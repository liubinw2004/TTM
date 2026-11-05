"""把「非中文条件标签」回填成中文（纯标签层，不碰特征/分片）。

**背景**：M3 前的文本条件体检（``scripts/02e_text_sanity.py``）发现 179 个条件里有
3 个是英文——``Box Ing`` / ``Cooperative Transport`` / ``Equipment Exercises``，
合计 240 个窗口（1.75%）。根因在标签解析链：这三个 l3 目录既不在精确表里，也过不了
关键词（``box_ing`` 带下划线，``boxing`` 匹配不上），于是落到机械转写，产出了
``Box Ing`` 这种英文标签，以及「慢慢地Box Ing」这种中英混排的扩写。

对**中文指令模型**来说英文条件是硬伤：任何一句中文 prompt 都触发不到它，而且它的
扩写是模板套英文词，语义空间和中文条件不在一处。

两个必须同时做、缺一不可的修复：

1. **源头**：在 :mod:`ttm.data.labels_zh` 的 ``L3_EXACT_ZH`` 补上这三条映射
   ——否则下一次全量预处理会重新产出同样的坏标签（本脚本只修已经产出的产物）；
2. **产物**：本脚本把 ``labels.jsonl`` 里已生成的英文标签回填成中文，再重跑
   ``02b_merge_labels.py`` 重建 ``labels_merged.jsonl``。

为什么不重跑 ``02_preprocess.py``：特征分片跟标签无关，重算要 30 分钟且会动
``norm_stats``——为一个纯标签问题动特征产物是白换风险。标签层的回填是幂等的、
可审的、可回滚的（首次运行留 ``.bak``）。

用法::

    python scripts/02f_relabel_english.py --config configs/tier_s.yaml          # 回填
    python scripts/02f_relabel_english.py --config configs/tier_s.yaml --dry-run  # 只看

之后必须接着跑（顺序不能反）::

    python scripts/02b_merge_labels.py --config configs/tier_s.yaml
    python scripts/02d_verify_data.py --config configs/tier_s.yaml     # a)–f) 门禁
    python scripts/02e_text_sanity.py --config configs/tier_s.yaml     # 文本体检

退出码：0 全部回填成功（或本来就干净）；1 仍有无法解析的英文标签（需要补映射）。
"""

from __future__ import annotations

import argparse
import json
import os
import re
import shutil
import sys
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

# --- sys.path 引导 -----------------------------------------------------------
_PROJECT_ROOT = Path(__file__).resolve().parents[1]
_SRC_DIR = _PROJECT_ROOT / "src"
if str(_SRC_DIR) not in sys.path:
    sys.path.insert(0, str(_SRC_DIR))

from ttm.common import paths as P  # noqa: E402
from ttm.common.config import load_config  # noqa: E402
from ttm.common.logutil import close_all, get_logger  # noqa: E402
from ttm.data.labels_zh import (  # noqa: E402
    LabelGenerator,
    normalize_key,
    zh_label_for_l3,
)

#: 汉字区间（用于判定「这个标签是不是中文」）。
_CJK = re.compile(r"[\u4e00-\u9fff]")


@dataclass
class Relabel:
    """一个英文标签的回填记录。"""

    old: str
    new: str
    category_l3: str
    n_windows: int
    variants: list[str] = field(default_factory=list)

    def to_dict(self) -> dict[str, Any]:
        """序列化。"""
        return {
            "old": self.old,
            "new": self.new,
            "category_l3": self.category_l3,
            "n_windows": int(self.n_windows),
            "n_variants": len(self.variants),
        }


def has_cjk(text: str) -> bool:
    """``text`` 是否含汉字。

    Args:
        text: 待检查字符串。

    Returns:
        含至少一个汉字返回 ``True``。
    """
    return bool(_CJK.search(str(text or "")))


def plan_relabels(
    rows: list[dict[str, Any]], *, generator: LabelGenerator | None = None
) -> tuple[list[Relabel], list[str]]:
    """统计需要回填的英文标签，并算出对应的中文标签与扩写。

    Args:
        rows: ``labels.jsonl`` 的行。
        generator: 可选复用 :class:`LabelGenerator`。

    Returns:
        ``(回填列表, 无法解析的标签列表)``。后者非空说明映射表还有洞。
    """
    maker = generator or LabelGenerator()
    seen: dict[str, Relabel] = {}
    unresolved: list[str] = []
    for row in rows:
        label = str(row.get("zh_label") or "")
        if not label or has_cjk(label):
            continue
        record = seen.get(label)
        if record is None:
            l3 = str(row.get("category_l3") or "")
            l2 = str(row.get("category_l2") or "")
            new = zh_label_for_l3(l3, l2) or zh_label_for_l3(l2, "")
            if not new:
                # 连 l2 都解析不出：记下来，让调用方去补映射，不要瞎猜
                if label not in unresolved:
                    unresolved.append(label)
                continue
            record = Relabel(
                old=label,
                new=new,
                category_l3=l3 or l2,
                n_windows=0,
                variants=maker.expand(new),
            )
            seen[label] = record
        record.n_windows += 1
    return list(seen.values()), unresolved


def apply_relabels(rows: list[dict[str, Any]], relabels: list[Relabel]) -> int:
    """就地回填标签与扩写（含细标签留档字段）。

    Args:
        rows: ``labels.jsonl`` 的行（会被就地修改）。
        relabels: :func:`plan_relabels` 的结果。

    Returns:
        实际改动的行数。
    """
    mapping = {item.old: item for item in relabels}
    changed = 0
    for row in rows:
        label = str(row.get("zh_label") or "")
        record = mapping.get(label)
        if record is None:
            continue
        fine = str(row.get("zh_label_fine") or "")
        row["zh_label"] = record.new
        row["zh_texts"] = list(record.variants)
        # 细标签留档字段：只有当它也等于那个英文标签时才一起换，
        # 否则会覆盖掉真正的细标签（例如合并过的行 fine ≠ zh_label）
        if fine == label:
            row["zh_label_fine"] = record.new
            row["zh_texts_fine"] = list(record.variants)
        changed += 1
    return changed


def read_rows(path: Path) -> list[dict[str, Any]]:
    """读 JSONL。

    Args:
        path: 文件路径。

    Returns:
        行字典列表。
    """
    rows: list[dict[str, Any]] = []
    with Path(path).open("r", encoding="utf-8") as handle:
        for line in handle:
            text = line.strip()
            if text:
                rows.append(json.loads(text))
    return rows


def write_rows(path: Path, rows: list[dict[str, Any]], *, backup: bool = True) -> Path:
    """原子写回 JSONL（首次写入前留 ``.bak``）。

    Args:
        path: 目标文件。
        rows: 行列表。
        backup: 是否在覆盖前备份到 ``<path>.bak``（已存在则不覆盖备份）。

    Returns:
        实际写入的路径。
    """
    target = P.resolve_out(path, mkparents=True)
    if backup and target.is_file():
        backup_path = target.with_suffix(target.suffix + ".bak")
        if not backup_path.exists():
            shutil.copy2(target, backup_path)
    temp = target.with_suffix(target.suffix + ".tmp")
    with temp.open("w", encoding="utf-8", newline="\n") as handle:
        for row in rows:
            handle.write(json.dumps(row, ensure_ascii=False) + "\n")
    os.replace(temp, target)
    return target


def _print_plan(relabels: list[Relabel], unresolved: list[str], *, dry_run: bool) -> None:
    """打印回填计划。"""
    print("=" * 84)
    print("非中文条件标签回填" + ("（dry-run，不写盘）" if dry_run else ""))
    print("=" * 84)
    if not relabels:
        print("没有需要回填的标签（全部条件都已含汉字）")
    else:
        print(f"  {'原标签':<24}{'新标签':<12}{'目录':<34}{'窗口':>6}")
        for item in relabels:
            print(
                f"  {item.old:<24}{item.new:<12}{item.category_l3[:32]:<34}{item.n_windows:>6}"
            )
        total = sum(item.n_windows for item in relabels)
        print("-" * 84)
        print(f"合计 {len(relabels)} 个标签 / {total} 个窗口")
        for item in relabels:
            print(f"  扩写示例（{item.new}）: {item.variants[:3]}")
    if unresolved:
        print("-" * 84)
        print("!! 以下标签无法解析出中文，请到 ttm/data/labels_zh.py 补映射:")
        for label in unresolved:
            print(f"   - {label}")
    print("=" * 84)


def build_argparser() -> argparse.ArgumentParser:
    """构造命令行参数解析器。"""
    parser = argparse.ArgumentParser(
        description="把 labels.jsonl 里的英文条件标签回填成中文（纯标签层）",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument("--config", default="configs/default.yaml", help="配置文件路径")
    parser.add_argument(
        "--labels", default="", help="标签文件（默认取配置里的 labels_path，即 labels.jsonl）"
    )
    parser.add_argument("--dry-run", action="store_true", help="只打印计划，不写盘")
    return parser


def main(argv: list[str] | None = None) -> int:
    """命令行入口。

    Args:
        argv: 参数列表（默认取 ``sys.argv[1:]``）。

    Returns:
        进程退出码。
    """
    args = build_argparser().parse_args(argv)
    cfg = load_config(args.config)
    log = get_logger("ttm.data.relabel", script="02f_relabel_english")

    labels_path = Path(args.labels) if args.labels else cfg.labels_path()
    if not labels_path.is_file():
        log.error("标签文件不存在: %s。请先运行 scripts/02_preprocess.py", labels_path)
        close_all()
        return 3

    rows = read_rows(labels_path)
    log.info("读入 %d 行: %s", len(rows), labels_path)
    relabels, unresolved = plan_relabels(rows)
    _print_plan(relabels, unresolved, dry_run=bool(args.dry_run))

    if unresolved:
        log.error("有 %d 个英文标签无法解析，请先补映射表再重跑", len(unresolved))
        close_all()
        return 1

    if args.dry_run:
        log.info("dry-run：未写盘")
        close_all()
        return 0

    if not relabels:
        log.info("标签已是全中文，无需回填")
        close_all()
        return 0

    changed = apply_relabels(rows, relabels)
    target = write_rows(labels_path, rows)
    log.info("已回填 %d 行并写回 %s（备份 .bak）", changed, target)
    print(f"已写回: {target.as_posix()}")
    print("下一步（顺序不能反）:")
    print(f"  1) python scripts/02b_merge_labels.py --config {args.config}")
    print(f"  2) python scripts/02d_verify_data.py --config {args.config}")
    print(f"  3) python scripts/02e_text_sanity.py --config {args.config}")
    close_all()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
