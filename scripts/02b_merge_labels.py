"""T05b 脚本：少样本标签合并（重标签后处理，不重跑预处理）。

用法::

    . D:/TTM/scripts/env.ps1
    python scripts/02b_merge_labels.py --config configs/tier_s.yaml
    python scripts/02b_merge_labels.py --config configs/tier_s.yaml --dry-run   # 只看报告不写盘

**这不是可选打磨，是条件可学性的硬约束**（主理人 2026-10-29 追加）：

Tier-S 实测 350 个细标签，其中 191 个是武术、总共只有 833 个窗口
（191 类 / 220 文件 / 每类 1.15 个文件），184 个不到 20 窗、中位 4 窗。
直接拿它们当条件 = 给模型 191 个「各 4 个样本」的条件，模型只会拟合噪声，
还会把本来能学的共性语义（「这是长拳」）一起冲掉。

规则：``W < 30`` 的细标签并到基名（长拳 → 长拳第七段…；刀/剑/棍/枪按器械种类
各自成组；虎鹤蛇龙豹 → 象形拳）。合并只动**标签层**——``labels.jsonl`` 与
``shards/*.npz`` 是解耦的，所以这个脚本不读 Z 盘、不重算特征，几秒跑完。

产物（``data/tier_{tier}/``）::

    labels_merged.jsonl   每窗口一行：zh_label/zh_texts（合并后，训练用）
                          + zh_label_fine/zh_texts_fine（原细标签，统计口径）
                          + label_merged/is_wushu
    runs/tier_{tier}/merge_report.json   合并决策与分布（机器可读）

``labels.jsonl`` **保持不动**：它是「合并前 distinct 细类数」的统计口径来源。

退出码：合并后 ``distinct >= 30`` 且最大单一标签占比 ``<= 15%`` 时为 0，否则 1。
"""

from __future__ import annotations

import argparse
import collections
import sys
from pathlib import Path

# --- sys.path 引导 -----------------------------------------------------------
_PROJECT_ROOT = Path(__file__).resolve().parents[1]
_SRC_DIR = _PROJECT_ROOT / "src"
if str(_SRC_DIR) not in sys.path:
    sys.path.insert(0, str(_SRC_DIR))

import numpy as np  # noqa: E402
import orjson  # noqa: E402

from ttm.common import paths as P  # noqa: E402
from ttm.common.config import dump_resolved, load_config  # noqa: E402
from ttm.common.logutil import close_all, get_logger  # noqa: E402
from ttm.common.seed import set_seed  # noqa: E402
from ttm.data.dataset import (  # noqa: E402
    build_sample_weights,
    expected_group_share,
    resolve_group_multiplier,
)
from ttm.data.label_merge import (  # noqa: E402
    MERGE_THRESHOLD,
    apply_plan_to_rows,
    plan_merges,
)

#: 验收线（主理人 2026-10-29）：合并后 distinct ≥ 30、最大单一标签占比 ≤ 15%。
GATE_MIN_DISTINCT = 30
GATE_MAX_SHARE = 0.15


def build_argparser() -> argparse.ArgumentParser:
    """构造命令行参数解析器。"""
    parser = argparse.ArgumentParser(
        description="少样本标签合并 → labels_merged.jsonl（TTM T05b）",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument(
        "--config", default="configs/tier_s.yaml", help="配置文件路径（相对项目根）"
    )
    parser.add_argument(
        "--threshold",
        type=int,
        default=0,
        help="合并阈值（0 = 用配置里的 data.rare_label_min_windows）",
    )
    parser.add_argument("--dry-run", action="store_true", help="只出报告不写盘")
    return parser


def read_jsonl(path: Path) -> list[dict]:
    """读 jsonl。

    Args:
        path: 文件路径。

    Returns:
        记录列表；文件不存在时返回空列表。
    """
    rows: list[dict] = []
    if not path.is_file():
        return rows
    with path.open("rb") as handle:
        for line in handle:
            text = line.strip()
            if text:
                rows.append(orjson.loads(text))
    return rows


def write_jsonl(path: Path, rows: list[dict]) -> None:
    """写 jsonl（经写盘守卫，只允许写 D:/TTM）。"""
    target = P.resolve_out(path, mkparents=True)
    with target.open("wb") as handle:
        for row in rows:
            handle.write(orjson.dumps(row))
            handle.write(b"\n")


def _print_report(cfg, plan, rows, index_rows, args) -> float:  # type: ignore[no-untyped-def]
    """打印合并报告，返回最大单一标签占比。"""
    total = len(rows)
    name, share = plan.max_share_after()
    print("=" * 92)
    print(f"TTM 少样本标签合并报告 · tier={cfg.data.tier} · 阈值 W < {plan.threshold}")
    print("=" * 92)
    print(f"窗口总数                : {total}")
    print(
        f"distinct 标签（合并前）  : {plan.distinct_before}   ← 统计口径，含全部细类"
    )
    print(
        f"distinct 标签（合并后）  : {plan.distinct_after}   ← **训练实际使用的条件数**"
    )
    print(f"被并掉的细标签          : {len(plan.mapping)}")
    print(f"整族合并（本身 ≥ 阈值）  : {list(plan.forced) or '无'}")
    print(f"最大单一标签            : {name} {plan.counts_after.get(name, 0)} 窗 = {share * 100:.2f}%")
    gate_distinct = plan.distinct_after >= GATE_MIN_DISTINCT
    gate_share = share <= GATE_MAX_SHARE
    print(
        f"验收线                  : distinct ≥ {GATE_MIN_DISTINCT} "
        f"{'✓' if gate_distinct else '✗'} / "
        f"最大单一占比 ≤ {GATE_MAX_SHARE * 100:.0f}% {'✓' if gate_share else '✗'}"
    )

    # -- residual：合并后仍不可学的标签（无更粗父级，如实上报不静默丢）------
    print("-" * 92)
    print(f"合并后仍 < {plan.threshold} 窗的标签（{len(plan.residual)} 个，无更粗父级，如实保留）:")
    for item in plan.residual:
        print(f"  {plan.counts_after[item]:5d} 窗   {item}")

    # -- 武术：合并后基名分布 -----------------------------------------------
    wushu_flags = [bool(row.get("is_wushu")) for row in rows]
    merged_labels = [str(row.get("zh_label", "")) for row in rows]
    wushu_counts: collections.Counter[str] = collections.Counter(
        label for label, flag in zip(merged_labels, wushu_flags) if flag
    )
    wushu_total = sum(wushu_counts.values())
    print("-" * 92)
    print(
        f"武术（合并后 {len(wushu_counts)} 个条件 / {wushu_total} 窗 = "
        f"{wushu_total / max(total, 1) * 100:.2f}%）:"
    )
    for label, count in wushu_counts.most_common():
        print(f"  {count:5d} 窗  {count / max(wushu_total, 1) * 100:5.1f}%  {label}")

    # -- 武术：191 个细类逐项全表（主理人要求「看全表」）--------------------
    clip_stats = _wushu_clip_table(rows, index_rows)
    print("-" * 92)
    print(f"武术细类逐项全表（{len(clip_stats)} 个细类；clip 数来自 index.jsonl）:")
    print(f"  {'窗口':>5}  {'clip':>5}  {'合并后基名':<12} 细类目录")
    for item in clip_stats:
        print(
            f"  {item['windows']:5d}  {item['clips']:5d}  {item['merged']:<12} {item['l3']}"
        )

    # -- 大类窗口占比 -------------------------------------------------------
    by_l1: collections.Counter[str] = collections.Counter()
    for row in rows:
        by_l1[str(row.get("category_l1", ""))] += 1
    print("-" * 92)
    print("大类窗口占比:")
    for key, count in by_l1.most_common():
        print(f"  {count:6d}  {count / max(total, 1) * 100:5.2f}%  {key}")

    # -- 训练期采样占比（数据占比 → 采样期望占比）---------------------------
    mask = np.asarray(wushu_flags, dtype=bool)
    data_share = float(mask.sum()) / max(len(mask), 1)
    base_weights = build_sample_weights(
        merged_labels,
        power=float(cfg.data.balance_power),
        max_weight_ratio=float(cfg.data.balance_max_ratio),
    )
    target_ratio = float(cfg.data.balanced_target_wushu_ratio)
    tuned = build_sample_weights(
        merged_labels,
        power=float(cfg.data.balance_power),
        max_weight_ratio=float(cfg.data.balance_max_ratio),
        group_mask=mask,
        target_group_share=target_ratio,
    )
    multiplier = resolve_group_multiplier(base_weights, mask, target_ratio)
    print("-" * 92)
    print(
        f"武术采样占比            : 数据 {data_share * 100:.2f}% → "
        f"power={cfg.data.balance_power} 后 {expected_group_share(base_weights, mask) * 100:.2f}% → "
        f"组倍率 ×{multiplier:.3f} 后 {expected_group_share(tuned, mask) * 100:.2f}%"
        f"（目标 {target_ratio * 100:.2f}%）"
    )
    print(
        f"权重跨度                : max/min = {tuned.max() / tuned.min():.1f} "
        f"（上限 balance_max_ratio={cfg.data.balance_max_ratio}）"
    )
    print("=" * 92)
    return share


def _wushu_clip_table(rows: list[dict], index_rows: list[dict]) -> list[dict]:
    """按细类汇总武术的 clip 数与窗口数（细类 → 合并后基名）。

    Args:
        rows: ``labels_merged.jsonl`` 的行（窗口级）。
        index_rows: ``index.jsonl`` 的行（clip 级，取 ``n_windows``）。

    Returns:
        按窗口数降序的列表，每项含 ``l3`` / ``merged`` / ``clips`` / ``windows``。
    """
    clips_per_l3: collections.Counter[str] = collections.Counter()
    for entry in index_rows:
        if "wushu" not in str(entry.get("category_l1", "")).lower():
            continue
        clips_per_l3[str(entry.get("category_l3", ""))] += 1

    agg: dict[str, dict] = {}
    for row in rows:
        if not row.get("is_wushu"):
            continue
        l3 = str(row.get("category_l3", ""))
        item = agg.setdefault(
            l3,
            {
                "l3": l3,
                "merged": str(row.get("zh_label", "")),
                "fine": str(row.get("zh_label_fine", "")),
                "windows": 0,
                "clips": clips_per_l3.get(l3, 0),
            },
        )
        item["windows"] += 1
    return sorted(agg.values(), key=lambda it: (-it["windows"], it["l3"]))


def main(argv: list[str] | None = None) -> int:
    """脚本入口。"""
    args = build_argparser().parse_args(argv)
    cfg = load_config(args.config)
    set_seed(cfg.runtime.seed, deterministic=False)
    log = get_logger("ttm.scripts.merge_labels", script="02b_merge_labels")
    dump_resolved(cfg, "merge_labels")
    log.info("加载配置: %s", cfg.source_path)

    labels_path = cfg.labels_path()
    index_path = cfg.index_path()
    rows = read_jsonl(labels_path)
    if not rows:
        log.error("labels.jsonl 不存在或为空: %s（请先跑 02_preprocess.py）", labels_path)
        close_all()
        return 1
    index_rows = read_jsonl(index_path)

    threshold = int(args.threshold) if args.threshold > 0 else int(cfg.data.rare_label_min_windows)
    labels = [str(row.get("zh_label", "")) for row in rows]
    flags = [
        bool(row.get("is_wushu"))
        if row.get("is_wushu") is not None
        else "wushu" in str(row.get("category_l1", "")).lower()
        for row in rows
    ]
    plan = plan_merges(labels, flags, threshold=threshold)
    merged_rows = apply_plan_to_rows(rows, plan)
    share = _print_report(cfg, plan, merged_rows, index_rows, args)

    if not args.dry_run:
        out_path = cfg.tier_dir() / cfg.data.labels_merged_file
        write_jsonl(out_path, merged_rows)
        log.info("已写出 %s（%d 行）", out_path, len(merged_rows))
        report_path = cfg.run_dir("merge_labels") / "merge_report.json"
        payload = plan.as_dict()
        payload["labels_merged_file"] = out_path.as_posix()
        payload["wushu_windows"] = sum(1 for flag in flags if flag)
        target = P.resolve_out(report_path, mkparents=True)
        target.write_bytes(orjson.dumps(payload, option=orjson.OPT_INDENT_2))
        log.info("已写出 %s", target)

    ok = plan.distinct_after >= GATE_MIN_DISTINCT and share <= GATE_MAX_SHARE
    close_all()
    return 0 if ok else 1


if __name__ == "__main__":
    raise SystemExit(main())
