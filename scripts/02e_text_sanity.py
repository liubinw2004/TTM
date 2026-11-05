"""文本条件体检（QA F-16 闸门）：179 个条件的中文扩写是否**真的聚得住**。

为什么必须在 M3 之前跑：

扩散模型的 CFG 条件就是 bge 句向量。如果同一个动作的 3 条扩写在向量空间里聚不
到一起（``sim_intra`` 不比 ``sim_inter`` 高出足够余量），那训练时的条件信号本身
是噪声——此时开 55 分钟训练等于白烧，而且训完之后你会误以为是「模型条件能力
不足」，去调 guidance / 加条件层，方向全错。

三个互相独立的坏味道，判据不同：

1. **聚不起来**（``sim_intra <= mean(sim_inter) + margin``）：
   扩写的语义漂移太大，「同一个意思换个说法」变成了「另一种意思」。
   这是扩写质量问题，要回去改标签。
2. **扩写几乎同句**（``sim_intra`` 普遍 > 0.9）：
   3 条变体字面差异太小，等于没做数据增强——训练时条件 dropout 之外没有任何
   增强收益，且极易过拟合到某一句固定说法。
3. **条件之间分不开**（``sim_intra`` 与 ``sim_inter`` 同时很高）：
   179 个条件在句向量空间里挤成一团，等于只有「一个条件」，CFG 无从区分。

口径说明：

* ``sim_intra``：同一条件内、两两变体的余弦相似度均值；
* ``sim_inter``：**逐条件**算——本条件的变体与其它条件所有变体之间的余弦相似度
  均值。用「逐条件」而不是全局均值，是因为全局均值会被 179 个条件之间的平均相似
  度稀释，掩盖掉「某个条件特别不聚」的情况；
* 只有变体数 ≥2 的条件才能算 ``sim_intra``；门禁样本只从变体数 ≥3 的条件里抽
  （抽到的条件才能保证「3 条变体」这个前提真实成立）。

用法::

    python scripts/02e_text_sanity.py --config configs/tier_s.yaml
    python scripts/02e_text_sanity.py --config configs/tier_s.yaml --sample 15 --wushu-sample 8

退出码：0 通过；1 不通过（聚不住 / 过度相似 / 抽不到足够样本）；3 标签缺失；2 文本编码器不可用。
"""

from __future__ import annotations

import argparse
import itertools
import json
import sys
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Sequence

# --- sys.path 引导 -----------------------------------------------------------
_PROJECT_ROOT = Path(__file__).resolve().parents[1]
_SRC_DIR = _PROJECT_ROOT / "src"
if str(_SRC_DIR) not in sys.path:
    sys.path.insert(0, str(_SRC_DIR))

import numpy as np  # noqa: E402

from ttm.common import paths as P  # noqa: E402
from ttm.common.config import load_config  # noqa: E402
from ttm.common.logutil import close_all, get_logger  # noqa: E402

#: 每个条件最多取几条变体（够算两两均值，又不至于把批次堆太大）。
MAX_VARIANTS = 3

#: ``sim_intra`` 高于它就算「扩写几乎同句」（等于没增强 + 容易过拟合固定说法）。
OVER_SIMILAR_THRESHOLD = 0.90

#: 默认门禁样本量 / 其中武术条件的下限（QA F-16 要求「≥15，武术 ≥8」）。
DEFAULT_SAMPLE = 15
DEFAULT_WUSHU_SAMPLE = 8


@dataclass
class ConditionStat:
    """单个条件的体检结果。"""

    label: str
    category_l1: str
    is_wushu: bool
    n_windows: int
    variants: list[str] = field(default_factory=list)
    sim_intra: float = float("nan")
    sim_inter: float = float("nan")

    @property
    def margin(self) -> float:
        """``sim_intra - sim_inter``（越大越聚得住）。"""
        if np.isnan(self.sim_intra) or np.isnan(self.sim_inter):
            return float("nan")
        return float(self.sim_intra - self.sim_inter)

    def to_dict(self) -> dict[str, Any]:
        """序列化。"""
        return {
            "label": self.label,
            "category_l1": self.category_l1,
            "is_wushu": bool(self.is_wushu),
            "n_windows": int(self.n_windows),
            "n_variants": len(self.variants),
            "variants": list(self.variants),
            "sim_intra": None if np.isnan(self.sim_intra) else round(float(self.sim_intra), 4),
            "sim_inter": None if np.isnan(self.sim_inter) else round(float(self.sim_inter), 4),
            "margin": None if np.isnan(self.margin) else round(float(self.margin), 4),
        }


def load_condition_rows(path: Path) -> list[dict[str, Any]]:
    """读标签文件（逐窗口行）。

    Args:
        path: ``labels_merged.jsonl`` 路径。

    Returns:
        行字典列表。

    Raises:
        FileNotFoundError: 文件不存在。
    """
    target = Path(path)
    if not target.is_file():
        raise FileNotFoundError(f"标签文件不存在: {target}")
    rows: list[dict[str, Any]] = []
    with target.open("r", encoding="utf-8") as handle:
        for line in handle:
            text = line.strip()
            if text:
                rows.append(json.loads(text))
    return rows


def group_conditions(rows: Sequence[dict[str, Any]]) -> list[ConditionStat]:
    """按 ``zh_label`` 聚成条件，收集各条件的变体与窗口数。

    Args:
        rows: 标签行。

    Returns:
        按 ``zh_label`` 排序的 :class:`ConditionStat` 列表。
    """
    buckets: dict[str, ConditionStat] = {}
    for row in rows:
        label = str(row.get("zh_label") or "")
        if not label:
            continue
        stat = buckets.get(label)
        if stat is None:
            stat = ConditionStat(
                label=label,
                category_l1=str(row.get("category_l1") or ""),
                is_wushu=bool(row.get("is_wushu")) or "wushu" in str(row.get("category_l1", "")).lower(),
                n_windows=0,
            )
            buckets[label] = stat
        stat.n_windows += 1
        for text in row.get("zh_texts") or []:
            text = str(text).strip()
            if text and text not in stat.variants:
                stat.variants.append(text)
    # 主标签（类中心）必须排在首位：它是最规范的表述，也是「条件」这个名字本身。
    # 不能只在缺失时插入——扩写列表里通常本来就含主标签（模板的第一条就是它），
    # 只判断「不在才插」会让首位取决于 zh_texts 的排列。
    for stat in buckets.values():
        if stat.label in stat.variants:
            stat.variants.remove(stat.label)
        stat.variants.insert(0, stat.label)
    return [buckets[key] for key in sorted(buckets)]


def _normalize(matrix: np.ndarray) -> np.ndarray:
    """按行做 L2 归一化（余弦相似度 = 归一化后的内积）。"""
    norms = np.linalg.norm(matrix, axis=1, keepdims=True)
    return matrix / np.clip(norms, 1e-12, None)


def compute_similarities(stats: Sequence[ConditionStat], encoder: Any, *, batch_size: int = 64) -> None:
    """就地填入每个条件的 ``sim_intra`` / ``sim_inter``。

    Args:
        stats: 条件列表（会被就地修改）。
        encoder: 文本编码器（``encode(list[str]) -> np.ndarray``）。
        batch_size: 编码批大小。

    Notes:
        只对变体数 ≥2 的条件算 ``sim_intra``；``sim_inter`` 用「本条件变体 vs
        其它条件全部变体」的均值，因此需要把所有文本一次性编码、按行切片比较。
    """
    texts: list[str] = []
    spans: dict[int, tuple[int, int]] = {}
    for index, stat in enumerate(stats):
        picked = stat.variants[:MAX_VARIANTS]
        stat.variants = picked
        start = len(texts)
        texts.extend(picked)
        spans[index] = (start, start + len(picked))
    if not texts:
        return
    vectors = _normalize(np.asarray(encoder.encode(texts, batch_size=batch_size), dtype=np.float64))
    full = np.arange(len(texts))

    for index, stat in enumerate(stats):
        start, end = spans[index]
        own = vectors[start:end]
        if own.shape[0] >= 2:
            pairs = list(itertools.combinations(range(own.shape[0]), 2))
            sims = [float(own[a] @ own[b]) for a, b in pairs]
            stat.sim_intra = float(np.mean(sims)) if sims else float("nan")
        others = np.setdiff1d(full, np.arange(start, end), assume_unique=True)
        if others.size and own.shape[0]:
            # [n_variants, n_others] 的平均
            stat.sim_inter = float((own @ vectors[others].T).mean())


def pick_sample(
    stats: Sequence[ConditionStat],
    *,
    sample: int = DEFAULT_SAMPLE,
    wushu_sample: int = DEFAULT_WUSHU_SAMPLE,
) -> list[ConditionStat]:
    """确定性地抽门禁样本：先满足武术下限，再跨类别铺开补足。

    抽样的「确定性」很重要：门禁结论必须可复现，不能用随机种子掷骰子。

    Args:
        stats: 条件列表。
        sample: 总样本量。
        wushu_sample: 其中武术条件的数量下限。

    Returns:
        被抽中的条件（顺序稳定）。
    """
    eligible = [item for item in stats if len(item.variants) >= MAX_VARIANTS]
    if len(eligible) <= sample:
        return list(eligible)

    picked: list[ConditionStat] = []
    # 1) 武术：按窗口数降序（窗口多的条件更影响训练），确定性
    wushu = sorted(
        (item for item in eligible if item.is_wushu), key=lambda item: (-item.n_windows, item.label)
    )
    picked.extend(wushu[: min(wushu_sample, sample)])

    # 2) 其余：按 category_l1 轮转取样，保证跨类别铺开而不是集中在一两个大类
    remaining = [item for item in eligible if item not in picked]
    by_category: dict[str, list[ConditionStat]] = {}
    for item in sorted(remaining, key=lambda item: (-item.n_windows, item.label)):
        by_category.setdefault(item.category_l1, []).append(item)
    cursors = {key: 0 for key in by_category}
    order = sorted(by_category)
    while len(picked) < sample:
        progressed = False
        for key in order:
            if len(picked) >= sample:
                break
            bucket = by_category[key]
            cursor = cursors[key]
            if cursor < len(bucket):
                picked.append(bucket[cursor])
                cursors[key] = cursor + 1
                progressed = True
        if not progressed:  # 所有类别都取空了
            break
    return picked


def build_report(
    stats: Sequence[ConditionStat],
    sample_stats: Sequence[ConditionStat],
    *,
    margin_required: float,
    labels_path: str,
) -> dict[str, Any]:
    """组织体检报告（含判定与失败原因）。

    Args:
        stats: 全部条件。
        sample_stats: 门禁样本。
        margin_required: 要求的 ``sim_intra - sim_inter`` 下限。
        labels_path: 标签文件路径（写进报告便于追溯）。

    Returns:
        报告字典。
    """
    usable = [item for item in stats if not np.isnan(item.sim_intra)]
    intra = np.asarray([item.sim_intra for item in usable], dtype=np.float64)
    over_share = float(np.mean(intra > OVER_SIMILAR_THRESHOLD)) if intra.size else 0.0

    sample_rows: list[dict[str, Any]] = []
    failures: list[str] = []
    for item in sample_stats:
        row = item.to_dict()
        passed = (not np.isnan(item.margin)) and item.margin > margin_required
        row["passed"] = bool(passed)
        sample_rows.append(row)
        if not passed:
            failures.append(
                f"{item.label}: sim_intra={item.sim_intra:.4f} / sim_inter={item.sim_inter:.4f}"
                f" / margin={item.margin:.4f} ≤ {margin_required:.2f}"
            )

    wushu_in_sample = sum(1 for item in sample_stats if item.is_wushu)
    reasons: list[str] = []
    if len(sample_stats) < DEFAULT_SAMPLE:
        reasons.append(f"可抽样本只有 {len(sample_stats)} 个（<{DEFAULT_SAMPLE}），条件变体不足")
    if wushu_in_sample < DEFAULT_WUSHU_SAMPLE:
        reasons.append(
            f"武术样本只有 {wushu_in_sample} 个（<{DEFAULT_WUSHU_SAMPLE}），"
            "武术 24 个条件里变体数 ≥3 的不够"
        )
    reasons.extend(failures)
    if over_share > 0.5:
        reasons.append(
            f"{over_share:.1%} 的条件 sim_intra > {OVER_SIMILAR_THRESHOLD}，"
            "扩写几乎同句（等于没做增强）"
        )

    return {
        "labels": str(labels_path),
        "passed": len(reasons) == 0,
        "margin_required": float(margin_required),
        "over_similar_threshold": OVER_SIMILAR_THRESHOLD,
        "conditions": {
            "total": len(stats),
            "with_variants": int(sum(1 for item in stats if len(item.variants) >= 2)),
            "wushu": int(sum(1 for item in stats if item.is_wushu)),
        },
        "all_conditions": {
            "sim_intra_median": round(float(np.median(intra)), 4) if intra.size else None,
            "sim_intra_p10": round(float(np.percentile(intra, 10)), 4) if intra.size else None,
            "sim_intra_min": round(float(intra.min()), 4) if intra.size else None,
            "sim_intra_max": round(float(intra.max()), 4) if intra.size else None,
            "over_similar_share": round(over_share, 4),
        },
        "sample": {
            "size": len(sample_rows),
            "wushu": wushu_in_sample,
            "passed": int(sum(1 for row in sample_rows if row["passed"])),
            "items": sample_rows,
        },
        "failures": reasons,
    }


def _print_report(report: dict[str, Any]) -> None:
    """打印体检报告（可直接贴进汇报）。"""
    stats = report["all_conditions"]
    sample = report["sample"]
    print("=" * 84)
    print("文本条件体检（QA F-16 闸门）")
    print("=" * 84)
    print(f"标签文件            : {report['labels']}")
    print(
        f"条件数              : {report['conditions']['total']} "
        f"（变体数 ≥2 的 {report['conditions']['with_variants']} 个 / 武术 {report['conditions']['wushu']} 个）"
    )
    print(f"sim_intra 中位数    : {stats['sim_intra_median']}")
    print(f"sim_intra P10 / min : {stats['sim_intra_p10']} / {stats['sim_intra_min']}")
    print(f"sim_intra max       : {stats['sim_intra_max']}")
    print(f"过度相似占比(>0.90) : {stats['over_similar_share']:.1%}")
    print("-" * 84)
    print(f"门禁样本            : {sample['size']} 个（武术 {sample['wushu']} 个），要求 margin > {report['margin_required']}")
    print(f"  {'条件':<14}{'类':<26}{'intra':>8}{'inter':>8}{'margin':>8}  判定")
    for row in sample["items"]:
        intra = "-" if row["sim_intra"] is None else f"{row['sim_intra']:.4f}"
        inter = "-" if row["sim_inter"] is None else f"{row['sim_inter']:.4f}"
        margin = "-" if row["margin"] is None else f"{row['margin']:.4f}"
        category = row["category_l1"][:24]
        flag = "PASS" if row["passed"] else "FAIL"
        print(f"  {row['label']:<14}{category:<26}{intra:>8}{inter:>8}{margin:>8}  {flag}")
    print("-" * 84)
    print(f"通过                : {sample['passed']}/{sample['size']}")
    if report["passed"]:
        print("结论                : PASS —— 扩写聚得住、条件分得开，可以开 M3")
    else:
        print("结论                : FAIL")
        for reason in report["failures"]:
            print(f"  - {reason}")
    print("=" * 84)


def build_argparser() -> argparse.ArgumentParser:
    """构造命令行参数解析器。"""
    parser = argparse.ArgumentParser(
        description="文本条件体检：179 个条件的中文扩写是否聚得住（M3 前置闸门）",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument("--config", default="configs/default.yaml", help="配置文件路径")
    parser.add_argument("--labels", default="", help="标签文件（默认取配置里的 train_labels_path）")
    parser.add_argument("--out", default="runs/eval/text_sanity.json", help="报告输出路径")
    parser.add_argument("--sample", type=int, default=DEFAULT_SAMPLE, help="门禁样本量")
    parser.add_argument(
        "--wushu-sample", type=int, default=DEFAULT_WUSHU_SAMPLE, help="样本中武术条件数量下限"
    )
    parser.add_argument(
        "--margin", type=float, default=0.1, help="要求的 sim_intra − sim_inter 下限"
    )
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
    log = get_logger("ttm.eval.text_sanity", script="02e_text_sanity")

    labels_path = Path(args.labels) if args.labels else cfg.train_labels_path()
    try:
        rows = load_condition_rows(labels_path)
    except FileNotFoundError as exc:
        log.error("%s", exc)
        close_all()
        return 3
    if not rows:
        log.error("标签文件是空的: %s", labels_path)
        close_all()
        return 3

    stats = group_conditions(rows)
    log.info(
        "条件聚合完成: %d 个条件 / %d 行（武术 %d 个）",
        len(stats),
        len(rows),
        sum(1 for item in stats if item.is_wushu),
    )

    from ttm.models.text_encoder import TextEncoder, TextEncoderUnavailable  # noqa: PLC0415

    try:
        encoder = TextEncoder(
            cfg.text.encoder,
            model_path=cfg.text.model_path or None,
            proj_dim=cfg.text.proj_dim,
            frozen=cfg.text.frozen,
            max_len=cfg.text.max_len,
            device="cpu",
            local_files_only=True,
        )
        # 预热：首次 encode 有 ~3 秒 lazy init，不预热会让第一个条件的耗时假得离谱
        encoder.encode(["预热"], batch_size=1)
    except TextEncoderUnavailable as exc:
        log.error("文本编码器不可用（%s），无法体检", exc)
        close_all()
        return 2

    compute_similarities(stats, encoder)
    sample_stats = pick_sample(stats, sample=int(args.sample), wushu_sample=int(args.wushu_sample))
    report = build_report(
        stats,
        sample_stats,
        margin_required=float(args.margin),
        labels_path=labels_path.as_posix(),
    )
    _print_report(report)

    out = P.resolve_out(args.out, mkparents=True)
    out.write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")
    print(f"报告文件: {out.as_posix()}")

    if report["passed"]:
        log.info("文本条件体检 PASS")
    else:
        for reason in report["failures"]:
            log.error("文本条件体检 FAIL: %s", reason)
    close_all()
    return 0 if report["passed"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
