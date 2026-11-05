#!/usr/bin/env python
"""各类别根高度中位数分布审计（M1' 追加诊断，**只读产物、不重跑预处理**）。

## 这个脚本为什么存在

M1' 的判废阈值是「足部最低高度 < :data:`ttm.data.features.MOTION_MIN_FOOT_HEIGHT_CM`
（-10cm）」——它只抓「明显穿地」的坏样本。球类实测有 110 条失败、根高度中位数稳定
落在 -21 ~ -31cm，是**系统性偏移**而不是随机噪声，所以整块剔除不划算（会丢 17.6%
的窗口），决定保留。

但 -10cm 这个硬阈值恰好存在一个盲区：**-5 ~ -20cm 之间的「半坏簇」**。
这些 clip 的脚只是略微下陷，判废规则会放行；可一旦进了训练集，模型就会把「这个人
比地面低十几厘米」当成正常分布的一部分去学，根高度这个量纲的基线被悄悄拉偏。
本脚本就是把这个盲区量化出来：

1. 逐窗口反归一化特征第 2 维（根高度，cm，见 ``FEAT_LAYOUT["root_height"]``）；
2. 只取 ``mask == 1`` 的有效帧（pad 帧特征全 0，反归一化后是「-mean/std」的假高度）；
3. 每个 **clip**（而不是每个窗口——窗口之间有 120 帧重叠，直接平均会重复计数）
   取全部有效帧的**中位数**（中位数而非均值：单个腾空帧能把均值拉飞）；
4. 按 ``category_l1`` 聚合，给中位数、p10、p90、极值，以及五个高度带的 clip 计数。

判定口径：某类别里落在「半坏带」(-20cm, -5cm] 的 clip 占比 **> 5%** 就标记
``SUSPECT``——那说明该类别内部存在一个偏移簇，需要按 clip 做地面校正；否则标
``CLEAN``，说明该类别只是整体有一致的基线偏移（可接受，模型学得到）或完全正常。

## 用法

```bash
python scripts/02c_root_height_audit.py --config configs/tier_s.yaml
# 只读旧产物对比时显式指定路径：
python scripts/02c_root_height_audit.py --config configs/tier_s.yaml \
    --labels data/tier_S/labels_merged.jsonl --out-dir runs/tier_s/root_audit
```

产物：``runs/tier_s/root_audit/root_height_audit.json``（机器读）与
``root_height_audit.md``（人读，含半坏带逐条 clip 清单）。
退出码恒为 0——这是诊断，不是门禁；门禁在 ``02d_verify_data.py``。
"""

from __future__ import annotations

import argparse
import sys
from collections import defaultdict
from pathlib import Path
from typing import Any, Iterable

import numpy as np
import orjson

from ttm.common import constants as C
from ttm.common import paths as P
from ttm.common.config import load_config
from ttm.data.features import NormStats

#: 高度分带 ``(下界, 上界, 名称)``，左开右闭。用于统计每个类别里各带的 clip 数。
HEIGHT_BANDS: tuple[tuple[float, float, str], ...] = (
    (float("-inf"), -20.0, "< -20cm 明显穿地"),
    (-20.0, -10.0, "-20 ~ -10cm 判废线内"),
    (-10.0, -5.0, "-10 ~ -5cm 半坏带（阈值盲区）"),
    (-5.0, 0.0, "-5 ~ 0cm 轻微下陷"),
    (0.0, float("inf"), ">= 0cm 正常"),
)

#: 「半坏带」覆盖的分带下标：``(-20cm, -10cm]`` + ``(-10cm, -5cm]``，合并即
#: ``(-20cm, -5cm]``。判定 SUSPECT 时把这两档一起算——它们同属「判废规则放行、
#: 但脚已经下陷」的区间，分开算会让「-15cm 簇」被漏掉一半。
SEMI_BAD_BAND_INDEXES: tuple[int, ...] = (1, 2)

#: 半坏带 clip 占比超过它就判定该类别存在「偏移簇」。
SUSPECT_RATIO: float = 0.05


def _read_labels(path: Path) -> dict[str, dict[str, Any]]:
    """读 ``labels*.jsonl``，返回 ``clip_id -> row``。

    Args:
        path: 标签文件路径。

    Returns:
        ``clip_id`` 到整行记录的映射（文件不存在时返回空字典并打印告警）。
    """
    if not path.is_file():
        print(f"[warn] 标签文件不存在: {path}", file=sys.stderr)
        return {}
    output: dict[str, dict[str, Any]] = {}
    with path.open("rb") as handle:
        for line in handle:
            line = line.strip()
            if not line:
                continue
            row = orjson.loads(line)
            output[str(row["clip_id"])] = row
    return output


def _band_index(height_cm: float) -> int:
    """把根高度中位数映射到 :data:`HEIGHT_BANDS` 的下标。

    Args:
        height_cm: clip 级根高度中位数（cm）。

    Returns:
        分带下标；高度为 NaN 时归到最后一档（避免统计里出现空洞）。
    """
    if not np.isfinite(height_cm):
        return len(HEIGHT_BANDS) - 1
    for index, (low, high, _name) in enumerate(HEIGHT_BANDS):
        if low < float(height_cm) <= high:
            return index
    # 恰好等于最小下界（-inf 不会命中）或落在 (-20, -20] 之类的边界情形
    return len(HEIGHT_BANDS) - 1


def collect_clip_heights(
    shard_dir: Path,
    norm_stats: NormStats,
    labels: dict[str, dict[str, Any]],
) -> dict[str, dict[str, Any]]:
    """扫描全部分片，算**逐 clip** 的根高度中位数。

    Args:
        shard_dir: ``shards/`` 目录（含 ``shard_*.npz``）。
        norm_stats: 归一化统计（用于把 z-score 还原成 cm）。
        labels: ``clip_id -> row``；缺失标签的窗口归到 ``"<未标注>"``。

    Returns:
        ``clip_root -> {"heights": np.ndarray, "category_l1": str, "zh_label": str,
        "is_wushu": bool, "windows": int, "split": str}``。
        ``heights`` 是该 clip 全部**有效帧**的根高度（cm），可能被多个重叠窗口重复收录
        （只有 120/150 帧重叠，中位数对这种重复不敏感，故不做去重）。
    """
    shard_paths = sorted(shard_dir.glob("shard_*.npz"))
    if not shard_paths:
        raise FileNotFoundError(f"分片目录为空: {shard_dir}")

    height_dim = C.FEAT_LAYOUT["root_height"].start
    mean = float(norm_stats.mean[height_dim])
    std = float(norm_stats.std[height_dim]) if float(norm_stats.std[height_dim]) else 1.0

    buckets: dict[str, list[np.ndarray]] = defaultdict(list)
    meta: dict[str, dict[str, Any]] = {}

    for path in shard_paths:
        with np.load(path) as payload:
            feat = np.asarray(payload["feat"], dtype=np.float32)  # [n,150,146]
            mask = np.asarray(payload["mask"])  # [n,150]
            clip_ids = np.asarray(payload["clip_id"]).astype(str)
        for local in range(feat.shape[0]):
            valid = np.asarray(mask[local]).astype(bool)
            if not valid.any():
                continue
            row = labels.get(str(clip_ids[local]), {})
            clip_root = str(row.get("clip_root") or row.get("take_id") or clip_ids[local])
            heights = feat[local, valid, height_dim].astype(np.float64) * std + mean
            buckets[clip_root].append(heights)
            if clip_root not in meta:
                meta[clip_root] = {
                    "category_l1": str(row.get("category_l1", "<未标注>")),
                    "category_l3": str(row.get("category_l3", "")),
                    "zh_label": str(row.get("zh_label", "<未标注>")),
                    "is_wushu": bool(row.get("is_wushu", False)),
                    "split": str(row.get("split", "unknown")),
                }

    output: dict[str, dict[str, Any]] = {}
    for clip_root, chunks in buckets.items():
        joined = np.concatenate(chunks) if chunks else np.zeros(0, dtype=np.float64)
        info = meta.get(clip_root, {})
        output[clip_root] = {
            "heights": joined,
            "windows": len(chunks),
            "category_l1": str(info.get("category_l1", "<未标注>")),
            "category_l3": str(info.get("category_l3", "")),
            "zh_label": str(info.get("zh_label", "<未标注>")),
            "is_wushu": bool(info.get("is_wushu", False)),
            "split": str(info.get("split", "unknown")),
        }
    return output


def summarize_by_category(clips: dict[str, dict[str, Any]]) -> dict[str, dict[str, Any]]:
    """按 ``category_l1`` 汇总逐 clip 中位数。

    Args:
        clips: :func:`collect_clip_heights` 的输出。

    Returns:
        ``category -> {"n_clips", "n_windows", "median", "p10", "p90", "min", "max",
        "bands": [count...], "semi_bad_ratio", "verdict", "worst": [(clip, median)...]}``。
    """
    per_category: dict[str, list[tuple[str, float, int]]] = defaultdict(list)
    for clip_root, item in clips.items():
        heights = np.asarray(item["heights"], dtype=np.float64)
        if heights.size == 0:
            continue
        per_category[str(item["category_l1"])].append(
            (clip_root, float(np.median(heights)), int(item["windows"]))
        )

    summary: dict[str, dict[str, Any]] = {}
    for category, rows in per_category.items():
        values = np.array([row[1] for row in rows], dtype=np.float64)
        bands = [0] * len(HEIGHT_BANDS)
        for value in values:
            bands[_band_index(float(value))] += 1
        semi_bad = sum(bands[index] for index in SEMI_BAD_BAND_INDEXES)
        ratio = float(semi_bad) / float(len(rows)) if rows else 0.0
        worst = sorted(rows, key=lambda item: item[1])[:5]
        summary[category] = {
            "n_clips": len(rows),
            "n_windows": int(sum(row[2] for row in rows)),
            "median": float(np.median(values)),
            "p10": float(np.percentile(values, 10)),
            "p90": float(np.percentile(values, 90)),
            "min": float(values.min()),
            "max": float(values.max()),
            "bands": bands,
            "semi_bad_ratio": ratio,
            "verdict": "SUSPECT" if ratio > SUSPECT_RATIO else "CLEAN",
            "worst": [{"clip_root": row[0], "median_cm": row[1]} for row in worst],
        }
    return summary


def summarize_by_label(clips: dict[str, dict[str, Any]]) -> dict[str, dict[str, Any]]:
    """按合并后的中文标签汇总（看「半坏簇」是否集中在某个具体动作上）。

    Args:
        clips: :func:`collect_clip_heights` 的输出。

    Returns:
        ``zh_label -> {"n_clips", "median", "semi_bad", "semi_bad_ratio", "is_wushu"}``。
    """
    per_label: dict[str, list[float]] = defaultdict(list)
    flags: dict[str, bool] = {}
    for item in clips.values():
        heights = np.asarray(item["heights"], dtype=np.float64)
        if heights.size == 0:
            continue
        label = str(item["zh_label"])
        per_label[label].append(float(np.median(heights)))
        flags[label] = flags.get(label, False) or bool(item["is_wushu"])

    summary: dict[str, dict[str, Any]] = {}
    for label, values in per_label.items():
        array = np.asarray(values, dtype=np.float64)
        semi_bad = int(sum(1 for value in array if -20.0 < value <= -5.0))
        summary[label] = {
            "n_clips": int(array.size),
            "median": float(np.median(array)),
            "semi_bad": semi_bad,
            "semi_bad_ratio": float(semi_bad) / float(array.size) if array.size else 0.0,
            "is_wushu": flags.get(label, False),
        }
    return summary


def _format_table(summary: dict[str, dict[str, Any]]) -> list[str]:
    """把按类别的汇总渲染成 Markdown 表（按 clip 数降序）。

    Args:
        summary: :func:`summarize_by_category` 的输出。

    Returns:
        Markdown 行列表。
    """
    header = (
        "| 类别 | clip | 窗 | 中位数 | p10 | p90 | min | < -20 | -20~-10 | "
        "-10~-5 | -5~0 | ≥ 0 | 半坏带占比 | 判定 |"
    )
    divider = "|" + "---|" * 14
    lines = [header, divider]
    for category, item in sorted(
        summary.items(), key=lambda kv: kv[1]["n_clips"], reverse=True
    ):
        bands = item["bands"]
        lines.append(
            "| {cat} | {clips} | {wins} | {median:+.1f} | {p10:+.1f} | {p90:+.1f} | "
            "{minv:+.1f} | {b0} | {b1} | {b2} | {b3} | {b4} | {ratio:.1%} | {verdict} |".format(
                cat=category,
                clips=item["n_clips"],
                wins=item["n_windows"],
                median=item["median"],
                p10=item["p10"],
                p90=item["p90"],
                minv=item["min"],
                b0=bands[0],
                b1=bands[1],
                b2=bands[2],
                b3=bands[3],
                b4=bands[4],
                ratio=item["semi_bad_ratio"],
                verdict=item["verdict"],
            )
        )
    return lines


def main(argv: Iterable[str] | None = None) -> int:
    """命令行入口。

    Args:
        argv: 参数列表（``None`` 时用 ``sys.argv``）。

    Returns:
        进程退出码（恒为 0）。
    """
    parser = argparse.ArgumentParser(description="各类别根高度中位数分布审计")
    parser.add_argument("--config", default="configs/tier_s.yaml", help="配置文件路径")
    parser.add_argument("--shards", default=None, help="覆盖分片目录")
    parser.add_argument("--labels", default=None, help="覆盖标签文件（默认读合并后）")
    parser.add_argument("--norm-stats", default=None, help="覆盖 norm_stats.npz")
    parser.add_argument(
        "--out-dir", default="runs/tier_s/root_audit", help="报告输出目录"
    )
    args = parser.parse_args(list(argv) if argv is not None else None)

    cfg = load_config(args.config)
    shard_dir = Path(args.shards) if args.shards else cfg.shards_path()
    labels_path = Path(args.labels) if args.labels else cfg.train_labels_path()
    norm_path = Path(args.norm_stats) if args.norm_stats else cfg.norm_stats_path()

    print(f"分片目录 : {shard_dir}")
    print(f"标签文件 : {labels_path}")
    print(f"归一统计 : {norm_path}")

    norm_stats = NormStats.load(norm_path)
    labels = _read_labels(labels_path)
    clips = collect_clip_heights(shard_dir, norm_stats, labels)
    by_category = summarize_by_category(clips)
    by_label = summarize_by_label(clips)

    suspects = sorted(
        (name, item) for name, item in by_category.items() if item["verdict"] == "SUSPECT"
    )
    global_medians = np.array(
        [float(np.median(item["heights"])) for item in clips.values() if item["heights"].size],
        dtype=np.float64,
    )

    lines: list[str] = [
        "# 根高度中位数分布审计（M1' 追加诊断）",
        "",
        f"* clip 总数：**{len(clips)}**",
        f"* 全局 clip 中位数：median **{np.median(global_medians):+.1f}cm** / "
        f"p10 {np.percentile(global_medians, 10):+.1f}cm / "
        f"p90 {np.percentile(global_medians, 90):+.1f}cm",
        f"* 判废阈值（足部最低高度）：-10cm；**半坏带** = (-20cm, -5cm] "
        f"（阈值放行、但脚已下陷的盲区）",
        f"* 判定：某类别半坏带 clip 占比 > {SUSPECT_RATIO:.0%} → `SUSPECT`（需按 clip 地面校正）",
        "",
        "## 按大类（category_l1）",
        "",
    ]
    lines.extend(_format_table(by_category))

    lines.extend(["", "## 半坏带逐条 clip 清单（最多 40 条，按高度升序）", ""])
    semi_rows: list[tuple[str, float, str, str]] = []
    for clip_root, item in clips.items():
        heights = np.asarray(item["heights"], dtype=np.float64)
        if heights.size == 0:
            continue
        median = float(np.median(heights))
        if -20.0 < median <= -5.0:
            semi_rows.append(
                (clip_root, median, str(item["category_l1"]), str(item["zh_label"]))
            )
    semi_rows.sort(key=lambda row: row[1])
    if semi_rows:
        lines.append("| clip | 中位数 | 类别 | 标签 |")
        lines.append("|---|---|---|---|")
        for clip_root, median, category, label in semi_rows[:40]:
            lines.append(f"| {clip_root} | {median:+.1f}cm | {category} | {label} |")
        lines.append("")
        lines.append(f"合计 **{len(semi_rows)}** 条 clip 落在半坏带。")
    else:
        lines.append("**零条** clip 落在半坏带（(-20cm, -5cm]）。")

    lines.extend(["", "## 武术（is_wushu）专项", ""])
    wushu_rows = [item for item in clips.values() if item["is_wushu"] and item["heights"].size]
    if wushu_rows:
        wushu_medians = np.array(
            [float(np.median(item["heights"])) for item in wushu_rows], dtype=np.float64
        )
        wushu_semi = int(sum(1 for value in wushu_medians if -20.0 < value <= -5.0))
        lines.append(
            f"* clip 数 **{len(wushu_rows)}**，中位数 **{np.median(wushu_medians):+.1f}cm**，"
            f"p10 {np.percentile(wushu_medians, 10):+.1f}cm，"
            f"p90 {np.percentile(wushu_medians, 90):+.1f}cm，"
            f"半坏带 {wushu_semi} 条（{wushu_semi / len(wushu_rows):.1%}）"
        )
    else:
        lines.append("* 无武术 clip。")

    lines.extend(["", "## 结论", ""])
    if suspects:
        lines.append(
            "发现 **{}** 个类别存在偏移簇（半坏带占比 > {:.0%}）：".format(
                len(suspects), SUSPECT_RATIO
            )
        )
        for name, item in suspects:
            semi_count = sum(item["bands"][index] for index in SEMI_BAD_BAND_INDEXES)
            lines.append(
                f"* `{name}`：{item['n_clips']} clip，半坏带 {semi_count} 条"
                f"（{item['semi_bad_ratio']:.1%}），中位数 {item['median']:+.1f}cm"
            )
        lines.append("")
        lines.append("→ 这些类别需要按 clip 做地面校正，再谈是否入训。")
    else:
        lines.append(
            "全部类别 `CLEAN`：没有任何类别在半坏带里聚集（均 ≤ {:.0%}）。".format(SUSPECT_RATIO)
        )
        lines.append("")
        lines.append(
            "→ 判废阈值（-10cm）没有漏掉成簇的半坏数据；保留球类的决定有数据支撑，无需地面校正。"
        )

    out_dir = P.safe_makedirs(args.out_dir)
    md_path = out_dir / "root_height_audit.md"
    md_path.write_text("\n".join(lines) + "\n", encoding="utf-8")

    payload = {
        "n_clips": len(clips),
        "global": {
            "median": float(np.median(global_medians)),
            "p10": float(np.percentile(global_medians, 10)),
            "p90": float(np.percentile(global_medians, 90)),
        },
        "bands": [name for _low, _high, name in HEIGHT_BANDS],
        "suspect_ratio_threshold": SUSPECT_RATIO,
        "by_category": by_category,
        "by_label": by_label,
        "semi_bad_clips": [
            {"clip_root": clip_root, "median_cm": median, "category_l1": category, "label": label}
            for clip_root, median, category, label in semi_rows
        ],
    }
    json_path = out_dir / "root_height_audit.json"
    json_path.write_bytes(orjson.dumps(payload, option=orjson.OPT_INDENT_2))

    print()
    print("\n".join(_format_table(by_category)))
    print()
    print(f"报告: {P.rel_to_root(md_path)} / {P.rel_to_root(json_path)}")
    print(
        "判定: " + ("SUSPECT × {}".format(len(suspects)) if suspects else "全部 CLEAN")
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
