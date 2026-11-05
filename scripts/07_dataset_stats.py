#!/usr/bin/env python
"""数据集统计报告（M1/T04 收尾验收用）。

读取某一个档位的扫描/预处理产物，输出可直接贴进汇报的统计：

1. **规模**：clip 数 / 窗口数 / train-val-test 切分 / 总时长；
2. **淘汰**：``failed.jsonl`` 按阶段与原因聚类的分布（root-only、静止占位等）；
3. **中文标签覆盖**：distinct 标签数、每标签的 clip/窗口样本量、Top-N 与**长尾**
   （样本量 ≤ 阈值的类别，用来判断是否有类别只有 1–2 条）；
4. **类别层级**：``category_l1`` / ``category_l2`` 的样本量分布；
5. **归一化统计**：z-score 四维的 mean/std。

用法::

    python scripts/07_dataset_stats.py --config configs/tier_s.yaml
    python scripts/07_dataset_stats.py --config configs/tier_s.yaml --tail 2 --json
"""

from __future__ import annotations

import argparse
import collections
import json
import re
import sys
from pathlib import Path
from typing import Any

_SRC = Path(__file__).resolve().parents[1] / "src"
if str(_SRC) not in sys.path:  # 兼容未做 editable install
    sys.path.insert(0, str(_SRC))

from ttm.common.config import load_config  # noqa: E402

#: 原因归类：把 ``failed.jsonl`` 的自由文本压成稳定的类别标签。
_REASON_RULES: tuple[tuple[str, str], ...] = (
    ("n_joints", "root-only / 非 23 关节骨架"),
    ("静止片段", "静止占位数据"),
    ("足部低于地面", "根高度约定损坏（脚穿地）"),
    ("NaN", "数值非法"),
    ("no_window", "有效帧不足（切不出窗口）"),
    ("parse:", "解析失败"),
    ("通道", "通道布局不符"),
    ("关节", "关节集合不符"),
    ("MOTION", "缺 MOTION 小节"),
)


def _classify(error: str) -> str:
    """把失败原因文本归类（保持稳定，便于跨次对比）。"""
    for token, label in _REASON_RULES:
        if token in error:
            return label
    return re.sub(r"[\d.]+", "N", error)[:48] or "未知"


def _read_jsonl(path: Path) -> list[dict[str, Any]]:
    """读 JSONL（不存在时返回空列表）。"""
    if not path.exists():
        return []
    rows: list[dict[str, Any]] = []
    with open(path, "r", encoding="utf-8") as handle:
        for line in handle:
            text = line.strip()
            if text:
                rows.append(json.loads(text))
    return rows


def collect(cfg: Any, *, tail_threshold: int = 2) -> dict[str, Any]:
    """汇总一个档位的全部统计。

    Args:
        cfg: 生效配置。
        tail_threshold: 长尾判定阈值（窗口样本量 ≤ 该值即视为长尾）。

    Returns:
        可直接 JSON 序列化的统计字典。
    """
    index_rows = _read_jsonl(cfg.index_path())
    label_rows = _read_jsonl(cfg.labels_path())
    failed_rows = _read_jsonl(cfg.failed_path())

    # --- 规模
    split_clips: collections.Counter[str] = collections.Counter(
        str(row.get("split", "?")) for row in index_rows
    )
    split_windows: collections.Counter[str] = collections.Counter(
        str(row.get("split", "?")) for row in label_rows
    )
    total_windows = sum(int(row.get("n_windows", 0)) for row in index_rows)
    total_seconds = sum(float(row.get("duration", 0.0)) for row in label_rows)

    # --- 淘汰分布
    by_stage: collections.Counter[str] = collections.Counter(
        str(row.get("stage", "?")) for row in failed_rows
    )
    by_reason: collections.Counter[str] = collections.Counter(
        _classify(str(row.get("error", ""))) for row in failed_rows
    )

    # --- 标签覆盖
    label_windows: collections.Counter[str] = collections.Counter()
    label_clips: dict[str, set[str]] = collections.defaultdict(set)
    for row in label_rows:
        label = str(row.get("zh_label") or "(空)")
        label_windows[label] += 1
        label_clips[label].add(str(row.get("clip_root") or row.get("clip_id") or ""))
    label_size = {label: len(clips) for label, clips in label_clips.items()}
    long_tail = sorted(
        (label for label, windows in label_windows.items() if windows <= tail_threshold),
        key=lambda label: label_windows[label],
    )

    # --- 类别层级（按窗口）
    l1_windows: collections.Counter[str] = collections.Counter(
        str(row.get("category_l1") or "(空)") for row in label_rows
    )
    l3_clips: collections.Counter[str] = collections.Counter(
        str(row.get("category_l3") or "(空)") for row in index_rows
    )

    # --- 窗口/时长分布
    windows_per_clip = [int(row.get("n_windows", 0)) for row in index_rows]
    n_texts = [len(row.get("zh_texts") or []) for row in label_rows]

    # --- 归一化统计
    norm_stats: dict[str, Any] = {}
    if cfg.norm_stats_path().exists():
        import numpy as np  # noqa: PLC0415

        with np.load(cfg.norm_stats_path()) as payload:
            mean = payload["mean"]
            std = payload["std"]
            norm_stats = {
                "layout_version": int(payload["layout_version"]),
                "dims": {
                    int(dim): {"mean": round(float(mean[dim]), 4), "std": round(float(std[dim]), 4)}
                    for dim in (0, 1, 2, 5)
                },
            }

    return {
        "tier": cfg.data.tier,
        "index": str(cfg.index_path()),
        "clips": {
            "total": len(index_rows),
            "by_split": dict(split_clips),
        },
        "windows": {
            "total": total_windows,
            "by_split": dict(split_windows),
            "per_clip_max": max(windows_per_clip) if windows_per_clip else 0,
            "per_clip_mean": round(
                sum(windows_per_clip) / len(windows_per_clip), 2
            )
            if windows_per_clip
            else 0.0,
            "total_seconds": round(total_seconds, 1),
            "total_hours": round(total_seconds / 3600.0, 2),
        },
        "failed": {
            "total": len(failed_rows),
            "by_stage": dict(by_stage),
            "by_reason": dict(by_reason.most_common()),
        },
        "labels": {
            "distinct": len(label_windows),
            "windows_top10": label_windows.most_common(10),
            "clips_top10": sorted(label_size.items(), key=lambda kv: -kv[1])[:10],
            "tail_threshold": tail_threshold,
            "tail_labels": {label: label_windows[label] for label in long_tail},
            "texts_per_window_min": min(n_texts) if n_texts else 0,
            "texts_per_window_max": max(n_texts) if n_texts else 0,
        },
        "categories": {
            "l1_windows": l1_windows.most_common(),
            "l3_clips_top10": l3_clips.most_common(10),
        },
        "norm_stats": norm_stats,
    }


def render(stats: dict[str, Any]) -> str:
    """把统计渲染成人类可读的报表文本。"""
    clips = stats["clips"]
    windows = stats["windows"]
    failed = stats["failed"]
    labels = stats["labels"]
    lines: list[str] = []
    lines.append("=" * 92)
    lines.append(f"TTM 数据集统计 · tier={stats['tier']}")
    lines.append("=" * 92)
    total = clips["total"]
    total_attempts = total + failed["total"]
    rate = (failed["total"] / total_attempts * 100.0) if total_attempts else 0.0
    lines.append(f"有效 clip           : {total}（淘汰 {failed['total']}，总尝试 {total_attempts}）")
    lines.append(f"淘汰率              : {rate:.2f}%")
    lines.append(f"切分                : " + " / ".join(f"{k}={v}" for k, v in sorted(clips["by_split"].items())))
    lines.append(
        f"训练窗口            : {windows['total']}（单 clip 最多 {windows['per_clip_max']}，"
        f"平均 {windows['per_clip_mean']}）"
    )
    lines.append(f"窗口切分            : " + " / ".join(f"{k}={v}" for k, v in sorted(windows["by_split"].items())))
    lines.append(f"总时长              : {windows['total_hours']} 小时（{windows['total_seconds']} 秒）")
    lines.append("")
    lines.append("淘汰原因分布：")
    for reason, count in failed["by_reason"].items():
        lines.append(f"  {reason:<32} {count:>6}")
    if failed["by_stage"]:
        lines.append("  按阶段: " + ", ".join(f"{k}={v}" for k, v in sorted(failed["by_stage"].items())))
    lines.append("")
    lines.append(f"中文标签            : {labels['distinct']} 个 distinct 标签")
    lines.append(f"每窗文本变体数      : {labels['texts_per_window_min']}–{labels['texts_per_window_max']} 条")
    lines.append("样本量 Top10（按窗口）：")
    for label, count in labels["windows_top10"]:
        lines.append(f"  {label:<16} {count:>6} 窗")
    lines.append(f"**长尾**（窗口数 ≤ {labels['tail_threshold']}）：{len(labels['tail_labels'])} 个标签")
    for label, count in labels["tail_labels"].items():
        lines.append(f"  ! {label:<16} {count:>6} 窗")
    lines.append("")
    lines.append("大类分布（按窗口）：")
    for name, count in stats["categories"]["l1_windows"]:
        lines.append(f"  {name:<44} {count:>6}")
    if stats["norm_stats"]:
        dims = stats["norm_stats"]["dims"]
        names = {0: "根前向速度 cm/s", 1: "根侧向速度 cm/s", 2: "根高度 cm", 5: "偏航角速度 rad/s"}
        lines.append("")
        lines.append("归一化统计（z-score 四维）：")
        for dim, payload in dims.items():
            lines.append(
                f"  [{dim}] {names.get(dim, ''):<18} mean={payload['mean']:>10} std={payload['std']:>10}"
            )
    lines.append("=" * 92)
    return "\n".join(lines)


def main(argv: list[str] | None = None) -> int:
    """命令行入口。

    Args:
        argv: 参数列表（默认取 ``sys.argv[1:]``）。

    Returns:
        进程退出码（产物缺失时返回 2）。
    """
    parser = argparse.ArgumentParser(description="TTM 数据集统计报告（M1 验收）")
    parser.add_argument("--config", default="configs/tier_s.yaml", help="配置文件路径")
    parser.add_argument("--tail", type=int, default=2, help="长尾阈值（窗口数 ≤ N 视为长尾）")
    parser.add_argument("--json", action="store_true", help="输出 JSON 而不是报表文本")
    args = parser.parse_args(argv)

    cfg = load_config(args.config)
    if not cfg.index_path().exists():
        print(f"索引不存在: {cfg.index_path()}。请先运行 scripts/01_scan.py", file=sys.stderr)
        return 2
    stats = collect(cfg, tail_threshold=args.tail)
    if args.json:
        print(json.dumps(stats, ensure_ascii=False, indent=2))
    else:
        print(render(stats))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
