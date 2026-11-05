#!/usr/bin/env python
"""VAE 重建可视化：抽 5 个 val 窗口，导出 pred / gt 的 BVH，并给出**分动作阶段**的误差归因。

## 这个脚本为哪条验收线服务

M2' 的目标线是 ``root_mae ≤ 3cm``，并带一条**降级验收**：

> 若仍 >3cm 但 ≤4cm，允许带条件通过——条件是补一个感知层面的检查：从 val 集抽 5 个
> 窗口做重建可视化（BVH 导出后肉眼看脚底/地面接触），确认没有明显漂浮或穿地，并在
> 报告里写明「4.x cm 的误差主要分布在什么动作阶段」（如果是腾空/跳跃帧，误差天然大，
> 可以接受；如果是站立帧，不可接受）。

所以本脚本输出三样东西，缺一不可：

1. **BVH 文件**（pred 与 gt 各一份/样本）：人能直接拖进 Blender / UE / BVH 播放器，
   肉眼看脚底贴不贴地。文件名带样本编号与中文标签，方便对照。
2. **分阶段误差表**：把有效帧按 GT 的足接触通道（``FEAT_LAYOUT["foot_contact"]``）
   分成 *腾空*（两脚都不接触）/ *着地*（至少一脚接触）/ *站立*（着地且根高度接近
   该样本的中位数，即没有蹲起跳的准备动作）三段，分别给根高度 MAE。
   —— 若「腾空帧误差大、站立帧误差小」，那 4.x cm 是姿态语义导致的，不是模型把
   地面搞错了；反之则是真问题。
3. **FK 足底高度**：用 :meth:`ttm.data.skeleton.SkeletonDef.fk` 把预测姿态算到世界
   坐标，取两脚（``FOOT_CONTACT_JOINTS``）的最低 y，统计**着地帧**里低于 -5cm
   （穿地）或高于 +5cm（漂浮）的帧占比。这是「肉眼检查」的数值化替身——肉眼看
   5 个片段只能抓到明显问题，这个占比能抓住「整体偏低 8cm」这种肉眼容易忽略的系统偏移。

## 用法

```bash
python scripts/03b_vae_recon_viz.py --config configs/tier_s.yaml \
    --ckpt runs/tier_s/vae/best.pt --num 5
```

产物：``runs/tier_s/vae/recon_viz/`` 下的 ``*.bvh``、``recon_viz.json``、``recon_viz.md``。
"""

from __future__ import annotations

import argparse
import re
import sys
from collections import defaultdict
from pathlib import Path
from typing import Any, Iterable

import numpy as np
import orjson
import torch

from ttm.common import constants as C
from ttm.common import paths as P
from ttm.common.config import load_config
from ttm.data.dataset import MotionDataset, auto_cache_size
from ttm.data.features import MotionFeature, NormStats
from ttm.data.skeleton import SkeletonDef
from ttm.export.bvh_io import motion_to_euler_deg, write_bvh
from ttm.train.train_vae import load_checkpoint, reconstruction_metrics

#: 判定「站立」的根高度容差（cm）：与样本根高度中位数的偏差在此之内。
STANDING_TOLERANCE_CM: float = 5.0

#: 足底高度判据：低于 -PENETRATION_CM 记为穿地，高于 +FLOAT_CM 记为漂浮。
FOOT_PENETRATION_CM: float = -5.0
FOOT_FLOAT_CM: float = 5.0


def _safe_name(text: str, fallback: str = "sample") -> str:
    """把中文标签清洗成可用作文件名的片段。

    Args:
        text: 原始文本（可能是中文、含空格或斜杠的标签）。
        fallback: 清洗后为空时的替代值。

    Returns:
        只含字母/数字/下划线/汉字/短横的字符串。
    """
    cleaned = re.sub(r"[^\w\u4e00-\u9fff\-]+", "_", str(text)).strip("_")
    return cleaned or fallback


def pick_samples(
    dataset: MotionDataset,
    num: int,
    *,
    prefer_wushu: int = 1,
    explicit: list[int] | None = None,
) -> list[tuple[int, str]]:
    """从 val 集里挑 ``num`` 个样本（按合并后标签分层，确定性）。

    Args:
        dataset: val :class:`MotionDataset`。
        num: 需要的样本数。
        prefer_wushu: 至少包含多少个武术样本（有的话）。
        explicit: 显式指定的样本下标（给出时直接用它，忽略分层逻辑）。

    Returns:
        ``[(index, label), ...]``。
    """
    labels = dataset.sample_labels()
    if explicit:
        return [
            (int(index), str(labels[int(index)]) if 0 <= int(index) < len(labels) else "")
            for index in explicit
        ]

    groups: dict[str, list[int]] = defaultdict(list)
    for index, label in enumerate(labels):
        groups[str(label)].append(index)
    if not groups:
        return []
    # 按样本数降序排列标签，再在这个序列上等距取点：既覆盖头部大类，也覆盖长尾。
    ordered = sorted(groups.items(), key=lambda kv: (-len(kv[1]), kv[0]))
    picks: list[tuple[int, str]] = []
    count = max(1, int(num))
    if count == 1:
        positions = [0]
    else:
        positions = [round(step * (len(ordered) - 1) / (count - 1)) for step in range(count)]
    chosen_labels: set[str] = set()
    for position in positions:
        label, indices = ordered[min(max(position, 0), len(ordered) - 1)]
        if label in chosen_labels:
            continue
        chosen_labels.add(label)
        picks.append((indices[len(indices) // 2], label))

    # 补足数量（去重后可能不足 num）＋ 保证至少含 prefer_wushu 个武术样本。
    wushu_mask = dataset.wushu_mask()
    wushu_present = sum(
        1 for index, _label in picks if index < wushu_mask.shape[0] and bool(wushu_mask[index])
    )
    wushu_candidates = [index for index in range(len(labels)) if bool(wushu_mask[index])]
    if wushu_present < prefer_wushu and wushu_candidates:
        replacement_index = wushu_candidates[len(wushu_candidates) // 2]
        if picks:
            picks[-1] = (replacement_index, str(labels[replacement_index]))
        else:
            picks.append((replacement_index, str(labels[replacement_index])))

    used = {index for index, _label in picks}
    for index in range(len(labels)):
        if len(picks) >= count:
            break
        if index in used:
            continue
        picks.append((index, str(labels[index])))
        used.add(index)
    return picks[:count]


def phase_breakdown(
    pred_denorm: np.ndarray,
    gt_denorm: np.ndarray,
) -> dict[str, Any]:
    """按动作阶段拆解根高度误差。

    Args:
        pred_denorm: ``[L,146]`` 预测（已反归一化，L = 有效帧数）。
        gt_denorm: ``[L,146]`` 真值（已反归一化）。

    Returns:
        ``{"airborne": {...}, "grounded": {...}, "standing": {...}, "counts": {...}}``，
        每段含 ``n_frames`` / ``root_height_mae_cm`` / ``root_height_max_cm``。
    """
    contact = gt_denorm[:, C.FEAT_LAYOUT["foot_contact"]]
    contact_left = contact[:, 0] > 0.5
    contact_right = contact[:, 1] > 0.5
    grounded = contact_left | contact_right
    airborne = ~grounded

    height_dim = C.FEAT_LAYOUT["root_height"].start
    pred_height = pred_denorm[:, height_dim]
    gt_height = gt_denorm[:, height_dim]
    error = np.abs(pred_height - gt_height)

    median_height = float(np.median(gt_height)) if gt_height.size else 0.0
    standing = grounded & (np.abs(gt_height - median_height) <= STANDING_TOLERANCE_CM)

    def _segment(mask: np.ndarray) -> dict[str, float]:
        selected = error[mask]
        return {
            "n_frames": int(mask.sum()),
            "root_height_mae_cm": float(selected.mean()) if selected.size else 0.0,
            "root_height_max_cm": float(selected.max()) if selected.size else 0.0,
        }

    return {
        "airborne": _segment(airborne),
        "grounded": _segment(grounded),
        "standing": _segment(standing),
        "counts": {
            "n_frames": int(error.size),
            "n_airborne": int(airborne.sum()),
            "n_grounded": int(grounded.sum()),
            "n_standing": int(standing.sum()),
            "gt_median_height_cm": median_height,
        },
    }


def foot_ground_contact(
    quats: np.ndarray, root_pos: np.ndarray, gt_grounded: np.ndarray
) -> dict[str, float]:
    """用 FK 算足底世界高度，统计着地帧的穿地 / 漂浮比例。

    Args:
        quats: ``[L,23,4]`` 四元数（xyzw）。
        root_pos: ``[L,3]`` 根位移（cm）。
        gt_grounded: ``[L]`` 布尔，GT 判定为「至少一脚接触」。

    Returns:
        ``{"n_grounded", "foot_min_mean_cm", "penetration_ratio", "float_ratio"}``。
    """
    if quats.shape[0] == 0:
        return {
            "n_grounded": 0,
            "foot_min_mean_cm": 0.0,
            "penetration_ratio": 0.0,
            "float_ratio": 0.0,
        }
    skeleton = SkeletonDef.default()
    euler_deg = motion_to_euler_deg(quats)
    euler_rad = np.deg2rad(euler_deg)
    positions = skeleton.fk(euler_rad, root_pos)  # [L,23,3]
    feet = positions[:, list(C.FOOT_CONTACT_JOINTS), 1]  # [L,2]
    foot_min = feet.min(axis=1)
    if not np.any(gt_grounded):
        return {
            "n_grounded": 0,
            "foot_min_mean_cm": float(foot_min.mean()),
            "penetration_ratio": 0.0,
            "float_ratio": 0.0,
        }
    selected = foot_min[gt_grounded]
    return {
        "n_grounded": int(selected.size),
        "foot_min_mean_cm": float(selected.mean()),
        "penetration_ratio": float((selected < FOOT_PENETRATION_CM).mean()),
        "float_ratio": float((selected > FOOT_FLOAT_CM).mean()),
    }


def run_one(
    dataset: MotionDataset,
    model: torch.nn.Module,
    norm_stats: NormStats,
    index: int,
    label: str,
    *,
    device: torch.device,
    out_dir: Path,
    write_files: bool = True,
) -> dict[str, Any]:
    """重构一个样本、导出 BVH、算指标。

    Args:
        dataset: val 数据集。
        model: 已 eval 的 VAE。
        norm_stats: 归一化统计。
        index: 样本下标。
        label: 合并后中文标签（用于文件名与报告）。
        device: 计算设备。
        out_dir: BVH 输出目录。
        write_files: 是否真的写 BVH 文件。做分布扫描时（``--num 60``）关掉——
            那是为了看误差在全体 val 上的分布，不需要 120 个 BVH 文件。

    Returns:
        该样本的指标字典。
    """
    sample = dataset[index]
    feat = torch.as_tensor(sample["feat"]).unsqueeze(0).to(device)  # [1,146,150]
    mask = torch.as_tensor(sample["mask"]).unsqueeze(0).to(device)  # [1,150]
    with torch.no_grad():
        output = model(feat, sample=False)  # 确定性：用 mu，不采样
    pred = output.recon[0].float().cpu().numpy().T  # [150,146]
    truth = feat[0].float().cpu().numpy().T  # [150,146]
    length = int(mask.sum().item())
    length = max(1, min(length, pred.shape[0]))

    unit_mask = np.ones(length, dtype=np.uint8)
    pred_slice = MotionFeature(feat=pred[:length], mask=unit_mask)
    truth_slice = MotionFeature(feat=truth[:length], mask=unit_mask)
    pred_denorm = pred_slice.denormalize(norm_stats).feat
    truth_denorm = truth_slice.denormalize(norm_stats).feat

    metrics = reconstruction_metrics(pred_denorm, truth_denorm)
    phases = phase_breakdown(pred_denorm, truth_denorm)

    pred_quats, pred_root = MotionFeature(feat=pred_denorm, mask=unit_mask).to_rotations_root()
    gt_quats, gt_root = MotionFeature(feat=truth_denorm, mask=unit_mask).to_rotations_root()

    contact = truth_denorm[:, C.FEAT_LAYOUT["foot_contact"]]
    grounded = (contact[:, 0] > 0.5) | (contact[:, 1] > 0.5)
    pred_feet = foot_ground_contact(pred_quats, pred_root, grounded)
    gt_feet = foot_ground_contact(gt_quats, gt_root, grounded)

    stem = f"val{index:05d}_{_safe_name(label)}"
    if write_files:
        pred_path = write_bvh(out_dir / f"{stem}_pred.bvh", pred_quats, pred_root)
        gt_path = write_bvh(out_dir / f"{stem}_gt.bvh", gt_quats, gt_root)
        bvh_pred = P.rel_to_root(pred_path)
        bvh_gt = P.rel_to_root(gt_path)
    else:
        bvh_pred = ""
        bvh_gt = ""

    return {
        "index": int(index),
        "clip_id": str(sample["clip_id"]),
        "win_idx": int(sample["win_idx"]),
        "label": str(label),
        "valid_frames": int(length),
        "joint_mae_deg": metrics["joint_mae_deg"],
        "root_height_mae_cm": metrics["root_height_mae_cm"],
        "root_speed_mae_cmps": metrics["root_speed_mae_cmps"],
        "phases": phases,
        "foot_pred": pred_feet,
        "foot_gt": gt_feet,
        "bvh_pred": bvh_pred,
        "bvh_gt": bvh_gt,
    }


def _format_markdown(rows: list[dict[str, Any]], summary: dict[str, Any]) -> list[str]:
    """渲染人读报告。

    Args:
        rows: 逐样本结果。
        summary: 汇总指标。

    Returns:
        Markdown 行列表。
    """
    lines: list[str] = [
        "# VAE 重建可视化（val 5 窗）",
        "",
        f"* 样本数：**{summary['n_samples']}**",
        f"* 关节角 MAE：**{summary['joint_mae_deg']:.2f}°**（目标 ≤ 8°）",
        f"* 根高度 MAE：**{summary['root_height_mae_cm']:.2f}cm**（目标 ≤ 3cm，"
        f"降级验收 ≤ 4cm）",
        f"* 根高度分布：p10 **{summary['root_percentiles_cm']['p10']:.2f}cm** / "
        f"p50 **{summary['root_percentiles_cm']['p50']:.2f}cm** / "
        f"p90 **{summary['root_percentiles_cm']['p90']:.2f}cm** / "
        f"max **{summary['root_percentiles_cm']['max']:.2f}cm**",
        f"* 根速度 MAE：{summary['root_speed_mae_cmps']:.2f} cm/s",
        "",
        "## 逐样本",
        "",
        "| # | 标签 | 有效帧 | joint ° | root cm | 腾空帧 root cm | 着地帧 root cm | 站立帧 root cm | BVH(pred) |",
        "|---|---|---|---|---|---|---|---|---|",
    ]
    for row in rows:
        phases = row["phases"]
        lines.append(
            "| {idx} | {label} | {frames} | {joint:.2f} | {root:.2f} | {air:.2f} ({na}) | "
            "{ground:.2f} ({ng}) | {stand:.2f} ({ns}) | `{path}` |".format(
                idx=row["index"],
                label=row["label"],
                frames=row["valid_frames"],
                joint=row["joint_mae_deg"],
                root=row["root_height_mae_cm"],
                air=phases["airborne"]["root_height_mae_cm"],
                na=phases["airborne"]["n_frames"],
                ground=phases["grounded"]["root_height_mae_cm"],
                ng=phases["grounded"]["n_frames"],
                stand=phases["standing"]["root_height_mae_cm"],
                ns=phases["standing"]["n_frames"],
                path=row["bvh_pred"],
            )
        )
    lines.extend(
        [
            "",
            "## 脚底/地面接触（FK 数值化替身）",
            "",
            "| # | 标签 | 预测足底均值 | 预测穿地帧 | 预测漂浮帧 | 真值足底均值 | 真值穿地帧 | 真值漂浮帧 |",
            "|---|---|---|---|---|---|---|---|",
        ]
    )
    for row in rows:
        pred_feet = row["foot_pred"]
        gt_feet = row["foot_gt"]
        lines.append(
            "| {idx} | {label} | {pm:+.1f}cm | {pp:.0%} | {pf:.0%} | {gm:+.1f}cm | "
            "{gp:.0%} | {gf:.0%} |".format(
                idx=row["index"],
                label=row["label"],
                pm=pred_feet["foot_min_mean_cm"],
                pp=pred_feet["penetration_ratio"],
                pf=pred_feet["float_ratio"],
                gm=gt_feet["foot_min_mean_cm"],
                gp=gt_feet["penetration_ratio"],
                gf=gt_feet["float_ratio"],
            )
        )
    lines.extend(
        [
            "",
            "## 最差 5 个样本（按根高度 MAE 降序）",
            "",
            "| # | 标签 | clip | win | root cm | joint ° |",
            "|---|---|---|---|---|---|",
        ]
    )
    for row in summary.get("worst_samples", []):
        lines.append(
            "| {idx} | {label} | `{clip}` | {win} | {root:.2f} | {joint:.2f} |".format(
                idx=row["index"],
                label=row["label"],
                clip=row["clip_id"],
                win=row["win_idx"],
                root=row["root_height_mae_cm"],
                joint=row["joint_mae_deg"],
            )
        )
    lines.extend(["", "## 结论", ""])
    lines.extend(summary["verdict_lines"])
    return lines


def main(argv: Iterable[str] | None = None) -> int:
    """命令行入口。

    Args:
        argv: 参数列表（``None`` 时用 ``sys.argv``）。

    Returns:
        0 = 一贯成功（不设门禁，判定在报告里）。
    """
    parser = argparse.ArgumentParser(description="VAE 重建可视化 + 分阶段误差归因")
    parser.add_argument("--config", default="configs/tier_s.yaml", help="配置文件路径")
    parser.add_argument("--ckpt", default="runs/tier_s/vae/best.pt", help="VAE checkpoint")
    parser.add_argument("--num", type=int, default=5, help="抽样数量")
    parser.add_argument(
        "--indices", default=None, help="显式样本下标（逗号分隔），给出时忽略分层抽样"
    )
    parser.add_argument("--out-dir", default="runs/tier_s/vae/recon_viz", help="输出目录")
    parser.add_argument(
        "--no-bvh",
        action="store_true",
        help="不写 BVH 文件（做分布扫描时用，如 --num 60 --no-bvh）",
    )
    parser.add_argument(
        "--all",
        action="store_true",
        help="评估**全部** val 窗口（不抽样、不写 BVH）。这是唯一可信的验收口径："
        "抽样 64 个窗口会让 root_mae 随抽样运气在 4cm 和 510cm 之间跳。",
    )
    parser.add_argument(
        "--device", default=None, help="覆盖设备（默认按配置，cuda 不可用回退 cpu）"
    )
    args = parser.parse_args(list(argv) if argv is not None else None)

    cfg = load_config(args.config)
    device = torch.device(
        args.device
        if args.device
        else (cfg.runtime.device if torch.cuda.is_available() else "cpu")
    )

    norm_stats = NormStats.load(cfg.norm_stats_path())
    dataset = MotionDataset(
        cfg.shards_path(),
        cache_size=auto_cache_size(cfg.shards_path()),
        window=cfg.data.window,
        feature_dim=cfg.motion.feature_dim,
        labels_path=cfg.train_labels_path(),
        split="val",
    )
    if len(dataset) == 0:
        print("val split 为空，无法可视化", file=sys.stderr)
        return 1

    model, payload = load_checkpoint(Path(args.ckpt), map_location=device)
    model.to(device)
    model.eval()
    print(
        f"ckpt: {args.ckpt}（epoch {payload.get('epoch', '?')}）；"
        f"val 样本 {len(dataset)}；设备 {device}"
    )

    explicit = (
        [int(item) for item in str(args.indices).split(",") if str(item).strip()]
        if args.indices
        else None
    )
    if args.all:
        val_labels = dataset.sample_labels()
        picks = [(index, str(val_labels[index])) for index in range(len(dataset))]
        print(f"--all: 评估全部 {len(picks)} 个 val 窗口（不写 BVH）")
    else:
        picks = pick_samples(dataset, args.num, explicit=explicit)
    out_dir = P.safe_makedirs(args.out_dir)

    rows: list[dict[str, Any]] = []
    for index, label in picks:
        row = run_one(
            dataset,
            model,
            norm_stats,
            index,
            label,
            device=device,
            out_dir=out_dir,
            write_files=not (args.no_bvh or args.all),
        )
        rows.append(row)
        if args.all and len(rows) % 100 == 0:
            print(f"  ... 已评估 {len(rows)}/{len(picks)} 个窗口")
        # 全量评估时不逐条刷屏，只把「根高度误差 > 50cm」的坏样本挑出来打印
        if not args.all or row["root_height_mae_cm"] > 50.0:
            print(
                f"[{row['index']:5d}] {row['label'][:12]:<12} "
            f"joint {row['joint_mae_deg']:6.2f}°  root {row['root_height_mae_cm']:7.2f}cm  "
            f"腾空 {row['phases']['airborne']['root_height_mae_cm']:6.2f} / "
            f"着地 {row['phases']['grounded']['root_height_mae_cm']:6.2f} / "
                f"站立 {row['phases']['standing']['root_height_mae_cm']:6.2f}"
            )

    summary = {
        "n_samples": len(rows),
        "joint_mae_deg": float(np.mean([row["joint_mae_deg"] for row in rows])) if rows else 0.0,
        "root_height_mae_cm": (
            float(np.mean([row["root_height_mae_cm"] for row in rows])) if rows else 0.0
        ),
        "root_speed_mae_cmps": (
            float(np.mean([row["root_speed_mae_cmps"] for row in rows])) if rows else 0.0
        ),
    }

    # 均值会被长尾拉飞，所以同时给分位数与最差样本；分布扫描（--num 60 --no-bvh）时
    # 这一块是判断「4.x cm 是普遍水平还是少数片段拖累」的主要依据。
    root_values = np.array([row["root_height_mae_cm"] for row in rows], dtype=np.float64)
    joint_values = np.array([row["joint_mae_deg"] for row in rows], dtype=np.float64)
    summary["root_percentiles_cm"] = {
        "p10": float(np.percentile(root_values, 10)) if rows else 0.0,
        "p50": float(np.percentile(root_values, 50)) if rows else 0.0,
        "p90": float(np.percentile(root_values, 90)) if rows else 0.0,
        "max": float(root_values.max()) if rows else 0.0,
    }
    summary["joint_percentiles_deg"] = {
        "p10": float(np.percentile(joint_values, 10)) if rows else 0.0,
        "p50": float(np.percentile(joint_values, 50)) if rows else 0.0,
        "p90": float(np.percentile(joint_values, 90)) if rows else 0.0,
        "max": float(joint_values.max()) if rows else 0.0,
    }
    summary["worst_samples"] = [
        {
            "index": row["index"],
            "label": row["label"],
            "clip_id": row["clip_id"],
            "win_idx": row["win_idx"],
            "root_height_mae_cm": row["root_height_mae_cm"],
            "joint_mae_deg": row["joint_mae_deg"],
        }
        for row in sorted(rows, key=lambda item: item["root_height_mae_cm"], reverse=True)[:5]
    ]
    per_label: dict[str, list[float]] = defaultdict(list)
    for row in rows:
        per_label[str(row["label"])].append(float(row["root_height_mae_cm"]))
    summary["by_label"] = {
        label: {"n": len(values), "root_mae_cm": float(np.mean(values))}
        for label, values in sorted(
            per_label.items(), key=lambda kv: float(np.mean(kv[1])), reverse=True
        )
    }

    # 归因：站立案误差 vs 腾空案误差——判断 4.x cm 是姿态语义还是地面估计错误。
    stand_err = [
        row["phases"]["standing"]["root_height_mae_cm"]
        for row in rows
        if row["phases"]["standing"]["n_frames"] > 0
    ]
    air_err = [
        row["phases"]["airborne"]["root_height_mae_cm"]
        for row in rows
        if row["phases"]["airborne"]["n_frames"] > 0
    ]
    standing_mean = float(np.mean(stand_err)) if stand_err else 0.0
    airborne_mean = float(np.mean(air_err)) if air_err else 0.0
    pen_ratio = (
        float(np.mean([row["foot_pred"]["penetration_ratio"] for row in rows])) if rows else 0.0
    )
    float_ratio = (
        float(np.mean([row["foot_pred"]["float_ratio"] for row in rows])) if rows else 0.0
    )
    summary["standing_root_mae_cm"] = standing_mean
    summary["airborne_root_mae_cm"] = airborne_mean
    # 按帧数加权：`standing_mean` 是「样本均值的均值」，小样本占比会被放大；
    # 加权版才是「整个 val 集里站立帧的平均误差」，报告里两个都给。
    weighted: dict[str, float] = {}
    for segment in ("airborne", "grounded", "standing"):
        frames = float(np.sum([row["phases"][segment]["n_frames"] for row in rows])) if rows else 0.0
        if frames > 0:
            error_sum = float(
                np.sum(
                    [
                        row["phases"][segment]["root_height_mae_cm"]
                        * row["phases"][segment]["n_frames"]
                        for row in rows
                    ]
                )
            )
            weighted[segment] = error_sum / frames
        else:
            weighted[segment] = 0.0
    summary["phase_weighted_cm"] = weighted
    summary["n_bad_root_gt_50cm"] = int(
        sum(1 for row in rows if row["root_height_mae_cm"] > 50.0)
    )
    summary["penetration_ratio"] = pen_ratio
    summary["float_ratio"] = float_ratio

    verdict: list[str] = []
    verdict.append(
        f"* 站立帧根高度 MAE **{standing_mean:.2f}cm**，腾空帧 **{airborne_mean:.2f}cm**。"
    )
    if standing_mean <= 3.0 and airborne_mean > standing_mean:
        verdict.append(
            "  → 误差集中在**腾空/跳跃**阶段，站立帧达标：这是姿态语义造成的（腾空时根高度"
            "本来就是自由变量），不是模型把地面搞错了。**可接受**。"
        )
    elif standing_mean > 3.0:
        verdict.append(
            "  → **站立帧误差超标**：这是模型对地面的估计本身有偏，属于不可接受的类型"
            "（会出现整体漂浮/下沉），需按主理人的口径升级为架构问题上报。"
        )
    else:
        verdict.append("  → 站立帧已达标，腾空帧误差不高于站立帧，无需额外解释。")
    verdict.append(
        f"* FK 足底：着地帧里 **{pen_ratio:.1%}** 低于 {FOOT_PENETRATION_CM:.0f}cm（穿地），"
        f"**{float_ratio:.1%}** 高于 +{FOOT_FLOAT_CM:.0f}cm（漂浮）。"
    )
    if pen_ratio <= 0.05 and float_ratio <= 0.05:
        verdict.append("  → 都在 5% 以内：肉眼看不出一致的穿地/漂浮，与数值一致。")
    else:
        verdict.append("  → **超过 5%**：请优先打开 BVH 目视确认脚底，并在报告里点名片段。")
    summary["verdict_lines"] = verdict

    out_dir_path = Path(out_dir)
    (out_dir_path / "recon_viz.md").write_text(
        "\n".join(_format_markdown(rows, summary)) + "\n", encoding="utf-8"
    )
    (out_dir_path / "recon_viz.json").write_bytes(
        orjson.dumps({"summary": summary, "samples": rows}, option=orjson.OPT_INDENT_2)
    )
    print()
    print("\n".join(verdict))
    print(f"报告: {P.rel_to_root(out_dir_path / 'recon_viz.md')}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
