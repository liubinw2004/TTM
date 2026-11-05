#!/usr/bin/env python
"""VAE checkpoint 评测：**全量 val** 与 **前 64 样本** 两种口径并列给出。

## 为什么要两个口径

M2 baseline 报出的 ``joint_mae = 6.49° / root_mae = 4.76cm`` 是用
``_evaluate_metrics(max_batches=4)`` 算的——只覆盖 val 的前 4 个 batch（64 个窗口）。
用同一个 ckpt 跑全量 val，``root_mae`` 是 **510cm**：那 64 个窗口恰好没抽到
「下楼梯」这类根高度跨度 3 米的片段。也就是说，**抽样口径会让验收线随运气波动**。

所以本脚本把两个数字都打出来：

* ``full`` —— 全量 val（1528 窗口），**验收以它为准**；
* ``first64`` —— 前 4 个 batch，仅用于和 baseline 做**同口径**对照。

另外给出：

* 逐 ``category_l1`` 的 MAE（找出「误差集中在谁身上」）；
* 按 GT 足接触划分的 **腾空 / 着地 / 站立** 三段根高度 MAE（判断 4.x cm 是不是
  腾空帧造成的）；
* 误差分位数（p50 / p90 / max）与最差 5 个样本——均值会被长尾拉飞，必须看分布。

## 用法

```bash
python scripts/03c_eval_vae.py --config configs/tier_s.yaml \
    --ckpt runs/tier_s/vae/best.pt --json runs/tier_s/vae/eval_full.json
```
"""

from __future__ import annotations

import argparse
import sys
from collections import defaultdict
from pathlib import Path
from typing import Any, Iterable

import numpy as np
import orjson
import torch
from torch.utils.data import DataLoader

from ttm.common import constants as C
from ttm.common import paths as P
from ttm.common.config import load_config
from ttm.data.dataset import MotionDataset, auto_cache_size, collate_samples
from ttm.data.features import MotionFeature, NormStats
from ttm.train.train_vae import load_checkpoint, reconstruction_metrics

#: 「站立」判据：着地且根高度与该样本中位数偏差在此以内（cm）。
STANDING_TOLERANCE_CM: float = 5.0


@torch.no_grad()
def evaluate(
    model: torch.nn.Module,
    dataset: MotionDataset,
    norm_stats: NormStats,
    device: torch.device,
    *,
    batch_size: int = 16,
    max_batches: int | None = None,
    collect: bool = False,
) -> dict[str, Any]:
    """在（部分或全部）val 上评估。

    Args:
        model: VAE。
        dataset: 评估集。
        norm_stats: 归一化统计。
        device: 设备。
        batch_size: batch 大小。
        max_batches: 限制 batch 数（``None`` = 全量）。
        collect: 是否收集逐样本明细（逐类别/分阶段/最差样本）。

    Returns:
        ``{"joint_mae_deg", "root_height_mae_cm", "root_speed_mae_cmps", "n_samples",
        "per_category"?, "phases"?, "percentiles"?, "worst"?}``。
    """
    model.eval()
    loader = DataLoader(
        dataset,
        batch_size=min(batch_size, max(1, len(dataset))),
        shuffle=False,
        collate_fn=collate_samples,
        num_workers=0,
    )
    per_sample: list[dict[str, Any]] = []
    for index, batch in enumerate(loader):
        if max_batches is not None and index >= max_batches:
            break
        feat = batch["feat"].to(device)
        mask = batch["mask"].to(device)
        output = model(feat)
        pred = output.recon.detach().float().cpu().numpy().transpose(0, 2, 1)  # [B,T,146]
        truth = feat.detach().float().cpu().numpy().transpose(0, 2, 1)
        lengths = [int(item) for item in mask.sum(dim=-1).cpu().tolist()]
        for offset in range(pred.shape[0]):
            length = max(1, min(lengths[offset], pred.shape[1]))
            unit = np.ones(length, dtype=np.uint8)
            pred_item = MotionFeature(feat=pred[offset][:length], mask=unit).denormalize(
                norm_stats
            ).feat
            truth_item = MotionFeature(feat=truth[offset][:length], mask=unit).denormalize(
                norm_stats
            ).feat
            metrics = reconstruction_metrics(pred_item, truth_item)
            clip_id = str(batch["clip_id"][offset])
            info = dataset.labels.get(clip_id, {})
            row = {
                "clip_id": clip_id,
                # ``collate_samples`` 只带 ``text``（第一条描述），标签要从 labels 索引取
                "label": str(info.get("zh_label", "")),
                "category_l1": str(info.get("category_l1", "?")),
                "joint_mae_deg": metrics["joint_mae_deg"],
                "root_height_mae_cm": metrics["root_height_mae_cm"],
                "root_speed_mae_cmps": metrics["root_speed_mae_cmps"],
            }
            if collect:
                row["phases"] = _phase_errors(pred_item, truth_item)
            per_sample.append(row)

    if not per_sample:
        return {"n_samples": 0, "joint_mae_deg": 0.0, "root_height_mae_cm": 0.0,
                "root_speed_mae_cmps": 0.0}

    joint = np.array([row["joint_mae_deg"] for row in per_sample], dtype=np.float64)
    root = np.array([row["root_height_mae_cm"] for row in per_sample], dtype=np.float64)
    speed = np.array([row["root_speed_mae_cmps"] for row in per_sample], dtype=np.float64)
    summary: dict[str, Any] = {
        "n_samples": len(per_sample),
        "joint_mae_deg": float(joint.mean()),
        "root_height_mae_cm": float(root.mean()),
        "root_speed_mae_cmps": float(speed.mean()),
        "percentiles": {
            "joint_p50_p90_max": [float(np.percentile(joint, 50)), float(np.percentile(joint, 90)), float(joint.max())],
            "root_p50_p90_max": [float(np.percentile(root, 50)), float(np.percentile(root, 90)), float(root.max())],
            "root_gt_50cm_share": float((root > 50.0).mean()),
        },
    }
    if not collect:
        return summary

    by_category: dict[str, list[float]] = defaultdict(list)
    by_category_joint: dict[str, list[float]] = defaultdict(list)
    for row in per_sample:
        by_category[str(row["category_l1"])].append(float(row["root_height_mae_cm"]))
        by_category_joint[str(row["category_l1"])].append(float(row["joint_mae_deg"]))
    summary["per_category"] = {
        name: {
            "n": len(values),
            "root_mae_cm": float(np.mean(values)),
            "joint_mae_deg": float(np.mean(by_category_joint[name])),
            "share_of_val": len(values) / float(len(per_sample)),
        }
        for name, values in sorted(by_category.items(), key=lambda kv: -len(kv[1]))
    }
    # 分阶段：按帧数加权的根高度 MAE（避免「样本均值的均值」被小样本放大）
    weighted: dict[str, float] = {}
    for segment in ("airborne", "grounded", "standing"):
        frames = sum(row["phases"][segment]["n_frames"] for row in per_sample)
        if frames > 0:
            total = sum(
                row["phases"][segment]["root_height_mae_cm"] * row["phases"][segment]["n_frames"]
                for row in per_sample
            )
            weighted[segment] = float(total) / float(frames)
        else:
            weighted[segment] = 0.0
    summary["phase_weighted_cm"] = weighted
    summary["phase_frames"] = {
        segment: int(sum(row["phases"][segment]["n_frames"] for row in per_sample))
        for segment in ("airborne", "grounded", "standing")
    }
    summary["worst"] = [
        {
            "clip_id": row["clip_id"],
            "label": row["label"],
            "category_l1": row["category_l1"],
            "root_height_mae_cm": row["root_height_mae_cm"],
            "joint_mae_deg": row["joint_mae_deg"],
        }
        for row in sorted(per_sample, key=lambda item: item["root_height_mae_cm"], reverse=True)[:5]
    ]
    return summary


def _phase_errors(pred_denorm: np.ndarray, truth_denorm: np.ndarray) -> dict[str, Any]:
    """按 GT 足接触把帧分成腾空/着地/站立，返回各段根高度 MAE。"""
    contact = truth_denorm[:, C.FEAT_LAYOUT["foot_contact"]]
    grounded = (contact[:, 0] > 0.5) | (contact[:, 1] > 0.5)
    airborne = ~grounded
    height_dim = C.FEAT_LAYOUT["root_height"].start
    error = np.abs(pred_denorm[:, height_dim] - truth_denorm[:, height_dim])
    median_height = float(np.median(truth_denorm[:, height_dim]))
    standing = grounded & (np.abs(truth_denorm[:, height_dim] - median_height) <= STANDING_TOLERANCE_CM)

    def _segment(mask: np.ndarray) -> dict[str, float]:
        selected = error[mask]
        return {
            "n_frames": int(mask.sum()),
            "root_height_mae_cm": float(selected.mean()) if selected.size else 0.0,
        }

    return {"airborne": _segment(airborne), "grounded": _segment(grounded), "standing": _segment(standing)}


def main(argv: Iterable[str] | None = None) -> int:
    """命令行入口。"""
    parser = argparse.ArgumentParser(description="VAE 全量 / 前 64 样本两种口径评测")
    parser.add_argument("--config", default="configs/tier_s.yaml", help="配置文件路径")
    parser.add_argument("--ckpt", default="runs/tier_s/vae/best.pt", help="VAE checkpoint")
    parser.add_argument("--split", default="val", choices=("val", "train", "test"), help="评估 split")
    parser.add_argument("--json", default="", help="结果 JSON 落盘路径（空 = 只打印）")
    parser.add_argument("--device", default="", help="覆盖设备")
    args = parser.parse_args(list(argv) if argv is not None else None)

    cfg = load_config(args.config)
    device = torch.device(
        args.device if args.device else (cfg.runtime.device if torch.cuda.is_available() else "cpu")
    )
    norm_stats = NormStats.load(cfg.norm_stats_path())
    dataset = MotionDataset(
        cfg.shards_path(),
        cache_size=auto_cache_size(cfg.shards_path()),
        window=cfg.data.window,
        feature_dim=cfg.motion.feature_dim,
        labels_path=cfg.train_labels_path(),
        split=args.split,
    )
    if len(dataset) == 0:
        print(f"{args.split} split 为空", file=sys.stderr)
        return 1
    model, payload = load_checkpoint(Path(args.ckpt), map_location=device)
    model.to(device)
    model.eval()
    print(f"ckpt: {args.ckpt}（epoch {payload.get('epoch', '?')}） / {args.split} {len(dataset)} 窗口 / {device}")

    full = evaluate(model, dataset, norm_stats, device, collect=True)
    first64 = evaluate(model, dataset, norm_stats, device, max_batches=4)

    print("=" * 84)
    print(f"全量口径（{full['n_samples']} 窗口，**验收以此为准**）")
    print("=" * 84)
    print(f"  关节角 MAE : {full['joint_mae_deg']:7.2f}°（目标 ≤ 8°）")
    print(f"  根高度 MAE : {full['root_height_mae_cm']:7.2f}cm（目标 ≤ 3cm，降级 ≤ 4cm）")
    print(f"  根速度 MAE : {full['root_speed_mae_cmps']:7.2f} cm/s")
    joint_p = full["percentiles"]["joint_p50_p90_max"]
    root_p = full["percentiles"]["root_p50_p90_max"]
    print(f"  关节分位数 : p50 {joint_p[0]:.2f}° / p90 {joint_p[1]:.2f}° / max {joint_p[2]:.2f}°")
    print(f"  根高分位数 : p50 {root_p[0]:.2f}cm / p90 {root_p[1]:.2f}cm / max {root_p[2]:.2f}cm")
    print(f"  根高 >50cm : {full['percentiles']['root_gt_50cm_share']:.2%} 的窗口")
    phase = full["phase_weighted_cm"]
    frames = full["phase_frames"]
    print(
        f"  分阶段根高 : 腾空 {phase['airborne']:.2f}cm（{frames['airborne']} 帧）/ "
        f"着地 {phase['grounded']:.2f}cm（{frames['grounded']} 帧）/ "
        f"站立 {phase['standing']:.2f}cm（{frames['standing']} 帧）"
    )
    print()
    print("=" * 84)
    print(f"同口径对照（前 {first64['n_samples']} 窗口，与 M2 baseline 的 6.49° / 4.76cm 可直接比）")
    print("=" * 84)
    print(f"  关节角 MAE : {first64['joint_mae_deg']:7.2f}°")
    print(f"  根高度 MAE : {first64['root_height_mae_cm']:7.2f}cm")
    print()
    print("逐大类（全量口径）")
    print(f"{'类别':<44} {'n':>5} {'占比':>7} {'root cm':>9} {'joint °':>9}")
    for name, item in full["per_category"].items():
        print(f"{name:<44} {item['n']:>5} {item['share_of_val']:>6.1%} {item['root_mae_cm']:>9.2f} {item['joint_mae_deg']:>9.2f}")
    print()
    print("最差 5 个窗口（按根高度 MAE）")
    for row in full["worst"]:
        print(
            f"  {row['root_height_mae_cm']:9.1f}cm  {row['joint_mae_deg']:6.1f}°  "
            f"{row['label']:<10} {row['clip_id']}"
        )

    if args.json:
        out = P.resolve_out(args.json, mkparents=True)
        out.write_bytes(
            orjson.dumps({"ckpt": args.ckpt, "split": args.split, "full": full, "first64": first64},
                         option=orjson.OPT_INDENT_2)
        )
        print(f"\n结果落盘: {P.rel_to_root(out)}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
