"""MotionVAE 训练：重构 L1（mask 感知）+ KL（β=1e-4），ckpt / loss 落盘。

对应 ``TTM-ARCH.md`` §4.2 时序图 B 的前半段与 §10 T05 验收标准：

* 输入 batch ``feat[B,146,150]`` + ``mask[B,150]``，pad 帧的 loss 必须为 0；
* ``loss = masked_L1 + kl_weight * KL``；
* 每 epoch 落 ``runs/vae/ckpt_XXXX.pt``，val loss 改善时更新 ``runs/vae/best.pt``；
* 每 step 追加一行 ``runs/vae/loss.jsonl``；
* 验收指标：关节角 MAE ≤ 8°、根位移 MAE ≤ 3cm（在 val split 上算）。

本模块依赖 torch；无 torch 环境下不要导入它。
"""

from __future__ import annotations

import json
import math
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Iterable, Sequence

import numpy as np
import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader

from ttm.common import constants as C
from ttm.common import paths as P
from ttm.common.config import TTMConfig
from ttm.common.logutil import get_logger
from ttm.common.seed import set_seed
from ttm.data.dataset import MotionDataset, auto_cache_size, collate_samples
from ttm.data.features import MotionFeature, NormStats
from ttm.models.vae import MotionVAE, VAEOutput, kl_divergence

from ttm import LAYOUT_VERSION

__all__ = [
    "TrainHistory",
    "TrainResult",
    "joint_angle_mae_deg",
    "load_checkpoint",
    "masked_l1",
    "posterior_diagnostics",
    "reconstruction_metrics",
    "save_checkpoint",
    "train_vae",
    "wushu_sampling_report",
]

#: 验收阈值（ARCH §10 T05）。
JOINT_MAE_TARGET_DEG = 8.0
ROOT_MAE_TARGET_CM = 3.0


@dataclass
class TrainHistory:
    """训练历史（每一步/每 epoch 的 loss）。"""

    steps: list[dict[str, Any]] = field(default_factory=list)
    epochs: list[dict[str, Any]] = field(default_factory=list)

    def add_step(self, record: dict[str, Any]) -> None:
        """记录一步。"""
        self.steps.append(record)

    def add_epoch(self, record: dict[str, Any]) -> None:
        """记录一个 epoch。"""
        self.epochs.append(record)

    def loss_drop_ratio(self) -> float:
        """首尾 epoch 训练 loss 的下降比例（用于「loss 下降 > 30%」验收）。"""
        train_values = [item["train_loss"] for item in self.epochs if "train_loss" in item]
        if len(train_values) < 2 or train_values[0] <= 0:
            return 0.0
        return float((train_values[0] - train_values[-1]) / train_values[0])


@dataclass
class TrainResult:
    """训练结果摘要。"""

    best_val_loss: float = math.inf
    epochs_done: int = 0
    params: int = 0
    history: TrainHistory = field(default_factory=TrainHistory)
    metrics: dict[str, float] = field(default_factory=dict)
    checkpoint: str = ""
    elapsed_s: float = 0.0
    peak_memory_mb: float = 0.0


def masked_l1(pred: torch.Tensor, target: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
    """mask 感知的 L1 重构损失：只有 ``mask==1`` 的帧参与平均。

    Args:
        pred: ``[B, 146, T]`` 预测。
        target: ``[B, 146, T]`` 真值。
        mask: ``[B, T]`` 有效帧掩码（1/0）。

    Returns:
        标量张量；全 mask 为 0 时返回 0（不产生 NaN）。
    """
    if pred.shape != target.shape:
        raise ValueError(f"pred{tuple(pred.shape)} 与 target{tuple(target.shape)} 形状不一致")
    if mask.dim() != 2 or mask.shape[0] != pred.shape[0] or mask.shape[1] != pred.shape[-1]:
        raise ValueError(f"mask 形状应为 [{pred.shape[0]},{pred.shape[-1]}]，实际 {tuple(mask.shape)}")
    weights = mask.unsqueeze(1).to(pred.dtype)  # [B,1,T]
    error = (pred - target).abs() * weights
    denominator = weights.sum() * pred.shape[1]
    if float(denominator) <= 0.0:
        return pred.sum() * 0.0
    return error.sum() / denominator


def joint_angle_mae_deg(pred_quat: np.ndarray, gt_quat: np.ndarray) -> float:
    """两组四元数之间的平均角度误差（度）。

    Args:
        pred_quat: ``[..., 4]`` xyzw。
        gt_quat: ``[..., 4]`` xyzw。

    Returns:
        平均角误差（度）。
    """
    a = np.asarray(pred_quat, dtype=np.float64).reshape(-1, 4)
    b = np.asarray(gt_quat, dtype=np.float64).reshape(-1, 4)
    a = a / np.clip(np.linalg.norm(a, axis=1, keepdims=True), 1e-12, None)
    b = b / np.clip(np.linalg.norm(b, axis=1, keepdims=True), 1e-12, None)
    dots = np.clip(np.abs(np.sum(a * b, axis=1)), 0.0, 1.0)
    angles = 2.0 * np.arccos(dots)  # 弧度
    return float(np.degrees(angles).mean())


def reconstruction_metrics(
    pred_feat: np.ndarray,
    gt_feat: np.ndarray,
    *,
    fps: int = C.FPS,
) -> dict[str, float]:
    """把两段（已反归一化的）146 维特征还原为旋转后比较误差。

    Args:
        pred_feat: ``[B,T,146]`` 或 ``[T,146]`` 预测（**已反归一化**）。
        gt_feat: 同形状真值。
        fps: 帧率。

    Returns:
        ``{"joint_mae_deg":..., "root_height_mae_cm":..., "root_speed_mae_cmps":...}``。
    """
    pred = np.asarray(pred_feat, dtype=np.float64)
    gt = np.asarray(gt_feat, dtype=np.float64)
    if pred.shape != gt.shape:
        raise ValueError(f"形状不一致: {pred.shape} vs {gt.shape}")
    if pred.ndim == 2:
        pred = pred[None, ...]
        gt = gt[None, ...]

    angle_errors: list[float] = []
    height_errors: list[float] = []
    speed_errors: list[float] = []
    for index in range(pred.shape[0]):
        pred_feature = MotionFeature(feat=pred[index], mask=np.ones(pred.shape[1], dtype=np.uint8), fps=fps)
        gt_feature = MotionFeature(feat=gt[index], mask=np.ones(gt.shape[1], dtype=np.uint8), fps=fps)
        pred_quat, pred_root = pred_feature.to_rotations_root()
        gt_quat, gt_root = gt_feature.to_rotations_root()
        angle_errors.append(joint_angle_mae_deg(pred_quat, gt_quat))
        height_errors.append(float(np.abs(pred_root[:, 1] - gt_root[:, 1]).mean()))
        pred_vel = pred_feature.feat[:, C.FEAT_LAYOUT["root_vel"]]
        gt_vel = gt_feature.feat[:, C.FEAT_LAYOUT["root_vel"]]
        speed_errors.append(float(np.abs(pred_vel - gt_vel).mean()))
    return {
        "joint_mae_deg": float(np.mean(angle_errors)) if angle_errors else 0.0,
        "root_height_mae_cm": float(np.mean(height_errors)) if height_errors else 0.0,
        "root_speed_mae_cmps": float(np.mean(speed_errors)) if speed_errors else 0.0,
    }


def _prune_checkpoints(run_dir: Path, *, keep: int) -> int:
    """只保留最近 ``keep`` 个 ``ckpt_XXXX.pt`` 快照，返回删除个数。

    单个 VAE ckpt 约 120MB，500 epoch 全留就是 60GB。``best.pt`` / ``last.pt``
    **不在**本函数的作用范围内（它们的文件名不匹配 ``ckpt_*.pt``）。

    Args:
        run_dir: 训练输出目录。
        keep: 需要保留的最近快照数（``>= 1``）。

    Returns:
        实际删除的文件数。
    """
    keep = max(1, int(keep))
    snapshots = sorted(run_dir.glob("ckpt_*.pt"))
    if len(snapshots) <= keep:
        return 0
    removed = 0
    for path in snapshots[:-keep]:
        try:
            path.unlink()
            removed += 1
        except OSError:  # pragma: no cover - 文件被占用（Windows）时跳过
            continue
    return removed


def save_checkpoint(
    path: str | Path,
    model: MotionVAE,
    *,
    optimizer: torch.optim.Optimizer | None = None,
    epoch: int = 0,
    extra: dict[str, Any] | None = None,
) -> Path:
    """保存 checkpoint（含结构超参，便于独立加载）。

    Args:
        path: 目标路径（经写盘守卫）。
        model: 模型。
        optimizer: 可选优化器（续训用）。
        epoch: 已完成 epoch 数。
        extra: 额外字段。

    Returns:
        实际写出的路径。
    """
    target = P.resolve_out(path, mkparents=True)
    payload: dict[str, Any] = {
        "model": model.state_dict(),
        "model_config": {
            "feature_dim": model.feature_dim,
            "latent_dim": model.latent_dim,
            "hidden": model.hidden,
            "downsample": model.downsample,
            "n_res_blocks": model.n_res_blocks,
        },
        "epoch": int(epoch),
        "layout_version": int(LAYOUT_VERSION),
    }
    if optimizer is not None:
        payload["optimizer"] = optimizer.state_dict()
    if extra:
        payload["extra"] = extra
    torch.save(payload, target)
    return target


def load_checkpoint(
    path: str | Path,
    model: MotionVAE | None = None,
    *,
    optimizer: torch.optim.Optimizer | None = None,
    map_location: str | torch.device = "cpu",
) -> tuple[MotionVAE, dict[str, Any]]:
    """加载 checkpoint。

    Args:
        path: ckpt 路径。
        model: 已构造的模型；``None`` 时按 ckpt 内的超参新建。
        optimizer: 可选优化器（恢复状态）。
        map_location: torch.load 的 map_location。

    Returns:
        ``(model, payload)``。

    Raises:
        FileNotFoundError: 文件不存在。
        ValueError: ckpt 内容不合法。
    """
    source = Path(path)
    if not source.is_file():
        raise FileNotFoundError(f"checkpoint 不存在: {source}")
    payload = torch.load(source, map_location=map_location, weights_only=False)
    if not isinstance(payload, dict) or "model" not in payload:
        raise ValueError(f"checkpoint 格式不正确: {source}")
    if model is None:
        config = payload.get("model_config", {})
        model = MotionVAE(
            feature_dim=int(config.get("feature_dim", C.FEATURE_DIM)),
            latent_dim=int(config.get("latent_dim", C.LATENT_DIM)),
            hidden=int(config.get("hidden", 512)),
            downsample=int(config.get("downsample", C.DOWNSAMPLE)),
            n_res_blocks=int(config.get("n_res_blocks", 2)),
        )
    model.load_state_dict(payload["model"])
    if optimizer is not None and "optimizer" in payload:
        optimizer.load_state_dict(payload["optimizer"])
    return model, payload


def _resolve_device(cfg: TTMConfig) -> torch.device:
    """按配置选择设备（cuda 不可用时回退 cpu 并告警）。"""
    requested = cfg.runtime.device
    if requested.startswith("cuda") and not torch.cuda.is_available():
        return torch.device("cpu")
    return torch.device(requested)


def _load_norm_stats(cfg: TTMConfig) -> NormStats:
    """读取归一化统计量；缺失时退化为恒等。"""
    path = cfg.norm_stats_path()
    if Path(path).is_file():
        return NormStats.load(path)
    return NormStats.identity()


def wushu_sampling_report(dataset: MotionDataset, cfg: TTMConfig) -> dict[str, float]:
    """算「武术数据占比 → 采样期望占比」的对照数字。

    主理人要求把训练期武术占比抬到约 15%，并且**要看实测数**而不是「设了个 0.5」。
    这里返回三档：数据占比、仅 ``power`` 均衡后的占比、加上组倍率之后的目标占比，
    三者的差就是「组倍率」到底做了多少事。

    Args:
        dataset: 训练集（需带 ``labels_path``）。
        cfg: 生效配置。

    Returns:
        ``{"data_share":..., "after_power":..., "after_multiplier":..., "multiplier":...}``。
    """
    from ttm.data.dataset import (  # noqa: PLC0415 - 只在此处需要
        build_sample_weights,
        expected_group_share,
        resolve_group_multiplier,
    )

    mask = dataset.wushu_mask()
    labels = dataset.sample_labels()
    if mask.size == 0 or not mask.any():
        return {"data_share": 0.0, "after_power": 0.0, "after_multiplier": 0.0, "multiplier": 1.0}
    power = float(cfg.data.balance_power)
    cap = float(cfg.data.balance_max_ratio)
    base = build_sample_weights(labels, power=power, max_weight_ratio=cap)
    target = float(cfg.data.balanced_target_wushu_ratio)
    tuned = build_sample_weights(
        labels,
        power=power,
        max_weight_ratio=cap,
        group_mask=mask,
        target_group_share=target,
    )
    return {
        "data_share": dataset.data_group_share(mask),
        "after_power": expected_group_share(base, mask),
        "after_multiplier": expected_group_share(tuned, mask),
        "multiplier": resolve_group_multiplier(base, mask, target),
    }


def _log_wushu_sampling(dataset: MotionDataset, cfg: TTMConfig, log: Any) -> None:
    """把武术采样占比打进日志（数据占比 → 采样期望占比）。"""
    stats = wushu_sampling_report(dataset, cfg)
    log.info(
        "武术采样占比: 数据 %.2f%% → power=%.2f 后 %.2f%% → 组倍率 ×%.3f 后 %.2f%%（目标 %.2f%%）",
        stats["data_share"] * 100.0,
        float(cfg.data.balance_power),
        stats["after_power"] * 100.0,
        stats["multiplier"],
        stats["after_multiplier"] * 100.0,
        float(cfg.data.balanced_target_wushu_ratio) * 100.0,
    )


def train_vae(
    cfg: TTMConfig,
    *,
    max_steps_per_epoch: int | None = None,
    max_epochs: int | None = None,
    resume: str | None = None,
    logger: Any | None = None,
    max_minutes: float | None = None,
) -> TrainResult:
    """训练 MotionVAE。

    Args:
        cfg: 生效配置。
        max_steps_per_epoch: 限制每 epoch 步数（冒烟/测试用）。
        max_epochs: 覆盖 ``cfg.vae.epochs``。
        resume: 从该 ckpt 续训。
        logger: 可注入 logger。
        max_minutes: 墙钟预算（分钟）。每跑完一个 epoch 检查一次，超预算就停止并
            保留 ``best.pt``——避免被外部 timeout 杀掉导致一个 ckpt 都没有。

    Returns:
        :class:`TrainResult`。

    Raises:
        FileNotFoundError: 分片目录不存在（需先跑预处理）。
    """
    log = logger or get_logger("ttm.train.vae", script="03_train_vae")
    seed = set_seed(cfg.runtime.seed)
    device = _resolve_device(cfg)
    if device.type == "cuda":
        torch.cuda.reset_peak_memory_stats(device)
    budget_s = float(max_minutes) * 60.0 if max_minutes else 0.0
    log.info("VAE 训练开始: device=%s seed=%d", device, seed)

    # 注意：resolve_out 只创建「父目录」，要建目录本身必须用 safe_makedirs
    run_dir = P.safe_makedirs(cfg.run_dir("vae"))
    # 训练读**合并后**的标签（少样本细标签已并到基名，条件可学）；
    # ``labels.jsonl`` 保留原细标签，只作统计口径。
    labels_path = cfg.train_labels_path()
    # 分片全部常驻内存，否则 shuffle 会让 LRU 抖动到 30s/step（实测）
    cache_size = auto_cache_size(cfg.shards_path())
    train_set = MotionDataset(
        cfg.shards_path(),
        cache_size=cache_size,
        window=cfg.data.window,
        feature_dim=cfg.motion.feature_dim,
        labels_path=labels_path,
        split="train",
        class_balanced=cfg.data.class_balanced,
        target_wushu_ratio=cfg.data.balanced_target_wushu_ratio,
    )
    try:
        val_set = MotionDataset(
            cfg.shards_path(),
            cache_size=cache_size,
            window=cfg.data.window,
            feature_dim=cfg.motion.feature_dim,
            labels_path=labels_path,
            split="val",
        )
    except FileNotFoundError:  # pragma: no cover - 极端情况
        val_set = None
    if len(train_set) == 0:
        raise ValueError("train split 没有样本，请检查 labels.jsonl 的 split 分布")
    if val_set is not None and len(val_set) == 0:
        log.warning("val split 为空，用 train 的最后一批代替验证")
        val_set = None

    batch_size = max(1, int(cfg.vae.batch_size))
    # 类别均衡：训练集按 1/sqrt(freq) 加权，验证集保持自然分布（否则 metric 不可比）。
    # DataLoader 规定 sampler 与 shuffle 互斥，所以启用采样器时必须把 shuffle 关掉。
    train_sampler = train_set.sampler()
    if train_sampler is not None:
        log.info(
            "训练集启用类别均衡采样: %d 个样本 / %d 个标签, power=%.2f",
            len(train_set),
            len(set(train_set.sample_labels())),
            float(cfg.data.balance_power),
        )
        _log_wushu_sampling(train_set, cfg, log)
    train_loader = DataLoader(
        train_set,
        batch_size=batch_size,
        sampler=train_sampler,
        shuffle=train_sampler is None,
        drop_last=False,
        collate_fn=collate_samples,
        num_workers=0,
    )
    val_loader = (
        DataLoader(
            val_set,
            batch_size=batch_size,
            shuffle=False,
            drop_last=False,
            collate_fn=collate_samples,
            num_workers=0,
        )
        if val_set is not None
        else None
    )

    model = MotionVAE(
        feature_dim=cfg.motion.feature_dim,
        latent_dim=cfg.vae.latent_dim,
        hidden=cfg.vae.hidden,
        downsample=cfg.vae.downsample,
        # logvar 必须夹上界：不夹的话个别样本会漂到 +6，单样本 KL ≈ 7.2e5，
        # val loss 随机跳出几十到上百的尖峰，best.pt 的选择被污染。
        logvar_clamp=(float(cfg.vae.logvar_min), float(cfg.vae.logvar_max)),
    ).to(device)
    log.info("模型: %s", model)
    optimizer = torch.optim.AdamW(model.parameters(), lr=float(cfg.vae.lr))

    start_epoch = 0
    if resume:
        model, payload = load_checkpoint(resume, model, optimizer=optimizer, map_location=device)
        start_epoch = int(payload.get("epoch", 0))
        log.info("从 %s 续训（已 %d epoch）", resume, start_epoch)

    norm_stats = _load_norm_stats(cfg)
    history = TrainHistory()
    result = TrainResult(params=model.param_count(), checkpoint="")
    use_amp = cfg.runtime.precision == "bf16" and device.type == "cuda"
    kl_weight = float(cfg.vae.kl_weight)
    warmup_epochs = max(0, int(cfg.vae.kl_warmup_epochs))
    epochs = int(max_epochs if max_epochs is not None else cfg.vae.epochs)
    loss_log = P.resolve_out(run_dir / "loss.jsonl", mkparents=True)
    started = time.perf_counter()
    global_step = 0

    # 全新训练截断、续训追加：否则重跑一次会把上一轮的 step 混进来，曲线直接失去意义
    with loss_log.open("a" if resume else "w", encoding="utf-8") as log_handle:
        for epoch in range(start_epoch, epochs):
            # KL 线性 warmup：前期让重建主导，把解码器「教会用 z」，再逐步压后验。
            # 注意 warmup 必须与 models.vae 的 logvar_bias 配合：在 (mu,logvar)=(0,0)
            # 这个驻点上 KL 梯度恒为 0，单独退火乘数是无操作。
            effective_kl = kl_weight * min(1.0, (epoch + 1) / warmup_epochs) if warmup_epochs else kl_weight
            model.train()
            total_loss = 0.0
            total_recon = 0.0
            total_kl = 0.0
            steps = 0
            for batch in train_loader:
                feat = batch["feat"].to(device)
                mask = batch["mask"].to(device)
                with torch.autocast(device_type=device.type, dtype=torch.bfloat16, enabled=use_amp):
                    output: VAEOutput = model(feat)
                    recon_loss = masked_l1(output.recon.float(), feat.float(), mask)
                    kl = kl_divergence(output.mu.float(), output.logvar.float())
                    loss = recon_loss + effective_kl * kl
                optimizer.zero_grad(set_to_none=True)
                loss.backward()
                torch.nn.utils.clip_grad_norm_(model.parameters(), 5.0)
                optimizer.step()

                if not torch.isfinite(loss):
                    raise FloatingPointError(f"VAE loss 出现 NaN/Inf（epoch {epoch}, step {steps}）")

                record = {
                    "step": global_step,
                    "epoch": epoch,
                    "loss": float(loss.detach().cpu()),
                    "recon": float(recon_loss.detach().cpu()),
                    "kl": float(kl.detach().cpu()),
                    "kl_weight": float(effective_kl),
                }
                history.add_step(record)
                log_handle.write(json.dumps(record) + "\n")
                log_handle.flush()

                total_loss += record["loss"]
                total_recon += record["recon"]
                total_kl += record["kl"]
                steps += 1
                global_step += 1
                if max_steps_per_epoch is not None and steps >= max_steps_per_epoch:
                    break

            train_loss = total_loss / max(1, steps)
            val_loss = (
                _evaluate_loss(model, val_loader, device, effective_kl, cfg) if val_loader else train_loss
            )
            epoch_record = {
                "epoch": epoch,
                "train_loss": train_loss,
                "train_recon": total_recon / max(1, steps),
                "train_kl": total_kl / max(1, steps),
                "val_loss": val_loss,
                "steps": steps,
                "kl_weight": float(effective_kl),
            }
            history.add_epoch(epoch_record)
            log.info(
                "epoch %03d | train %.6f (recon %.6f / kl %.4f / w %.2e) | val %.6f",
                epoch,
                train_loss,
                epoch_record["train_recon"],
                epoch_record["train_kl"],
                effective_kl,
                val_loss,
            )

            save_checkpoint(run_dir / f"ckpt_{epoch:04d}.pt", model, optimizer=optimizer, epoch=epoch + 1)
            _prune_checkpoints(run_dir, keep=int(cfg.vae.save_keep_last))
            if val_loss <= result.best_val_loss:
                result.best_val_loss = val_loss
                result.checkpoint = str(
                    save_checkpoint(run_dir / "best.pt", model, optimizer=optimizer, epoch=epoch + 1)
                )
            result.epochs_done = epoch + 1
            if budget_s > 0:
                spent = time.perf_counter() - started
                if spent >= budget_s:
                    log.warning(
                        "已达墙钟预算 %.1f 分钟（实际 %.1f 分钟），停止训练并保留 best.pt",
                        budget_s / 60.0,
                        spent / 60.0,
                    )
                    break

    # --- 最终指标（在 val 上算关节角/根位移误差）
    eval_set = val_set if val_set is not None else train_set
    # 全量评估（``max_batches=None``）：抽样 64 个窗口会让验收线随抽样运气波动。
    result.metrics = _evaluate_metrics(model, eval_set, device, norm_stats)
    # 后验健康度：只看 joint_mae 分不清「重建差」还是「后验塌缩」。
    # 塌缩的典型指纹是 mu 标准差≈0 + 重建帧间方差远低于真值（实测只有 1/17）。
    result.metrics.update(posterior_diagnostics(model, eval_set, device, max_batches=2))
    # 数据占比 → 采样期望占比（主理人要求 M1'/M2 报告里给出实测对照）
    sampling = wushu_sampling_report(train_set, cfg)
    result.metrics.update({f"sampling_{key}": value for key, value in sampling.items()})
    result.history = history
    result.elapsed_s = time.perf_counter() - started
    if device.type == "cuda":
        result.peak_memory_mb = float(torch.cuda.max_memory_allocated(device) / (1024**2))
    log.info(
        "VAE 训练结束: best_val=%.6f, joint_mae=%.2f°, root_height_mae=%.2fcm, 用时 %.1fs",
        result.best_val_loss,
        result.metrics.get("joint_mae_deg", float("nan")),
        result.metrics.get("root_height_mae_cm", float("nan")),
        result.elapsed_s,
    )
    log.info(
        "后验健康度: mu_std=%.4f, logvar_mean=%.3f, 重建帧间方差=%.6f / 真值 %.6f（比 %.2f）",
        result.metrics.get("mu_std", float("nan")),
        result.metrics.get("logvar_mean", float("nan")),
        result.metrics.get("recon_frame_var", float("nan")),
        result.metrics.get("target_frame_var", float("nan")),
        result.metrics.get("var_ratio", float("nan")),
    )
    if result.metrics.get("mu_std", 1.0) < 0.01:
        log.warning(
            "后验仍然塌缩：mu_std=%.5f < 0.01。检查 logvar_bias 与 kl_weight"
            "（KL 是对 %d 维求和，1e-4 实际等效权重远超重建项）",
            result.metrics.get("mu_std", 0.0),
            int(model.token_length) * int(model.latent_dim),
        )
    if result.metrics.get("joint_mae_deg", 0.0) > JOINT_MAE_TARGET_DEG:
        log.warning("关节角 MAE 超过验收线 %.1f°（训练轮数不足或数据量太小）", JOINT_MAE_TARGET_DEG)
    if result.metrics.get("root_height_mae_cm", 0.0) > ROOT_MAE_TARGET_CM:
        log.warning("根高度 MAE 超过验收线 %.1fcm", ROOT_MAE_TARGET_CM)
    return result


@torch.no_grad()
def posterior_diagnostics(
    model: MotionVAE,
    dataset: Any,
    device: torch.device,
    *,
    max_batches: int = 2,
) -> dict[str, float]:
    """量化后验是否塌缩（训练结束后调用）。

    为什么单独做这个诊断：``joint_mae`` 只反映重建精度，一个塌缩到「输出平均姿态」
    的 VAE 也能拿到不错的 L1，但它的潜空间对扩散模型毫无用处。实测塌缩模型的
    指纹是 ``mu.std() ≈ 1e-4``、``logvar ≈ 0``（即后验=先验、std=1 纯噪声），
    且重建序列的帧间方差只有真值的 1/17。

    Args:
        model: 训练好的 VAE。
        dataset: ``MotionDataset``（取前 ``max_batches`` 个 batch）。
        device: 计算设备。
        max_batches: 采样多少批（每批 16 个窗口）。

    Returns:
        ``{"mu_std", "logvar_mean", "z_std", "recon_frame_var", "target_frame_var", "var_ratio"}``。
    """
    was_training = model.training
    model.eval()
    mu_values: list[float] = []
    logvar_values: list[float] = []
    z_values: list[float] = []
    recon_var: list[float] = []
    target_var: list[float] = []
    batch_size = 16
    total = len(dataset)
    for start in range(0, min(total, batch_size * max_batches), batch_size):
        indices = list(range(start, min(start + batch_size, total)))
        if not indices:
            continue
        samples = [dataset[index] for index in indices]
        batch = collate_samples(samples)
        feat = batch["feat"].to(device)
        output: VAEOutput = model(feat)
        mu_values.append(float(output.mu.float().std().cpu()))
        logvar_values.append(float(output.logvar.float().mean().cpu()))
        z_values.append(float(output.z.float().std().cpu()))
        # 帧间方差：对时间维求一阶差分后的方差，衡量「动作是否有变化」
        recon = output.recon.float()
        if recon.shape[-1] != feat.shape[-1]:
            recon = recon[..., : feat.shape[-1]]
        recon_var.append(float(recon.diff(dim=-1).var().cpu()))
        target_var.append(float(feat.float().diff(dim=-1).var().cpu()))
    if was_training:
        model.train()

    def _mean(values: list[float]) -> float:
        return float(np.mean(values)) if values else 0.0

    recon_mean = _mean(recon_var)
    target_mean = _mean(target_var)
    return {
        "mu_std": _mean(mu_values),
        "logvar_mean": _mean(logvar_values),
        "z_std": _mean(z_values),
        "recon_frame_var": recon_mean,
        "target_frame_var": target_mean,
        "var_ratio": (recon_mean / target_mean) if target_mean > 0 else 0.0,
    }


@torch.no_grad()
def _evaluate_loss(
    model: MotionVAE,
    loader: Iterable[dict[str, Any]] | None,
    device: torch.device,
    kl_weight: float,
    cfg: TTMConfig,
) -> float:
    """在验证集上算平均 loss（采样版，与训练一致）。"""
    if loader is None:
        return math.inf
    model.eval()
    total = 0.0
    count = 0
    for batch in loader:
        feat = batch["feat"].to(device)
        mask = batch["mask"].to(device)
        output = model(feat)
        recon = masked_l1(output.recon.float(), feat.float(), mask)
        kl = kl_divergence(output.mu.float(), output.logvar.float())
        total += float((recon + kl_weight * kl).detach().cpu())
        count += 1
    model.train()
    return total / max(1, count)


@torch.no_grad()
def _evaluate_metrics(
    model: MotionVAE,
    dataset: MotionDataset,
    device: torch.device,
    norm_stats: NormStats,
    *,
    max_batches: int | None = None,
) -> dict[str, float]:
    """算「关节角 MAE / 根高度 MAE」。

    特征都是归一化后的，先用 ``norm_stats`` 反归一化，再用
    :func:`reconstruction_metrics` 还原旋转比较。

    Args:
        model: 待评估模型。
        dataset: 评估集（通常是 val split）。
        device: 计算设备。
        norm_stats: 归一化统计。
        max_batches: 最多评估多少个 batch；``None``（默认）= **全量评估**。

            **默认值的由来**：旧版写死 ``4``，即只用 4×16 = 64 个窗口代表整个 val。
            实测这 64 个窗口恰好没抽到「下楼梯」这类根高度跨度 3 米的片段，
            报出 ``root_mae = 4.76cm`` 的好看的数字；换成全量 val 后同一 ckpt 是
            510cm。抽样口径会让验收线失去意义，所以默认改成全量。
    """
    model.eval()
    loader = DataLoader(
        dataset,
        batch_size=min(16, max(1, len(dataset))),
        shuffle=False,
        collate_fn=collate_samples,
        num_workers=0,
    )
    preds: list[np.ndarray] = []
    truths: list[np.ndarray] = []
    valid_lengths: list[int] = []
    for index, batch in enumerate(loader):
        if max_batches is not None and index >= int(max_batches):
            break
        feat = batch["feat"].to(device)
        mask = batch["mask"].to(device)
        output = model(feat)
        pred = output.recon.detach().float().cpu().numpy().transpose(0, 2, 1)  # [B,T,146]
        truth = feat.detach().float().cpu().numpy().transpose(0, 2, 1)
        preds.append(pred)
        truths.append(truth)
        valid_lengths.extend(int(item) for item in mask.sum(dim=-1).cpu().tolist())
    if not preds:
        return {"joint_mae_deg": 0.0, "root_height_mae_cm": 0.0, "root_speed_mae_cmps": 0.0}
    pred_concat = np.concatenate(preds, axis=0)
    truth_concat = np.concatenate(truths, axis=0)

    # **必须按 mask 裁到有效帧再比较**。pad 帧的特征是全 0，反归一化后是一段
    # 完全不存在的「姿态」（根高度恒为 -mean/std，脚陷到地下十几米，边界上还有
    # 巨大速度阶跃）。旧版这里用 ``mask=np.ones(...)``，等于把 pad 帧当真值，
    # 实测把整个 val 的 root_height_mae 从 4cm 抬到 514cm（1.12 一个类 1399cm）。
    # 这不是小数点问题，是「验收线到底过没过」的判断依据。
    pred_denorm_list: list[np.ndarray] = []
    truth_denorm_list: list[np.ndarray] = []
    for item_pred, item_truth, valid in zip(pred_concat, truth_concat, valid_lengths):
        length = max(1, min(int(valid), int(item_pred.shape[0])))
        unit_mask = np.ones(length, dtype=np.uint8)
        pred_denorm_list.append(
            MotionFeature(feat=item_pred[:length], mask=unit_mask).denormalize(norm_stats).feat
        )
        truth_denorm_list.append(
            MotionFeature(feat=item_truth[:length], mask=unit_mask).denormalize(norm_stats).feat
        )
    # 各样本有效帧数不同，``reconstruction_metrics`` 支持逐条传入（[B,T,146] 需等长），
    # 所以逐条算再平均，避免为凑齐形状又引入 padding。
    metrics = [
        reconstruction_metrics(pred_item, truth_item)
        for pred_item, truth_item in zip(pred_denorm_list, truth_denorm_list)
    ]
    if not metrics:
        return {"joint_mae_deg": 0.0, "root_height_mae_cm": 0.0, "root_speed_mae_cmps": 0.0}
    return {
        key: float(np.mean([item[key] for item in metrics]))
        for key in ("joint_mae_deg", "root_height_mae_cm", "root_speed_mae_cmps")
    }
