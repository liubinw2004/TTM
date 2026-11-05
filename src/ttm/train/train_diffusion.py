"""潜空间扩散（DiT）训练：冻结 VAE 编码 + 冻结 bge 编码 + 可训练投影/DiT。

流程（``TTM-ARCH.md`` §6 T07）：

::

    146 维特征窗口 [B,150,146]
      │ 冻结 MotionVAE.encode → μ（取均值，不加噪）        [B,38,128]
      │ 中文标注 zh_texts 采样一条 → 冻结 bge → 512d          [B,512]
      │ TextProjector（可训练）                              [B,256]
      ▼
    DiT ε-预测：loss = MSE(ε_pred, ε_true)，条件 dropout 10%（CFG）

关键设计：

* **VAE 冻结**：只用编码器提供潜变量目标，梯度不回传（省显存、防塌缩）；
* **文本嵌入缓存**：数据集里 distinct 文本只有几百条，一次性编码成 512d 查表，
  训练时只跑可训练投影层（否则每步都跑 bge 前向，慢 10 倍以上）；
* **token 掩码**：150 帧 → 38 token，用 ``avg_pool1d(k=4,s=4,ceil_mode=True)``
  把帧级 mask 映射成 token 级 mask，尾窗 padding 不参与 loss；
* **CFG**：训练期以 ``cfg.diffusion.cfg_dropout``（默认 0.1）把条件置零，
  推理期才能用 ``guidance > 1`` 做无分类器引导。
"""

from __future__ import annotations

import json
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Sequence

import numpy as np
import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader

from ttm import LAYOUT_VERSION
from ttm.common import constants as C
from ttm.common import paths as P
from ttm.common.config import TTMConfig
from ttm.common.logutil import get_logger
from ttm.common.seed import set_seed, torch_generator
from ttm.data.dataset import MotionDataset, auto_cache_size, collate_samples
from ttm.models.diffusion import DiffusionTransformer, GaussianDiffusion
from ttm.models.text_encoder import TextEncoder, TextEncoderUnavailable
from ttm.models.vae import MotionVAE
# 采样占比的日志与 VAE 侧共用一份实现：口径必须完全一致，否则两个里程碑报出来的
# 「武术采样占比」对不上，排查时会以为是数据被换了。
from ttm.train.train_vae import _log_wushu_sampling

__all__ = [
    "DiffusionHistory",
    "DiffusionTrainResult",
    "TextConditionBank",
    "frames_to_token_mask",
    "load_vae_for_diffusion",
    "resolve_vae_checkpoint",
    "train_diffusion",
]

#: 每个 epoch 记录一次的 loss 采样点数量（loss.jsonl 每 step 一行，这里只做摘要）。
CURVE_POINTS = 8


@dataclass
class DiffusionHistory:
    """训练曲线（loss 记录 + 峰值显存）。"""

    train_loss: list[float] = field(default_factory=list)
    val_loss: list[float] = field(default_factory=list)
    steps: list[int] = field(default_factory=list)
    peak_memory_mb: float = 0.0
    elapsed_s: float = 0.0

    def loss_drop_ratio(self) -> float:
        """首末 loss 的相对下降比例（含验证 loss 时取训练 loss）。"""
        if len(self.train_loss) < 2 or self.train_loss[0] <= 0:
            return 0.0
        return float((self.train_loss[0] - self.train_loss[-1]) / self.train_loss[0])

    def to_dict(self) -> dict[str, Any]:
        """序列化（写 summary.json 用）。"""
        return {
            "train_loss": self.train_loss,
            "val_loss": self.val_loss,
            "steps": self.steps,
            "peak_memory_mb": round(self.peak_memory_mb, 1),
            "elapsed_s": round(self.elapsed_s, 1),
            "loss_drop_ratio": round(self.loss_drop_ratio(), 4),
            "curve_points": CURVE_POINTS,
        }


@dataclass
class DiffusionTrainResult:
    """训练结果。"""

    history: DiffusionHistory
    best_val: float
    param_count: int
    ckpt_path: str
    run_dir: str

    def to_dict(self) -> dict[str, Any]:
        """序列化（写 summary.json 用）。"""
        return {
            "best_val": round(float(self.best_val), 6),
            "param_count": int(self.param_count),
            "ckpt": self.ckpt_path,
            "run_dir": self.run_dir,
            "history": self.history.to_dict(),
        }


def frames_to_token_mask(mask: torch.Tensor, downsample: int = C.DOWNSAMPLE) -> torch.Tensor:
    """把帧级掩码 ``[B,T]`` 映射成 token 级掩码 ``[B,ceil(T/d)]``（1 = 有效）。

    与 VAE 的 ``ceil(T/d)`` 长度约定保持一致：只要一个 token 的感受野里有有效帧，
    该 token 就算有效。

    Args:
        mask: ``[B,T]``（1 = 有效，0 = padding）。
        downsample: 下采样倍数（4）。

    Returns:
        ``[B, ceil(T/d)]`` float32（1.0 / 0.0）。
    """
    if mask.dim() != 2:
        raise ValueError(f"mask 形状应为 [B,T]，实际 {tuple(mask.shape)}")
    pooled = F.avg_pool1d(
        mask.unsqueeze(1).float(),
        kernel_size=int(downsample),
        stride=int(downsample),
        ceil_mode=True,
    ).squeeze(1)
    return (pooled > 0.0).float()


class TextConditionBank:
    """把数据集里的中文描述一次性编码成 512d 向量，训练时只查表。

    每条窗口样本在 ``labels.jsonl`` 里带 8–15 条同义表述；训练时随机取一条做
    数据增强（同一动作的不同说法 → 条件增强），因此需要按 ``clip_id`` 保存候选列表。
    """

    def __init__(self, encoder: TextEncoder, *, device: torch.device) -> None:
        """初始化。

        Args:
            encoder: 已加载的文本编码器（bge，冻结）。
            device: 存储设备。
        """
        self.encoder = encoder
        self.device = device
        self._index: dict[str, int] = {}
        self._vectors: torch.Tensor | None = None
        self._clip_candidates: dict[str, list[int]] = {}
        self._fallback = 0

    @property
    def size(self) -> int:
        """去重后的文本条数。"""
        return len(self._index)

    @property
    def dim(self) -> int:
        """句向量维数（512）。"""
        if self._vectors is None:
            return 0
        return int(self._vectors.shape[1])

    def build(self, labels: dict[str, dict[str, Any]], *, batch_size: int = 64) -> None:
        """编码全部文本并建立「clip_id → 候选向量下标」索引。

        Args:
            labels: ``clip_id → label 记录``（来自 ``labels.jsonl``）。
            batch_size: 编码批大小。

        Raises:
            TextEncoderUnavailable: 文本编码器不可用（由调用方决定是否降级）。
        """
        unique: list[str] = []
        seen: dict[str, int] = {}
        for clip_id, info in labels.items():
            texts = [str(item) for item in (info.get("zh_texts") or [])]
            label = str(info.get("zh_label") or "")
            if label:
                texts.append(label)
            candidates: list[int] = []
            for text in dict.fromkeys(texts):
                if not text:
                    continue
                if text not in seen:
                    seen[text] = len(unique)
                    unique.append(text)
                candidates.append(seen[text])
            if candidates:
                self._clip_candidates[str(clip_id)] = candidates
        if not unique:
            raise ValueError("labels.jsonl 里没有任何文本，无法建立条件库")
        vectors = self.encoder.encode(unique, batch_size=batch_size)
        self._index = seen
        self._vectors = torch.from_numpy(np.asarray(vectors, dtype=np.float32)).to(self.device)
        # 兜底：无条件时用全零向量（等价 null condition）
        self._fallback = -1

    def null(self, proj_dim: int, batch_size: int = 1) -> torch.Tensor:
        """无条件向量（全零），用于 CFG 的 null 分支。"""
        return torch.zeros(batch_size, proj_dim, device=self.device)

    def gather(self, clip_ids: Sequence[str], generator: torch.Generator | None = None) -> torch.Tensor:
        """为一组 ``clip_id`` 随机取一条描述，返回 ``[B,512]`` 句向量。

        Args:
            clip_ids: batch 内的 clip_id。
            generator: 随机数发生器（保证可复现）。

        Returns:
            ``[B,512]`` float32（在 ``self.device`` 上）。
        """
        if self._vectors is None:
            raise RuntimeError("TextConditionBank.build() 尚未调用")
        rows: list[int] = []
        for clip_id in clip_ids:
            candidates = self._clip_candidates.get(str(clip_id))
            if not candidates:
                rows.append(-1)
                continue
            if len(candidates) == 1:
                rows.append(candidates[0])
                continue
            idx = int(
                torch.randint(0, len(candidates), (1,), generator=generator).item()
            )
            rows.append(candidates[idx])
        picked = torch.as_tensor(
            [row if row >= 0 else 0 for row in rows], dtype=torch.long, device=self.device
        )
        vectors = self._vectors.index_select(0, picked)
        # 没有文本的样本强制置空（等价 null condition）
        empty = torch.as_tensor([row < 0 for row in rows], dtype=torch.bool, device=self.device)
        if bool(empty.any()):
            vectors = vectors.masked_fill(empty.unsqueeze(-1), 0.0)
        return vectors


def resolve_vae_checkpoint(cfg: TTMConfig) -> Path:
    """解析 VAE checkpoint 路径（``runtime.vae_ckpt`` 为空时按档位推导）。

    Args:
        cfg: 生效配置。

    Returns:
        VAE checkpoint 路径。

    Notes:
        ``runtime.vae_ckpt`` 默认留空，改由 ``run_dir("vae")`` 推导出
        ``runs/tier_<档位>/vae/best.pt``；这里集中一处解析，避免 ckpt 里
        记下空字符串导致推理端找不到权重。
    """
    return Path(cfg.runtime.vae_ckpt) if cfg.runtime.vae_ckpt else cfg.run_dir("vae") / "best.pt"


def load_vae_for_diffusion(cfg: TTMConfig, device: torch.device) -> MotionVAE:
    """加载并冻结 VAE（只用作潜变量编码器）。

    Args:
        cfg: 生效配置。
        device: 设备。

    Returns:
        已 ``eval()`` 且参数 ``requires_grad=False`` 的 :class:`MotionVAE`。

    Raises:
        FileNotFoundError: ckpt 不存在（需先跑 M2）。
    """
    from ttm.train.train_vae import load_checkpoint  # noqa: PLC0415 - 复用同一套 ckpt 格式

    ckpt = resolve_vae_checkpoint(cfg)
    if not ckpt.exists():
        raise FileNotFoundError(f"VAE ckpt 不存在: {ckpt}（请先运行 scripts/03_train_vae.py）")
    model, _payload = load_checkpoint(ckpt, map_location=device)
    model.to(device)
    model.eval()
    for parameter in model.parameters():
        parameter.requires_grad = False
    return model


def _save_diffusion_checkpoint(
    path: Path,
    *,
    model: DiffusionTransformer,
    projector: torch.nn.Module,
    optimizer: torch.optim.Optimizer,
    epoch: int,
    step: int,
    best_val: float,
    vae_ckpt: str,
    text_model: str,
) -> Path:
    """保存扩散 ckpt（DiT + 文本投影 + 优化器 + 元信息）。"""
    target = P.resolve_out(path, mkparents=True)
    torch.save(
        {
            "model": model.state_dict(),
            "projector": projector.state_dict(),
            "optimizer": optimizer.state_dict(),
            "epoch": int(epoch),
            "step": int(step),
            "best_val": float(best_val),
            "layout_version": int(LAYOUT_VERSION),
            "vae_ckpt": str(vae_ckpt),
            "text_model": str(text_model),
            "model_config": model.describe(),
        },
        target,
    )
    return target


def _load_diffusion_checkpoint(
    path: str | Path,
    *,
    model: DiffusionTransformer,
    projector: torch.nn.Module,
    optimizer: torch.optim.Optimizer | None = None,
    device: torch.device,
) -> dict[str, Any]:
    """加载扩散 ckpt（兼容只读元信息的场景）。"""
    payload = torch.load(Path(path), map_location=device, weights_only=False)
    model.load_state_dict(payload["model"])
    projector.load_state_dict(payload["projector"])
    if optimizer is not None and "optimizer" in payload:
        optimizer.load_state_dict(payload["optimizer"])
    return payload


def train_diffusion(
    cfg: TTMConfig,
    *,
    max_steps_per_epoch: int | None = None,
    max_epochs: int | None = None,
    resume: str | None = None,
    logger: Any | None = None,
    max_minutes: float | None = None,
) -> DiffusionTrainResult:
    """训练潜空间 DiT（CFG + DDPM ε-预测）。

    Args:
        cfg: 生效配置。
        max_steps_per_epoch: 限制每 epoch 步数（冒烟/测试用）。
        max_epochs: 覆盖 ``cfg.diffusion.epochs``。
        resume: 从该 ckpt 续训。
        logger: 可注入 logger。
        max_minutes: 墙钟预算（分钟）。**每跑完一个 epoch** 检查一次，超预算就
            停止并落盘——保证在有限时间里总能拿到可用的 ``best.pt``，而不是被
            外部 timeout 杀掉导致一个 ckpt 都没有。

    Returns:
        :class:`DiffusionTrainResult`。

    Raises:
        FileNotFoundError: 分片或 VAE ckpt 缺失。
        ValueError: train split 为空。
    """
    log = logger or get_logger("ttm.train.diffusion", script="04_train_diffusion")
    seed = set_seed(cfg.runtime.seed)
    device = torch.device(
        cfg.runtime.device if torch.cuda.is_available() or cfg.runtime.device == "cpu" else "cpu"
    )
    if device.type == "cuda":
        # 峰值显存只统计本次训练，避免被之前的进程/推理峰值污染
        torch.cuda.reset_peak_memory_stats(device)
    budget_s = float(max_minutes) * 60.0 if max_minutes else 0.0
    log.info("扩散训练开始: device=%s seed=%d", device, seed)
    started = time.perf_counter()

    # 注意：resolve_out 只创建「父目录」，要建目录本身必须用 safe_makedirs
    run_dir = P.safe_makedirs(cfg.run_dir("diffusion"))
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
        with_torch=True,
        class_balanced=cfg.data.class_balanced,
        target_wushu_ratio=cfg.data.balanced_target_wushu_ratio,
    )
    if len(train_set) == 0:
        raise ValueError("train split 没有样本，请检查 labels.jsonl 的 split 分布")
    try:
        val_set: MotionDataset | None = MotionDataset(
            cfg.shards_path(),
            cache_size=cache_size,
            window=cfg.data.window,
            feature_dim=cfg.motion.feature_dim,
            labels_path=labels_path,
            split="val",
            with_torch=True,
        )
        if len(val_set) == 0:
            val_set = None
    except FileNotFoundError:  # pragma: no cover
        val_set = None

    # --- 冻结的 VAE（潜变量目标）
    vae = load_vae_for_diffusion(cfg, device)
    log.info("VAE 已加载并冻结: %s", vae)

    # --- 冻结的文本编码器 + 条件库
    encoder = TextEncoder(
        cfg.text.encoder,
        model_path=cfg.text.model_path or None,
        proj_dim=cfg.text.proj_dim,
        frozen=cfg.text.frozen,
        max_len=cfg.text.max_len,
        device="cpu",  # bge 只在建库时跑一次，CPU 足够且省显存
        local_files_only=True,
    )
    bank = TextConditionBank(encoder, device=device)
    try:
        bank.build(train_set.labels)
        bank_source = f"bge:{encoder.model_source}"
        log.info("条件库就绪: %d 条 distinct 文本（%d 个 clip）", bank.size, len(train_set.labels))
    except (TextEncoderUnavailable, ValueError) as exc:
        log.warning("文本编码器不可用（%s），退化为无条件训练（guidance 将无效）", exc)
        bank_source = "null"

    # --- 可训练：文本投影 + DiT
    projector = encoder.projector.to(device).train()
    model = DiffusionTransformer(
        latent_dim=cfg.vae.latent_dim,
        d_model=cfg.diffusion.d_model,
        n_layers=cfg.diffusion.n_layers,
        n_heads=cfg.diffusion.n_heads,
        cond_dim=cfg.text.proj_dim,
        dropout=0.0,
    ).to(device)
    model.train()
    # 系数表必须跟设备走：留在 CPU 上会在 q_sample 里索引 CUDA 的时间步时炸。
    diffusion = GaussianDiffusion(timesteps=cfg.diffusion.timesteps, device=device)

    params = list(model.parameters()) + list(projector.parameters())
    optimizer = torch.optim.AdamW(params, lr=float(cfg.diffusion.lr), weight_decay=0.01)
    use_amp = device.type == "cuda" and cfg.runtime.precision in ("bf16", "fp16")
    amp_dtype = torch.bfloat16 if cfg.runtime.precision == "bf16" else torch.float16

    batch_size = max(1, int(cfg.diffusion.batch_size))
    # 类别均衡：训练集按 1/sqrt(freq) 加权，验证集保持自然分布（否则 metric 不可比）。
    # DataLoader 规定 sampler 与 shuffle 互斥，启用采样器时必须关掉 shuffle。
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

    history = DiffusionHistory()
    best_val = float("inf")
    start_epoch = 0
    global_step = 0
    if resume:
        payload = _load_diffusion_checkpoint(
            resume, model=model, projector=projector, optimizer=optimizer, device=device
        )
        best_val = float(payload.get("best_val", best_val))
        start_epoch = int(payload.get("epoch", 0)) + 1
        global_step = int(payload.get("step", 0))
        log.info("续训: %s（epoch=%d, best_val=%.5f）", resume, start_epoch, best_val)

    epochs = int(max_epochs if max_epochs is not None else cfg.diffusion.epochs)
    loss_path = P.resolve_out(run_dir / "loss.jsonl", mkparents=True)
    # 全新训练截断、续训追加：否则重跑一次会把上一轮的 step 混进来，曲线直接失去意义
    loss_file = open(loss_path, "a" if resume else "w", encoding="utf-8")
    # 两个生成器，设备必须分开——这是踩过的坑：
    # ``p_losses`` 里是 ``torch.randn_like(latent)``，latent 在 CUDA 上，
    # 传 CPU generator 会直接抛 ``Expected a 'cuda' device type for generator but found 'cpu'``；
    # 而 ``bank.gather`` 是在 CPU 的 clip_id 列表上做 ``torch.randint``，只能用 CPU generator。
    # 用同一个生成器两边都过不去，所以显式拆开。
    generator_cpu = torch_generator(seed)
    generator = torch_generator(seed, device=device.type)
    token_mask_ok = True
    vae_ckpt_recorded = resolve_vae_checkpoint(cfg).as_posix()
    budget_hit = False

    try:
        for epoch in range(start_epoch, epochs):
            if budget_hit:
                break
            epoch_started = time.perf_counter()
            total = 0.0
            count = 0
            for step, batch in enumerate(train_loader):
                if max_steps_per_epoch and step >= max_steps_per_epoch:
                    break
                feat = batch["feat"].to(device, non_blocking=True)  # [B,146,150]
                frame_mask = batch["mask"].to(device=device, dtype=torch.float32)
                with torch.no_grad():
                    mu, _logvar = vae.encode(feat)
                token_mask = frames_to_token_mask(frame_mask)
                if token_mask.shape[1] != mu.shape[1]:
                    # 极端形状兜底（理论上 ceil(T/4) 恒等）
                    token_mask_ok = False
                    token_mask = torch.ones(
                        mu.shape[0], mu.shape[1], device=device, dtype=torch.float32
                    )
                cond = projector(
                    bank.gather(batch["clip_id"], generator=generator_cpu).to(device)
                )

                with torch.autocast(
                    device_type=device.type, dtype=amp_dtype, enabled=use_amp
                ):
                    output = diffusion.p_losses(
                        model,
                        mu,
                        cond,
                        cfg_dropout=float(cfg.diffusion.cfg_dropout),
                        mask=token_mask,
                        generator=generator,
                    )
                loss = output.loss
                optimizer.zero_grad(set_to_none=True)
                loss.backward()
                torch.nn.utils.clip_grad_norm_(params, 1.0)
                optimizer.step()

                value = float(loss.detach().cpu())
                total += value
                count += 1
                global_step += 1
                loss_file.write(
                    json.dumps(
                        {
                            "epoch": epoch,
                            "step": global_step,
                            "loss": round(value, 6),
                            "dropped": int(output.n_dropped),
                        }
                    )
                    + "\n"
                )
                if step % 50 == 0:
                    loss_file.flush()

            train_loss = total / max(1, count)
            val_loss = (
                _evaluate_diffusion(model, diffusion, vae, projector, bank, val_loader, device, cfg)
                if val_loader is not None
                else train_loss
            )
            history.train_loss.append(round(train_loss, 6))
            history.val_loss.append(round(val_loss, 6))
            history.steps.append(global_step)
            elapsed = time.perf_counter() - epoch_started
            log.info(
                "epoch %03d | train %.5f | val %.5f | %d step | %.1fs",
                epoch,
                train_loss,
                val_loss,
                count,
                elapsed,
            )
            _save_diffusion_checkpoint(
                run_dir / "last.pt",
                model=model,
                projector=projector,
                optimizer=optimizer,
                epoch=epoch,
                step=global_step,
                best_val=min(best_val, val_loss),
                vae_ckpt=vae_ckpt_recorded,
                text_model=bank_source,
            )
            if val_loss < best_val:
                best_val = val_loss
                _save_diffusion_checkpoint(
                    run_dir / "best.pt",
                    model=model,
                    projector=projector,
                    optimizer=optimizer,
                    epoch=epoch,
                    step=global_step,
                    best_val=best_val,
                    vae_ckpt=vae_ckpt_recorded,
                    text_model=bank_source,
                )
            if budget_s > 0:
                spent = time.perf_counter() - started
                if spent >= budget_s:
                    budget_hit = True
                    log.warning(
                        "已达墙钟预算 %.1f 分钟（实际 %.1f 分钟），停止训练并保留 best.pt",
                        budget_s / 60.0,
                        spent / 60.0,
                    )
    finally:
        loss_file.close()

    history.elapsed_s = time.perf_counter() - started
    if device.type == "cuda":
        history.peak_memory_mb = float(torch.cuda.max_memory_allocated(device) / (1024**2))
    if not token_mask_ok:
        log.warning("token 掩码长度与潜变量长度不一致，已退化为全 1（请检查 DOWNSAMPLE）")
    if not (run_dir / "best.pt").is_file():
        log.error("训练结束但 best.pt 不存在：val 从未改善，请检查 val split 与 lr")

    result = DiffusionTrainResult(
        history=history,
        best_val=best_val,
        param_count=model.param_count(),
        ckpt_path=str((run_dir / "best.pt").as_posix()),
        run_dir=str(run_dir.as_posix()),
    )
    (run_dir / "summary.json").write_text(
        json.dumps(result.to_dict(), ensure_ascii=False, indent=2), encoding="utf-8"
    )
    log.info(
        "扩散训练结束: best_val=%.5f, loss 下降 %.1f%%, 峰值显存 %.0f MB, 用时 %.1f min",
        best_val,
        history.loss_drop_ratio() * 100,
        history.peak_memory_mb,
        history.elapsed_s / 60.0,
    )
    return result


def _evaluate_diffusion(
    model: DiffusionTransformer,
    diffusion: GaussianDiffusion,
    vae: MotionVAE,
    projector: torch.nn.Module,
    bank: TextConditionBank,
    loader: DataLoader | None,
    device: torch.device,
    cfg: TTMConfig,
) -> float:
    """在验证集上计算平均 ε-预测损失（不采样，只算 loss）。"""
    if loader is None:
        return float("inf")
    model.eval()
    projector.eval()
    total = 0.0
    count = 0
    with torch.no_grad():
        for batch in loader:
            feat = batch["feat"].to(device)
            frame_mask = batch["mask"].to(device=device, dtype=torch.float32)
            mu, _ = vae.encode(feat)
            token_mask = frames_to_token_mask(frame_mask)
            cond = projector(bank.gather(batch["clip_id"]))
            output = diffusion.p_losses(
                model, mu, cond, cfg_dropout=0.0, mask=token_mask
            )
            total += float(output.loss.detach().cpu())
            count += 1
    model.train()
    projector.train()
    return total / max(1, count)
