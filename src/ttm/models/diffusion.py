"""潜空间扩散：DiT（Transformer）+ DDPM 训练调度 + DDIM 采样。

结构（``TTM-ARCH.md`` §1.2 / §5.8）：

::

    x_t  [B, 38, 128]   潜序列（VAE 下采样后 38 tokens）
      │  Linear(128 -> d_model=256) + 位置编码（可学习）
      ▼
    8 × DiTBlock（自注意力 + 交叉注意力吃文本条件 + FFN，pre-LN）
      │  LayerNorm + Linear(d_model -> 128)
      ▼
    eps  [B, 38, 128]   预测噪声

为什么可行（§7.4 显存估算）：潜序列只有 38 tokens，注意力开销可忽略；
8 层 d=256 的激活量约 0.16 GB，训练峰值 ≈ 1.0–1.5 GB ≪ 5 GB 上限。

DDPM 训练：标准 ε-预测 + MSE；条件 dropout ``cfg_dropout`` 用于 CFG。
DDIM 采样：η=0 的确定性 DDIM，50 步；CFG 双前向（有条件 / 无条件）。
"""

from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Any, Sequence

import torch
import torch.nn as nn
import torch.nn.functional as F

from ttm.common import constants as C

__all__ = [
    "DDIMResult",
    "DDPMOutput",
    "DiffusionTransformer",
    "DiTBlock",
    "GaussianDiffusion",
    "sinusoidal_embedding",
]


def sinusoidal_embedding(timesteps: torch.Tensor, dim: int, *, max_period: float = 10000.0) -> torch.Tensor:
    """标准正弦时间步编码（Transformer 版）。

    Args:
        timesteps: ``[B]`` 整数/浮点时间步。
        dim: 输出维数（必须为偶数）。
        max_period: 频率基准。

    Returns:
        ``[B, dim]``。
    """
    if dim % 2 != 0:
        raise ValueError(f"sinusoidal_embedding 的 dim 必须为偶数，实际 {dim}")
    half = dim // 2
    device = timesteps.device
    exponent = -math.log(max_period) * torch.arange(half, device=device, dtype=torch.float32) / half
    frequencies = torch.exp(exponent)
    args = timesteps.float().unsqueeze(-1) * frequencies.unsqueeze(0)
    return torch.cat([torch.cos(args), torch.sin(args)], dim=-1)


class DiTBlock(nn.Module):
    """一层：自注意力 → 交叉注意力（文本条件）→ FFN，全部 pre-LN + 残差。"""

    def __init__(
        self,
        d_model: int,
        n_heads: int,
        cond_dim: int,
        *,
        ffn_mult: int = 4,
        dropout: float = 0.0,
    ) -> None:
        """初始化。

        Args:
            d_model: 隐层维度（256）。
            n_heads: 注意力头数（8）。
            cond_dim: 条件向量维度（256）。
            ffn_mult: FFN 扩张倍数（4 → 1024）。
            dropout: dropout 比例。
        """
        super().__init__()
        if d_model % n_heads != 0:
            raise ValueError(f"d_model({d_model}) 必须能被 n_heads({n_heads}) 整除")
        self.d_model = int(d_model)
        self.n_heads = int(n_heads)
        self.head_dim = self.d_model // self.n_heads
        self.cond_dim = int(cond_dim)

        self.ln_self = nn.LayerNorm(self.d_model)
        self.self_qkv = nn.Linear(self.d_model, 3 * self.d_model)
        self.self_out = nn.Linear(self.d_model, self.d_model)

        self.ln_cross = nn.LayerNorm(self.d_model)
        self.cross_q = nn.Linear(self.d_model, self.d_model)
        self.cross_kv = nn.Linear(self.cond_dim, 2 * self.d_model)
        self.cross_out = nn.Linear(self.d_model, self.d_model)

        self.ln_ff = nn.LayerNorm(self.d_model)
        self.ff1 = nn.Linear(self.d_model, self.d_model * ffn_mult)
        self.ff2 = nn.Linear(self.d_model * ffn_mult, self.d_model)

        self.act = nn.GELU()
        self.dropout = nn.Dropout(dropout)

    def _heads(self, tensor: torch.Tensor) -> torch.Tensor:
        """``[B,T,d]`` → ``[B,h,T,dh]``。"""
        batch, length, _ = tensor.shape
        return tensor.view(batch, length, self.n_heads, self.head_dim).transpose(1, 2)

    def _merge(self, tensor: torch.Tensor) -> torch.Tensor:
        """``[B,h,T,dh]`` → ``[B,T,d]``。"""
        batch, _, length, _ = tensor.shape
        return tensor.transpose(1, 2).contiguous().view(batch, length, self.d_model)

    def forward(self, x: torch.Tensor, cond: torch.Tensor) -> torch.Tensor:
        """前向。

        Args:
            x: ``[B, T, d_model]``。
            cond: ``[B, cond_dim]`` 或 ``[B, S, cond_dim]``。

        Returns:
            ``[B, T, d_model]``。
        """
        # --- 自注意力
        residual = x
        hidden = self.ln_self(x)
        query, key, value = self.self_qkv(hidden).chunk(3, dim=-1)
        attended = F.scaled_dot_product_attention(
            self._heads(query), self._heads(key), self._heads(value)
        )
        x = residual + self.dropout(self.self_out(self._merge(attended)))

        # --- 交叉注意力（文本条件）
        residual = x
        hidden = self.ln_cross(x)
        cond_tokens = cond.unsqueeze(1) if cond.dim() == 2 else cond
        cross_k, cross_v = self.cross_kv(cond_tokens).chunk(2, dim=-1)
        attended = F.scaled_dot_product_attention(
            self._heads(self.cross_q(hidden)), self._heads(cross_k), self._heads(cross_v)
        )
        x = residual + self.dropout(self.cross_out(self._merge(attended)))

        # --- FFN
        residual = x
        hidden = self.ln_ff(x)
        x = residual + self.dropout(self.ff2(self.dropout(self.act(self.ff1(hidden)))))
        return x


class DiffusionTransformer(nn.Module):
    """潜空间 ε-预测 Transformer（DiT）。"""

    def __init__(
        self,
        latent_dim: int = C.LATENT_DIM,
        d_model: int = 256,
        n_layers: int = 8,
        n_heads: int = 8,
        cond_dim: int = 256,
        *,
        max_tokens: int = 64,
        ffn_mult: int = 4,
        dropout: float = 0.0,
    ) -> None:
        """初始化。

        Args:
            latent_dim: 潜变量通道数（128）。
            d_model: 隐层维度（256）。
            n_layers: DiTBlock 层数（8）。
            n_heads: 注意力头数（8）。
            cond_dim: 文本条件维度（256）。
            max_tokens: 支持的最大 token 数（64 > 38，留长动作余量）。
            ffn_mult: FFN 扩张倍数。
            dropout: dropout 比例（默认 0，便于确定性断言）。
        """
        super().__init__()
        self.latent_dim = int(latent_dim)
        self.d_model = int(d_model)
        self.n_layers = int(n_layers)
        self.n_heads = int(n_heads)
        self.cond_dim = int(cond_dim)
        self.max_tokens = int(max_tokens)

        self.input_proj = nn.Linear(self.latent_dim, self.d_model)
        self.position = nn.Parameter(torch.zeros(1, self.max_tokens, self.d_model))
        nn.init.normal_(self.position, std=0.02)

        self.time_mlp = nn.Sequential(
            nn.Linear(self.d_model, self.d_model * 4),
            nn.SiLU(),
            nn.Linear(self.d_model * 4, self.d_model),
        )
        # 条件嵌入用 2 层 MLP 而非 LayerNorm：
        # LayerNorm 会把**任何常量向量**都归一到 0（均值相消），使「全 0 条件」
        # 与「全 1 条件」不可区分，也会丢掉文本嵌入的模长信息；MLP 不损失可辨性。
        self.cond_mlp = nn.Sequential(
            nn.Linear(self.cond_dim, self.cond_dim),
            nn.SiLU(),
            nn.Linear(self.cond_dim, self.cond_dim),
        )

        self.blocks = nn.ModuleList(
            [
                DiTBlock(self.d_model, self.n_heads, self.cond_dim, ffn_mult=ffn_mult, dropout=dropout)
                for _ in range(self.n_layers)
            ]
        )
        self.final_norm = nn.LayerNorm(self.d_model)
        self.output_proj = nn.Linear(self.d_model, self.latent_dim)

    # -- 属性 ---------------------------------------------------------------
    def param_count(self) -> int:
        """可训练参数总量（预期 8–12M）。"""
        return int(sum(p.numel() for p in self.parameters() if p.requires_grad))

    def tokens_for_frames(self, n_frames: int, downsample: int = C.DOWNSAMPLE) -> int:
        """给定目标帧数推算潜 token 数。

        Args:
            n_frames: 目标帧数（如 150 / 360）。
            downsample: VAE 下采样倍数。

        Returns:
            ``ceil(n_frames / downsample)``。
        """
        return max(1, math.ceil(int(n_frames) / int(downsample)))

    # -- 前向 ---------------------------------------------------------------
    def forward(self, x_t: torch.Tensor, t: torch.Tensor, cond: torch.Tensor) -> torch.Tensor:
        """预测噪声 ε。

        Args:
            x_t: ``[B, T, latent_dim]`` 加噪潜序列。
            t: ``[B]`` 时间步（0..T-1）。
            cond: ``[B, cond_dim]`` 文本条件。

        Returns:
            ``[B, T, latent_dim]`` 预测噪声。
        """
        if x_t.dim() != 3 or x_t.shape[-1] != self.latent_dim:
            raise ValueError(f"x_t 形状应为 [B,T,{self.latent_dim}]，实际 {tuple(x_t.shape)}")
        if t.dim() != 1 or t.shape[0] != x_t.shape[0]:
            raise ValueError(f"t 形状应为 [{x_t.shape[0]}]，实际 {tuple(t.shape)}")
        if cond.dim() != 2 or cond.shape[-1] != self.cond_dim:
            raise ValueError(f"cond 形状应为 [B,{self.cond_dim}]，实际 {tuple(cond.shape)}")

        tokens = x_t.shape[1]
        hidden = self.input_proj(x_t)
        hidden = hidden + self._position_embedding(tokens, hidden.device, hidden.dtype)
        hidden = hidden + self.time_mlp(sinusoidal_embedding(t, self.d_model)).unsqueeze(1)

        condition = self.cond_mlp(cond)
        for block in self.blocks:
            hidden = block(hidden, condition)
        return self.output_proj(self.final_norm(hidden))

    def _position_embedding(self, tokens: int, device: torch.device, dtype: torch.dtype) -> torch.Tensor:
        """取长度为 ``tokens`` 的位置编码（超出预训练长度时线性插值）。"""
        if tokens <= self.max_tokens:
            return self.position[:, :tokens, :].to(device=device, dtype=dtype)
        weight = self.position.transpose(1, 2)  # [1,d,max]
        resized = F.interpolate(weight, size=tokens, mode="linear", align_corners=False)
        return resized.transpose(1, 2).to(device=device, dtype=dtype)

    def describe(self) -> dict[str, Any]:
        """结构摘要。"""
        return {
            "latent_dim": self.latent_dim,
            "d_model": self.d_model,
            "n_layers": self.n_layers,
            "n_heads": self.n_heads,
            "cond_dim": self.cond_dim,
            "max_tokens": self.max_tokens,
            "params": self.param_count(),
        }

    def __repr__(self) -> str:  # pragma: no cover - 调试用
        info = self.describe()
        return (
            f"DiffusionTransformer(d_model={info['d_model']}, layers={info['n_layers']}, "
            f"heads={info['n_heads']}, latent={info['latent_dim']}, params={info['params']:,})"
        )


@dataclass
class DDPMOutput:
    """一次 DDPM 训练前向的输出。"""

    loss: torch.Tensor
    eps_pred: torch.Tensor
    eps_true: torch.Tensor
    n_dropped: int = 0


@dataclass
class DDIMResult:
    """DDIM 采样结果。"""

    latent: torch.Tensor
    steps: int
    guidance: float


class GaussianDiffusion:
    """DDPM 训练调度 + DDIM 采样（纯 torch，无外部依赖）。"""

    def __init__(
        self,
        timesteps: int = 1000,
        *,
        beta_start: float = 1e-4,
        beta_end: float = 0.02,
        device: torch.device | str = "cpu",
    ) -> None:
        """初始化并预计算所有系数。

        Args:
            timesteps: 扩散步数（1000）。
            beta_start: β 起始值。
            beta_end: β 终止值。
            device: 系数张量所在设备。
        """
        if timesteps < 2:
            raise ValueError(f"timesteps 必须 >= 2，实际 {timesteps}")
        self.timesteps = int(timesteps)
        self.device = torch.device(device)

        betas = torch.linspace(beta_start, beta_end, self.timesteps, dtype=torch.float64)
        alphas = 1.0 - betas
        alphas_cumprod = torch.cumprod(alphas, dim=0)
        self.betas = betas.to(self.device)
        self.alphas = alphas.to(self.device)
        self.alphas_cumprod = alphas_cumprod.to(self.device)
        self.sqrt_alphas_cumprod = torch.sqrt(alphas_cumprod).to(self.device)
        self.sqrt_one_minus_alphas_cumprod = torch.sqrt(1.0 - alphas_cumprod).to(self.device)

    def to(self, device: torch.device | str) -> "GaussianDiffusion":
        """迁移所有系数到指定设备。"""
        self.device = torch.device(device)
        for name in (
            "betas",
            "alphas",
            "alphas_cumprod",
            "sqrt_alphas_cumprod",
            "sqrt_one_minus_alphas_cumprod",
        ):
            setattr(self, name, getattr(self, name).to(self.device))
        return self

    # -- 训练 ---------------------------------------------------------------
    def q_sample(
        self,
        x0: torch.Tensor,
        t: torch.Tensor,
        noise: torch.Tensor | None = None,
        *,
        generator: torch.Generator | None = None,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """前向扩散：``x_t = √ᾱ_t · x0 + √(1-ᾱ_t) · ε``。

        Args:
            x0: ``[B, T, D]`` 干净潜变量。
            t: ``[B]`` 时间步。
            noise: 可选噪声；``None`` 时按 ``generator`` 采样。
            generator: 随机数发生器（可复现）。

        Returns:
            ``(x_t, noise)``。
        """
        if noise is None:
            noise = torch.randn(x0.shape, dtype=x0.dtype, device=x0.device, generator=generator)
        # 系数表的设备可能与 x0 不同（``GaussianDiffusion()`` 默认在 CPU 上建表，
        # 而 x0 在 CUDA 上）。**索引必须用表所在设备的 index**，否则会抛
        # ``indices should be either on cpu or on the same device as the indexed
        # tensor (cpu)``——这个坑实测在扩散训练第 1 步就炸，且因为脚本吞了堆栈
        # 只报一句 "训练中断"，很难定位。取出来的系数再搬到 x0 的设备/精度。
        table_device = self.sqrt_alphas_cumprod.device
        index = t.to(device=table_device, dtype=torch.long)
        sqrt_alpha = (
            self.sqrt_alphas_cumprod[index].to(device=x0.device, dtype=x0.dtype).view(-1, 1, 1)
        )
        sqrt_one_minus = (
            self.sqrt_one_minus_alphas_cumprod[index]
            .to(device=x0.device, dtype=x0.dtype)
            .view(-1, 1, 1)
        )
        return sqrt_alpha * x0 + sqrt_one_minus * noise, noise

    def p_losses(
        self,
        model: DiffusionTransformer,
        x0: torch.Tensor,
        cond: torch.Tensor,
        *,
        cond_null: torch.Tensor | None = None,
        cfg_dropout: float = 0.1,
        mask: torch.Tensor | None = None,
        generator: torch.Generator | None = None,
    ) -> DDPMOutput:
        """训练损失：ε-预测的 MSE（可选 CFG 条件 dropout）。

        Args:
            model: DiT。
            x0: ``[B, T, D]`` 潜变量（来自冻结的 VAE 编码）。
            cond: ``[B, cond_dim]`` 文本条件。
            cond_null: 无条件向量（默认零向量）。
            cfg_dropout: 条件置空概率（0.1）。
            mask: ``[B, T]`` token 级掩码（``1`` = 有效）；``None`` 时全有效。
            generator: 随机数发生器。

        Returns:
            :class:`DDPMOutput`（``n_dropped`` 为本次被置空的样本数）。
        """
        batch = x0.shape[0]
        if cond_null is None:
            cond_null = torch.zeros_like(cond)
        n_dropped = 0
        used_cond = cond
        if cfg_dropout > 0.0 and batch > 0:
            drop_mask = (
                torch.rand(batch, device=cond.device, generator=generator) < float(cfg_dropout)
            )
            n_dropped = int(drop_mask.sum().item())
            if n_dropped:
                used_cond = torch.where(drop_mask.unsqueeze(-1), cond_null, cond)

        t = torch.randint(
            low=0,
            high=self.timesteps,
            size=(batch,),
            device=x0.device,
            generator=generator,
        )
        x_t, noise = self.q_sample(x0, t, generator=generator)
        eps_pred = model(x_t, t, used_cond)

        if mask is None:
            loss = F.mse_loss(eps_pred, noise)
        else:
            weights = mask.to(eps_pred.dtype).unsqueeze(-1)
            denominator = weights.sum() * eps_pred.shape[-1]
            if float(denominator) <= 0.0:
                loss = eps_pred.sum() * 0.0
            else:
                loss = ((eps_pred - noise).pow(2) * weights).sum() / denominator
        return DDPMOutput(loss=loss, eps_pred=eps_pred, eps_true=noise, n_dropped=n_dropped)

    # -- 采样 ---------------------------------------------------------------
    @torch.no_grad()
    def ddim_sample(
        self,
        model: DiffusionTransformer,
        shape: Sequence[int],
        cond: torch.Tensor,
        *,
        steps: int = 50,
        guidance: float = 3.0,
        cond_null: torch.Tensor | None = None,
        generator: torch.Generator | None = None,
    ) -> DDIMResult:
        """DDIM 采样（η=0，确定性）+ CFG。

        Args:
            model: DiT（须为 ``eval()``）。
            shape: 输出形状 ``(B, T, D)``。
            cond: ``[B, cond_dim]`` 文本条件。
            steps: DDIM 步数（50）。
            guidance: CFG 强度（3.0）；``1.0`` 表示不加引导。
            cond_null: 无条件向量（零向量）。
            generator: 随机数发生器（可复现）。

        Returns:
            :class:`DDIMResult`。
        """
        steps = int(max(1, min(steps, self.timesteps)))
        device = cond.device
        batch = shape[0]
        if cond_null is None:
            cond_null = torch.zeros_like(cond)
        was_training = model.training
        model.eval()

        x = torch.randn(tuple(shape), device=device, generator=generator)
        # 均匀取 steps 个时间步（DDIM 的确定性采样序列）
        timestep_seq = torch.linspace(
            self.timesteps - 1, 0, steps=steps, device=device
        ).round().long()
        for index, t_value in enumerate(timestep_seq):
            t_batch = torch.full((batch,), int(t_value.item()), device=device, dtype=torch.long)
            if guidance != 1.0:
                eps_cond = model(x, t_batch, cond)
                eps_uncond = model(x, t_batch, cond_null)
                eps = eps_uncond + float(guidance) * (eps_cond - eps_uncond)
            else:
                eps = model(x, t_batch, cond)

            alpha = self.alphas_cumprod[int(t_value.item())]
            alpha_prev = (
                self.alphas_cumprod[int(timestep_seq[index + 1].item())]
                if index + 1 < steps
                else torch.tensor(1.0, dtype=alpha.dtype, device=device)
            )
            x0 = (x - torch.sqrt(1.0 - alpha) * eps) / torch.sqrt(alpha)
            x0 = x0.clamp(-10.0, 10.0)
            sigma = torch.sqrt(
                torch.clamp((1.0 - alpha_prev) / (1.0 - alpha) * (1.0 - alpha / alpha_prev), min=0.0)
            )
            direction = torch.sqrt(torch.clamp(1.0 - alpha_prev - sigma**2, min=0.0)) * eps
            x = torch.sqrt(alpha_prev) * x0 + direction
            if sigma > 0:  # pragma: no cover - η=0 时 sigma 恒为 0
                noise = torch.randn(x.shape, device=device, generator=generator)
                x = x + sigma * noise

        if was_training:
            model.train()
        return DDIMResult(latent=x, steps=steps, guidance=float(guidance))
