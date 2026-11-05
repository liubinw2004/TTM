"""MotionVAE：1D 卷积自编码器（latent 128，时序下采样 4× → 38 tokens）。

结构（``TTM-ARCH.md`` §1.2 / §10 T05）：

::

    输入 feat [B, 146, 150]
      │  Conv1d(146 -> hidden, k=7)  + GELU
      │  ResBlock(hidden)
      │  Conv1d(hidden -> hidden, k=4, s=4, p=1)   # 150 -> 38（= ceil(150/4)）
      │  ResBlock(hidden)
      └─ Conv1d(hidden -> 2*latent, k=3)  ->  (mu, logvar)  [B, 38, 128]

    潜变量 z [B, 38, 128]
      │  Conv1d(latent -> hidden, k=3) + GELU
      │  ResBlock(hidden)
      │  ConvTranspose1d(hidden -> hidden, k=4, s=4, p=1)  # 38 -> 150（精确对称）
      │  ResBlock(hidden)
      └─ Conv1d(hidden -> 146, k=7)  ->  feat' [B, 146, 150]

为什么用 1D 卷积：时序局部性强、参数小、下采样天然对齐「帧 → token」；
不需要 ``einops``（ARCH §6 明确不引入），纯 ``transpose/reshape`` 即可。
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

import torch
import torch.nn as nn
import torch.nn.functional as F

from ttm.common import constants as C

__all__ = [
    "MotionVAE",
    "ResBlock",
    "VAEOutput",
    "kl_divergence",
    "resample_length_after_conv",
]


def resample_length_after_conv(length: int, kernel: int, stride: int, padding: int) -> int:
    """计算 Conv1d 输出的时间长度（与 PyTorch 语义一致）。

    Args:
        length: 输入长度。
        kernel: 卷积核长度。
        stride: 步长。
        padding: 单侧 padding。

    Returns:
        输出长度 ``floor((L + 2p - k)/s) + 1``。
    """
    return (int(length) + 2 * int(padding) - int(kernel)) // int(stride) + 1


@dataclass
class VAEOutput:
    """一次前向的完整输出（便于训练循环与调试）。"""

    recon: torch.Tensor
    mu: torch.Tensor
    logvar: torch.Tensor
    z: torch.Tensor
    extra: dict[str, Any] = field(default_factory=dict)


class ResBlock(nn.Module):
    """单层残差块：``x + Conv1d(GELU(Conv1d(x)))``。"""

    def __init__(self, channels: int, kernel_size: int = 3) -> None:
        """初始化。

        Args:
            channels: 通道数。
            kernel_size: 卷积核长度（默认 3，保持长度不变）。
        """
        super().__init__()
        padding = kernel_size // 2
        self.conv1 = nn.Conv1d(channels, channels, kernel_size, padding=padding)
        self.conv2 = nn.Conv1d(channels, channels, kernel_size, padding=padding)
        self.act = nn.GELU()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """前向。

        Args:
            x: ``[B, C, T]``。

        Returns:
            ``[B, C, T]``。
        """
        residual = x
        out = self.act(self.conv1(x))
        out = self.conv2(out)
        return self.act(out + residual)


class MotionVAE(nn.Module):
    """运动特征自编码器（卷积版 VAE）。"""

    def __init__(
        self,
        feature_dim: int = C.FEATURE_DIM,
        latent_dim: int = C.LATENT_DIM,
        hidden: int = 512,
        downsample: int = C.DOWNSAMPLE,
        *,
        n_res_blocks: int = 2,
        kernel_size: int = 7,
        logvar_bias: float = -2.0,
        logvar_clamp: tuple[float, float] = (-10.0, 2.0),
    ) -> None:
        """初始化。

        Args:
            feature_dim: 输入/输出特征维数（146）。
            latent_dim: 潜变量通道数（128）。
            hidden: 中间通道数。
            downsample: 时序下采样倍数（4，须能整除卷积核与步长）。
            n_res_blocks: 编码器/解码器各阶段后的残差块数量。
            kernel_size: 首尾卷积核长度（奇数，默认 7）。
            logvar_bias: ``logvar`` 分支的偏置初始值。

                **为什么必须有这个**：``logvar`` 默认初始为 0，即后验标准差
                ``exp(0.5*0) = 1``，与归一化后的特征同量级——采样出的 ``z``
                几乎全是噪声，解码器的最优反应就是「忽略 z、输出平均姿态」。
                而 ``(mu, logvar) = (0, 0)`` 恰好是 KL 项
                ``-0.5*(1 + logvar - mu² - exp(logvar))`` 的**驻点**（梯度为 0），
                KL 只会把参数吸在这个点上，无法把它们推出去。实测就是这样锁死的：
                ``mu std = 1e-4``、重建帧间方差只有真实数据的 1/17、
                ``joint_mae`` 卡在 19.4°。

                取 ``-2.0`` → 初始 ``std = exp(-1) ≈ 0.37``，``z ≈ mu``，
                解码器从一开始就「看得见」信号，重建梯度才能回流到 ``mu``，
                从而打破这个死锁。KL warmup 只负责后续平滑压缩，单独用它无效
                （在驻点上乘任何系数都还是 0 梯度）。
            logvar_clamp: ``logvar`` 的硬上下界。**KL 项必须有上界**：它对
                ``exp(logvar)`` 是线性的，而梯度只有 ``kl_weight=1e-5`` 量级，
                所以个别样本的 ``logvar`` 会一路漂到 +6（std≈400）——此时单样本 KL
                就有 ``0.5*(e^6-1-6)*4864 ≈ 7.2e5``，val loss 会从 0.039 跳到 80+。
                实测 M2 的 200 个 epoch 里有 8 次这种尖峰（train 稳定在 0.030），
                best.pt 的选择因此被污染。
        """
        super().__init__()
        if downsample < 1:
            raise ValueError(f"downsample 必须 >= 1，实际 {downsample}")
        if kernel_size % 2 == 0:
            raise ValueError(f"kernel_size 必须为奇数，实际 {kernel_size}")
        self.feature_dim = int(feature_dim)
        self.latent_dim = int(latent_dim)
        self.hidden = int(hidden)
        self.downsample = int(downsample)
        self.n_res_blocks = int(n_res_blocks)

        mid = max(32, self.hidden // 2)
        pad = kernel_size // 2

        # ---------------- encoder ----------------
        self.enc_in = nn.Conv1d(self.feature_dim, mid, kernel_size, padding=pad)
        self.enc_res1 = nn.Sequential(*[ResBlock(mid) for _ in range(self.n_res_blocks)])
        self.enc_down = nn.Conv1d(
            mid, self.hidden, kernel_size=self.downsample, stride=self.downsample, padding=1
        )
        self.enc_res2 = nn.Sequential(*[ResBlock(self.hidden) for _ in range(self.n_res_blocks)])
        self.enc_out = nn.Conv1d(self.hidden, 2 * self.latent_dim, 3, padding=1)
        self.act = nn.GELU()

        # ``enc_out`` 的后半通道是 logvar。默认 bias=0 → std=1，采样几乎全是噪声，
        # 解码器会学会忽略 z（后验塌缩）。这里把 logvar 的初始偏置压低，
        # 让初始 std≈0.37，保证训练一开始 z 就携带可解码的信号。
        self.logvar_bias = float(logvar_bias)
        low, high = (float(logvar_clamp[0]), float(logvar_clamp[1]))
        if not low < high:
            raise ValueError(f"logvar_clamp 应为 (下界, 上界)，实际 {logvar_clamp}")
        self.logvar_clamp = (low, high)
        with torch.no_grad():
            self.enc_out.bias[self.latent_dim :] = self.logvar_bias

        # ---------------- decoder ----------------
        self.dec_in = nn.Conv1d(self.latent_dim, self.hidden, 3, padding=1)
        self.dec_res1 = nn.Sequential(*[ResBlock(self.hidden) for _ in range(self.n_res_blocks)])
        self.dec_up = nn.ConvTranspose1d(
            self.hidden,
            mid,
            kernel_size=self.downsample,
            stride=self.downsample,
            padding=1,
        )
        self.dec_res2 = nn.Sequential(*[ResBlock(mid) for _ in range(self.n_res_blocks)])
        self.dec_out = nn.Conv1d(mid, self.feature_dim, kernel_size, padding=pad)

    # -- 基本属性 -----------------------------------------------------------
    @property
    def token_length(self) -> int:
        """150 帧窗口对应的潜 token 数（38）。"""
        return resample_length_after_conv(C.WINDOW, self.downsample, self.downsample, 1)

    def param_count(self) -> int:
        """返回可训练参数总量。"""
        return int(sum(p.numel() for p in self.parameters() if p.requires_grad))

    # -- 编解码 -------------------------------------------------------------
    def encode(self, feat: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        """编码为 ``(mu, logvar)``。

        Args:
            feat: ``[B, 146, T]``（也接受 ``[B, T, 146]``，自动转置）。

        Returns:
            ``mu``/``logvar``：``[B, T', 128]``。
        """
        x = self._as_channel_first(feat)
        h = self.act(self.enc_in(x))
        h = self.enc_res1(h)
        h = self.act(self.enc_down(h))
        h = self.enc_res2(h)
        h = self.enc_out(h)  # [B, 2*latent, T']
        mu, logvar = torch.chunk(h, 2, dim=1)
        # logvar 必须夹住，否则个别样本会漂到 +6（std≈400），KL 单样本就能贡献
        # 0.5*(e^6-1-6)*4864 ≈ 7.2e5 → val loss 从 0.039 跳到 80+。
        # 实测 M2：200 个 epoch 里有 8 个 epoch 的 val 呈 6~112 的尖峰，而 train
        # 稳定在 0.030——不是模型坏了，是 KL 项没上界。默认上限 2.0（std≤7.4，
        # 对归一化特征已经足够宽），下限 -10.0（std≥0.0067，防塌缩成确定性编码）。
        lo, hi = self.logvar_clamp
        logvar = logvar.clamp(min=lo, max=hi)
        return mu.transpose(1, 2).contiguous(), logvar.transpose(1, 2).contiguous()

    def decode(self, z: torch.Tensor, *, target_len: int | None = None) -> torch.Tensor:
        """从潜变量解码特征。

        Args:
            z: ``[B, T', 128]``（也接受 ``[B, 128, T']``，自动转置）。
            target_len: 期望输出帧数；``None`` 时由卷积自然决定（150）。

        Returns:
            ``[B, 146, T]``。
        """
        latent = self._as_latent_channel_first(z)
        h = self.act(self.dec_in(latent))
        h = self.dec_res1(h)
        h = self.act(self.dec_up(h))
        h = self.dec_res2(h)
        out = self.dec_out(h)
        if target_len is not None and out.shape[-1] != int(target_len):
            out = F.interpolate(out, size=int(target_len), mode="linear", align_corners=False)
        return out

    def reparameterize(
        self,
        mu: torch.Tensor,
        logvar: torch.Tensor,
        *,
        generator: torch.Generator | None = None,
        sample: bool = True,
    ) -> torch.Tensor:
        """重参数化采样 ``z = mu + eps * std``。

        Args:
            mu: ``[B, T', 128]``。
            logvar: ``[B, T', 128]``（已裁剪到合理区间前会被 clamp）。
            generator: 可选的随机数发生器（用于可复现）。
            sample: ``False`` 时直接返回 ``mu``（推理/确定性前向）。

        Returns:
            ``[B, T', 128]``。
        """
        if not sample:
            return mu
        logvar_clamped = torch.clamp(logvar, -30.0, 20.0)
        std = torch.exp(0.5 * logvar_clamped)
        noise = torch.randn(
            mu.shape,
            dtype=mu.dtype,
            device=mu.device,
            generator=generator,
        )
        return mu + std * noise

    def forward(self, feat: torch.Tensor, *, sample: bool = True) -> VAEOutput:
        """完整前向：编码 → 重参数化 → 解码。

        Args:
            feat: ``[B, 146, T]``。
            sample: 是否采样（``False`` 时用 ``mu``，便于确定性重构评估）。

        Returns:
            :class:`VAEOutput`。
        """
        target_len = int(feat.shape[-1])
        mu, logvar = self.encode(feat)
        z = self.reparameterize(mu, logvar, sample=sample)
        recon = self.decode(z, target_len=target_len)
        return VAEOutput(recon=recon, mu=mu, logvar=logvar, z=z)

    # -- 形状工具 -----------------------------------------------------------
    def _as_channel_first(self, feat: torch.Tensor) -> torch.Tensor:
        """把 ``[B,T,146]`` 或 ``[B,146,T]`` 统一为 ``[B,146,T]``。"""
        if feat.dim() != 3:
            raise ValueError(f"feat 必须是 3 维，实际 {tuple(feat.shape)}")
        if feat.shape[1] == self.feature_dim:
            return feat
        if feat.shape[2] == self.feature_dim:
            return feat.transpose(1, 2).contiguous()
        raise ValueError(
            f"feat 任一维必须是 {self.feature_dim}，实际 {tuple(feat.shape)}"
        )

    def _as_latent_channel_first(self, z: torch.Tensor) -> torch.Tensor:
        """把 ``[B,T',128]`` 或 ``[B,128,T']`` 统一为 ``[B,128,T']``。"""
        if z.dim() != 3:
            raise ValueError(f"z 必须是 3 维，实际 {tuple(z.shape)}")
        if z.shape[1] == self.latent_dim:
            return z
        if z.shape[2] == self.latent_dim:
            return z.transpose(1, 2).contiguous()
        raise ValueError(f"z 任一维必须是 {self.latent_dim}，实际 {tuple(z.shape)}")

    # -- 存取 ---------------------------------------------------------------
    def describe(self) -> dict[str, Any]:
        """返回结构摘要（日志/ckpt 用）。"""
        return {
            "feature_dim": self.feature_dim,
            "latent_dim": self.latent_dim,
            "hidden": self.hidden,
            "downsample": self.downsample,
            "token_length": self.token_length,
            "params": self.param_count(),
            "logvar_bias": self.logvar_bias,
        }

    def __repr__(self) -> str:  # pragma: no cover - 调试用
        info = self.describe()
        return (
            f"MotionVAE(feature_dim={info['feature_dim']}, latent={info['latent_dim']}, "
            f"hidden={info['hidden']}, downsample={info['downsample']}, "
            f"tokens={info['token_length']}, params={info['params']:,})"
        )


def kl_divergence(mu: torch.Tensor, logvar: torch.Tensor) -> torch.Tensor:
    """标准高斯先验下的逐样本 KL 散度（对通道与时间求和后取 batch 平均）。

    Args:
        mu: ``[B, T', D]``。
        logvar: ``[B, T', D]``。

    Returns:
        标量张量。
    """
    logvar_clamped = torch.clamp(logvar, -30.0, 20.0)
    per_sample = -0.5 * torch.sum(1.0 + logvar_clamped - mu.pow(2) - logvar_clamped.exp(), dim=(1, 2))
    return per_sample.mean()
