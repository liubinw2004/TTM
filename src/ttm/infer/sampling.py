"""潜空间采样：文本条件 → DDIM（η=0）+ CFG → 潜变量窗口。

只做「潜变量」这一层，解码与融合分别在 :mod:`ttm.infer.blend` /
:mod:`ttm.infer.generate` 里，保持职责单一，便于单测替换任一环节。
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Sequence

import torch

from ttm.common import constants as C
from ttm.models.diffusion import DiffusionTransformer, GaussianDiffusion

__all__ = ["LatentSampler", "SamplingConfig", "tokens_for_frames"]


def tokens_for_frames(n_frames: int, downsample: int = C.DOWNSAMPLE) -> int:
    """帧数 → 潜 token 数（与 VAE ``ceil(T/d)`` 一致）。"""
    import math  # noqa: PLC0415

    return max(1, math.ceil(int(n_frames) / int(downsample)))


@dataclass(frozen=True)
class SamplingConfig:
    """采样超参。"""

    #: DDIM 步数（越少越快，20–50 是常规区间）。
    steps: int = 50
    #: CFG 强度；1.0 表示不加引导。
    guidance: float = 3.0
    #: 窗口之间的重叠帧数（推理侧，ARCH §7）。
    overlap: int = C.OVERLAP
    #: 窗口帧数（= 训练窗口 150）。
    window: int = C.WINDOW
    #: 随机种子。
    seed: int = 42

    @property
    def stride(self) -> int:
        """窗口步长（帧）= window − overlap。"""
        return max(1, int(self.window) - int(self.overlap))

    def windows_for_frames(self, n_frames: int) -> int:
        """目标帧数需要几个窗口（ARCH §7：``K = ceil((N−O)/S)``）。"""
        n_frames = int(n_frames)
        if n_frames <= self.window:
            return 1
        return 1 + -(-(n_frames - self.window) // self.stride)


class LatentSampler:
    """封装 DiT + DDIM，把条件向量采样成潜变量窗口序列。"""

    def __init__(
        self,
        model: DiffusionTransformer,
        diffusion: GaussianDiffusion,
        *,
        latent_dim: int = C.LATENT_DIM,
        cond_dim: int = 256,
        device: torch.device | str = "cpu",
    ) -> None:
        """初始化。

        Args:
            model: DiT（会被置为 ``eval()``）。
            diffusion: DDPM/DDIM 调度器。
            latent_dim: 潜变量通道数。
            cond_dim: 条件维数。
            device: 设备。
        """
        self.model = model
        self.diffusion = diffusion
        self.latent_dim = int(latent_dim)
        self.cond_dim = int(cond_dim)
        self.device = torch.device(device)
        self.model.eval()

    def _generator(self, seed: int) -> torch.Generator:
        """按 seed 构造与**采样设备一致**的发生器。

        ``torch.randn(..., device="cuda", generator=g)`` 要求 ``g`` 也在 CUDA 上，
        否则抛 ``Expected a 'cuda' device type for generator but found 'cpu'``。
        DDIM 的 ``η = 0`` 路径本身是确定性的，但代码里仍保留 ``randn`` 分支
        （η > 0 时用来注入噪声），所以发生器必须跟设备走。

        Args:
            seed: 随机种子。

        Returns:
            与 :attr:`device` 同设备的 ``torch.Generator``。
        """
        generator = torch.Generator(device=self.device.type)
        generator.manual_seed(int(seed))
        return generator

    def sample_window(
        self,
        cond: torch.Tensor,
        *,
        n_tokens: int = C.VAE_TOKEN_LEN,
        steps: int = 50,
        guidance: float = 3.0,
        seed: int = 42,
    ) -> torch.Tensor:
        """采样一个窗口的潜变量。

        Args:
            cond: ``[1, cond_dim]`` 或 ``[cond_dim]``。
            n_tokens: token 数（150 帧 → 38）。
            steps: DDIM 步数。
            guidance: CFG 强度。
            seed: 随机种子（同 seed 逐元素可复现）。

        Returns:
            ``[1, n_tokens, latent_dim]``。
        """
        if cond.dim() == 1:
            cond = cond.unsqueeze(0)
        if cond.shape[-1] != self.cond_dim:
            raise ValueError(f"cond 最后一维应为 {self.cond_dim}，实际 {tuple(cond.shape)}")
        cond = cond.to(self.device)
        result = self.diffusion.ddim_sample(
            self.model,
            (1, int(n_tokens), self.latent_dim),
            cond,
            steps=int(steps),
            guidance=float(guidance),
            cond_null=torch.zeros_like(cond),
            generator=self._generator(seed),
        )
        return result.latent

    def sample_sequence(
        self,
        cond: torch.Tensor,
        *,
        n_windows: int,
        n_tokens: int = C.VAE_TOKEN_LEN,
        config: SamplingConfig | None = None,
    ) -> list[torch.Tensor]:
        """采样 ``n_windows`` 个窗口（每个窗口用 ``seed + i``，保证同族可复现）。

        Args:
            cond: 条件向量。
            n_windows: 窗口数（≥1）。
            n_tokens: 每窗 token 数。
            config: 采样配置（``None`` 时用默认值）。

        Returns:
            潜变量列表，每项 ``[1, n_tokens, latent_dim]``。
        """
        cfg = config or SamplingConfig()
        count = max(1, int(n_windows))
        return [
            self.sample_window(
                cond,
                n_tokens=n_tokens,
                steps=cfg.steps,
                guidance=cfg.guidance,
                seed=cfg.seed + index,
            )
            for index in range(count)
        ]

    def describe(self) -> dict[str, object]:
        """结构摘要（写进生成日志）。"""
        return {
            "latent_dim": self.latent_dim,
            "cond_dim": self.cond_dim,
            "device": str(self.device),
            "params": self.model.param_count(),
            "timesteps": int(self.diffusion.timesteps),
        }


def stack_latents(windows: Sequence[torch.Tensor]) -> torch.Tensor:
    """把窗口列表堆成 ``[K, n_tokens, latent_dim]``（便于一次性解码）。"""
    if not windows:
        raise ValueError("windows 为空")
    return torch.cat([window if window.dim() == 3 else window.unsqueeze(0) for window in windows], dim=0)
