"""训练层：MotionVAE 训练（T05）与潜空间扩散训练（T07 追加）。

与 :mod:`ttm.models` 相同，顶层不 import 子模块（子模块依赖 torch）。

用法::

    from ttm.train.train_vae import train_vae
"""

from __future__ import annotations

__all__ = ["train_vae"]
