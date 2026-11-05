"""模型层：MotionVAE、中文文本编码器、潜空间扩散 Transformer。

注意：本包的子模块都依赖 torch，因此 :mod:`ttm.models` 顶层**不 import 任何
子模块**（避免无 torch 环境下 `import ttm.models` 直接失败）。
请显式导入所需模块，例如 ::

    from ttm.models.vae import MotionVAE
    from ttm.models.diffusion import DiffusionTransformer
"""

from __future__ import annotations

__all__ = ["diffusion", "text_encoder", "vae"]
