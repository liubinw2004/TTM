"""推理层：潜空间采样（sampling）、窗口融合（blend）、端到端生成（generate）。

子模块依赖 torch，因此顶层不做 eager import，按需显式导入::

    from ttm.infer.generate import MotionGenerator, load_generator
"""

from __future__ import annotations

__all__ = ["blend", "generate", "sampling"]
