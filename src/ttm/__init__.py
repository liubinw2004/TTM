"""TTM — 中文文本驱动骨骼动作生成系统。

本包为 TTM 项目的 Python 实现，按分层管线组织::

    data -> models -> train -> infer -> retarget/export -> serve

层与层之间只通过 `TTM-ARCH.md` §5 定义的数据契约耦合
（146 维运动特征 / npz 分片 / JSONL 索引 / ttm-motion/v1 Motion JSON）。

所有落盘路径必须经 :func:`ttm.common.paths.resolve_out`，
所有随机性必须经 :func:`ttm.common.seed.set_seed`。
"""

from __future__ import annotations

__all__ = [
    "__version__",
    "LAYOUT_VERSION",
    "PROJECT_NAME",
]

#: 项目名（用于日志、产物目录命名、HTTP 服务标题）。
PROJECT_NAME: str = "ttm_motion"

#: 包版本号。遵循语义化版本，首版为 0.1.0。
__version__: str = "0.1.0"

#: 运动特征/数据契约的布局版本号（TTM-ARCH.md §5.1「版本契约」）。
#:
#: 任何对 146 维特征布局、npz 字段集合的变更都必须递增该值，并同步写入
#: ``norm_stats.npz`` 与 Motion JSON 的 ``meta.layout_version``。
LAYOUT_VERSION: int = 1
