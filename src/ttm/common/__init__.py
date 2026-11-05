"""通用基础设施：路径与写盘守卫、日志、随机种子、配置、骨架常量。

本子包的模块**不依赖** numpy 之外的重型依赖（`config`/`constants`/`skeleton`
会用到 numpy），以保证 `python scripts/00_env_check.py` 在最小环境下即可运行。
"""

from __future__ import annotations

from ttm.common import logutil, paths, seed

__all__ = [
    "logutil",
    "paths",
    "seed",
]
