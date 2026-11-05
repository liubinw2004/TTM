"""随机种子统一设置。

契约（TTM-ARCH.md §5.1「随机种子」）：同一 ``seed`` + 同一指令 + 同一 ckpt
⇒ 逐帧可复现。因此**所有**随机性都必须经本模块的入口：

* :func:`set_seed` — 一次性设置 ``random`` / ``numpy`` / ``torch`` / ``torch.cuda``；
* :func:`numpy_rng` — 取得独立的 ``numpy.random.Generator``（不污染全局状态）；
* :func:`torch_generator` — 取得 ``torch.Generator``（DataLoader / 采样用）。

torch 为**可选依赖**：本模块在 torch 缺失时静默跳过其种子设置，
保证 ``00_env_check.py`` 等基础设施脚本在最小环境下可运行。
"""

from __future__ import annotations

import os
import random
from typing import Any, Final

import numpy as np

__all__ = [
    "DEFAULT_SEED",
    "derived_seed",
    "numpy_rng",
    "set_seed",
    "set_torch_deterministic",
    "torch_generator",
]

#: 默认随机种子（与 ``configs/default.yaml`` 的 ``runtime.seed`` 一致）。
DEFAULT_SEED: Final[int] = 42


def _seed_torch(seed: int, deterministic: bool) -> bool:
    """设置 torch / torch.cuda 种子。

    Args:
        seed: 种子值。
        deterministic: 是否尝试开启确定性算法（推理侧默认不强求，训练侧可选）。

    Returns:
        ``True`` 表示 torch 可用且已设置；``False`` 表示 torch 未安装。
    """
    try:
        import torch  # 延迟导入：torch 是可选依赖
    except Exception:  # pragma: no cover - 无 torch 环境
        return False

    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    if deterministic:
        # 注意：确定性算法可能使个别算子不可用，仅在明确要求时开启。
        try:
            torch.use_deterministic_algorithms(True, warn_only=True)
        except Exception:  # pragma: no cover - 旧版本 torch
            pass
        torch.backends.cudnn.benchmark = False
        torch.backends.cudnn.deterministic = True
    return True


def set_seed(seed: int | None = None, *, deterministic: bool = False) -> int:
    """同时设置 Python / numpy / torch 的全局随机种子。

    Args:
        seed: 种子值；``None`` 时使用环境变量 ``TTM_SEED`` 或 :data:`DEFAULT_SEED`。
        deterministic: 是否开启 torch 确定性算法（``cudnn`` 关闭 benchmark）。

    Returns:
        实际生效的种子值（便于日志记录）。
    """
    if seed is None:
        raw = os.environ.get("TTM_SEED", "").strip()
        seed = int(raw) if raw else DEFAULT_SEED
    seed = int(seed)

    os.environ["PYTHONHASHSEED"] = str(seed)
    random.seed(seed)
    np.random.seed(seed % (2**32))
    _seed_torch(seed, deterministic)
    return seed


def set_torch_deterministic(enabled: bool = True) -> None:
    """单独开关 torch 确定性算法（不改变种子）。

    Args:
        enabled: ``True`` 开启确定性算法与 ``cudnn.deterministic``。
    """
    try:
        import torch  # 延迟导入
    except Exception:  # pragma: no cover
        return
    try:
        torch.use_deterministic_algorithms(enabled, warn_only=True)
    except Exception:  # pragma: no cover
        pass
    torch.backends.cudnn.benchmark = not enabled
    torch.backends.cudnn.deterministic = enabled


def numpy_rng(seed: int | None = None) -> np.random.Generator:
    """创建独立的 numpy 随机数发生器（不修改全局 ``np.random`` 状态）。

    Args:
        seed: 种子；``None`` 时使用 :data:`DEFAULT_SEED`。

    Returns:
        :class:`numpy.random.Generator`（PCG64）。
    """
    effective = DEFAULT_SEED if seed is None else int(seed)
    return np.random.default_rng(effective % (2**32))


def torch_generator(seed: int | None = None, *, device: str = "cpu") -> Any:
    """创建 ``torch.Generator``；torch 未安装时返回 ``None``。

    Args:
        seed: 种子；``None`` 时使用 :data:`DEFAULT_SEED`。
        device: ``"cpu"`` 或 ``"cuda"``。

    Returns:
        ``torch.Generator`` 实例，或 ``None``（无 torch）。
    """
    try:
        import torch  # 延迟导入
    except Exception:  # pragma: no cover - 无 torch 环境
        return None
    effective = DEFAULT_SEED if seed is None else int(seed)
    generator = torch.Generator(device=device)
    generator.manual_seed(effective)
    return generator


def derived_seed(base: int, *index: int) -> int:
    """由基础种子与若干索引派生稳定子种子（长动作多窗口采样用）。

    使用简单的线性同余混合，保证跨平台一致、且相邻窗口种子不同。

    Args:
        base: 基础种子。
        *index: 索引（如窗口号）。

    Returns:
        派生后的种子（非负、落在 32 位范围内）。
    """
    value = int(base) % (2**32)
    for item in index:
        value = (value * 1_000_003 + int(item) * 2_654_435_761 + 12_345) % (2**32)
    return value
