"""路径解析与写盘守卫。

硬约束（TTM-PRD.md §8「写入范围」）：只允许写 ``D:/TTM``；``Z:`` 严格只读；
``C:`` 零写入。

本模块是该约束的**唯一**执行点：

* :func:`assert_write_allowed` — 校验单个路径是否在写盘白名单内，越界抛
  :class:`WriteNotAllowedError`。
* :func:`resolve_out` — 所有“要写出文件”的代码点统一入口：解析相对路径、
  校验白名单、按需创建父目录。

模块同时集中管理 ``runs/`` ``data/`` ``.cache/`` 等产物目录的派生规则，
避免任何脚本硬编码路径。
"""

from __future__ import annotations

import os
from pathlib import Path
from typing import Final

__all__ = [
    "CACHE_ENV_VARS",
    "WriteNotAllowedError",
    "assert_write_allowed",
    "cache_dir",
    "data_dir",
    "env_report",
    "logs_dir",
    "norm_stats_path",
    "project_root",
    "rel_to_root",
    "resolve_out",
    "runs_dir",
    "safe_makedirs",
    "shards_dir",
    "tier_data_dir",
    "tier_filename",
]

#: 允许写盘的根目录环境变量名（默认为 ``D:/TTM``）。
ENV_ROOT: Final[str] = "TTM_ROOT"

#: 默认项目根目录。项目内所有路径都相对它派生。
DEFAULT_PROJECT_ROOT: Final[str] = "D:/TTM"

#: 必须重定向到 ``<root>/.cache`` 的环境变量清单（TTM-ARCH.md §5.1）。
CACHE_ENV_VARS: Final[tuple[str, ...]] = (
    "PIP_CACHE_DIR",
    "HF_HOME",
    "TORCH_HOME",
    "TMPDIR",
    "TEMP",
    "TMP",
)

#: 禁止写入的盘符（小写、无冒号）。``Z:`` 是只读网络盘，``C:`` 是系统盘。
FORBIDDEN_DRIVES: Final[frozenset[str]] = frozenset({"c", "z"})


class WriteNotAllowedError(PermissionError):
    """尝试写入写盘白名单之外的路径时抛出。

    Attributes:
        path: 被拒绝的原始路径字符串。
        root: 当前生效的白名单根目录。
    """

    def __init__(self, path: str | os.PathLike[str], root: Path) -> None:
        self.path = str(path)
        self.root = root
        super().__init__(
            f"写盘被拒绝: {self.path!r} 不在白名单 {str(root)!r} 之内。"
            f"TTM 只允许写入 {DEFAULT_PROJECT_ROOT}（C: 零写入 / Z: 只读）。"
        )


def project_root() -> Path:
    """返回当前生效的项目根目录（写盘白名单根）。

    优先读取环境变量 ``TTM_ROOT``（由 ``scripts/env.ps1`` / ``env.sh`` 注入），
    否则使用 :data:`DEFAULT_PROJECT_ROOT`。

    Returns:
        归一化后的绝对路径（不做 realpath，避免网络盘/软链耗时）。
    """
    raw = os.environ.get(ENV_ROOT, "").strip()
    root = raw if raw else DEFAULT_PROJECT_ROOT
    return Path(os.path.abspath(os.path.expanduser(root)))


def _normcase(p: str | os.PathLike[str]) -> str:
    """把路径归一化为可比较的小写绝对路径字符串（Windows 大小写不敏感）。"""
    absolute = os.path.abspath(os.path.expanduser(str(p)))
    return os.path.normcase(os.path.normpath(absolute))


def _drive_of(p: str | os.PathLike[str]) -> str:
    """返回路径的盘符（小写、无冒号）；无盘符时返回空串。"""
    drive = os.path.splitdrive(os.path.abspath(str(p)))[0]
    return drive.replace(":", "").lower()


def rel_to_root(p: str | os.PathLike[str]) -> str:
    """把绝对路径转成相对项目根的 POSIX 风格字符串（用于日志与 JSON 记录）。

    Args:
        p: 任意路径。

    Returns:
        相对项目根的路径字符串（如 ``data/tier_S/index.jsonl``）；若不在项目
        根之内，则原样返回绝对路径。
    """
    root = project_root()
    absolute = os.path.abspath(os.path.expanduser(str(p)))
    try:
        return Path(os.path.relpath(absolute, root)).as_posix()
    except ValueError:
        # Windows 下跨盘符无法 relpath。
        return Path(absolute).as_posix()


def assert_write_allowed(path: str | os.PathLike[str]) -> Path:
    """校验 ``path`` 是否允许写入，返回其绝对路径。

    Args:
        path: 待写入的文件或目录路径（可为相对路径，相对项目根解析）。

    Returns:
        归一化后的 :class:`pathlib.Path` 绝对路径。

    Raises:
        WriteNotAllowedError: 路径不在项目根之内，或落在 ``C:`` / ``Z:`` 盘。
    """
    root = project_root()
    raw = str(path)
    if not raw.strip():
        raise WriteNotAllowedError(raw, root)

    candidate = os.path.abspath(os.path.expanduser(raw))
    norm_candidate = _normcase(candidate)
    norm_root = _normcase(root)

    # 1) 盘符黑名单先行，给出更精确的错误语义。
    drive = _drive_of(candidate)
    if drive in FORBIDDEN_DRIVES:
        raise WriteNotAllowedError(candidate, root)

    # 2) 必须落在白名单根之内（相等或作为子路径）。
    if norm_candidate == norm_root:
        return Path(candidate)
    prefix = norm_root + os.sep
    if not norm_candidate.startswith(prefix):
        raise WriteNotAllowedError(candidate, root)
    return Path(candidate)


def safe_makedirs(path: str | os.PathLike[str]) -> Path:
    """创建目录（递归），并强制经过写盘守卫。

    Args:
        path: 待创建的目录路径。

    Returns:
        创建后的目录绝对路径。
    """
    target = assert_write_allowed(path)
    target.mkdir(parents=True, exist_ok=True)
    return target


def resolve_out(path: str | os.PathLike[str], *, mkparents: bool = True) -> Path:
    """所有写文件代码点的统一入口。

    相对路径按项目根解析为绝对路径；随后经 :func:`assert_write_allowed`
    校验；``mkparents=True`` 时自动创建父目录。

    Args:
        path: 目标路径，可相对项目根。
        mkparents: 是否自动创建父目录（默认 True）。

    Returns:
        可直接用于 ``open(..., "wb")`` 的绝对路径。

    Raises:
        WriteNotAllowedError: 路径越界。
    """
    root = project_root()
    raw = str(path)
    expanded = os.path.expanduser(raw)
    candidate = expanded if os.path.isabs(expanded) else os.path.join(str(root), expanded)
    target = assert_write_allowed(candidate)
    if mkparents:
        target.parent.mkdir(parents=True, exist_ok=True)
    return target


def cache_dir() -> Path:
    """返回 ``<root>/.cache``（pip/HF/torch/tmp 全部重定向基准）。"""
    return project_root() / ".cache"


def runs_dir() -> Path:
    """返回 ``<root>/runs``（ckpt/logs/samples 运行期产物根）。"""
    return project_root() / "runs"


def data_dir() -> Path:
    """返回 ``<root>/data``（预处理产物根）。"""
    return project_root() / "data"


def logs_dir() -> Path:
    """返回 ``<root>/runs/logs``（脚本日志目录）。"""
    return runs_dir() / "logs"


def tier_data_dir(
    tier: str, base_data_dir: str | os.PathLike[str] | None = None
) -> Path:
    """返回某档位的数据产物目录 ``<data>/{tier_filename(tier)}``。

    Args:
        tier: ``S`` / ``M`` / ``L``（大小写不敏感）。
        base_data_dir: 数据根目录；``None`` 时用 :func:`data_dir`。
            允许配置里覆盖（如 ``data.unit_test`` / ``data.smoke``）。

    Returns:
        档位产物目录。
    """
    base = data_dir() if base_data_dir is None else Path(base_data_dir)
    return base / tier_filename(tier)


def tier_filename(tier: str) -> str:
    """返回档位对应的文件名后缀形式 ``tier_S``（用于 index/标签文件命名）。"""
    return f"tier_{str(tier).strip().upper()}"


def shards_dir(tier: str, base_data_dir: str | os.PathLike[str] | None = None) -> Path:
    """返回某档位的 npz 分片目录 ``.../tier_{tier}/shards``。"""
    return tier_data_dir(tier, base_data_dir) / "shards"


def norm_stats_path(tier: str, base_data_dir: str | os.PathLike[str] | None = None) -> Path:
    """返回某档位的归一化统计文件路径 ``.../tier_{tier}/norm_stats.npz``。"""
    return tier_data_dir(tier, base_data_dir) / "norm_stats.npz"


def env_report() -> dict[str, str]:
    """返回当前环境变量重定向状况（供 ``00_env_check.py`` 打印/断言）。

    Returns:
        形如 ``{"PIP_CACHE_DIR": "D:/TTM/.cache/pip", ...}`` 的字典；
        未设置的变量值为空串。
    """
    report: dict[str, str] = {}
    for name in CACHE_ENV_VARS:
        report[name] = os.environ.get(name, "").strip()
    report[ENV_ROOT] = os.environ.get(ENV_ROOT, "").strip()
    return report
