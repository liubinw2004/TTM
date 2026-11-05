"""统一日志：控制台 + ``runs/logs/{script}.log`` 双写。

约定（TTM-ARCH.md §5.1「日志」）：

* 控制台输出 INFO 及以上，格式紧凑；
* 文件输出含 ``funcName:lineno``，便于定位；
* 同名 logger 只装配一次 handler，避免重复行；
* 日志文件路径强制经 :func:`ttm.common.paths.resolve_out` 校验写盘白名单。
"""

from __future__ import annotations

import logging
import os
import sys
from pathlib import Path
from typing import Final

from ttm.common import paths

__all__ = [
    "DEFAULT_LOGGER_NAME",
    "close_all",
    "get_logger",
    "log_file_path",
]

#: 未显式指定名字时使用的 logger 名。
DEFAULT_LOGGER_NAME: Final[str] = "ttm"

#: 控制台格式：时间 | 级别 | name | 消息。
_CONSOLE_FORMAT: Final[str] = "%(asctime)s | %(levelname)-7s | %(name)s | %(message)s"

#: 文件格式：额外带函数名与行号。
_FILE_FORMAT: Final[str] = (
    "%(asctime)s | %(levelname)-7s | %(name)s | %(funcName)s:%(lineno)d | %(message)s"
)

#: 日期格式（秒级即可，训练日志量大时避免毫秒噪音）。
_DATE_FORMAT: Final[str] = "%Y-%m-%d %H:%M:%S"

#: 已装配过的 logger 名，用于幂等保护。
_CONFIGURED: set[str] = set()


def log_file_path(script: str) -> Path:
    """返回脚本对应的日志文件路径 ``<root>/runs/logs/{script}.log``。

    Args:
        script: 脚本名（如 ``"01_scan"``），非法字符会被替换为下划线。
    """
    safe = "".join(ch if (ch.isalnum() or ch in "._-") else "_" for ch in str(script))
    return paths.logs_dir() / f"{safe or 'ttm'}.log"


def _ensure_console_handler(logger: logging.Logger) -> None:
    """给 logger 装配（一次）控制台 handler。"""
    for handler in logger.handlers:
        if isinstance(handler, logging.StreamHandler) and not isinstance(
            handler, logging.FileHandler
        ):
            return
    console = logging.StreamHandler(stream=sys.stdout)
    console.setLevel(logging.INFO)
    console.setFormatter(logging.Formatter(_CONSOLE_FORMAT, datefmt=_DATE_FORMAT))
    logger.addHandler(console)


def _ensure_file_handler(logger: logging.Logger, script: str) -> None:
    """给 logger 装配（一次）文件 handler，路径经写盘守卫校验。"""
    for handler in logger.handlers:
        if isinstance(handler, logging.FileHandler):
            return
    try:
        target = paths.resolve_out(log_file_path(script), mkparents=True)
    except paths.WriteNotAllowedError:
        # 白名单行为异常时降级为纯控制台日志，绝不因日志导致脚本崩溃。
        return
    file_handler = logging.FileHandler(target, mode="a", encoding="utf-8")
    file_handler.setLevel(logging.DEBUG)
    file_handler.setFormatter(logging.Formatter(_FILE_FORMAT, datefmt=_DATE_FORMAT))
    logger.addHandler(file_handler)


def get_logger(
    name: str = DEFAULT_LOGGER_NAME,
    *,
    script: str | None = None,
    level: int | None = None,
) -> logging.Logger:
    """获取（并首次装配）一个双写 logger。

    Args:
        name: logger 名；建议直接用脚本/模块名，如 ``"ttm.scan"``。
        script: 日志文件名（不含扩展名）；默认取 ``name`` 的最后一段。
        level: logger 级别；默认读环境变量 ``TTM_LOG_LEVEL``，未设置则 INFO。

    Returns:
        已装配 handler 的 :class:`logging.Logger`。
    """
    logger = logging.getLogger(name)
    logger.setLevel(_env_level() if level is None else level)
    logger.propagate = False
    if name not in _CONFIGURED:
        _ensure_console_handler(logger)
        _ensure_file_handler(logger, script or name.rsplit(".", 1)[-1] or DEFAULT_LOGGER_NAME)
        _CONFIGURED.add(name)
    return logger


def close_all() -> None:
    """关闭并移除所有本模块装配过的 handler（测试收尾/进程退出用）。"""
    for name in list(_CONFIGURED):
        logger = logging.getLogger(name)
        for handler in list(logger.handlers):
            try:
                handler.flush()
            finally:
                handler.close()
                logger.removeHandler(handler)
    _CONFIGURED.clear()


def _install_excepthook(logger: logging.Logger) -> None:
    """把未捕获异常也写进日志文件（可用但非必需，供脚本调用）。"""
    previous = sys.excepthook

    def _hook(exc_type, exc_value, exc_tb):  # type: ignore[no-untyped-def]
        if issubclass(exc_type, KeyboardInterrupt):
            previous(exc_type, exc_value, exc_tb)
            return
        logger.error("未捕获异常", exc_info=(exc_type, exc_value, exc_tb))
        previous(exc_type, exc_value, exc_tb)

    sys.excepthook = _hook


def _env_level() -> int:
    """从环境变量 ``TTM_LOG_LEVEL`` 读取日志级别（默认 INFO）。"""
    raw = os.environ.get("TTM_LOG_LEVEL", "").strip().upper()
    return getattr(logging, raw, logging.INFO) if raw else logging.INFO
