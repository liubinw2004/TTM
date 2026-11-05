"""TTM 推理服务包（T10）。

``uvicorn server.app:app`` 或 ``python server/app.py`` 启动；
具体路由与模型定义见 :mod:`server.app`。
"""

from __future__ import annotations

__all__ = ["app"]


def __getattr__(name: str):  # noqa: ANN202 - 延迟导入，避免仅导入包时就加载权重
    """按需暴露 ``app``（避免 ``import server`` 触发配置/权重加载）。"""
    if name == "app":
        from server.app import app as _app  # noqa: PLC0415

        return _app
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
