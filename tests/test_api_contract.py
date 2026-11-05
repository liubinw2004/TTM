"""接口契约：``guidance`` 配置回落 + prompt 长度软拦截（QA 必修 3 / 建议 4）。

背景：``GenerateRequest.guidance`` 曾经默认写死 ``3.0``，而生产配置是 ``7.5``——
不显式传参的调用方（QA 脚本、前端）会**静默**用低引导强度，画质下降却查不出来。
现在默认 ``None``，由运行时回落到 ``cfg.runtime.guidance``，并在响应里回显实际值。

prompt 长度：``max_length=200`` 是防滥用的硬上限；40~200 字区间只回
``warnings``，不拒绝请求（训练集指令都很短，长指令效果会掉但不是错误）。
"""

from __future__ import annotations

import pytest

from server.app import PROMPT_SOFT_LIMIT, GenerateRequest, GenerateResponse, prompt_warnings


def test_guidance_defaults_to_none() -> None:
    """未显式传 guidance 时是 ``None``（交给运行时回落），不是写死的常量。"""
    request = GenerateRequest(prompt="挥手")
    assert request.guidance is None


def test_guidance_accepts_explicit_value() -> None:
    """显式传值优先（调用方想自己控 CFG 时必须能覆盖配置）。"""
    request = GenerateRequest(prompt="挥手", guidance=2.5)
    assert request.guidance == 2.5


def test_response_echoes_guidance_and_warnings() -> None:
    """响应必须带 ``guidance``（实际使用值）与 ``warnings`` 两个字段。"""
    fields = GenerateResponse.model_fields
    assert "guidance" in fields
    assert "warnings" in fields
    assert fields["warnings"].default_factory is list


def test_prompt_hard_limit() -> None:
    """超过 200 字的 prompt 直接拒绝（防滥用），边界 200 仍接受。"""
    GenerateRequest(prompt="字" * 200)
    with pytest.raises(Exception):  # noqa: B017 - pydantic 校验错误类型随版本变化
        GenerateRequest(prompt="字" * 201)


def test_prompt_warnings_quiet_for_short_prompt() -> None:
    """训练分布内的短指令不产生噪音警告。"""
    assert prompt_warnings("慢慢走上楼梯") == []
    assert prompt_warnings("字" * PROMPT_SOFT_LIMIT) == []


def test_prompt_warnings_soft_for_long_prompt() -> None:
    """刚过软上限就警告，且提示里带上实际字数。"""
    long_prompt = "字" * (PROMPT_SOFT_LIMIT + 1)
    warnings = prompt_warnings(long_prompt)
    assert len(warnings) == 1
    assert "效果可能下降" in warnings[0]
    assert str(PROMPT_SOFT_LIMIT + 1) in warnings[0]
