"""配置契约：yaml 键 / dataclass 字段 / ``_parse_*`` 消费三方必须对齐。

**为什么需要这个文件**（M4 返工的真实事故）：``configs/tier_s.yaml`` 加了
``runtime.trim_head_frames: 12``、``RuntimeConfig`` 也加了字段，但
``_parse_runtime()`` 没有 ``sec.get(...)`` 它。``_Section.finish()`` 按「parser
实际消费过的键」判未知字段，于是 ``load_config`` 直接抛 ``ConfigError``——而
``load_config`` 是**所有入口的第一步**，server / 02~08 脚本全部起不来。

这类 bug 靠人眼抓不可靠（三处只差一处，diff 上看不出来），所以固化成断言：

1. ``configs/*.yaml`` 每个文件都必须能被 ``load_config`` 加载；
2. yaml 里出现的每个键，都必须是所属 dataclass 的字段（防拼写错误）；
3. **每个 dataclass 字段都必须被 parser 消费**——把全部字段写成 yaml 再加载，
   漏读的字段会被 ``finish()`` 当成未知键抓出来（本文件的核心用例）。
"""

from __future__ import annotations

import dataclasses
import typing
from pathlib import Path

import pytest

from ttm.common.config import TTMConfig, load_config

PROJECT_ROOT = Path(__file__).resolve().parents[1]
CONFIG_DIR = PROJECT_ROOT / "configs"

#: 配置小节名 → dataclass（``TTMConfig`` 的字段类型）。
SECTION_TYPES: dict[str, type] = {
    name: hint
    for name, hint in typing.get_type_hints(TTMConfig).items()
    if dataclasses.is_dataclass(hint)
}


def _config_files() -> list[Path]:
    """``configs/`` 下全部 yaml（按名字排序，报错时顺序稳定）。"""
    return sorted(CONFIG_DIR.glob("*.yaml"))


def _raw_sections(path: Path) -> dict[str, dict]:
    """读出 yaml 的 ``{小节: {键: 值}}``（只关心映射型小节）。"""
    import yaml  # noqa: PLC0415 - 仅测试需要

    raw = yaml.safe_load(path.read_text(encoding="utf-8")) or {}
    return {
        name: value
        for name, value in raw.items()
        if name in SECTION_TYPES and isinstance(value, dict)
    }


def test_config_dir_is_not_empty() -> None:
    """没有 yaml 可测说明路径算错了，不能让本文件静默空跑。"""
    assert _config_files(), f"{CONFIG_DIR} 下没有 yaml"


@pytest.mark.parametrize("path", _config_files(), ids=lambda item: item.name)
def test_every_config_is_loadable(path: Path) -> None:
    """每个 yaml 都必须能被 ``load_config`` 加载（漏接 parser 会在这里炸）。"""
    cfg = load_config(path)
    assert isinstance(cfg, TTMConfig)
    assert cfg.source_path


@pytest.mark.parametrize("path", _config_files(), ids=lambda item: item.name)
def test_yaml_keys_are_dataclass_fields(path: Path) -> None:
    """yaml 里的每个键都必须是所属小节 dataclass 的字段（防拼写错误）。"""
    problems: list[str] = []
    for section, values in _raw_sections(path).items():
        allowed = {item.name for item in dataclasses.fields(SECTION_TYPES[section])}
        for key in values:
            if key not in allowed:
                problems.append(f"{section}.{key}（{SECTION_TYPES[section].__name__} 无此字段）")
    assert not problems, f"{path.name} 存在未知键: {problems}"


def _base_payload() -> dict[str, dict]:
    """以 ``configs/tier_s.yaml`` 为底（含必需小节），把所有 dataclass 字段补全。

    直接用 dataclass 默认值造 yaml 会缺 ``data`` 这种必需小节，所以先读真实配置
    再补全——这样既覆盖「所有字段」又保持可加载。
    """
    import yaml  # noqa: PLC0415 - 仅测试需要

    base: dict[str, dict] = dict(
        yaml.safe_load((CONFIG_DIR / "tier_s.yaml").read_text(encoding="utf-8")) or {}
    )
    for section, cls in SECTION_TYPES.items():
        current = dict(base.get(section) or {})
        for item in dataclasses.fields(cls):
            current.setdefault(item.name, _yaml_safe(getattr(cls(), item.name)))
        base[section] = current
    return base


def _write_yaml(path: Path, payload: dict[str, dict]) -> Path:
    """写 yaml（用 ``yaml.safe_dump`` 而不是 json——PyYAML 解析 ``1e-05`` 时会当字符串）。"""
    import yaml  # noqa: PLC0415 - 仅测试需要

    path.write_text(
        yaml.safe_dump(payload, allow_unicode=True, sort_keys=False), encoding="utf-8"
    )
    return path


def test_every_dataclass_field_is_consumed_by_parser(work_dir: Path) -> None:
    """把全部 dataclass 字段写成 yaml 再加载——任何一个 parser 漏读都会报未知键。

    这是本文件的核心用例：它精确复现并拦住「yaml 有 → dataclass 有 → parser 没读」
    这一类事故（漏 ``sec.get`` 时 ``_Section.finish()`` 会抛出 ``ConfigError``）。
    后半段再逐字段核对「写进去的值确实读出来了」，防止 ``sec.get`` 读错了键。
    """
    payload = _base_payload()
    target = _write_yaml(work_dir / "all_fields.yaml", payload)

    cfg = load_config(target)  # 漏读字段 → ConfigError（用例失败点）

    problems: list[str] = []
    for section, cls in SECTION_TYPES.items():
        parsed = getattr(cfg, section)
        for item in dataclasses.fields(cls):
            expected = payload[section][item.name]
            actual = getattr(parsed, item.name)
            if _norm(actual) != _norm(expected):
                problems.append(f"{section}.{item.name}: 写入 {expected!r} → 读出 {actual!r}")
    assert not problems, f"字段没被正确解析: {problems}"


def _norm(value: object) -> object:
    """比对用的归一化（列表→元组、字符串统一小写去空格）。"""
    if isinstance(value, (list, tuple)):
        return tuple(_norm(item) for item in value)
    if isinstance(value, str):
        return value.strip().lower()
    return value


def _yaml_safe(value: object) -> object:
    """把 dataclass 默认值转成 yaml/json 可写形式。"""
    if isinstance(value, tuple):
        return list(value)
    if isinstance(value, Path):
        return str(value)
    return value


def _with_runtime(**overrides: object) -> dict[str, dict]:
    """在完整 payload 上覆盖 ``runtime`` 的若干键。"""
    payload = _base_payload()
    payload["runtime"] = {**payload["runtime"], **overrides}
    return payload


def test_trim_head_frames_roundtrip(work_dir: Path) -> None:
    """``runtime.trim_head_frames`` 必须真的从 yaml 读进来（不是只用 dataclass 默认值）。"""
    target = _write_yaml(work_dir / "trim.yaml", _with_runtime(trim_head_frames=7))
    cfg = load_config(target)
    assert cfg.runtime.trim_head_frames == 7


def test_trim_head_frames_bounds(work_dir: Path) -> None:
    """超过一个窗口（150 帧）的裁头是配置错误，不能静默接受。"""
    from ttm.common.config import ConfigError  # noqa: PLC0415

    target = _write_yaml(work_dir / "trim_bad.yaml", _with_runtime(trim_head_frames=999))
    with pytest.raises(ConfigError):
        load_config(target)
