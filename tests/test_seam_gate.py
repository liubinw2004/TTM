"""接缝连续性门禁的单测（M3 硬性要求：滑窗融合后接缝根速度必须连续）。

这些用例**不加载任何真实模型**：``verify_seams`` 支持注入生成器，所以用鸭子类型
的假生成器就能覆盖门禁的全部判定分支。真正需要 GPU + 训练产物的那一层留给
``scripts/05_generate.py`` 与训练收尾阶段跑。

覆盖重点（每条都对应一个「会静默放过问题」的坑）：

1. 全连续 → 通过；
2. 有一个接缝越界 → 不通过（且报告里能看出是哪条指令、多少 cm/s）；
3. **窗口数不足、没有接缝可验** → 不通过（``max(空数组) = 0`` 会伪装成「完美」）；
4. 单条指令抛异常 → 不通过（不能因为「其他两条过了」就放过）；
5. 报告落盘 JSON 的字段完整（``passed`` / ``worst`` / ``failures`` / ``probes``）；
6. ``--verify-seam=auto`` 的三态解析；
7. 默认时长必须撑出 ≥2 个窗口（防止有人把 ``DEFAULT_SEAM_DURATION_S`` 调小到
   6 秒以下，让门禁退化成空洞通过）。
"""

from __future__ import annotations

import json
import shutil
from pathlib import Path
from types import SimpleNamespace
from typing import Any

import numpy as np
import pytest

from ttm.common import constants as C
from ttm.infer.generate import SEAM_JUMP_MAX_CMPS
from ttm.infer.sampling import SamplingConfig
from ttm.infer.seam_gate import (
    DEFAULT_SEAM_DURATION_S,
    SEAM_GATE_PROMPTS,
    SeamGateReport,
    SeamProbe,
    verify_seams,
)

pytestmark = pytest.mark.requires_torch

#: 本模块独占的产物目录（与其他测试模块隔离）。
_RUNS_DIR = "runs/unit_test_seam_gate"


class _FakeMotion:
    """假生成结果：只带门禁读的几个字段。"""

    def __init__(
        self,
        *,
        n_windows: int = 2,
        n_frames: int = 240,
        jumps: np.ndarray | None = None,
        total_ms: float = 1400.0,
    ) -> None:
        self.n_windows = int(n_windows)
        self.n_frames = int(n_frames)
        self.seam_jumps = (
            np.asarray(jumps, dtype=np.float64)
            if jumps is not None
            else np.asarray([12.5], dtype=np.float64)
        )
        self.timing_ms = {"total_ms": float(total_ms)}


class _FakeGenerator:
    """假生成器：``generate`` 返回预设结果或抛预设异常。"""

    def __init__(
        self,
        *,
        jumps_by_prompt: dict[str, np.ndarray] | None = None,
        n_windows: int = 2,
        n_frames: int = 240,
        errors: dict[str, Exception] | None = None,
    ) -> None:
        self.jumps_by_prompt = jumps_by_prompt or {}
        self.n_windows = int(n_windows)
        self.n_frames = int(n_frames)
        self.errors = errors or {}
        self.calls: list[str] = []

    def generate(self, prompt: str, options: Any = None) -> _FakeMotion:  # noqa: ANN401
        """返回预设结果（或抛预设异常）。"""
        self.calls.append(str(prompt))
        if str(prompt) in self.errors:
            raise self.errors[str(prompt)]
        jumps = self.jumps_by_prompt.get(str(prompt))
        return _FakeMotion(
            n_windows=self.n_windows,
            n_frames=self.n_frames,
            jumps=jumps,
        )


def _cfg() -> SimpleNamespace:
    """最小配置替身（只在 ``steps`` / ``guidance`` / ``seed`` 为 ``None`` 时被读）。"""
    return SimpleNamespace(
        diffusion=SimpleNamespace(ddim_steps=50),
        runtime=SimpleNamespace(guidance=3.0, seed=42),
    )


def _cleanup() -> None:
    """删掉本模块的产物目录。"""
    target = Path(__file__).resolve().parents[1] / _RUNS_DIR
    if target.is_dir():
        shutil.rmtree(target, ignore_errors=True)


# ---------------------------------------------------------------------------
# 1) 全连续 → 通过
# ---------------------------------------------------------------------------
def test_gate_passes_when_all_probes_continuous() -> None:
    """三条指令接缝都远低于阈值时，门禁应判通过。"""
    generator = _FakeGenerator(
        jumps_by_prompt={prompt: np.asarray([8.0, 11.2]) for prompt in SEAM_GATE_PROMPTS}
    )
    report = verify_seams(_cfg(), generator=generator, steps=50, guidance=3.0, seed=42)

    assert report.passed is True
    assert report.failures() == []
    assert len(report.probes) == len(SEAM_GATE_PROMPTS)
    for probe in report.probes:
        assert probe.n_seams == 2
        assert probe.error == ""
        assert probe.ok is True
    # worst 取的是最大值那条
    assert report.worst is not None
    assert report.worst.seam_max_cmps == pytest.approx(11.2)


# ---------------------------------------------------------------------------
# 2) 一个接缝越界 → 不通过
# ---------------------------------------------------------------------------
def test_gate_fails_on_jump_over_threshold() -> None:
    """任一接缝越过 100 cm/s 就必须判失败，并在失败原因里给出具体数字。

    369.6 cm/s 是真实观测过的值（VAE-only smoke ckpt 上的 8 秒双窗动作），
    用它做回归用例，比造一个整数更有意义。
    """
    jumps = {
        SEAM_GATE_PROMPTS[0]: np.asarray([5.0]),
        SEAM_GATE_PROMPTS[1]: np.asarray([369.6]),
        SEAM_GATE_PROMPTS[2]: np.asarray([9.0]),
    }
    report = verify_seams(_cfg(), generator=_FakeGenerator(jumps_by_prompt=jumps))

    assert report.passed is False
    reasons = report.failures()
    assert len(reasons) == 1
    assert SEAM_GATE_PROMPTS[1] in reasons[0]
    assert "369.6" in reasons[0]
    assert report.worst is not None
    assert report.worst.prompt == SEAM_GATE_PROMPTS[1]


# ---------------------------------------------------------------------------
# 3) 没有接缝 → 不通过（空洞通过不算通过）
# ---------------------------------------------------------------------------
def test_gate_fails_when_no_seam_exists() -> None:
    """只有 1 个窗口时接缝数恒为 0，``max(空) = 0`` 会伪装成完美，必须显式判失败。"""
    generator = _FakeGenerator(
        n_windows=1,
        n_frames=150,
        jumps_by_prompt={
            prompt: np.asarray([], dtype=np.float64) for prompt in SEAM_GATE_PROMPTS
        },
    )
    report = verify_seams(_cfg(), generator=generator)

    assert report.passed is False
    reasons = report.failures()
    assert len(reasons) == len(SEAM_GATE_PROMPTS)
    assert all("没有接缝可验" in reason for reason in reasons)


# ---------------------------------------------------------------------------
# 4) 单条指令抛异常 → 不通过
# ---------------------------------------------------------------------------
def test_gate_catches_per_prompt_exception() -> None:
    """一条炸了不能拖垮整批，但整批必须判失败。"""
    generator = _FakeGenerator(
        jumps_by_prompt={prompt: np.asarray([7.0]) for prompt in SEAM_GATE_PROMPTS},
        errors={SEAM_GATE_PROMPTS[2]: RuntimeError("CUDA out of memory")},
    )
    report = verify_seams(_cfg(), generator=generator)

    assert report.passed is False
    # 其他两条仍然有结果
    assert len(report.probes) == len(SEAM_GATE_PROMPTS)
    failed = [probe for probe in report.probes if probe.error]
    assert len(failed) == 1
    assert "CUDA out of memory" in failed[0].error
    reasons = report.failures()
    assert any("生成失败" in reason for reason in reasons)


# ---------------------------------------------------------------------------
# 5) 报告落盘
# ---------------------------------------------------------------------------
def test_gate_writes_json_report() -> None:
    """``write_to`` 给出的路径应写出完整字段的 JSON。"""
    _cleanup()
    try:
        generator = _FakeGenerator(
            jumps_by_prompt={prompt: np.asarray([13.0]) for prompt in SEAM_GATE_PROMPTS}
        )
        report = verify_seams(
            _cfg(),
            generator=generator,
            write_to=f"{_RUNS_DIR}/seam_gate.json",
        )
        assert report.passed is True

        target = Path(__file__).resolve().parents[1] / _RUNS_DIR / "seam_gate.json"
        assert target.is_file(), f"报告未写出: {target}"
        payload = json.loads(target.read_text(encoding="utf-8"))

        assert payload["passed"] is True
        assert payload["threshold_cmps"] == pytest.approx(SEAM_JUMP_MAX_CMPS)
        assert payload["duration_s"] == pytest.approx(DEFAULT_SEAM_DURATION_S)
        assert payload["failures"] == []
        assert len(payload["probes"]) == len(SEAM_GATE_PROMPTS)
        assert payload["worst"]["seam_max_cmps"] == pytest.approx(13.0)
        for probe in payload["probes"]:
            for key in (
                "prompt",
                "n_frames",
                "n_windows",
                "n_seams",
                "seam_max_cmps",
                "seam_mean_cmps",
                "ok",
                "error",
            ):
                assert key in probe, f"探测结果缺字段 {key}"
    finally:
        _cleanup()


# ---------------------------------------------------------------------------
# 6) --verify-seam 三态解析
# ---------------------------------------------------------------------------
def test_should_verify_seam_modes() -> None:
    """``auto`` 在冒烟（``--max-steps > 0``）时跳过、正式训练时必跑。"""
    module = _load_script_module()
    should = module._should_verify_seam

    assert should("on", 5) is True
    assert should("on", 0) is True
    assert should("off", 0) is False
    assert should("off", 5) is False
    # auto：冒烟跳过、正式跑
    assert should("auto", 5) is False
    assert should("auto", 0) is True


def _load_script_module():  # noqa: ANN202 - 测试辅助
    """把 ``scripts/04_train_diffusion.py`` 作为模块导入（它不是包的一部分）。"""
    import importlib.util
    import sys

    if "ttm_script_04" in sys.modules:
        return sys.modules["ttm_script_04"]
    path = Path(__file__).resolve().parents[1] / "scripts" / "04_train_diffusion.py"
    spec = importlib.util.spec_from_file_location("ttm_script_04", path)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    sys.modules["ttm_script_04"] = module
    spec.loader.exec_module(module)
    return module


# ---------------------------------------------------------------------------
# 7) 默认时长必须撑出 ≥2 个窗口
# ---------------------------------------------------------------------------
def test_default_duration_yields_at_least_one_real_seam() -> None:
    """默认 8 秒必须能撑出 2 个窗口，否则门禁退化成空洞通过。

    这条是给未来改常量的人留的护栏：``DEFAULT_SEAM_DURATION_S`` 一旦调到 6 秒
    以下（150 帧窗口 + 120 帧步长 = 270 帧 = 9 秒... 实际 8 秒=240 帧仍是 2 窗），
    窗口数会掉到 1，接缝数恒为 0，测试会立刻失败。
    """
    sampling = SamplingConfig()
    n_frames = int(round(DEFAULT_SEAM_DURATION_S * C.FPS))
    n_windows = sampling.windows_for_frames(n_frames)

    assert n_windows >= 2, (
        f"默认时长 {DEFAULT_SEAM_DURATION_S}s = {n_frames} 帧只够 {n_windows} 个窗口"
        "（窗口 150 / 步长 120），没有接缝可验"
    )
    # 窗口数 - 1 就是接缝数
    assert n_windows - 1 >= 1


# ---------------------------------------------------------------------------
# 8) 报告的边界语义
# ---------------------------------------------------------------------------
def test_empty_report_is_not_passed() -> None:
    """一条都没探测成功时不能判通过（``all([]) == True`` 的经典坑）。"""
    report = SeamGateReport()
    assert report.passed is False
    assert report.worst is None
    assert report.failures() == ["一条指令都没探测成功"]


def test_probe_serialization_rounds_numbers() -> None:
    """``SeamProbe.to_dict`` 应把浮点压到合理精度，便于人读。"""
    probe = SeamProbe(
        prompt="打一套长拳第七段",
        n_frames=240,
        n_windows=2,
        n_seams=1,
        seam_max_cmps=12.3456789,
        seam_mean_cmps=9.87654321,
        total_ms=1234.5678,
        ok=True,
    )
    payload = probe.to_dict()
    assert payload["seam_max_cmps"] == pytest.approx(12.346)
    assert payload["seam_mean_cmps"] == pytest.approx(9.877)
    assert payload["total_ms"] == pytest.approx(1234.6)
