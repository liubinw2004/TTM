"""F-17b：根高振荡诊断（``root_y_diagnostics``）。

**为什么需要它**：F-17a（``|median(f0..f11) − median(f12..f23)| ≤ 30cm``）会把
「动作本身大幅上下」误判成启动瞬态。实测量到的反例是「在房间里走一圈」seed43：
root Y 在 45↔147cm 之间往复（峰峰 103cm、周期 1.5~2.5s），**裁不裁头都不正常**，
但裁头会把判据窗口正好挪到爬坡段，读数从 0.89cm 变 70.19cm。

主理人裁决：这类片段记 ``KNOWN_LIMITATION``（既不是 PASS 也不是 FAIL），依据就是
本文件的诊断函数——它必须能把「振荡」和「GT 那种 0.1cm 的正常行走抖动」区分开。
**区分度全部来自振幅门**：GT 行走满窗的 dY 变号中位数本来就有 32 次（在 0.07cm
范围内抖），只看变号次数会把正常行走判成振荡。
"""

from __future__ import annotations

import numpy as np
import pytest

from ttm.infer.generate import (
    ROOT_Y_OSC_AMPLITUDE_CM,
    ROOT_Y_OSC_MIN_SIGN_CHANGES,
    root_y_diagnostics,
)


def _root_pos(height: np.ndarray) -> np.ndarray:
    """把一维高度包成 ``[T,3]`` 根位移。"""
    array = np.zeros((height.size, 3), dtype=np.float64)
    array[:, 1] = height
    return array


def _sine(amplitude: float, period_frames: int, frames: int = 150) -> np.ndarray:
    """以 ``period_frames`` 为周期、峰峰 = ``2*amplitude`` 的正弦。"""
    t = np.arange(frames, dtype=np.float64)
    return 100.0 + amplitude * np.sin(2.0 * np.pi * t / period_frames)


def test_flat_walk_is_not_oscillation() -> None:
    """GT 那种根高恒定（峰峰 0.1cm）的正常行走，绝不能被判成振荡。"""
    rng = np.random.default_rng(7)
    height = 90.0 + rng.normal(scale=0.03, size=150)  # ±0.1cm 量级的抖动

    result = root_y_diagnostics(_root_pos(height))

    assert result["oscillating"] is False
    assert result["peak_to_peak_cm"] < 1.0


def test_gt_like_jitter_not_misjudged_despite_many_sign_changes() -> None:
    """反例保护：抖动再多，只要振幅不到门就**不算**振荡（GT 变号中位 32 次）。"""
    rng = np.random.default_rng(11)
    height = np.cumsum(rng.normal(scale=0.05, size=150))  # 随机游走但总幅度很小
    height = 90.0 + (height - height.mean()) * 0.5

    result = root_y_diagnostics(_root_pos(height))

    assert result["peak_to_peak_cm"] < ROOT_Y_OSC_AMPLITUDE_CM
    assert result["oscillating"] is False


def test_large_vertical_oscillation_is_detected() -> None:
    """峰峰 100cm、周期约 1.5s（45 帧）的往复 → 判为振荡，并报出周期。"""
    height = _sine(amplitude=50.0, period_frames=45)  # 峰峰 100cm

    result = root_y_diagnostics(_root_pos(height))

    assert result["oscillating"] is True
    assert result["peak_to_peak_cm"] == pytest.approx(100.0, abs=0.5)
    assert result["sign_changes"] >= ROOT_Y_OSC_MIN_SIGN_CHANGES
    # 45 帧 @30fps = 1.5s
    assert result["period_s"] == pytest.approx(1.5, abs=0.2)


def test_stairs_climb_is_not_oscillation() -> None:
    """**纯单调爬升**（无往复）不能被判成振荡——这是去趋势振幅的作用。"""
    height = 90.0 + np.linspace(0.0, 250.0, 150)

    result = root_y_diagnostics(_root_pos(height))

    assert result["peak_to_peak_cm"] >= ROOT_Y_OSC_AMPLITUDE_CM
    assert result["detrended_peak_to_peak_cm"] < 1.0, "单调爬升去趋势后应几乎为 0"
    assert result["sign_changes"] < ROOT_Y_OSC_MIN_SIGN_CHANGES
    assert result["oscillating"] is False


def test_detrended_amplitude_separates_climb_from_oscillation() -> None:
    """同峰峰值：单调爬升 vs 上下往复，只有后者被判振荡。"""
    climb = 90.0 + np.linspace(-50.0, 50.0, 150)
    wander = 90.0 + 50.0 * np.sin(2.0 * np.pi * np.arange(150) / 45.0)

    climb_result = root_y_diagnostics(_root_pos(climb))
    wander_result = root_y_diagnostics(_root_pos(wander))

    assert abs(climb_result["peak_to_peak_cm"] - wander_result["peak_to_peak_cm"]) < 3.0
    assert climb_result["oscillating"] is False
    assert wander_result["oscillating"] is True


def test_local_jump_matches_f17a_reading() -> None:
    """``local_jump_cm`` 必须与 QA 的 F-17a 口径逐值一致（同一份 meta 可单点门控）。"""
    height = np.concatenate([np.full(12, 50.0), np.full(12, 130.0), np.full(126, 130.0)])

    result = root_y_diagnostics(_root_pos(height))
    expected = abs(float(np.median(height[:12])) - float(np.median(height[12:24])))

    assert result["local_jump_cm"] == pytest.approx(expected, abs=1e-6)


def test_amplitude_gate_boundary() -> None:
    """振幅门是唯一区分度来源，边界两侧必须给出相反结论。"""
    below = _sine(amplitude=(ROOT_Y_OSC_AMPLITUDE_CM - 2.0) / 2.0, period_frames=45)
    above = _sine(amplitude=(ROOT_Y_OSC_AMPLITUDE_CM + 2.0) / 2.0, period_frames=45)

    assert root_y_diagnostics(_root_pos(below))["oscillating"] is False
    assert root_y_diagnostics(_root_pos(above))["oscillating"] is True


def test_short_or_bad_input() -> None:
    """帧数不足返回零值；形状不符必须报错（不做静默兜底）。"""
    short = root_y_diagnostics(np.zeros((2, 3), dtype=np.float64))
    assert short["oscillating"] is False
    assert short["peak_to_peak_cm"] == 0.0

    with pytest.raises(ValueError, match=r"\[T,3\]"):
        root_y_diagnostics(np.zeros((10, 2), dtype=np.float64))


def test_generated_oscillating_clip_is_flagged(acceptance_generator) -> None:
    """真实生成：'在房间里走一圈' seed43 必须被标记为振荡（已知限制的样本）。

    注意：``oscillating`` 是**筛查**而不是判决——爬升类（楼梯）去趋势后同样可能
    >40cm（GT 楼梯 p90 就有 60cm），所以 F-17b 只在 F-17a 已经失败时才参照它；
    楼梯类 F-17a 本来就通过，不会走到这一步。
    """
    from ttm.infer.generate import GenerateOptions  # noqa: PLC0415

    # 用生产配置的裁头帧数复现 QA 那条失败用例（trim=0 时 F-17a 反而「侥幸」通过）
    motion = acceptance_generator.generate(
        "在房间里走一圈",
        GenerateOptions(
            duration_s=5.0, steps=50, seed=43, export_bvh=False, guidance=7.5,
            trim_head_frames=12,
        ),
    )
    result = motion.meta["root_y_diagnostics"]

    assert result["oscillating"] is True
    assert result["detrended_peak_to_peak_cm"] >= ROOT_Y_OSC_AMPLITUDE_CM
    # 同一条 clip 的 F-17a 读数也在 meta 里（QA 单点门控用）
    assert result["local_jump_cm"] > 30.0, "这条 clip 的 F-17a 应该是失败的那条"

    # 楼梯类：F-17a 通过（≤30cm），所以不进 F-17b 分支
    stairs = acceptance_generator.generate(
        "慢慢走上楼梯",
        GenerateOptions(
            duration_s=5.0, steps=50, seed=43, export_bvh=False, guidance=7.5,
            trim_head_frames=12,
        ),
    )
    assert stairs.meta["root_y_diagnostics"]["local_jump_cm"] <= 30.0
