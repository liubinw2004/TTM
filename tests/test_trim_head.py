"""F-17：启动瞬态裁头（``trim_head_frames``）。

现象：生成动作开头 12~24 帧里根高有 25~41cm 的横向跳变（最差「舞一段剑术」
40.67cm）。GT 同口径标定 p50=0.01 / p99=0.58 / max=1.59cm、越阈 0%，所以这不是
判据过严，而是生成侧缺陷：首窗开头的 latent 尚未收敛。

修法：按「用户时长 + 裁头帧数」申请窗口，导出前丢掉开头 ``trim_head_frames`` 帧。
**用户侧时长不变**；ground_origin 的平移量在裁头后的首帧重新计算（顺序不可反）。

本文件锁四件事：
1. 帧数预算不变式（纯函数，不依赖权重）；
2. 配置默认值与「裸 GenerateOptions 不裁头」的边界；
3. 裁头后首帧最低足触地点仍在 ±2cm（真实生成）；
4. 接缝指标在裁头后仍打在正确的帧号上（纯函数）。
"""

from __future__ import annotations

import numpy as np
import pytest

from ttm.common.config import RuntimeConfig, load_config
from ttm.infer.blend import seam_velocity_jumps
from ttm.infer.generate import (
    GenerateOptions,
    ground_origin_offset_cm,
    resolve_frame_budget,
    shift_root_y,
)

#: QA F-17 判据：前 12 帧与随后 12 帧的根高差上限（cm）。
F17_TRANSIENT_MAX_CM = 30.0

#: QA F-17 的口径窗口（帧）。
F17_TRANSIENT_WINDOW = 12

#: ground_origin 之后首帧最低足踝允许的残差（cm）。
FOOT_ON_GROUND_TOL_CM = 2.0


def _startup_jump(root_pos: np.ndarray) -> float:
    """``|median(f0..f11) − median(f12..f23)|``（cm），即 QA 的 F-17 判据口径。"""
    array = np.asarray(root_pos, dtype=np.float64)
    head = float(np.median(array[:F17_TRANSIENT_WINDOW, 1]))
    tail = float(np.median(array[F17_TRANSIENT_WINDOW : 2 * F17_TRANSIENT_WINDOW, 1]))
    return abs(tail - head)


# ---------------------------------------------------------------------------
# 1. 帧数预算（纯函数）
# ---------------------------------------------------------------------------
def test_frame_budget_keeps_user_duration() -> None:
    """申请帧数 = 用户帧数 + 裁头帧数；用户帧数只由时长决定。"""
    user, target = resolve_frame_budget(5.0, 12)
    assert user == 150
    assert target == 162  # 5s 请求 + 12 帧裁头 → 2 个窗口（stride=120）

    user, target = resolve_frame_budget(5.0, 0)
    assert (user, target) == (150, 150)

    user, target = resolve_frame_budget(1.0, 12)
    assert (user, target) == (30, 42)


def test_frame_budget_clamps_negative_trim() -> None:
    """负的裁头帧数等同关闭，不会把帧数算少。"""
    assert resolve_frame_budget(5.0, -4) == (150, 150)


def test_frame_budget_rejects_non_positive_duration() -> None:
    """时长非正必须报错，不能悄悄生成 0 帧动作。"""
    with pytest.raises(ValueError, match="duration_s"):
        resolve_frame_budget(0.0, 12)


# ---------------------------------------------------------------------------
# 2. 配置默认值与边界
# ---------------------------------------------------------------------------
def test_config_ships_trim_head_twelve() -> None:
    """生产配置默认裁 12 帧（GT 标定：首窗瞬态在 12 帧内收敛）。"""
    assert RuntimeConfig().trim_head_frames == 12
    cfg = load_config("configs/tier_s.yaml")
    assert int(cfg.runtime.trim_head_frames) == 12


def test_bare_generate_options_do_not_trim() -> None:
    """裸 ``GenerateOptions()`` 默认不裁头——既有单测与老产物口径不受影响。

    与 ``ground_origin`` 默认开不同：裁头会改变采样窗口数（时延翻倍），
    必须由生产入口显式注入，避免把「参数调整」偷渡进单元测试。
    """
    assert GenerateOptions().trim_head_frames == 0


# ---------------------------------------------------------------------------
# 3. 真实生成：帧数不变 + 首帧足触地
# ---------------------------------------------------------------------------
def test_generate_trim_preserves_frame_count(acceptance_generator) -> None:
    """裁头后：帧数不变、窗口数变多、meta 记录裁头信息。"""
    plain = acceptance_generator.generate(
        "慢慢走上楼梯",
        GenerateOptions(duration_s=5.0, steps=20, seed=42, export_bvh=False, trim_head_frames=0),
    )
    trimmed = acceptance_generator.generate(
        "慢慢走上楼梯",
        GenerateOptions(duration_s=5.0, steps=20, seed=42, export_bvh=False, trim_head_frames=12),
    )

    assert plain.n_frames == 150
    assert trimmed.n_frames == 150, "裁头改变了用户侧时长"
    assert trimmed.n_windows >= plain.n_windows
    assert trimmed.meta["trim_head_frames"] == 12
    assert trimmed.meta["frames_before_trim"] == 162


def test_generate_trim_first_frame_foot_on_ground(acceptance_generator) -> None:
    """裁头后 ground_origin 仍成立：平移后首帧最低足踝 ∈[-2,+2]cm。"""
    motion = acceptance_generator.generate(
        "慢慢走上楼梯",
        GenerateOptions(duration_s=5.0, steps=20, seed=43, export_bvh=False, trim_head_frames=12),
    )
    offset = float(motion.ground_offset_cm)
    shifted = shift_root_y(motion.root_pos, offset)
    # 平移后重新求「首帧最低足触地点」，应当回到 0（±2cm 量化/平滑残差）
    residual = float(ground_origin_offset_cm(motion.quats, shifted))

    assert abs(residual) <= FOOT_ON_GROUND_TOL_CM, (
        f"裁头后首帧最低足踝残差 {residual:.2f}cm 超出 ±{FOOT_ON_GROUND_TOL_CM}cm"
    )


def test_generate_trim_suppresses_startup_transient(acceptance_generator) -> None:
    """裁头后 F-17 判据（``|median(f0..f11) − median(f12..f23)| ≤ 30cm``）通过。"""
    motion = acceptance_generator.generate(
        "打一套陈氏太极拳，动作舒缓连贯",
        GenerateOptions(duration_s=5.0, steps=20, seed=44, export_bvh=False, trim_head_frames=12),
    )
    jump = _startup_jump(motion.root_pos)
    assert jump <= F17_TRANSIENT_MAX_CM, (
        f"裁头后启动瞬态仍有 {jump:.2f}cm > {F17_TRANSIENT_MAX_CM}cm"
        "（若确为 latent 首窗收敛问题，报主理人，不要靠加帧数硬凑）"
    )


# ---------------------------------------------------------------------------
# 4. 接缝指标与裁头的帧号对齐
# ---------------------------------------------------------------------------
def test_seam_metric_aligns_after_head_trim() -> None:
    """裁头把接缝整体前移，``head_trim`` 必须让指标打在正确的帧号上。

    构造：320 帧平滑根轨迹 + 接缝处一个已知速度台阶。真正的接缝在原始时间轴上
    是 120 / 240（stride=120）；裁掉开头 12 帧后落到 108 / 228。
    ``head_trim=0`` 会把指标打在 120 / 240（已经错位的帧号）→ 读不到台阶；
    ``head_trim=12`` 才对齐到 108 / 228 → 台阶被检出。
    """
    fps = 30
    frames = 400
    trim = 12
    stride = 120  # window 150 - overlap 30

    rng = np.random.default_rng(5)
    root = np.zeros((frames, 3), dtype=np.float64)
    root[:, 1] = np.cumsum(np.sin(np.linspace(0.0, 6.0, frames)) * 2.0)
    root[:, 2] = np.cumsum(np.cos(np.linspace(0.0, 5.0, frames)) * 1.5)
    # 在**裁头后**的接缝位置（108 / 228）埋速度台阶
    for seam in (stride - trim, 2 * stride - trim):
        root[seam - 1 : seam + 2, 1] += np.array([0.0, 30.0, 30.0])

    aligned = seam_velocity_jumps(root, 3, fps=fps, head_trim=trim)
    misaligned = seam_velocity_jumps(root, 3, fps=fps, head_trim=0)

    assert aligned.shape == misaligned.shape == (2,)
    assert aligned.min() > 300.0, f"对齐后没检出接缝台阶: {aligned}"
    assert misaligned.max() < 150.0, f"未对齐却检出大台阶（说明测试构造无效）: {misaligned}"


def test_seam_metric_head_trim_larger_than_stride_does_not_crash() -> None:
    """``head_trim`` 大于 stride 时首个接缝会落到负帧号，必须被裁剪住。"""
    root = np.zeros((400, 3), dtype=np.float64)
    root[:, 1] = np.linspace(0.0, 100.0, 400)

    jumps = seam_velocity_jumps(np.asarray(root), 3, fps=30, head_trim=200)

    assert jumps.shape == (2,)
    assert np.all(np.isfinite(jumps))
