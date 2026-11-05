"""ground_origin（地面锚定）的单测。

背景（根高排查的下游结论）：特征表示是**绝对根高**，楼梯类生成会从数据里
继承「世界坐标随楼层累加」的根高（实测终值 403cm），BVH/预览原样播放角色会
从 4 米高空开始。修法是把整条动作刚性平移，使**首帧最低足触地点 ≈ 0**——
用首帧足部而不是首帧根高，因为首帧可能正处于腾空。

主理人验收断言（QA 会复用）：

* 楼梯样例平移后首帧足高 ∈ [-2, +2]cm；
* **竖直爬升总量不变**（只许平移不许缩放）。
"""

from __future__ import annotations

import numpy as np
import pytest
from scipy.spatial.transform import Rotation

from ttm.common import constants as C
from ttm.infer.generate import (
    GenerateOptions,
    GeneratedMotion,
    ground_origin_offset_cm,
    shift_root_y,
)


# ---------------------------------------------------------------------------
# 合成动作构造
# ---------------------------------------------------------------------------
def _identity_quats(frames: int) -> np.ndarray:
    """``[F,23,4]`` 全单位四元数（xyzw, w=1）。"""
    quats = np.zeros((frames, C.N_JOINTS, 4), dtype=np.float64)
    quats[..., 3] = 1.0
    return quats


def _foot_world_y_identity(root_y: float) -> float:
    """单位旋转下足关节世界高度 = root_y + rest 偏移链的 y 之和。

    这是与 ``ground_origin_offset_cm`` **独立**的实现（纯标量链式求和，
    不走矩阵递推），用作交叉验证。
    """
    total = 0.0
    for joint in C.FOOT_CONTACT_JOINTS:
        idx = int(joint)
        while idx >= 0:
            total += float(C.DEFAULT_REST_OFFSETS_CM[idx][1])
            idx = int(C.PARENTS[idx])
        break  # 两只脚链长相同，算一条即可（min 由对称性保证）
    return root_y + total


def _stairs_motion(frames: int = 120, *, start_y: float = 100.0, climb: float = 300.0):
    """合成「上楼」：根从 ``start_y`` 匀速爬升 ``climb`` cm，姿态为单位旋转。"""
    quats = _identity_quats(frames)
    t = np.linspace(0.0, 1.0, frames)
    root_pos = np.stack(
        [
            np.linspace(0.0, 60.0, frames),  # X 前进
            start_y + climb * t,  # Y 爬升
            np.zeros(frames),
        ],
        axis=1,
    )
    return quats, root_pos


# ---------------------------------------------------------------------------
# 核心断言（主理人验收口径）
# ---------------------------------------------------------------------------
def test_stairs_shift_brings_first_foot_to_ground() -> None:
    """楼梯样例平移后首帧足高 ∈ [-2, +2]cm（验收断言 1）。"""
    quats, root_pos = _stairs_motion()
    offset = ground_origin_offset_cm(quats, root_pos)
    # 交叉验证：offset 必须等于独立 FK 算出的首帧足高
    assert offset == pytest.approx(_foot_world_y_identity(float(root_pos[0, 1])), abs=1e-6)
    shifted = shift_root_y(root_pos, offset)
    # 用同一独立实现复核平移后的首帧足高
    after = _foot_world_y_identity(float(shifted[0, 1]))
    assert -2.0 <= after <= 2.0


def test_vertical_climb_unchanged() -> None:
    """竖直爬升总量不变、逐帧变化不变、XZ 不动（验收断言 2：只许平移不许缩放）。"""
    quats, root_pos = _stairs_motion()
    offset = ground_origin_offset_cm(quats, root_pos)
    shifted = shift_root_y(root_pos, offset)
    # 爬升总量：末帧 − 首帧
    assert shifted[-1, 1] - shifted[0, 1] == pytest.approx(root_pos[-1, 1] - root_pos[0, 1])
    # 逐帧差分完全一致（刚性平移的定义）
    np.testing.assert_allclose(np.diff(shifted[:, 1]), np.diff(root_pos[:, 1]), atol=1e-12)
    # 水平轨迹不动
    np.testing.assert_allclose(shifted[:, 0], root_pos[:, 0], atol=0.0)
    np.testing.assert_allclose(shifted[:, 2], root_pos[:, 2], atol=0.0)
    # 幅值没被缩放：整体极差不变
    assert np.ptp(shifted[:, 1]) == pytest.approx(np.ptp(root_pos[:, 1]))


def test_foot_height_range_preserved() -> None:
    """足高极差（max(足高)−min(足高)）平移前后差 < 0.5cm（验收断言：只平移不缩放）。

    比「根爬升总量不变」更严：足高随姿态逐帧变化，极差不变才能证明平移没有
    挤压/拉伸任何一帧。
    """
    from ttm.infer.generate import _world_joint_positions

    quats, root_pos = _stairs_motion()
    offset = ground_origin_offset_cm(quats, root_pos)
    joints = list(C.FOOT_CONTACT_JOINTS)
    before = _world_joint_positions(quats, root_pos)[:, joints, 1]
    after = _world_joint_positions(quats, shift_root_y(root_pos, offset))[:, joints, 1]
    assert abs((after.max() - after.min()) - (before.max() - before.min())) < 0.5


def test_ground_level_motion_is_noop() -> None:
    """非楼梯样例（足已在地面附近）：平移量 < 5cm，近似 no-op（验收断言）。

    实测口径：真机「挥手打招呼」offset=1.4cm；这里用更严的 2cm 下限锁死。
    """
    frames = 60
    quats = _identity_quats(frames)
    ground_root_y = -_foot_world_y_identity(0.0)  # 让首帧足高恰为 0 的根高
    root_pos = np.stack(
        [np.zeros(frames), np.full(frames, ground_root_y), np.zeros(frames)], axis=1
    )
    offset = ground_origin_offset_cm(quats, root_pos)
    assert abs(offset) < 2.0
    shifted = shift_root_y(root_pos, offset)
    assert abs(_foot_world_y_identity(float(shifted[0, 1]))) <= 2.0


def test_airborne_first_frame_uses_foot_not_root() -> None:
    """首帧腾空（跳跃起跳）：offset = 首帧足高（>0），与首帧根高无关。

    按 spec 用**足部**锚定：首帧在空中时，把足部放到地面而不是把骨盆放到地面。
    """
    frames = 60
    quats = _identity_quats(frames)
    airborne_root_y = -_foot_world_y_identity(0.0) + 80.0  # 足在 80cm 高的空中
    root_pos = np.stack(
        [np.zeros(frames), np.full(frames, airborne_root_y), np.zeros(frames)], axis=1
    )
    offset = ground_origin_offset_cm(quats, root_pos)
    assert offset == pytest.approx(80.0, abs=1e-6)  # = 首帧足高，而不是根高
    shifted = shift_root_y(root_pos, offset)
    assert abs(_foot_world_y_identity(float(shifted[0, 1]))) <= 2.0


def test_non_identity_pose_fk_path() -> None:
    """非单位姿态：helper 走完整 FK（矩阵递推），与独立逐关节重算一致。"""
    rng = np.random.default_rng(7)
    frames = 40
    local = Rotation.random(frames * C.N_JOINTS, random_state=rng)
    quats = local.as_quat().reshape(frames, C.N_JOINTS, 4)
    root_pos = np.stack(
        [np.zeros(frames), np.full(frames, 150.0), np.zeros(frames)], axis=1
    )
    offset = ground_origin_offset_cm(quats, root_pos)
    # 独立复核：scipy 矩阵递推一帧（局部旋转要逐级组合成世界旋转）
    local = Rotation.from_quat(quats[0]).as_matrix()  # [23,3,3]
    world = np.zeros((C.N_JOINTS, 3))
    world_rot = np.empty_like(local)
    world_rot[0] = local[0]
    world[0] = root_pos[0]
    for idx in range(1, C.N_JOINTS):
        parent = int(C.PARENTS[idx])
        world_rot[idx] = world_rot[parent] @ local[idx]
        world[idx] = world[parent] + world_rot[parent] @ C.DEFAULT_REST_OFFSETS_CM[idx]
    feet = world[list(C.FOOT_CONTACT_JOINTS), 1]
    assert offset == pytest.approx(float(feet.min()), abs=1e-9)


def test_shift_returns_copy_not_inplace() -> None:
    """``shift_root_y`` 不修改入参（root_pos 是要保留世界坐标的那份）。"""
    root_pos = np.array([[0.0, 100.0, 0.0], [1.0, 200.0, 2.0]])
    snapshot = root_pos.copy()
    shifted = shift_root_y(root_pos, 30.0)
    np.testing.assert_array_equal(root_pos, snapshot)
    assert shifted[0, 1] == pytest.approx(70.0)


def test_invalid_shapes_raise() -> None:
    """形状不符必须显式报错（不要安静地算出垃圾 offset）。"""
    bad_quats = np.zeros((5, 3, 4))
    bad_quats[..., 3] = 1.0
    with pytest.raises(ValueError):
        ground_origin_offset_cm(bad_quats, np.zeros((5, 3)))
    good_quats = _identity_quats(5)
    with pytest.raises(ValueError):
        ground_origin_offset_cm(good_quats, np.zeros((4, 3)))


def test_generate_options_default_ground_origin_on() -> None:
    """默认开启：服务端/脚本必须测随默认发货的行为。"""
    assert GenerateOptions().ground_origin is True


def test_generated_motion_carries_offset() -> None:
    """``GeneratedMotion.ground_offset_cm`` 默认 0 且可携带（服务端读它做 BVH 平移）。"""
    quats = _identity_quats(3)
    root_pos = np.zeros((3, 3))
    motion = GeneratedMotion(
        prompt="x",
        quats=quats,
        root_pos=root_pos,
        feature=None,
        n_windows=1,
        seam_jumps=np.zeros(0),
    )
    assert motion.ground_offset_cm == 0.0
    motion.ground_offset_cm = 96.5
    assert shift_root_y(motion.root_pos, motion.ground_offset_cm)[0, 1] == pytest.approx(-96.5)
