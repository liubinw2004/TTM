"""T04 单测：146 维特征布局 / 6D 一致性 / 归一化往返 / 足接触 / 中文标签 / 预处理。

全部离线可跑（合成 BVH 由 conftest 生成）。
"""

from __future__ import annotations

import shutil
from pathlib import Path

import numpy as np
import pytest
from conftest import build_bvh_text, write_bvh, write_config

from ttm import LAYOUT_VERSION
from ttm.common import constants as C
from ttm.common import paths as P
from ttm.common.config import load_config
from ttm.data.bvh import BvhParser
from ttm.data.features import (
    MOTION_MIN_FOOT_HEIGHT_CM,
    FeatureExtractor,
    MotionFeature,
    NormStats,
    count_windows,
    motion_quality_issue,
    quat_to_six_d,
    rotmat_to_6d,
    six_d_to_quat,
    six_d_to_rotmat,
    slice_windows_arrays,
    trim_placeholder_frames,
)
from ttm.data.labels_zh import LabelGenerator, coverage_check, zh_label_for_directory
from ttm.data.preprocess import SHARD_FIELDS, PreprocessPipeline, read_labels
from ttm.data.scan import Scanner
from ttm.data.skeleton import SkeletonDef


@pytest.fixture()
def skeleton() -> SkeletonDef:
    """默认 23 关节骨架。"""
    return SkeletonDef.default()


@pytest.fixture()
def extractor(skeleton: SkeletonDef) -> FeatureExtractor:
    """默认特征提取器。"""
    return FeatureExtractor(skeleton=skeleton, fps=30)


def _make_motion(
    n_frames: int = 300,
    *,
    animated: bool = True,
    root_yaw_deg: float = 0.0,
    forward_speed: float = 0.0,
    bob_cm: float = 0.0,
    yaw_wave_deg: float = 0.0,
    speed_wave_cmps: float = 0.0,
) -> tuple[np.ndarray, np.ndarray]:
    """构造 ``(euler_deg[T,23,3], root_pos[T,3])``（角度制）。

    Args:
        n_frames: 帧数。
        animated: 是否加肢体旋转抖动。
        root_yaw_deg: 根关节绕 Y 的基准旋转（度）。
        forward_speed: 根沿前向的平均速度（cm/s），用于速度测试。
        bob_cm: 骨盆高度正弦起伏幅度（cm），用于根高度方差。
        yaw_wave_deg: 偏航角正弦摆动幅度（度），用于 heading / 角速度方差。
        speed_wave_cmps: 前向速度正弦起伏幅度（cm/s）。

    Returns:
        ``(euler_deg, root_pos_cm)``。
    """
    t = np.arange(n_frames, dtype=np.float64) / 30.0
    euler = np.zeros((n_frames, C.N_JOINTS, 3), dtype=np.float64)
    # 根绕 Y：基准 + 0.25Hz 摆动（索引 0 = Y，YXZ 欧拉序）
    euler[:, 0, 0] = root_yaw_deg + yaw_wave_deg * np.sin(2 * np.pi * 0.25 * t)
    if animated:
        for joint in range(1, C.N_JOINTS):
            euler[:, joint, 0] = 5.0 * np.sin(2 * np.pi * 0.5 * t + joint * 0.3)
            euler[:, joint, 1] = 3.0 * np.cos(2 * np.pi * 0.8 * t + joint * 0.2)
            euler[:, joint, 2] = 1.5 * np.sin(2 * np.pi * 1.2 * t + joint * 0.1)

    yaw_rad = np.deg2rad(root_yaw_deg)
    forward = np.array([np.sin(yaw_rad), 0.0, np.cos(yaw_rad)])
    distance = forward_speed * t + speed_wave_cmps * np.sin(2 * np.pi * 0.2 * t)
    root_pos = np.zeros((n_frames, 3), dtype=np.float64)
    root_pos[:, 1] = 91.7 + bob_cm * np.sin(2 * np.pi * 0.5 * t)
    root_pos += forward[None, :] * distance[:, None]
    return euler, root_pos


# ---------------------------------------------------------------------------
# 1. 布局与数值合法性
# ---------------------------------------------------------------------------


def test_feature_shape_and_layout(extractor: FeatureExtractor) -> None:
    """特征 shape (T,146) 且各段切片符合 §5.3 布局。"""
    euler, root = _make_motion(300)
    feature = extractor.extract(euler, root)

    assert isinstance(feature, MotionFeature)
    assert feature.feat.shape == (300, C.FEATURE_DIM)
    assert feature.feat.dtype == np.float32
    assert feature.mask.shape == (300,)
    assert (feature.mask == 1).all()
    assert feature.layout_version == LAYOUT_VERSION
    assert feature.fps == 30

    for name, expected_len in (
        ("root_vel", 2),
        ("root_height", 1),
        ("heading", 2),
        ("heading_vel", 1),
        ("root_tilt_6d", 6),
        ("joints_6d", 132),
        ("foot_contact", 2),
    ):
        assert feature.segment(name).shape == (300, expected_len), name

    assert np.all(np.isfinite(feature.feat))


def test_root_height_channel(extractor: FeatureExtractor) -> None:
    """[2] 必须是骨盆世界高度（cm）。"""
    euler, root = _make_motion(120, animated=False)
    root[:, 1] = np.linspace(90.0, 92.0, 120)
    feature = extractor.extract(euler, root)
    assert np.allclose(feature.segment("root_height")[:, 0], root[:, 1], atol=1e-4)


def test_heading_channels_for_yawed_root(extractor: FeatureExtractor) -> None:
    """根绕 Y 转 30° → heading = (sin30, cos30)。"""
    euler, root = _make_motion(60, animated=False, root_yaw_deg=30.0)
    feature = extractor.extract(euler, root)
    heading = feature.segment("heading")
    assert np.allclose(heading[:, 0], np.sin(np.deg2rad(30.0)), atol=1e-6)
    assert np.allclose(heading[:, 1], np.cos(np.deg2rad(30.0)), atol=1e-6)
    # sin^2 + cos^2 == 1
    assert np.allclose(np.sum(heading**2, axis=1), 1.0, atol=1e-6)


def test_six_d_segments_are_unit_norm(extractor: FeatureExtractor) -> None:
    """6D 段每对 3 维向量的模长必须为 1 ± 1e-5（旋转矩阵前两行的性质）。"""
    euler, root = _make_motion(120)
    feature = extractor.extract(euler, root)

    tilt = feature.segment("root_tilt_6d").reshape(-1, 2, 3)
    assert np.allclose(np.linalg.norm(tilt, axis=-1), 1.0, atol=1e-5)

    joints = feature.segment("joints_6d").reshape(-1, 22, 2, 3)
    norms = np.linalg.norm(joints, axis=-1)
    assert np.allclose(norms, 1.0, atol=1e-5)


def test_root_tilt_identity(extractor: FeatureExtractor) -> None:
    """无旋转时根倾斜 6D = [1,0,0, 0,1,0]。"""
    euler, root = _make_motion(10, animated=False)
    feature = extractor.extract(euler, root)
    tilt = feature.segment("root_tilt_6d")
    expected = np.array([1.0, 0.0, 0.0, 0.0, 1.0, 0.0], dtype=np.float32)
    assert np.allclose(tilt, expected, atol=1e-6)


def test_root_velocity_in_heading_frame(extractor: FeatureExtractor) -> None:
    """沿当前朝向前进时，[0] 应等于速度、[1] 应接近 0。"""
    speed = 60.0  # cm/s
    euler, root = _make_motion(120, animated=False, forward_speed=speed)
    feature = extractor.extract(euler, root)
    vel = feature.segment("root_vel")
    # 首帧是单边差分（np.gradient 的边界处理），只检查中间帧
    assert np.allclose(vel[1:-1, 0], speed, atol=1e-2)
    assert np.allclose(vel[1:-1, 1], 0.0, atol=1e-2)


def test_heading_velocity_yaw_rate(extractor: FeatureExtractor) -> None:
    """匀速偏航时 [5] 应等于角速度（rad/s）。"""
    n = 60
    rate_deg_per_s = 90.0
    euler = np.zeros((n, C.N_JOINTS, 3))
    t = np.arange(n) / 30.0
    euler[:, 0, 0] = rate_deg_per_s * t  # 根 Y 角线性增加
    root = np.tile(np.array([0.0, 91.7, 0.0]), (n, 1))
    feature = extractor.extract(euler, root)
    omega = feature.segment("heading_vel")[:, 0]
    assert np.allclose(omega[1:-1], np.deg2rad(rate_deg_per_s), atol=1e-3)


def test_foot_contact_standing(extractor: FeatureExtractor) -> None:
    """站立不动：足接触段 ≈ 1（T04 验收标准 ③）。"""
    euler, root = _make_motion(90, animated=False)
    feature = extractor.extract(euler, root)
    contacts = feature.segment("foot_contact")
    assert contacts.shape == (90, 2)
    assert contacts[5:].min() == pytest.approx(1.0)
    assert set(np.unique(contacts).tolist()) <= {0.0, 1.0}


def test_foot_contact_lifted_foot(extractor: FeatureExtractor, skeleton: SkeletonDef) -> None:
    """抬起右腿（大腿绕 X 转 90°）后右侧接触应释放。"""
    n = 90
    euler = np.zeros((n, C.N_JOINTS, 3))
    euler[:, C.bone_index("RightUpLeg"), 1] = 90.0  # 欧拉 (Y,X,Z) 的第 2 分量 = X
    root = np.tile(np.array([0.0, 91.7, 0.0]), (n, 1))
    feature = extractor.extract(euler, root)
    contacts = feature.segment("foot_contact")
    # 左脚着地，右脚抬到髋高度
    assert contacts[5:, 0].min() == pytest.approx(1.0)
    assert contacts[5:, 1].max() == pytest.approx(0.0)


# ---------------------------------------------------------------------------
# 2. 6D / 四元数互转
# ---------------------------------------------------------------------------


def test_six_d_roundtrip() -> None:
    """旋转矩阵 → 6D → 旋转矩阵 必须无损。"""
    rng = np.random.default_rng(3)
    quats = rng.normal(size=(64, 4))
    quats /= np.linalg.norm(quats, axis=1, keepdims=True)
    from scipy.spatial.transform import Rotation

    matrices = Rotation.from_quat(quats).as_matrix()
    six = rotmat_to_6d(matrices)
    restored = six_d_to_rotmat(six)
    assert np.allclose(restored, matrices, atol=1e-10)

    quats_again = six_d_to_quat(six)
    # 四元数双覆盖：q 与 -q 表示同一旋转
    dots = np.abs(np.sum(quats * quats_again, axis=1))
    assert np.allclose(dots, 1.0, atol=1e-8)

    six_again = quat_to_six_d(quats)
    assert np.allclose(six_again, six, atol=1e-8)


def test_six_d_rejects_bad_shape() -> None:
    """形状不符必须报错。"""
    with pytest.raises(ValueError):
        rotmat_to_6d(np.zeros((4, 3)))
    with pytest.raises(ValueError):
        six_d_to_rotmat(np.zeros((4, 5)))
    with pytest.raises(ValueError):
        quat_to_six_d(np.zeros((4, 3)))


def test_six_d_survives_noise() -> None:
    """6D 带噪声时 Gram-Schmidt 仍返回正交矩阵（行正交、行列式为 +1）。"""
    six = np.array([[1.0, 0.0, 0.0, 0.0, 1.0, 1e-3]])
    matrix = six_d_to_rotmat(six)
    assert matrix.shape == (1, 3, 3)
    assert np.allclose(matrix @ matrix.transpose(0, 2, 1), np.eye(3), atol=1e-10)
    assert np.isclose(np.linalg.det(matrix[0]), 1.0, atol=1e-10)


def test_six_d_degenerate_returns_identity() -> None:
    """退化 6D（全 0 / 两行平行）→ 单位阵，不抛错（训练初期模型输出接近 0）。"""
    zeros = six_d_to_rotmat(np.zeros((2, 6)))
    assert np.allclose(zeros, np.eye(3)[None, :, :])
    parallel = six_d_to_rotmat(np.array([[1.0, 0.0, 0.0, 3.0, 0.0, 0.0]]))
    assert np.isclose(np.linalg.det(parallel[0]), 1.0, atol=1e-10)
    assert np.allclose(parallel[0] @ parallel[0].T, np.eye(3), atol=1e-10)
    # 退化输入转四元数不抛错（scipy 需要合法旋转矩阵）
    quats = six_d_to_quat(np.zeros((2, 6)))
    assert np.allclose(quats, np.array([[0.0, 0.0, 0.0, 1.0]] * 2))


# ---------------------------------------------------------------------------
# 3. 归一化
# ---------------------------------------------------------------------------


def test_normalization_roundtrip(extractor: FeatureExtractor) -> None:
    """归一化 → 反归一化往返误差 < 1e-6（float64 路径）。"""
    # 四个 z-score 维都必须有非零方差，否则「std≈1」的断言不成立：
    #   dim0/1 根速度 ← speed_wave_cmps，dim2 根高 ← bob_cm，dim5 角速度 ← yaw_wave_deg
    euler, root = _make_motion(
        300,
        forward_speed=45.0,
        bob_cm=4.0,
        yaw_wave_deg=25.0,
        speed_wave_cmps=20.0,
    )
    feature = extractor.extract(euler, root, dtype=np.float64)
    stats = NormStats.from_features([feature.feat])
    original = feature.feat.copy()

    restored = feature.normalize(stats).denormalize(stats)
    assert restored.feat.shape == feature.feat.shape
    error = np.abs(restored.feat - original).max()
    assert error < 1e-6, f"往返误差 {error:.3e} > 1e-6"

    # 归一化后 z-score 四维应近似零均值、单位方差
    normalized = feature.normalize(stats).feat
    for dim in C.NORM_DIMS:
        assert abs(normalized[:, dim].mean()) < 1e-9
        assert abs(normalized[:, dim].std() - 1.0) < 1e-6

    # 未归一化维必须保持原值
    for dim in (3, 4, 6, 100, 144):
        assert np.allclose(normalized[:, dim], original[:, dim], atol=1e-12)


def test_norm_stats_constant_dim_is_identity(extractor: FeatureExtractor) -> None:
    """常量维（如匀速直行时的角速度）统计 std 退化为 1，归一化结果为 0 且可逆。"""
    euler, root = _make_motion(120, forward_speed=0.0)  # 完全静止 → 四维全常量
    feature = extractor.extract(euler, root, dtype=np.float64)
    stats = NormStats.from_features([feature.feat])

    for dim in C.NORM_DIMS:
        assert stats.std[dim] == pytest.approx(1.0)

    normalized = feature.normalize(stats).feat
    for dim in C.NORM_DIMS:
        assert np.allclose(normalized[:, dim], 0.0, atol=1e-12)
    restored = feature.normalize(stats).denormalize(stats)
    assert np.allclose(restored.feat, feature.feat, atol=1e-9)


def test_norm_stats_identity() -> None:
    """恒等统计量不改变数值。"""
    stats = NormStats.identity()
    assert np.allclose(stats.mean, 0.0)
    assert np.allclose(stats.std, 1.0)
    assert stats.layout_version == LAYOUT_VERSION


def test_norm_stats_save_load(work_dir: Path) -> None:
    """统计量落盘/读回一致，字段严格为 mean/std/layout_version。"""
    stats = NormStats.from_features([np.random.default_rng(1).normal(size=(50, C.FEATURE_DIM))])
    path = stats.save(work_dir / "norm_stats.npz")
    assert path.exists()

    with np.load(path) as payload:
        assert set(payload.files) == {"mean", "std", "layout_version"}
        assert payload["mean"].shape == (C.FEATURE_DIM,)
        assert payload["std"].shape == (C.FEATURE_DIM,)
        assert int(payload["layout_version"]) == LAYOUT_VERSION

    loaded = NormStats.load(path)
    assert np.allclose(loaded.mean, stats.mean)
    assert np.allclose(loaded.std, stats.std)


def test_norm_stats_layout_version_mismatch(work_dir: Path) -> None:
    """layout_version 不匹配必须报错（防止用旧统计量跑新布局）。"""
    target = P.resolve_out(work_dir / "bad_stats.npz", mkparents=True)
    np.savez(
        target,
        mean=np.zeros(C.FEATURE_DIM),
        std=np.ones(C.FEATURE_DIM),
        layout_version=np.int32(LAYOUT_VERSION + 1),
    )
    with pytest.raises(ValueError):
        NormStats.load(target)


def test_norm_stats_rejects_bad_shape() -> None:
    """形状非法必须报错。"""
    with pytest.raises(ValueError):
        NormStats(mean=np.zeros(10), std=np.ones(10))
    with pytest.raises(ValueError):
        NormStats(mean=np.zeros(C.FEATURE_DIM), std=np.zeros(C.FEATURE_DIM))


# ---------------------------------------------------------------------------
# 4. 特征 ↔ 旋转还原
# ---------------------------------------------------------------------------


def test_to_rotations_root_identity(extractor: FeatureExtractor) -> None:
    """静态姿态还原：23 个四元数都应是单位四元数，高度与输入一致。"""
    euler, root = _make_motion(20, animated=False)
    feature = extractor.extract(euler, root)
    quats, root_pos = feature.to_rotations_root()

    assert quats.shape == (20, 23, 4)
    assert root_pos.shape == (20, 3)
    assert np.allclose(np.abs(quats[..., 3]), 1.0, atol=1e-6)
    assert np.allclose(root_pos[:, 1], 91.7, atol=1e-4)


def test_to_rotations_root_matches_source(extractor: FeatureExtractor) -> None:
    """非平凡姿态：还原的四元数必须与源欧拉角一致。"""
    from scipy.spatial.transform import Rotation

    euler, root = _make_motion(60, animated=True, root_yaw_deg=25.0)
    feature = extractor.extract(euler, root)
    quats, _ = feature.to_rotations_root()

    expected = Rotation.from_euler(C.EULER_ORDER, euler.reshape(-1, 3), degrees=True).as_quat()
    expected = expected.reshape(60, 23, 4)
    dots = np.abs(np.sum(quats * expected, axis=-1))
    assert np.allclose(dots, 1.0, atol=1e-5)


# ---------------------------------------------------------------------------
# 5. 窗口切分
# ---------------------------------------------------------------------------


def test_slice_windows_full_and_tail(extractor: FeatureExtractor) -> None:
    """301 帧 → 2 个整窗；121 帧 → 1 个 pad 窗（mask 尾 29 帧为 0）。"""
    euler, root = _make_motion(301)
    feature = extractor.extract(euler, root)
    windows = extractor.slice_windows(feature)
    assert len(windows) == 2
    for feat_win, mask_win, win_idx in windows:
        assert feat_win.shape == (150, C.FEATURE_DIM)
        assert mask_win.shape == (150,)
        assert int(mask_win.sum()) == 150
    assert [item[2] for item in windows] == [0, 1]

    euler2, root2 = _make_motion(121)
    feature2 = extractor.extract(euler2, root2)
    windows2 = extractor.slice_windows(feature2)
    assert len(windows2) == 1
    feat_win, mask_win, _ = windows2[0]
    assert int(mask_win.sum()) == 121
    assert np.all(feat_win[121:] == 0.0)
    assert np.all(mask_win[121:] == 0)


def test_slice_windows_too_short(extractor: FeatureExtractor) -> None:
    """< 75 帧不产生窗口。"""
    euler, root = _make_motion(74)
    feature = extractor.extract(euler, root)
    assert extractor.slice_windows(feature) == []
    assert extractor.window_count(74) == 0
    assert extractor.window_count(75) == 1


def test_motion_feature_validation() -> None:
    """MotionFeature 对形状与版本做严格校验。"""
    with pytest.raises(ValueError):
        MotionFeature(feat=np.zeros((10, 100)), mask=np.ones(10, dtype=np.uint8))
    with pytest.raises(ValueError):
        MotionFeature(
            feat=np.zeros((10, C.FEATURE_DIM)),
            mask=np.ones(9, dtype=np.uint8),
        )
    with pytest.raises(ValueError):
        MotionFeature(
            feat=np.zeros((10, C.FEATURE_DIM)),
            mask=np.ones(10, dtype=np.uint8),
            layout_version=LAYOUT_VERSION + 1,
        )


# ---------------------------------------------------------------------------
# 6. 中文标签
# ---------------------------------------------------------------------------


def test_zh_label_taxonomy() -> None:
    """taxonomy 精确匹配覆盖已知大类/细类/武术类别。"""
    assert zh_label_for_directory("1.7.Daily_Life_Behavior") == "日常行为"
    # 不带方向词 → 泛化的「上下楼梯」（旧版误标成「上楼梯」）
    assert zh_label_for_directory("1.7.3.3.Stairs_Up_Down") == "上下楼梯"
    assert zh_label_for_directory("1.7.3.1.Basic_Gait") == "基本步态"
    assert zh_label_for_directory("1.13.Object_Carrying_And_Manipulation") == "持物与操作"
    assert zh_label_for_directory("5.3.Pop_Dance") == "流行舞蹈"
    # 武术：基名 + 序号展开，不再塌成基名
    assert zh_label_for_directory("Bajiquan") == "八极拳"
    assert zh_label_for_directory("Changquan1") == "长拳第一段"
    assert zh_label_for_directory("Changquan12") == "长拳第十二段"
    assert zh_label_for_directory("Chen_style_Tai_Chi2") == "陈式太极拳第二路"
    # Broadsword 是刀不是剑：旧版因 KEYWORD_RULES 里 sword 排在 broadsword 前而错标
    assert zh_label_for_directory("Basic_Broadsword") == "基本刀术"
    assert zh_label_for_directory("Basic_Sword") == "基本剑术"


def test_zh_label_wushu_index_expansion() -> None:
    """武术 148 个带序号类别必须逐个展开，不能塌成基名。"""
    assert zh_label_for_directory("Tai_Chi18") == "太极拳第十八式"
    assert zh_label_for_directory("Nanquan7") == "南拳第七路"
    assert zh_label_for_directory("Optional_Straight_Sword14") == "自选剑术第十四段"
    assert zh_label_for_directory("Wing_Chun3") == "咏春第三段"
    labels = {
        zh_label_for_directory(f"Changquan{index}") for index in range(1, 13)
    }
    assert len(labels) == 12, f"12 套长拳必须得到 12 个不同标签，实得 {sorted(labels)}"


def test_zh_label_cleans_invisible_chars() -> None:
    """零宽字符必须清洗（实测 ``Penetrating_Arm_Fist\\u200b``）。"""
    assert zh_label_for_directory("Penetrating_Arm_Fist\u200b") == "通背拳"
    assert zh_label_for_directory("Penetrating_Arm_Fist") == "通背拳"
    # 带空格的目录名也要能命中
    assert zh_label_for_directory("International_Compulsory_Cudgel Routine_I") == "国际规定棍术一路"


def test_zh_label_l2_disambiguation() -> None:
    """同名 l3 靠 l2 消歧，不能塌成同一个标签。"""
    from ttm.data.labels_zh import zh_label_for_l3

    assert zh_label_for_l3("3.3.1.2.Pass_Ball", "3.3.1.Basketball") == "篮球传球"
    assert zh_label_for_l3("3.3.4.2.Pass_Ball", "3.3.4.Volley_Ball") == "排球传球"
    assert zh_label_for_l3("2.4.1.4.Block", "2.4.1.Box_Ing") == "拳击格挡"
    assert zh_label_for_l3("2.4.2.3.Block", "2.4.2.Fencing") == "击剑格挡"


def test_zh_label_no_l1_fallback_for_known_l3() -> None:
    """已知 l3 禁止落 l1 大类名兜底（旧版 39% 窗口被「复杂行为动作序列」吞掉）。"""
    from ttm.data.labels_zh import CATEGORY_ZH

    l1_labels = set(CATEGORY_ZH.values())
    probes = [
        "1.12.7.1.Front_Desk_Process",
        "1.12.4.3.Pointing_Flow",
        "2.3.1.12.Backflip",
        "3.3.5.3.Loop_Drive",
        "1.12.1.4.Trash_Disposal_Flow",
    ]
    for probe in probes:
        label = zh_label_for_directory(probe)
        assert label, f"{probe} 未产出标签"
        assert label not in l1_labels, f"{probe} 落到了大类兜底「{label}」"


def test_zh_label_keyword_fallback() -> None:
    """未见过的细类名靠关键词兜底，且方向词优先于泛化词。"""
    # 未指明上下 stairs → 泛化「上下楼梯」
    assert zh_label_for_directory("1.7.3.9.Walking_On_Stairs_Slowly") == "上下楼梯"
    assert zh_label_for_directory("Walking_Up_Stairs") == "上楼梯"
    assert zh_label_for_directory("Walking_Down_Stairs") == "下楼梯"
    assert zh_label_for_directory("1.7.9.3.Sweeping_The_Kitchen") == "扫地"
    assert zh_label_for_directory("unknown_category") == ""


def test_zh_label_granularity_gate() -> None:
    """验收门槛：已知 l3 的 distinct 标签数 ≥ 40，且关键词不串味。"""
    from ttm.data.labels_zh import KEYWORD_RULES, all_known_labels

    known = all_known_labels()
    assert len(known) >= 40, f"distinct 标签只有 {len(known)} 个，低于 40"

    tokens = [token for token, _ in KEYWORD_RULES]
    # 排序不变量：楼梯/武术类（优先级 0）必须整体排在泛化动词之前，
    # 否则 ``walking``(7 字符) 会比 ``stair``(6) 先命中，楼梯被标成行走。
    assert tokens.index("stair") < tokens.index("walking")
    # 刀不能被剑抢走：broadsword 必须先于 sword 命中
    assert tokens.index("broadsword") < tokens.index("sword")
    assert tokens.index("cudgel") < tokens.index("sword")


def test_zh_label_coverage_of_prd_categories() -> None:
    """PRD §3.1 列出的 14 个大类必须全部命中映射表。"""
    categories = [
        "1.7.Daily_Life_Behavior",
        "1.9.Special_Worker_Actions",
        "1.12.Action_Sequence_Complex_Behavior",
        "1.13.Object_Carrying_And_Manipulation",
        "2.1.Basic_Environment_Interaction_Class",
        "2.2.Complex_Terrain_Interaction",
        "2.3.Extreme_Env_Interaction",
        "2.4.Adversarial_Interaction",
        "3.1.Object_Carrying",
        "3.2.Environment_Operation",
        "3.3.Ball_Interaction",
        "4.1.Box_Ing",
        "4.3.Equipment_Exercises",
        "5.3.Pop_Dance",
    ]
    hit, total = coverage_check(categories)
    assert hit == total, f"大类映射覆盖 {hit}/{total}"


def test_label_generator_variants() -> None:
    """每条 clip 至少 3 条、最多 15 条互不重复的中文描述（映射类 ≥8）。"""
    generator = LabelGenerator()
    label, texts = generator.generate(
        category_l1="1.7.Daily_Life_Behavior",
        category_l2="1.7.3.Walking_Movement_With_Posture",
        category_l3="1.7.3.3.Stairs_Up_Down",
    )
    assert label == "上下楼梯"
    assert len(texts) == len(set(texts))
    assert len(texts) >= 8
    assert len(texts) <= 15
    assert "上下楼梯" in texts
    assert any("楼梯" in text or "台阶" in text for text in texts)
    assert all(isinstance(text, str) and text.strip() for text in texts)


def test_label_generator_unmapped_fallback() -> None:
    """未映射类也保证 ≥3 条描述，且标签可读。"""
    generator = LabelGenerator()
    label, texts = generator.generate(
        category_l1="9.9.Unknown", category_l2="9.9.9.Mystery_Move", category_l3=""
    )
    assert label
    assert len(texts) >= 3
    assert len(texts) == len(set(texts))


def test_label_generator_is_deterministic() -> None:
    """同一输入两次生成结果一致（可复现）。"""
    generator = LabelGenerator()
    first = generator.generate(category_l3="1.7.3.3.Stairs_Up_Down")
    second = generator.generate(category_l3="1.7.3.3.Stairs_Up_Down")
    assert first == second


# ---------------------------------------------------------------------------
# 7. 预处理端到端（离线合成数据）
# ---------------------------------------------------------------------------


def _build_pp_tree(root: Path) -> None:
    """构造 3 个可入训 clip + 1 个坏文件的小数据集。"""
    base = root / "1.7.Daily_Life_Behavior" / "1.7.3.Walking_Movement_With_Posture"
    write_bvh(
        base / "1.7.3.1.Basic_Gait" / "20251211" / "Take_001" / "Take_001_actor_0.bvh",
        build_bvh_text(n_frames=481),
    )
    write_bvh(
        base / "1.7.3.1.Basic_Gait" / "20251211" / "Take_002" / "Take_002_actor_0.bvh",
        build_bvh_text(n_frames=601),
    )
    write_bvh(
        base / "1.7.3.3.Stairs_Up_Down" / "20260326" / "Take_115" / "Take_115_stair_0.bvh",
        build_bvh_text(n_frames=1201),
    )
    write_bvh(
        base / "1.7.3.3.Stairs_Up_Down" / "20260326" / "Take_116" / "Take_116_bad.bvh",
        "HIERARCHY\nROOT bad\n{\n\tOFFSET 0 0 0\n"
        "\tCHANNELS 6 Xposition Yposition Zposition Yrotation Xrotation Zrotation\n}\n"
        "MOTION\nFrames: 60\nFrame Time: 0.00833333\n" + "0 0 0 0 0 0\n" * 60,
    )


def test_preprocess_end_to_end(work_dir: Path) -> None:
    """预处理：分片字段/形状/归一化正确，labels/norm_stats/failed 齐备。"""
    dataset_root = work_dir / "BVH_Body"
    _build_pp_tree(dataset_root)
    config_path = write_config(
        work_dir / "pp_cfg.yaml",
        bvh_roots=[dataset_root.as_posix()],
        data_dir="data/unit_test_pp",
        max_files=100,
        shard_size=2,
        val_ratio=0.0,
        test_ratio=0.0,
    )
    cfg = load_config(config_path)

    scan_result = Scanner(cfg, workers=1).run()
    assert len(scan_result.entries) == 3
    assert len(scan_result.failed) == 1
    Scanner(cfg, workers=1).write(scan_result)

    pipeline = PreprocessPipeline(cfg, workers=2)
    result = pipeline.run()

    # 481→121 帧(1 窗) / 601→151 帧(1 窗) / 1201→301 帧(2 窗) = 4 窗
    assert result.n_clips == 3
    assert result.n_windows == 4
    assert result.n_failed == 0
    assert len(result.shards) == 2  # shard_size=2 → 4 条样本切成 2 片
    assert result.norm_stats is not None
    assert result.norm_stats.layout_version == LAYOUT_VERSION
    assert result.norm_stats.count == 4 * 150  # 全部来自 train（val/test 比例 0）

    # --- 分片契约（ARCH §5.5）
    # 路径口径以 cfg 为准（data_dir 可被 smoke/unit-test 覆盖）
    assert cfg.shards_path() == P.shards_dir(cfg.data.tier, cfg.project.data_path())
    shard_paths = sorted(cfg.shards_path().glob("shard_*.npz"))
    assert len(shard_paths) == 2
    total = 0
    all_heights: list[float] = []
    for path in shard_paths:
        with np.load(path) as payload:
            assert set(payload.files) == set(SHARD_FIELDS), payload.files
            feat = payload["feat"]
            mask = payload["mask"]
            assert feat.dtype == np.float32
            assert feat.ndim == 3 and feat.shape[1:] == (150, C.FEATURE_DIM)
            assert mask.dtype == np.uint8
            assert mask.shape == (feat.shape[0], 150)
            assert payload["clip_id"].dtype == np.dtype("<U80")
            assert payload["win_idx"].dtype == np.int16
            assert int(payload["layout_version"]) == LAYOUT_VERSION
            assert int(payload["fps"]) == 30
            total += feat.shape[0]
            all_heights.append(float(feat[:, :, C.FEAT_LAYOUT["root_height"].start].mean()))
            # 归一化后根高度维应接近 0 均值（train 统计）
    assert total == 4
    assert all(abs(value) < 3.0 for value in all_heights), all_heights

    # --- norm_stats.npz
    with np.load(cfg.norm_stats_path()) as payload:
        assert int(payload["layout_version"]) == LAYOUT_VERSION
        stds = payload["std"]
        for dim in C.NORM_DIMS:
            assert stds[dim] > 0

    # --- labels.jsonl（窗口级）
    labels = read_labels(cfg.labels_path())
    assert len(labels) == 4
    required = {"clip_id", "zh_label", "zh_texts", "duration", "fps", "n_frames", "split"}
    assert required <= set(labels[0])
    for row in labels:
        assert row["clip_id"].endswith(f"_win{row['win_idx']:02d}")
        assert row["n_frames"] == 150
        assert row["fps"] == 30
        assert len(row["zh_texts"]) >= 3
        assert row["zh_label"]
        assert row["split"] == "train"

    # --- 清理
    shutil.rmtree(P.project_root() / "data" / "unit_test_pp", ignore_errors=True)
    shutil.rmtree(P.project_root() / "runs" / "unit_test", ignore_errors=True)


# ---------------------------------------------------------------------------
# 4. 数据质量守卫（实测踩坑：T-pose 占位首帧 / 整段静止的占位数据）
# ---------------------------------------------------------------------------
def test_trim_placeholder_frames_drops_leading_tpose() -> None:
    """首帧全 0 旋转的 T-pose 占位帧被裁掉，其余帧原样保留。"""
    euler, root = _make_motion(10, animated=True)
    tpose = np.zeros((1, C.N_JOINTS, 3), dtype=np.float64)
    root_tpose = np.zeros((1, 3), dtype=np.float64)
    root_tpose[0, 1] = 91.7
    stacked_euler = np.concatenate([tpose, euler], axis=0)
    stacked_root = np.concatenate([root_tpose, root], axis=0)

    trimmed_euler, trimmed_root, n_trimmed = trim_placeholder_frames(stacked_euler, stacked_root)
    assert n_trimmed == 1
    assert trimmed_euler.shape == (10, C.N_JOINTS, 3)
    assert np.allclose(trimmed_euler, euler)
    assert np.allclose(trimmed_root, root)


def test_trim_placeholder_frames_keeps_at_least_one() -> None:
    """整段都是 0 旋转时不裁空（退化由 quality gate 负责淘汰）。"""
    euler = np.zeros((5, C.N_JOINTS, 3), dtype=np.float64)
    root = np.tile(np.array([0.0, 91.7, 0.0]), (5, 1))
    trimmed_euler, _, n_trimmed = trim_placeholder_frames(euler, root)
    assert n_trimmed == 4
    assert trimmed_euler.shape[0] == 1


def test_motion_quality_accepts_valid_motion(skeleton: SkeletonDef) -> None:
    """正常行走动作应通过质量检查。"""
    euler, root = _make_motion(120, forward_speed=45.0, bob_cm=2.0, yaw_wave_deg=15.0)
    assert motion_quality_issue(skeleton, euler, root) is None


def test_motion_quality_rejects_static_clip(skeleton: SkeletonDef) -> None:
    """整段静止（旋转极差 0）→ 淘汰原因含「静止」。"""
    euler = np.zeros((60, C.N_JOINTS, 3), dtype=np.float64)
    root = np.tile(np.array([0.0, 91.7, 0.0]), (60, 1))
    issue = motion_quality_issue(skeleton, euler, root)
    assert issue is not None and "静止" in issue


def test_motion_quality_rejects_below_ground(skeleton: SkeletonDef) -> None:
    """根高度塌到地面以下（实测坏数据 y≈2.28cm）→ 淘汰原因含「足部低于地面」。"""
    euler, root = _make_motion(60, animated=True)
    root[:, 1] = 2.28
    issue = motion_quality_issue(skeleton, euler, root)
    assert issue is not None and "足部低于地面" in issue
    assert MOTION_MIN_FOOT_HEIGHT_CM < 0


def test_motion_quality_rejects_non_finite(skeleton: SkeletonDef) -> None:
    """含 NaN 直接淘汰。"""
    euler, root = _make_motion(30, animated=True)
    root[3, 1] = np.nan
    issue = motion_quality_issue(skeleton, euler, root)
    assert issue is not None and "NaN" in issue


def test_preprocess_drops_degenerate_clip(work_dir: Path) -> None:
    """端到端：静止的坏 clip 计入 failed，不进入分片。"""
    dataset_root = P.safe_makedirs(work_dir / "quality_tree")
    take_ok = P.safe_makedirs(
        dataset_root / "1.7.3.1.Basic_Gait" / "20260101" / "Take_001"
    )
    take_bad = P.safe_makedirs(
        dataset_root / "1.7.3.1.Basic_Gait" / "20260101" / "Take_002"
    )
    write_bvh(take_ok / "Take_001_walk.bvh", build_bvh_text(n_frames=400))
    # 静止片段：animated=False 且首帧 T-pose → 全 0 旋转
    write_bvh(take_bad / "Take_002_static.bvh", build_bvh_text(n_frames=400, animated=False))

    config_path = write_config(
        work_dir / "quality_cfg.yaml",
        bvh_roots=[dataset_root.as_posix()],
        data_dir="data/unit_test_quality",
        max_files=10,
        val_ratio=0.0,
        test_ratio=0.0,
    )
    cfg = load_config(config_path)
    scan_result = Scanner(cfg, workers=1).run()
    assert len(scan_result.entries) == 2
    Scanner(cfg, workers=1).write(scan_result)

    result = PreprocessPipeline(cfg, workers=1).run()
    assert result.n_clips == 1
    assert result.n_failed == 1
    failed = read_labels(cfg.failed_path())
    assert len(failed) >= 1
    assert any("degenerate" in str(row.get("error", "")) for row in failed)

    shutil.rmtree(P.project_root() / "data" / "unit_test_quality", ignore_errors=True)
    shutil.rmtree(P.project_root() / "runs" / "unit_test", ignore_errors=True)


def test_preprocess_requires_index(work_dir: Path) -> None:
    """没有 index.jsonl 时给出明确错误（提示先跑 01_scan）。"""
    config_path = write_config(
        work_dir / "pp_cfg2.yaml",
        bvh_roots=[(work_dir / "nowhere").as_posix()],
        data_dir="data/unit_test_pp2",
    )
    cfg = load_config(config_path)
    with pytest.raises(FileNotFoundError):
        PreprocessPipeline(cfg, workers=1).run()


# ---------------------------------------------------------------------------
# 尾窗起点（M1' 实测 bug：越界产生全 padding 窗口）
# ---------------------------------------------------------------------------
def _fake_clip(n_frames: int) -> tuple[np.ndarray, np.ndarray]:
    """构造 ``[T,146]`` 特征与全 1 mask。"""
    feat = np.arange(n_frames * C.FEATURE_DIM, dtype=np.float32).reshape(n_frames, C.FEATURE_DIM)
    return feat, np.ones(n_frames, dtype=np.uint8)


def test_long_clip_tail_window_is_not_empty() -> None:
    """超长 clip（step=450）的尾窗不许越界成全 padding。

    实测 ``n_frames=1209`` 时旧代码取 ``cursor=3*450=1350 > 1209``，产出
    mask 全 0 的窗口——全库共 265 个（13696 的 1.9%），等于往训练集里掺
    1.9% 的纯噪声样本（重建项 0、KL 照算）。
    """
    feat, mask = _fake_clip(1209)
    windows = slice_windows_arrays(feat, mask, stride=450)
    assert len(windows) == count_windows(1209, stride=450) == 4
    assert all(int(win_mask.sum()) > 0 for _f, win_mask, _i in windows)
    assert [int(win_mask.sum()) for _f, win_mask, _i in windows] == [150, 150, 150, 150]


def test_tail_window_starts_at_last_valid_start() -> None:
    """尾窗起点必须夹到 ``n_frames - window``，否则会漏掉尾部 150–300 帧。"""
    feat, mask = _fake_clip(1400)
    windows = slice_windows_arrays(feat, mask, stride=450)
    last_feat, last_mask, last_idx = windows[-1]
    assert last_idx == len(windows) - 1
    assert int(last_mask.sum()) == 150
    # 尾窗内容 == 原 clip 的最后 150 帧
    np.testing.assert_allclose(last_feat, feat[-150:])


def test_no_empty_window_for_any_length() -> None:
    """任意长度（>= window）都不许出现全 0 mask 窗口。"""
    for n_frames in (150, 151, 380, 900, 901, 1209, 1350, 1400, 2850):
        feat, mask = _fake_clip(n_frames)
        stride = 450 if n_frames > 900 else 150
        windows = slice_windows_arrays(feat, mask, stride=stride)
        assert len(windows) == count_windows(n_frames, stride=stride)
        assert all(int(win_mask.sum()) > 0 for _f, win_mask, _i in windows), n_frames


def test_short_clip_tail_keeps_pad_and_mask() -> None:
    """``n_frames < window`` 的极短 clip 仍走 pad+mask（这是 mask 的唯一用途）。"""
    feat, mask = _fake_clip(90)
    windows = slice_windows_arrays(feat, mask)
    assert len(windows) == 1
    win_feat, win_mask, _idx = windows[0]
    assert int(win_mask.sum()) == 90
    assert win_feat.shape == (C.WINDOW, C.FEATURE_DIM)
    assert np.all(win_feat[90:] == 0)
