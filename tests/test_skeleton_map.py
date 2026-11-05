"""T02 单测：骨架常量一致性 / UE 映射全覆盖 / 特征布局 / FK / 重采样 / 配置校验。

全部离线可跑（不依赖 Z 盘、不依赖 torch）。临时文件一律落在
``D:/TTM/.cache/tmp/pytest`` 之内（写盘硬约束）。
"""

from __future__ import annotations

import dataclasses
import shutil
from pathlib import Path

import numpy as np
import pytest

from ttm import LAYOUT_VERSION
from ttm.common import config as config_mod
from ttm.common import constants as C
from ttm.common import paths as P
from ttm.data.skeleton import SkeletonDef

# fixtures（work_dir 等）由 tests/conftest.py 提供，统一落在 D:/TTM/.cache/tmp/pytest


# ---------------------------------------------------------------------------
# 1. 骨架常量
# ---------------------------------------------------------------------------
EXPECTED_BONE_ORDER = (
    "Hips",
    "Spine",
    "Spine1",
    "Spine2",
    "Spine3",
    "Neck",
    "Head",
    "LeftShoulder",
    "LeftArm",
    "LeftForeArm",
    "LeftHand",
    "RightShoulder",
    "RightArm",
    "RightForeArm",
    "RightHand",
    "LeftUpLeg",
    "LeftLeg",
    "LeftFoot",
    "LeftToeBase",
    "RightUpLeg",
    "RightLeg",
    "RightFoot",
    "RightToeBase",
)


def test_bone_order_matches_contract() -> None:
    """BONE_ORDER 必须与 ARCH §5.2 逐字一致，且 23 个不重名。"""
    assert C.BONE_ORDER == EXPECTED_BONE_ORDER
    assert len(C.BONE_ORDER) == 23
    assert len(set(C.BONE_ORDER)) == 23
    assert C.N_JOINTS == 23
    assert C.N_NON_ROOT_JOINTS == 22
    assert C.NON_ROOT_JOINTS == EXPECTED_BONE_ORDER[1:]


def test_parents_valid() -> None:
    """PARENTS 长度 23、单根、父索引严格小于子索引、无环。"""
    parents = C.PARENTS
    assert len(parents) == 23
    assert parents[0] == -1
    assert sum(1 for p in parents if p < 0) == 1
    for idx in range(1, 23):
        assert 0 <= parents[idx] < idx, f"parents[{idx}]={parents[idx]} 非拓扑序"
    # 显式走一遍父链，确保无环
    for idx in range(23):
        seen = {idx}
        cursor = idx
        while parents[cursor] >= 0:
            cursor = parents[cursor]
            assert cursor not in seen
            seen.add(cursor)
        assert cursor == 0, f"关节 {idx} 的父链未收敛到根"
    # 树形结构语义抽查
    assert parents[C.bone_index("Spine")] == C.bone_index("Hips")
    assert parents[C.bone_index("Spine3")] == C.bone_index("Spine2")
    assert parents[C.bone_index("Neck")] == C.bone_index("Spine3")
    assert parents[C.bone_index("LeftShoulder")] == C.bone_index("Spine3")
    assert parents[C.bone_index("LeftUpLeg")] == C.bone_index("Hips")
    assert parents[C.bone_index("LeftToeBase")] == C.bone_index("LeftFoot")


def test_ue_bone_map_full_coverage() -> None:
    """UE_BONE_MAP 覆盖 23 个关节中的 22 个（Spine3 fold），UE 侧 +root 共 23 骨。"""
    spine3 = C.SPINE_MERGE_PAIR[0]
    assert spine3 == "Spine3"
    assert spine3 not in C.UE_BONE_MAP, "Spine3 必须 fold 进 spine_03，不得单独成骨"

    expected_sources = set(C.BONE_ORDER) - {spine3}
    assert set(C.UE_BONE_MAP) == expected_sources
    assert len(C.UE_BONE_MAP) == 22
    assert len(C.UE_BONE_MAP) == len(C.BONE_ORDER) - 1

    ue_targets = set(C.UE_BONE_MAP.values())
    assert len(ue_targets) == 22, "目标骨名不得重复"
    assert ue_targets | {C.UE_ROOT_BONE} == set(C.UE_BONE_ORDER)
    assert C.UE_ROOT_BONE == "root"
    assert len(C.UE_BONE_ORDER) == 23
    assert C.UE_BONE_ORDER[0] == "root"

    # 抽查关键映射（PRD §3.3 / ARCH §5.2）
    assert C.UE_BONE_MAP["Hips"] == "pelvis"
    assert C.UE_BONE_MAP["Spine2"] == "spine_03"
    assert C.UE_BONE_MAP["Neck"] == "neck_01"
    assert C.UE_BONE_MAP["LeftShoulder"] == "clavicle_l"
    assert C.UE_BONE_MAP["LeftToeBase"] == "ball_l"
    assert C.UE_BONE_MAP["RightToeBase"] == "ball_r"


def test_ue_parents_derived_from_bvh_parents() -> None:
    """UE_PARENTS 必须能由 BVH PARENTS + Spine3 fold 规则推导出来。"""

    def ue_index_of(bvh_idx: int) -> int:
        """BVH 关节索引 → UE 骨索引（Spine3 折叠到 spine_03）。"""
        if bvh_idx < 0:
            return 0  # UE 的 root
        if bvh_idx == C.bone_index("Spine3"):
            return C.UE_BONE_ORDER.index("spine_03")
        return bvh_idx + 1 if bvh_idx <= C.bone_index("Spine3") else bvh_idx

    def bvh_index_of(ue_idx: int) -> int:
        """UE 骨索引 → BVH 关节索引（跳过 UE 的 root）。"""
        if ue_idx <= C.UE_BONE_ORDER.index("spine_03"):
            return ue_idx - 1
        return ue_idx

    assert len(C.UE_PARENTS) == 23
    assert C.UE_PARENTS[0] == -1
    for ue_idx in range(1, 23):
        bvh_idx = bvh_index_of(ue_idx)
        expected = ue_index_of(C.PARENTS[bvh_idx])
        assert C.UE_PARENTS[ue_idx] == expected, (
            f"UE_PARENTS[{ue_idx}]({C.UE_BONE_ORDER[ue_idx]}) 应为 {expected}，"
            f"实际 {C.UE_PARENTS[ue_idx]}"
        )
    # UE 侧同样必须满足拓扑序（供 ue_import 单趟递推）
    for ue_idx in range(1, 23):
        assert 0 <= C.UE_PARENTS[ue_idx] < ue_idx


def test_feat_layout() -> None:
    """146 维特征布局：连续、无重叠、边界与 ARCH §5.3 表格逐段一致。"""
    assert C.FEATURE_DIM == 146
    assert LAYOUT_VERSION == 1

    expected = {
        "root_vel": (0, 2),
        "root_height": (2, 3),
        "heading": (3, 5),
        "heading_vel": (5, 6),
        "root_tilt_6d": (6, 12),
        "joints_6d": (12, 144),
        "foot_contact": (144, 146),
    }
    actual = {name: (slc.start, slc.stop) for name, slc in C.FEAT_LAYOUT.items()}
    assert actual == expected

    cursor = 0
    for name, slc in C.FEAT_LAYOUT.items():
        assert slc.start == cursor, f"{name} 起点 {slc.start} != {cursor}（布局必须连续）"
        cursor = slc.stop
    assert cursor == C.FEATURE_DIM

    # 22 个非根关节 × 6D
    assert C.FEAT_LAYOUT["joints_6d"].stop - C.FEAT_LAYOUT["joints_6d"].start == 6 * 22
    for joint_idx in range(22):
        slc = C.joint_6d_slice(joint_idx)
        assert slc.stop - slc.start == 6
        assert slc.start == 12 + 6 * joint_idx
    with pytest.raises(IndexError):
        C.joint_6d_slice(22)

    # z-score 只作用 4 个连续维
    assert C.NORM_DIMS == (0, 1, 2, 5)
    assert C.HEADING_DIMS == (3, 4)


def test_vae_token_len_contract() -> None:
    """150 帧 / 4× 下采样 = 38 tokens（潜序列长度契约）。"""
    assert C.WINDOW == 150
    assert C.DOWNSAMPLE == 4
    assert C.VAE_TOKEN_LEN == 38
    assert C.token_len_for(150, 4) == 38
    assert C.token_len_for(120, 4) == 30
    assert 4 * C.VAE_TOKEN_LEN >= C.WINDOW
    assert C.LATENT_DIM == 128
    assert C.FPS == 30 and C.SOURCE_FPS == 120


def test_verify_constants_passes() -> None:
    """内置一致性自检必须通过。"""
    C.verify_constants()


def test_verify_constants_detects_bad_layout(monkeypatch: pytest.MonkeyPatch) -> None:
    """篡改布局后自检必须失败（防止契约被静默破坏）。"""
    broken = dict(C.FEAT_LAYOUT)
    broken["foot_contact"] = slice(144, 145)  # 少一维
    monkeypatch.setattr(C, "FEAT_LAYOUT", broken)
    with pytest.raises(AssertionError):
        C.verify_constants()


# ---------------------------------------------------------------------------
# 2. SkeletonDef：FK / 重采样 / 足接触
# ---------------------------------------------------------------------------
def test_skeleton_default_valid() -> None:
    """默认骨架能构造并通过校验。"""
    skeleton = SkeletonDef.default()
    assert skeleton.n_joints == 23
    assert skeleton.euler_order == "YXZ"
    assert skeleton.units == "cm"
    assert skeleton.up_axis == "Y"
    assert skeleton.rest_offsets_cm.shape == (23, 3)
    assert skeleton.bone_index("Hips") == 0
    assert skeleton.parent_of(1) == 0
    assert skeleton.children_of(0) == (1, 15, 19)
    meta = skeleton.as_metadata()
    assert meta["n_joints"] == 23
    assert len(meta["rest_offsets_cm"]) == 23


def test_rest_offsets_hips_height_sane() -> None:
    """rest 偏移的脊椎链与腿链必须对得上 Hips 高度（91.7cm ± 3cm）。"""
    skeleton = SkeletonDef.default()
    offsets = skeleton.rest_offsets_cm
    spine_chain = ["Spine", "Spine1", "Spine2", "Spine3", "Neck", "Head"]
    total_up = sum(offsets[skeleton.bone_index(name), 1] for name in spine_chain)
    assert 40.0 < total_up < 80.0, f"脊椎链总高 {total_up} 不合理"

    leg_chain = ["LeftUpLeg", "LeftLeg", "LeftFoot"]
    total_down = sum(offsets[skeleton.bone_index(name), 1] for name in leg_chain)
    assert abs((C.DEFAULT_HIPS_HEIGHT_CM + total_down) - 0.0) < 3.0, (
        f"腿链末端应接近地面，实际 y={C.DEFAULT_HIPS_HEIGHT_CM + total_down:.2f}"
    )


def test_fk_rest_pose_reconstructs_offsets() -> None:
    """FK 于 rest pose（零旋转）必须重建出累积偏移，误差 < 0.01cm。"""
    skeleton = SkeletonDef.default()
    frames = 5
    euler = np.zeros((frames, 23, 3), dtype=np.float64)
    root = np.tile(np.array([0.0, C.DEFAULT_HIPS_HEIGHT_CM, 0.0]), (frames, 1))

    world = skeleton.fk(euler, root)
    assert world.shape == (frames, 23, 3)

    # 手工累加偏移作为期望值
    expected = np.zeros((23, 3), dtype=np.float64)
    expected[0] = root[0]
    for idx in range(1, 23):
        expected[idx] = expected[C.PARENTS[idx]] + skeleton.rest_offsets_cm[idx]

    for frame in range(frames):
        err = np.abs(world[frame] - expected).max()
        assert err < 0.01, f"第 {frame} 帧 rest-pose FK 误差 {err:.6f}cm > 0.01cm"

    head = world[0, C.bone_index("Head")]
    assert head[1] > C.DEFAULT_HIPS_HEIGHT_CM, "头必须在髋之上"


def test_fk_matches_single_frame_api() -> None:
    """批量 FK 与单帧 FK 必须一致。"""
    skeleton = SkeletonDef.default()
    rng = np.random.default_rng(7)
    euler = rng.uniform(-0.5, 0.5, size=(3, 23, 3))
    root = rng.uniform(-10, 10, size=(3, 3)) + np.array([0.0, 90.0, 0.0])
    batch = skeleton.fk(euler, root)
    single = skeleton.world_joint_pos(euler[1], root[1])
    assert np.allclose(batch[1], single, atol=1e-10)


def test_fk_respects_rotation() -> None:
    """根关节绕 Y 轴转 90° 后，LeftUpLeg 的横向偏移应转到 -Z 方向。"""
    skeleton = SkeletonDef.default()
    euler = np.zeros((1, 23, 3))
    euler[0, 0, 0] = np.pi / 2  # 根绕 Y 旋转 90°（欧拉分量顺序为 Y, X, Z）
    root = np.array([[0.0, C.DEFAULT_HIPS_HEIGHT_CM, 0.0]])
    world = skeleton.fk(euler, root)
    hip_offset = skeleton.rest_offsets_cm[skeleton.bone_index("LeftUpLeg")]
    got = world[0, skeleton.bone_index("LeftUpLeg")] - world[0, 0]
    assert np.linalg.norm(got) == pytest.approx(np.linalg.norm(hip_offset), abs=1e-9)
    # (x, y, z) -> (z, y, -x)
    assert got[1] == pytest.approx(hip_offset[1], abs=1e-9)
    assert got[2] == pytest.approx(-hip_offset[0], abs=1e-6)
    assert got[0] == pytest.approx(hip_offset[2], abs=1e-6)


def test_resample_120_to_30_exact() -> None:
    """120fps → 30fps（4× 整除）逐点误差 < 1e-3。"""
    skeleton = SkeletonDef.default()
    fps_src, fps_dst = 120, 30
    duration = 4.0
    n_src = int(duration * fps_src) + 1  # 481
    t = np.arange(n_src) / fps_src
    signal = np.sin(2 * np.pi * t)

    out = skeleton.resample(signal, fps_src, fps_dst)
    expected_len = int(duration * fps_dst) + 1  # 121
    assert out.shape == (expected_len,)
    expected = np.sin(2 * np.pi * (np.arange(expected_len) / fps_dst))
    assert np.abs(out - expected).max() < 1e-3

    # 多通道 + 非 0 轴
    multi = np.stack([signal, signal * 2.0], axis=1)  # [481, 2]
    out_multi = skeleton.resample(multi, fps_src, fps_dst, axis=0)
    assert out_multi.shape == (expected_len, 2)
    assert np.abs(out_multi[:, 1] - expected * 2.0).max() < 1e-3
    out_axis1 = skeleton.resample(multi.T, fps_src, fps_dst, axis=1)
    assert out_axis1.shape == (2, expected_len)


def test_resample_rotations_constant_is_stable() -> None:
    """常量旋转序列重采样后必须保持不变（球面插值不引入漂移）。"""
    skeleton = SkeletonDef.default()
    angles = np.deg2rad(np.tile(np.array([10.0, 20.0, 30.0]), (480, 23, 1)))
    out = skeleton.resample_rotations(angles, 120, 30)
    assert out.shape == (120, 23, 3)
    assert np.abs(out - angles[0]).max() < 1e-6


def test_resample_motion_aligned() -> None:
    """角度与位移两条时间轴必须严格对齐。"""
    skeleton = SkeletonDef.default()
    rng = np.random.default_rng(11)
    angles = rng.uniform(-30, 30, size=(240, 23, 3))
    root = rng.uniform(-50, 50, size=(240, 3))
    new_angles, new_root = skeleton.resample_motion(angles, root, 120, 30)
    assert new_angles.shape[0] == new_root.shape[0] == 60
    assert new_root[0] == pytest.approx(root[0], abs=1e-9)


def test_foot_contacts_standing() -> None:
    """站立不动：双脚接触恒为 1（高度 0.5cm、速度 0）。"""
    skeleton = SkeletonDef.default()
    frames = 60
    world = np.zeros((frames, 23, 3))
    world[:, :, 1] = 40.0  # 除脚外都在 40cm 高
    world[:, 17, :] = np.array([9.0, 0.5, 5.0])  # LeftFoot
    world[:, 21, :] = np.array([-9.0, 0.5, 5.0])  # RightFoot

    contacts = skeleton.foot_contacts(world, fps=30)
    assert contacts.shape == (frames, 2)
    assert contacts.dtype == np.float32
    assert contacts[5:].min() == pytest.approx(1.0)
    assert set(np.unique(contacts).tolist()) <= {0.0, 1.0}


def test_foot_contacts_moving_is_released() -> None:
    """高速移动的脚不判为接触。"""
    skeleton = SkeletonDef.default()
    frames = 60
    world = np.zeros((frames, 23, 3))
    world[:, :, 1] = 40.0
    # 每帧移动 100cm ⇒ 3000 cm/s ≫ 15 cm/s
    world[:, 17, 0] = np.arange(frames) * 100.0
    world[:, 17, 1] = 0.5
    world[:, 21, 0] = np.arange(frames) * 100.0
    world[:, 21, 1] = 0.5
    contacts = skeleton.foot_contacts(world, fps=30)
    assert contacts[3:].max() == pytest.approx(0.0)


def test_foot_contacts_high_foot_is_released() -> None:
    """抬起的脚（高度 > 阈值）不判为接触。"""
    skeleton = SkeletonDef.default()
    frames = 30
    world = np.zeros((frames, 23, 3))
    world[:, :, 1] = 40.0
    world[:, 17, 1] = 30.0  # 抬到 30cm
    world[:, 21, 1] = 30.0
    contacts = skeleton.foot_contacts(world, fps=30)
    assert contacts.max() == pytest.approx(0.0)


# ---------------------------------------------------------------------------
# 3. 配置系统
# ---------------------------------------------------------------------------
@pytest.mark.parametrize(
    ("name", "tier", "data_dir", "max_files"),
    [
        ("default.yaml", "M", "data", 12000),
        ("smoke.yaml", "S", "data/smoke", 50),
        ("tier_s.yaml", "S", "data", 4300),
    ],
)
def test_load_shipped_configs(name: str, tier: str, data_dir: str, max_files: int) -> None:
    """三份随包配置都能加载，且关键字段符合设计。"""
    cfg = config_mod.load_config(f"configs/{name}")
    assert cfg.data.tier == tier
    assert cfg.project.data_dir == data_dir
    assert cfg.data.max_files == max_files
    assert cfg.motion.n_joints == 23
    assert cfg.motion.feature_dim == 146
    assert cfg.data.fps == 30
    assert cfg.data.window == 150
    assert cfg.data.overlap == 30
    assert cfg.vae.latent_dim == 128
    assert cfg.vae.downsample == 4
    assert C.token_len_for(cfg.data.window, cfg.vae.downsample) == 38
    assert cfg.diffusion.d_model % cfg.diffusion.n_heads == 0
    assert cfg.runtime.seed == 42
    assert cfg.retarget.spine_merge in config_mod.VALID_SPINE_MERGE
    assert cfg.text.encoder == "BAAI/bge-small-zh-v1.5"
    assert cfg.text.frozen is True
    assert cfg.source_path.endswith(name)


def test_default_config_matches_dataclass_defaults() -> None:
    """dataclass 默认值必须与 default.yaml 的语义一致（ckpt 留空由 run_dir 推导）。"""
    cfg = config_mod.load_config("configs/default.yaml")
    plain = config_mod.default_config()
    assert cfg.motion == plain.motion
    assert cfg.vae == plain.vae
    assert cfg.diffusion == plain.diffusion
    assert cfg.text == plain.text
    assert cfg.serve == plain.serve
    assert cfg.retarget == plain.retarget
    assert cfg.runtime == plain.runtime
    # ckpt 路径留空 → 由 run_dir() 按档位推导，避免跨档位互相覆盖
    assert cfg.runtime.vae_ckpt == ""
    assert cfg.runtime.diffusion_ckpt == ""


def test_config_paths_derivation() -> None:
    """派生路径必须落在 ``data/tier_{tier}`` 约定位置。"""
    cfg = config_mod.load_config("configs/tier_s.yaml")
    assert cfg.tier_dir().as_posix().endswith("data/tier_S")
    assert cfg.tier_name() == "tier_S"
    assert cfg.index_path().name == "index.jsonl"
    assert cfg.labels_path().name == "labels.jsonl"
    assert cfg.failed_path().name == "failed.jsonl"
    assert cfg.norm_stats_path().name == "norm_stats.npz"
    assert cfg.shards_path().name == "shards"
    # 训练产物按档位隔离：runs/tier_s/{vae,diffusion}
    assert cfg.run_dir("vae").as_posix().endswith("runs/tier_s/vae")
    assert cfg.run_dir("diffusion").as_posix().endswith("runs/tier_s/diffusion")


def _write_yaml(path: Path, text: str) -> Path:
    """把 YAML 文本写到临时目录（经写盘守卫校验）。"""
    target = P.resolve_out(path, mkparents=True)
    target.write_text(text, encoding="utf-8")
    return target


def test_bad_config_unknown_field(work_dir: Path) -> None:
    """拼错的字段名必须报错并指出位置。"""
    bad = _write_yaml(
        work_dir / "bad.yaml",
        "data:\n  tier: S\n  fps: 30\n  windows: 150\n",
    )
    with pytest.raises(config_mod.ConfigError) as excinfo:
        config_mod.load_config(bad)
    assert "data" in str(excinfo.value)
    assert "windows" in str(excinfo.value)


def test_bad_config_wrong_type(work_dir: Path) -> None:
    """类型不匹配必须报错。"""
    bad = _write_yaml(work_dir / "bad_type.yaml", "data:\n  tier: 123\n")
    with pytest.raises(config_mod.ConfigError) as excinfo:
        config_mod.load_config(bad)
    assert "data.tier" in str(excinfo.value)


def test_bad_config_bad_tier(work_dir: Path) -> None:
    """枚举越界必须报错。"""
    bad = _write_yaml(work_dir / "bad_tier.yaml", "data:\n  tier: XL\n")
    with pytest.raises(config_mod.ConfigError):
        config_mod.load_config(bad)


def test_bad_config_feature_dim_contract(work_dir: Path) -> None:
    """feature_dim 被改成 260 时必须报错（PRD 里的 260 是立项前估算）。"""
    bad = _write_yaml(
        work_dir / "bad_dim.yaml",
        "data:\n  tier: S\nmotion:\n  feature_dim: 260\n",
    )
    with pytest.raises(config_mod.ConfigError) as excinfo:
        config_mod.load_config(bad)
    assert "motion.feature_dim" in str(excinfo.value)


def test_bad_config_ratio_sum(work_dir: Path) -> None:
    """val_ratio + test_ratio >= 1 必须报错。"""
    bad = _write_yaml(
        work_dir / "bad_ratio.yaml",
        "data:\n  tier: S\n  val_ratio: 0.6\n  test_ratio: 0.5\n",
    )
    with pytest.raises(config_mod.ConfigError):
        config_mod.load_config(bad)


def test_bad_config_overlap(work_dir: Path) -> None:
    """overlap >= window 必须报错。"""
    bad = _write_yaml(work_dir / "bad_overlap.yaml", "data:\n  tier: S\n  overlap: 150\n")
    with pytest.raises(config_mod.ConfigError):
        config_mod.load_config(bad)


def test_bad_config_unknown_section(work_dir: Path) -> None:
    """未知顶层小节必须报错。"""
    bad = _write_yaml(work_dir / "bad_section.yaml", "dataloader:\n  workers: 8\n")
    with pytest.raises(config_mod.ConfigError) as excinfo:
        config_mod.load_config(bad)
    assert "dataloader" in str(excinfo.value)


def test_bad_config_missing_file() -> None:
    """配置文件不存在时给出清晰错误。"""
    with pytest.raises(config_mod.ConfigError):
        config_mod.load_config("configs/does_not_exist_xyz.yaml")


def test_dump_resolved_roundtrip(work_dir: Path) -> None:
    """生效配置能落盘并原样读回（字段语义不变）。"""
    cfg = config_mod.load_config("configs/smoke.yaml")
    target = config_mod.dump_resolved(cfg, "unit_test_t02")
    assert target.exists()
    text = target.read_text(encoding="utf-8")
    assert "layout_version" in text
    assert "feature_dim" in text
    assert "146" in text

    # 落盘内容能被重新解析（通过 reload 相同结构）
    import yaml

    reloaded = yaml.safe_load(text)
    assert reloaded["data"]["tier"] == "S"
    assert reloaded["motion"]["feature_dim"] == 146
    assert reloaded["_meta"]["n_joints"] == 23
    assert reloaded["_meta"]["vae_token_len"] == 38
    assert reloaded["_meta"]["layout_version"] == LAYOUT_VERSION
    shutil.rmtree(target.parent, ignore_errors=True)


def test_config_ratio_percent_form(work_dir: Path) -> None:
    """比例字段支持百分比写法（10 → 0.1）。"""
    path = _write_yaml(
        work_dir / "pct.yaml",
        "data:\n  tier: S\n  val_ratio: 10\n  test_ratio: 5\n",
    )
    cfg = config_mod.load_config(path)
    assert cfg.data.val_ratio == pytest.approx(0.1)
    assert cfg.data.test_ratio == pytest.approx(0.05)
