"""T08 单测：滑窗融合 / 根轨迹连续化 / DDIM 采样 / 端到端生成与 BVH 导出。

全部离线可跑：用随机初始化的小模型验证**形状与连续性契约**，不依赖训练产物。
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest

torch = pytest.importorskip("torch")

from ttm.common import constants as C  # noqa: E402
from ttm.data.bvh import BvhParser  # noqa: E402
from ttm.data.features import MotionFeature, NormStats, six_d_to_rotmat  # noqa: E402
from ttm.export.bvh_io import motion_to_euler_deg, write_bvh  # noqa: E402
from ttm.infer.blend import (  # noqa: E402
    blend_feature_windows,
    expected_frames,
    seam_velocity_jumps,
    smooth_root_track,
    window_starts,
)
from ttm.infer.generate import GenerateOptions, MotionGenerator  # noqa: E402
from ttm.infer.sampling import LatentSampler, SamplingConfig, tokens_for_frames  # noqa: E402
from ttm.models.diffusion import DiffusionTransformer, GaussianDiffusion  # noqa: E402
from ttm.models.text_encoder import TextProjector  # noqa: E402
from ttm.models.vae import MotionVAE  # noqa: E402

pytestmark = pytest.mark.requires_torch


# ---------------------------------------------------------------------------
# 1. 窗口几何
# ---------------------------------------------------------------------------
def test_window_geometry() -> None:
    """步长 = window − overlap；K 个窗口融合后帧数 = W + (K−1)·S。"""
    cfg = SamplingConfig()
    assert cfg.stride == C.WINDOW - C.OVERLAP == 120
    assert window_starts(3) == [0, 120, 240]
    assert expected_frames(1) == 150
    assert expected_frames(3) == 150 + 2 * 120
    assert cfg.windows_for_frames(150) == 1
    assert cfg.windows_for_frames(360) == 3  # 12 秒 @30fps
    assert tokens_for_frames(150) == 38
    assert tokens_for_frames(360) == 90


# ---------------------------------------------------------------------------
# 2. 特征融合
# ---------------------------------------------------------------------------
def _make_feature_window(
    n_frames: int = C.WINDOW,
    *,
    forward_speed: float = 100.0,
    yaw: float = 0.0,
    offset: float = 0.0,
) -> np.ndarray:
    """构造一个可分辨的合成窗口（根速度已知、heading 已知）。

    注意只设置 ``root_vel`` 的**前向**通道（dim 0），侧向通道保持 0——若整段赋值
    会把侧向速度也设成同一个值，根速度模长会变成 ``√2 · forward_speed``。
    """
    feat = np.zeros((n_frames, C.FEATURE_DIM), dtype=np.float64)
    feat[:, C.FEAT_LAYOUT["root_vel"].start] = forward_speed
    feat[:, C.FEAT_LAYOUT["root_height"]] = 91.7
    feat[:, C.FEAT_LAYOUT["heading"]] = [np.sin(yaw), np.cos(yaw)]
    feat[:, C.FEAT_LAYOUT["heading_vel"]] = 0.0
    root_tilt = np.tile(np.array([1.0, 0.0, 0.0, 0.0, 1.0, 0.0]), (n_frames, 1))
    feat[:, C.FEAT_LAYOUT["root_tilt_6d"]] = root_tilt
    joints = np.tile(np.array([1.0, 0.0, 0.0, 0.0, 1.0, 0.0]), (n_frames, C.N_JOINTS - 1, 1))
    feat[:, C.FEAT_LAYOUT["joints_6d"]] = joints.reshape(n_frames, -1)
    feat[:, C.FEAT_LAYOUT["foot_contact"]] = [1.0, 1.0]
    if offset:
        feat[:, C.FEAT_LAYOUT["root_height"]] += offset
    return feat


def test_blend_windows_shape_and_heading_norm() -> None:
    """融合后长度正确，heading 仍是单位向量，足接触仍为 0/1。"""
    windows = [_make_feature_window(yaw=0.1 * index) for index in range(3)]
    blended = blend_feature_windows(windows)
    assert blended.shape == (expected_frames(3), C.FEATURE_DIM)
    heading = blended[:, C.FEAT_LAYOUT["heading"]]
    norms = np.linalg.norm(heading, axis=1)
    assert np.allclose(norms, 1.0, atol=1e-5)
    contacts = blended[:, C.FEAT_LAYOUT["foot_contact"]]
    assert set(np.unique(contacts).tolist()) <= {0.0, 1.0}


def test_blend_single_window_passthrough() -> None:
    """单窗口时数值不应被改写（只转 dtype）。"""
    window = _make_feature_window(forward_speed=42.0)
    blended = blend_feature_windows([window])
    assert np.allclose(blended, window, atol=1e-6)


def test_blend_rejects_bad_shape() -> None:
    """形状不符合契约时报错。"""
    with pytest.raises(ValueError):
        blend_feature_windows([])
    with pytest.raises(ValueError):
        blend_feature_windows([np.zeros((150, 100))])


# ---------------------------------------------------------------------------
# 3. 根轨迹连续化（核心断言）
# ---------------------------------------------------------------------------
def _root_from_features(feature: MotionFeature) -> tuple[np.ndarray, np.ndarray]:
    """特征 → (quats, root_pos)（走生产同一条路径）。"""
    return feature.denormalize(NormStats.identity()).to_rotations_root()


def test_root_velocity_continuity_at_seams() -> None:
    """接缝处根速度必须连续：融合后的速度台阶远小于「各窗口独立积分再拼接」。

    构造让缺陷可以被真实观测：三个窗口的**前进速度不同**（100/120/140 cm/s）。
    模拟「不做全局积分」的常见错误做法——每窗各自从原点积分，再按步长 120 贴到
    总时间轴上（后窗覆盖重叠区），于是相邻窗在 ``start_i`` 处速度从 100 直接跳到
    120、再到 140（单帧 20 cm/s 的台阶）。正确做法是融合后**全局只积分一次**，
    速度在 30 帧重叠区里线性过渡。
    """
    stats = NormStats.identity()
    stride = C.WINDOW - C.OVERLAP
    windows = [
        _make_feature_window(forward_speed=100.0),
        _make_feature_window(forward_speed=120.0),
        _make_feature_window(forward_speed=140.0),
    ]
    n_windows = len(windows)
    total = expected_frames(n_windows)
    assert total == C.WINDOW + (n_windows - 1) * stride

    # (a) 错误做法：每窗独立积分 → 按链式偏移贴到总轴上（重叠区被后窗覆盖）
    naive_root = np.zeros((total, 3), dtype=np.float64)
    for index, window in enumerate(windows):
        feature = MotionFeature(
            feat=window, mask=np.ones(window.shape[0], dtype=np.uint8), fps=C.FPS
        )
        _, root = feature.to_rotations_root()
        start = index * stride
        # 链式偏移：让后窗起点与总轴已有位置对齐（位置连续，但速度不连续）
        offset = naive_root[start] - root[0] if index > 0 else np.zeros(3)
        naive_root[start : start + root.shape[0]] = root + offset
    naive_jumps = seam_velocity_jumps(
        naive_root, n_windows, fps=C.FPS, window=C.WINDOW, overlap=C.OVERLAP
    )
    assert naive_jumps.size == n_windows - 1
    # 硬拼必然在接缝处留下 20 cm/s 级的速度台阶
    assert naive_jumps.max() > 10.0, naive_jumps

    # (b) 正确做法：融合后**全局只积分一次**
    blended = blend_feature_windows(windows)
    feature = MotionFeature(
        feat=blended, mask=np.ones(blended.shape[0], dtype=np.uint8), fps=C.FPS
    ).denormalize(stats)
    _, root = feature.to_rotations_root()
    assert root.shape[0] == total
    jumps = seam_velocity_jumps(
        root, n_windows, fps=C.FPS, window=C.WINDOW, overlap=C.OVERLAP
    )
    # 速度只在重叠区线性过渡（20 cm/s / 30 帧 ≈ 0.7 cm/s 每帧）→ 台阶应极小
    assert jumps.max() < naive_jumps.max() / 5.0, (jumps, naive_jumps)
    assert jumps.max() < 5.0, jumps

    # (c) 融合方向正确：速度从 100 平滑升到 140，重叠区落在两者之间（不是突变）
    speed = np.linalg.norm(np.gradient(root[:, [0, 2]], axis=0) * C.FPS, axis=1)
    assert abs(speed[0] - 100.0) < 5.0, speed[0]
    assert abs(speed[-1] - 140.0) < 5.0, speed[-1]
    for seam_index in range(n_windows - 1):
        seam = seam_index * stride + C.WINDOW // 2
        assert 100.0 < speed[seam] < 140.0, (seam_index, speed[seam])


def test_smooth_root_track_reduces_jitter() -> None:
    """EMA 平滑降低速度抖动，``smooth=0`` 时原样返回。"""
    t = np.arange(300, dtype=np.float64)
    clean = np.stack([t * 2.0, np.full_like(t, 91.7), np.zeros_like(t)], axis=1)
    noisy = clean.copy()
    noisy[:, 0] += np.where(t % 2 == 0, 1.5, -1.5)  # 1 帧级抖动
    raw_velocity = np.gradient(noisy[:, 0]) * C.FPS
    smoothed = smooth_root_track(noisy, smooth=0.6)
    smooth_velocity = np.gradient(smoothed[:, 0]) * C.FPS
    assert smooth_velocity.std() < raw_velocity.std()
    assert np.allclose(smooth_root_track(noisy, smooth=0.0), noisy)


# ---------------------------------------------------------------------------
# 4. DDIM 采样
# ---------------------------------------------------------------------------
def _tiny_dit(cond_dim: int = 64) -> DiffusionTransformer:
    """小号 DiT（测试用，几万参数）。"""
    return DiffusionTransformer(
        latent_dim=C.LATENT_DIM, d_model=64, n_layers=1, n_heads=4, cond_dim=cond_dim, max_tokens=64
    )


def test_latent_sampler_shapes_and_determinism() -> None:
    """同 seed 逐元素一致；形状 = [1,38,128]；guidance 会改变结果。"""
    torch.manual_seed(0)
    model = _tiny_dit().eval()
    sampler = LatentSampler(model, GaussianDiffusion(timesteps=100), cond_dim=64)
    cond = torch.randn(1, 64)

    first = sampler.sample_window(cond, steps=5, guidance=3.0, seed=7)
    second = sampler.sample_window(cond, steps=5, guidance=3.0, seed=7)
    assert first.shape == (1, C.VAE_TOKEN_LEN, C.LATENT_DIM)
    assert torch.allclose(first, second)

    other_seed = sampler.sample_window(cond, steps=5, guidance=3.0, seed=8)
    assert not torch.allclose(first, other_seed)

    no_guidance = sampler.sample_window(cond, steps=5, guidance=1.0, seed=7)
    assert not torch.allclose(first, no_guidance)

    sequence = sampler.sample_sequence(cond, n_windows=3, config=SamplingConfig(steps=5))
    assert len(sequence) == 3
    assert not torch.allclose(sequence[0], sequence[1])  # seed+i


def test_latent_sampler_rejects_bad_cond() -> None:
    """条件维数不符时报错。"""
    sampler = LatentSampler(_tiny_dit(64), GaussianDiffusion(timesteps=10), cond_dim=64)
    with pytest.raises(ValueError):
        sampler.sample_window(torch.randn(1, 32), steps=2)


# ---------------------------------------------------------------------------
# 5. 端到端生成 + BVH 导出
# ---------------------------------------------------------------------------
class _StubTextEncoder:
    """固定句向量的假编码器（避免测试依赖 bge 权重）。"""

    def __init__(self, dim: int = 512) -> None:
        self.dim = dim
        self.calls: list[str] = []

    def encode(self, texts, *, batch_size: int = 16):
        vectors = []
        for text in texts:
            self.calls.append(text)
            rng = np.random.default_rng(abs(hash(text)) % (2**32))
            vectors.append(rng.normal(size=self.dim).astype(np.float32))
        return np.stack(vectors)


@pytest.fixture()
def generator() -> MotionGenerator:
    """随机初始化的小号生成器（不加载任何 ckpt）。"""
    torch.manual_seed(0)
    vae = MotionVAE(feature_dim=C.FEATURE_DIM, latent_dim=C.LATENT_DIM, hidden=32, downsample=4)
    dit = DiffusionTransformer(
        latent_dim=C.LATENT_DIM, d_model=32, n_layers=1, n_heads=4, cond_dim=32, max_tokens=64
    )
    projector = TextProjector(512, 32)
    return MotionGenerator(
        text_encoder=_StubTextEncoder(),  # type: ignore[arg-type]
        vae=vae,
        dit=dit,
        projector=projector,
        norm_stats=NormStats.identity(),
        diffusion=GaussianDiffusion(timesteps=50),
        device="cpu",
    )


def test_generate_single_window(generator: MotionGenerator) -> None:
    """1 个窗口：输出帧数 = 目标帧数，四元数与根位移形状正确。"""
    motion = generator.generate("走上楼梯", GenerateOptions(duration_s=5.0, steps=3))
    assert motion.n_windows == 1
    assert motion.quats.shape == (motion.n_frames, C.N_JOINTS, 4)
    assert motion.root_pos.shape == (motion.n_frames, 3)
    assert motion.n_frames == C.WINDOW
    norms = np.linalg.norm(motion.quats, axis=-1)
    assert np.allclose(norms, 1.0, atol=1e-4)
    assert motion.seam_jumps.size == 0
    for key in ("text_ms", "sample_ms", "decode_ms", "blend_ms", "post_ms", "total_ms"):
        assert key in motion.timing_ms


def test_generate_long_motion_windows(generator: MotionGenerator) -> None:
    """12 秒长动作：走 3 个窗口，帧数截断到目标，接缝跳变受控。"""
    motion = generator.generate("打一套陈式太极", GenerateOptions(duration_s=12.0, steps=3))
    assert motion.n_windows == 3
    assert motion.n_frames == 360
    assert motion.seam_jumps.size == 2
    assert motion.seam_jumps.max() < 400.0  # 随机初始化模型也要保证没有爆炸


def test_generate_rejects_empty_prompt(generator: MotionGenerator) -> None:
    """空指令直接报错。"""
    with pytest.raises(ValueError):
        generator.generate("   ")


def test_export_bvh_roundtrip(generator: MotionGenerator, work_dir: Path) -> None:
    """导出的 BVH 能被自家解析器读回，且关节数/帧数/首帧姿态一致。"""
    motion = generator.generate("慢慢走上楼梯", GenerateOptions(duration_s=5.0, steps=2))
    # ground_origin 关掉：本用例验证的是「raw 轨迹 → BVH → 回读」的往返保真
    path = generator.export(motion, work_dir, name="unit", ground_origin=False)
    assert path.exists() and path.suffix == ".bvh"

    document = BvhParser.parse(path)
    assert document.n_frames == motion.n_frames
    assert document.n_joints == C.N_JOINTS
    assert tuple(document.skeleton.bone_names) == tuple(C.BONE_ORDER)
    # 欧拉角往返一致（同一坐标系、同一欧拉序）
    euler_expected = motion_to_euler_deg(motion.quats)
    assert np.allclose(document.euler_deg, euler_expected, atol=1e-4)
    assert np.allclose(document.root_pos_cm, motion.root_pos, atol=1e-4)


def test_export_bvh_ground_origin_shifts_root(generator: MotionGenerator, work_dir: Path) -> None:
    """ground_origin（默认开）：BVH 的根 Y 整体平移了 offset，且爬升总量不变。"""
    from ttm.infer.generate import ground_origin_offset_cm, shift_root_y

    motion = generator.generate("慢慢走上楼梯", GenerateOptions(duration_s=5.0, steps=2))
    path = generator.export(motion, work_dir, name="unit_ground")  # 默认 ground_origin=True
    document = BvhParser.parse(path)
    offset = motion.ground_offset_cm
    expected = shift_root_y(motion.root_pos, offset)
    assert np.allclose(document.root_pos_cm, expected, atol=1e-4)
    # 首帧足触地点落到 0 附近（offset 的定义），爬升总量不变
    # （BVH 只写 4 位小数，两帧各 ±5e-5，所以爬升差比对放宽到 1e-3）
    assert offset == pytest.approx(ground_origin_offset_cm(motion.quats, motion.root_pos))
    assert document.root_pos_cm[-1, 1] - document.root_pos_cm[0, 1] == pytest.approx(
        motion.root_pos[-1, 1] - motion.root_pos[0, 1], abs=1e-3
    )


def test_bvh_export_writes_expected_channels() -> None:
    """导出的文本里根是 6 通道、其余是 3 通道（下游 UE/three.js 依赖）。"""
    quats = np.zeros((2, C.N_JOINTS, 4))
    quats[..., 3] = 1.0
    root = np.array([[0.0, 91.7, 0.0], [1.0, 91.8, 2.0]])
    lines = []
    tmp = Path("D:/TTM/.cache/tmp")
    tmp.mkdir(parents=True, exist_ok=True)
    path = write_bvh(tmp / "channels_unit.bvh", quats, root, fps=30)
    text = path.read_text(encoding="utf-8")
    lines = text.splitlines()
    assert "CHANNELS 6 Xposition Yposition Zposition Yrotation Xrotation Zrotation" in text
    assert text.count("CHANNELS 3 Yrotation Xrotation Zrotation") == C.N_JOINTS - 1
    motion_index = lines.index("MOTION")
    assert lines[motion_index + 1] == "Frames: 2"
    data_tokens = lines[motion_index + 3].split()
    assert len(data_tokens) == 6 + (C.N_JOINTS - 1) * 3
    path.unlink()


def test_six_d_pipeline_preserves_rotation() -> None:
    """融合路径里的 6D → 矩阵仍是合法旋转（防止生成出镜像骨）。"""
    blended = blend_feature_windows([_make_feature_window() for _ in range(2)])
    joints = blended[:, C.FEAT_LAYOUT["joints_6d"]].reshape(-1, C.N_JOINTS - 1, 6)
    matrices = six_d_to_rotmat(joints)
    dets = np.linalg.det(matrices)
    assert np.allclose(dets, 1.0, atol=1e-6)
