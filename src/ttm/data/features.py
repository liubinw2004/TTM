"""运动特征提取（**146 维**）与归一化。

布局见 :data:`ttm.common.constants.FEAT_LAYOUT`（``layout_version = 1``）：

===========  ====  ==================================================
维度区间      维数  内容
===========  ====  ==================================================
[0, 2)         2   根水平速度 (前向, 侧向)，heading 系，cm/s，z-score
[2, 3)         1   根高度 y（骨盆世界高度），cm，z-score
[3, 5)         2   朝向 (sin θ, cos θ)，θ = atan2(前向.x, 前向.z)
[5, 6)         1   偏航角速度 ω，rad/s，z-score
[6, 12)        6   根倾斜 6D：R_yaw⁻¹ · R_root 的前两行
[12, 144)    132   22 个非根关节 × 6D 局部旋转（顺序 = ``BONE_ORDER[1:]``）
[144, 146)     2   足接触 (contact_L, contact_R) ∈ {0,1}
===========  ====  ==================================================

坐标系与旋转约定：全部在 **BVH 系（Y-up / 右手 / cm）** 内计算；旋转内部用
**6D**（旋转矩阵前两行，紧凑且连续，避免四元数双覆盖与欧拉角回绕）。

归一化：**只**对 :data:`ttm.common.constants.NORM_DIMS` = ``(0,1,2,5)``
四个连续维做 z-score（统计只在 train split 上算），其余维 mean=0/std=1 恒等。
"""

from __future__ import annotations

import os
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Sequence

import numpy as np
from scipy.spatial.transform import Rotation

from ttm import LAYOUT_VERSION
from ttm.common import constants as C
from ttm.common import paths as P
from ttm.data.skeleton import SkeletonDef

__all__ = [
    "FeatureExtractor",
    "MotionFeature",
    "NormStats",
    "count_windows",
    "motion_quality_issue",
    "quat_to_six_d",
    "rotmat_to_6d",
    "six_d_to_quat",
    "six_d_to_rotmat",
    "slice_windows_arrays",
    "trim_placeholder_frames",
]

#: 前向轴（根局部坐标系内）。``θ = atan2(前向.x, 前向.z)`` 与 heading 通道一致。
FORWARD_AXIS: np.ndarray = np.array([0.0, 0.0, 1.0], dtype=np.float64)

#: 「非静止」判据：全片段关节旋转极差下限（度）。实测坏样本为 0.000°。
MOTION_MIN_ROTATION_RANGE_DEG: float = 0.5

#: 足部最低高度阈值（cm）。站立时脚底约 0–12cm，占位数据的 -85cm 属明显损坏。
MOTION_MIN_FOOT_HEIGHT_CM: float = -10.0

#: 6D → 矩阵时判定「向量退化」的范数阈值。
_SIX_D_EPSILON: float = 1e-9

#: 6D 退化时用于重建正交基的参考轴。
_AXIS_X: np.ndarray = np.array([1.0, 0.0, 0.0], dtype=np.float64)
_AXIS_Y: np.ndarray = np.array([0.0, 1.0, 0.0], dtype=np.float64)


# ---------------------------------------------------------------------------
# 窗口切分规则（**唯一实现处**，scan / preprocess / dataset 全部复用）
# ---------------------------------------------------------------------------
def trim_placeholder_frames(
    euler_deg: np.ndarray,
    root_pos_cm: np.ndarray,
    *,
    zero_tolerance: float = 1e-6,
) -> tuple[np.ndarray, np.ndarray, int]:
    """裁掉开头的「T-pose 占位帧」。

    实测（2026-10 快照）：``Z:/1000hours/BVH_Body`` 里大量片段的首帧是建模工具
    导出的默认姿态——**所有关节旋转恒为 0、根关节落在原点附近**，与第 2 帧的真实
    动作之间存在上百 cm 的瞬移。若直接提特征，会在起点产生虚假的巨大根速度。

    规则：从第 0 帧起，凡是「所有关节旋转都精确为 0」的连续帧都丢掉；但至少保留
    1 帧（整段都是 0 旋转的片段交给 :func:`motion_quality_issue` 判退化后淘汰）。

    Args:
        euler_deg: ``[T,23,3]`` 欧拉角（角度制）。
        root_pos_cm: ``[T,3]`` 根位移。
        zero_tolerance: 判定「旋转为 0」的容差（度）。

    Returns:
        ``(euler_deg_trimmed, root_pos_cm_trimmed, n_trimmed)``。
    """
    euler_deg = np.asarray(euler_deg, dtype=np.float64)
    root_pos_cm = np.asarray(root_pos_cm, dtype=np.float64)
    if euler_deg.ndim != 3 or root_pos_cm.ndim != 2 or euler_deg.shape[0] != root_pos_cm.shape[0]:
        raise ValueError(
            f"trim_placeholder_frames 形状不匹配: euler={euler_deg.shape}, root={root_pos_cm.shape}"
        )
    frames = euler_deg.shape[0]
    if frames <= 1:
        return euler_deg, root_pos_cm, 0
    is_tpose = np.all(np.abs(euler_deg) <= zero_tolerance, axis=(1, 2))
    lead = 0
    while lead < frames - 1 and is_tpose[lead]:
        lead += 1
    if lead == 0:
        return euler_deg, root_pos_cm, 0
    return euler_deg[lead:], root_pos_cm[lead:], lead


def motion_quality_issue(
    skeleton: SkeletonDef,
    euler_deg: np.ndarray,
    root_pos_cm: np.ndarray,
    *,
    min_rotation_range_deg: float = MOTION_MIN_ROTATION_RANGE_DEG,
    min_foot_height_cm: float = MOTION_MIN_FOOT_HEIGHT_CM,
) -> str | None:
    """检查一段动作是否「可训练」，返回淘汰原因（``None`` 表示通过）。

    三条实测有效的红线：

    1. **数值有效**：存在 NaN/Inf 直接淘汰；
    2. **非静止**：全片段关节旋转极差 < ``min_rotation_range_deg`` 视为坏样本
       （实测 ``3.3.1.1.Dribbling_Ball/Take_001..005`` 整段都是同一帧的姿态占位数据）；
    3. **人体可信**：``min(左脚,右脚)`` 世界高度的中位数低于 ``min_foot_height_cm``
       说明脚陷到地面以下，物理上不可能（同批占位数据的根高度只有 2.28cm，
       脚在 -85cm）。

    Args:
        skeleton: 骨架定义。
        euler_deg: ``[T,23,3]`` 欧拉角（角度制）。
        root_pos_cm: ``[T,3]`` 根位移。
        min_rotation_range_deg: 「非静止」判据（度）。
        min_foot_height_cm: 足部最低高度阈值（cm）。

    Returns:
        淘汰原因字符串，或 ``None``。
    """
    euler_deg = np.asarray(euler_deg, dtype=np.float64)
    root_pos_cm = np.asarray(root_pos_cm, dtype=np.float64)
    if not np.all(np.isfinite(euler_deg)) or not np.all(np.isfinite(root_pos_cm)):
        return "degenerate: 含 NaN/Inf"
    if euler_deg.shape[0] < 2:
        return "degenerate: 帧数不足 2"

    rotation_range = float(np.ptp(euler_deg, axis=0).max())
    if rotation_range < min_rotation_range_deg:
        return f"degenerate: 静止片段（关节旋转极差 {rotation_range:.3f}° < {min_rotation_range_deg}°）"

    world = skeleton.fk(np.deg2rad(euler_deg), root_pos_cm)
    feet = world[:, list(C.FOOT_CONTACT_JOINTS), 1]  # [T,2]
    median_foot = float(np.median(feet.min(axis=1)))
    if median_foot < min_foot_height_cm:
        return (
            f"degenerate: 足部低于地面（中位 {median_foot:.1f}cm < {min_foot_height_cm}cm，"
            "根高度约定可能损坏）"
        )
    return None


def window_stride_for(
    n_frames: int,
    *,
    window: int = C.WINDOW,
    long_clip_stride: int = 0,
) -> int:
    """给出某个 clip 的**窗口步长**（非重叠 150，超长 clip 降密度）。

    为什么要降密度：``1.12.Action_Sequence_Complex_Behavior`` 的长链任务单条能到
    2850 帧（95 秒，19 个窗口），而 ``3.2`` 的短片段只有 300 帧（2 个窗口）。
    如果一律按 150 非重叠切，配额会被长套路吃光——实测旧版 ``1.12`` 一个类占掉
    19193/22682 = 85% 的窗口。超长 clip 用更大的步长抽样，保留首尾与中段覆盖，
    但不按帧数线性吃预算。

    阈值取 ``2 * long_clip_stride``：默认 450 → 超过 900 帧（30 秒 @30fps）才降密度。

    Args:
        n_frames: 30fps 下的帧数。
        window: 窗口帧数。
        long_clip_stride: 超长 clip 的窗口步长；``0`` 表示不降密度。

    Returns:
        窗口步长（帧），恒 ``>= window``。
    """
    base = max(1, int(window))
    long_stride = int(long_clip_stride)
    if long_stride <= 0 or n_frames <= 2 * long_stride:
        return base
    return max(base, long_stride)


def count_windows(
    n_frames: int,
    window: int = C.WINDOW,
    min_tail: int = C.MIN_TAIL_FRAMES,
    stride: int | None = None,
) -> int:
    """按 ``TTM-ARCH.md`` §5.5 规则计算一个 clip 能切出多少训练窗口。

    规则：源 clip 重采样到 30fps 后，按步长 ``stride`` 切整窗（默认
    ``stride = window``，即非重叠）；最后一个整窗之后剩余帧 ``>= min_tail``
    时再保留一个 pad+mask 窗，否则丢弃。

    Args:
        n_frames: 30fps 下的帧数。
        window: 窗口帧数（默认 150）。
        min_tail: 尾窗保留阈值（默认 75）。
        stride: 窗口步长；``None`` 等价于 ``window``（非重叠）。

    Returns:
        窗口数（可能为 0，表示该 clip 太短无法入训）。

    Examples:
        >>> count_windows(150)
        1
        >>> count_windows(360)  # 2 个整窗 + 60 帧余量（< 75）丢弃
        2
        >>> count_windows(900, window=150, stride=450)  # 超长 clip 降密度
        3
    """
    if n_frames <= 0 or window <= 0:
        return 0
    step = max(1, int(stride) if stride else int(window))
    if n_frames < window:
        return 1 if n_frames >= min_tail else 0
    full = (int(n_frames) - int(window)) // step + 1
    last_start = (full - 1) * step
    tail = int(n_frames) - (last_start + int(window))
    return full + (1 if tail >= min_tail else 0)


def slice_windows_arrays(
    feat: np.ndarray,
    mask: np.ndarray,
    *,
    window: int = C.WINDOW,
    min_tail: int = C.MIN_TAIL_FRAMES,
    stride: int | None = None,
) -> list[tuple[np.ndarray, np.ndarray, int]]:
    """把 ``[T,146]`` 特征按 §5.5 规则切成 ``window`` 长的窗口。

    Args:
        feat: ``[T, 146]`` 特征数组。
        mask: ``[T]`` 有效帧掩码。
        window: 窗口帧数。
        min_tail: 尾窗保留阈值。
        stride: 窗口步长；``None`` 等价于 ``window``（非重叠）。

    Returns:
        ``[(feat_win[window,146], mask_win[window], win_idx), ...]``；
        不足 ``window`` 的尾窗用 0 padding 且 ``mask`` 对应位置置 0。

    Note:
        **尾窗起点必须夹到** ``frames - window``：直接用 ``win_idx * step`` 会在
        ``step > 剩余帧`` 时越界，实测制造过两类坏数据：

        * 265 个窗口（13696 的 1.9%）**mask 全 0**——起点 1350 而 clip 只有 1209 帧，
          整个窗口全是 padding。它们对重建项贡献 0，却照样被编码器编码、照样算 KL，
          是纯噪声样本；
        * 超长 clip（> 900 帧 → step=450）的尾窗被推到 ``last_start + 450``，
          而它前面那段（``last_start+window`` 到 ``last_start+450``，约 150–300 帧）
          **从来没被任何窗口覆盖**——静默丢数据，比多算一个窗口严重得多。

        夹到 ``frames - window`` 之后：尾窗总是「clip 的最后 150 帧」，一个窗口都不丢，
        也不产生全 padding 窗口。只有 ``frames < window`` 的极短 clip 才走 pad 路径。
    """
    feat = np.asarray(feat)
    mask = np.asarray(mask)
    if feat.ndim != 2 or feat.shape[1] != C.FEATURE_DIM:
        raise ValueError(f"slice_windows_arrays 需要 [T,{C.FEATURE_DIM}]，实际 {feat.shape}")
    frames = feat.shape[0]
    step = max(1, int(stride) if stride else int(window))
    last_start = max(0, frames - int(window))
    windows: list[tuple[np.ndarray, np.ndarray, int]] = []
    for win_idx in range(count_windows(frames, window=window, min_tail=min_tail, stride=step)):
        cursor = max(0, min(win_idx * step, last_start))
        chunk = feat[cursor : cursor + window]
        chunk_mask = mask[cursor : cursor + window]
        actual = chunk.shape[0]
        feat_win = np.zeros((window, C.FEATURE_DIM), dtype=feat.dtype)
        mask_win = np.zeros(window, dtype=np.uint8)
        feat_win[:actual] = chunk
        mask_win[:actual] = chunk_mask
        windows.append((feat_win, mask_win, win_idx))
    return windows


# ---------------------------------------------------------------------------
# 6D 旋转表示工具（Zhou et al. 2019）
# ---------------------------------------------------------------------------
def rotmat_to_6d(mats: np.ndarray) -> np.ndarray:
    """把旋转矩阵转成 6D（矩阵前两行按行拼接）。

    Args:
        mats: ``[..., 3, 3]`` 旋转矩阵。

    Returns:
        ``[..., 6]`` float64。
    """
    array = np.asarray(mats, dtype=np.float64)
    if array.shape[-2:] != (3, 3):
        raise ValueError(f"rotmat_to_6d 需要 [...,3,3]，实际 {array.shape}")
    return np.concatenate([array[..., 0, :], array[..., 1, :]], axis=-1)


def six_d_to_rotmat(six_d: np.ndarray) -> np.ndarray:
    """把 6D 表示还原为旋转矩阵（对数值噪声做 Gram-Schmidt 正交化）。

    数值鲁棒性：扩散/VAE 在训练初期会输出**接近 0** 的 6D（实测 batch 内出现过
    全 0 向量）。此时 Gram-Schmidt 会得到全 0 矩阵，``scipy`` 转四元数时会抛
    ``Non-positive determinant``。这里对退化情形退化为单位旋转的前两行，保证
    永远返回合法旋转矩阵（单位阵），使指标计算不会中断训练。

    Args:
        six_d: ``[..., 6]``。

    Returns:
        ``[..., 3, 3]`` float64，行向量正交归一（退化输入 → 单位阵）。
    """
    array = np.asarray(six_d, dtype=np.float64)
    if array.shape[-1] != 6:
        raise ValueError(f"six_d_to_rotmat 需要 [...,6]，实际 {array.shape}")
    first = array[..., :3]
    second = array[..., 3:]

    first_norm = np.linalg.norm(first, axis=-1, keepdims=True)
    degenerate_first = first_norm < _SIX_D_EPSILON
    unit_x = np.zeros_like(first)
    unit_x[..., 0] = 1.0
    first = np.where(degenerate_first, unit_x, first / np.clip(first_norm, 1e-12, None))

    second = second - np.sum(first * second, axis=-1, keepdims=True) * first
    second_norm = np.linalg.norm(second, axis=-1, keepdims=True)
    degenerate_second = second_norm < _SIX_D_EPSILON

    # 退化时取一个与 first 不共线的参考轴再正交化，得到合法的第二行
    ref_axis = np.where(np.abs(first[..., 1:2]) < 0.9, _AXIS_Y, _AXIS_X)
    fallback = ref_axis - np.sum(first * ref_axis, axis=-1, keepdims=True) * first
    fallback = fallback / np.clip(
        np.linalg.norm(fallback, axis=-1, keepdims=True), 1e-12, None
    )
    second = np.where(degenerate_second, fallback, second / np.clip(second_norm, 1e-12, None))

    third = np.cross(first, second)
    return np.stack([first, second, third], axis=-2)


def six_d_to_quat(six_d: np.ndarray) -> np.ndarray:
    """6D → 四元数 ``xyzw``。

    Args:
        six_d: ``[..., 6]``。

    Returns:
        ``[..., 4]`` float64（``xyzw``，与 scipy 一致）。
    """
    matrices = six_d_to_rotmat(six_d)
    flat = matrices.reshape(-1, 3, 3)
    quats = Rotation.from_matrix(flat).as_quat()
    return quats.reshape(*matrices.shape[:-2], 4)


def quat_to_six_d(quat: np.ndarray) -> np.ndarray:
    """四元数 ``xyzw`` → 6D。

    Args:
        quat: ``[..., 4]``。

    Returns:
        ``[..., 6]`` float64。
    """
    array = np.asarray(quat, dtype=np.float64)
    if array.shape[-1] != 4:
        raise ValueError(f"quat_to_six_d 需要 [...,4]，实际 {array.shape}")
    flat = array.reshape(-1, 4)
    matrices = Rotation.from_quat(flat).as_matrix()
    return rotmat_to_6d(matrices.reshape(*array.shape[:-1], 3, 3))


def _yaw_rotation(yaw: np.ndarray) -> np.ndarray:
    """构造绕 Y 轴旋转 ``yaw`` 的旋转矩阵。

    Args:
        yaw: ``[..., ]`` 弧度。

    Returns:
        ``[..., 3, 3]`` 旋转矩阵。
    """
    angles = np.asarray(yaw, dtype=np.float64)
    cos = np.cos(angles)
    sin = np.sin(angles)
    zeros = np.zeros_like(cos)
    ones = np.ones_like(cos)
    row0 = np.stack([cos, zeros, sin], axis=-1)
    row1 = np.stack([zeros, ones, zeros], axis=-1)
    row2 = np.stack([-sin, zeros, cos], axis=-1)
    return np.stack([row0, row1, row2], axis=-2)


# ---------------------------------------------------------------------------
# 归一化统计
# ---------------------------------------------------------------------------
@dataclass(frozen=True)
class NormStats:
    """z-score 统计量（``norm_stats.npz`` 的内存表示）。

    Attributes:
        mean: ``[146]`` float64，未归一化维恒为 0。
        std: ``[146]`` float64，未归一化维恒为 1（避免除零）。
        layout_version: 特征布局版本。
        count: 参与统计的帧数（不落盘，仅日志/测试用）。
    """

    mean: np.ndarray
    std: np.ndarray
    layout_version: int = LAYOUT_VERSION
    count: int = 0

    def __post_init__(self) -> None:
        """校验形状并强制未归一化维为恒等值。"""
        mean = np.asarray(self.mean, dtype=np.float64).reshape(-1)
        std = np.asarray(self.std, dtype=np.float64).reshape(-1)
        if mean.shape != (C.FEATURE_DIM,) or std.shape != (C.FEATURE_DIM,):
            raise ValueError(
                f"mean/std 形状应为 ({C.FEATURE_DIM},)，实际 {mean.shape}/{std.shape}"
            )
        if not np.all(np.isfinite(mean)) or not np.all(np.isfinite(std)):
            raise ValueError("mean/std 含 NaN/Inf")
        if np.any(std <= 0):
            raise ValueError("std 必须为正（未归一化维应为 1.0）")
        object.__setattr__(self, "mean", mean)
        object.__setattr__(self, "std", std)

    # -- 构造 ---------------------------------------------------------------
    @classmethod
    def identity(cls) -> "NormStats":
        """返回恒等统计量（不改变数值）。"""
        return cls(
            mean=np.zeros(C.FEATURE_DIM, dtype=np.float64),
            std=np.ones(C.FEATURE_DIM, dtype=np.float64),
        )

    @classmethod
    def from_features(cls, features: Sequence[np.ndarray]) -> "NormStats":
        """在给定特征（应仅来自 train split）上计算统计量。

        Args:
            features: ``[T,146]`` 数组序列。

        Returns:
            :class:`NormStats`；只对 ``NORM_DIMS`` 计算真实统计，其余维为恒等。
        """
        mean = np.zeros(C.FEATURE_DIM, dtype=np.float64)
        std = np.ones(C.FEATURE_DIM, dtype=np.float64)
        if not features:
            return cls(mean=mean, std=std, count=0)

        stacked = np.concatenate(
            [np.asarray(item, dtype=np.float64).reshape(-1, C.FEATURE_DIM) for item in features],
            axis=0,
        )
        if stacked.shape[0] == 0:
            return cls(mean=mean, std=std, count=0)
        for dim in C.NORM_DIMS:
            column = stacked[:, dim]
            mean[dim] = float(column.mean())
            std[dim] = float(column.std())
        # 防除零：方差极小的维度退化为 1
        for dim in C.NORM_DIMS:
            if std[dim] < 1e-6:
                std[dim] = 1.0
        return cls(mean=mean, std=std, count=int(stacked.shape[0]))

    # -- 读写 ---------------------------------------------------------------
    def save(self, path: str | os.PathLike[str]) -> Path:
        """保存为 ``norm_stats.npz``（经写盘守卫）。

        字段严格为 ``mean`` / ``std`` / ``layout_version``（ARCH §5.3）。

        Args:
            path: 目标路径。

        Returns:
            实际写出的路径。
        """
        target = P.resolve_out(path, mkparents=True)
        np.savez(
            target,
            mean=self.mean.astype(np.float64),
            std=self.std.astype(np.float64),
            layout_version=np.int32(self.layout_version),
        )
        return target

    @classmethod
    def load(cls, path: str | os.PathLike[str]) -> "NormStats":
        """从 ``norm_stats.npz`` 读取统计量。

        Args:
            path: 文件路径。

        Returns:
            :class:`NormStats`。

        Raises:
            FileNotFoundError: 文件不存在。
            ValueError: 缺字段或 layout_version 不匹配。
        """
        source = Path(path)
        if not source.is_file():
            raise FileNotFoundError(f"norm_stats 不存在: {source}")
        with np.load(source) as payload:
            for key in ("mean", "std", "layout_version"):
                if key not in payload:
                    raise ValueError(f"norm_stats 缺少字段 {key!r}: {source}")
            mean = np.asarray(payload["mean"], dtype=np.float64)
            std = np.asarray(payload["std"], dtype=np.float64)
            version = int(np.asarray(payload["layout_version"]).reshape(-1)[0])
        if version != LAYOUT_VERSION:
            raise ValueError(
                f"layout_version 不匹配：文件 {version} != 代码 {LAYOUT_VERSION}（需重新预处理）"
            )
        return cls(mean=mean, std=std, layout_version=version)

    def to_dict(self) -> dict[str, Any]:
        """返回统计摘要（日志用）。"""
        return {
            "layout_version": self.layout_version,
            "count": self.count,
            "mean": {int(dim): float(self.mean[dim]) for dim in C.NORM_DIMS},
            "std": {int(dim): float(self.std[dim]) for dim in C.NORM_DIMS},
        }


# ---------------------------------------------------------------------------
# 运动特征
# ---------------------------------------------------------------------------
@dataclass(frozen=True)
class MotionFeature:
    """一段动作的 146 维特征序列。

    Attributes:
        feat: ``[T,146]`` 特征数组（dtype 由调用方决定，预处理产物为 float32）。
        mask: ``[T]`` 有效帧掩码（1 = 有效，0 = padding）。
        fps: 帧率（本项目固定 30）。
        layout_version: 布局版本。
    """

    feat: np.ndarray
    mask: np.ndarray
    fps: int = C.FPS
    layout_version: int = LAYOUT_VERSION

    def __post_init__(self) -> None:
        """校验形状与布局版本。"""
        feat = np.asarray(self.feat)
        mask = np.asarray(self.mask)
        if feat.ndim != 2 or feat.shape[1] != C.FEATURE_DIM:
            raise ValueError(f"feat 形状应为 [T,{C.FEATURE_DIM}]，实际 {feat.shape}")
        if mask.shape != (feat.shape[0],):
            raise ValueError(f"mask 形状应为 ({feat.shape[0]},)，实际 {mask.shape}")
        if self.layout_version != LAYOUT_VERSION:
            raise ValueError(
                f"layout_version 不匹配：{self.layout_version} != {LAYOUT_VERSION}"
            )
        object.__setattr__(self, "feat", feat)
        object.__setattr__(self, "mask", mask.astype(np.uint8, copy=False))

    # -- 基本属性 -----------------------------------------------------------
    @property
    def n_frames(self) -> int:
        """帧数。"""
        return int(self.feat.shape[0])

    @property
    def duration_s(self) -> float:
        """时长（秒）。"""
        return float(self.n_frames) / float(self.fps) if self.fps else 0.0

    def slice(self, start: int, stop: int) -> "MotionFeature":
        """按帧切片（保持 dtype）。

        Args:
            start: 起始帧（含）。
            stop: 结束帧（不含）。

        Returns:
            新的 :class:`MotionFeature`。
        """
        return MotionFeature(
            feat=self.feat[start:stop],
            mask=self.mask[start:stop],
            fps=self.fps,
            layout_version=self.layout_version,
        )

    # -- 归一化 -------------------------------------------------------------
    def normalize(self, stats: NormStats) -> "MotionFeature":
        """按统计量做 z-score 归一化（只作用于 ``NORM_DIMS``）。

        Args:
            stats: 统计量。

        Returns:
            归一化后的新对象（dtype 与输入一致）。
        """
        dtype = self.feat.dtype
        data = self.feat.astype(np.float64, copy=True)
        for dim in C.NORM_DIMS:
            data[:, dim] = (data[:, dim] - stats.mean[dim]) / stats.std[dim]
        return MotionFeature(
            feat=data.astype(dtype),
            mask=self.mask.copy(),
            fps=self.fps,
            layout_version=self.layout_version,
        )

    def denormalize(self, stats: NormStats) -> "MotionFeature":
        """归一化的逆运算。

        Args:
            stats: 统计量。

        Returns:
            反归一化后的新对象。
        """
        dtype = self.feat.dtype
        data = self.feat.astype(np.float64, copy=True)
        for dim in C.NORM_DIMS:
            data[:, dim] = data[:, dim] * stats.std[dim] + stats.mean[dim]
        return MotionFeature(
            feat=data.astype(dtype),
            mask=self.mask.copy(),
            fps=self.fps,
            layout_version=self.layout_version,
        )

    # -- 还原为旋转 ---------------------------------------------------------
    def to_rotations_root(self) -> tuple[np.ndarray, np.ndarray]:
        """把特征还原为 23 关节四元数与根位移。

        ``to_rotations_root()`` 语义（ARCH 类图）：返回 ``(quat[F,23,4] xyzw,
        root_pos[F,3])``。注意特征里**不含根的水平绝对位置**（用 heading 系速度
        表示，刻意做成平移不变），因此这里通过积分还原 XZ 轨迹，起点为 0；
        绝对摆放由消费端（重定向/导出）决定。

        Returns:
            ``(quat, root_pos_cm)``。
        """
        sin_theta = self.feat[:, C.FEAT_LAYOUT["heading"].start].astype(np.float64)
        cos_theta = self.feat[:, C.FEAT_LAYOUT["heading"].start + 1].astype(np.float64)
        yaw = np.arctan2(sin_theta, cos_theta)

        root_tilt = six_d_to_rotmat(self.feat[:, C.FEAT_LAYOUT["root_tilt_6d"]])
        root_rot = _yaw_rotation(yaw) @ root_tilt

        joints_6d = self.feat[:, C.FEAT_LAYOUT["joints_6d"]].astype(np.float64)
        joints = six_d_to_rotmat(joints_6d.reshape(self.n_frames, C.N_JOINTS - 1, 6))
        matrices = np.concatenate([root_rot[:, None, :, :], joints], axis=1)  # [T,23,3,3]
        quats = Rotation.from_matrix(matrices.reshape(-1, 3, 3)).as_quat()
        quats = quats.reshape(self.n_frames, C.N_JOINTS, 4)

        # 根位移：y 直接来自特征；XZ 由「heading 系速度」积分还原
        height = self.feat[:, C.FEAT_LAYOUT["root_height"].start].astype(np.float64)
        vel_forward = self.feat[:, C.FEAT_LAYOUT["root_vel"].start].astype(np.float64)
        vel_side = self.feat[:, C.FEAT_LAYOUT["root_vel"].start + 1].astype(np.float64)
        vel_x = vel_forward * sin_theta + vel_side * cos_theta
        vel_z = vel_forward * cos_theta - vel_side * sin_theta
        dt = 1.0 / float(self.fps) if self.fps else 1.0 / C.FPS
        pos_x = np.cumsum(vel_x) * dt
        pos_z = np.cumsum(vel_z) * dt
        root_pos = np.stack([pos_x, height, pos_z], axis=1)
        return quats, root_pos

    # -- 工具 ---------------------------------------------------------------
    def to_float32(self) -> "MotionFeature":
        """返回 float32 版本（写入 npz 分片前的统一转换）。"""
        return MotionFeature(
            feat=self.feat.astype(np.float32),
            mask=self.mask.copy(),
            fps=self.fps,
            layout_version=self.layout_version,
        )

    def segment(self, name: str) -> np.ndarray:
        """取出某个命名段的子数组。

        Args:
            name: :data:`ttm.common.constants.FEAT_LAYOUT` 的键。

        Returns:
            ``[T, k]`` 子数组。
        """
        return self.feat[:, C.feat_slice(name)]


# ---------------------------------------------------------------------------
# 特征提取器
# ---------------------------------------------------------------------------
@dataclass(frozen=True)
class FeatureExtractor:
    """把「欧拉角 + 根位移」转成 146 维特征。"""

    skeleton: SkeletonDef
    fps: int = C.FPS
    contact_height_cm: float = C.CONTACT_HEIGHT_CM
    contact_speed_cmps: float = C.CONTACT_SPEED_CMPS

    def extract(
        self,
        euler_deg: np.ndarray,
        root_pos_cm: np.ndarray,
        *,
        dtype: Any = np.float32,
    ) -> MotionFeature:
        """提取完整 clip 的特征。

        Args:
            euler_deg: ``[T,23,3]`` 角度制局部欧拉角（YXZ）。
            root_pos_cm: ``[T,3]`` 根关节世界位移（cm）。
            dtype: 输出 dtype（预处理用 float32）。

        Returns:
            :class:`MotionFeature`，``mask`` 全为 1。
        """
        angles = np.asarray(euler_deg, dtype=np.float64)
        root = np.asarray(root_pos_cm, dtype=np.float64)
        if angles.ndim != 3 or angles.shape[1:] != (C.N_JOINTS, 3):
            raise ValueError(f"euler_deg 形状应为 [T,{C.N_JOINTS},3]，实际 {angles.shape}")
        if root.shape != (angles.shape[0], 3):
            raise ValueError(f"root_pos_cm 形状应为 [{angles.shape[0]},3]，实际 {root.shape}")
        frames = angles.shape[0]
        if frames == 0:
            return MotionFeature(
                feat=np.zeros((0, C.FEATURE_DIM), dtype=dtype),
                mask=np.zeros(0, dtype=np.uint8),
                fps=self.fps,
            )

        euler_rad = np.deg2rad(angles)
        local = self.skeleton.local_rotmats(euler_rad)  # [T,23,3,3]
        root_rot = local[:, 0]

        # --- 朝向 θ = atan2(前向.x, 前向.z)
        forward = root_rot @ FORWARD_AXIS
        yaw = np.arctan2(forward[:, 0], forward[:, 2])
        sin_theta = np.sin(yaw)
        cos_theta = np.cos(yaw)

        # --- 根水平速度（heading 系：前向 / 侧向），cm/s
        dt = 1.0 / float(self.fps)
        if frames >= 2:
            velocity = np.gradient(root[:, [0, 2]], dt, axis=0)  # [T,2]
            yaw_unwrapped = np.unwrap(yaw)
            omega = np.gradient(yaw_unwrapped, dt)
        else:
            velocity = np.zeros((frames, 2), dtype=np.float64)
            omega = np.zeros(frames, dtype=np.float64)
        vel_forward = velocity[:, 0] * sin_theta + velocity[:, 1] * cos_theta
        vel_side = velocity[:, 0] * cos_theta - velocity[:, 1] * sin_theta

        # --- 偏航角速度（rad/s，先去回绕）
        #   （omega 已在上面帧数不足时置零）

        # --- 根倾斜 6D：R_yaw⁻¹ · R_root
        tilt = np.transpose(_yaw_rotation(yaw), (0, 2, 1)) @ root_rot
        tilt_6d = rotmat_to_6d(tilt)

        # --- 22 个非根关节 6D
        joints_6d = rotmat_to_6d(local[:, 1:])  # [T,22,6]

        # --- 足接触
        world = self.skeleton.fk(euler_rad, root)
        contacts = self.skeleton.foot_contacts(
            world,
            fps=self.fps,
            height_cm=self.contact_height_cm,
            speed_cmps=self.contact_speed_cmps,
        )

        feat = np.zeros((frames, C.FEATURE_DIM), dtype=np.float64)
        feat[:, C.FEAT_LAYOUT["root_vel"]] = np.stack([vel_forward, vel_side], axis=1)
        feat[:, C.FEAT_LAYOUT["root_height"]] = root[:, 1:2]
        feat[:, C.FEAT_LAYOUT["heading"]] = np.stack([sin_theta, cos_theta], axis=1)
        feat[:, C.FEAT_LAYOUT["heading_vel"]] = omega[:, None]
        feat[:, C.FEAT_LAYOUT["root_tilt_6d"]] = tilt_6d
        feat[:, C.FEAT_LAYOUT["joints_6d"]] = joints_6d.reshape(frames, -1)
        feat[:, C.FEAT_LAYOUT["foot_contact"]] = contacts

        return MotionFeature(
            feat=feat.astype(dtype),
            mask=np.ones(frames, dtype=np.uint8),
            fps=self.fps,
        )

    # -- 窗口切分 -----------------------------------------------------------
    def slice_windows(
        self,
        feature: MotionFeature,
        *,
        window: int = C.WINDOW,
        min_tail: int = C.MIN_TAIL_FRAMES,
        stride: int | None = None,
    ) -> list[tuple[np.ndarray, np.ndarray, int]]:
        """按 §5.5 规则把 clip 切成训练窗口。

        Args:
            feature: 整段特征。
            window: 窗口帧数。
            min_tail: 尾窗保留阈值。
            stride: 窗口步长；``None`` 等价于 ``window``（非重叠）。

        Returns:
            ``[(feat_win[window,146], mask_win[window], win_idx), ...]``；
            不足 ``window`` 的尾窗用 0 padding 且 ``mask=0``。
        """
        frames = feature.n_frames
        if frames <= 0 or window <= 0:
            return []
        return slice_windows_arrays(
            feature.feat, feature.mask, window=window, min_tail=min_tail, stride=stride
        )

    def window_count(
        self, n_frames: int, *, window: int = C.WINDOW, min_tail: int = C.MIN_TAIL_FRAMES
    ) -> int:
        """返回窗口数（不实际切片，扫描阶段用）。

        Args:
            n_frames: 帧数。
            window: 窗口帧数。
            min_tail: 尾窗保留阈值。

        Returns:
            窗口数。
        """
        return count_windows(n_frames, window=window, min_tail=min_tail)
