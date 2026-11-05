"""骨架常量与运动特征布局 —— **全项目的数据契约唯一实现处**。

本模块严格实现 ``TTM-ARCH.md`` §5.2 与 §5.3：

* :data:`BONE_ORDER` / :data:`PARENTS` / :data:`DEFAULT_REST_OFFSETS_CM` — 23 关节骨架；
* :data:`UE_BONE_MAP` / :data:`UE_BONE_ORDER` — BVH → UE 骨名映射（含 Spine3 降级策略）；
* :data:`FEAT_LAYOUT` — **146 维**运动特征的分段切片（``layout_version = 1``）；
* 帧率 / 窗口 / latents / 单位 / 欧拉序等全局常量。

约定（不得违反）：

* 内部坐标系 **BVH 系：Y-up、右手、单位 cm**；
* 旋转内部用 **6D**（旋转矩阵前两行），导出用四元数 **xyzw**；
* BVH 欧拉序 **YXZ**、角度制，仅在解析边界换算一次；
* 帧率统一 **30 fps**（源数据 120 fps，4× 精确整除）。

任何布局改动都必须递增 :data:`ttm.LAYOUT_VERSION`。
"""

from __future__ import annotations

import math
from typing import Final

import numpy as np

__all__ = [
    "BONE_ORDER",
    "BONE_PARENT_NAMES",
    "CLIP_ID_MAXLEN",
    "CONTACT_HEIGHT_CM",
    "CONTACT_HYSTERESIS_FRAMES",
    "CONTACT_SPEED_CMPS",
    "DEFAULT_REST_OFFSETS_CM",
    "DOWNSAMPLE",
    "EULER_ORDER",
    "FEATURE_DIM",
    "FEAT_LAYOUT",
    "FOOT_CONTACT_JOINTS",
    "FPS",
    "HEADING_DIMS",
    "LATENT_DIM",
    "MIN_TAIL_FRAMES",
    "MOTION_JSON_FORMAT",
    "NORM_DIMS",
    "N_JOINTS",
    "NON_ROOT_JOINTS",
    "N_NON_ROOT_JOINTS",
    "OVERLAP",
    "PARENTS",
    "SKELETON_NAME",
    "SOURCE_FPS",
    "SPINE_MERGE_PAIR",
    "UE_BONE_MAP",
    "UE_BONE_ORDER",
    "UE_PARENTS",
    "UE_ROOT_BONE",
    "UP_AXIS",
    "UNITS_LENGTH",
    "VAE_TOKEN_LEN",
    "WINDOW",
    "bone_index",
    "feat_slice",
    "joint_6d_slice",
    "parents_array",
    "rest_offsets_array",
    "token_len_for",
    "verify_constants",
]

# ---------------------------------------------------------------------------
# 1. 骨架：23 关节（索引即数组维序，已按源 BVH 实测顺序锁定）
# ---------------------------------------------------------------------------
#: 23 个 BVH 关节名，顺序即特征数组/权重张量的关节维序。
BONE_ORDER: Final[tuple[str, ...]] = (
    "Hips",  # 0  根
    "Spine",  # 1
    "Spine1",  # 2
    "Spine2",  # 3
    "Spine3",  # 4
    "Neck",  # 5
    "Head",  # 6
    "LeftShoulder",  # 7
    "LeftArm",  # 8
    "LeftForeArm",  # 9
    "LeftHand",  # 10
    "RightShoulder",  # 11
    "RightArm",  # 12
    "RightForeArm",  # 13
    "RightHand",  # 14
    "LeftUpLeg",  # 15
    "LeftLeg",  # 16
    "LeftFoot",  # 17
    "LeftToeBase",  # 18
    "RightUpLeg",  # 19
    "RightLeg",  # 20
    "RightFoot",  # 21
    "RightToeBase",  # 22
)

#: 每个关节的父关节索引，``-1`` 表示根。
PARENTS: Final[tuple[int, ...]] = (
    -1,  # Hips
    0,  # Spine      <- Hips
    1,  # Spine1     <- Spine
    2,  # Spine2     <- Spine1
    3,  # Spine3     <- Spine2
    4,  # Neck       <- Spine3
    5,  # Head       <- Neck
    4,  # LeftShoulder  <- Spine3
    7,  # LeftArm       <- LeftShoulder
    8,  # LeftForeArm   <- LeftArm
    9,  # LeftHand      <- LeftForeArm
    4,  # RightShoulder <- Spine3
    11,  # RightArm      <- RightShoulder
    12,  # RightForeArm  <- RightArm
    13,  # RightHand     <- RightForeArm
    0,  # LeftUpLeg     <- Hips
    15,  # LeftLeg       <- LeftUpLeg
    16,  # LeftFoot      <- LeftLeg
    17,  # LeftToeBase   <- LeftFoot
    0,  # RightUpLeg    <- Hips
    19,  # RightLeg      <- RightUpLeg
    20,  # RightFoot     <- RightLeg
    21,  # RightToeBase  <- RightFoot
)

#: 关节总数。
N_JOINTS: Final[int] = 23

#: 非根关节名（特征中 22×6D 的顺序即此元组）。
NON_ROOT_JOINTS: Final[tuple[str, ...]] = BONE_ORDER[1:]

#: 非根关节数。
N_NON_ROOT_JOINTS: Final[int] = N_JOINTS - 1

#: 每个关节父关节的**名字**（``None`` 表示根），便于日志与调试。
BONE_PARENT_NAMES: Final[tuple[str | None, ...]] = tuple(
    None if parent < 0 else BONE_ORDER[parent] for parent in PARENTS
)

#: 默认 rest pose 的关节局部偏移（cm，Y-up）。根关节偏移为 0（位移走 motion 通道）。
#:
#: 说明：源 BVH 的 OFFSET 在解析时从文件读取（权威值）；本表是**兜底默认值**，
#: 用于「无源文件」场景（合成数据、单测、导出前的 rest pose 校验）。
#: 数值按 Hips 世界高度 ≈ 91.7 cm 的正立姿态构造，与本项目实测数据一致。
DEFAULT_REST_OFFSETS_CM: Final[tuple[tuple[float, float, float], ...]] = (
    (0.0, 0.0, 0.0),  # 0  Hips（位移来自 motion 通道）
    (0.0, 12.0, 0.0),  # 1  Spine
    (0.0, 9.0, 0.0),  # 2  Spine1
    (0.0, 9.0, 0.0),  # 3  Spine2
    (0.0, 9.0, 0.0),  # 4  Spine3
    (0.0, 10.0, 0.0),  # 5  Neck
    (0.0, 9.0, 0.0),  # 6  Head
    (3.5, 8.0, 0.0),  # 7  LeftShoulder
    (12.0, 0.0, 0.0),  # 8  LeftArm
    (25.0, 0.0, 0.0),  # 9  LeftForeArm
    (7.0, 0.0, 0.0),  # 10 LeftHand
    (-3.5, 8.0, 0.0),  # 11 RightShoulder
    (-12.0, 0.0, 0.0),  # 12 RightArm
    (-25.0, 0.0, 0.0),  # 13 RightForeArm
    (-7.0, 0.0, 0.0),  # 14 RightHand
    (9.0, -9.5, 0.0),  # 15 LeftUpLeg
    (0.0, -42.0, 0.0),  # 16 LeftLeg
    (0.0, -40.0, 0.0),  # 17 LeftFoot
    (0.0, -6.0, 11.0),  # 18 LeftToeBase
    (-9.0, -9.5, 0.0),  # 19 RightUpLeg
    (0.0, -42.0, 0.0),  # 20 RightLeg
    (0.0, -40.0, 0.0),  # 21 RightFoot
    (0.0, -6.0, 11.0),  # 22 RightToeBase
)

#: 默认 Hips 世界高度（cm），源数据实测值。
DEFAULT_HIPS_HEIGHT_CM: Final[float] = 91.7

#: 用于足接触检测的关节（左/右 Foot），顺序决定特征 [144] / [145] 的通道语义。
FOOT_CONTACT_JOINTS: Final[tuple[int, int]] = (17, 21)

# ---------------------------------------------------------------------------
# 2. UE 骨架映射（22 骨 + root）
# ---------------------------------------------------------------------------
#: UE 侧多出的轨道根骨名；恒等旋转、零位移。
UE_ROOT_BONE: Final[str] = "root"

#: BVH 关节名 → UE 骨名。``Spine3`` 不单独出现（并入 ``spine_03``，见
#: :data:`SPINE_MERGE_PAIR` 与 ``retarget/ue_map.py``）。
UE_BONE_MAP: Final[dict[str, str]] = {
    "Hips": "pelvis",
    "Spine": "spine_01",
    "Spine1": "spine_02",
    "Spine2": "spine_03",
    # "Spine3" -> 并入 spine_03（fold 策略），不产生独立目标骨
    "Neck": "neck_01",
    "Head": "head",
    "LeftShoulder": "clavicle_l",
    "LeftArm": "upperarm_l",
    "LeftForeArm": "lowerarm_l",
    "LeftHand": "hand_l",
    "RightShoulder": "clavicle_r",
    "RightArm": "upperarm_r",
    "RightForeArm": "lowerarm_r",
    "RightHand": "hand_r",
    "LeftUpLeg": "thigh_l",
    "LeftLeg": "calf_l",
    "LeftFoot": "foot_l",
    "LeftToeBase": "ball_l",
    "RightUpLeg": "thigh_r",
    "RightLeg": "calf_r",
    "RightFoot": "foot_r",
    "RightToeBase": "ball_r",
}

#: UE 侧 23 根骨的完整顺序（``root`` 在 0，其余与 BONE_ORDER[0:4]+BONE_ORDER[5:] 对齐）。
UE_BONE_ORDER: Final[tuple[str, ...]] = (
    "root",  # 0  UE 额外轨道根（恒等）
    "pelvis",  # 1  <- Hips
    "spine_01",  # 2  <- Spine
    "spine_02",  # 3  <- Spine1
    "spine_03",  # 4  <- Spine2 (⊗ Spine3)
    "neck_01",  # 5  <- Neck
    "head",  # 6  <- Head
    "clavicle_l",  # 7  <- LeftShoulder
    "upperarm_l",  # 8  <- LeftArm
    "lowerarm_l",  # 9  <- LeftForeArm
    "hand_l",  # 10 <- LeftHand
    "clavicle_r",  # 11 <- RightShoulder
    "upperarm_r",  # 12 <- RightArm
    "lowerarm_r",  # 13 <- RightForeArm
    "hand_r",  # 14 <- RightHand
    "thigh_l",  # 15 <- LeftUpLeg
    "calf_l",  # 16 <- LeftLeg
    "foot_l",  # 17 <- LeftFoot
    "ball_l",  # 18 <- LeftToeBase
    "thigh_r",  # 19 <- RightUpLeg
    "calf_r",  # 20 <- RightLeg
    "foot_r",  # 21 <- RightFoot
    "ball_r",  # 22 <- RightToeBase
)

#: UE 侧父骨索引（与 :data:`UE_BONE_ORDER` 同长，``-1`` 为根）。
#: 由 :data:`PARENTS` 平移 +1 派生：BVH 根（Hips）的父变成 UE 的 ``root``。
UE_PARENTS: Final[tuple[int, ...]] = (
    -1,  # root
    0,  # pelvis <- root
    1,  # spine_01
    2,  # spine_02
    3,  # spine_03
    4,  # neck_01
    5,  # head
    4,  # clavicle_l
    7,  # upperarm_l
    8,  # lowerarm_l
    9,  # hand_l
    4,  # clavicle_r
    11,  # upperarm_r
    12,  # lowerarm_r
    13,  # hand_r
    1,  # thigh_l <- pelvis
    15,  # calf_l
    16,  # foot_l
    17,  # ball_l
    1,  # thigh_r <- pelvis
    19,  # calf_r
    20,  # foot_r
    21,  # ball_r
)

#: Spine3 降级策略涉及的 (源关节名, 目标 UE 骨名)。
SPINE_MERGE_PAIR: Final[tuple[str, str]] = ("Spine3", "spine_03")

# ---------------------------------------------------------------------------
# 3. 运动特征：146 维（layout_version = 1）
# ---------------------------------------------------------------------------
#: 特征总维数。
FEATURE_DIM: Final[int] = 146

#: 命名切片布局 —— 146 维的**唯一**权威定义（TTM-ARCH.md §5.3）。
FEAT_LAYOUT: Final[dict[str, slice]] = {
    # [0,2) 根水平速度 (vx, vz)，朝向系（heading 前/侧向），cm/s，z-score
    "root_vel": slice(0, 2),
    # [2,3) 根高度 y（骨盆世界高度），cm，z-score
    "root_height": slice(2, 3),
    # [3,5) 朝向 (sin θ, cos θ)，θ = 偏航角，天然有界不归一化
    "heading": slice(3, 5),
    # [5,6) 偏航角速度 ω，rad/s，z-score
    "heading_vel": slice(5, 6),
    # [6,12) 根倾斜 6D：R_yaw⁻¹·R_root 的前两行，不归一化（每列单位范数）
    "root_tilt_6d": slice(6, 12),
    # [12,144) 22 个非根关节 × 6D 局部旋转，顺序 = BONE_ORDER[1:]
    "joints_6d": slice(12, 144),
    # [144,146) 足接触 (contact_L, contact_R) ∈ {0,1}
    "foot_contact": slice(144, 146),
}

#: 需要 z-score 的连续维（其余维 mean=0/std=1 恒等，不参与归一化）。
NORM_DIMS: Final[tuple[int, ...]] = (0, 1, 2, 5)

#: 朝向两维（sin, cos）。
HEADING_DIMS: Final[tuple[int, int]] = (3, 4)

# ---------------------------------------------------------------------------
# 4. 时间 / 窗口 / 潜变量
# ---------------------------------------------------------------------------
#: 内部与导出统一帧率。
FPS: Final[int] = 30

#: 源 BVH 帧率（120 / 4 == 30，精确整除，线性/球面插值无损相位）。
SOURCE_FPS: Final[int] = 120

#: 训练/推理窗口帧数（5 秒）。
WINDOW: Final[int] = 150

#: 长动作滑窗重叠帧数。
OVERLAP: Final[int] = 30

#: 尾窗保留阈值：尾窗长度 ≥ 该值才保留（不足则 pad + mask）。
MIN_TAIL_FRAMES: Final[int] = 75

#: clip_id 在 npz 分片里的最大字符数（``<U80``，ARCH §5.5）。
#: 窗口级 id = ``{clip_id}_win{idx}``，因此 clip_id 本体必须留出后缀空间，
#: 否则落盘会被 ``<U80`` 静默截断，与 labels.jsonl 的 id 对不上。
CLIP_ID_MAXLEN: Final[int] = 80

#: VAE 时序下采样倍数（Conv1d stride）。
DOWNSAMPLE: Final[int] = 4

#: 潜空间维数。
LATENT_DIM: Final[int] = 128

#: 150 帧窗口经 4× 下采样后的 token 数 = ceil(150/4) = 38。
VAE_TOKEN_LEN: Final[int] = math.ceil(WINDOW / DOWNSAMPLE)

# ---------------------------------------------------------------------------
# 5. 单位 / 轴向 / 欧拉序
# ---------------------------------------------------------------------------
#: 长度单位（内部与导出统一 cm，永不引入米）。
UNITS_LENGTH: Final[str] = "cm"

#: 内部 up 轴（BVH 约定）。
UP_AXIS: Final[str] = "Y"

#: BVH 欧拉角顺序（root 与所有关节一致）。
EULER_ORDER: Final[str] = "YXZ"

#: Motion JSON 的格式标识。
MOTION_JSON_FORMAT: Final[str] = "ttm-motion/v1"

#: Motion JSON 的骨架标识。
SKELETON_NAME: Final[str] = "bvh23"

# ---------------------------------------------------------------------------
# 6. 足接触检测阈值
# ---------------------------------------------------------------------------
#: 判定为接触的足部高度阈值（cm）。
CONTACT_HEIGHT_CM: Final[float] = 5.0

#: 判定为接触的足部速度阈值（cm/s）。
CONTACT_SPEED_CMPS: Final[float] = 15.0

#: 接触状态翻转所需的连续一致帧数（滞回，抑制抖动）。
CONTACT_HYSTERESIS_FRAMES: Final[int] = 2

# ---------------------------------------------------------------------------
# 7. 辅助函数
# ---------------------------------------------------------------------------


def bone_index(name: str) -> int:
    """返回关节名在 :data:`BONE_ORDER` 中的索引。

    Args:
        name: 关节名（大小写敏感，与 BVH 文件一致）。

    Returns:
        0..22 的索引。

    Raises:
        KeyError: 关节名不在 23 关节表中。
    """
    try:
        return BONE_ORDER.index(name)
    except ValueError as exc:
        raise KeyError(f"未知关节名 {name!r}；合法值见 constants.BONE_ORDER") from exc


def feat_slice(name: str) -> slice:
    """返回特征布局中某个命名段对应的 :class:`slice`。

    Args:
        name: :data:`FEAT_LAYOUT` 的键（如 ``"root_vel"``）。

    Returns:
        该段的切片对象。

    Raises:
        KeyError: 未定义的段名。
    """
    if name not in FEAT_LAYOUT:
        raise KeyError(f"未知特征段 {name!r}；合法值: {sorted(FEAT_LAYOUT)}")
    return FEAT_LAYOUT[name]


def joint_6d_slice(joint_idx: int) -> slice:
    """返回第 ``joint_idx`` 个**非根**关节（``BONE_ORDER[1:]`` 序）的 6D 切片。

    Args:
        joint_idx: 非根关节序号 0..21（0 对应 ``Spine``，21 对应 ``RightToeBase``）。

    Returns:
        长度 6 的 :class:`slice`。

    Raises:
        IndexError: 序号越界。
    """
    if not 0 <= joint_idx < N_NON_ROOT_JOINTS:
        raise IndexError(f"非根关节序号越界: {joint_idx}（合法 0..{N_NON_ROOT_JOINTS - 1}）")
    base = FEAT_LAYOUT["joints_6d"].start
    return slice(base + 6 * joint_idx, base + 6 * (joint_idx + 1))


def parents_array() -> np.ndarray:
    """返回 ``int32[23]`` 的父关节索引数组（便于矢量化 FK）。"""
    return np.asarray(PARENTS, dtype=np.int32)


def rest_offsets_array() -> np.ndarray:
    """返回 ``float64[23,3]`` 的默认 rest 偏移（cm）。"""
    return np.asarray(DEFAULT_REST_OFFSETS_CM, dtype=np.float64)


def token_len_for(window: int, downsample: int = DOWNSAMPLE) -> int:
    """计算给定窗口与下采样倍数对应的潜 token 数。

    Args:
        window: 窗口帧数。
        downsample: 时序下采样倍数。

    Returns:
        ``ceil(window / downsample)``。
    """
    if window <= 0 or downsample <= 0:
        raise ValueError(f"window/downsample 必须为正: window={window}, downsample={downsample}")
    return math.ceil(window / downsample)


def verify_constants() -> None:
    """自检常量表的内部一致性；不一致时抛 :class:`AssertionError`。

    检查项：

    1. ``len(BONE_ORDER) == len(PARENTS) == len(DEFAULT_REST_OFFSETS_CM) == 23``；
    2. 父子关系无环、有且仅有 1 个根、且每个父索引 < 23；
    3. :data:`UE_BONE_MAP` 覆盖 22 个非根关节（Spine3 走 fold），UE 侧多出 ``root``；
    4. :data:`UE_BONE_ORDER` 长度 23 且无重复，:data:`UE_PARENTS` 与之一致；
    5. :data:`FEAT_LAYOUT` 各段连续无重叠、总长恰好 146；
    6. ``VAE_TOKEN_LEN == ceil(WINDOW / DOWNSAMPLE) == 38``。
    """
    assert len(BONE_ORDER) == N_JOINTS, f"BONE_ORDER 长度 {len(BONE_ORDER)} != {N_JOINTS}"
    assert len(set(BONE_ORDER)) == N_JOINTS, "BONE_ORDER 存在重名关节"
    assert len(PARENTS) == N_JOINTS, f"PARENTS 长度 {len(PARENTS)} != {N_JOINTS}"
    assert len(DEFAULT_REST_OFFSETS_CM) == N_JOINTS, "REST_OFFSETS 行数 != 23"

    # --- 父子关系：单根 + 无环
    roots = [i for i, parent in enumerate(PARENTS) if parent < 0]
    assert roots == [0], f"应当只有索引 0 为根，实际 roots={roots}"
    for idx, parent in enumerate(PARENTS):
        assert -1 <= parent < N_JOINTS, f"PARENTS[{idx}]={parent} 越界"
        assert parent != idx, f"PARENTS[{idx}] 自引用"
    for idx in range(N_JOINTS):
        seen = {idx}
        cursor = idx
        while PARENTS[cursor] >= 0:
            cursor = PARENTS[cursor]
            assert cursor not in seen, f"关节 {idx} 的父链存在环"
            seen.add(cursor)

    # --- UE 映射
    mapped_sources = set(UE_BONE_MAP)
    expected_sources = set(BONE_ORDER) - {SPINE_MERGE_PAIR[0]}
    assert mapped_sources == expected_sources, (
        f"UE_BONE_MAP 覆盖不符；缺少 {sorted(expected_sources - mapped_sources)}，"
        f"多出 {sorted(mapped_sources - expected_sources)}"
    )
    assert len(UE_BONE_MAP) == N_JOINTS - 1, (
        f"UE_BONE_MAP 应有 {N_JOINTS - 1} 条（23 个关节去掉 fold 掉的 Spine3），"
        f"实际 {len(UE_BONE_MAP)}"
    )
    targets = set(UE_BONE_MAP.values())
    assert len(targets) == len(UE_BONE_MAP), "UE_BONE_MAP 目标骨名重复"
    assert targets | {UE_ROOT_BONE} == set(UE_BONE_ORDER), (
        f"UE 目标骨集合与 UE_BONE_ORDER 不一致；差集 {sorted((targets | {UE_ROOT_BONE}) ^ set(UE_BONE_ORDER))}"
    )
    assert len(UE_BONE_ORDER) == N_JOINTS, "UE_BONE_ORDER 长度 != 23"
    assert len(set(UE_BONE_ORDER)) == N_JOINTS, "UE_BONE_ORDER 存在重名"
    assert len(UE_PARENTS) == N_JOINTS, "UE_PARENTS 长度 != 23"
    assert UE_PARENTS[0] == -1, "UE root 必须是根"
    ue_roots = [i for i, parent in enumerate(UE_PARENTS) if parent < 0]
    assert ue_roots == [0], f"UE 侧应只有 root 为根，实际 {ue_roots}"

    # --- 特征布局：连续、无重叠、总长 146
    cursor = 0
    for name, slc in FEAT_LAYOUT.items():
        assert slc.start == cursor, f"FEAT_LAYOUT[{name!r}].start={slc.start} 应为 {cursor}"
        assert slc.step is None or slc.step == 1, f"FEAT_LAYOUT[{name!r}] 不支持 step"
        cursor = slc.stop
    assert cursor == FEATURE_DIM, f"FEAT_LAYOUT 总长 {cursor} != {FEATURE_DIM}"
    assert FEAT_LAYOUT["joints_6d"].stop - FEAT_LAYOUT["joints_6d"].start == 6 * N_NON_ROOT_JOINTS

    # --- 时间/潜变量
    assert VAE_TOKEN_LEN == token_len_for(WINDOW, DOWNSAMPLE), "VAE_TOKEN_LEN 与 window/downsample 不一致"
    assert VAE_TOKEN_LEN == 38, f"VAE_TOKEN_LEN 应为 38，实际 {VAE_TOKEN_LEN}"
    assert 4 * VAE_TOKEN_LEN >= WINDOW, "下采样后 token 必须覆盖整个窗口（padding 补齐）"
