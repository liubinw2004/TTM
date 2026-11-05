"""T09：把内部 23 关节动作重定向到 UE 骨架，并导出 BVH / Motion JSON。

映射规则（``TTM-ARCH.md`` §7 T09）：

* 内部 :data:`ttm.common.constants.BONE_ORDER`（23 关节，BVH 命名）→
  :data:`ttm.common.constants.UE_BONE_ORDER`（``root`` + 22 骨，UE 命名）；
* ``Spine3`` **没有独立的 UE 骨**：默认折叠进 ``spine_03``（旋转按局部序相乘
  ``q(spine_03) = q(Spine2) ∘ q(Spine3)``）；若调用方给出目标骨架且其中**存在**
  ``spine_04``，则改为单独输出 ``spine_04``（不折叠）；
* ``root`` 是 UE 侧额外轨道根：恒等旋转 + 零位移，根位移挂在 ``pelvis`` 上
  （BVH 的位移通道写在根骨 ``pelvis`` 上，与 three.js / UE 的 retarget 约定一致）；
* **rest pose 补偿**：``out = q_rest_dst ∘ (q_rest_src⁻¹ ∘ q_src)``。两边 rest 都是
  单位四元数时退化为纯换名，因此不传 rest 也能用；UE Mannequin 是 A-pose，
  实际投放时把它的 rest 局部旋转传进 ``dst_rest_quats`` 即可。

导出格式见 :data:`ttm.common.constants.MOTION_JSON_FORMAT`（``ttm-motion/v1``），
前端（``web/app.js``）与 UE 导入工具（``tools/ue_import.py``）共用这一份 JSON。
"""

from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Sequence

import numpy as np
from scipy.spatial.transform import Rotation

from ttm.common import constants as C
from ttm.common import paths as P

__all__ = [
    "RetargetPlan",
    "apply_rest_delta",
    "build_plan",
    "canonicalize_quats",
    "retarget_motion",
    "write_motion_json",
    "write_ue_bvh",
]

#: 可选的目标骨名；目标骨架里有它就不折叠 Spine3。
SPINE3_ALIAS: str = "spine_04"


def canonicalize_quats(rotations: np.ndarray) -> np.ndarray:
    """四元数半球规约：``qw < 0`` 的帧取反，**之后再归一化**。

    ``q`` 与 ``-q`` 表示同一旋转，但 slerp / 线性插值在两者之间会走「长弧」，
    UE 导入与前端插值都会出现翻转歧义（BVH 走欧拉所以看不出来，属静默错误）。
    QA 实测导出 JSON 里 qw<0 占 4.2%（集中在 pelvis / thigh_l / thigh_r）。

    **顺序不能反**：先翻转（只是乘 -1，不改变范数），后归一化；反过来的话
    「先归一化再翻转」在数值上虽然也成立，但一旦调用方传入未归一化四元数，
    先归一化会把误差混进符号判定之外的环节，无法定位问题。这里统一按
    「翻转 → 归一化」执行，并对缺省范数做 epsilon 保护。

    Args:
        rotations: ``[F, B, 4]`` 四元数（xyzw，w 在最后一列）。

    Returns:
        规约后的新数组（``qw >= 0`` 且逐帧单位范数），不修改入参。
    """
    array = np.asarray(rotations, dtype=np.float64)
    if array.ndim != 3 or array.shape[-1] != 4:
        raise ValueError(f"rotations 形状应为 [F,B,4]，实际 {array.shape}")
    flipped = np.where(array[..., 3:4] < 0.0, -array, array)
    norms = np.linalg.norm(flipped, axis=-1, keepdims=True)
    return flipped / np.clip(norms, 1e-12, None)


@dataclass(frozen=True)
class RetargetPlan:
    """一次重定向的映射表（不可变，可复用）。

    Attributes:
        bone_names: 目标骨名，``bone_names[0] == "root"``。
        parents: 父骨索引（同长，``-1`` 为根）。
        source_joints: 每根目标骨对应的**主**源关节索引；``root`` 忽略此值。
        fold_joints: 需要额外并入的源关节索引（``-1`` 表示无），目前只有
            ``spine_03 <- Spine3`` 一种情况。
        rest_offsets_cm: ``[B,3]`` rest 偏移（cm），供 BVH OFFSET 与前端摆位使用。
        spine3_folded: 是否发生了 ``Spine3`` 折叠降级。
    """

    bone_names: tuple[str, ...]
    parents: tuple[int, ...]
    source_joints: tuple[int, ...]
    fold_joints: tuple[int, ...]
    rest_offsets_cm: np.ndarray
    spine3_folded: bool

    @property
    def n_bones(self) -> int:
        """目标骨数量。"""
        return len(self.bone_names)

    def index_of(self, name: str) -> int:
        """按骨名取索引。

        Args:
            name: 目标骨名。

        Returns:
            骨索引。

        Raises:
            KeyError: 骨名不在本计划内。
        """
        try:
            return self.bone_names.index(name)
        except ValueError as exc:
            raise KeyError(f"目标骨架里没有骨: {name}") from exc

    def to_metadata(self) -> dict[str, Any]:
        """导出为 JSON 的 ``skeleton`` 段。"""
        return {
            "name": f"ue{self.n_bones}",
            "format": C.MOTION_JSON_FORMAT,
            "unit": C.UNITS_LENGTH,
            "up_axis": C.UP_AXIS,
            "euler_order": C.EULER_ORDER,
            "spine3_folded": self.spine3_folded,
            "bones": [
                {
                    "name": name,
                    "parent": int(self.parents[index]),
                    "source_joint": (
                        C.BONE_ORDER[self.source_joints[index]]
                        if self.source_joints[index] >= 0
                        else None
                    ),
                    "rest_offset_cm": [float(value) for value in self.rest_offsets_cm[index]],
                }
                for index, name in enumerate(self.bone_names)
            ],
        }


def build_plan(
    *,
    target_bone_names: Sequence[str] | None = None,
    fold_spine3: bool | None = None,
    rest_offsets_cm: np.ndarray | Sequence[Sequence[float]] | None = None,
) -> RetargetPlan:
    """构造重定向映射表（含 Spine3 折叠/降级判定）。

    Args:
        target_bone_names: 目标骨架已有的骨名。``None`` 表示按
            :data:`ttm.common.constants.UE_BONE_ORDER`（已折叠）处理。
        fold_spine3: 强制是否折叠；``None`` 时按 ``target_bone_names`` 里有没有
            ``spine_04`` 自动判定（**没有就降级折叠**，这是 T09 的硬要求）。
        rest_offsets_cm: ``[23,3]`` rest 偏移（源骨架）。

    Returns:
        :class:`RetargetPlan`。

    Raises:
        ValueError: rest 偏移形状不符。
    """
    if rest_offsets_cm is None:
        offsets = np.asarray(C.DEFAULT_REST_OFFSETS_CM, dtype=np.float64)
    else:
        offsets = np.asarray(rest_offsets_cm, dtype=np.float64)
    if offsets.shape != (C.N_JOINTS, 3):
        raise ValueError(f"rest_offsets_cm 形状应为 [{C.N_JOINTS},3]，实际 {offsets.shape}")

    if fold_spine3 is None:
        if target_bone_names is None:
            folded = True  # UE_BONE_ORDER 本身就是折叠版
        else:
            folded = SPINE3_ALIAS not in set(target_bone_names)
    else:
        folded = bool(fold_spine3)

    names: list[str] = [C.UE_ROOT_BONE]
    parents: list[int] = [-1]
    source: list[int] = [0]  # root 占位（旋转恒等、无位移）
    folds: list[int] = [-1]
    offsets_out: list[np.ndarray] = [np.zeros(3, dtype=np.float64)]
    target_of_source: dict[int, int] = {}

    spine_src = C.BONE_ORDER.index(C.SPINE_MERGE_PAIR[0])  # Spine3 的源索引
    spine_bone = C.SPINE_MERGE_PAIR[1]  # "spine_03"（折叠时的宿主骨）

    for joint_index, joint_name in enumerate(C.BONE_ORDER):
        if folded and joint_index == spine_src:
            # 折叠：Spine3 不新增目标骨，与父关节（Spine2 → spine_03）共用同一个。
            # 必须显式登记映射，否则 Neck 的父索引会取不到。
            target_of_source[joint_index] = target_of_source[C.PARENTS[joint_index]]
            continue

        if joint_name == C.SPINE_MERGE_PAIR[0]:
            bone_name = SPINE3_ALIAS  # 未折叠：单独输出 spine_04
        elif joint_index == 0:
            bone_name = C.UE_BONE_MAP[joint_name]
        else:
            bone_name = C.UE_BONE_MAP.get(joint_name, f"bvh_{joint_name.lower()}")

        parent_source = C.PARENTS[joint_index]
        parent_index = 0 if parent_source < 0 else target_of_source[parent_source]
        target_of_source[joint_index] = len(names)
        names.append(bone_name)
        parents.append(parent_index)
        source.append(joint_index)
        # 折叠时，宿主骨（spine_03）额外并入 Spine3 的局部旋转
        fold_source = spine_src if (folded and bone_name == spine_bone) else -1
        folds.append(fold_source)
        offsets_out.append(offsets[joint_index].copy())

    return RetargetPlan(
        bone_names=tuple(names),
        parents=tuple(parents),
        source_joints=tuple(source),
        fold_joints=tuple(folds),
        rest_offsets_cm=np.stack(offsets_out, axis=0),
        spine3_folded=bool(folded),
    )


def _as_rotation(quats: np.ndarray | Sequence[float] | None, count: int) -> Rotation:
    """把四元数数组转成 :class:`Rotation`（``None`` → ``count`` 个单位旋转）。"""
    if quats is None:
        return Rotation.identity(count)
    array = np.asarray(quats, dtype=np.float64)
    if array.shape != (count, 4):
        raise ValueError(f"rest 四元数形状应为 [{count},4]，实际 {array.shape}")
    return Rotation.from_quat(array)


def apply_rest_delta(
    quats: np.ndarray,
    plan: RetargetPlan,
    *,
    src_rest_quats: np.ndarray | None = None,
    dst_rest_quats: np.ndarray | None = None,
) -> np.ndarray:
    """按 ``out = q_rest_dst ∘ (q_rest_src⁻¹ ∘ q_src)`` 做 rest pose 补偿。

    折叠骨（``spine_03``）在此处把 ``Spine3`` 的局部旋转乘上去。

    Args:
        quats: ``[F,23,4]`` 源动作四元数（xyzw，内部骨架序）。
        plan: 重定向映射表。
        src_rest_quats: ``[23,4]`` 源 rest **局部**四元数；``None`` 视为单位。
        dst_rest_quats: ``[B,4]`` 目标 rest **局部**四元数；``None`` 视为单位。

    Returns:
        ``[F,B,4]`` 目标骨架四元数（xyzw）。

    Raises:
        ValueError: 形状不符。
    """
    array = np.asarray(quats, dtype=np.float64)
    if array.ndim != 3 or array.shape[1:] != (C.N_JOINTS, 4):
        raise ValueError(f"quats 形状应为 [F,{C.N_JOINTS},4]，实际 {array.shape}")
    frames = array.shape[0]
    src_rest = _as_rotation(src_rest_quats, C.N_JOINTS)
    dst_rest = _as_rotation(dst_rest_quats, plan.n_bones)

    out = np.zeros((frames, plan.n_bones, 4), dtype=np.float64)
    out[..., 3] = 1.0  # root 与所有未覆盖骨默认单位旋转
    for bone in range(1, plan.n_bones):
        joint = plan.source_joints[bone]
        delta = src_rest[joint].inv() * Rotation.from_quat(array[:, joint])
        fold = plan.fold_joints[bone]
        if fold >= 0:
            delta = delta * (src_rest[fold].inv() * Rotation.from_quat(array[:, fold]))
        out[:, bone] = (dst_rest[bone] * delta).as_quat()
    return out


def retarget_motion(
    quats: np.ndarray,
    root_pos_cm: np.ndarray,
    *,
    plan: RetargetPlan | None = None,
    src_rest_quats: np.ndarray | None = None,
    dst_rest_quats: np.ndarray | None = None,
) -> tuple[np.ndarray, np.ndarray]:
    """重定向到 UE 骨架。

    Args:
        quats: ``[F,23,4]`` 源四元数。
        root_pos_cm: ``[F,3]`` 根位移（cm）。
        plan: 映射表；``None`` 时用默认（折叠版）。
        src_rest_quats: 源 rest 局部旋转。
        dst_rest_quats: 目标 rest 局部旋转。

    Returns:
        ``(rotations[F,B,4], root_pos[F,3])``；位移原样保留（``pelvis`` 承载）。
    """
    active = plan or build_plan()
    rotations = apply_rest_delta(
        quats, active, src_rest_quats=src_rest_quats, dst_rest_quats=dst_rest_quats
    )
    positions = np.asarray(root_pos_cm, dtype=np.float64)
    if positions.ndim != 2 or positions.shape[1] != 3:
        raise ValueError(f"root_pos_cm 形状应为 [F,3]，实际 {positions.shape}")
    if positions.shape[0] != rotations.shape[0]:
        raise ValueError(
            f"帧数不一致: quats {rotations.shape[0]} vs root_pos {positions.shape[0]}"
        )
    return rotations, positions


def write_motion_json(
    path: str | Path,
    rotations: np.ndarray,
    root_pos_cm: np.ndarray,
    *,
    plan: RetargetPlan | None = None,
    fps: int = C.FPS,
    prompt: str = "",
    extra: dict[str, Any] | None = None,
) -> Path:
    """写出 ``ttm-motion/v1`` JSON（前端与 UE 导入共用）。

    结构::

        {
          "format": "ttm-motion/v1", "fps": 30, "n_frames": F,
          "skeleton": {...bones/parents/rest_offset...},
          "prompt": "...",
          "root_positions_cm": [[x,y,z], ...],
          "tracks": [{"bone": "pelvis", "rotations_xyzw": [[x,y,z,w], ...]}, ...]
        }

    Args:
        path: 输出路径（经写盘守卫）。
        rotations: ``[F,B,4]`` 目标骨架四元数。
        root_pos_cm: ``[F,3]`` 根位移。
        plan: 映射表；``None`` 用默认。
        fps: 帧率。
        prompt: 生成用的中文指令（便于追溯）。
        extra: 额外写进 JSON 顶层的字段。

    Returns:
        写出的路径。

    Raises:
        ValueError: 形状不符。
    """
    active = plan or build_plan()
    # 半球规约（F-07）：qw<0 → 取反，再归一化。写出前统一做，保证任何消费端
    # （UE/前端/ue_import）拿到的都是同一半球，插值不会走长弧。
    array = canonicalize_quats(rotations)
    if array.shape[1:] != (active.n_bones, 4):
        raise ValueError(
            f"rotations 形状应为 [F,{active.n_bones},4]，实际 {array.shape}"
        )
    positions = np.asarray(root_pos_cm, dtype=np.float64)
    if positions.shape != (array.shape[0], 3):
        raise ValueError(
            f"root_pos_cm 形状应为 [{array.shape[0]},3]，实际 {positions.shape}"
        )

    payload: dict[str, Any] = {
        "format": C.MOTION_JSON_FORMAT,
        "fps": int(fps),
        "n_frames": int(array.shape[0]),
        "n_bones": int(active.n_bones),
        "skeleton": active.to_metadata(),
        "prompt": str(prompt),
        "root_positions_cm": positions.round(6).tolist(),
        "tracks": [
            {
                "bone": name,
                "rotations_xyzw": array[:, index, :].round(6).tolist(),
            }
            for index, name in enumerate(active.bone_names)
        ],
    }
    if extra:
        payload.update(extra)

    target = P.resolve_out(Path(path), mkparents=True)
    target.write_text(json.dumps(payload, ensure_ascii=False), encoding="utf-8")
    return target


def write_ue_bvh(
    path: str | Path,
    rotations: np.ndarray,
    root_pos_cm: np.ndarray,
    *,
    plan: RetargetPlan | None = None,
    fps: int = C.FPS,
) -> Path:
    """把**已重定向**的动作写成 UE 骨名的 BVH（便于直接在 UE/Maya 里看）。

    Args:
        path: 输出路径。
        rotations: ``[F,B,4]`` 目标骨架四元数。
        root_pos_cm: ``[F,3]`` 根位移。
        plan: 映射表；``None`` 用默认。
        fps: 帧率。

    Returns:
        写出的路径。
    """
    from ttm.export.bvh_io import write_bvh  # noqa: PLC0415 - 避免循环导入

    active = plan or build_plan()
    return write_bvh(
        path,
        rotations,
        root_pos_cm,
        fps=fps,
        bone_names=active.bone_names,
        parents=active.parents,
        offsets_cm=active.rest_offsets_cm,
        root_name=C.UE_ROOT_BONE,
    )
