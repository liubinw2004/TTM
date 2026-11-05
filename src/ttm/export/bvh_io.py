"""BVH 导出：把「23 关节四元数 + 根位移」写成标准 BVH 文本。

与 :mod:`ttm.data.bvh` 严格互逆（解析 → 特征 → 生成 → 导出 应保持骨架一致）：

* 欧拉序 **YXZ**、**角度制**（与源数据一致，避免下游重定向再换算）；
* 根关节 6 通道（``Xposition Yposition Zposition Yrotation Xrotation Zrotation``），
  其余 22 关节 3 通道（``Yrotation Xrotation Zrotation``）；
* 单位 **cm**、Y-up 右手系（BVH 系）；
* 关节顺序与 :data:`ttm.common.constants.BONE_ORDER` 一致，父子关系取自
  :data:`ttm.common.constants.PARENTS`，``OFFSET`` 用默认 rest offsets。

设计取舍：不写 ``End Site`` 块（源数据的 End Site 只是末端偏移，导出时不需要，
且写出来会让某些下游解析器多算一个关节）。
"""

from __future__ import annotations

from pathlib import Path
from typing import Final, Sequence

import numpy as np
from scipy.spatial.transform import Rotation

from ttm.common import constants as C
from ttm.common import paths as P

__all__ = [
    "BVH_FRAME_TIME_30FPS",
    "build_hierarchy_lines",
    "motion_to_euler_deg",
    "write_bvh",
]

#: 30fps 下的 Frame Time（秒）。
BVH_FRAME_TIME_30FPS: Final[float] = 1.0 / float(C.FPS)

#: 根关节的通道声明（顺序不可改，解析端按名字匹配）。
_ROOT_CHANNELS: Final[str] = (
    "CHANNELS 6 Xposition Yposition Zposition Yrotation Xrotation Zrotation"
)

#: 非根关节的通道声明。
_JOINT_CHANNELS: Final[str] = "CHANNELS 3 Yrotation Xrotation Zrotation"


def _children(parents: Sequence[int], index: int) -> list[int]:
    """返回某关节的子关节索引（升序 = BONE_ORDER 顺序）。"""
    return [child for child, parent in enumerate(parents) if parent == index]


def _emit_joint(
    lines: list[str],
    index: int,
    depth: int,
    *,
    bone_names: Sequence[str],
    parents: Sequence[int],
    offsets: np.ndarray,
) -> None:
    """递归输出一个关节块（标准 2 空格缩进）。"""
    pad = "\t" * depth
    keyword = "ROOT" if index == 0 else "JOINT"
    lines.append(f"{pad}{keyword} {bone_names[index]}")
    lines.append(f"{pad}{{")
    offset = offsets[index]
    lines.append(f"{pad}\tOFFSET {offset[0]:.6f} {offset[1]:.6f} {offset[2]:.6f}")
    lines.append(f"{pad}\t{_ROOT_CHANNELS if index == 0 else _JOINT_CHANNELS}")
    for child in _children(parents, index):
        _emit_joint(
            lines,
            child,
            depth + 1,
            bone_names=bone_names,
            parents=parents,
            offsets=offsets,
        )
    lines.append(f"{pad}}}")


def build_hierarchy_lines(
    *,
    bone_names: Sequence[str] = C.BONE_ORDER,
    parents: Sequence[int] = C.PARENTS,
    offsets_cm: np.ndarray | None = None,
) -> list[str]:
    """生成 ``HIERARCHY`` 小节的行列表。

    Args:
        bone_names: 23 个关节名。
        parents: 父索引（``-1`` 为根）。
        offsets_cm: ``[23,3]`` rest offsets；``None`` 时用默认骨架。

    Returns:
        行列表（不含 ``MOTION``）。
    """
    if len(bone_names) != C.N_JOINTS or len(parents) != C.N_JOINTS:
        raise ValueError(f"关节表长度应为 {C.N_JOINTS}，实际 {len(bone_names)}/{len(parents)}")
    offsets = (
        np.asarray(offsets_cm, dtype=np.float64)
        if offsets_cm is not None
        else C.rest_offsets_array()
    )
    if offsets.shape != (C.N_JOINTS, 3):
        raise ValueError(f"offsets 形状应为 ({C.N_JOINTS},3)，实际 {offsets.shape}")
    lines: list[str] = ["HIERARCHY"]
    _emit_joint(
        lines, 0, 0, bone_names=bone_names, parents=parents, offsets=offsets
    )
    return lines


def motion_to_euler_deg(quats: np.ndarray) -> np.ndarray:
    """``quat[F,23,4] xyzw`` → ``euler_deg[F,23,3]``（YXZ，角度制）。

    Args:
        quats: ``[F,23,4]`` 单位四元数（xyzw）。

    Returns:
        ``[F,23,3]`` 欧拉角（度），分量顺序 ``(Y, X, Z)``。
    """
    array = np.asarray(quats, dtype=np.float64)
    if array.ndim != 3 or array.shape[1:] != (C.N_JOINTS, 4):
        raise ValueError(f"quats 形状应为 [F,{C.N_JOINTS},4]，实际 {array.shape}")
    flat = array.reshape(-1, 4)
    # scipy 的 as_euler 返回顺序与 seq 字符串一致：(Y, X, Z)
    euler = Rotation.from_quat(flat).as_euler(C.EULER_ORDER, degrees=True)
    return euler.reshape(array.shape[0], C.N_JOINTS, 3)


def write_bvh(
    path: str | Path,
    quats: np.ndarray,
    root_pos_cm: np.ndarray,
    *,
    fps: int = C.FPS,
    bone_names: Sequence[str] = C.BONE_ORDER,
    parents: Sequence[int] = C.PARENTS,
    offsets_cm: np.ndarray | None = None,
    root_name: str | None = None,
) -> Path:
    """把一段动作写成 BVH 文件（经写盘守卫）。

    Args:
        path: 目标路径（必须位于 ``D:/TTM`` 内）。
        quats: ``[F,23,4]`` 四元数（xyzw）。
        root_pos_cm: ``[F,3]`` 根位移（cm，BVH 系）。
        fps: 帧率（写进 ``Frame Time``）。
        bone_names: 关节名（默认 :data:`BONE_ORDER`）。
        parents: 父索引（默认 :data:`PARENTS`）。
        offsets_cm: rest offsets；``None`` 时用默认骨架。
        root_name: 根关节名；``None`` 时用 ``bone_names[0]``。

    Returns:
        实际写出的路径。

    Raises:
        ValueError: 形状不匹配或帧数为 0。
    """
    quats = np.asarray(quats, dtype=np.float64)
    root_pos_cm = np.asarray(root_pos_cm, dtype=np.float64)
    if root_pos_cm.ndim != 2 or root_pos_cm.shape[1] != 3:
        raise ValueError(f"root_pos_cm 形状应为 [F,3]，实际 {root_pos_cm.shape}")
    frames = int(root_pos_cm.shape[0])
    if frames == 0:
        raise ValueError("帧数为 0，无法导出 BVH")
    if quats.shape != (frames, C.N_JOINTS, 4):
        raise ValueError(
            f"quats 形状应为 ({frames},{C.N_JOINTS},4)，实际 {quats.shape}"
        )
    if fps <= 0:
        raise ValueError(f"fps 必须为正，实际 {fps}")

    names = list(bone_names)
    if root_name:
        names[0] = str(root_name)
    euler_deg = motion_to_euler_deg(quats)

    lines = build_hierarchy_lines(bone_names=names, parents=parents, offsets_cm=offsets_cm)
    lines.append("MOTION")
    lines.append(f"Frames: {frames}")
    lines.append(f"Frame Time: {1.0 / float(fps):.9f}")
    for frame in range(frames):
        row: list[str] = [f"{value:.6f}" for value in root_pos_cm[frame]]
        row.extend(f"{value:.6f}" for value in euler_deg[frame, 0])
        for joint in range(1, C.N_JOINTS):
            row.extend(f"{value:.6f}" for value in euler_deg[frame, joint])
        lines.append(" ".join(row))

    target = P.resolve_out(path, mkparents=True)
    target.write_text("\n".join(lines) + "\n", encoding="utf-8")
    return target
