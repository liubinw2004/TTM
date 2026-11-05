"""骨架定义与运动学工具：正运动学（FK）、帧率重采样、足接触检测。

本模块只处理**已在内存中的数组**，不碰文件系统（文件解析在
:mod:`ttm.data.bvh`）。这样单测可以完全离线、无外部依赖地跑。

坐标与旋转约定（``TTM-ARCH.md`` §5.1）：

* 内部统一 **BVH 系：Y-up、右手系、单位 cm**；
* 欧拉序 **YXZ**、角度制（进出本模块的 ``euler_deg`` 为角度，``euler_rad`` 为弧度）；
* BVH FK 语义：``p[i] = p[parent] + R_world[parent] @ offset[i]``，
  ``R_world[i] = R_world[parent] @ R_local[i]``，根的 ``p`` 直接来自 motion 通道。
"""

from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Any, Sequence

import numpy as np
from scipy.spatial.transform import Rotation, Slerp

from ttm.common import constants as C

__all__ = ["SkeletonDef"]


@dataclass(frozen=True, eq=False)
class SkeletonDef:
    """23 关节骨架定义（BVH 系）。

    Attributes:
        bone_names: 关节名，顺序即数组维序（默认为 :data:`ttm.common.constants.BONE_ORDER`）。
        parents: ``int32[n]``，父关节索引，``-1`` 为根。
        rest_offsets_cm: ``float64[n,3]``，rest pose 下各关节相对父骨的偏移（cm）。
        euler_order: 欧拉序，固定 ``"YXZ"``。
        units: 长度单位，固定 ``"cm"``。
        up_axis: up 轴，固定 ``"Y"``。
    """

    bone_names: tuple[str, ...]
    parents: np.ndarray
    rest_offsets_cm: np.ndarray
    euler_order: str = C.EULER_ORDER
    units: str = C.UNITS_LENGTH
    up_axis: str = C.UP_AXIS

    # -- 构造 ---------------------------------------------------------------
    @classmethod
    def default(cls) -> "SkeletonDef":
        """返回基于 :mod:`ttm.common.constants` 常量的默认骨架。"""
        skeleton = cls(
            bone_names=tuple(C.BONE_ORDER),
            parents=C.parents_array(),
            rest_offsets_cm=C.rest_offsets_array(),
        )
        skeleton.validate()
        return skeleton

    @classmethod
    def from_parts(
        cls,
        bone_names: Sequence[str],
        parents: Sequence[int],
        rest_offsets_cm: np.ndarray | Sequence[Sequence[float]],
        *,
        euler_order: str = C.EULER_ORDER,
        units: str = C.UNITS_LENGTH,
        up_axis: str = C.UP_AXIS,
    ) -> "SkeletonDef":
        """由解析结果构造骨架（:mod:`ttm.data.bvh` 的入口）。

        Args:
            bone_names: 关节名序列（顺序 = BVH 文件中的层级顺序）。
            parents: 父关节索引序列。
            rest_offsets_cm: ``[n,3]`` rest 偏移（cm）。
            euler_order: 欧拉序。
            units: 长度单位。
            up_axis: up 轴。

        Returns:
            已校验的 :class:`SkeletonDef`。

        Raises:
            ValueError: 骨架不合法（关节数不符、父子关系非拓扑序等）。
        """
        offsets = np.asarray(rest_offsets_cm, dtype=np.float64)
        skeleton = cls(
            bone_names=tuple(str(name) for name in bone_names),
            parents=np.asarray(parents, dtype=np.int32),
            rest_offsets_cm=offsets,
            euler_order=str(euler_order),
            units=str(units),
            up_axis=str(up_axis),
        )
        skeleton.validate()
        return skeleton

    # -- 基本属性 -----------------------------------------------------------
    @property
    def n_joints(self) -> int:
        """关节数。"""
        return len(self.bone_names)

    @property
    def parents_list(self) -> list[int]:
        """父关节索引的 Python 列表（便于序列化）。"""
        return [int(item) for item in self.parents]

    @property
    def rest_offsets_list(self) -> list[list[float]]:
        """rest 偏移的 Python 嵌套列表（便于写 Motion JSON）。"""
        return [[float(v) for v in row] for row in self.rest_offsets_cm]

    def bone_index(self, name: str) -> int:
        """返回关节名索引。

        Raises:
            KeyError: 关节名不存在。
        """
        try:
            return self.bone_names.index(name)
        except ValueError as exc:
            raise KeyError(f"骨架中不存在关节 {name!r}；已有 {self.bone_names}") from exc

    def parent_of(self, index: int) -> int:
        """返回第 ``index`` 个关节的父索引（``-1`` 为根）。"""
        return int(self.parents[index])

    def children_of(self, index: int) -> tuple[int, ...]:
        """返回第 ``index`` 个关节的直接子关节索引元组。"""
        return tuple(int(i) for i, p in enumerate(self.parents) if int(p) == index)

    # -- 校验 ---------------------------------------------------------------
    def validate(self) -> None:
        """校验骨架结构自洽。

        Raises:
            ValueError: 关节数不一致 / 多根 / 父索引越界 / 非拓扑序 / 偏移维度错误。
        """
        n = self.n_joints
        if n == 0:
            raise ValueError("骨架为空")
        if len(self.parents) != n:
            raise ValueError(f"parents 长度 {len(self.parents)} != 关节数 {n}")
        if self.rest_offsets_cm.shape != (n, 3):
            raise ValueError(f"rest_offsets_cm 形状 {self.rest_offsets_cm.shape} != ({n}, 3)")

        roots = [i for i in range(n) if int(self.parents[i]) < 0]
        if len(roots) != 1:
            raise ValueError(f"骨架必须恰有 1 个根，实际 {len(roots)} 个（索引 {roots}）")
        if roots[0] != 0:
            raise ValueError(f"根关节必须是索引 0，实际 {roots[0]}")

        for idx in range(n):
            parent = int(self.parents[idx])
            if idx == 0:
                if parent != -1:
                    raise ValueError(f"根关节 parents[0] 应为 -1，实际 {parent}")
                continue
            if not 0 <= parent < n:
                raise ValueError(f"parents[{idx}]={parent} 越界")
            # FK 按索引顺序单趟递推，要求父索引严格小于子索引（拓扑序）
            if parent >= idx:
                raise ValueError(
                    f"parents[{idx}]={parent} 违反拓扑序（父索引必须 < 子索引），请按层级顺序排列关节"
                )
        if self.euler_order not in ("YXZ", "XYZ", "ZXY", "ZYX", "YZX", "XZY"):
            raise ValueError(f"不支持的欧拉序 {self.euler_order!r}")

    # -- 旋转与 FK ----------------------------------------------------------
    def local_rotmats(self, euler_rad: np.ndarray) -> np.ndarray:
        """把局部欧拉角（弧度）转成旋转矩阵。

        Args:
            euler_rad: ``[F, n, 3]`` 弧度制局部欧拉角。

        Returns:
            ``[F, n, 3, 3]`` float64 旋转矩阵。
        """
        angles = np.asarray(euler_rad, dtype=np.float64)
        if angles.ndim != 3 or angles.shape[1:] != (self.n_joints, 3):
            raise ValueError(
                f"euler_rad 形状应为 (F,{self.n_joints},3)，实际 {angles.shape}"
            )
        flat = angles.reshape(-1, 3)
        matrices = Rotation.from_euler(self.euler_order, flat, degrees=False).as_matrix()
        return matrices.reshape(angles.shape[0], self.n_joints, 3, 3)

    def world_rotmats(self, euler_rad: np.ndarray) -> np.ndarray:
        """由局部欧拉角递推世界旋转矩阵。

        Args:
            euler_rad: ``[F, n, 3]`` 弧度制局部欧拉角。

        Returns:
            ``[F, n, 3, 3]`` float64 世界旋转矩阵。
        """
        local = self.local_rotmats(euler_rad)
        frames = local.shape[0]
        world = np.empty_like(local)
        world[:, 0] = local[:, 0]
        for idx in range(1, self.n_joints):
            parent = int(self.parents[idx])
            world[:, idx] = world[:, parent] @ local[:, idx]
        assert world.shape[0] == frames
        return world

    def fk(self, euler_rad: np.ndarray, root_pos: np.ndarray) -> np.ndarray:
        """正运动学：由局部旋转 + 根位移求所有关节的世界坐标。

        Args:
            euler_rad: ``[F, n, 3]`` 弧度制局部欧拉角。
            root_pos: ``[F, 3]`` 根关节世界位置（cm）。

        Returns:
            ``[F, n, 3]`` float64 关节世界坐标（cm）。
        """
        local = self.local_rotmats(euler_rad)
        positions = np.asarray(root_pos, dtype=np.float64)
        if positions.shape != (local.shape[0], 3):
            raise ValueError(
                f"root_pos 形状应为 ({local.shape[0]},3)，实际 {positions.shape}"
            )
        world_rot = np.empty_like(local)
        world_pos = np.zeros((local.shape[0], self.n_joints, 3), dtype=np.float64)
        world_rot[:, 0] = local[:, 0]
        world_pos[:, 0] = positions
        for idx in range(1, self.n_joints):
            parent = int(self.parents[idx])
            world_rot[:, idx] = world_rot[:, parent] @ local[:, idx]
            world_pos[:, idx] = world_pos[:, parent] + world_rot[:, parent] @ self.rest_offsets_cm[idx]
        return world_pos

    def world_joint_pos(self, frame_euler_rad: np.ndarray, root_pos: np.ndarray) -> np.ndarray:
        """单帧正运动学。

        Args:
            frame_euler_rad: ``[n, 3]`` 弧度制局部欧拉角。
            root_pos: ``[3]`` 根关节世界位置（cm）。

        Returns:
            ``[n, 3]`` 关节世界坐标（cm）。
        """
        euler = np.asarray(frame_euler_rad, dtype=np.float64).reshape(1, self.n_joints, 3)
        root = np.asarray(root_pos, dtype=np.float64).reshape(1, 3)
        return self.fk(euler, root)[0]

    # -- 重采样 -------------------------------------------------------------
    @staticmethod
    def resampled_length(n_frames: int, src_fps: float, dst_fps: float) -> int:
        """计算重采样后的帧数（保持时长不变，且不超出源覆盖范围）。

        取 ``floor(duration * dst_fps) + 1`` 而不是四舍五入：后者在某些帧数下
        会让最后一个目标采样点落到源时间轴之外（``Slerp`` 会直接抛错）。

        Args:
            n_frames: 源帧数。
            src_fps: 源帧率。
            dst_fps: 目标帧率。

        Returns:
            目标帧数（至少 1）。
        """
        if n_frames <= 0:
            return 0
        if n_frames == 1:
            return 1
        duration = (n_frames - 1) / float(src_fps)
        steps = int(math.floor(duration * float(dst_fps) + 1e-6))
        return max(1, steps + 1)

    def resample(
        self,
        data: np.ndarray,
        src_fps: float,
        dst_fps: float,
        *,
        axis: int = 0,
        n_dst: int | None = None,
    ) -> np.ndarray:
        """对**任意**沿时间轴排列的数组做线性插值重采样。

        帧率比恰为整数（120→30 为 4×）时，目标采样点精确落在源采样点上，
        结果与直接抽取等价（误差 ~1e-16）。

        Args:
            data: 源数组，时间轴由 ``axis`` 指定。
            src_fps: 源帧率。
            dst_fps: 目标帧率。
            axis: 时间轴位置（默认 0）。
            n_dst: 目标帧数；``None`` 时按时长等比推算。

        Returns:
            重采样后的数组（时间轴长度变为 ``n_dst``）。
        """
        array = np.asarray(data, dtype=np.float64)
        if array.ndim == 0:
            raise ValueError("data 至少 1 维")
        array = np.moveaxis(array, axis, 0)
        n_src = array.shape[0]
        if n_dst is None:
            n_dst = self.resampled_length(n_src, src_fps, dst_fps)
        if n_src == 0 or n_dst == 0:
            out = np.zeros((0, *array.shape[1:]), dtype=np.float64)
            return np.moveaxis(out, 0, axis)
        if n_src == 1:
            out = np.repeat(array, n_dst, axis=0)
            return np.moveaxis(out, 0, axis)

        t_src = np.arange(n_src, dtype=np.float64) / float(src_fps)
        t_dst = np.arange(n_dst, dtype=np.float64) / float(dst_fps)
        flat = array.reshape(n_src, -1)
        out_flat = np.empty((n_dst, flat.shape[1]), dtype=np.float64)
        for col in range(flat.shape[1]):
            out_flat[:, col] = np.interp(t_dst, t_src, flat[:, col])
        out = out_flat.reshape(n_dst, *array.shape[1:])
        return np.moveaxis(out, 0, axis)

    def resample_rotations(
        self,
        euler_deg: np.ndarray,
        src_fps: float,
        dst_fps: float,
        *,
        n_dst: int | None = None,
    ) -> np.ndarray:
        """对欧拉角序列做**球面插值**重采样（四元数域），避免角度线性插值的收缩。

        Args:
            euler_deg: ``[F, n, 3]`` 角度制欧拉角。
            src_fps: 源帧率。
            dst_fps: 目标帧率。
            n_dst: 目标帧数；``None`` 时按时长推算。

        Returns:
            ``[F', n, 3]`` 角度制欧拉角（仍为 ``YXZ``）。
        """
        angles = np.asarray(euler_deg, dtype=np.float64)
        if angles.ndim != 3 or angles.shape[1:] != (self.n_joints, 3):
            raise ValueError(
                f"euler_deg 形状应为 (F,{self.n_joints},3)，实际 {angles.shape}"
            )
        n_src = angles.shape[0]
        if n_dst is None:
            n_dst = self.resampled_length(n_src, src_fps, dst_fps)
        if n_src <= 1 or n_dst <= 1:
            return self.resample(angles, src_fps, dst_fps, axis=0, n_dst=n_dst)

        t_src = np.arange(n_src, dtype=np.float64) / float(src_fps)
        t_dst = np.arange(n_dst, dtype=np.float64) / float(dst_fps)
        out = np.empty((n_dst, self.n_joints, 3), dtype=np.float64)
        for joint in range(self.n_joints):
            rotation = Rotation.from_euler(self.euler_order, angles[:, joint, :], degrees=True)
            interpolated = Slerp(t_src, rotation)(t_dst)
            out[:, joint, :] = interpolated.as_euler(self.euler_order, degrees=True)
        return out

    def resample_motion(
        self,
        euler_deg: np.ndarray,
        root_pos_cm: np.ndarray,
        src_fps: float,
        dst_fps: float,
        *,
        n_dst: int | None = None,
    ) -> tuple[np.ndarray, np.ndarray]:
        """把整段动作从 ``src_fps`` 重采样到 ``dst_fps``。

        旋转走球面插值，根位移走线性插值（两条时间轴严格对齐）。

        Args:
            euler_deg: ``[F, n, 3]`` 角度制欧拉角。
            root_pos_cm: ``[F, 3]`` 根位移（cm）。
            src_fps: 源帧率。
            dst_fps: 目标帧率。
            n_dst: 目标帧数；``None`` 时按时长推算。

        Returns:
            ``(euler_deg', root_pos_cm')``。
        """
        angles = np.asarray(euler_deg, dtype=np.float64)
        root_pos = np.asarray(root_pos_cm, dtype=np.float64)
        if angles.shape[0] != root_pos.shape[0]:
            raise ValueError(f"角度帧数 {angles.shape[0]} 与位移帧数 {root_pos.shape[0]} 不一致")
        if n_dst is None:
            n_dst = self.resampled_length(angles.shape[0], src_fps, dst_fps)
        new_angles = self.resample_rotations(angles, src_fps, dst_fps, n_dst=n_dst)
        new_root = self.resample(root_pos, src_fps, dst_fps, axis=0, n_dst=n_dst)
        return new_angles, new_root

    # -- 足接触 -------------------------------------------------------------
    def foot_contacts(
        self,
        world_pos: np.ndarray,
        *,
        fps: float = C.FPS,
        height_cm: float = C.CONTACT_HEIGHT_CM,
        speed_cmps: float = C.CONTACT_SPEED_CMPS,
        hysteresis_frames: int = C.CONTACT_HYSTERESIS_FRAMES,
    ) -> np.ndarray:
        """由关节世界坐标检测左/右足接触。

        判据（``TTM-ARCH.md`` §5.3 [144,146)）：足部高度 < ``height_cm`` **且**
        速度 < ``speed_cmps`` 记为接触；再经 ``hysteresis_frames`` 帧滞回抑制抖动。

        Args:
            world_pos: ``[F, n, 3]`` 关节世界坐标（cm）。
            fps: 帧率（用于速度差分）。
            height_cm: 高度阈值（cm）。
            speed_cmps: 速度阈值（cm/s）。
            hysteresis_frames: 状态翻转所需的连续一致帧数（>=1）。

        Returns:
            ``[F, 2]`` float32，列顺序 ``(contact_L, contact_R)``，取值 0.0 / 1.0。
        """
        positions = np.asarray(world_pos, dtype=np.float64)
        if positions.ndim != 3 or positions.shape[1] != self.n_joints:
            raise ValueError(f"world_pos 形状应为 (F,{self.n_joints},3)，实际 {positions.shape}")
        frames = positions.shape[0]
        if frames == 0:
            return np.zeros((0, 2), dtype=np.float32)

        out = np.zeros((frames, 2), dtype=np.float32)
        for slot, joint in enumerate(C.FOOT_CONTACT_JOINTS):
            if joint >= self.n_joints:
                continue
            track = positions[:, joint, :]
            height = track[:, 1]
            if frames > 1:
                deltas = np.diff(track, axis=0)
                speeds = np.linalg.norm(deltas, axis=1) * float(fps)
                speeds = np.concatenate([speeds[:1], speeds])  # 首帧前向填充
            else:
                speeds = np.zeros(1, dtype=np.float64)
            raw = (height < float(height_cm)) & (speeds < float(speed_cmps))
            out[:, slot] = self._hysteresis(raw, max(1, int(hysteresis_frames)))
        return out

    @staticmethod
    def _hysteresis(flags: np.ndarray, frames: int) -> np.ndarray:
        """对布尔序列做 ``frames`` 帧滞回（状态翻转需连续 ``frames`` 帧一致）。

        Args:
            flags: ``[F]`` 布尔序列。
            frames: 连续一致帧数阈值。

        Returns:
            ``[F]`` float32 0/1 序列。
        """
        result = np.zeros(flags.shape[0], dtype=np.float32)
        state = 0
        pending = 0
        pending_value = 0
        for idx, flag in enumerate(flags):
            value = 1 if bool(flag) else 0
            if value == state:
                pending = 0
                pending_value = value
            else:
                if pending_value == value:
                    pending += 1
                else:
                    pending_value = value
                    pending = 1
                if pending >= frames:
                    state = value
                    pending = 0
            result[idx] = float(state)
        return result

    # -- 元数据 -------------------------------------------------------------
    def as_metadata(self) -> dict[str, Any]:
        """返回可序列化的骨架元数据（写 npz / Motion JSON / 日志用）。"""
        return {
            "skeleton": C.SKELETON_NAME,
            "bone_names": list(self.bone_names),
            "parents": self.parents_list,
            "rest_offsets_cm": self.rest_offsets_list,
            "n_joints": self.n_joints,
            "euler_order": self.euler_order,
            "units": self.units,
            "up_axis": self.up_axis,
        }

    def __len__(self) -> int:
        """关节数。"""
        return self.n_joints

    def __repr__(self) -> str:  # pragma: no cover - 仅调试用
        hips_height = float(self.rest_offsets_cm.sum(axis=0)[1]) if self.n_joints else math.nan
        return (
            f"SkeletonDef(n_joints={self.n_joints}, euler={self.euler_order!r}, "
            f"units={self.units!r}, rest_chain_height={hips_height:.1f}cm)"
        )
