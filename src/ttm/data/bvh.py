"""BVH 解析器：流式文本解析 + 关节数校验 + 通道重排。

文件格式（已对 ``Z:/1000hours`` 实测确认）::

    HIERARCHY
    ROOT Hips
    {
        OFFSET 0 0 0
        CHANNELS 6 Xposition Yposition Zposition Yrotation Xrotation Zrotation
        JOINT Spine
        {
            OFFSET -0 8.08814 -2.57809
            CHANNELS 3 Yrotation Xrotation Zrotation
            ...
            End Site
            {
                OFFSET -0 15 0
            }
        }
        ...
    }
    MOTION
    Frames: 11802
    Frame Time: 0.00833333
    <每帧 total_channels 个浮点数>

设计要点（``TTM-ARCH.md`` §1.1 D1/D2、§5.1）：

* **单趟顺序读**：一次 ``open(..., "rb")``，头部逐行读（≤ :data:`MAX_HEADER_BYTES`），
  记下 MOTION 数据区的**字节偏移**；数据区按 :data:`READ_CHUNK_BYTES` 分块累计，
  最后**一次性向量化** ``np.fromstring``（逐行 ``split`` 是禁手，11,802 帧的武术
  文件会慢一个数量级）。全程不回头二次打开文件；
* **关节数严格校验**：``n_joints != 23`` 直接抛 :class:`InvalidBvh`
  （数据集里混有只有 ROOT 的 root-only 轨迹文件、空 Take 目录）；
* **容错**：不认识的 token 跳过不报错；坏文件由上层捕获并计入 ``failed.jsonl``；
* **通道重排**：即使文件里关节的遍历顺序与 :data:`ttm.common.constants.BONE_ORDER`
  不同，也按名字重排到契约顺序；root 名字允许与 ``Hips`` 不同（记录在
  :attr:`BvhDocument.root_name`）。
"""

from __future__ import annotations

import os
from dataclasses import dataclass
from pathlib import Path
from typing import Final, Sequence

import numpy as np
from scipy.spatial.transform import Rotation

from ttm.common import constants as C
from ttm.data.skeleton import SkeletonDef

__all__ = [
    "BvhDocument",
    "BvhHeader",
    "BvhParser",
    "InvalidBvh",
    "MAX_HEADER_BYTES",
    "READ_CHUNK_BYTES",
    "load_bvh",
    "peek_bvh",
]

#: 头部允许的最大字节数（实测头部约 3.5 KB，给足余量）。
MAX_HEADER_BYTES: Final[int] = 256 * 1024

#: MOTION 数据的读取块大小（1 MB）。
READ_CHUNK_BYTES: Final[int] = 1 << 20

#: 旋转通道名（顺序即 BVH 欧拉序 ``YXZ``）。
ROTATION_CHANNELS: Final[tuple[str, ...]] = ("Yrotation", "Xrotation", "Zrotation")

#: 位置通道名（仅根关节允许出现）。
POSITION_CHANNELS: Final[tuple[str, ...]] = ("Xposition", "Yposition", "Zposition")


class InvalidBvh(ValueError):
    """BVH 文件不合法（无法使用）时抛出。

    Attributes:
        path: 文件路径。
        reason: 人类可读的拒绝原因（会写进 ``failed.jsonl``）。
    """

    def __init__(self, path: str | os.PathLike[str], reason: str) -> None:
        self.path = str(path)
        self.reason = str(reason)
        super().__init__(f"BVH 不合法 [{Path(self.path).name}]: {self.reason}")


@dataclass(frozen=True)
class _JointNode:
    """解析中间态：BVH 层级中的一个关节。"""

    name: str
    offset: tuple[float, float, float] = (0.0, 0.0, 0.0)
    channels: tuple[str, ...] = ()
    parent: int = -1


@dataclass(frozen=True)
class _HeaderInfo:
    """文件头部读取结果（内部使用）。"""

    text: str
    n_frames: int
    frame_time: float
    data_offset: int


@dataclass(frozen=True)
class BvhHeader:
    """头部信息（扫描器只需它来判断文件是否可用）。"""

    path: str
    root_name: str
    joint_names: tuple[str, ...]
    n_joints: int
    n_frames: int
    frame_time: float
    total_channels: int

    @property
    def fps(self) -> float:
        """源帧率。"""
        if self.frame_time <= 0:
            return 0.0
        return 1.0 / self.frame_time

    @property
    def is_usable(self) -> bool:
        """是否满足本项目契约（23 关节 + 合法帧数/帧时间 + 72 通道）。"""
        return (
            self.n_joints == C.N_JOINTS
            and self.n_frames > 0
            and self.frame_time > 0
            and self.total_channels == 6 + 3 * (C.N_JOINTS - 1)
        )

    def rejection_reason(self) -> str:
        """返回不可用原因（可用时返回空串）。"""
        if self.n_joints != C.N_JOINTS:
            return f"n_joints={self.n_joints} != {C.N_JOINTS}（root-only 或非标准骨架）"
        if self.n_frames <= 0:
            return f"n_frames={self.n_frames}"
        if self.frame_time <= 0:
            return f"frame_time={self.frame_time}"
        expected = 6 + 3 * (C.N_JOINTS - 1)
        if self.total_channels != expected:
            return f"total_channels={self.total_channels} != {expected}"
        return ""


@dataclass(frozen=True)
class BvhDocument:
    """解析后的 BVH 文档（内部表示，BVH 坐标系）。

    Attributes:
        source_path: 源文件路径（只读盘上的位置，仅用于追溯）。
        skeleton: 骨架定义（23 关节，rest 偏移来自文件）。
        euler_deg: ``[F,23,3]`` 角度制局部欧拉角，顺序 ``(Y, X, Z)``。
        root_pos_cm: ``[F,3]`` 根关节世界位移（cm）。
        frame_time: 源 Frame Time（秒）。
        root_name: 文件里 ROOT 的名字（通常 ``Hips``）。
    """

    source_path: str
    skeleton: SkeletonDef
    euler_deg: np.ndarray
    root_pos_cm: np.ndarray
    frame_time: float
    root_name: str = "Hips"

    # -- 基本属性 -----------------------------------------------------------
    @property
    def n_frames(self) -> int:
        """帧数。"""
        return int(self.euler_deg.shape[0])

    @property
    def n_joints(self) -> int:
        """关节数（恒为 23）。"""
        return int(self.euler_deg.shape[1])

    @property
    def fps(self) -> float:
        """源帧率（由 Frame Time 换算）。"""
        return 1.0 / self.frame_time if self.frame_time > 0 else 0.0

    @property
    def duration_s(self) -> float:
        """时长（秒）。"""
        if self.n_frames <= 1 or self.frame_time <= 0:
            return 0.0
        return (self.n_frames - 1) * self.frame_time

    @property
    def euler_order(self) -> str:
        """欧拉序（固定 ``YXZ``）。"""
        return C.EULER_ORDER

    # -- 派生量 -------------------------------------------------------------
    def to_rotations_rad(self) -> np.ndarray:
        """局部欧拉角（弧度）。"""
        return np.deg2rad(self.euler_deg)

    def to_quaternions(self) -> np.ndarray:
        """局部旋转四元数 ``[F,23,4]``，顺序 ``xyzw``（scipy 原生顺序）。"""
        flat = Rotation.from_euler(C.EULER_ORDER, self.euler_deg.reshape(-1, 3), degrees=True)
        return flat.as_quat().reshape(self.n_frames, self.n_joints, 4)

    def world_positions(self) -> np.ndarray:
        """FK 得到 ``[F,23,3]`` 关节世界坐标（cm）。"""
        return self.skeleton.fk(self.to_rotations_rad(), self.root_pos_cm)

    def resample(self, dst_fps: float) -> tuple[np.ndarray, np.ndarray]:
        """把动作重采样到目标帧率。

        Args:
            dst_fps: 目标帧率（本项目固定 30）。

        Returns:
            ``(euler_deg', root_pos_cm')``。

        Raises:
            InvalidBvh: 源 Frame Time 非法。
        """
        src_fps = self.fps
        if src_fps <= 0:
            raise InvalidBvh(self.source_path, f"非法 Frame Time {self.frame_time}")
        return self.skeleton.resample_motion(self.euler_deg, self.root_pos_cm, src_fps, dst_fps)

    def meta_dict(self) -> dict[str, object]:
        """返回用于日志/索引的元信息。"""
        return {
            "source": self.source_path,
            "root_name": self.root_name,
            "n_frames": self.n_frames,
            "n_joints": self.n_joints,
            "fps": self.fps,
            "duration_s": self.duration_s,
            "frame_time": self.frame_time,
        }

    def __repr__(self) -> str:  # pragma: no cover - 调试用
        return (
            f"BvhDocument({Path(self.source_path).name}, frames={self.n_frames}, "
            f"joints={self.n_joints}, fps={self.fps:.2f})"
        )


class BvhParser:
    """BVH 文本解析器（无状态，可复用/可并发）。"""

    # -- 头部 ---------------------------------------------------------------
    @staticmethod
    def peek(path: str | os.PathLike[str]) -> BvhHeader:
        """只解析头部（关节数 / 帧数 / 帧时间），不读 MOTION 数据。

        扫描器用它快速淘汰 root-only 文件与空动作，避免把整个文件读进内存。

        Args:
            path: BVH 文件路径。

        Returns:
            :class:`BvhHeader`。

        Raises:
            InvalidBvh: 文件不可读 / 缺 HIERARCHY / 缺 MOTION / 缺帧数。
        """
        header = _read_header(path)
        joints = _parse_hierarchy(header.text, path)
        return BvhHeader(
            path=str(path),
            root_name=joints[0].name if joints else "",
            joint_names=tuple(node.name for node in joints),
            n_joints=len(joints),
            n_frames=header.n_frames,
            frame_time=header.frame_time,
            total_channels=sum(len(node.channels) for node in joints),
        )

    # -- 完整解析 -----------------------------------------------------------
    @staticmethod
    def parse(path: str | os.PathLike[str], *, strict: bool = True) -> BvhDocument:
        """完整解析一个 BVH 文件。

        Args:
            path: BVH 文件路径。
            strict: ``True``（默认）按本项目契约校验——必须 23 关节、名字集合
                与 :data:`ttm.common.constants.BONE_ORDER` 一致，并按契约顺序重排；
                ``False`` 时**只校验通道布局**（根 6 通道、其余 3 通道），关节名与
                顺序沿用文件里的，用于检查重定向后导出的 UE 骨名 BVH。

        Returns:
            :class:`BvhDocument`。

        Raises:
            InvalidBvh: 关节数/通道布局不符、帧数或数据缺失等。
        """
        header = _read_header(path)
        joints = _parse_hierarchy(header.text, path)
        _validate_joints(joints, path, strict=strict)

        euler_index, root_pos_index, total_channels = _build_channel_map(joints, path)
        if strict:
            canonical = _canonical_order(joints, path)
            bone_names: Sequence[str] = C.BONE_ORDER
            parents: Sequence[int] = C.PARENTS
        else:
            canonical = list(range(len(joints)))
            bone_names = tuple(node.name for node in joints)
            parents = tuple(int(node.parent) for node in joints)

        values = _read_motion_values(path, header.data_offset)
        if values.size == 0:
            raise InvalidBvh(path, "MOTION 数据为空")
        available_frames = values.size // total_channels
        if available_frames == 0:
            raise InvalidBvh(path, f"数据点数量 {values.size} 少于单帧通道数 {total_channels}")

        n_frames = (
            min(header.n_frames, available_frames) if header.n_frames > 0 else available_frames
        )
        data = values[: n_frames * total_channels].reshape(n_frames, total_channels)

        euler_deg = np.empty((n_frames, len(canonical), 3), dtype=np.float64)
        for canonical_idx, file_idx in enumerate(canonical):
            cols = euler_index[file_idx]
            euler_deg[:, canonical_idx, 0] = data[:, cols[0]]
            euler_deg[:, canonical_idx, 1] = data[:, cols[1]]
            euler_deg[:, canonical_idx, 2] = data[:, cols[2]]

        pos_cols = root_pos_index[canonical[0]]
        root_pos_cm = np.stack([data[:, col] for col in pos_cols], axis=1).astype(np.float64)

        skeleton = SkeletonDef.from_parts(
            bone_names=bone_names,
            parents=parents,
            rest_offsets_cm=_canonical_offsets(joints, canonical),
        )

        return BvhDocument(
            source_path=str(path),
            skeleton=skeleton,
            euler_deg=euler_deg,
            root_pos_cm=root_pos_cm,
            frame_time=header.frame_time,
            root_name=joints[0].name,
        )


# ---------------------------------------------------------------------------
# 内部：单趟读文件
# ---------------------------------------------------------------------------
def _read_header(path: str | os.PathLike[str]) -> _HeaderInfo:
    """单趟读取头部，返回头部文本、帧数、帧时间与数据区字节偏移。

    做法：以二进制模式逐行读，累计字节偏移；读到 ``MOTION`` 后继续读两行
    ``Frames:`` / ``Frame Time:``；其后的字节偏移即数据区起点。

    Args:
        path: 文件路径。

    Returns:
        :class:`_HeaderInfo`。

    Raises:
        InvalidBvh: 文件不可读 / 超过头部上限 / 缺 HIERARCHY 或 MOTION / 缺帧信息。
    """
    header_lines: list[str] = []
    consumed = 0
    n_frames = -1
    frame_time = -1.0
    data_offset = 0

    try:
        handle = open(path, "rb")
    except OSError as exc:
        raise InvalidBvh(path, f"无法打开文件: {exc}") from exc

    with handle:
        saw_motion = False
        try:
            while True:
                raw = handle.readline()
                if not raw:
                    if not saw_motion:
                        raise InvalidBvh(path, "缺少 MOTION 小节")
                    break
                consumed += len(raw)
                if consumed > MAX_HEADER_BYTES and not saw_motion:
                    raise InvalidBvh(
                        path, f"头部超过 {MAX_HEADER_BYTES} 字节仍未遇到 MOTION"
                    )
                text = raw.decode("utf-8", errors="replace").strip()
                if not text or text.startswith("#"):
                    continue
                lower = text.lower()
                if not saw_motion:
                    if lower.startswith("motion"):
                        saw_motion = True
                        continue
                    header_lines.append(text)
                    continue
                # MOTION 之后：先要 Frames / Frame Time，然后是数据
                if lower.startswith("frames:"):
                    n_frames = _parse_int(text, path, "Frames")
                    continue
                if lower.startswith("frame time"):
                    frame_time = _parse_float(text, path, "Frame Time")
                    data_offset = consumed
                    break
                if lower.startswith("frames"):
                    n_frames = _parse_int(text, path, "Frames")
                    continue
                # 直接进数据（缺 Frame Time 行）——交给下面的校验报错
                break
        except UnicodeDecodeError as exc:  # pragma: no cover
            raise InvalidBvh(path, f"文本解码失败: {exc}") from exc

    text_all = "\n".join(header_lines)
    if "hierarchy" not in text_all.lower():
        raise InvalidBvh(path, "缺少 HIERARCHY 小节")
    if n_frames < 0:
        raise InvalidBvh(path, "缺少 Frames 行")
    if frame_time <= 0:
        raise InvalidBvh(path, "缺少或非法的 Frame Time")
    return _HeaderInfo(
        text=text_all, n_frames=n_frames, frame_time=frame_time, data_offset=data_offset
    )


def _parse_int(text: str, path: str | os.PathLike[str], what: str) -> int:
    """解析 ``Key: 123`` 形式的整数字段。"""
    payload = text.split(":", 1)[-1].strip() if ":" in text else text.split()[-1]
    try:
        return int(float(payload))
    except ValueError as exc:
        raise InvalidBvh(path, f"{what} 行无法解析: {text!r}") from exc


def _parse_float(text: str, path: str | os.PathLike[str], what: str) -> float:
    """解析 ``Key: 0.0083`` 形式的浮点字段。"""
    payload = text.split(":", 1)[-1].strip() if ":" in text else text.split()[-1]
    try:
        return float(payload)
    except ValueError as exc:
        raise InvalidBvh(path, f"{what} 行无法解析: {text!r}") from exc


def _parse_hierarchy(header_text: str, path: str | os.PathLike[str]) -> list[_JointNode]:
    """解析 ``HIERARCHY`` 小节为关节列表（DFS 前序）。

    Args:
        header_text: 头部文本（不含 MOTION）。
        path: 文件路径（仅用于报错）。

    Returns:
        关节列表；索引即文件中的出现顺序（对树而言即拓扑序）。

    Raises:
        InvalidBvh: 层级语法损坏或没有 ROOT。
    """
    tokens = header_text.replace("{", " { ").replace("}", " } ").split()
    if not tokens:
        raise InvalidBvh(path, "HIERARCHY 为空")

    joints: list[_JointNode] = []
    stack: list[int] = []
    index = 0
    seen_root = False

    while index < len(tokens):
        token = tokens[index]
        upper = token.upper()

        if upper in ("ROOT", "JOINT"):
            if index + 2 >= len(tokens):
                raise InvalidBvh(path, f"{token} 之后缺少名称或左花括号")
            name = tokens[index + 1]
            brace = tokens[index + 2]
            if brace != "{":
                raise InvalidBvh(path, f"{token} {name} 之后缺少左花括号，实际 {brace!r}")
            parent = stack[-1] if stack else -1
            joints.append(_JointNode(name=name, parent=parent))
            stack.append(len(joints) - 1)
            seen_root = seen_root or upper == "ROOT"
            index += 3
            continue

        if upper == "END":
            # ``End Site { OFFSET x y z }`` —— 跳过整个块，不计入关节数
            depth = 0
            index += 1
            while index < len(tokens):
                if tokens[index] == "{":
                    depth += 1
                elif tokens[index] == "}":
                    depth -= 1
                    if depth == 0:
                        index += 1
                        break
                index += 1
            continue

        if upper == "OFFSET":
            if not stack:
                raise InvalidBvh(path, "OFFSET 出现在任何关节之外")
            if index + 3 >= len(tokens):
                raise InvalidBvh(path, "OFFSET 分量不足 3 个")
            try:
                offset = (
                    float(tokens[index + 1]),
                    float(tokens[index + 2]),
                    float(tokens[index + 3]),
                )
            except ValueError as exc:
                raise InvalidBvh(
                    path, f"OFFSET 分量不是数字: {tokens[index + 1:index + 4]}"
                ) from exc
            target = stack[-1]
            joints[target] = _JointNode(
                name=joints[target].name,
                offset=offset,
                channels=joints[target].channels,
                parent=joints[target].parent,
            )
            index += 4
            continue

        if upper == "CHANNELS":
            if not stack:
                raise InvalidBvh(path, "CHANNELS 出现在任何关节之外")
            if index + 1 >= len(tokens):
                raise InvalidBvh(path, "CHANNELS 之后缺少数量")
            try:
                count = int(tokens[index + 1])
            except ValueError as exc:
                raise InvalidBvh(
                    path, f"CHANNELS 数量不是整数: {tokens[index + 1]!r}"
                ) from exc
            if count < 1:
                raise InvalidBvh(path, f"CHANNELS 数量非法: {count}")
            names = tuple(tokens[index + 2 : index + 2 + count])
            if len(names) != count:
                raise InvalidBvh(path, f"CHANNELS 声明 {count} 个但实际给出 {len(names)} 个")
            target = stack[-1]
            joints[target] = _JointNode(
                name=joints[target].name,
                offset=joints[target].offset,
                channels=names,
                parent=joints[target].parent,
            )
            index += 2 + count
            continue

        if token == "}":
            if not stack:
                raise InvalidBvh(path, "花括号不匹配（多余的 }）")
            stack.pop()
            index += 1
            continue

        # 其余 token（HIERARCHY / 未知扩展）跳过
        index += 1

    if not joints:
        raise InvalidBvh(path, "HIERARCHY 中没有 ROOT/JOINT")
    if not seen_root:
        raise InvalidBvh(path, "HIERARCHY 中没有 ROOT")
    if stack:
        raise InvalidBvh(path, "花括号不匹配（缺少 }）")
    return joints


def _read_motion_values(path: str | os.PathLike[str], data_offset: int) -> np.ndarray:
    """从数据区偏移开始分块读取并一次性向量化解析。

    Args:
        path: 文件路径。
        data_offset: 数据区起始字节偏移（由 :func:`_read_header` 给出）。

    Returns:
        一维 float64 数组（顺序为 ``帧0通道0..n, 帧1通道0..n, ...``）。

    Raises:
        InvalidBvh: 文件被并发修改或读取失败。
    """
    chunks: list[bytes] = []
    try:
        with open(path, "rb") as handle:
            handle.seek(data_offset)
            while True:
                block = handle.read(READ_CHUNK_BYTES)
                if not block:
                    break
                chunks.append(block)
    except OSError as exc:
        raise InvalidBvh(path, f"读取 MOTION 数据失败: {exc}") from exc

    if not chunks:
        return np.zeros(0, dtype=np.float64)
    payload = b"".join(chunks).decode("utf-8", errors="replace")
    if not payload.strip():
        return np.zeros(0, dtype=np.float64)
    if "," in payload:
        payload = payload.replace(",", " ")
    return np.fromstring(payload, dtype=np.float64, sep=" ")


# ---------------------------------------------------------------------------
# 内部：契约校验与重排
# ---------------------------------------------------------------------------
def _validate_joints(
    joints: Sequence[_JointNode],
    path: str | os.PathLike[str],
    *,
    strict: bool = True,
) -> None:
    """校验关节数量与通道布局是否满足本项目契约。

    Args:
        joints: 解析得到的关节列表。
        path: 文件路径（仅用于报错）。
        strict: ``True`` 时额外校验「恰 23 关节 + 名字集合等于 ``BONE_ORDER``」；
            ``False`` 时跳过这两项，只保留通道布局校验（重定向后的 UE 骨名 BVH）。

    Raises:
        InvalidBvh: 关节数不为 23 / 名称集合不符 / 通道布局不符。
    """
    if strict:
        if len(joints) != C.N_JOINTS:
            raise InvalidBvh(
                path,
                f"n_joints={len(joints)} != {C.N_JOINTS}"
                "（root-only 轨迹文件或非标准骨架，按设计淘汰）",
            )

        names_ci = [node.name.strip().lower() for node in joints]
        expected_ci = [name.lower() for name in C.BONE_ORDER]
        if set(names_ci[1:]) != set(expected_ci[1:]):
            missing = sorted(set(expected_ci[1:]) - set(names_ci[1:]))
            extra = sorted(set(names_ci[1:]) - set(expected_ci[1:]))
            raise InvalidBvh(path, f"非根关节名不符；缺少 {missing}，多出 {extra}")
        if len(set(names_ci[1:])) != C.N_JOINTS - 1:
            raise InvalidBvh(path, "非根关节名存在重复")

    for order, node in enumerate(joints):
        channels = tuple(channel.strip() for channel in node.channels)
        if not channels:
            raise InvalidBvh(path, f"关节 {node.name} 缺少 CHANNELS 声明")
        if order == 0:
            if len(channels) != 6:
                raise InvalidBvh(
                    path,
                    f"ROOT {node.name} 的通道数应为 6（3 位置 + 3 旋转），实际 {len(channels)}",
                )
            if (
                tuple(channels[3:]) != ROTATION_CHANNELS
                or tuple(channels[:3]) != POSITION_CHANNELS
            ):
                raise InvalidBvh(
                    path,
                    f"ROOT 通道顺序应为 {POSITION_CHANNELS + ROTATION_CHANNELS}，实际 {channels}",
                )
        else:
            if len(channels) != 3:
                raise InvalidBvh(
                    path, f"关节 {node.name} 的通道数应为 3（仅旋转），实际 {len(channels)}"
                )
            if set(channels) != set(ROTATION_CHANNELS):
                raise InvalidBvh(
                    path, f"关节 {node.name} 的通道应为 {ROTATION_CHANNELS} 的排列，实际 {channels}"
                )


def _canonical_order(joints: Sequence[_JointNode], path: str | os.PathLike[str]) -> list[int]:
    """返回「契约顺序 → 文件内索引」的映射（长度 23）。

    Args:
        joints: 文件中的关节列表（DFS 前序）。
        path: 文件路径（仅用于报错）。

    Returns:
        列表，第 i 项是 ``BONE_ORDER[i]`` 在文件中的索引；索引 0 恒为根。

    Raises:
        InvalidBvh: 找不到某个契约关节。
    """
    by_name: dict[str, int] = {}
    for file_idx, node in enumerate(joints):
        by_name[node.name.strip().lower()] = file_idx
    order = [0]  # 根按位置对齐（名字可能不是 Hips）
    for name in C.BONE_ORDER[1:]:
        key = name.lower()
        if key not in by_name:
            raise InvalidBvh(path, f"缺少关节 {name}")
        order.append(by_name[key])
    return order


def _canonical_offsets(joints: Sequence[_JointNode], canonical: Sequence[int]) -> np.ndarray:
    """按契约顺序收集 rest 偏移 ``[23,3]``（cm）。"""
    offsets = np.zeros((C.N_JOINTS, 3), dtype=np.float64)
    for canonical_idx, file_idx in enumerate(canonical):
        offsets[canonical_idx] = np.asarray(joints[file_idx].offset, dtype=np.float64)
    return offsets


def _build_channel_map(
    joints: Sequence[_JointNode], path: str | os.PathLike[str]
) -> tuple[list[tuple[int, int, int]], list[tuple[int, int, int]], int]:
    """建立「关节 → 帧内通道列号」映射。

    Args:
        joints: 文件中的关节列表。
        path: 文件路径（仅用于报错）。

    Returns:
        ``(euler_index, root_pos_index, total_channels)``：

        * ``euler_index[j]`` 为关节 j 的 ``(Y, X, Z)`` 旋转所在列号；
        * ``root_pos_index[j]`` 为 ``(X, Y, Z)`` 位置列号（非根填 ``(-1,-1,-1)``）。

    Raises:
        InvalidBvh: 通道名不在允许集合内。
    """
    cursor = 0
    euler_index: list[tuple[int, int, int]] = []
    root_pos_index: list[tuple[int, int, int]] = []
    for order, node in enumerate(joints):
        columns: dict[str, int] = {}
        for channel in node.channels:
            name = channel.strip()
            if name not in ROTATION_CHANNELS and name not in POSITION_CHANNELS:
                raise InvalidBvh(path, f"关节 {node.name} 含未知通道 {name!r}")
            columns[name] = cursor
            cursor += 1

        rotation_cols = [columns[name] for name in ROTATION_CHANNELS if name in columns]
        if len(rotation_cols) != 3:
            raise InvalidBvh(path, f"关节 {node.name} 缺少旋转通道")
        euler_index.append((rotation_cols[0], rotation_cols[1], rotation_cols[2]))

        position_cols = [columns[name] if name in columns else -1 for name in POSITION_CHANNELS]
        if order > 0 and any(col >= 0 for col in position_cols):
            raise InvalidBvh(path, f"非根关节 {node.name} 不允许含位置通道")
        root_pos_index.append((position_cols[0], position_cols[1], position_cols[2]))
    return euler_index, root_pos_index, cursor


# ---------------------------------------------------------------------------
# 便捷函数
# ---------------------------------------------------------------------------
def load_bvh(path: str | os.PathLike[str]) -> BvhDocument:
    """:meth:`BvhParser.parse` 的函数式封装。"""
    return BvhParser.parse(path)


def peek_bvh(path: str | os.PathLike[str]) -> BvhHeader:
    """:meth:`BvhParser.peek` 的函数式封装。"""
    return BvhParser.peek(path)
