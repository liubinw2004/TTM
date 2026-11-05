"""窗口融合与根轨迹连续化（**最容易翻车的一环**，独立成模块便于单测）。

长动作走滑窗生成，相邻窗口在重叠区（默认 30 帧）必须平滑过渡：

1. **特征级融合**：重叠区用线性（Hann 型）权重交叉淡化，``heading`` 的
   ``(sinθ, cosθ)`` 融合后重新归一化，足接触取「或」再二值化；
2. **全局积分**：融合后的特征**只积分一次**（:meth:`MotionFeature.to_rotations_root`），
   所以根位移天然连续——若各窗口各自积分再拼接，接缝处必然出现位移跳变；
3. **接缝速度连续性**：即便全局积分，重叠区两窗速度不一致时接缝处仍会有速度台阶，
   因此提供 :func:`seam_velocity_jumps` 量化，并可用 :func:`smooth_root_track`
   （EMA，强度 ``runtime.smooth``）进一步平滑根轨迹。

测试断言见 ``tests/test_sampling.py``。
"""

from __future__ import annotations

from typing import Sequence

import numpy as np

from ttm.common import constants as C

__all__ = [
    "blend_feature_windows",
    "expected_frames",
    "seam_velocity_jumps",
    "smooth_root_track",
    "window_starts",
]


def window_starts(
    n_windows: int, *, window: int = C.WINDOW, overlap: int = C.OVERLAP
) -> list[int]:
    """各窗口在总时间轴上的起始帧（步长 = window − overlap）。"""
    stride = max(1, int(window) - int(overlap))
    return [index * stride for index in range(max(0, int(n_windows)))]


def expected_frames(
    n_windows: int, *, window: int = C.WINDOW, overlap: int = C.OVERLAP
) -> int:
    """``n_windows`` 个窗口融合后的总帧数。"""
    count = max(1, int(n_windows))
    return int(int(window) + (count - 1) * (max(1, int(window) - int(overlap))))


def _ramp(length: int, *, rising: bool) -> np.ndarray:
    """长度 ``length`` 的线性权重（端点含 0/1，避免重叠区出现硬台阶）。"""
    if length <= 0:
        return np.zeros(0, dtype=np.float64)
    if length == 1:
        return np.ones(1, dtype=np.float64)
    base = np.linspace(0.0, 1.0, length, dtype=np.float64)
    return base if rising else base[::-1].copy()


def blend_feature_windows(
    windows: Sequence[np.ndarray],
    *,
    overlap: int = C.OVERLAP,
    window: int = C.WINDOW,
) -> np.ndarray:
    """把多个窗口的特征（``[W,146]``）在重叠区交叉淡化后拼成一条长序列。

    融合策略：

    * 重叠区按「前窗权重 1−w、后窗权重 w」做线性交叉淡化（``w`` 由 0→1）；
    * ``heading``（dim 3:5）融合后重新归一化，保证仍是单位圆上的方向；
    * ``foot_contact``（dim 144:146）先融合再按 0.5 阈值二值化。

    Args:
        windows: 每项 ``[window,146]``（帧数可略小，按实际长度处理）。
        overlap: 重叠帧数。
        window: 窗口帧数（仅用于形状校验与默认步长）。

    Returns:
        ``[T_total,146]`` float32。

    Raises:
        ValueError: 窗口为空或特征维度不是 146。
    """
    if not windows:
        raise ValueError("windows 为空，无法融合")
    arrays = [np.asarray(item, dtype=np.float64) for item in windows]
    for index, array in enumerate(arrays):
        if array.ndim != 2 or array.shape[1] != C.FEATURE_DIM:
            raise ValueError(
                f"第 {index} 个窗口特征形状应为 [W,{C.FEATURE_DIM}]，实际 {array.shape}"
            )
    if len(arrays) == 1:
        return arrays[0].astype(np.float32)

    overlap = max(0, int(overlap))
    total = expected_frames(len(arrays), window=window, overlap=overlap)
    acc = np.zeros((total, C.FEATURE_DIM), dtype=np.float64)
    weight = np.zeros((total, 1), dtype=np.float64)
    smooth_dims = [dim for dim in C.NORM_DIMS]

    for index, array in enumerate(arrays):
        start = index * max(1, int(window) - overlap)
        length = array.shape[0]
        if start + length > total:  # 末窗可能略长，截断到总长
            length = total - start
            array = array[:length]
        if length <= 0:
            continue
        weights = np.ones(length, dtype=np.float64)
        if overlap > 0 and index > 0:
            head = min(overlap, length)
            weights[:head] = _ramp(head, rising=True)
        if overlap > 0 and index < len(arrays) - 1:
            tail = min(overlap, length)
            weights[length - tail :] = np.minimum(
                weights[length - tail :], _ramp(tail, rising=False)
            )
        acc[start : start + length] += array * weights[:, None]
        weight[start : start + length] += weights[:, None]

    blended = acc / np.clip(weight, 1e-8, None)

    # heading 重新归一化（融合后不再是单位向量）
    heading = blended[:, C.FEAT_LAYOUT["heading"]]
    norm = np.linalg.norm(heading, axis=1, keepdims=True)
    blended[:, C.FEAT_LAYOUT["heading"]] = heading / np.clip(norm, 1e-8, None)

    # 足接触二值化
    contacts = blended[:, C.FEAT_LAYOUT["foot_contact"]]
    blended[:, C.FEAT_LAYOUT["foot_contact"]] = (contacts >= 0.5).astype(np.float64)

    # 连续维做一次 3 帧滑动平均，压掉重叠区的权重锯齿
    for dim in smooth_dims:
        column = blended[:, dim]
        kernel = np.ones(3, dtype=np.float64) / 3.0
        padded = np.pad(column, (1, 1), mode="edge")
        blended[:, dim] = np.convolve(padded, kernel, mode="valid")
    return blended.astype(np.float32)


def seam_velocity_jumps(
    root_pos: np.ndarray,
    n_windows: int,
    *,
    fps: int = C.FPS,
    window: int = C.WINDOW,
    overlap: int = C.OVERLAP,
    head_trim: int = 0,
) -> np.ndarray:
    """返回每个接缝处的根速度台阶（cm/s，接缝带内**最大单帧速度变化**）。

    **接缝在哪**：第 ``i`` 个接缝是第 ``i+1`` 个窗口的起点，即帧
    ``(i+1) * (window − overlap)``。注意不是 ``i * stride``——后者在第 0 个接缝上
    指的是**片段开头**（帧 0），实测会把「生成头两帧的启动伪影」误判成接缝问题
    （174.2 cm/s 全部落在 frame 2，而真正接缝附近的单帧变化只有 7~16 cm/s）。

    **带子取多宽**：只取两个窗口**同时在贡献**的重叠区
    ``[start_i − 1, start_i + overlap]``。这个区间之外只有单个窗口在说话，
    那里的跳变是模型自己动作的性质，不是拼接引入的——把它算进接缝指标，
    指标就不再是「接缝指标」了。

    Args:
        root_pos: ``[T,3]`` 根位移（cm）。
        n_windows: 窗口数。
        fps: 帧率。
        window: 窗口帧数。
        overlap: 重叠帧数。
        head_trim: 已从开头丢掉的帧数（F-17 裁头）。裁头会把所有接缝整体前移
            ``head_trim`` 帧，因此接缝帧号要同量回退，否则指标打在错误的位置上。

    Returns:
        ``[n_windows-1]`` 数组（cm/s）；``n_windows <= 1`` 时为空数组。
    """
    root_pos = np.asarray(root_pos, dtype=np.float64)
    if root_pos.ndim != 2 or root_pos.shape[1] != 3:
        raise ValueError(f"root_pos 形状应为 [T,3]，实际 {root_pos.shape}")
    frames = root_pos.shape[0]
    if frames < 3 or n_windows <= 1:
        return np.zeros(0, dtype=np.float64)
    velocity = np.gradient(root_pos, axis=0) * float(fps)  # [T,3] cm/s
    # steps[t] = v[t+1] - v[t]（单帧速度变化量）
    steps = np.linalg.norm(np.diff(velocity, axis=0), axis=1)  # [T-1]
    window = max(2, int(window))
    overlap = max(1, int(overlap))
    shift = max(0, int(head_trim))
    stride = max(1, window - overlap)
    jumps: list[float] = []
    for index in range(int(n_windows) - 1):
        start = (index + 1) * stride - shift  # 后窗第一帧 = 第 index 个接缝
        low = int(np.clip(start - 1, 0, steps.shape[0] - 1))
        high = int(np.clip(start + overlap, low + 1, steps.shape[0]))
        band = steps[low:high]
        jumps.append(float(band.max()) if band.size else 0.0)
    return np.asarray(jumps, dtype=np.float64)


def smooth_root_track(root_pos: np.ndarray, *, smooth: float = 0.2) -> np.ndarray:
    """对根轨迹做 EMA 平滑（消除接缝处的速度台阶）。

    ``smooth=0`` 表示不平滑；``smooth`` 越大平滑越强（上限 0.95）。

    Args:
        root_pos: ``[T,3]`` 根位移。
        smooth: 平滑强度（``runtime.smooth``）。

    Returns:
        ``[T,3]`` 平滑后的根位移（float64）。
    """
    root_pos = np.asarray(root_pos, dtype=np.float64)
    if root_pos.ndim != 2 or root_pos.shape[1] != 3:
        raise ValueError(f"root_pos 形状应为 [T,3]，实际 {root_pos.shape}")
    alpha = float(np.clip(smooth, 0.0, 0.95))
    if alpha <= 0.0 or root_pos.shape[0] < 2:
        return root_pos.copy()
    # 双向 EMA（前向 + 反向）避免引入相位滞后
    forward = root_pos.copy()
    for index in range(1, root_pos.shape[0]):
        forward[index] = alpha * forward[index - 1] + (1.0 - alpha) * root_pos[index]
    backward = forward.copy()
    for index in range(root_pos.shape[0] - 2, -1, -1):
        backward[index] = alpha * backward[index + 1] + (1.0 - alpha) * forward[index]
    return backward
