"""数据加载：npz 分片随机读取 + 固定窗口采样。

契约（``TTM-ARCH.md`` §5.5）：分片字段为
``feat[n,150,146] float32`` / ``mask[n,150] uint8`` / ``clip_id[n]<U80`` /
``win_idx[n]int16`` / ``layout_version`` / ``fps``。

设计要点：

* **分片惰性加载 + LRU 缓存**：每个分片 ~45 MB，缓存最近 ``cache_size`` 个，
  既不把 1 GB 全量读进内存，也避免每次取样都解压 npz；
* **只读 D 盘**：分片目录来自预处理产物，训练/推理**零次触碰 Z 盘**；
* **输出 channel-first**：``feat`` 转成 ``[146, 150]``，与 ``MotionVAE`` 的
  卷积输入布局一致；``mask`` 为 ``[150]``，供 loss 屏蔽 pad 帧；
* **split 过滤**：可选读 ``labels.jsonl`` 按 ``split`` 过滤样本（避免同动作跨集泄漏）。
"""

from __future__ import annotations

import os
import zipfile
from collections import OrderedDict
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Iterator, Sequence

import numpy as np
import orjson

from ttm import LAYOUT_VERSION
from ttm.common import constants as C

__all__ = [
    "MotionDataset",
    "ShardRecord",
    "auto_cache_size",
    "build_sample_weights",
    "expected_group_share",
    "load_labels_index",
    "make_balanced_sampler",
    "resolve_group_multiplier",
]

#: 默认可同时缓存在内存里的分片数。
DEFAULT_CACHE_SIZE = 2

#: 类别均衡采样的默认幂次。``0.5`` 表示按 ``1/sqrt(freq)`` 加权：既把长尾类抬起来，
#: 又不会像 ``1/freq`` 那样把只有 4 个窗口的武术细分抬到和 1128 窗口的「手指引路」
#: 同等频次（那等于过拟合到几个样本上）。
DEFAULT_BALANCE_POWER = 0.5

#: 单个样本权重相对最轻样本的最大倍数（即 ``max/min`` 比值上限）。
#:
#: 取值必须**大于真实频次跨度的 sqrt**，否则截断会把长尾削掉、反而抬高头部占比。
#: 实测频次跨度 9851:4（2462 倍），``sqrt = 49.6``，所以取 60 留余量；
#: 它只是防病态分布的保险丝（如 1:10000 那种），正常数据不会触发。
DEFAULT_MAX_WEIGHT_RATIO = 60.0


def resolve_group_multiplier(
    weights: np.ndarray, group_mask: np.ndarray, target_share: float
) -> float:
    """求「组内权重统一乘多少」才能让该组的期望采样占比等于目标值。

    期望占比：``s(m) = m·Wg / (m·Wg + Wo)``（``Wg`` = 组内权重之和，
    ``Wo`` = 组外权重之和）。它对 ``m`` 单调递增，且有闭式解::

        m = target · Wo / ((1 - target) · Wg)

    不用二分查找：闭式解精确、无迭代次数问题，且能在单测里直接反解验证。
    主理人要求「把 ``balance_power`` 调到武术采样占比约 15%」——**光调 power
    是调不准的**：power 同时改变所有类的权重，武术占比对 power 的响应不是 1:1，
    实测 power 从 0.5 调到 0.7 时武术占比只从 22% 动到 27%，还会顺带把别的
    长尾（如 191 类里的其他小类）一起抬起来。所以拆成「全局 power 定形状 +
    组倍率定占比」两步，占比是解出来的，不是试出来的。

    Args:
        weights: 逐样本基础权重（:func:`build_sample_weights` 的输出）。
        group_mask: 逐样本布尔掩码（``True`` = 目标组，如武术）。
        target_share: 目标期望占比，``(0, 1)`` 之间。

    Returns:
        组倍率 ``m``；组为空、组占满、或 ``target_share`` 越界时返回 ``1.0``。
    """
    if not 0.0 < float(target_share) < 1.0:
        return 1.0
    mask = np.asarray(group_mask, dtype=bool)
    values = np.asarray(weights, dtype=np.float64)
    if mask.shape != values.shape or not mask.any() or mask.all():
        return 1.0
    inside = float(values[mask].sum())
    outside = float(values[~mask].sum())
    if inside <= 0.0 or outside <= 0.0:
        return 1.0
    target = float(target_share)
    return target * outside / ((1.0 - target) * inside)


def expected_group_share(weights: np.ndarray, group_mask: np.ndarray) -> float:
    """给定逐样本权重，算某组的**期望采样占比**。

    ``WeightedRandomSampler(replacement=True)`` 下每个样本被抽中的概率正比于权重，
    所以期望占比就是 ``组内权重和 / 总权重和``。

    Args:
        weights: 逐样本权重。
        group_mask: 逐样本布尔掩码。

    Returns:
        期望占比（``0.0`` ~ ``1.0``）；总权重为 0 时返回 ``0.0``。
    """
    values = np.asarray(weights, dtype=np.float64)
    mask = np.asarray(group_mask, dtype=bool)
    if values.size == 0 or mask.shape != values.shape:
        return 0.0
    total = float(values.sum())
    if total <= 0.0:
        return 0.0
    return float(values[mask].sum()) / total


def build_sample_weights(
    labels: Sequence[str],
    *,
    power: float = DEFAULT_BALANCE_POWER,
    max_weight_ratio: float = DEFAULT_MAX_WEIGHT_RATIO,
    group_mask: np.ndarray | None = None,
    target_group_share: float = 0.0,
) -> np.ndarray:
    """按标签频次计算「类别均衡」的逐样本权重。

    权重形式为 ``freq[label] ** (-power)``，再整体归一到均值 1。

    Args:
        labels: 每个样本的标签（长度 = 样本数；空串视为一个独立类别「未标注」）。
        power: 幂次。``1.0`` → 完全均衡（各类期望占比相等）；``0.5`` → 按
            ``1/sqrt(freq)`` 折中；``0.0`` → 等权（等价于不均衡）。
        max_weight_ratio: 权重相对最轻样本的最大倍数，超出则截断。防止样本数只有个位数
            的类别被过采样到远高于其语义重要性。
        group_mask: 可选，逐样本布尔掩码，标记需要单独抬/压权重的组（如武术）。
        target_group_share: 该组的**目标期望采样占比**；``0`` 表示不调整。
            实际倍率由 :func:`resolve_group_multiplier` 闭式解出。

    Returns:
        形状 ``[n]`` 的 ``float64`` 权重数组；``labels`` 为空时返回空数组。
    """
    count = len(labels)
    if count == 0:
        return np.zeros(0, dtype=np.float64)

    frequencies: dict[str, int] = {}
    for label in labels:
        key = str(label) or "未标注"
        frequencies[key] = frequencies.get(key, 0) + 1

    exponent = float(power)
    raw = np.empty(count, dtype=np.float64)
    for index, label in enumerate(labels):
        key = str(label) or "未标注"
        raw[index] = float(frequencies[key]) ** (-exponent)

    # 归一化必须**在截断之前**选定基准，否则「先截断再除以均值」会把截断抵消掉
    # （实测 10000:1 的极端分布截到 5 倍后一除以均值又变回 9.99 倍）。
    # 这里把「最轻样本」定为 1，截断成 max/min ≤ cap 的比值上限，最后再归一到均值 1：
    # 比值上限保住，均值 1 也保住。
    lowest = float(raw.min())
    if lowest <= 0.0:  # 理论不可达（频次恒 ≥ 1，故 raw 恒 > 0）
        return np.ones(count, dtype=np.float64)
    relative = raw / lowest
    limit = max(1.0, float(max_weight_ratio))
    relative = np.clip(relative, 1.0, limit)
    weights = relative / float(relative.mean())

    # 组占比调整：在**基础权重之上**乘一个组内统一倍率，再整体归一到均值 1。
    # 必须放在截断之后：否则倍率会把 max_weight_ratio 的相对上限一起放大，
    # 等于悄悄取消了长尾保护。
    if group_mask is not None and target_group_share:
        mask = np.asarray(group_mask, dtype=bool)
        if mask.shape == weights.shape and mask.any() and not mask.all():
            multiplier = resolve_group_multiplier(weights, mask, target_group_share)
            weights = weights.copy()
            weights[mask] *= multiplier
            weights = weights / float(weights.mean())
    return weights



def make_balanced_sampler(
    weights: np.ndarray, *, num_samples: int | None = None, generator: Any | None = None
) -> Any:
    """由逐样本权重构造 ``torch.utils.data.WeightedRandomSampler``。

    Args:
        weights: :func:`build_sample_weights` 的输出。
        num_samples: 每个 epoch 采多少样本（默认 = 样本总数）。
        generator: 可选的 ``torch.Generator``（可复现）。

    Returns:
        ``WeightedRandomSampler`` 实例。

    Raises:
        RuntimeError: 未安装 torch。
    """
    try:
        from torch.utils.data import WeightedRandomSampler  # noqa: PLC0415
    except ModuleNotFoundError as exc:  # pragma: no cover - 无 torch 环境
        raise RuntimeError("类别均衡采样需要 torch") from exc

    tensor = weights
    try:
        import torch  # noqa: PLC0415

        tensor = torch.as_tensor(weights, dtype=torch.double)
    except ModuleNotFoundError:  # pragma: no cover
        pass
    return WeightedRandomSampler(
        tensor,
        num_samples=int(num_samples) if num_samples else int(len(weights)),
        replacement=True,
        generator=generator,
    )


def auto_cache_size(
    shard_dir: str | os.PathLike[str], *, cap: int = 128, floor: int = 4
) -> int:
    """按分片总数给出「不抖动」的 LRU 容量。

    为什么需要这个：训练时 ``shuffle=True``，一个 batch 的样本会散落在几乎所有分片上。
    如果缓存容量只有 2，每取一个样本就要把整个 npz（约 45 MB）重新读一遍并解压，
    实测吞吐会掉到 **30 秒/step**（GPU 利用率 1%）。把全部分片常驻内存后
    （45 分片 ≈ 2 GB），同样的 step 只要 17 ms。

    Args:
        shard_dir: ``shards/`` 目录。
        cap: 容量上限（避免异常多分片时吃爆内存）。
        floor: 容量下限。

    Returns:
        建议的 ``cache_size``。
    """
    count = len(list(Path(shard_dir).glob("*.npz")))
    if count <= 0:
        return floor
    return int(min(max(floor, count), cap))


@dataclass(frozen=True)
class ShardRecord:
    """分片内的一个样本位置。"""

    shard: Path
    index: int


def _npy_shape(path: Path, member: str) -> tuple[tuple[int, ...], np.dtype]:
    """只读 npz 里某个成员的 shape/dtype（不解压数据本体）。

    Args:
        path: npz 文件路径。
        member: 成员名（如 ``"feat.npy"``）。

    Returns:
        ``(shape, dtype)``。

    Raises:
        KeyError: 成员不存在。
        ValueError: 头部格式无法解析。
    """
    with zipfile.ZipFile(path) as archive:
        names = set(archive.namelist())
        if member not in names:
            raise KeyError(f"{path} 缺少成员 {member}（已有 {sorted(names)}）")
        with archive.open(member) as handle:
            version = np.lib.format.read_magic(handle)
            if version == (1, 0):
                shape, _fortran, dtype = np.lib.format.read_array_header_1_0(handle)
            elif version == (2, 0):
                shape, _fortran, dtype = np.lib.format.read_array_header_2_0(handle)
            else:  # pragma: no cover - numpy 未来版本
                raise ValueError(f"不支持的 .npy 版本 {version}")
    return tuple(int(item) for item in shape), np.dtype(dtype)


def load_labels_index(path: str | os.PathLike[str]) -> dict[str, dict[str, Any]]:
    """读 ``labels.jsonl`` 并建 ``clip_id → 记录`` 映射。

    Args:
        path: labels.jsonl 路径。

    Returns:
        映射字典；文件不存在时返回空字典。
    """
    source = Path(path)
    mapping: dict[str, dict[str, Any]] = {}
    if not source.is_file():
        return mapping
    with source.open("rb") as handle:
        for line in handle:
            text = line.strip()
            if not text:
                continue
            record = orjson.loads(text)
            clip_id = str(record.get("clip_id", ""))
            if clip_id:
                mapping[clip_id] = record
    return mapping


class _ShardCache:
    """分片数组的 LRU 缓存。"""

    def __init__(self, capacity: int = DEFAULT_CACHE_SIZE) -> None:
        self.capacity = max(1, int(capacity))
        self._cache: OrderedDict[Path, dict[str, np.ndarray]] = OrderedDict()

    def get(self, path: Path) -> dict[str, np.ndarray]:
        """取一个分片的数组字典（命中缓存则直接返回）。"""
        if path in self._cache:
            self._cache.move_to_end(path)
            return self._cache[path]
        with np.load(path) as payload:
            arrays = {key: np.asarray(payload[key]) for key in payload.files}
        self._cache[path] = arrays
        while len(self._cache) > self.capacity:
            self._cache.popitem(last=False)
        return arrays


class MotionDataset:
    """npz 分片数据集（map-style，可直接喂 DataLoader）。

    Attributes:
        window: 每个样本的帧数（150）。
        feature_dim: 特征维数（146）。
        cache_size: 分片 LRU 容量。
    """

    def __init__(
        self,
        shard_dir: str | os.PathLike[str],
        *,
        window: int = C.WINDOW,
        feature_dim: int = C.FEATURE_DIM,
        labels_path: str | os.PathLike[str] | None = None,
        split: str | None = None,
        cache_size: int = DEFAULT_CACHE_SIZE,
        with_torch: bool = True,
        class_balanced: bool = False,
        target_wushu_ratio: float = 0.0,
        skip_empty_windows: bool = True,
    ) -> None:
        """初始化（扫描分片并建立索引，不加载数据本体）。

        Args:
            shard_dir: ``shards/`` 目录。
            window: 期望窗口帧数（用于校验）。
            feature_dim: 期望特征维数（用于校验）。
            labels_path: 可选，``labels.jsonl``（用于 split 过滤与文本条件）。
            split: ``"train"`` / ``"val"`` / ``"test"`` / ``None``（不过滤）。
            cache_size: 分片 LRU 容量。
            with_torch: 为 ``False`` 时 ``__getitem__`` 返回 numpy 数组（便于无 torch 测试）。
            class_balanced: 为 ``True`` 时 :meth:`sampler` 返回按 ``1/sqrt(freq)``
                加权的采样器。**只在训练集启用**，验证/测试集必须关掉，否则
                指标会被重加权，失去可比性。
            skip_empty_windows: 跳过 ``mask`` 全 0 的窗口。

                **这不是保险丝，是修过一个真 bug 之后的防线**：旧版尾窗起点取
                ``win_idx * step``，当 ``step > 剩余帧`` 时会越界，实测产出 265 个
                全 padding 窗口（13696 的 1.9%）。它们对重建项贡献 0，却照样被编码、
                照样算 KL——等于往训练集里掺 1.9% 的纯噪声样本。
                数据侧已修（``features.slice_windows_arrays``），这里再兜一层：
                任何来源的零 mask 样本都不进训练/验证。
            target_wushu_ratio: 武术在训练 batch 中的**目标期望占比**（``0`` = 不调整）。
                数据侧武术只占 6.1%（220 个文件 / 每类 1.15 个，物理上限，配额策略
                改不动），而用户的核心诉求就是「武术动作」，所以只能在训练期加权补。
                倍率由 :func:`resolve_group_multiplier` 闭式解出，不是拍脑袋给的。

        Raises:
            FileNotFoundError: 目录不存在或没有分片。
            ValueError: 分片字段/形状不符合契约。
        """
        self.shard_dir = Path(shard_dir)
        self.window = int(window)
        self.feature_dim = int(feature_dim)
        self.cache_size = max(1, int(cache_size))
        self.with_torch = bool(with_torch)
        self.class_balanced = bool(class_balanced)
        self.target_wushu_ratio = float(target_wushu_ratio)
        self.skip_empty_windows = bool(skip_empty_windows)
        self.split = split

        if not self.shard_dir.is_dir():
            raise FileNotFoundError(f"分片目录不存在: {self.shard_dir}")
        self.shard_paths: list[Path] = sorted(self.shard_dir.glob("shard_*.npz"))
        if not self.shard_paths:
            raise FileNotFoundError(f"分片目录为空（未找到 shard_*.npz）: {self.shard_dir}")

        self.labels: dict[str, dict[str, Any]] = (
            load_labels_index(labels_path) if labels_path else {}
        )
        #: 因 ``mask`` 全 0 被跳过的窗口数（>0 说明上游产出了坏窗口，值得告警）
        self.skipped_empty = 0

        records: list[ShardRecord] = []
        clip_ids: list[str] = []
        win_indices: list[int] = []
        for path in self.shard_paths:
            shape, _dtype = _npy_shape(path, "feat.npy")
            if len(shape) != 3 or shape[1:] != (self.window, self.feature_dim):
                raise ValueError(
                    f"分片 {path.name} 的 feat 形状 {shape} 不符合契约 "
                    f"[n,{self.window},{self.feature_dim}]"
                )
            _npy_shape(path, "mask.npy")
            with np.load(path) as payload:
                ids = np.asarray(payload["clip_id"]).astype(str)
                wins = np.asarray(payload["win_idx"]).astype(np.int32)
                # mask 只有 [n,150] uint8，整个分片也就几十 KB，直接读出来做过滤，
                # 免得把全 padding 窗口当样本喂进模型。
                valid_frames = (
                    np.asarray(payload["mask"]).sum(axis=1)
                    if self.skip_empty_windows
                    else None
                )
                version = int(np.asarray(payload["layout_version"]).reshape(-1)[0])
                fps = int(np.asarray(payload["fps"]).reshape(-1)[0])
            if version != LAYOUT_VERSION:
                raise ValueError(
                    f"分片 {path.name} 的 layout_version={version} != {LAYOUT_VERSION}（需重新预处理）"
                )
            if fps != C.FPS:
                raise ValueError(
                    f"分片 {path.name} 的 fps={fps} != {C.FPS}（分片必须是 30fps 特征）"
                )
            for local in range(shape[0]):
                if valid_frames is not None and int(valid_frames[local]) <= 0:
                    self.skipped_empty += 1
                    continue
                record = ShardRecord(shard=path, index=local)
                clip_id = str(ids[local])
                if split is not None:
                    info = self.labels.get(clip_id)
                    if info is None:
                        continue
                    if str(info.get("split", "train")) != split:
                        continue
                records.append(record)
                clip_ids.append(clip_id)
                win_indices.append(int(wins[local]))

        self.records = records
        self.clip_ids = clip_ids
        self.win_indices = win_indices
        self._cache = _ShardCache(self.cache_size)

    # -- 基本属性 -----------------------------------------------------------
    def __len__(self) -> int:
        """样本数。"""
        return len(self.records)

    # -- 类别均衡 -----------------------------------------------------------
    def sample_labels(self) -> list[str]:
        """返回逐样本的中文标签（无标签时为空串）。

        Returns:
            长度 = 样本数的标签列表。
        """
        output: list[str] = []
        for clip_id in self.clip_ids:
            info = self.labels.get(clip_id, {})
            output.append(str(info.get("zh_label", "")))
        return output

    def class_weights(
        self,
        *,
        power: float = DEFAULT_BALANCE_POWER,
        max_weight_ratio: float = DEFAULT_MAX_WEIGHT_RATIO,
        target_wushu_ratio: float | None = None,
    ) -> np.ndarray:
        """逐样本的类别均衡权重（见 :func:`build_sample_weights`）。

        Args:
            power: 幂次，``0.5`` 表示 ``1/sqrt(freq)``。
            max_weight_ratio: 权重相对均值的最大倍数。
            target_wushu_ratio: 武术目标期望占比；``None`` 时用构造参数。

        Returns:
            形状 ``[n]`` 的权重数组。
        """
        target = self.target_wushu_ratio if target_wushu_ratio is None else float(target_wushu_ratio)
        mask = self.wushu_mask() if target else None
        return build_sample_weights(
            self.sample_labels(),
            power=power,
            max_weight_ratio=max_weight_ratio,
            group_mask=mask,
            target_group_share=target if mask is not None else 0.0,
        )

    def wushu_mask(self) -> np.ndarray:
        """逐样本的「是否武术」掩码。

        优先读 ``labels.jsonl`` 的 ``is_wushu`` 字段（重标签后处理写入，最可靠）；
        缺失时退化为看 ``category_l1`` 是否含 ``wushu``（``4.wushu`` 实测就是这个
        写法；不能按 ``4.`` 前缀判断，会把 ``4.1.Box_Ing`` / ``4.3.Equipment_Exercises``
        算进武术）。

        Returns:
            形状 ``[n]`` 的 ``bool`` 数组。
        """
        mask = np.zeros(len(self.clip_ids), dtype=bool)
        for index, clip_id in enumerate(self.clip_ids):
            info = self.labels.get(clip_id, {})
            flag = info.get("is_wushu")
            if flag is None:
                flag = "wushu" in str(info.get("category_l1", "")).lower()
            mask[index] = bool(flag)
        return mask

    def data_group_share(self, mask: np.ndarray) -> float:
        """某组在**数据里**的样本占比（与采样占比对照用）。"""
        values = np.asarray(mask, dtype=bool)
        if values.size == 0:
            return 0.0
        return float(values.sum()) / float(values.size)

    def sampler(self, *, num_samples: int | None = None, generator: Any | None = None) -> Any:
        """构造用于 ``DataLoader(sampler=...)`` 的采样器。

        训练集建议 ``class_balanced=True``；验证/测试集保持默认的随机顺序遍历
        （关掉重加权，否则 metric 不可比）。

        Args:
            num_samples: 每个 epoch 的采样数（默认 = 样本总数）。
            generator: 可选的 ``torch.Generator``。

        Returns:
            ``class_balanced=True`` 时返回 ``WeightedRandomSampler``，
            否则返回 ``None``（由 DataLoader 自行 ``shuffle``）。
        """
        if not self.class_balanced or len(self.records) == 0:
            return None
        return make_balanced_sampler(
            self.class_weights(), num_samples=num_samples, generator=generator
        )

    def __repr__(self) -> str:  # pragma: no cover - 调试用
        return (
            f"MotionDataset(shards={len(self.shard_paths)}, samples={len(self)}, "
            f"split={self.split!r}, window={self.window})"
        )

    # -- 取样 ---------------------------------------------------------------
    def __getitem__(self, index: int) -> dict[str, Any]:
        """取一个样本。

        Args:
            index: 样本序号。

        Returns:
            字典，键为 ``feat``（``[146,150]``）、``mask``（``[150]``）、
            ``clip_id``、``win_idx``，以及（有标签时）``zh_label`` / ``text`` / ``zh_texts``。
        """
        record = self.records[index]
        arrays = self._cache.get(record.shard)
        feat = np.asarray(arrays["feat"][record.index], dtype=np.float32)  # [W,146]
        mask = np.asarray(arrays["mask"][record.index], dtype=np.uint8)  # [W]
        clip_id = str(self.clip_ids[index])
        info = self.labels.get(clip_id, {})
        texts = info.get("zh_texts") or []
        sample: dict[str, Any] = {
            "feat": np.ascontiguousarray(feat.T),  # channel-first [146, W]
            "mask": np.ascontiguousarray(mask),  # [W]
            "clip_id": clip_id,
            "win_idx": int(self.win_indices[index]),
            "zh_label": str(info.get("zh_label", "")),
            "zh_texts": list(texts),
            "text": str(texts[0]) if texts else str(info.get("zh_label", "")),
            "split": str(info.get("split", "train")),
        }
        if self.with_torch:
            import torch  # noqa: PLC0415 - 延迟导入，保证无 torch 环境可用 numpy 路径

            sample["feat"] = torch.from_numpy(sample["feat"])
            sample["mask"] = torch.from_numpy(sample["mask"]).float()
        return sample

    def __iter__(self) -> Iterator[dict[str, Any]]:
        """按顺序遍历（便于无 DataLoader 的调试）。"""
        for index in range(len(self)):
            yield self[index]

    # -- 辅助 ---------------------------------------------------------------
    def split_counts(self) -> dict[str, int]:
        """按 split 统计样本数（需提供 labels_path）。"""
        counts: dict[str, int] = {}
        for clip_id in self.clip_ids:
            name = str(self.labels.get(clip_id, {}).get("split", "unknown"))
            counts[name] = counts.get(name, 0) + 1
        return counts

    def feature_stats(self, max_samples: int = 64) -> dict[str, float]:
        """抽样检查特征的数值范围（调试/自检用）。

        Args:
            max_samples: 最多抽样多少个样本。

        Returns:
            ``{"min":..., "max":..., "mean":..., "std":...}``。
        """
        count = min(max_samples, len(self))
        if count == 0:
            return {"min": 0.0, "max": 0.0, "mean": 0.0, "std": 0.0}
        stacked = np.stack([self[index]["feat"] for index in range(count)], axis=0)
        return {
            "min": float(stacked.min()),
            "max": float(stacked.max()),
            "mean": float(stacked.mean()),
            "std": float(stacked.std()),
        }


def collate_samples(batch: Sequence[dict[str, Any]]) -> dict[str, Any]:
    """把样本列表拼成 batch（DataLoader 的 ``collate_fn``）。

    同时兼容 :class:`MotionDataset` 的 ``with_torch=True``（返回张量）与
    ``with_torch=False``（返回 numpy 数组）两种模式。

    Args:
        batch: :meth:`MotionDataset.__getitem__` 的返回列表。

    Returns:
        含 ``feat[B,146,T]`` / ``mask[B,T]`` / ``clip_id[list]`` / ``text[list]`` 的字典。
    """
    import torch  # noqa: PLC0415

    if not batch:
        return {
            "feat": torch.empty(0),
            "mask": torch.empty(0),
            "clip_id": [],
            "text": [],
            "zh_texts": [],
            "win_idx": [],
        }
    feats = torch.stack([torch.as_tensor(item["feat"]) for item in batch], dim=0)
    masks = torch.stack([torch.as_tensor(item["mask"]) for item in batch], dim=0)
    return {
        "feat": feats,
        "mask": masks,
        "clip_id": [item["clip_id"] for item in batch],
        "text": [item.get("text", "") for item in batch],
        "zh_texts": [item.get("zh_texts", []) for item in batch],
        "win_idx": [item.get("win_idx", 0) for item in batch],
    }
