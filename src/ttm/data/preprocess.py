"""全量预处理：单趟流式读 BVH → 146 维特征 → 分片落盘（npz）+ 标签。

流程（``TTM-ARCH.md`` §4.1 时序图 A、§5.5）：

1. 读 ``index.jsonl``（T03 产物，已含 split/类别/窗口数）；
2. 对每个 clip：**一次**打开 BVH → 解析 → 120fps→30fps 重采样 → 提取 146 维特征
   → 按 150 帧非重叠切窗（尾窗 ≥75 帧保留，pad+mask）；
3. 原始特征追加进 ``shards/shard_XXXXX.npz``（**单趟顺序读 Z 盘，产物全落 D 盘**）；
4. 训练集窗口喂入流式统计器（只统计 ``NORM_DIMS`` 四个连续维）；
5. 全部处理完后，用 train split 的统计量把**已落盘的分片就地重写为归一化特征**
   （只在 D 盘读写，零额外 Z 盘 IO —— 这是「先写原始、再统一归一化」的两阶段
  写盘法，用于规避「统计量必须看过全量 train 才能算」与「单趟读 Z 盘」的冲突）；
6. 产出 ``norm_stats.npz``、``labels.jsonl``（窗口级，含中文标签与 8–15 条变体）、
   ``failed.jsonl``（追加，含 stage 字段）。

``Z:`` 是只读慢速网络盘：本模块对每个文件只 ``open`` 一次；调试请用
``scripts/02_preprocess.py --limit N``。
"""

from __future__ import annotations

import os
import threading
import time
from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Iterable, Iterator, Sequence

import numpy as np
import orjson
from tqdm import tqdm

from ttm import LAYOUT_VERSION
from ttm.common import constants as C
from ttm.common import paths as P
from ttm.common.config import TTMConfig
from ttm.common.logutil import get_logger
from ttm.data.bvh import BvhParser, InvalidBvh
from ttm.data.features import (
    FeatureExtractor,
    MotionFeature,
    NormStats,
    motion_quality_issue,
    trim_placeholder_frames,
    window_stride_for,
)
from ttm.data.labels_zh import LabelGenerator
from ttm.data.scan import read_index
from ttm.data.skeleton import SkeletonDef

__all__ = [
    "PreprocessPipeline",
    "PreprocessResult",
    "ShardWriter",
    "normalize_shards",
    "read_labels",
]

#: 默认并发数（Z 盘网络 RTT 摊薄）。
DEFAULT_WORKERS = 8

#: 失败率上限默认值：超过则整批视为失败（ARCH §5.1「错误处理」）。
#: 可在配置里用 ``data.max_failure_rate`` 覆盖（超小冒烟档样本极少，需放宽）。
MAX_FAILURE_RATE = 0.01

#: npz 分片字段集合（ARCH §5.5，**不得增删**）。
SHARD_FIELDS: tuple[str, ...] = (
    "feat",
    "mask",
    "clip_id",
    "win_idx",
    "layout_version",
    "fps",
)

#: clip_id 在 npz 中的最大字符数（唯一来源：:data:`ttm.common.constants.CLIP_ID_MAXLEN`）。
CLIP_ID_MAXLEN = C.CLIP_ID_MAXLEN


@dataclass
class PreprocessResult:
    """预处理统计结果。"""

    n_clips: int = 0
    n_windows: int = 0
    n_failed: int = 0
    n_skipped: int = 0
    shards: list[str] = field(default_factory=list)
    norm_stats: NormStats | None = None
    elapsed_s: float = 0.0
    max_failure_rate: float = MAX_FAILURE_RATE
    stats: dict[str, Any] = field(default_factory=dict)

    @property
    def failure_rate(self) -> float:
        """失败率（失败 clip 数 / 总 clip 数）。"""
        total = self.n_clips + self.n_failed
        return (self.n_failed / total) if total else 0.0

    def is_acceptable(self) -> bool:
        """失败率是否在允许范围内（默认 ≤ 1%，可由配置覆盖）。"""
        return self.failure_rate <= self.max_failure_rate


class ShardWriter:
    """``shards/shard_XXXXX.npz`` 写入器（按 ``shard_size`` 自动分片）。"""

    def __init__(
        self,
        out_dir: str | os.PathLike[str],
        *,
        shard_size: int = 512,
        fps: int = C.FPS,
        layout_version: int = LAYOUT_VERSION,
    ) -> None:
        """初始化。

        Args:
            out_dir: 分片输出目录。
            shard_size: 每个分片最多多少条样本。
            fps: 特征帧率（写入 npz 标量）。
            layout_version: 布局版本（写入 npz 标量）。
        """
        self.out_dir = Path(out_dir)
        self.shard_size = max(1, int(shard_size))
        self.fps = int(fps)
        self.layout_version = int(layout_version)

        self._feat: list[np.ndarray] = []
        self._mask: list[np.ndarray] = []
        self._clip_id: list[str] = []
        self._win_idx: list[int] = []
        self.written: list[Path] = []
        self._lock = threading.Lock()

    @property
    def pending(self) -> int:
        """当前缓冲区里的样本数。"""
        return len(self._feat)

    def add(self, clip_id: str, win_idx: int, feat: np.ndarray, mask: np.ndarray) -> None:
        """加入一条样本（窗口）。

        Args:
            clip_id: 窗口级 clip_id（``{clip}_win{XX}``）。
            win_idx: 窗口序号。
            feat: ``[W,146]`` 特征。
            mask: ``[W]`` 掩码。
        """
        array = np.asarray(feat, dtype=np.float32)
        mask_array = np.asarray(mask, dtype=np.uint8)
        if array.ndim != 2 or array.shape[1] != C.FEATURE_DIM:
            raise ValueError(f"feat 形状应为 [W,{C.FEATURE_DIM}]，实际 {array.shape}")
        if mask_array.shape != (array.shape[0],):
            raise ValueError(f"mask 形状应为 ({array.shape[0]},)，实际 {mask_array.shape}")
        with self._lock:
            self._feat.append(array)
            self._mask.append(mask_array)
            text_id = str(clip_id)
            if len(text_id) > CLIP_ID_MAXLEN:
                raise ValueError(
                    f"clip_id 过长（{len(text_id)} > {CLIP_ID_MAXLEN}），会被 <U{CLIP_ID_MAXLEN} 截断: {text_id}"
                )
            self._clip_id.append(text_id)
            self._win_idx.append(int(win_idx))
            if self.pending >= self.shard_size:
                # 注意：这里必须调用**无锁**内部实现，threading.Lock 不可重入
                self._flush_locked()

    def flush(self) -> Path | None:
        """把缓冲区写成一个 npz 分片，并清空缓冲区。

        Returns:
            写出的分片路径；缓冲区为空时返回 ``None``。
        """
        with self._lock:
            return self._flush_locked()

    def _flush_locked(self) -> Path | None:
        """落盘实现（**调用方必须已持有** ``self._lock``）。

        Returns:
            写出的分片路径；缓冲区为空时返回 ``None``。
        """
        if not self._feat:
            return None
        feat = np.stack(self._feat, axis=0).astype(np.float32)
        mask = np.stack(self._mask, axis=0).astype(np.uint8)
        clip_id = np.asarray(self._clip_id, dtype=f"<U{CLIP_ID_MAXLEN}")
        win_idx = np.asarray(self._win_idx, dtype=np.int16)
        target = P.resolve_out(
            self.out_dir / f"shard_{len(self.written):05d}.npz", mkparents=True
        )
        np.savez(
            target,
            feat=feat,
            mask=mask,
            clip_id=clip_id,
            win_idx=win_idx,
            layout_version=np.int32(self.layout_version),
            fps=np.int16(self.fps),
        )
        self.written.append(target)
        self._feat.clear()
        self._mask.clear()
        self._clip_id.clear()
        self._win_idx.clear()
        return target

    def close(self) -> list[Path]:
        """落盘剩余数据并返回全部分片路径。

        Returns:
            分片路径列表。
        """
        self.flush()
        return list(self.written)


class _RunningStats:
    """只针对 ``NORM_DIMS`` 的流式均值/方差统计器。"""

    def __init__(self) -> None:
        self.dims = tuple(C.NORM_DIMS)
        self.count = 0
        self.sums = np.zeros(len(self.dims), dtype=np.float64)
        self.sumsq = np.zeros(len(self.dims), dtype=np.float64)

    def update(self, feat: np.ndarray) -> None:
        """累积一个窗口（只取有效帧）。

        Args:
            feat: ``[W,146]`` 特征。
        """
        if feat.size == 0:
            return
        columns = np.asarray(feat, dtype=np.float64)[:, list(self.dims)]
        if columns.shape[0] == 0:
            return
        self.count += columns.shape[0]
        self.sums += columns.sum(axis=0)
        self.sumsq += np.square(columns).sum(axis=0)

    def to_norm_stats(self) -> NormStats:
        """转成 :class:`NormStats`。"""
        mean = np.zeros(C.FEATURE_DIM, dtype=np.float64)
        std = np.ones(C.FEATURE_DIM, dtype=np.float64)
        if self.count > 0:
            means = self.sums / self.count
            variances = np.maximum(self.sumsq / self.count - np.square(means), 0.0)
            stds = np.sqrt(variances)
            for offset, dim in enumerate(self.dims):
                mean[dim] = float(means[offset])
                std[dim] = float(stds[offset]) if stds[offset] > 1e-6 else 1.0
        return NormStats(mean=mean, std=std, count=self.count)


def normalize_shards(shard_paths: Sequence[Path], stats: NormStats) -> int:
    """把已落盘的分片就地重写为归一化特征（仅 D 盘读写）。

    Args:
        shard_paths: 分片路径列表。
        stats: 归一化统计量。

    Returns:
        重写的分片数。
    """
    dims = list(C.NORM_DIMS)
    for path in shard_paths:
        with np.load(path) as payload:
            missing = [key for key in SHARD_FIELDS if key not in payload]
            if missing:
                raise ValueError(f"分片 {path} 缺字段 {missing}")
            feat = np.asarray(payload["feat"], dtype=np.float32)
            mask = np.asarray(payload["mask"], dtype=np.uint8)
            clip_id = np.asarray(payload["clip_id"])
            win_idx = np.asarray(payload["win_idx"], dtype=np.int16)
            layout_version = int(np.asarray(payload["layout_version"]).reshape(-1)[0])
            fps = int(np.asarray(payload["fps"]).reshape(-1)[0])
        normalized = feat.astype(np.float64)
        for dim in dims:
            normalized[:, :, dim] = (normalized[:, :, dim] - stats.mean[dim]) / stats.std[dim]
        np.savez(
            path,
            feat=normalized.astype(np.float32),
            mask=mask,
            clip_id=clip_id,
            win_idx=win_idx,
            layout_version=np.int32(layout_version),
            fps=np.int16(fps),
        )
    return len(shard_paths)


def read_labels(path: str | os.PathLike[str]) -> list[dict[str, Any]]:
    """读 ``labels.jsonl``（测试/调试用）。"""
    records: list[dict[str, Any]] = []
    source = Path(path)
    if not source.is_file():
        return records
    with source.open("rb") as handle:
        for line in handle:
            text = line.strip()
            if text:
                records.append(orjson.loads(text))
    return records


@dataclass
class _ClipJob:
    """单个 clip 的处理结果（线程池返回）。"""

    entry: dict[str, Any]
    windows: list[tuple[np.ndarray, np.ndarray, int]]
    error: str = ""
    n_frames_30: int = 0
    trimmed_lead_frames: int = 0


class PreprocessPipeline:
    """把 ``index.jsonl`` 变成可训练的分片数据集。"""

    def __init__(
        self,
        cfg: TTMConfig,
        *,
        limit: int | None = None,
        workers: int = DEFAULT_WORKERS,
        logger: Any | None = None,
    ) -> None:
        """初始化。

        Args:
            cfg: 生效配置。
            limit: 最多处理多少个 clip（调试用）。
            workers: 并发线程数。
            logger: 可注入 logger。
        """
        self.cfg = cfg
        self.limit = int(limit) if limit else 0
        self.workers = max(1, int(workers))
        self.log = logger or get_logger("ttm.preprocess", script="02_preprocess")
        self.skeleton = SkeletonDef.default()
        self.extractor = FeatureExtractor(
            skeleton=self.skeleton,
            fps=cfg.data.fps,
            contact_height_cm=cfg.motion.contact_height_cm,
            contact_speed_cmps=cfg.motion.contact_speed_cmps,
        )
        self.labels = LabelGenerator()

    # -- 单文件处理 ---------------------------------------------------------
    def _process_one(self, entry: dict[str, Any]) -> _ClipJob:
        """解析 + 重采样 + 提特征 + 切窗（本函数在设计上是纯计算 + 一次文件读）。"""
        source = str(entry.get("source") or entry.get("path") or "")
        try:
            document = BvhParser.parse(source)
        except InvalidBvh as exc:
            return _ClipJob(entry=entry, windows=[], error=f"parse: {exc.reason}")
        except Exception as exc:  # noqa: BLE001 - 坏文件一律记失败并继续
            return _ClipJob(entry=entry, windows=[], error=f"parse: {type(exc).__name__}: {exc}")

        try:
            euler_deg, root_pos = document.resample(self.cfg.data.fps)
            # 首帧常是建模工具导出的 T-pose 占位帧（全 0 旋转 + 根在原点），
            # 与第 2 帧之间有上百 cm 瞬移，必须裁掉，否则产生虚假根速度。
            euler_deg, root_pos, trimmed = trim_placeholder_frames(euler_deg, root_pos)
            issue = motion_quality_issue(self.skeleton, euler_deg, root_pos)
            if issue:
                return _ClipJob(entry=entry, windows=[], error=issue, n_frames_30=int(euler_deg.shape[0]))
            feature = self.extractor.extract(euler_deg, root_pos)
            # 降密度步长必须与扫描期（scan._build_entry）算的一致，否则
            # index.jsonl 里的 n_windows 与实际落盘窗口数对不上，配额统计失真。
            stride = window_stride_for(
                feature.n_frames,
                window=self.cfg.data.window,
                long_clip_stride=self.cfg.data.long_clip_stride_frames,
            )
            windows = self.extractor.slice_windows(
                feature,
                window=self.cfg.data.window,
                min_tail=self.cfg.data.min_tail_frames,
                stride=stride,
            )
        except Exception as exc:  # noqa: BLE001
            return _ClipJob(
                entry=entry, windows=[], error=f"features: {type(exc).__name__}: {exc}"
            )
        return _ClipJob(
            entry=entry,
            windows=windows,
            n_frames_30=int(feature.n_frames),
            trimmed_lead_frames=int(trimmed),
        )

    # -- 主流程 -------------------------------------------------------------
    def run(self) -> PreprocessResult:
        """执行预处理。

        Returns:
            :class:`PreprocessResult`。

        Raises:
            FileNotFoundError: ``index.jsonl`` 不存在（需先跑 ``01_scan.py``）。
        """
        started = time.perf_counter()
        index_path = P.assert_write_allowed(self.cfg.index_path())
        if not index_path.is_file():
            raise FileNotFoundError(
                f"找不到 index.jsonl: {index_path}。请先运行 scripts/01_scan.py --config {self.cfg.source_path}"
            )

        entries = read_index(index_path)
        if self.limit > 0:
            entries = entries[: self.limit]
        if not entries:
            self.log.warning("index.jsonl 为空，无可处理数据")

        result = PreprocessResult(
            stats={"index": str(index_path)},
            max_failure_rate=float(self.cfg.data.max_failure_rate),
        )
        writer = ShardWriter(
            P.safe_makedirs(self.cfg.shards_path()),
            shard_size=self.cfg.data.shard_size,
            fps=self.cfg.data.fps,
        )
        train_stats = _RunningStats()
        label_rows: list[dict[str, Any]] = []
        failed_rows: list[dict[str, Any]] = []
        skipped_windows = 0
        trimmed_total = 0

        progress = tqdm(total=len(entries), desc="预处理", unit="clip", ncols=88)
        chunk = max(1, self.workers * 4)
        with ThreadPoolExecutor(max_workers=self.workers) as pool:
            for start in range(0, len(entries), chunk):
                batch = entries[start : start + chunk]
                for job in pool.map(self._process_one, batch):
                    if job.error:
                        failed_rows.append(
                            {
                                "path": job.entry.get("source", ""),
                                "stage": "preprocess",
                                "error": job.error,
                            }
                        )
                        result.n_failed += 1
                        progress.update(1)
                        continue

                    windows = job.windows
                    if not windows:
                        skipped_windows += 1
                        failed_rows.append(
                            {
                                "path": job.entry.get("source", ""),
                                "stage": "preprocess",
                                "error": f"no_window: n_frames@30fps={job.n_frames_30}",
                            }
                        )
                        progress.update(1)
                        continue

                    label, texts = self.labels.generate(
                        category_l1=str(job.entry.get("category_l1", "")),
                        category_l2=str(job.entry.get("category_l2", "")),
                        category_l3=str(job.entry.get("category_l3", "")),
                        stem=str(job.entry.get("actor", "")),
                    )
                    base_id = str(job.entry.get("clip_id", "clip"))
                    split = str(job.entry.get("split", "train"))
                    for feat_win, mask_win, win_idx in windows:
                        window_id = f"{base_id}_win{win_idx:02d}"
                        writer.add(window_id, win_idx, feat_win, mask_win)
                        if split == "train":
                            train_stats.update(feat_win)
                        label_rows.append(
                            {
                                "clip_id": window_id,
                                "clip_root": base_id,
                                "take_id": str(job.entry.get("take_id", "")),
                                "split": split,
                                "category_l1": str(job.entry.get("category_l1", "")),
                                "category_l2": str(job.entry.get("category_l2", "")),
                                "category_l3": str(job.entry.get("category_l3", "")),
                                "zh_label": label,
                                "zh_texts": texts,
                                "duration": round(feat_win.shape[0] / float(self.cfg.data.fps), 6),
                                "fps": self.cfg.data.fps,
                                "n_frames": int(feat_win.shape[0]),
                                "win_idx": int(win_idx),
                                "source": str(job.entry.get("source", "")),
                            }
                        )
                    result.n_clips += 1
                    result.n_windows += len(windows)
                    trimmed_total += job.trimmed_lead_frames
                    progress.update(1)
        progress.close()

        shards = writer.close()
        result.shards = [shard.name for shard in shards]
        result.n_skipped = skipped_windows

        # --- 归一化统计量（train split）
        norm_stats = train_stats.to_norm_stats()
        used_identity = norm_stats.count == 0
        if used_identity:
            self.log.warning("train split 没有可用窗口，norm_stats 退化为恒等（无归一化）")
        stats_path = norm_stats.save(self.cfg.norm_stats_path())
        result.norm_stats = norm_stats

        # --- 两阶段写盘的第 2 阶段：就地归一化已落盘的分片
        normalize_shards(shards, norm_stats)

        # --- 标签与失败日志
        labels_path = self._write_jsonl(self.cfg.labels_path(), label_rows)
        failed_path = self._write_jsonl(self.cfg.failed_path(), failed_rows, mode="ab")

        result.elapsed_s = time.perf_counter() - started
        result.stats.update(
            {
                "shards": len(shards),
                "labels": labels_path.as_posix(),
                "failed": failed_path.as_posix(),
                "norm_stats": stats_path.as_posix(),
                "norm_identity": used_identity,
                "layout_version": LAYOUT_VERSION,
                "tier": self.cfg.data.tier,
                "window": self.cfg.data.window,
                "fps": self.cfg.data.fps,
                "trimmed_lead_frames": trimmed_total,
                "failure_rate": result.failure_rate,
                "workers": self.workers,
            }
        )
        self.log.info(
            "预处理完成: %d clip / %d 窗口 / %d 分片 / 失败 %d（%.2f%%）/ 用时 %.1fs",
            result.n_clips,
            result.n_windows,
            len(shards),
            result.n_failed,
            result.failure_rate * 100,
            result.elapsed_s,
        )
        return result

    @staticmethod
    def _write_jsonl(
        path: str | os.PathLike[str], rows: Iterable[dict[str, Any]], *, mode: str = "wb"
    ) -> Path:
        """按行写 JSONL（orjson）。

        Args:
            path: 目标路径。
            rows: 记录迭代器。
            mode: ``"wb"`` 覆盖 / ``"ab"`` 追加。

        Returns:
            实际写出的路径。
        """
        target = P.resolve_out(path, mkparents=True)
        with target.open(mode) as handle:
            for row in rows:
                handle.write(orjson.dumps(row))
                handle.write(b"\n")
        return target

    # -- 便捷 ---------------------------------------------------------------
    def iter_features(self, entries: Sequence[dict[str, Any]]) -> Iterator[MotionFeature]:
        """按顺序产出每个 clip 的完整特征（调试用，不落盘）。

        Args:
            entries: 索引记录。

        Yields:
            :class:`MotionFeature`。
        """
        for entry in entries:
            job = self._process_one(entry)
            if job.error or not job.windows:
                continue
            feat = np.concatenate([win[0] for win in job.windows], axis=0)
            mask = np.concatenate([win[1] for win in job.windows], axis=0)
            yield MotionFeature(feat=feat, mask=mask, fps=self.cfg.data.fps)
