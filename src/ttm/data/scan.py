"""数据集扫描器：目录枚举 → Take 分组去重 → 8:1:1 切分 → ``index.jsonl``。

职责（``TTM-ARCH.md`` §4.1 时序图 A、§5.6）：

1. **目录枚举**：按 ``configs/*.yaml`` 的 ``data.bvh_roots`` 与
   ``data.include_categories`` 白名单**按序**遍历（白名单作为子树根，天然实现优先级，
   且不必扫描无关子树——Z 盘很慢）；
2. **Take 分组去重**：同一 ``Take_*`` 目录下的多个演员是同一动作的不同体型，
   每个 Take 只保留 ``data.actors_per_take`` 个（默认 1），避免过拟合到某一体型；
3. **头部校验**：只读文件头（:meth:`ttm.data.bvh.BvhParser.peek`）数关节数，
   ``n_joints != 23`` 的 root-only 轨迹文件直接淘汰并计入 ``failed.jsonl``；
4. **切分**：按 ``take_id`` 的稳定哈希切 train/val/test，**保证三集 take_id 无交集**；
5. **产物**：``data/tier_{tier}/index.jsonl`` + ``failed.jsonl``。

空 Take 目录会被自然跳过（没有 ``.bvh`` 文件就没有候选），不产生噪音。
"""

from __future__ import annotations

import dataclasses
import hashlib
import os
import threading
import time
from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Iterable, Iterator, Sequence

import orjson

from ttm.common import constants as C
from ttm.common import paths as P
from ttm.common.config import TTMConfig
from ttm.common.logutil import get_logger
from ttm.data.bvh import BvhHeader, BvhParser, InvalidBvh
from ttm.data import features
from ttm.data.features import window_stride_for
from ttm.data.skeleton import SkeletonDef

__all__ = [
    "ScanEntry",
    "ScanResult",
    "Scanner",
    "count_windows",
    "split_for_take",
]

#: 合法的 split 名。
SPLITS: tuple[str, ...] = ("train", "val", "test")

#: 目录名形如 ``20260326`` / ``20251214_SD101`` 时视为「日期目录」，不计入类别层级。
_DATE_PREFIX_LEN = 8

#: 默认并发数（Z 盘网络 RTT 摊薄的实测较优值）。
DEFAULT_WORKERS = 8


def count_windows(
    n_frames: int,
    window: int = C.WINDOW,
    min_tail: int = C.MIN_TAIL_FRAMES,
    stride: int | None = None,
) -> int:
    """按 §5.5 规则计算一个 clip 能切出多少训练窗口（转发到唯一实现处）。

    Args:
        n_frames: 30fps 下的帧数。
        window: 窗口帧数（默认 150）。
        min_tail: 尾窗保留阈值（默认 75）。
        stride: 窗口步长；``None`` 等价于 ``window``（非重叠）。

    Returns:
        窗口数（可能为 0，表示该 clip 太短无法入训）。
    """
    return features.count_windows(n_frames, window=window, min_tail=min_tail, stride=stride)


def split_for_take(take_id: str, val_ratio: float, test_ratio: float) -> str:
    """用 ``take_id`` 的稳定哈希决定所属 split。

    同一个 take 永远落在同一集，且 train/val/test 的 take 集合天然无交集。

    Args:
        take_id: Take 唯一标识。
        val_ratio: 验证集比例。
        test_ratio: 测试集比例。

    Returns:
        ``"train"`` / ``"val"`` / ``"test"``。
    """
    digest = hashlib.sha1(take_id.encode("utf-8")).hexdigest()
    bucket = int(digest[:8], 16) / 0xFFFFFFFF  # [0,1]
    if bucket < test_ratio:
        return "test"
    if bucket < test_ratio + val_ratio:
        return "val"
    return "train"


def _is_date_dir(name: str) -> bool:
    """判断目录名是否为日期目录（``20260326`` / ``20251214_SD101``）。"""
    head = name[:_DATE_PREFIX_LEN]
    return len(head) == _DATE_PREFIX_LEN and head.isdigit()


def _is_take_dir(name: str) -> bool:
    """判断目录名是否为 Take 目录（``Take_207``）。"""
    return name.lower().startswith("take_")


@dataclass(frozen=True)
class ScanEntry:
    """``index.jsonl`` 的一行：一个可入训的 clip。"""

    clip_id: str
    path: str
    take_id: str
    split: str
    zh_label: str
    category_l1: str
    category_l2: str
    category_l3: str
    n_frames: int
    n_windows: int
    source: str
    src_fps: float = 0.0
    duration_s: float = 0.0
    actor: str = ""
    root_name: str = ""

    def to_dict(self) -> dict[str, Any]:
        """转成 JSON 可序列化 dict（字段顺序稳定，便于 diff）。"""
        return {
            "clip_id": self.clip_id,
            "path": self.path,
            "take_id": self.take_id,
            "split": self.split,
            "zh_label": self.zh_label,
            "category_l1": self.category_l1,
            "category_l2": self.category_l2,
            "category_l3": self.category_l3,
            "n_frames": self.n_frames,
            "n_windows": self.n_windows,
            "source": self.source,
            "src_fps": self.src_fps,
            "duration_s": self.duration_s,
            "actor": self.actor,
            "root_name": self.root_name,
        }


@dataclass
class ScanResult:
    """扫描结果与统计。"""

    entries: list[ScanEntry] = field(default_factory=list)
    failed: list[dict[str, Any]] = field(default_factory=list)
    stats: dict[str, Any] = field(default_factory=dict)
    elapsed_s: float = 0.0

    def split_counts(self) -> dict[str, int]:
        """各 split 的 clip 数。"""
        counts = {name: 0 for name in SPLITS}
        for entry in self.entries:
            counts[entry.split] = counts.get(entry.split, 0) + 1
        return counts

    def take_ids(self, split: str) -> set[str]:
        """某 split 的 take_id 集合。"""
        return {entry.take_id for entry in self.entries if entry.split == split}

    def total_windows(self) -> int:
        """总窗口数。"""
        return sum(entry.n_windows for entry in self.entries)


class Scanner:
    """扫描器：把只读盘上的 BVH 目录树变成带标签与切分的索引。"""

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
            limit: 最多采集多少个 clip（调试用，覆盖 ``data.max_files``）。
            workers: 并发 peek 文件的线程数。
            logger: 可注入的 logger（默认按脚本名创建）。
        """
        self.cfg = cfg
        self.limit = int(limit) if limit else int(cfg.data.max_files)
        self.workers = max(1, int(workers))
        self.log = logger or get_logger("ttm.scan", script="01_scan")
        self._lock = threading.Lock()

    # -- 主流程 -------------------------------------------------------------
    def run(self) -> ScanResult:
        """执行扫描。

        Returns:
            :class:`ScanResult`（含 entries / failed / stats）。
        """
        started = time.perf_counter()
        result = ScanResult()
        stats: dict[str, Any] = {
            "tier": self.cfg.data.tier,
            "roots": list(self.cfg.data.bvh_roots),
            "include_categories": list(self.cfg.data.include_categories or []),
            "files_seen": 0,
            "takes_seen": 0,
            "takes_empty": 0,
            "takes_deduped": 0,
            "takes_skipped_by_cap": 0,
            "clips_selected": 0,
            "clips_rejected_header": 0,
            "clips_rejected_too_short": 0,
            "clips_rejected_duplicate_id": 0,
        }
        seen_ids: set[str] = set()
        candidates: list[ScanEntry] = []
        # 候选上限只是内存保险丝（真正截断由窗口预算 + LFQ 完成）
        collect_cap = max(self.limit * 20, 200_000)

        for root, take_id, files, category_parts in self._iter_take_groups():
            stats["takes_seen"] += 1
            if not files:
                stats["takes_empty"] += 1
                continue
            stats["files_seen"] += len(files)

            selected = files[: max(1, int(self.cfg.data.actors_per_take))]
            if len(files) > len(selected):
                stats["takes_deduped"] += 1

            headers = self._peek_many(selected)
            for path, header, error in headers:
                if error is not None:
                    self._record_failure(
                        result, path, stage="scan", error=f"{type(error).__name__}: {error}"
                    )
                    stats["clips_rejected_header"] += 1
                    continue
                assert header is not None
                reason = header.rejection_reason()
                if reason:
                    self._record_failure(result, path, stage="scan", error=reason)
                    stats["clips_rejected_header"] += 1
                    continue

                entry, too_short = self._build_entry(path, header, take_id, category_parts)
                if too_short:
                    self._record_failure(
                        result, path, stage="scan", error=f"too_short: n_frames@30fps={entry.n_frames}"
                    )
                    stats["clips_rejected_too_short"] += 1
                    continue
                if entry.clip_id in seen_ids:
                    unique = self._dedupe_clip_id(entry.clip_id, seen_ids)
                    entry = dataclasses.replace(entry, clip_id=unique)
                    stats["clips_rejected_duplicate_id"] += 1
                seen_ids.add(entry.clip_id)
                candidates.append(entry)
                if len(candidates) >= collect_cap:  # pragma: no cover - 极端保险
                    self.log.warning("候选数触达内存保险丝 %d，提前停止枚举", collect_cap)
                    break
            if len(candidates) >= collect_cap:  # pragma: no cover
                stats["takes_skipped_by_cap"] += 1
                break

        stats["clips_candidates"] = len(candidates)
        result.entries = self._select_balanced(candidates, stats)

        self._ensure_all_splits(result, stats)
        stats["clips_selected"] = len(result.entries)
        stats["windows_total"] = result.total_windows()
        stats["split_counts"] = result.split_counts()
        stats["failed"] = len(result.failed)
        stats["limit"] = self.limit
        result.stats = stats
        result.elapsed_s = time.perf_counter() - started
        self.log.info(
            "扫描完成: 候选 %d → 入训 %d clip / %d 窗口, failed=%d, 用时 %.1fs",
            len(candidates),
            stats["clips_selected"],
            stats["windows_total"],
            stats["failed"],
            result.elapsed_s,
        )
        return result

    # -- 均衡配额 -----------------------------------------------------------
    @staticmethod
    def _leaf_key(entry: ScanEntry) -> str:
        """细类的分组键（l1/l2/l3 三级全带上，避免同名 l3 跨子类被合并）。"""
        return f"{entry.category_l1}|{entry.category_l2}|{entry.category_l3}"

    def _select_balanced(
        self, candidates: Sequence[ScanEntry], stats: dict[str, Any]
    ) -> list[ScanEntry]:
        """在窗口预算内按「细类 round-robin」挑选 clip。

        为什么不是「按文件数截断」或「按细类顺序取满即停」：文件长度相差近 10 倍
        （``1.12`` 的长链任务单条 2850 帧 = 19 个窗口，``3.2`` 的短片段 300 帧 =
        2 个窗口）。按文件数配平等于按错单位配平——实测旧版 ``1.12`` 一个类吃掉
        19193/22682 = 85% 的窗口。

        所以调度单位是**窗口**：每轮挑「已累计窗口数最少」的细类补 1 个 clip
        （least-filled queue），直到总窗口预算耗尽或所有细类取空。这样长尾细类
        先被喂饱，长套路细类自然先触顶停手。

        Args:
            candidates: 全部通过头部校验的候选条。
            stats: 统计字典（原地写入均衡性指标）。

        Returns:
            被选中的条目（按原始候选顺序，保证 ``index.jsonl`` 稳定可 diff）。
        """
        cfg = self.cfg.data
        window_budget = max(1, int(cfg.max_windows))
        clip_budget = max(1, int(self.limit))
        leaf_cap = max(1, int(cfg.max_windows_per_category))
        share_cap = float(cfg.category_window_share_cap)
        if share_cap > 0.0:
            # 单类窗口占比软上限（默认 20%），与硬上限取小
            leaf_cap = min(leaf_cap, max(1, int(round(window_budget * share_cap))))

        buckets: dict[str, list[int]] = {}
        for index, entry in enumerate(candidates):
            buckets.setdefault(self._leaf_key(entry), []).append(index)
        for key in buckets:
            buckets[key].sort(key=lambda idx: candidates[idx].path)

        order = sorted(buckets)
        cursor = {key: 0 for key in order}
        used_windows = {key: 0 for key in order}
        used_clips = {key: 0 for key in order}
        chosen: list[int] = []
        total_windows = 0

        while total_windows < window_budget and len(chosen) < clip_budget:
            best_key: str | None = None
            best_windows: int | None = None
            for key in order:
                if cursor[key] >= len(buckets[key]):
                    continue
                if used_windows[key] >= leaf_cap:
                    continue
                nxt = candidates[buckets[key][cursor[key]]]
                if used_windows[key] + nxt.n_windows > leaf_cap:
                    # 这一个 clip 自己就超上限 → 该细类停止供料，不再尝试后续
                    cursor[key] = len(buckets[key])
                    continue
                if best_windows is None or used_windows[key] < best_windows:
                    best_windows = used_windows[key]
                    best_key = key
            if best_key is None:
                break
            index = buckets[best_key][cursor[best_key]]
            cursor[best_key] += 1
            entry = candidates[index]
            chosen.append(index)
            used_windows[best_key] += entry.n_windows
            used_clips[best_key] += 1
            total_windows += entry.n_windows

        chosen.sort()
        selected = [candidates[index] for index in chosen]

        # --- 均衡性统计（主理人验收口径）---
        by_leaf: dict[str, dict[str, int]] = {}
        for entry in selected:
            key = self._leaf_key(entry)
            slot = by_leaf.setdefault(key, {"clips": 0, "windows": 0, "label": entry.zh_label})
            slot["clips"] += 1
            slot["windows"] += entry.n_windows
        total = sum(slot["windows"] for slot in by_leaf.values()) or 1
        ranked = sorted(by_leaf.items(), key=lambda item: -item[1]["windows"])
        stats["balanced"] = {
            "window_budget": window_budget,
            "clip_budget": clip_budget,
            "leaf_window_cap": leaf_cap,
            "share_cap": share_cap,
            "leaves_available": len(buckets),
            "leaves_selected": len(by_leaf),
            "windows_selected": total,
            "window_budget_hit": total_windows >= window_budget,
            "clip_budget_hit": len(chosen) >= clip_budget,
            "max_leaf_share": (ranked[0][1]["windows"] / total) if ranked else 0.0,
            # 这个才是配额约束的口径：单类窗口不得超过「总窗口预算」的 20%。
            # 不是「实际入选总量」的 20%——预算没填满时（素材总量不足）按后者
            # 计算会自相矛盾：把已选中的 clip 丢掉只会让总量更小、占比更高。
            "max_leaf_share_of_budget": (
                ranked[0][1]["windows"] / window_budget if ranked else 0.0
            ),
            "top_leaves": [
                {
                    "leaf": key.split("|")[-1],
                    "label": slot["label"],
                    "clips": slot["clips"],
                    "windows": slot["windows"],
                    "share": slot["windows"] / total,
                }
                for key, slot in ranked[:15]
            ],
            "leaves_below_min_clips": [
                {"leaf": key.split("|")[-1], "clips": slot["clips"]}
                for key, slot in sorted(by_leaf.items())
                if slot["clips"] < int(cfg.min_clips_per_leaf)
            ],
            "leaves_dropped": len(buckets) - len(by_leaf),
        }
        return selected

    # -- 枚举 ---------------------------------------------------------------
    def _iter_take_groups(self) -> Iterator[tuple[str, str, list[str], list[str]]]:
        """按「白名单顺序 → 目录顺序」逐 Take 组产出候选文件。

        每个被访问的目录若直接包含 ``.bvh`` 文件，就视为一个 Take 组
        （正常数据是 ``.../Take_207/Take_207_actor_0.bvh``，但也能兼容文件直接
        放在日期目录的情况）。

        Yields:
            ``(root, take_id, files, category_parts)``。
        """
        roots = list(self.cfg.data.bvh_roots)
        categories = list(self.cfg.data.include_categories or [])
        for root in roots:
            root_path = Path(root)
            if not root_path.is_dir():
                self.log.warning("BVH 根目录不可访问，跳过: %s", root)
                continue
            bases: list[tuple[Path, str]] = []
            if categories:
                for cat in categories:
                    candidate = root_path / cat
                    if candidate.is_dir():
                        bases.append((candidate, cat))
                    else:
                        self.log.debug("类别前缀在 %s 下不存在，跳过: %s", root_path.name, cat)
            if not bases:
                # 该根目录一个白名单前缀都没命中（如武术根 ``Z:/1000hours/4.wushu``
                # 下面是 ``4.3`` 而不是 ``1.7.Daily_Life_Behavior``）。
                # 这时扫描整个根目录——否则新增数据源会被配置静默跳过，
                # 而「武术一条都没进训练集」正是这次返工的原因。
                #
                # 前缀必须补成**根目录名**：否则武术的 l1 会是 ``4.3``，
                # 与 BVH_Body 的 ``4.3.Equipment_Exercises``（器械健身）在报告里
                # 撞名，统计时会把两者混在一起。
                if categories:
                    self.log.info(
                        "根目录 %s 未命中任何 include_categories 前缀，改为扫描全量",
                        root_path.name,
                    )
                    bases = [(root_path, root_path.name)]
                else:
                    bases = [(root_path, "")]
            for base, prefix in bases:
                for take_dir, files in self._walk_take_dirs(base):
                    rel = take_dir.relative_to(root_path).as_posix()
                    take_id = f"{root_path.name}/{rel}" if rel else root_path.name
                    yield root, take_id, files, self._category_parts(rel, prefix)

    @staticmethod
    def _walk_take_dirs(base: Path) -> Iterator[tuple[Path, list[str]]]:
        """遍历一个子树，产出「Take 目录 → 排序后的文件列表」。

        包含 ``.bvh`` 的目录产出一个非空列表；名字形如 ``Take_*`` 但**没有**
        任何 ``.bvh`` 的目录产出一个空列表（用于统计空 Take 目录，真实数据里存在）。

        Args:
            base: 子树根。

        Yields:
            ``(directory, absolute_file_paths)``，文件已按名字排序（保证确定性）。
        """
        for current, dirnames, filenames in os.walk(base):
            dirnames.sort()
            bvh_files = sorted(
                str(Path(current) / name) for name in filenames if name.lower().endswith(".bvh")
            )
            if bvh_files:
                yield Path(current), bvh_files
            elif _is_take_dir(Path(current).name):
                yield Path(current), []

    def _category_parts(self, rel_dir: str, prefix: str) -> list[str]:
        """从 Take 目录相对路径提取类别层级（去掉日期与 Take 目录名）。

        Args:
            rel_dir: Take 目录相对 BVH 根的 POSIX 路径。
            prefix: 命中的类别白名单前缀（用于补全被裁剪的层级）。

        Returns:
            类别层级列表，如 ``["1.7.Daily_Life_Behavior", "1.7.3.Walking_...", "1.7.3.1.Basic_Gait"]``。
        """
        parts: list[str] = []
        if prefix:
            parts.extend(prefix.split("/"))
        for part in rel_dir.split("/"):
            if not part or _is_date_dir(part) or _is_take_dir(part):
                continue
            if parts and parts[-1] == part:
                continue
            parts.append(part)
        return parts

    # -- 头部校验 -----------------------------------------------------------
    def _peek_many(
        self, paths: Sequence[str]
    ) -> list[tuple[str, BvhHeader | None, BaseException | None]]:
        """并发读取多个文件的头部。"""
        if len(paths) == 0:
            return []
        if len(paths) == 1:
            path = paths[0]
            try:
                return [(path, BvhParser.peek(path), None)]
            except BaseException as exc:  # noqa: BLE001 - 坏文件一律记 failed
                return [(path, None, exc)]
        with ThreadPoolExecutor(max_workers=min(self.workers, len(paths))) as pool:
            futures = [(path, pool.submit(BvhParser.peek, path)) for path in paths]
            output: list[tuple[str, BvhHeader | None, BaseException | None]] = []
            for path, future in futures:
                try:
                    output.append((path, future.result(), None))
                except BaseException as exc:  # noqa: BLE001
                    output.append((path, None, exc))
            return output

    # -- 组条目 -------------------------------------------------------------
    def _build_entry(
        self,
        path: str,
        header: BvhHeader,
        take_id: str,
        category_parts: Sequence[str],
    ) -> tuple[ScanEntry, bool]:
        """由头部信息组装一条索引记录。

        Args:
            path: 文件绝对路径。
            header: 头部信息。
            take_id: Take 唯一标识。
            category_parts: 类别层级。

        Returns:
            ``(entry, too_short)``；``too_short`` 为 ``True`` 时该 clip 不入训。
        """
        src_fps = header.fps if header.fps > 0 else float(C.SOURCE_FPS)
        n_frames = SkeletonDef.resampled_length(header.n_frames, src_fps, self.cfg.data.fps)
        stride = window_stride_for(
            n_frames,
            window=self.cfg.data.window,
            long_clip_stride=self.cfg.data.long_clip_stride_frames,
        )
        n_windows = count_windows(
            n_frames, self.cfg.data.window, self.cfg.data.min_tail_frames, stride=stride
        )

        stem = Path(path).stem
        parent_parts = Path(path).parent.parts
        date = ""
        for part in reversed(parent_parts):
            if _is_date_dir(part):
                date = part
                break

        cat_l1 = category_parts[0] if len(category_parts) > 0 else ""
        cat_l2 = category_parts[1] if len(category_parts) > 1 else ""
        cat_l3 = category_parts[2] if len(category_parts) > 2 else (
            category_parts[-1] if category_parts else stem
        )

        clip_id = "/".join(part for part in (cat_l3, date, stem) if part)
        # npz 的 clip_id 字段是 <U80（ARCH §5.5）：窗口级 id 还要再加 "_winNN" 后缀。
        # 这里提前拦住，避免落盘时被静默截断导致 labels.jsonl 与分片对不上。
        if len(clip_id) + len("_win000") > C.CLIP_ID_MAXLEN:
            raise ValueError(
                f"clip_id 过长（{len(clip_id)} + 窗口后缀 > {C.CLIP_ID_MAXLEN}）: {clip_id}"
            )
        entry = ScanEntry(
            clip_id=clip_id,
            path=P.rel_to_root(path) if _is_under_project(path) else path,
            take_id=take_id,
            split=split_for_take(take_id, self.cfg.data.val_ratio, self.cfg.data.test_ratio),
            zh_label=_zh_label_for(cat_l3, cat_l2, cat_l1, stem),
            category_l1=cat_l1,
            category_l2=cat_l2,
            category_l3=cat_l3,
            n_frames=n_frames,
            n_windows=n_windows,
            source=str(path),
            src_fps=float(src_fps),
            duration_s=float(header.n_frames) / src_fps if src_fps > 0 else 0.0,
            actor=stem,
            root_name=header.root_name,
        )
        return entry, n_windows == 0

    def _record_failure(self, result: ScanResult, path: str, *, stage: str, error: str) -> None:
        """记录一条失败（写 ``failed.jsonl`` 的内容）。"""
        with self._lock:
            result.failed.append({"path": str(path), "stage": stage, "error": str(error)})
        self.log.debug("坏文件已跳过 [%s] %s: %s", stage, path, error)

    @staticmethod
    def _dedupe_clip_id(clip_id: str, seen: set[str]) -> str:
        """为重复的 clip_id 生成唯一后缀。"""
        counter = 1
        while f"{clip_id}_dup{counter}" in seen:
            counter += 1
        return f"{clip_id}_dup{counter}"

    def _ensure_all_splits(self, result: ScanResult, stats: dict[str, Any]) -> None:
        """小样本下保证 val/test 非空（整 Take 迁移，不破坏 take 唯一性）。

        Args:
            result: 扫描结果（原地修改 split）。
            stats: 统计字典（记录是否触发补偿）。
        """
        take_count = len({entry.take_id for entry in result.entries})
        if take_count < 10:
            return
        for missing in ("test", "val"):
            if any(entry.split == missing for entry in result.entries):
                continue
            candidates = sorted(
                {entry.take_id for entry in result.entries if entry.split == "train"}
            )
            if not candidates:
                continue
            victim = candidates[-1]
            moved = 0
            for idx, entry in enumerate(result.entries):
                if entry.take_id == victim:
                    result.entries[idx] = dataclasses.replace(entry, split=missing)
                    moved += 1
            stats[f"rebalanced_to_{missing}"] = moved

    # -- 落盘 ---------------------------------------------------------------
    def write(self, result: ScanResult) -> tuple[Path, Path]:
        """写出 ``index.jsonl`` 与 ``failed.jsonl``。

        Args:
            result: 扫描结果。

        Returns:
            ``(index_path, failed_path)``。
        """
        index_path = P.resolve_out(self.cfg.index_path(), mkparents=True)
        failed_path = P.resolve_out(self.cfg.failed_path(), mkparents=True)

        with index_path.open("wb") as handle:
            for entry in result.entries:
                handle.write(orjson.dumps(entry.to_dict()))
                handle.write(b"\n")
        with failed_path.open("wb") as handle:
            for item in result.failed:
                handle.write(orjson.dumps(item))
                handle.write(b"\n")

        self.log.info("已写出 %s（%d 行）", index_path, len(result.entries))
        self.log.info("已写出 %s（%d 行）", failed_path, len(result.failed))
        return index_path, failed_path


# ---------------------------------------------------------------------------
# 辅助
# ---------------------------------------------------------------------------
def _is_under_project(path: str) -> bool:
    """判断路径是否在项目根内（决定写相对路径还是绝对路径）。"""
    try:
        P.assert_write_allowed(path)
        return True
    except P.WriteNotAllowedError:
        return False


def _zh_label_for(cat_l3: str, cat_l2: str, cat_l1: str, stem: str) -> str:
    """取中文标签：优先用 :mod:`ttm.data.labels_zh`，缺失时机械转写。

    扫描阶段（T03）早于标签模块（T04），因此这里做**惰性导入**：
    标签模块存在时用它，否则回退到「去掉序号前缀的英文段名」。

    注意必须把 ``cat_l2`` 一起传下去：同名 l3（``Pass_Ball`` 同存于篮球与排球、
    ``Block`` 同存于拳击与击剑）只能靠 l2 消歧，只传 l3 会让不同运动塌成同一标签。

    Args:
        cat_l3: 细类目录名。
        cat_l2: 子类目录名。
        cat_l1: 大类目录名。
        stem: 文件名主干（兜底）。

    Returns:
        中文标签字符串。
    """
    target = cat_l3 or cat_l2 or cat_l1 or stem
    try:  # 惰性导入，避免 T03 阶段硬依赖 T04 产物
        from ttm.data import labels_zh  # noqa: PLC0415

        label = labels_zh.zh_label_for_l3(target, cat_l2)
        if not label:
            label = labels_zh.zh_label_for_directory(target)
    except Exception:  # noqa: BLE001 - 标签模块缺失或未映射时回退
        label = ""
    if label:
        return label
    return _fallback_label(target)


def _fallback_label(directory: str) -> str:
    """把英文目录名机械转写为可读标签（去序号、下划线转空格）。"""
    text = directory.strip()
    if not text:
        return "未知动作"
    # 去掉形如 "1.7.3.3." 的编号前缀
    while text and (text[0].isdigit() or text[0] == "."):
        text = text[1:]
    text = text.replace("_", " ").strip()
    return text or directory


def read_index(path: str | os.PathLike[str]) -> list[dict[str, Any]]:
    """读取 ``index.jsonl``（调试/测试用）。

    Args:
        path: 索引文件路径。

    Returns:
        记录列表。
    """
    records: list[dict[str, Any]] = []
    with open(path, "rb") as handle:
        for line in handle:
            text = line.strip()
            if text:
                records.append(orjson.loads(text))
    return records


def iter_entry_paths(entries: Iterable[ScanEntry]) -> Iterator[str]:
    """遍历条目中的源文件路径。"""
    for entry in entries:
        yield entry.source
