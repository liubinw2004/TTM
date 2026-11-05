"""扫描器「窗口预算 + 细类 round-robin」配额策略的单元测试。

背景（主理人 M1' 决策 A）：旧版按 ``max_files`` **文件数**截断，但片段长度相差近
10 倍（``1.12`` 的长链任务单条 2850 帧 = 19 个窗口，``3.2`` 的短片段 300 帧 =
2 个窗口），等于用错了配平单位——实测 ``1.12`` 一个类吃掉 85% 的窗口。

这里用合成 ``ScanEntry`` 直接测 :meth:`ttm.data.scan.Scanner._select_balanced`，
不碰 Z 盘，跑得快且可断言精确数字。
"""

from __future__ import annotations

from pathlib import Path

import pytest
import yaml

from ttm.common.config import load_config
from ttm.data.features import count_windows, window_stride_for
from ttm.data.scan import ScanEntry, Scanner


def _entry(
    leaf: str,
    index: int,
    *,
    n_frames: int = 900,
    n_windows: int = 6,
    l2: str = "sub",
) -> ScanEntry:
    """构造一个合成索引条目。"""
    return ScanEntry(
        clip_id=f"{leaf}_{index}",
        path=f"/synthetic/{leaf}/{index}.bvh",
        take_id=f"synthetic/{leaf}/{index}",
        split="train",
        zh_label=leaf,
        category_l1="l1",
        category_l2=l2,
        category_l3=leaf,
        n_frames=n_frames,
        n_windows=n_windows,
        source=f"/synthetic/{leaf}/{index}.bvh",
    )


def _scanner(tmp_path: Path, **overrides) -> Scanner:
    """按给定覆盖项构造一个不访问磁盘的扫描器。"""
    data = {
        "bvh_roots": ["D:/TTM/.cache/does-not-exist"],
        "max_windows": 1000,
        "max_windows_per_category": 10000,
        "category_window_share_cap": 0.0,
        "min_clips_per_leaf": 2,
        "max_files": 10000,
    }
    data.update(overrides)
    path = tmp_path / "cfg.yaml"
    payload = {
        "project": {"root": "D:/TTM", "data_dir": "data/unit_test", "runs_dir": "runs/unit_test"},
        "data": data,
    }
    path.write_text(yaml.safe_dump(payload, allow_unicode=True), encoding="utf-8")
    return Scanner(load_config(path))


# ---------------------------------------------------------------------------
# 窗口步长
# ---------------------------------------------------------------------------
def test_window_stride_keeps_short_clips_dense() -> None:
    """短片段不降密度（步长 = 窗口长度 150）。"""
    assert window_stride_for(300, window=150, long_clip_stride=450) == 150
    assert window_stride_for(900, window=150, long_clip_stride=450) == 150


def test_window_stride_downsamples_long_clips() -> None:
    """超过 2×450 = 900 帧才降密度。"""
    assert window_stride_for(901, window=150, long_clip_stride=450) == 450
    assert window_stride_for(2850, window=150, long_clip_stride=450) == 450
    # 关闭降密度
    assert window_stride_for(2850, window=150, long_clip_stride=0) == 150


def test_count_windows_with_stride() -> None:
    """步长公式与非重叠公式在 ``stride == window`` 时完全一致。"""
    for frames in (0, 60, 100, 150, 299, 360, 400, 900, 2850):
        assert count_windows(frames, 150, 75) == count_windows(frames, 150, 75, stride=150)
    # 2850 帧：150 步长 19 窗，450 步长 7 窗
    assert count_windows(2850, 150, 75, stride=150) == 19
    assert count_windows(2850, 150, 75, stride=450) == 7


# ---------------------------------------------------------------------------
# 配额策略
# ---------------------------------------------------------------------------
def test_round_robin_feeds_tail_before_head(tmp_path: Path) -> None:
    """长尾细类先被喂饱：每个细类都拿到配额，头部不会独吞。"""
    head = [_entry("head", index, n_windows=19) for index in range(100)]
    tails = [_entry(f"tail{leaf}", index, n_windows=2) for leaf in range(20) for index in range(8)]
    scanner = _scanner(
        tmp_path, max_windows=400, category_window_share_cap=0.2, max_windows_per_category=10000
    )
    stats: dict = {}
    selected = scanner._select_balanced(head + tails, stats)

    leaves = {entry.category_l3 for entry in selected}
    assert len(leaves) == 21, f"应覆盖全部 21 个细类，实得 {len(leaves)}"
    head_windows = sum(e.n_windows for e in selected if e.category_l3 == "head")
    assert head_windows <= 0.2 * 400 + 1e-9, f"头部细类占 {head_windows} 窗，超预算 20%"
    assert stats["balanced"]["max_leaf_share_of_budget"] <= 0.2 + 1e-9


def test_window_budget_stops_selection(tmp_path: Path) -> None:
    """总窗口预算达到即停。"""
    entries = [_entry("a", index, n_windows=10) for index in range(50)]
    scanner = _scanner(tmp_path, max_windows=100, max_windows_per_category=1000)
    selected = scanner._select_balanced(entries, {})
    assert sum(e.n_windows for e in selected) == 100
    assert len(selected) == 10


def test_per_leaf_window_cap(tmp_path: Path) -> None:
    """单个细类的窗口硬上限生效。"""
    entries = [_entry("a", index, n_windows=10) for index in range(50)]
    scanner = _scanner(tmp_path, max_windows=10000, max_windows_per_category=50)
    selected = scanner._select_balanced(entries, {})
    assert sum(e.n_windows for e in selected) == 50
    assert len(selected) == 5


def test_share_cap_limits_dominant_leaf(tmp_path: Path) -> None:
    """占比上限按「总窗口预算」计算：最大细类不超过预算的 20%。

    注：这里刻意让素材总量（400 窗）小于预算（1000 窗），也就是预算填不满的
    情况。此时「占已选总量的比例」必然超过 20%（分母更小），所以约束口径必须
    是「占预算」——否则会出现「为了降占比而丢数据」的荒谬结果。
    """
    head = [_entry("head", index, n_windows=20) for index in range(100)]
    tails = [_entry(f"tail{leaf}", 0, n_windows=5) for leaf in range(40)]
    scanner = _scanner(
        tmp_path,
        max_windows=1000,
        max_windows_per_category=10000,
        category_window_share_cap=0.2,
    )
    stats: dict = {}
    selected = scanner._select_balanced(head + tails, stats)

    head_windows = sum(e.n_windows for e in selected if e.category_l3 == "head")
    assert head_windows <= 0.2 * 1000 + 1e-9, f"头部占 {head_windows} 窗，超预算 20%"
    assert stats["balanced"]["max_leaf_share_of_budget"] <= 0.20 + 1e-9
    assert stats["balanced"]["window_budget_hit"] is False


def test_selection_preserves_candidate_order(tmp_path: Path) -> None:
    """输出按候选原始顺序，保证 index.jsonl 稳定可 diff。"""
    entries = [_entry(f"leaf{index % 5}", index, n_windows=5) for index in range(30)]
    scanner = _scanner(tmp_path, max_windows=10000)
    selected = scanner._select_balanced(entries, {})
    positions = [entries.index(entry) for entry in selected]
    assert positions == sorted(positions)


def test_stats_report_long_tail_below_min_clips(tmp_path: Path) -> None:
    """统计里必须点名「clip 数不足 min_clips_per_leaf」的细类，而不是静默丢弃。"""
    entries = [_entry("rich", index, n_windows=1) for index in range(10)]
    entries += [_entry("poor", 0, n_windows=1)]
    scanner = _scanner(
        Path(".") / ".cache", max_windows=10000, category_window_share_cap=0.0, min_clips_per_leaf=5
    )
    stats: dict = {}
    scanner._select_balanced(entries, stats)
    below = {item["leaf"] for item in stats["balanced"]["leaves_below_min_clips"]}
    assert below == {"poor"}


def test_no_candidates_is_safe(tmp_path: Path) -> None:
    """零候选不抛异常。"""
    scanner = _scanner(tmp_path)
    assert scanner._select_balanced([], {}) == []
