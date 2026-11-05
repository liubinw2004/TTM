"""M2' 前置工具（根高度审计 / 数据门禁 / 重建可视化）的单元测试。

这三个脚本都属于「门禁」性质——它们要能在**训练之前**发现数据问题，所以判定逻辑
本身必须是可验证的。这里刻意不碰 Z 盘、不依赖真实分片：用合成 npz 与合成数组
直接测纯函数与断言逻辑，跑得快且能断言精确数字。

覆盖的关键判定：

* ``02c`` 的分带映射与 ``SUSPECT`` 判定（半坏带占比 > 5%）；
* ``02d`` 的 a)–e) 五条断言各自**能正确地失败**——一条永远为 PASS 的断言等于没有；
* ``03b`` 的动作阶段切分与「至少含一个武术样本」的分层抽样。
"""

from __future__ import annotations

import importlib.util
import sys
from pathlib import Path
from types import ModuleType
from typing import Any

import numpy as np
import pytest

from ttm import LAYOUT_VERSION
from ttm.common import constants as C
from ttm.data.features import NormStats

_ROOT = Path(__file__).resolve().parents[1]


def _load_script(name: str) -> ModuleType:
    """按文件路径加载 ``scripts/`` 下的脚本模块（脚本名不是合法标识符）。

    Args:
        name: 脚本文件名（如 ``"02c_root_height_audit.py"``）。

    Returns:
        加载后的模块对象。
    """
    path = _ROOT / "scripts" / name
    module_name = f"_script_{path.stem.replace('-', '_')}"
    if module_name in sys.modules:
        return sys.modules[module_name]
    spec = importlib.util.spec_from_file_location(module_name, path)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    sys.modules[module_name] = module
    spec.loader.exec_module(module)
    return module


@pytest.fixture(scope="module")
def audit_module() -> ModuleType:
    """``02c_root_height_audit.py`` 模块。"""
    return _load_script("02c_root_height_audit.py")


@pytest.fixture(scope="module")
def gate_module() -> ModuleType:
    """``02d_verify_data.py`` 模块。"""
    return _load_script("02d_verify_data.py")


@pytest.fixture(scope="module")
def viz_module() -> ModuleType:
    """``03b_vae_recon_viz.py`` 模块。"""
    return _load_script("03b_vae_recon_viz.py")


# ---------------------------------------------------------------------------
# 02c：根高度审计
# ---------------------------------------------------------------------------
def test_band_index_covers_all_five_bands(audit_module: ModuleType) -> None:
    """五个分带的边界必须落在预期下标上（分带是左开右闭）。"""
    assert audit_module._band_index(-25.0) == 0
    assert audit_module._band_index(-20.0) == 0  # (-inf, -20]：-20 属「明显穿地」
    assert audit_module._band_index(-15.0) == 1  # (-20, -10]
    assert audit_module._band_index(-10.0) == 1  # 右闭：-10 属第一档
    assert audit_module._band_index(-9.9) == 2  # (-10, -5]
    assert audit_module._band_index(-7.0) == 2
    assert audit_module._band_index(-5.0) == 2
    assert audit_module._band_index(-0.1) == 3
    assert audit_module._band_index(0.0) == 3  # 右闭：0 属「-5~0cm 轻微下陷」
    assert audit_module._band_index(0.1) == 4
    assert audit_module._band_index(95.0) == 4


def test_band_index_handles_nan(audit_module: ModuleType) -> None:
    """NaN 归到最后一档，避免统计里出现空洞。"""
    assert audit_module._band_index(float("nan")) == len(audit_module.HEIGHT_BANDS) - 1


def test_semi_bad_bands_cover_minus_twenty_to_minus_five(audit_module: ModuleType) -> None:
    """``SEMI_BAD_BAND_INDEXES`` 必须覆盖 (-20cm, -5cm] 这一段（两档合并）。"""
    bands = audit_module.HEIGHT_BANDS
    lows = [bands[index][0] for index in audit_module.SEMI_BAD_BAND_INDEXES]
    highs = [bands[index][1] for index in audit_module.SEMI_BAD_BAND_INDEXES]
    assert min(lows) == -20.0
    assert max(highs) == -5.0
    assert all("半坏带" in bands[index][2] or "判废" in bands[index][2]
               for index in audit_module.SEMI_BAD_BAND_INDEXES)


def _clip(heights: list[float], category: str = "cat", label: str = "lbl") -> dict[str, Any]:
    """构造 ``collect_clip_heights`` 里单个 clip 的记录。"""
    return {
        "heights": np.asarray(heights, dtype=np.float64),
        "windows": 1,
        "category_l1": category,
        "category_l3": "",
        "zh_label": label,
        "is_wushu": False,
        "split": "train",
    }


def test_summarize_by_category_flags_offset_cluster(audit_module: ModuleType) -> None:
    """半坏带 clip 占比 > 5% 时必须判 SUSPECT，且分带计数与占比自洽。"""
    clips = {}
    # 19 个正常 clip + 1 个落在半坏带（-12cm）→ 5.0%，不超阈值，仍应 CLEAN
    for index in range(19):
        clips[f"ok{index}"] = _clip([90.0, 91.0, 92.0])
    clips["semi"] = _clip([-12.0, -12.5, -11.0])
    summary = audit_module.summarize_by_category(clips)["cat"]
    assert summary["n_clips"] == 20
    semi_total = sum(summary["bands"][i] for i in audit_module.SEMI_BAD_BAND_INDEXES)
    assert semi_total == 1
    assert summary["semi_bad_ratio"] == pytest.approx(0.05)
    assert summary["verdict"] == "CLEAN"  # 边界：等于 5% 不算超过

    # 再加一个落在 (-20,-10] 的 clip → 2/21 ≈ 9.5% > 5% → SUSPECT
    clips["semi2"] = _clip([-14.0, -15.0])
    summary = audit_module.summarize_by_category(clips)["cat"]
    semi_total = sum(summary["bands"][i] for i in audit_module.SEMI_BAD_BAND_INDEXES)
    assert semi_total == 2
    assert summary["verdict"] == "SUSPECT"


def test_summarize_by_category_uses_median_not_mean(audit_module: ModuleType) -> None:
    """单个腾空帧不能把 clip 中位数拉飞（用中位数而不是均值）。"""
    clips = {"a": _clip([90.0, 90.0, 90.0, 90.0, 400.0])}
    summary = audit_module.summarize_by_category(clips)["cat"]
    assert summary["median"] == pytest.approx(90.0)
    assert summary["max"] == pytest.approx(90.0)  # max 是「clip 中位数」的最大值


def test_summarize_by_label_counts_semi_bad(audit_module: ModuleType) -> None:
    """按标签汇总时半坏带条数必须精确。"""
    clips = {
        "a": _clip([-7.0], label="扫地"),
        "b": _clip([-8.0], label="扫地"),
        "c": _clip([90.0], label="扫地"),
        "d": _clip([-30.0], label="拖地"),
    }
    summary = audit_module.summarize_by_label(clips)
    assert summary["扫地"]["n_clips"] == 3
    assert summary["扫地"]["semi_bad"] == 2
    assert summary["扫地"]["semi_bad_ratio"] == pytest.approx(2 / 3)
    assert summary["拖地"]["semi_bad"] == 0  # -30cm 属「明显穿地」，不在半坏带


# ---------------------------------------------------------------------------
# 02d：数据门禁
# ---------------------------------------------------------------------------
def _alternating_block(n_windows: int, window: int) -> np.ndarray:
    """构造 ``[n_windows, window]`` 的 ``+1 / -1`` 交替块（精确 mean=0、std=1）。"""
    flat = np.where(np.arange(n_windows * window) % 2 == 0, 1.0, -1.0)
    return flat.reshape(n_windows, window).astype(np.float32)


def _write_shard(
    path: Path,
    *,
    n_windows: int,
    empty_indices: tuple[int, ...] = (),
    prefix: str = "clip",
) -> list[str]:
    """写一个符合契约的合成分片。

    Args:
        path: 目标 npz 路径。
        n_windows: 窗口数。
        empty_indices: 需要把 mask 置 0 的窗口下标（用于测「零 mask」断言）。
        prefix: ``clip_id`` 前缀。

    Returns:
        该分片的 ``clip_id`` 列表。
    """
    feat = np.zeros((n_windows, C.WINDOW, C.FEATURE_DIM), dtype=np.float32)
    # 断言 d) 要求「归一化后 mean≈0 / std≈1」，所以合成分片必须写**已标准化**的特征。
    # 用 ±1 交替而不是随机数：随机小样本的均值/标准差本身就有几个百分点噪声，
    # 会让门禁的容差判定变成「掷骰子」，而 ±1 交替是**精确** mean=0 / std=1。
    standardized = _alternating_block(n_windows, C.WINDOW)
    for dim in C.NORM_DIMS:
        feat[:, :, dim] = standardized
    mask = np.ones((n_windows, C.WINDOW), dtype=np.uint8)
    for index in empty_indices:
        mask[index] = 0
    clip_ids = np.array([f"{prefix}_{i:03d}" for i in range(n_windows)], dtype=f"<U{80}")
    np.savez(
        path,
        feat=feat,
        mask=mask,
        clip_id=clip_ids,
        win_idx=np.zeros(n_windows, dtype=np.int16),
        layout_version=np.asarray(LAYOUT_VERSION),
        fps=np.asarray(C.FPS),
    )
    return [str(item) for item in clip_ids]


def _write_labels(path: Path, clip_ids: list[str], *, labels: list[str]) -> None:
    """写 ``labels_merged.jsonl``（只填门禁关心的字段）。"""
    import orjson

    with path.open("wb") as handle:
        for index, clip_id in enumerate(clip_ids):
            row = {
                "clip_id": clip_id,
                "zh_label": labels[index % len(labels)],
                "split": "train",
                "is_wushu": False,
                "category_l1": "x",
            }
            handle.write(orjson.dumps(row) + b"\n")


def _norm_stats(std_dim2: float) -> NormStats:
    """构造一份 ``norm_stats``（只改 dim 2 的 std，便于测漂移断言）。"""
    mean = np.zeros(C.FEATURE_DIM, dtype=np.float64)
    std = np.ones(C.FEATURE_DIM, dtype=np.float64)
    for dim in C.NORM_DIMS:
        mean[dim] = 84.0
        std[dim] = 30.0
    std[C.FEAT_LAYOUT["root_height"].start] = float(std_dim2)
    return NormStats(mean=mean, std=std)


def test_scan_shards_counts_windows_and_empty(gate_module: ModuleType, work_dir: Path) -> None:
    """``scan_shards`` 必须准确区分总窗口数与零 mask 窗口数。"""
    shard_dir = work_dir / "shards"
    shard_dir.mkdir()
    first = _write_shard(shard_dir / "shard_00000.npz", n_windows=4, empty_indices=(1, 3), prefix="a")
    second = _write_shard(shard_dir / "shard_00001.npz", n_windows=3, prefix="b")
    scan = gate_module.scan_shards(shard_dir)
    assert scan["n_windows"] == 7
    assert scan["n_empty"] == 2
    assert scan["n_shards"] == 2
    assert scan["clip_ids"] == first + second
    assert scan["layout_versions"] == [LAYOUT_VERSION]
    assert scan["fps_values"] == [C.FPS]


def test_check_norm_stats_drift_passes_for_small_change(
    gate_module: ModuleType, work_dir: Path
) -> None:
    """std 变化 2% 应判「已重算」（阈值 5%）。"""
    new_path = _norm_stats(30.6).save(work_dir / "new.npz")
    prev_path = _norm_stats(30.0).save(work_dir / "prev.npz")
    result = gate_module.check_norm_stats_drift(Path(new_path), Path(prev_path))
    assert result.passed, result.detail


def test_check_norm_stats_drift_fails_when_unchanged_stats_claim_rerun(
    gate_module: ModuleType, work_dir: Path
) -> None:
    """std 变化 40% 必须判 FAIL——这是「窗口集合变了但统计没重算」的信号。"""
    new_path = _norm_stats(42.0).save(work_dir / "new.npz")
    prev_path = _norm_stats(30.0).save(work_dir / "prev.npz")
    result = gate_module.check_norm_stats_drift(Path(new_path), Path(prev_path))
    assert not result.passed


def test_check_norm_stats_drift_fails_without_prev_snapshot(
    gate_module: ModuleType, work_dir: Path
) -> None:
    """缺少旧快照时必须 FAIL（不能「无法证明」就当通过）。"""
    new_path = _norm_stats(30.0).save(work_dir / "new.npz")
    result = gate_module.check_norm_stats_drift(Path(new_path), work_dir / "missing.npz")
    assert not result.passed


def test_build_report_is_green_on_clean_data(gate_module: ModuleType, work_dir: Path) -> None:
    """干净数据上 a)–f) 必须全绿。"""
    shard_dir = work_dir / "shards"
    shard_dir.mkdir()
    clip_ids = _write_shard(shard_dir / "shard_00000.npz", n_windows=5, prefix="c")
    labels_path = work_dir / "labels_merged.jsonl"
    _write_labels(labels_path, clip_ids, labels=["扫地", "拖地"])
    norm_path = _norm_stats(30.3).save(work_dir / "norm.npz")
    prev_path = _norm_stats(30.0).save(work_dir / "prev.npz")

    report, facts = gate_module.build_report(
        cfg_path="x",
        shard_dir=shard_dir,
        labels_path=labels_path,
        norm_path=Path(norm_path),
        prev_norm_path=Path(prev_path),
        expect_windows=5,
        expect_distinct=2,
        expect_wushu=0,
    )
    assert report.passed, [item.line() for item in report.checks if not item.passed]
    assert facts["n_windows"] == 5
    assert facts["n_empty_windows"] == 0
    assert facts["distinct_labels"] == 2


def test_build_report_fails_on_zero_mask_windows(
    gate_module: ModuleType, work_dir: Path
) -> None:
    """断言 c) 必须能真的失败：存在零 mask 窗口就拦下。"""
    shard_dir = work_dir / "shards"
    shard_dir.mkdir()
    clip_ids = _write_shard(
        shard_dir / "shard_00000.npz", n_windows=4, empty_indices=(2,), prefix="d"
    )
    labels_path = work_dir / "labels_merged.jsonl"
    _write_labels(labels_path, clip_ids, labels=["扫地"])
    norm_path = _norm_stats(30.0).save(work_dir / "norm.npz")
    prev_path = _norm_stats(30.0).save(work_dir / "prev.npz")

    report, _facts = gate_module.build_report(
        cfg_path="x",
        shard_dir=shard_dir,
        labels_path=labels_path,
        norm_path=Path(norm_path),
        prev_norm_path=Path(prev_path),
        expect_windows=4,
        expect_distinct=1,
        expect_wushu=0,
    )
    assert not report.passed
    failed = {item.key for item in report.checks if not item.passed}
    assert failed == {"c"}


def test_build_report_fails_on_window_count_mismatch(
    gate_module: ModuleType, work_dir: Path
) -> None:
    """断言 a) 必须能真的失败：窗口数与定案值不符就拦下。"""
    shard_dir = work_dir / "shards"
    shard_dir.mkdir()
    clip_ids = _write_shard(shard_dir / "shard_00000.npz", n_windows=3, prefix="e")
    labels_path = work_dir / "labels_merged.jsonl"
    _write_labels(labels_path, clip_ids, labels=["扫地"])
    norm_path = _norm_stats(30.0).save(work_dir / "norm.npz")
    prev_path = _norm_stats(30.0).save(work_dir / "prev.npz")

    report, _facts = gate_module.build_report(
        cfg_path="x",
        shard_dir=shard_dir,
        labels_path=labels_path,
        norm_path=Path(norm_path),
        prev_norm_path=Path(prev_path),
        expect_windows=13696,
        expect_distinct=1,
        expect_wushu=0,
    )
    assert not report.passed
    assert {item.key for item in report.checks if not item.passed} == {"a"}


def test_build_report_fails_when_labels_miss_a_window(
    gate_module: ModuleType, work_dir: Path
) -> None:
    """断言 b)：``labels_merged`` 少一行就要拦下（否则数据集里会出现无标签样本）。"""
    shard_dir = work_dir / "shards"
    shard_dir.mkdir()
    clip_ids = _write_shard(shard_dir / "shard_00000.npz", n_windows=4, prefix="f")
    labels_path = work_dir / "labels_merged.jsonl"
    _write_labels(labels_path, clip_ids[:-1], labels=["扫地"])  # 故意少一行
    norm_path = _norm_stats(30.0).save(work_dir / "norm.npz")
    prev_path = _norm_stats(30.0).save(work_dir / "prev.npz")

    report, _facts = gate_module.build_report(
        cfg_path="x",
        shard_dir=shard_dir,
        labels_path=labels_path,
        norm_path=Path(norm_path),
        prev_norm_path=Path(prev_path),
        expect_windows=4,
        expect_distinct=1,
        expect_wushu=0,
    )
    assert not report.passed
    assert {item.key for item in report.checks if not item.passed} == {"b"}


def test_build_report_fails_on_wushu_condition_drift(
    gate_module: ModuleType, work_dir: Path
) -> None:
    """断言 e)：武术条件数不等于定案值（24）时必须拦下。"""
    shard_dir = work_dir / "shards"
    shard_dir.mkdir()
    clip_ids = _write_shard(shard_dir / "shard_00000.npz", n_windows=3, prefix="g")
    labels_path = work_dir / "labels_merged.jsonl"
    _write_labels(labels_path, clip_ids, labels=["长拳"])
    norm_path = _norm_stats(30.0).save(work_dir / "norm.npz")
    prev_path = _norm_stats(30.0).save(work_dir / "prev.npz")

    report, _facts = gate_module.build_report(
        cfg_path="x",
        shard_dir=shard_dir,
        labels_path=labels_path,
        norm_path=Path(norm_path),
        prev_norm_path=Path(prev_path),
        expect_windows=3,
        expect_distinct=1,
        expect_wushu=24,
    )
    assert not report.passed
    assert {item.key for item in report.checks if not item.passed} == {"e"}


def test_count_wushu_conditions_dedupes(gate_module: ModuleType) -> None:
    """武术条件数按 ``zh_label`` 去重，且只看 ``is_wushu`` 行。"""
    rows = [
        {"zh_label": "长拳", "is_wushu": True},
        {"zh_label": "长拳", "is_wushu": True},
        {"zh_label": "太极拳", "is_wushu": True},
        {"zh_label": "扫地", "is_wushu": False},
    ]
    assert gate_module.count_wushu_conditions(rows) == 2


# ---------------------------------------------------------------------------
# 03b：重建可视化
# ---------------------------------------------------------------------------
def _synth_sequence(
    contacts: list[tuple[float, float]], heights: list[float], errors: list[float]
) -> tuple[np.ndarray, np.ndarray]:
    """构造 ``pred_denorm`` / ``gt_denorm`` 两段特征（只填高度与接触维）。"""
    frames = len(heights)
    gt = np.zeros((frames, C.FEATURE_DIM), dtype=np.float64)
    pred = np.zeros((frames, C.FEATURE_DIM), dtype=np.float64)
    height_dim = C.FEAT_LAYOUT["root_height"].start
    contact_slice = C.FEAT_LAYOUT["foot_contact"]
    for index in range(frames):
        gt[index, height_dim] = heights[index]
        pred[index, height_dim] = heights[index] + errors[index]
        gt[index, contact_slice.start] = contacts[index][0]
        gt[index, contact_slice.start + 1] = contacts[index][1]
        pred[index, contact_slice.start] = contacts[index][0]
        pred[index, contact_slice.start + 1] = contacts[index][1]
    return pred, gt


def test_phase_breakdown_splits_by_contact_and_height(viz_module: ModuleType) -> None:
    """腾空/着地/站立三段必须按 GT 接触 + 根高度中位数正确切分。"""
    contacts = [
        (1.0, 1.0),  # 着地、站立（高度=中位数）
        (1.0, 0.0),  # 着地、站立
        (0.0, 0.0),  # 腾空
        (0.0, 0.0),  # 腾空
        (1.0, 0.0),  # 着地，但高度远离中位数（下蹲）→ 不算站立
    ]
    heights = [90.0, 90.0, 120.0, 130.0, 60.0]
    errors = [1.0, 1.0, 10.0, 20.0, 4.0]
    pred, gt = _synth_sequence(contacts, heights, errors)
    phases = viz_module.phase_breakdown(pred, gt)

    assert phases["airborne"]["n_frames"] == 2
    assert phases["airborne"]["root_height_mae_cm"] == pytest.approx(15.0)
    assert phases["grounded"]["n_frames"] == 3
    assert phases["grounded"]["root_height_mae_cm"] == pytest.approx(2.0)
    # 站立帧：着地且 |高度 - 中位数(90)| <= 5cm → 只有前两帧
    assert phases["standing"]["n_frames"] == 2
    assert phases["standing"]["root_height_mae_cm"] == pytest.approx(1.0)
    assert phases["counts"]["n_frames"] == 5
    assert phases["counts"]["gt_median_height_cm"] == pytest.approx(90.0)


def test_phase_breakdown_handles_all_airborne(viz_module: ModuleType) -> None:
    """全腾空时着地段必须返回 0 而不是 NaN。"""
    contacts = [(0.0, 0.0)] * 3
    pred, gt = _synth_sequence(contacts, [100.0, 110.0, 120.0], [1.0, 2.0, 3.0])
    phases = viz_module.phase_breakdown(pred, gt)
    assert phases["grounded"]["n_frames"] == 0
    assert phases["grounded"]["root_height_mae_cm"] == 0.0
    assert phases["airborne"]["root_height_mae_cm"] == pytest.approx(2.0)


def test_foot_ground_contact_handles_empty_input(viz_module: ModuleType) -> None:
    """零帧输入必须返回全 0，不抛异常。"""
    result = viz_module.foot_ground_contact(
        np.zeros((0, C.N_JOINTS, 4)), np.zeros((0, 3)), np.zeros(0, dtype=bool)
    )
    assert result["n_grounded"] == 0
    assert result["penetration_ratio"] == 0.0


def test_foot_ground_contact_detects_penetration(viz_module: ModuleType) -> None:
    """把整段根高度压到地下 50cm，着地帧应被判穿地。"""
    frames = 5
    quats = np.tile(np.array([0.0, 0.0, 0.0, 1.0]), (frames, C.N_JOINTS, 1))
    root = np.tile(np.array([0.0, -50.0, 0.0]), (frames, 1))
    grounded = np.ones(frames, dtype=bool)
    result = viz_module.foot_ground_contact(quats, root, grounded)
    assert result["n_grounded"] == frames
    assert result["penetration_ratio"] == pytest.approx(1.0)
    assert result["foot_min_mean_cm"] < -20.0


class _StubDataset:
    """``pick_samples`` 需要的最小接口（分层抽样不依赖真实分片）。"""

    def __init__(self, labels: list[str], wushu: list[bool]) -> None:
        self._labels = labels
        self._wushu = np.asarray(wushu, dtype=bool)

    def sample_labels(self) -> list[str]:
        """逐样本标签。"""
        return list(self._labels)

    def wushu_mask(self) -> np.ndarray:
        """逐样本武术掩码。"""
        return self._wushu


def test_pick_samples_is_stratified_and_deterministic(viz_module: ModuleType) -> None:
    """分层抽样必须覆盖不同标签，且两次调用结果一致。"""
    labels = ["扫地"] * 20 + ["拖地"] * 10 + ["长拳"] * 5 + ["太极拳"] * 2
    wushu = [False] * 30 + [True] * 7
    dataset = _StubDataset(labels, wushu)
    first = viz_module.pick_samples(dataset, 3, prefer_wushu=0)  # type: ignore[arg-type]
    second = viz_module.pick_samples(dataset, 3, prefer_wushu=0)  # type: ignore[arg-type]
    assert first == second
    assert len({label for _index, label in first}) == 3
    for index, label in first:
        assert labels[index] == label


def test_pick_samples_prefers_wushu_when_available(viz_module: ModuleType) -> None:
    """``prefer_wushu=1`` 时结果里必须至少有一个武术样本。"""
    labels = ["扫地"] * 40 + ["拖地"] * 20
    wushu = [False] * 60
    # 把最后两个样本标成武术（同时给它们独立标签，模拟长尾）
    labels[-2:] = ["长拳", "长拳"]
    wushu[-2:] = [True, True]
    dataset = _StubDataset(labels, wushu)
    picks = viz_module.pick_samples(dataset, 3, prefer_wushu=1)  # type: ignore[arg-type]
    flags = dataset.wushu_mask()
    assert any(bool(flags[index]) for index, _label in picks)


def test_pick_samples_respects_explicit_indices(viz_module: ModuleType) -> None:
    """显式下标必须原样采用（便于复现某次可视化）。"""
    dataset = _StubDataset(["扫地", "拖地", "长拳"], [False, False, True])
    picks = viz_module.pick_samples(dataset, 2, explicit=[2, 0])  # type: ignore[arg-type]
    assert picks == [(2, "长拳"), (0, "扫地")]


def test_safe_name_strips_path_separators(viz_module: ModuleType) -> None:
    """文件名清洗必须去掉路径分隔符，避免写到子目录。"""
    assert "/" not in viz_module._safe_name("1.12/扫地")
    assert viz_module._safe_name("///") == "sample"
    assert viz_module._safe_name("长拳第七段") == "长拳第七段"


def test_normalization_consistency_passes_for_standardized_features(
    gate_module: ModuleType, work_dir: Path
) -> None:
    """归一化过的特征（mean 0 / std 1）必须判自洽。"""
    shard_dir = work_dir / "shards"
    shard_dir.mkdir()
    n, window = 20, 30
    feat = np.zeros((n, window, C.FEATURE_DIM), dtype=np.float32)
    for dim in C.NORM_DIMS:
        feat[:, :, dim] = _alternating_block(n, window)
    clip_ids = np.array([f"c{i}" for i in range(n)], dtype="<U80")
    np.savez(
        shard_dir / "shard_00000.npz",
        feat=feat,
        mask=np.ones((n, window), dtype=np.uint8),
        clip_id=clip_ids,
        win_idx=np.zeros(n, dtype=np.int16),
        layout_version=np.asarray(LAYOUT_VERSION),
        fps=np.asarray(C.FPS),
    )
    splits = {f"c{i}": "train" for i in range(n)}
    result = gate_module.normalization_consistency(shard_dir, splits)
    assert result["count"] == n * window
    assert result["max_abs_mean"] < 1e-3
    assert result["max_abs_std_dev"] < 1e-3


def test_normalization_consistency_detects_stale_stats(
    gate_module: ModuleType, work_dir: Path
) -> None:
    """分片用另一套统计量归一化（均值偏离 5）时必须判不自洽。"""
    shard_dir = work_dir / "shards"
    shard_dir.mkdir()
    n, window = 2, 5
    feat = np.zeros((n, window, C.FEATURE_DIM), dtype=np.float32)
    for dim in C.NORM_DIMS:
        feat[:, :, dim] = _alternating_block(n, window) + 5.0
    clip_ids = np.array([f"d{i}" for i in range(n)], dtype="<U80")
    np.savez(
        shard_dir / "shard_00000.npz",
        feat=feat,
        mask=np.ones((n, window), dtype=np.uint8),
        clip_id=clip_ids,
        win_idx=np.zeros(n, dtype=np.int16),
        layout_version=np.asarray(LAYOUT_VERSION),
        fps=np.asarray(C.FPS),
    )
    splits = {f"d{i}": "train" for i in range(n)}
    result = gate_module.normalization_consistency(shard_dir, splits)
    assert result["max_abs_mean"] > gate_module.NORM_TOLERANCE


def test_normalization_consistency_ignores_other_splits(
    gate_module: ModuleType, work_dir: Path
) -> None:
    """val/test 窗口不参与 train 统计量的自洽性检查。"""
    shard_dir = work_dir / "shards"
    shard_dir.mkdir()
    feat = np.full((2, 4, C.FEATURE_DIM), 9.0, dtype=np.float32)
    clip_ids = np.array(["train0", "val0"], dtype="<U80")
    np.savez(
        shard_dir / "shard_00000.npz",
        feat=feat,
        mask=np.ones((2, 4), dtype=np.uint8),
        clip_id=clip_ids,
        win_idx=np.zeros(2, dtype=np.int16),
        layout_version=np.asarray(LAYOUT_VERSION),
        fps=np.asarray(C.FPS),
    )
    splits = {"train0": "train", "val0": "val"}
    result = gate_module.normalization_consistency(shard_dir, splits)
    assert result["count"] == 4  # 只有 train0 的 4 帧
