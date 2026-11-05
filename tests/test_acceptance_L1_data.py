"""L1 数据层验收（QA-PLAN v1.3 §1，D-01 ~ D-07）。

只读校验 ``data/tier_S``：分片 / labels_merged / norm_stats / index / merge_report。
全部离线、fast 圈可跑。阈值集中在 ``acceptance_support.TH``。
"""

from __future__ import annotations

from collections import Counter

import numpy as np
import pytest

from acceptance_support import (
    DATA_DIR,
    TH,
    add_note,
    iter_shards,
    load_index,
    load_labels_merged,
    record,
    shard_window_count,
)

# ---------------------------------------------------------------------------
# session 级缓存：分片只扫一遍
# ---------------------------------------------------------------------------
_CACHE: dict[str, object] = {}


def _scan_cache() -> dict[str, object]:
    """一次性扫描：窗口总数 / mask 全 0 / 首帧 mask / clip_id 集合。"""
    if "scan" in _CACHE:
        return _CACHE["scan"]  # type: ignore[return-value]
    total = 0
    zero_mask = 0
    bad_prefix = 0
    ids: set[str] = set()
    for _path, z in iter_shards():
        feat = z["feat"]
        mask = z["mask"]
        total += int(feat.shape[0])
        zero_mask += int((mask.sum(axis=1) == 0).sum())
        bad_prefix += int((mask[:, 0] == 0).sum())
        ids.update(str(item) for item in z["clip_id"].tolist())
    _CACHE["scan"] = {
        "total": total,
        "zero_mask": zero_mask,
        "bad_prefix": bad_prefix,
        "ids": ids,
    }
    return _CACHE["scan"]  # type: ignore[return-value]


# ---------------------------------------------------------------------------
# D-01 窗口总数三方一致
# ---------------------------------------------------------------------------
def test_D01_window_count_three_way_consistency() -> None:
    labels = load_labels_merged()
    scan = _scan_cache()
    total = shard_window_count()
    label_ids = {row["clip_id"] for row in labels}
    shard_ids = scan["ids"]
    index = load_index()
    index_sum = sum(int(row["n_windows"]) for row in index)

    ok = (
        total == TH["D01_EXPECT_WINDOWS"]
        and len(labels) == TH["D01_EXPECT_WINDOWS"]
        and label_ids == shard_ids
        and len(label_ids) == TH["D01_EXPECT_WINDOWS"]
    )
    record(
        "D-01",
        "PASS" if ok else "FAIL",
        measured={
            "shard_windows": total,
            "labels_rows": len(labels),
            "unique_clip_ids": len(label_ids),
            "id_set_equal": label_ids == shard_ids,
            "index_jsonl_n_windows_sum": index_sum,
        },
        route="data" if not ok else "none",
        notes=(
            "三方一致：分片窗口数 == labels_merged 行数 == 唯一 clip_id 数 == 13696"
            if ok
            else "分片/labels/唯一性三方比对失败"
        ),
    )
    # index.jsonl 是 clip 级计划数，与实际落盘窗口存在口径差（长 clip 降密度 / 尾窗裁剪）。
    # 非阻塞，但如实上报：
    if index_sum != TH["D01_EXPECT_WINDOWS"]:
        add_note(
            f"index.jsonl 的 n_windows 合计 {index_sum} 与分片实际 {TH['D01_EXPECT_WINDOWS']} 差 "
            f"{index_sum - TH['D01_EXPECT_WINDOWS']}（clip 级计划数 vs 实切窗口数的口径差，"
            "README #9/#21 的长 clip 降密度/尾窗阈值所致）；分片与 labels 两方一致，不阻塞。",
            route="data",
        )
    assert ok, f"D-01 失败：shard={total} labels={len(labels)} ids_equal={label_ids == shard_ids}"


# ---------------------------------------------------------------------------
# D-02 mask 合法性
# ---------------------------------------------------------------------------
def test_D02_mask_legality() -> None:
    scan = _scan_cache()
    ok = int(scan["zero_mask"]) == 0 and int(scan["bad_prefix"]) == 0
    record(
        "D-02",
        "PASS" if ok else "FAIL",
        measured={
            "all_zero_mask_windows": int(scan["zero_mask"]),
            "windows_with_mask0_first_frame": int(scan["bad_prefix"]),
        },
        route="data" if not ok else "none",
        notes="mask 全 0 窗口 == 0；每个窗口首帧必须有效（前缀连续性抽验）",
    )
    assert ok, f"D-02 失败：全0窗口={scan['zero_mask']} 首帧无效={scan['bad_prefix']}"


# ---------------------------------------------------------------------------
# D-03 norm_stats 合理性
# ---------------------------------------------------------------------------
def test_D03_norm_stats_self_consistency() -> None:
    from ttm import LAYOUT_VERSION
    from ttm.common import constants as C
    from ttm.data.features import NormStats

    stats = NormStats.load(DATA_DIR / "norm_stats.npz")
    mean, std = np.asarray(stats.mean), np.asarray(stats.std)
    problems: list[str] = []

    # ① 形状与恒等维：只有 NORM_DIMS 四维参与 z-score，其余必须 mean=0/std=1
    if mean.shape != (C.FEATURE_DIM,) or std.shape != (C.FEATURE_DIM,):
        problems.append(f"shape 错误: {mean.shape}/{std.shape}")
    identity = [dim for dim in range(C.FEATURE_DIM) if dim not in C.NORM_DIMS]
    if identity and not np.allclose(mean[identity], 0.0, atol=1e-12):
        problems.append("恒等维 mean != 0")
    if identity and not np.allclose(std[identity], 1.0, atol=1e-12):
        problems.append("恒等维 std != 1")

    # ② 自洽性：**全量** train 窗口（分片本身就是归一化后的特征），各归一维
    #    mean≈0 / std≈1 —— 归一化定义式判据（02d 同口径，非新旧 std 漂移）。
    #    注意不能抽样：按类有序的分片里抽前 1000 窗会拿到单类别切片，
    #    子集统计天然偏离全局（02d 门禁是全量 163 万帧口径）。
    labels = load_labels_merged()
    train_ids = {row["clip_id"] for row in labels if row["split"] == "train"}
    acc_sum = np.zeros(len(C.NORM_DIMS))
    acc_sq = np.zeros(len(C.NORM_DIMS))
    count = 0
    roundtrip_err = 0.0
    windows_checked = 0
    for path, z in iter_shards():
        ids = z["clip_id"]
        row_mask = np.isin(ids, list(train_ids))
        if not row_mask.any():
            continue
        feat = z["feat"].astype(np.float64)[row_mask]
        mask = z["mask"].astype(bool)[row_mask]
        # 反归一化 → 再归一化往返（逐 shard 全量）
        denorm = feat.copy()
        for dim in C.NORM_DIMS:
            denorm[:, :, dim] = feat[:, :, dim] * std[dim] + mean[dim]
        renorm = denorm.copy()
        for dim in C.NORM_DIMS:
            renorm[:, :, dim] = (denorm[:, :, dim] - mean[dim]) / std[dim]
        roundtrip_err = max(roundtrip_err, float(np.abs(renorm - feat).max()))
        # 自洽统计（只统计有效帧）
        for i, dim in enumerate(C.NORM_DIMS):
            values = feat[:, :, dim][mask]
            acc_sum[i] += values.sum()
            acc_sq[i] += (values**2).sum()
        count += int(mask.sum())
        windows_checked += feat.shape[0]

    norm_mean = acc_sum / max(count, 1)
    norm_std = np.sqrt(np.clip(acc_sq / max(count, 1) - norm_mean**2, 0.0, None))
    # 注意：分片特征已是归一化后的数据，denorm 后应恢复原始量纲 → 自洽性直接用
    # 「分片值」统计（它们就是归一化后的 train 窗口）：
    if np.abs(norm_mean).max() > TH["D03_SELFCONSIST_ATOL"] or np.abs(norm_std - 1).max() > TH[
        "D03_SELFCONSIST_ATOL"
    ]:
        problems.append(
            f"自洽性超差: max|mean|={np.abs(norm_mean).max():.6f}, max|std-1|={np.abs(norm_std - 1).max():.6f}"
        )
    if roundtrip_err > TH["D03_ROUNDTRIP_ATOL"]:
        problems.append(f"往返误差 {roundtrip_err:.3e} > {TH['D03_ROUNDTRIP_ATOL']:.1e}")

    version = int(np.load(DATA_DIR / "norm_stats.npz", allow_pickle=True)["layout_version"])
    if version != LAYOUT_VERSION:
        problems.append(f"layout_version={version} != {LAYOUT_VERSION}")

    ok = not problems
    record(
        "D-03",
        "PASS" if ok else "FAIL",
        measured={
            "layout_version": version,
            "norm_dims": list(C.NORM_DIMS),
            "selfconsist_max_abs_mean": round(float(np.abs(norm_mean).max()), 8),
            "selfconsist_max_abs_std_minus_1": round(float(np.abs(norm_std - 1).max()), 8),
            "roundtrip_max_abs_err": round(float(roundtrip_err), 10),
            "windows_checked": windows_checked,
            "frames_counted": count,
        },
        route="data" if not ok else "none",
        notes="；".join(problems) if problems else "恒等维严格 0/1；train 归一化后 mean≈0/std≈1；往返误差 <1e-6",
    )
    assert ok, f"D-03 失败: {problems}"


# ---------------------------------------------------------------------------
# D-04 条件数双口径 179 / 178
# ---------------------------------------------------------------------------
def test_D04_condition_count_179_178() -> None:
    labels = load_labels_merged()
    conds = {row["zh_label"] for row in labels}
    train_conds = {row["zh_label"] for row in labels if row["split"] == "train"}
    diff = conds - train_conds

    ok = (
        len(conds) == TH["D04_DISTINCT_FULL"]
        and len(train_conds) == TH["D04_DISTINCT_TRAIN"]
        and diff == TH["D04_EXPECT_DIFF"]
    )
    record(
        "D-04",
        "PASS" if ok else "FAIL",
        measured={
            "distinct_full": len(conds),
            "distinct_train": len(train_conds),
            "full_minus_train": sorted(diff),
        },
        route="data" if not ok else "none",
        notes=(
            "179 全集 / 178 train 可学；差集恰为 {通背拳}（EX-01 口径例外：按 clip 切分整块落 val，"
            "不计入 hit 率分母，见 QA-PLAN §5.4.2）"
            if ok
            else "条件数或差集不符"
        ),
    )
    assert ok, f"D-04 失败: full={len(conds)} train={len(train_conds)} diff={sorted(diff)}"


# ---------------------------------------------------------------------------
# D-05 武术 24 条件每类 ≥3 窗
# ---------------------------------------------------------------------------
def test_D05_wushu_condition_floor() -> None:
    labels = load_labels_merged()
    counter = Counter(row["zh_label"] for row in labels if row.get("is_wushu"))
    thin = {name: count for name, count in counter.items() if count < TH["D05_WUSHU_MIN_WINDOWS"]}
    ok = len(counter) == TH["D05_WUSHU_CONDITIONS"] and not thin
    record(
        "D-05",
        "PASS" if ok else "FAIL",
        measured={
            "wushu_conditions": len(counter),
            "min_windows": min(counter.values()) if counter else 0,
            "min_label": min(counter, key=counter.get) if counter else "",
            "below_floor": thin,
        },
        route="data" if not ok else "none",
        notes="24 个武术基名条件全部 ≥3 窗" if ok else f"武术条件数或下限不达标: {thin}",
    )
    assert ok, f"D-05 失败: conditions={len(counter)} thin={thin}"


# ---------------------------------------------------------------------------
# D-06 residual 21 项如实上报
# ---------------------------------------------------------------------------
def test_D06_merge_residual_reported_honestly() -> None:
    import json

    labels = load_labels_merged()
    conds = {row["zh_label"] for row in labels}
    report = json.loads(
        (DATA_DIR.parent.parent / "runs/tier_s/merge_labels/merge_report.json")
        .read_text(encoding="utf-8")
    )
    residual = report.get("residual_below_threshold", [])

    problems: list[str] = []
    if len(residual) != TH["D06_RESIDUAL_COUNT"]:
        problems.append(f"residual 条数 {len(residual)} != {TH['D06_RESIDUAL_COUNT']}")
    # README #16 契约：残留在阈值以下的基名**如实上报并保留为条件**（不许塞进「武术其他」）。
    hidden = [name for name in residual if name not in conds]
    if hidden:
        problems.append(f"residual 未出现在条件集（被静默丢弃）: {hidden}")
    junk = [name for name in conds if "其他" in str(name) and "wushu" in str(name).lower()]
    if junk:
        problems.append(f"出现捡垃圾标签: {junk}")

    ok = not problems
    record(
        "D-06",
        "PASS" if ok else "FAIL",
        measured={
            "residual_count": len(residual),
            "residual_missing_from_conditions": hidden,
            "junk_labels": junk,
            "distinct_after": report.get("distinct_after"),
            "merged_labels": report.get("merged_labels"),
        },
        route="spec" if ok else "data",
        notes=(
            "residual 21 项全部如实上报且保留为条件（README #16 契约）。"
            "⚠ QA-PLAN v1.0 D-06 写的是「residual 不进条件」——与实际契约相反，"
            "需按 README 契约做规格修订（本测试按真实契约断言）。"
            if ok
            else "；".join(problems)
        ),
    )
    if ok:
        add_note(
            "D-06 口径修订建议：QA-PLAN v1.0 的「residual 21 项不进条件」与实现契约（README #16："
            "残留基名如实上报并保留为条件、不许塞「武术其他」）相反；测试按 README 契约断言，"
            "QA-PLAN 待规格修订。",
            route="spec",
        )
    assert ok, f"D-06 失败: {problems}"


# ---------------------------------------------------------------------------
# D-07 split 泄漏检查
# ---------------------------------------------------------------------------
def test_D07_split_leakage() -> None:
    index = load_index()
    by_split: dict[str, set[str]] = {"train": set(), "val": set(), "test": set()}
    clip_splits: dict[str, set[str]] = {}
    for row in index:
        by_split.setdefault(row["split"], set()).add(row["take_id"])
        clip_splits.setdefault(row["clip_id"], set()).add(row["split"])

    overlap_tv = by_split["train"] & by_split["val"]
    overlap_tt = by_split["train"] & by_split["test"]
    overlap_vt = by_split["val"] & by_split["test"]
    multi = [clip for clip, splits in clip_splits.items() if len(splits) > 1]
    ok = not overlap_tv and not overlap_tt and not overlap_vt and not multi
    record(
        "D-07",
        "PASS" if ok else "FAIL",
        measured={
            "train_val_overlap": len(overlap_tv),
            "train_test_overlap": len(overlap_tt),
            "val_test_overlap": len(overlap_vt),
            "clips_in_multiple_splits": len(multi),
            "takes_per_split": {key: len(value) for key, value in by_split.items()},
        },
        route="data" if not ok else "none",
        notes="train/val/test 的 take_id 两两无交集；clip 不跨 split" if ok else "存在 split 泄漏",
    )
    assert ok, f"D-07 失败: tv={len(overlap_tv)} tt={len(overlap_tt)} vt={len(overlap_vt)} multi={len(multi)}"
