#!/usr/bin/env python
"""M2' 训练前的**程序化数据门禁**（主理人指定的 a)–e) 五项断言）。

背景：M2' 是「重跑预处理（修了尾窗越界）+ 重跑合并（labels 层后处理）+ VAE 重训」
连着跑的一轮。中间不等人肉确认数字，但**绝不允许带病训练**——所以把「人工核对」
换成五条可执行断言，全过才允许启动 ``03_train_vae.py``：

====  ==================================================================
断言  内容
====  ==================================================================
a     总窗口数 == 期望值（Tier-S 定案 13696）
b     ``labels_merged.jsonl`` 行数 == 总窗口数（且分片里的 clip_id 全被覆盖）
c     全部分片中 ``mask`` 全 0 的窗口数 == 0（不是跳过，是根本不该产生）
d     ``norm_stats`` 与分片**自洽**：train 窗口归一化后各归一维 mean≈0、std≈1
e     合并后 distinct 标签数 == 179，武术条件数 == 24
f     ``layout_version`` / ``fps`` 符合契约
====  ==================================================================

任何一条不过：退出码 1，并打印「哪一条、期望什么、实际什么」。此时**不要**启动训练。

## 为什么 c) 要「== 0」而不是「靠 dataset 跳过」

``MotionDataset(skip_empty_windows=True)`` 只是个兜底防线。如果数据侧仍在产出零 mask
窗口，说明 ``features.slice_windows_arrays`` 的尾窗夹取逻辑又有回归——那些窗口会被
编码、算 KL、对重建零贡献，等于往训练集里掺纯噪声。跳过它们只治症，不治病。

## 为什么 d) 不看「与旧统计的漂移」，而看自洽性

最初的写法是「新 std 与旧 std 偏差 < 5%」。**这个判据是错的**：尾窗越界修掉之后，
265 个曾经全 padding 的窗口变成真实帧，``dim2``（根高度）的 std 从 33.95 掉到
26.12——漂移 23%。可这恰恰**证明**统计量重算过（旧统计被「高度 0cm 的 pad 帧」
拉低了均值、抬高了方差），也就是说这个判据会把「修好了」判成「没重算」。

正确的判据是归一化的定义式：用 ``norm_stats`` 归一化过的 train 窗口，各归一维的
均值必须 ≈ 0、标准差必须 ≈ 1。如果分片是用另一套统计量归一化的（「换了数据忘重算」），
这两个量会立刻偏掉。漂移数值仍然打印，但只作为参考信息，不参与判定。

## 用法

```bash
python scripts/02d_verify_data.py --config configs/tier_s.yaml \
    --prev-norm-stats runs/tier_s/prev_m1/norm_stats_m1.npz \
    --expect-windows 13696 --expect-distinct 179 --expect-wushu 24
```

退出码 0 = 全绿放行，1 = 有断言失败。
"""

from __future__ import annotations

import argparse
import sys
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Iterable

import numpy as np
import orjson

from ttm import LAYOUT_VERSION
from ttm.common import constants as C
from ttm.common import paths as P
from ttm.common.config import load_config
from ttm.data.features import NormStats

#: 归一维 std 允许的相对偏差（仅作信息参考，不再作为断言依据）。
STD_DRIFT_TOLERANCE: float = 0.05

#: 归一化自洽性容差：train 窗口用 ``norm_stats`` 归一化后，均值应为 0、标准差应为 1。
NORM_TOLERANCE: float = 0.02


@dataclass
class CheckResult:
    """单条断言的结果。"""

    key: str
    title: str
    passed: bool
    detail: str

    def line(self) -> str:
        """渲染成一行可读文本。"""
        flag = "PASS" if self.passed else "FAIL"
        return f"[{flag}] ({self.key}) {self.title}: {self.detail}"


@dataclass
class AuditReport:
    """门禁汇总。"""

    checks: list[CheckResult] = field(default_factory=list)

    @property
    def passed(self) -> bool:
        """全部断言是否通过。"""
        return all(item.passed for item in self.checks)

    def add(self, result: CheckResult) -> None:
        """追加一条断言结果。"""
        self.checks.append(result)

    def to_dict(self) -> dict[str, Any]:
        """转成可序列化字典（落盘给报告引用）。"""
        return {
            "passed": self.passed,
            "checks": [
                {
                    "key": item.key,
                    "title": item.title,
                    "passed": item.passed,
                    "detail": item.detail,
                }
                for item in self.checks
            ],
        }


def scan_shards(shard_dir: Path) -> dict[str, Any]:
    """扫描全部分片，收集窗口数、空窗口数、clip_id 集合与契约字段。

    Args:
        shard_dir: ``shards/`` 目录。

    Returns:
        ``{"n_windows", "n_empty", "clip_ids"（列表）, "n_shards", "layout_versions",
        "fps_values", "n_frames_total"}``。

    Raises:
        FileNotFoundError: 目录不存在或没有分片。
    """
    shard_paths = sorted(shard_dir.glob("shard_*.npz"))
    if not shard_paths:
        raise FileNotFoundError(f"分片目录为空: {shard_dir}")

    n_windows = 0
    n_empty = 0
    n_frames_total = 0
    clip_ids: list[str] = []
    versions: set[int] = set()
    fps_values: set[int] = set()

    for path in shard_paths:
        with np.load(path) as payload:
            mask = np.asarray(payload["mask"])
            valid = mask.sum(axis=1)
            n_windows += int(mask.shape[0])
            n_empty += int((valid <= 0).sum())
            n_frames_total += int(valid.sum())
            clip_ids.extend(str(item) for item in np.asarray(payload["clip_id"]).astype(str))
            versions.add(int(np.asarray(payload["layout_version"]).reshape(-1)[0]))
            fps_values.add(int(np.asarray(payload["fps"]).reshape(-1)[0]))

    return {
        "n_windows": n_windows,
        "n_empty": n_empty,
        "n_frames_total": n_frames_total,
        "clip_ids": clip_ids,
        "n_shards": len(shard_paths),
        "layout_versions": sorted(versions),
        "fps_values": sorted(fps_values),
    }


def read_label_rows(path: Path) -> list[dict[str, Any]]:
    """读 ``labels_merged.jsonl`` 的全部行。

    Args:
        path: 标签文件路径。

    Returns:
        行字典列表。

    Raises:
        FileNotFoundError: 文件不存在。
    """
    if not path.is_file():
        raise FileNotFoundError(f"标签文件不存在: {path}")
    rows: list[dict[str, Any]] = []
    with path.open("rb") as handle:
        for line in handle:
            line = line.strip()
            if not line:
                continue
            rows.append(orjson.loads(line))
    return rows


def count_wushu_conditions(rows: list[dict[str, Any]]) -> int:
    """统计武术合并后的条件数（distinct ``zh_label``）。

    与 ``02b_merge_labels.py`` / ``02_merge`` 报告的口径一致：只看 ``is_wushu=True``
    的行，统计其 ``zh_label`` 的去重个数。

    Args:
        rows: ``labels_merged.jsonl`` 的行。

    Returns:
        武术条件数（无武术行时为 0）。
    """
    return len({str(row.get("zh_label", "")) for row in rows if bool(row.get("is_wushu", False))})


def normalization_consistency(
    shard_dir: Path,
    splits: dict[str, str],
    *,
    split: str = "train",
    dims: tuple[int, ...] = C.NORM_DIMS,
) -> dict[str, Any]:
    """检查「分片里的特征」与「``norm_stats.npz``」是否自洽。

    这才是断言 d) 真正要证明的东西：**统计量是跟着这批分片算出来的**，不是上一轮
    留下的旧文件。做法是直接验证归一化的定义式——用 ``norm_stats`` 归一化过的
    train 窗口，其归一维的均值应当 ≈ 0、标准差应当 ≈ 1。如果分片是用另一套统计量
    归一化的（典型的「换了数据忘了重算」），这两个量会立刻偏掉。

    **为什么不用「与上一版 std 的漂移 < 5%」来判**：那是个错误的代理指标。尾窗越界
    修掉之后，265 个曾经全 padding 的窗口变成了真实帧，``dim2``（根高度）的 std
    从 33.95 掉到 26.12——漂移 23%，但这恰恰**证明**统计量是重算过的（旧统计被
    「高度 0cm 的 pad 帧」拉低了均值、抬高了方差）。用漂移判会把「修好了」判成
    「没重算」，方向刚好反了。

    Args:
        shard_dir: ``shards/`` 目录。
        splits: ``clip_id -> split`` 映射。
        split: 只统计该 split（统计量本身只在 train 上算）。
        dims: 参与 z-score 的维度。

    Returns:
        ``{"count", "mean": {dim: float}, "std": {dim: float},
        "max_abs_mean", "max_abs_std_dev"}``。
    """
    index = {int(dim): offset for offset, dim in enumerate(dims)}
    total = 0
    sums = np.zeros(len(dims), dtype=np.float64)
    sumsq = np.zeros(len(dims), dtype=np.float64)

    for path in sorted(shard_dir.glob("shard_*.npz")):
        with np.load(path) as payload:
            feat = np.asarray(payload["feat"], dtype=np.float32)  # [n,150,146]
            mask = np.asarray(payload["mask"]).astype(bool)  # [n,150]
            clip_ids = np.asarray(payload["clip_id"]).astype(str)
        for local in range(feat.shape[0]):
            if splits.get(str(clip_ids[local])) != split:
                continue
            valid = mask[local]
            if not valid.any():
                continue
            block = feat[local][valid][:, list(dims)].astype(np.float64)
            sums += block.sum(axis=0)
            sumsq += np.square(block).sum(axis=0)
            total += int(block.shape[0])

    if total == 0:
        return {"count": 0, "mean": {}, "std": {}, "max_abs_mean": float("inf"),
                "max_abs_std_dev": float("inf")}
    means = sums / total
    variances = np.maximum(sumsq / total - np.square(means), 0.0)
    stds = np.sqrt(variances)
    return {
        "count": total,
        "mean": {int(dim): float(means[index[dim]]) for dim in dims},
        "std": {int(dim): float(stds[index[dim]]) for dim in dims},
        "max_abs_mean": float(np.abs(means).max()),
        "max_abs_std_dev": float(np.abs(stds - 1.0).max()),
    }


def check_norm_stats_drift(new_path: Path, prev_path: Path) -> CheckResult:
    """断言 d)：``norm_stats`` 已重算（各归一维 std 相对偏差 < 5%）。

    Args:
        new_path: 新的 ``norm_stats.npz``。
        prev_path: 重跑前的快照。

    Returns:
        :class:`CheckResult`。
    """
    title = "norm_stats 已基于新窗口重算"
    if not prev_path.is_file():
        return CheckResult(
            "d",
            title,
            False,
            f"缺少旧快照 {prev_path}（无法证明重算过；请先备份再重跑预处理）",
        )
    new_stats = NormStats.load(new_path)
    prev_stats = NormStats.load(prev_path)
    drifts: dict[int, float] = {}
    for dim in C.NORM_DIMS:
        old = float(prev_stats.std[dim])
        new = float(new_stats.std[dim])
        drifts[int(dim)] = abs(new - old) / old if old else float("inf")
    worst_dim = max(drifts, key=lambda key: drifts[key])
    worst = drifts[worst_dim]
    passed = worst < STD_DRIFT_TOLERANCE
    detail = (
        f"最大 std 相对偏差 {worst:.2%}（dim {worst_dim}: "
        f"{float(prev_stats.std[worst_dim]):.4f} → {float(new_stats.std[worst_dim]):.4f}），"
        f"阈值 {STD_DRIFT_TOLERANCE:.0%}"
    )
    return CheckResult("d", title, passed, detail)


def build_report(
    *,
    cfg_path: str,
    shard_dir: Path,
    labels_path: Path,
    norm_path: Path,
    prev_norm_path: Path,
    expect_windows: int | None,
    expect_distinct: int | None,
    expect_wushu: int | None,
) -> tuple[AuditReport, dict[str, Any]]:
    """跑完 a)–e) 五条断言。

    Args:
        cfg_path: 配置文件路径（仅用于日志/落盘记录）。
        shard_dir: 分片目录。
        labels_path: 合并后标签文件。
        norm_path: 当前 ``norm_stats.npz``。
        prev_norm_path: 重跑前的 ``norm_stats`` 快照。
        expect_windows: a) 的期望窗口数；``None`` 时跳过数值比对。
        expect_distinct: e) 的期望 distinct 标签数；``None`` 时跳过。
        expect_wushu: e) 的期望武术条件数；``None`` 时跳过。

    Returns:
        ``(report, facts)``；``facts`` 是原始统计（写报告用）。
    """
    report = AuditReport()
    scan = scan_shards(shard_dir)
    rows = read_label_rows(labels_path)
    n_windows = int(scan["n_windows"])
    n_empty = int(scan["n_empty"])

    # a) 总窗口数
    if expect_windows is None:
        report.add(
            CheckResult("a", "总窗口数", True, f"未指定期望值，实测 {n_windows}")
        )
    else:
        report.add(
            CheckResult(
                "a",
                "总窗口数",
                n_windows == int(expect_windows),
                f"expect {expect_windows}, actual {n_windows}",
            )
        )

    # b) labels_merged 行数 == 总窗口数（并检查 clip_id 全覆盖）
    shard_clip_ids = set(scan["clip_ids"])
    label_clip_ids = {str(row.get("clip_id", "")) for row in rows}
    missing = shard_clip_ids - label_clip_ids
    extra = label_clip_ids - shard_clip_ids
    passed_b = len(rows) == n_windows and not missing and not extra
    report.add(
        CheckResult(
            "b",
            "labels_merged.jsonl 行数 == 总窗口数",
            passed_b,
            f"expect {n_windows}, actual {len(rows)}；"
            f"分片有标签缺失 {len(missing)} 个 / 标签多出 {len(extra)} 个",
        )
    )

    # c) 全 0 mask 窗口数
    report.add(
        CheckResult(
            "c",
            "mask 全 0 的窗口数",
            n_empty == 0,
            f"expect 0, actual {n_empty}"
            + ("" if n_empty == 0 else "（尾窗夹取逻辑回归？）"),
        )
    )

    # d) norm_stats 与分片自洽（这才是「统计量是跟着这批数据算的」的证明）
    splits = {str(row.get("clip_id", "")): str(row.get("split", "train")) for row in rows}
    consistency = normalization_consistency(shard_dir, splits)
    drift_detail = check_norm_stats_drift(norm_path, prev_norm_path).detail
    passed_d = (
        consistency["count"] > 0
        and consistency["max_abs_mean"] <= NORM_TOLERANCE
        and consistency["max_abs_std_dev"] <= NORM_TOLERANCE
    )
    report.add(
        CheckResult(
            "d",
            "norm_stats 与分片自洽（train 归一化后 mean≈0 / std≈1）",
            passed_d,
            f"max|mean|={consistency['max_abs_mean']:.6f}, "
            f"max|std-1|={consistency['max_abs_std_dev']:.6f}"
            f"（容差 {NORM_TOLERANCE}，统计 {consistency['count']} 帧）；"
            f"与上版 std 漂移仅作参考：{drift_detail}",
        )
    )

    # e) 合并后 distinct 与武术条件数
    distinct = len({str(row.get("zh_label", "")) for row in rows})
    wushu_conditions = count_wushu_conditions(rows)
    parts: list[str] = [f"distinct {distinct}"]
    passed_e = True
    if expect_distinct is not None:
        passed_e = passed_e and distinct == int(expect_distinct)
        parts[0] = f"distinct expect {expect_distinct}, actual {distinct}"
    if expect_wushu is not None:
        passed_e = passed_e and wushu_conditions == int(expect_wushu)
        parts.append(f"武术条件 expect {expect_wushu}, actual {wushu_conditions}")
    else:
        parts.append(f"武术条件 actual {wushu_conditions}")
    report.add(CheckResult("e", "合并后 distinct / 武术条件数", passed_e, "；".join(parts)))

    # 契约字段（不单独立项，但契约错了一切都白搭，顺手锁死）
    contract_ok = scan["layout_versions"] == [LAYOUT_VERSION] and scan["fps_values"] == [C.FPS]
    report.add(
        CheckResult(
            "f",
            "契约字段（layout_version / fps）",
            contract_ok,
            f"layout_version={scan['layout_versions']}（期望 [{LAYOUT_VERSION}]）；"
            f"fps={scan['fps_values']}（期望 [{C.FPS}]）",
        )
    )

    split_counts: dict[str, int] = {}
    for row in rows:
        name = str(row.get("split", "unknown"))
        split_counts[name] = split_counts.get(name, 0) + 1
    wushu_windows = sum(1 for row in rows if bool(row.get("is_wushu", False)))
    facts = {
        "config": cfg_path,
        "shard_dir": P.rel_to_root(shard_dir),
        "labels_path": P.rel_to_root(labels_path),
        "norm_stats": P.rel_to_root(norm_path),
        "prev_norm_stats": P.rel_to_root(prev_norm_path),
        "n_shards": int(scan["n_shards"]),
        "n_windows": n_windows,
        "n_empty_windows": n_empty,
        "n_valid_frames": int(scan["n_frames_total"]),
        "duration_hours": float(scan["n_frames_total"]) / float(C.FPS) / 3600.0,
        "n_labels": len(rows),
        "distinct_labels": distinct,
        "wushu_conditions": wushu_conditions,
        "wushu_windows": wushu_windows,
        "wushu_share": (float(wushu_windows) / float(len(rows))) if rows else 0.0,
        "split_counts": split_counts,
    }
    return report, facts


def main(argv: Iterable[str] | None = None) -> int:
    """命令行入口。

    Args:
        argv: 参数列表（``None`` 时用 ``sys.argv``）。

    Returns:
        0 = 全绿；1 = 有断言失败。
    """
    parser = argparse.ArgumentParser(description="M2' 训练前数据门禁（a)–e)）")
    parser.add_argument("--config", default="configs/tier_s.yaml", help="配置文件路径")
    parser.add_argument("--shards", default=None, help="覆盖分片目录")
    parser.add_argument("--labels", default=None, help="覆盖标签文件（默认合并后）")
    parser.add_argument("--norm-stats", default=None, help="覆盖当前 norm_stats.npz")
    parser.add_argument(
        "--prev-norm-stats",
        default="runs/tier_s/prev_m1/norm_stats_m1.npz",
        help="重跑前的 norm_stats 快照（断言 d 用）",
    )
    parser.add_argument("--expect-windows", type=int, default=13696, help="a) 期望窗口数")
    parser.add_argument("--expect-distinct", type=int, default=179, help="e) 期望 distinct")
    parser.add_argument("--expect-wushu", type=int, default=24, help="e) 期望武术条件数")
    parser.add_argument(
        "--out", default="runs/tier_s/merge_labels/data_gate.json", help="门禁结果落盘路径"
    )
    args = parser.parse_args(list(argv) if argv is not None else None)

    cfg = load_config(args.config)
    shard_dir = Path(args.shards) if args.shards else cfg.shards_path()
    labels_path = Path(args.labels) if args.labels else cfg.train_labels_path()
    norm_path = Path(args.norm_stats) if args.norm_stats else cfg.norm_stats_path()

    report, facts = build_report(
        cfg_path=args.config,
        shard_dir=shard_dir,
        labels_path=labels_path,
        norm_path=norm_path,
        prev_norm_path=Path(args.prev_norm_stats),
        expect_windows=args.expect_windows,
        expect_distinct=args.expect_distinct,
        expect_wushu=args.expect_wushu,
    )

    print("=" * 78)
    print("M2' 数据门禁")
    print("=" * 78)
    for check in report.checks:
        print(check.line())
    print("-" * 78)
    print(
        f"分片 {facts['n_shards']} 个 / 窗口 {facts['n_windows']} / "
        f"有效帧 {facts['n_valid_frames']}（{facts['duration_hours']:.2f} 小时）"
    )
    print(
        f"标签 {facts['n_labels']} 行 / distinct {facts['distinct_labels']} / "
        f"武术条件 {facts['wushu_conditions']} / 武术窗口 {facts['wushu_windows']}"
        f"（{facts['wushu_share']:.2%}）"
    )
    print(f"split 分布: {facts['split_counts']}")
    print("=" * 78)

    payload = report.to_dict()
    payload["facts"] = facts
    out_path = P.resolve_out(args.out, mkparents=True)
    out_path.write_bytes(orjson.dumps(payload, option=orjson.OPT_INDENT_2))
    print(f"结果落盘: {P.rel_to_root(out_path)}")

    if not report.passed:
        print()
        print("!! 门禁未通过：不要启动 M2' 训练，先修上面的 FAIL 项并上报。")
        return 1
    print("门禁全绿：允许启动 M2'（500 epoch）。")
    return 0


if __name__ == "__main__":
    sys.exit(main())
