"""T03 脚本：扫描 BVH 目录树，生成 ``index.jsonl`` + ``failed.jsonl``。

用法::

    . D:/TTM/scripts/env.ps1
    python scripts/01_scan.py --config configs/tier_s.yaml              # 全量（Tier-S 上限 3000）
    python scripts/01_scan.py --config configs/smoke.yaml --limit 20    # 小样本调试

产物::

    data/tier_{tier}/index.jsonl    每行一个可入训 clip（含 split / 类别 / 窗口数）
    data/tier_{tier}/failed.jsonl   每个被淘汰的文件（含 stage / error 原因）
    runs/scan/config_resolved.yaml  本次生效配置（可复现性）

退出码：采集到至少 1 个 clip 时为 0，否则为 1。
``Z:`` 是只读慢速网络盘：本脚本只做**单趟**目录遍历 + 单趟头部读；
不带 ``--limit`` 时全量扫描 26,157 个文件仍可能耗时数分钟。
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

# --- sys.path 引导（即便未做 editable install 也能 import ttm）----------------
_PROJECT_ROOT = Path(__file__).resolve().parents[1]
_SRC_DIR = _PROJECT_ROOT / "src"
if str(_SRC_DIR) not in sys.path:
    sys.path.insert(0, str(_SRC_DIR))

from ttm.common import paths as P  # noqa: E402
from ttm.common.config import dump_resolved, load_config  # noqa: E402
from ttm.common.logutil import close_all, get_logger  # noqa: E402
from ttm.common.seed import set_seed  # noqa: E402
from ttm.data.scan import SPLITS, Scanner  # noqa: E402


def build_argparser() -> argparse.ArgumentParser:
    """构造命令行参数解析器。"""
    parser = argparse.ArgumentParser(
        description="扫描 BVH 目录树生成 index.jsonl（TTM T03）",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument(
        "--config", default="configs/default.yaml", help="配置文件路径（相对项目根）"
    )
    parser.add_argument(
        "--limit",
        type=int,
        default=0,
        help="最多采集多少个 clip（0 = 用配置里的 data.max_files）；调试时建议 20-200",
    )
    parser.add_argument("--workers", type=int, default=8, help="并发读文件头的线程数")
    parser.add_argument("--dry-run", action="store_true", help="只扫描不写盘")
    return parser


def _print_summary(result, cfg, index_path=None) -> None:  # type: ignore[no-untyped-def]
    """打印扫描摘要。"""
    stats = result.stats
    balanced = stats.get("balanced", {})
    print("=" * 84)
    print(f"TTM 扫描摘要 · tier={cfg.data.tier} · 用时 {result.elapsed_s:.1f}s")
    print("=" * 84)
    print(f"BVH 根目录          : {', '.join(cfg.data.bvh_roots)}")
    print(f"类别白名单          : {cfg.data.include_categories or '全量'}")
    print(f"遍历 Take 目录      : {stats.get('takes_seen', 0)}（空目录 {stats.get('takes_empty', 0)}）")
    print(f"枚举 .bvh 文件      : {stats.get('files_seen', 0)}")
    print(f"Take 去重           : {stats.get('takes_deduped', 0)} 个 Take 丢弃了多余演员")
    print(f"候选 clip           : {stats.get('clips_candidates', 0)}")
    print(f"入训 clip           : {stats.get('clips_selected', 0)}（clip 上限 {stats.get('limit', 0)}）")
    print(f"总训练窗口          : {stats.get('windows_total', 0)}")
    counts = stats.get("split_counts", {})
    print(
        "切分                : "
        + " / ".join(f"{name}={counts.get(name, 0)}" for name in SPLITS)
    )
    print("淘汰统计：")
    print(f"  头部校验失败      : {stats.get('clips_rejected_header', 0)}")
    print(f"  过短（<75 帧）    : {stats.get('clips_rejected_too_short', 0)}")
    print(f"  clip_id 重复      : {stats.get('clips_rejected_duplicate_id', 0)}")
    print(f"  失败总数          : {stats.get('failed', 0)}")

    if balanced:
        print("-" * 84)
        print("配额均衡（窗口口径，细类 round-robin / least-filled queue）：")
        print(
            f"  预算 / 实际       : {balanced.get('windows_selected', 0)} / "
            f"{balanced.get('window_budget', 0)} 窗口"
            + ("（预算用尽）" if balanced.get("window_budget_hit") else "（素材不足，预算未满）")
        )
        print(
            f"  细类覆盖          : {balanced.get('leaves_selected', 0)} / "
            f"{balanced.get('leaves_available', 0)} 个细类进入训练集"
            f"（丢弃 {balanced.get('leaves_dropped', 0)} 个）"
        )
        print(f"  单类窗口上限      : {balanced.get('leaf_window_cap', 0)} 窗")
        print(
            f"  最大单一细类占比  : {balanced.get('max_leaf_share', 0.0):.1%}"
            f"（占预算 {balanced.get('max_leaf_share_of_budget', 0.0):.1%}）"
        )
        top = balanced.get("top_leaves", [])
        if top:
            print("  Top 细类（按窗口）：")
            for item in top[:10]:
                print(
                    f"    {item['windows']:6d} 窗 / {item['clips']:4d} clip / "
                    f"{item['share']:5.1%}  {item['label'] or item['leaf']}"
                )
        below = balanced.get("leaves_below_min_clips", [])
        print(
            f"  clip 数不足 {cfg.data.min_clips_per_leaf} 的细类 : {len(below)} 个"
            + (f"（如 {', '.join(item['leaf'] for item in below[:6])}）" if below else "")
        )

    # --- 来源分布（按 category_l1），主理人要求单独看武术 ---
    by_source: dict[str, list[int]] = {}
    for entry in result.entries:
        slot = by_source.setdefault(entry.category_l1 or "(空)", [0, 0])
        slot[0] += 1
        slot[1] += entry.n_windows
    if by_source:
        total_windows = sum(slot[1] for slot in by_source.values()) or 1
        print("-" * 84)
        print("来源分布（大类 → clip / 窗口 / 窗口占比）：")
        for name, (clips, windows) in sorted(by_source.items(), key=lambda kv: -kv[1][1]):
            print(f"  {clips:5d} clip / {windows:6d} 窗 / {windows / total_windows:5.1%}  {name}")

    if index_path is not None:
        print(f"索引已写出          : {index_path}")
    print("=" * 84)


def _print_wushu_report(result, cfg) -> None:  # type: ignore[no-untyped-def]
    """逐项列出武术类别（根目录名含 ``wushu``）的 clip 数与窗口数。

    主理人明确要求「武术 24 套路逐项 clip 数」——素材里其实是 191 个类别，
    这里全量列出，缺素材的类别也会出现（值为 0），避免「静默丢弃」。

    注意不能按 ``category_l1.startswith("4.")`` 过滤：BVH_Body 也有
    ``4.1.Box_Ing`` / ``4.3.Equipment_Exercises``，会被一起算进武术。
    """
    wushu = [entry for entry in result.entries if "wushu" in entry.category_l1.lower()]
    if not wushu:
        print("武术类别            : 0（未进入训练集！）")
        return
    tally: dict[str, list[int]] = {}
    for entry in wushu:
        slot = tally.setdefault(entry.zh_label or entry.category_l3, [0, 0])
        slot[0] += 1
        slot[1] += entry.n_windows
    print("-" * 84)
    print(f"武术类别逐项（{len(tally)} 个，按 clip 数降序）：")
    for label, (clips, windows) in sorted(tally.items(), key=lambda kv: (-kv[1][0], kv[0])):
        print(f"  {clips:3d} clip / {windows:4d} 窗  {label}")
    total_clips = sum(slot[0] for slot in tally.values())
    total_windows = sum(slot[1] for slot in tally.values())
    all_windows = sum(entry.n_windows for entry in result.entries) or 1
    print(
        f"合计                : {total_clips} clip / {total_windows} 窗"
        f"（占全部窗口 {total_windows / all_windows:.1%}）"
    )
    print("=" * 84)


def main(argv: list[str] | None = None) -> int:
    """命令行入口。

    Args:
        argv: 参数列表（默认取 ``sys.argv[1:]``）。

    Returns:
        进程退出码。
    """
    args = build_argparser().parse_args(argv)
    cfg = load_config(args.config)
    set_seed(cfg.runtime.seed)
    log = get_logger("ttm.scan", script="01_scan")
    log.info("开始扫描: %s", cfg.summary())
    log.info("配置文件: %s", cfg.source_path)

    scanner = Scanner(
        cfg,
        limit=args.limit if args.limit and args.limit > 0 else None,
        workers=args.workers,
        logger=log,
    )
    result = scanner.run()

    index_path = None
    if not args.dry_run:
        index_path, failed_path = scanner.write(result)
        resolved = dump_resolved(cfg, "scan")
        log.info("生效配置已落盘: %s", resolved)
        if result.failed:
            log.warning("有 %d 个文件被淘汰，详见 %s", len(result.failed), failed_path)
    else:
        log.info("--dry-run：不写盘")

    _print_summary(result, cfg, index_path)
    _print_wushu_report(result, cfg)
    close_all()
    return 0 if result.entries else 1


if __name__ == "__main__":
    raise SystemExit(main())
