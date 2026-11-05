"""T04 脚本：全量预处理 → 146 维特征分片 + 中文标签 + 归一化统计。

用法::

    . D:/TTM/scripts/env.ps1
    python scripts/02_preprocess.py --config configs/tier_s.yaml             # 全量（Tier-S）
    python scripts/02_preprocess.py --config configs/smoke.yaml --limit 20   # 小样本调试

前置：``python scripts/01_scan.py --config ...`` 已生成 ``index.jsonl``。

产物（``data/tier_{tier}/``）::

    shards/shard_XXXXX.npz   feat[n,150,146] float32 / mask[n,150] uint8 /
                             clip_id[n]<U80 / win_idx[n]int16 /
                             layout_version / fps        （ARCH §5.5，字段不得增删）
    norm_stats.npz           mean[146] / std[146] / layout_version
    labels.jsonl             每窗口一行：zh_label + zh_texts(8-15 条)
    failed.jsonl             坏文件追加（stage=preprocess）

退出码：失败率 ≤ 1% 且产出至少一个窗口时为 0，否则为 1。
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

# --- sys.path 引导 -----------------------------------------------------------
_PROJECT_ROOT = Path(__file__).resolve().parents[1]
_SRC_DIR = _PROJECT_ROOT / "src"
if str(_SRC_DIR) not in sys.path:
    sys.path.insert(0, str(_SRC_DIR))

from ttm.common.config import dump_resolved, load_config  # noqa: E402
from ttm.common.logutil import close_all, get_logger  # noqa: E402
from ttm.common.seed import set_seed  # noqa: E402
from ttm.data.preprocess import PreprocessPipeline  # noqa: E402


def build_argparser() -> argparse.ArgumentParser:
    """构造命令行参数解析器。"""
    parser = argparse.ArgumentParser(
        description="预处理 BVH → 146 维特征分片（TTM T04）",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument(
        "--config", default="configs/default.yaml", help="配置文件路径（相对项目根）"
    )
    parser.add_argument(
        "--limit", type=int, default=0, help="最多处理多少个 clip（0 = 全部）；调试建议 20-200"
    )
    parser.add_argument("--workers", type=int, default=8, help="并发线程数（Z 盘 RTT 摊薄）")
    return parser


def _print_summary(result, cfg) -> None:  # type: ignore[no-untyped-def]
    """打印预处理摘要。"""
    stats = result.stats
    print("=" * 84)
    print(f"TTM 预处理摘要 · tier={cfg.data.tier} · 用时 {result.elapsed_s:.1f}s")
    print("=" * 84)
    print(f"入训 clip           : {result.n_clips}")
    print(f"训练窗口            : {result.n_windows}")
    print(f"npz 分片            : {len(result.shards)} 个（shard_size={cfg.data.shard_size}）")
    print(f"跳过（无窗口）      : {result.n_skipped}")
    print(
        f"失败                : {result.n_failed}（失败率 {result.failure_rate * 100:.2f}%，"
        f"上限 {result.max_failure_rate * 100:.0f}%）"
    )
    if result.norm_stats is not None:
        print("归一化统计（z-score 四维）：")
        for dim in range(len(result.norm_stats.mean)):
            if dim in (0, 1, 2, 5):
                names = {0: "根前向速度 cm/s", 1: "根侧向速度 cm/s", 2: "根高度 cm", 5: "偏航角速度 rad/s"}
                print(
                    f"  [{dim:3d}] {names[dim]:<18} mean={result.norm_stats.mean[dim]:10.4f} "
                    f"std={result.norm_stats.std[dim]:9.4f}"
                )
        print(f"  参与统计帧数      : {result.norm_stats.count}")
        print(f"  layout_version    : {result.norm_stats.layout_version}")
    print(f"标签文件            : {stats.get('labels')}")
    print(f"失败日志            : {stats.get('failed')}")
    print(f"统计量文件          : {stats.get('norm_stats')}")
    print(f"分片目录            : {cfg.shards_path()}")
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
    log = get_logger("ttm.preprocess", script="02_preprocess")
    log.info("开始预处理: %s", cfg.summary())
    log.info("配置文件: %s", cfg.source_path)

    pipeline = PreprocessPipeline(
        cfg,
        limit=args.limit if args.limit and args.limit > 0 else None,
        workers=args.workers,
        logger=log,
    )
    try:
        result = pipeline.run()
    except FileNotFoundError as exc:
        log.error("%s", exc)
        close_all()
        return 2

    resolved = dump_resolved(cfg, "preprocess")
    log.info("生效配置已落盘: %s", resolved)
    _print_summary(result, cfg)

    if not result.is_acceptable():
        log.error(
            "失败率 %.2f%% 超过上限 %.0f%%，按设计整批退出非零（ARCH §5.1）",
            result.failure_rate * 100,
            result.max_failure_rate * 100,
        )
        close_all()
        return 1
    close_all()
    return 0 if result.n_windows > 0 else 1


if __name__ == "__main__":
    raise SystemExit(main())
