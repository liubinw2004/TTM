"""T05 脚本：训练 MotionVAE（重构 L1 + KL）→ ``runs/vae/best.pt``。

用法::

    . D:/TTM/scripts/env.ps1
    python scripts/03_train_vae.py --config configs/smoke.yaml                    # 10 epoch 冒烟
    python scripts/03_train_vae.py --config configs/tier_s.yaml --resume runs/vae/ckpt_0040.pt

前置：``scripts/02_preprocess.py`` 已生成分片与 ``labels.jsonl``。

产物（``runs/vae/``）::

    ckpt_XXXX.pt       每 epoch 一个
    best.pt            val loss 最优
    loss.jsonl         每 step 一行 {step, epoch, loss, recon, kl}
    config_resolved.yaml

退出码：训练成功为 0；torch 缺失为 2；数据缺失为 3。
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


def _labels_summary(path: Path) -> tuple[int, int]:
    """统计标签文件的行数与 distinct 条件数。

    训练到底读的是 ``labels.jsonl``（350 个细标签）还是 ``labels_merged.jsonl``
    （179 个合并后条件）——这一条搞反了整条链路白跑，所以要在摘要里显式打出来，
    不靠「我记得配好了」。

    Args:
        path: 标签文件路径。

    Returns:
        ``(行数, distinct 条件数)``；文件不存在时返回 ``(0, 0)``。
    """
    import orjson  # noqa: PLC0415 - 只在本函数用到

    if not path.is_file():
        return 0, 0
    rows = 0
    labels: set[str] = set()
    with path.open("rb") as handle:
        for line in handle:
            line = line.strip()
            if not line:
                continue
            record = orjson.loads(line)
            rows += 1
            labels.add(str(record.get("zh_label", "")))
    return rows, len(labels)


def build_argparser() -> argparse.ArgumentParser:
    """构造命令行参数解析器。"""
    parser = argparse.ArgumentParser(
        description="训练 MotionVAE（TTM T05）",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument("--config", default="configs/default.yaml", help="配置文件路径")
    parser.add_argument("--epochs", type=int, default=0, help="覆盖配置里的 vae.epochs（0 = 用配置）")
    parser.add_argument(
        "--max-steps", type=int, default=0, help="每 epoch 最多多少步（0 = 不限制）；冒烟调试用"
    )
    parser.add_argument("--resume", default="", help="从该 checkpoint 续训")
    parser.add_argument(
        "--max-minutes",
        type=float,
        default=0.0,
        help="墙钟预算（分钟，0 = 不限制）；每 epoch 检查一次，超预算就停并保留 best.pt",
    )
    return parser


def main(argv: list[str] | None = None) -> int:
    """命令行入口。

    Args:
        argv: 参数列表（默认取 ``sys.argv[1:]``）。

    Returns:
        进程退出码。
    """
    args = build_argparser().parse_args(argv)
    cfg = load_config(args.config)
    log = get_logger("ttm.train.vae", script="03_train_vae")

    try:
        import torch  # noqa: F401, PLC0415
    except Exception as exc:  # noqa: BLE001
        log.error("torch 不可用（%s）。请先按 README 安装 cu128 版 torch。", exc)
        close_all()
        return 2

    from ttm.train.train_vae import train_vae  # noqa: PLC0415 - torch 就绪后再导入

    shards_dir = cfg.shards_path()
    if not shards_dir.is_dir():
        log.error("分片目录不存在: %s。请先运行 scripts/02_preprocess.py", shards_dir)
        close_all()
        return 3

    log.info("配置: %s", cfg.summary())
    dump_resolved(cfg, "vae")
    try:
        result = train_vae(
            cfg,
            max_steps_per_epoch=args.max_steps if args.max_steps > 0 else None,
            max_epochs=args.epochs if args.epochs > 0 else None,
            resume=args.resume or None,
            logger=log,
            max_minutes=args.max_minutes if args.max_minutes > 0 else None,
        )
    except (FileNotFoundError, ValueError) as exc:
        log.error("%s", exc)
        close_all()
        return 3

    print("=" * 84)
    print("TTM VAE 训练摘要")
    print("=" * 84)
    labels_path = cfg.train_labels_path()
    label_rows, label_conditions = _labels_summary(labels_path)
    print(
        f"训练标签            : {labels_path.as_posix()}（{label_rows} 行 / "
        f"{label_conditions} 个条件）"
    )
    print(f"参数量              : {result.params:,}")
    print(f"完成 epoch          : {result.epochs_done}")
    print(f"best val loss       : {result.best_val_loss:.6f}")
    print(f"loss 下降比例       : {result.history.loss_drop_ratio() * 100:.1f}%")
    print(f"关节角 MAE          : {result.metrics.get('joint_mae_deg', float('nan')):.2f}°（验收 ≤ 8°）")
    print(f"根高度 MAE          : {result.metrics.get('root_height_mae_cm', float('nan')):.2f} cm（验收 ≤ 3cm）")
    print(f"根速度 MAE          : {result.metrics.get('root_speed_mae_cmps', float('nan')):.2f} cm/s")
    print(f"峰值显存            : {result.peak_memory_mb:.0f} MB")
    # 后验健康度：joint_mae 好看但后验塌缩的模型对扩散训练毫无用处，必须单独看。
    print(
        "后验健康度          : "
        f"mu_std={result.metrics.get('mu_std', float('nan')):.4f}"
        f" / logvar_mean={result.metrics.get('logvar_mean', float('nan')):.3f}"
        f" / 帧间方差比={result.metrics.get('var_ratio', float('nan')):.2f}"
    )
    if result.metrics.get("mu_std", 1.0) < 0.01:
        print("  !! 警告：后验坍缩（mu_std < 0.01），潜空间对扩散模型不可用")
    # 武术采样占比：主理人要求给出「数据占比 → 采样期望占比」的实测对照
    if "sampling_data_share" in result.metrics:
        print(
            "武术采样占比        : "
            f"数据 {result.metrics['sampling_data_share'] * 100:.2f}% → "
            f"power 后 {result.metrics['sampling_after_power'] * 100:.2f}% → "
            f"组倍率 ×{result.metrics['sampling_multiplier']:.3f} 后 "
            f"{result.metrics['sampling_after_multiplier'] * 100:.2f}%"
        )
    print(f"最优 checkpoint     : {result.checkpoint}")
    print(f"耗时                : {result.elapsed_s:.1f}s")
    print("=" * 84)
    close_all()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
