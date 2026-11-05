"""T07 脚本：训练潜空间扩散（DiT）→ ``runs/tier_s/diffusion/best.pt``。

用法::

    . D:/TTM/scripts/env.ps1
    python scripts/04_train_diffusion.py --config configs/smoke.yaml --max-steps 5   # 冒烟
    python scripts/04_train_diffusion.py --config configs/tier_s.yaml               # 正式

前置：``scripts/02_preprocess.py`` 已生成分片 + ``labels.jsonl``，
``scripts/03_train_vae.py`` 已产出 ``runs/tier_s/vae/best.pt``。

关键约束（ARCH §6 T07）：

* VAE 与 bge **全部冻结**，只有 ``TextProjector`` 与 DiT 参与训练；
* 训练期条件 dropout（``cfg.diffusion.cfg_dropout``，默认 0.1）把条件置零，
  这是推理期 ``guidance > 1`` 能生效的前提；
* 文本嵌入一次性缓存（distinct 文本数量有限），每步只跑可训练投影层。

产物（``runs/tier_s/diffusion/``）::

    last.pt / best.pt  checkpoint（含 projector / optimizer / vae_ckpt / text_model）
    loss.jsonl         每 step 一行 {step, epoch, loss}
    summary.json       训练摘要（含峰值显存）
    seam_gate.json     接缝连续性门禁报告（--verify-seam，默认 auto）
    config_resolved.yaml

退出码：成功 0；torch 缺失 2；数据/权重缺失 3；训练异常 1；
**训练完成但接缝门禁不通过 5**（与 ``05_generate.py`` 的 4 区分开：4 是「生成脚本
自己发现接缝不连续」，5 是「训练收尾门禁判不通过」，两者都可能出现，不该混成
同一个码）。

接缝门禁为什么放在这里：滑窗拼接（overlap 30 帧交叉淡化 + 根位移只积分一次）
是否成立，在**训练产物**上就是可以程序化判定的。跑完训练立刻自己验一遍，比
「跑完生成脚本再看打印」少一个环节、少一轮往返。
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

# --- sys.path 引导 -----------------------------------------------------------
_PROJECT_ROOT = Path(__file__).resolve().parents[1]
_SRC_DIR = _PROJECT_ROOT / "src"
if str(_SRC_DIR) not in sys.path:
    sys.path.insert(0, str(_SRC_DIR))

from ttm.common import paths as P  # noqa: E402
from ttm.common.config import dump_resolved, load_config  # noqa: E402
from ttm.common.logutil import close_all, get_logger  # noqa: E402


def build_argparser() -> argparse.ArgumentParser:
    """构造命令行参数解析器。"""
    parser = argparse.ArgumentParser(
        description="训练潜空间扩散 Transformer（TTM T07）",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument("--config", default="configs/default.yaml", help="配置文件路径")
    parser.add_argument(
        "--epochs", type=int, default=0, help="覆盖配置里的 diffusion.epochs（0 = 用配置）"
    )
    parser.add_argument(
        "--max-steps", type=int, default=0, help="每 epoch 最多多少步（0 = 不限制）；冒烟调试用"
    )
    parser.add_argument("--resume", default="", help="从该 checkpoint 续训")
    parser.add_argument(
        "--device", default="", help="强制设备（如 cpu / cuda）；默认取配置"
    )
    parser.add_argument(
        "--max-minutes",
        type=float,
        default=0.0,
        help="墙钟预算（分钟，0 = 不限制）；每 epoch 检查一次，超预算就停并保留 best.pt",
    )
    parser.add_argument(
        "--verify-seam",
        choices=("auto", "on", "off"),
        default="auto",
        help=(
            "训练结束后跑接缝连续性门禁。auto = 冒烟（--max-steps > 0）跳过、"
            "正式训练必跑；on = 强制跑；off = 强制跳过"
        ),
    )
    parser.add_argument(
        "--seam-duration",
        type=float,
        default=8.0,
        help="接缝门禁每条指令的时长（秒）；8 秒 = 2 窗口 = 1 个真实接缝，再短就没接缝可验",
    )
    return parser


def _should_verify_seam(mode: str, max_steps: int) -> bool:
    """把 ``--verify-seam`` 的三态解析成布尔。

    ``auto`` 的规则：``--max-steps > 0`` 视为冒烟（模型只训了几十步，接缝必然
    不连续，跑门禁只会产出噪声结论），跳过；否则视为正式训练，必跑。

    Args:
        mode: ``auto`` / ``on`` / ``off``。
        max_steps: ``--max-steps`` 的值（0 = 不限制）。

    Returns:
        是否跑门禁。
    """
    if mode == "on":
        return True
    if mode == "off":
        return False
    return int(max_steps) <= 0


def _print_seam_gate(report) -> None:  # noqa: ANN001 - 仅内部展示用
    """打印接缝门禁结果（可直接贴进汇报）。"""
    print("-" * 84)
    print(f"接缝连续性门禁（阈值 {report.threshold_cmps:.0f} cm/s / 时长 {report.duration_s:.1f}s）")
    print(f"  {'指令':<24}{'窗口':>6}{'帧':>7}{'接缝':>6}{'最大跳变':>12}{'均值':>10}  判定")
    for probe in report.probes:
        if probe.error:
            print(f"  {probe.prompt:<24}{'-':>6}{'-':>7}{'-':>6}{'-':>12}{'-':>10}  ERROR")
            continue
        verdict = "PASS" if probe.ok else "FAIL"
        print(
            f"  {probe.prompt:<24}{probe.n_windows:>6}{probe.n_frames:>7}"
            f"{probe.n_seams:>6}{probe.seam_max_cmps:>11.1f}{probe.seam_mean_cmps:>10.1f}"
            f"  {verdict}"
        )
    if report.passed:
        worst = report.worst
        print(
            f"  接缝连续性: PASS — 最差 {worst.seam_max_cmps:.1f} cm/s"
            f" ≤ {report.threshold_cmps:.0f} cm/s（{len(report.probes)} 条指令 / "
            f"{sum(p.n_seams for p in report.probes)} 个接缝）"
        )
    else:
        print("  接缝连续性: FAIL")
        for reason in report.failures():
            print(f"    - {reason}")
    print("=" * 84)


def _print_summary(result, logger) -> None:  # noqa: ANN001 - 仅内部展示用
    """打印训练摘要（可直接贴进汇报）。"""
    history = result.history
    train = history.train_loss
    val = history.val_loss
    print("=" * 84)
    print("TTM 扩散（DiT）训练摘要")
    print("=" * 84)
    print(f"参数量（可训练）      : {result.param_count:,}")
    print(f"完成 epoch            : {len(val)}")
    print(f"总步数                : {history.steps[-1] if history.steps else 0}")
    print(f"best val loss         : {result.best_val:.6f}")
    print(f"loss 下降比例         : {history.loss_drop_ratio() * 100:.1f}%")
    if train:
        print(f"train loss 首 / 末    : {train[0]:.6f} → {train[-1]:.6f}")
    if val:
        print(f"val   loss 首 / 末    : {val[0]:.6f} → {val[-1]:.6f}")
    print(f"峰值显存              : {history.peak_memory_mb:.0f} MB")
    print(f"耗时                  : {history.elapsed_s:.1f}s")
    print(f"最优 checkpoint       : {result.ckpt_path}")
    print("-" * 84)
    print("loss 曲线（每 epoch 末的 val / train）:")
    for epoch, (train_loss, val_loss) in enumerate(zip(train, val)):
        bar = "#" * max(1, int(round(val_loss * 40))) if val_loss else ""
        print(f"  epoch {epoch:03d}  train={train_loss:.6f}  val={val_loss:.6f}  {bar}")
    print("=" * 84)
    logger.info("训练摘要已输出到 stdout")


def main(argv: list[str] | None = None) -> int:
    """命令行入口。

    Args:
        argv: 参数列表（默认取 ``sys.argv[1:]``）。

    Returns:
        进程退出码。
    """
    args = build_argparser().parse_args(argv)
    cfg = load_config(args.config)
    if args.device:
        cfg.runtime.device = args.device
    log = get_logger("ttm.train.diffusion", script="04_train_diffusion")

    try:
        import torch  # noqa: F401, PLC0415
    except Exception as exc:  # noqa: BLE001
        log.error("torch 不可用（%s）。请先按 README 安装 cu128 版 torch。", exc)
        close_all()
        return 2

    shards_dir = cfg.shards_path()
    if not shards_dir.is_dir():
        log.error("分片目录不存在: %s。请先运行 scripts/02_preprocess.py", shards_dir)
        close_all()
        return 3
    if not cfg.labels_path().is_file():
        log.error("标签文件不存在: %s。请先运行 scripts/02_preprocess.py", cfg.labels_path())
        close_all()
        return 3

    vae_ckpt = Path(cfg.runtime.vae_ckpt) if cfg.runtime.vae_ckpt else cfg.run_dir("vae") / "best.pt"
    if not vae_ckpt.is_file():
        log.error("VAE checkpoint 不存在: %s。请先运行 scripts/03_train_vae.py", vae_ckpt)
        close_all()
        return 3

    from ttm.train.train_diffusion import train_diffusion  # noqa: PLC0415 - torch 就绪后再导入

    log.info("配置: %s", cfg.summary())
    dump_resolved(cfg, "diffusion")
    try:
        result = train_diffusion(
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
    except RuntimeError as exc:  # 显存不足等
        # 必须带堆栈：这类错误（device/generator 不匹配、index 跨设备）只看一句话
        # 根本定位不了，而重跑一次要 30 秒加载 + 60 分钟训练。
        log.error("训练中断（RuntimeError）: %s", exc, exc_info=True)
        close_all()
        return 1

    _print_summary(result, log)

    summary_path = P.resolve_out(
        Path(result.run_dir) / "summary.json", mkparents=True
    )
    summary_path.write_text(
        json.dumps(result.to_dict(), ensure_ascii=False, indent=2), encoding="utf-8"
    )
    log.info("训练摘要已落盘: %s", summary_path)
    print(f"摘要文件: {summary_path.as_posix()}")

    # --- 训练后接缝门禁（主理人 M3 硬性要求：滑窗 overlap 30 帧融合后，接缝处
    #     的根速度必须连续）。放在训练脚本收尾而不是只放生成脚本，是为了让
    #     「训练产物能不能直接交付」这件事由程序判定，而不是靠人看表格。
    if not _should_verify_seam(args.verify_seam, args.max_steps):
        log.info("接缝门禁已跳过（--verify-seam=%s）", args.verify_seam)
        close_all()
        return 0

    from ttm.infer.seam_gate import verify_seams  # noqa: PLC0415 - torch 已就绪

    try:
        seam_report = verify_seams(
            cfg,
            duration_s=float(args.seam_duration),
            logger=log,
            write_to=Path(result.run_dir) / "seam_gate.json",
        )
    except FileNotFoundError as exc:
        log.error("接缝门禁无法运行: %s", exc)
        close_all()
        return 3
    except RuntimeError as exc:  # 显存不足等
        log.error("接缝门禁中断（RuntimeError）: %s", exc, exc_info=True)
        close_all()
        return 1

    _print_seam_gate(seam_report)
    close_all()
    # 训练本身是成功的，所以 0 / 5 要能区分开：拿到 5 说明 ckpt 能用但接缝不连续，
    # 不是「训练失败」，CI 与人工都不该把它当同一种情况处理。
    return 0 if seam_report.passed else 5


if __name__ == "__main__":
    raise SystemExit(main())
