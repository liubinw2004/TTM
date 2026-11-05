"""T08 脚本：用训练好的 VAE + DiT 生成动作并导出 BVH。

用法::

    . D:/TTM/scripts/env.ps1
    python scripts/05_generate.py --config configs/tier_s.yaml
    python scripts/05_generate.py --config configs/tier_s.yaml \
        --prompt "慢慢走上楼梯" --duration 5 --guidance 3.0 --steps 50
    python scripts/05_generate.py --config configs/smoke.yaml --dry-run   # 只查权重是否就位

默认生成三条固定指令（含一条武术），落盘到 ``deliverables/generated/``。

报告内容（主理人验收项）：

* 每条指令的 **单次推理时延分解**（text / sample / decode / blend / post）；
* 长动作的**接缝根速度台阶**（``seam_jumps``，cm/s，越小越连续）；
* 生成帧数、窗口数、BVH 落盘路径。

退出码：成功 0；torch 缺失 2；权重/数据缺失 3。
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

from ttm.common import constants as C  # noqa: E402
from ttm.common import paths as P  # noqa: E402
from ttm.common.config import load_config  # noqa: E402
from ttm.common.logutil import close_all, get_logger  # noqa: E402

#: 默认演示指令（第三条是武术，用于验证长动作滑窗融合）。
DEFAULT_PROMPTS: tuple[str, ...] = (
    "慢慢走上楼梯",
    "挥手打招呼然后转身走开",
    "打一套陈式太极拳，动作舒缓连贯",
)

#: 时延分解的展示顺序（与 ``GeneratedMotion.timing_ms`` 的键一致）。
TIMING_KEYS: tuple[str, ...] = ("text_ms", "sample_ms", "decode_ms", "blend_ms", "post_ms")


def build_argparser() -> argparse.ArgumentParser:
    """构造命令行参数解析器。"""
    parser = argparse.ArgumentParser(
        description="文本驱动生成动作并导出 BVH（TTM T08）",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument("--config", default="configs/default.yaml", help="配置文件路径")
    parser.add_argument(
        "--prompt",
        action="append",
        default=None,
        help="中文指令，可重复传入；缺省用内置三条（含武术）",
    )
    parser.add_argument("--duration", type=float, default=5.0, help="目标时长（秒）")
    parser.add_argument("--steps", type=int, default=50, help="DDIM 采样步数（η=0，确定性）")
    parser.add_argument(
        "--guidance",
        type=float,
        default=None,
        help="CFG 引导强度（1.0 = 关闭）；缺省回落到配置 runtime.guidance（生产 7.5）",
    )
    parser.add_argument("--smooth", type=float, default=0.2, help="根轨迹 EMA 平滑强度（0 = 关闭）")
    parser.add_argument("--seed", type=int, default=42, help="随机种子（确定性复现）")
    parser.add_argument("--out-dir", default="deliverables/generated", help="BVH 输出目录")
    parser.add_argument(
        "--motion-json",
        action="store_true",
        help="同时导出 ttm-motion/v1 JSON（UE 导入与前端播放都要它；BVH 只是给人看）",
    )
    parser.add_argument(
        "--ground-origin",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="BVH 侧把首帧最低足触地点平移到 0（楼梯类生成从地面开始爬）；JSON 始终保留世界坐标",
    )
    parser.add_argument(
        "--trim-head",
        type=int,
        default=None,
        help="裁掉的启动瞬态帧数（F-17）；缺省回落到配置 runtime.trim_head_frames（12），0 = 关闭",
    )
    parser.add_argument("--device", default="", help="强制设备（如 cpu / cuda）；默认取配置")
    parser.add_argument("--dry-run", action="store_true", help="只检查权重与数据是否就位")
    return parser


def _check_assets(cfg, log) -> list[str]:  # noqa: ANN001 - 内部工具
    """检查生成所需的所有产物是否就位。

    Returns:
        缺失项描述列表（空列表表示全部就位）。
    """
    missing: list[str] = []
    vae_ckpt = Path(cfg.runtime.vae_ckpt) if cfg.runtime.vae_ckpt else cfg.run_dir("vae") / "best.pt"
    dif_ckpt = (
        Path(cfg.runtime.diffusion_ckpt)
        if cfg.runtime.diffusion_ckpt
        else cfg.run_dir("diffusion") / "best.pt"
    )
    stats = cfg.norm_stats_path()
    if not vae_ckpt.is_file():
        missing.append(f"VAE checkpoint 缺失: {vae_ckpt}")
    if not dif_ckpt.is_file():
        missing.append(f"扩散 checkpoint 缺失: {dif_ckpt}")
    if not stats.is_file():
        missing.append(f"norm_stats 缺失: {stats}（先跑 scripts/02_preprocess.py）")
    model_path = Path(cfg.text.model_path) if cfg.text.model_path else None
    if model_path is not None and not model_path.is_dir():
        missing.append(f"bge 权重目录缺失: {model_path}")
    for item in missing:
        log.error("%s", item)
    return missing


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
    log = get_logger("ttm.infer.generate", script="05_generate")

    try:
        import torch  # noqa: F401, PLC0415
    except Exception as exc:  # noqa: BLE001
        log.error("torch 不可用（%s）。请先按 README 安装 cu128 版 torch。", exc)
        close_all()
        return 2

    missing = _check_assets(cfg, log)
    if missing:
        close_all()
        return 3
    if args.dry_run:
        print("权重与数据均已就位（dry-run）。")
        close_all()
        return 0

    from ttm.infer.generate import GenerateOptions, load_generator  # noqa: PLC0415

    generator = load_generator(cfg, logger=log)
    out_dir = P.safe_makedirs(Path(args.out_dir))
    # guidance / trim_head 都从配置回落——命令行不再写死常量，避免与配置静默漂移
    guidance = (
        float(args.guidance) if args.guidance is not None else float(cfg.runtime.guidance)
    )
    trim_head = (
        int(args.trim_head)
        if args.trim_head is not None
        else int(cfg.runtime.trim_head_frames)
    )
    log.info(
        "生成参数: duration=%.1fs steps=%d guidance=%.2f smooth=%.2f seed=%d trim_head=%d → %s",
        args.duration,
        args.steps,
        guidance,
        args.smooth,
        args.seed,
        trim_head,
        out_dir,
    )

    prompts = tuple(args.prompt) if args.prompt else DEFAULT_PROMPTS
    records: list[dict] = []
    for index, prompt in enumerate(prompts):
        options = GenerateOptions(
            duration_s=args.duration,
            steps=args.steps,
            guidance=guidance,
            smooth=args.smooth,
            seed=args.seed + index,
            ground_origin=bool(args.ground_origin),
            trim_head_frames=trim_head,
            # 由本脚本统一命名导出（要带序号前缀），因此关掉 generate() 的自动落盘
            export_bvh=False,
            out_dir=out_dir.as_posix(),
        )
        motion = generator.generate(prompt, options)
        stem = f"{index:02d}_{_safe_name(prompt)}"
        path = generator.export(motion, out_dir, name=stem, ground_origin=bool(args.ground_origin))
        record = motion.to_dict()
        record["bvh_path"] = path.as_posix()
        if args.motion_json:
            # 与服务端 ``/generate?export=json`` 走同一条重定向链路，避免
            # 「命令行导出的 BVH」和「服务端导出的 JSON」在两套骨名之间漂移。
            from ttm.export.retarget import (  # noqa: PLC0415 - 重定向依赖较重，用到才导
                build_plan,
                retarget_motion,
                write_motion_json,
            )

            plan = build_plan(fold_spine3=cfg.retarget.spine_merge != "split")
            rotations, positions = retarget_motion(motion.quats, motion.root_pos, plan=plan)
            json_path = write_motion_json(
                out_dir / f"{stem}.json",
                rotations,
                positions,
                plan=plan,
                fps=C.FPS,
                prompt=prompt,
                extra={
                    "n_windows": int(motion.n_windows),
                    "seam_jumps": [float(item) for item in motion.seam_jumps],
                    # JSON 保留原始世界坐标；offset 供预览/UE 做地面锚定
                    "ground_origin": {
                        "applied": bool(args.ground_origin),
                        "offset_cm": round(float(motion.ground_offset_cm), 3),
                    },
                },
            )
            record["motion_json_path"] = json_path.as_posix()
        records.append(record)
        log.info("已导出: %s%s", path, "（+ Motion JSON）" if args.motion_json else "")

    print("=" * 96)
    print("TTM 文本驱动生成报告")
    print("=" * 96)
    header = f"{'指令':<28} {'帧数':>6} {'窗数':>5} {'接缝':>8} {'总时延':>9}  时延分解 (ms)"
    print(header)
    print("-" * 96)
    for record in records:
        timing = record.get("timing_ms", {})
        breakdown = " / ".join(f"{key[:-3]}={timing.get(key, 0.0):.1f}" for key in TIMING_KEYS)
        jumps = record.get("seam_jumps", [])
        seam_value = record.get("seam_jump_max_cmps")
        if seam_value is None and jumps:
            seam_value = max(jumps)
        seam_text = f"{float(seam_value):.1f}" if seam_value else "-"
        print(
            f"{record['prompt']:<28} {record['n_frames']:>6} {record['n_windows']:>5} "
            f"{seam_text:>8} {timing.get('total_ms', 0.0):>7.0f}ms  {breakdown}"
        )
    print("-" * 96)
    total_ms = sum(item.get("timing_ms", {}).get("total_ms", 0.0) for item in records)
    print(f"合计 {len(records)} 条，总时延 {total_ms:.0f} ms，平均 {total_ms / len(records):.0f} ms/条")
    seam_fail = [item for item in records if not bool(item.get("seam_ok", True))]
    threshold = float(records[0].get("seam_jump_threshold_cmps", 100.0)) if records else 100.0
    if seam_fail:
        print(
            f"接缝连续性: **FAIL** — {len(seam_fail)}/{len(records)} 条超过 {threshold:.0f} cm/s："
            + "、".join(
                f"{item['prompt']}({item.get('seam_jump_max_cmps', 0.0):.1f})" for item in seam_fail
            )
        )
    else:
        print(f"接缝连续性: PASS — 全部接缝 < {threshold:.0f} cm/s")
    print("=" * 96)

    report = P.resolve_out(out_dir / "generate_report.json", mkparents=True)
    report.write_text(
        json.dumps(
            {
                "config": args.config,
                "duration_s": args.duration,
                "steps": args.steps,
                "guidance": guidance,
                "trim_head_frames": trim_head,
                "smooth": args.smooth,
                "seed": args.seed,
                "motions": records,
            },
            ensure_ascii=False,
            indent=2,
        ),
        encoding="utf-8",
    )
    print(f"报告: {report.as_posix()}")
    close_all()
    # 退出码 4 = 生成完成但接缝不连续。**不静默返回 0**：接缝跳变是「用户能一眼
    # 看出人物瞬移」的硬伤，验收时必须能被脚本化检测到。
    return 4 if seam_fail else 0


def _safe_name(prompt: str, *, limit: int = 20) -> str:
    """把中文指令转成安全文件名（保留中文与数字）。"""
    cleaned = "".join(ch if ch.isalnum() else "_" for ch in str(prompt).strip())
    collapsed = "_".join(part for part in cleaned.split("_") if part)
    return (collapsed or "motion")[:limit]


if __name__ == "__main__":
    raise SystemExit(main())
