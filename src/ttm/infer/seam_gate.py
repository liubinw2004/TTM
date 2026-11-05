"""训练后的接缝连续性门禁：**训练完立刻自己验一遍**，不等人工回报。

为什么要有这个模块（而不是把检查只放在 ``scripts/05_generate.py``）：

* 滑窗拼接（overlap 30 帧 + 交叉淡化 + 根位移只积分一次）的逻辑在**训练产物**
  上是否成立，是一个可以程序化判定的问题。等跑完生成脚本再看打印，等于多绕一
  个环节、多一轮往返；
* ``05_generate.py`` 面向「人看」，它的输出是表格；门禁面向「程序判」，它的输出
  是 ``passed`` 布尔与 JSON。两者共用同一个阈值
  :data:`ttm.infer.generate.SEAM_JUMP_MAX_CMPS`，口径不会分叉。

**必须至少有一个真实接缝**：8 秒（240 帧）= 2 个窗口 = 1 个接缝；5 秒只有 1 个
窗口、0 个接缝，此时 ``max(jumps) = 0`` 会「通过」，但那是一个**空洞的通过**
（没有接缝可跳）。所以本模块显式断言 ``n_seams >= 1``，把「没测到」判成失败，
而不是判成通过。
"""

from __future__ import annotations

import json
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Sequence

from ttm.common import constants as C
from ttm.common import paths as P
from ttm.common.config import TTMConfig
from ttm.common.logutil import get_logger
from ttm.infer.generate import (
    SEAM_JUMP_MAX_CMPS,
    GenerateOptions,
    MotionGenerator,
    load_generator,
)

__all__ = [
    "DEFAULT_SEAM_DURATION_S",
    "SEAM_GATE_PROMPTS",
    "SeamGateReport",
    "SeamProbe",
    "verify_seams",
]

#: 门禁默认探测时长（秒）。
#:
#: 8 秒 = 240 帧 → 2 个窗口 → **恰好 1 个接缝**。这是能验到接缝的最短时长，
#: 再短就只有 1 个窗口、没有接缝可验（退化成空洞通过）。
DEFAULT_SEAM_DURATION_S: float = 8.0

#: 门禁默认探测指令：覆盖「位移型」「长动作型」「精细操作型」三类。
#:
#: 第三条带武术——M2' 的逐类评测里 ``4.wushu`` 是最差的类（root 5.98cm /
#: joint 13.20°，明显差于全局 1.87cm / 5.90°），如果接缝会出问题，最可能
#: 先在它身上出问题。
SEAM_GATE_PROMPTS: tuple[str, ...] = (
    "慢慢走上楼梯",
    "打一套长拳第七段",
    "拿起桌上的杯子放到右边",
)


@dataclass
class SeamProbe:
    """单个指令的探测结果。"""

    prompt: str
    n_frames: int = 0
    n_windows: int = 0
    n_seams: int = 0
    seam_max_cmps: float = 0.0
    seam_mean_cmps: float = 0.0
    total_ms: float = 0.0
    ok: bool = False
    error: str = ""

    def to_dict(self) -> dict[str, Any]:
        """序列化。"""
        return {
            "prompt": self.prompt,
            "n_frames": int(self.n_frames),
            "n_windows": int(self.n_windows),
            "n_seams": int(self.n_seams),
            "seam_max_cmps": round(float(self.seam_max_cmps), 3),
            "seam_mean_cmps": round(float(self.seam_mean_cmps), 3),
            "total_ms": round(float(self.total_ms), 1),
            "ok": bool(self.ok),
            "error": self.error,
        }


@dataclass
class SeamGateReport:
    """整批探测的汇总。"""

    threshold_cmps: float = SEAM_JUMP_MAX_CMPS
    duration_s: float = DEFAULT_SEAM_DURATION_S
    steps: int = 0
    guidance: float = 0.0
    seed: int = 42
    probes: list[SeamProbe] = field(default_factory=list)

    @property
    def passed(self) -> bool:
        """全部门禁条件是否满足。

        三个条件缺一不可：

        1. 至少探测成功一条指令（全炸了不能算通过）；
        2. **每一条**指令都至少有 1 个真实接缝（``n_seams >= 1``）；
        3. 全部接缝 ``max <= threshold``。
        """
        if not self.probes:
            return False
        for probe in self.probes:
            if probe.error:
                return False
            if probe.n_seams < 1:
                return False
            if probe.seam_max_cmps > self.threshold_cmps:
                return False
        return True

    @property
    def worst(self) -> SeamProbe | None:
        """接缝跳变最大的那条（没有探测结果时为 ``None``）。"""
        if not self.probes:
            return None
        return max(self.probes, key=lambda item: item.seam_max_cmps)

    def failures(self) -> list[str]:
        """不通过的条目及原因（全通过时为空列表）。"""
        reasons: list[str] = []
        for probe in self.probes:
            if probe.error:
                reasons.append(f"{probe.prompt}: 生成失败（{probe.error}）")
            elif probe.n_seams < 1:
                reasons.append(
                    f"{probe.prompt}: 只生成 {probe.n_windows} 个窗口、"
                    f"{probe.n_frames} 帧，**没有接缝可验**（空洞通过不算通过）"
                )
            elif probe.seam_max_cmps > self.threshold_cmps:
                reasons.append(
                    f"{probe.prompt}: 接缝跳变 {probe.seam_max_cmps:.1f} cm/s "
                    f"> {self.threshold_cmps:.1f} cm/s"
                )
        if not self.probes:
            reasons.append("一条指令都没探测成功")
        return reasons

    def to_dict(self) -> dict[str, Any]:
        """序列化（写 ``seam_gate.json`` 用）。"""
        worst = self.worst
        return {
            "passed": bool(self.passed),
            "threshold_cmps": float(self.threshold_cmps),
            "duration_s": float(self.duration_s),
            "steps": int(self.steps),
            "guidance": round(float(self.guidance), 3),
            "seed": int(self.seed),
            "worst": worst.to_dict() if worst is not None else None,
            "failures": self.failures(),
            "probes": [probe.to_dict() for probe in self.probes],
        }


def _probe_one(
    generator: MotionGenerator,
    prompt: str,
    *,
    duration_s: float,
    steps: int,
    guidance: float,
    seed: int,
) -> SeamProbe:
    """探测单条指令（异常转成 ``error`` 字段，不向上抛）。

    Args:
        generator: 已就绪的生成器。
        prompt: 中文指令。
        duration_s: 时长（秒）。
        steps: DDIM 步数。
        guidance: CFG 强度。
        seed: 随机种子。

    Returns:
        :class:`SeamProbe`。**不写 BVH**——门禁是判定，不是交付物；BVH 由
        ``scripts/05_generate.py`` 负责，两者的落盘职责分开。
    """
    probe = SeamProbe(prompt=str(prompt))
    try:
        options = GenerateOptions(
            duration_s=float(duration_s),
            steps=int(steps),
            guidance=float(guidance),
            seed=int(seed),
            export_bvh=False,
            out_dir=None,
        )
        motion = generator.generate(prompt, options)
    except Exception as exc:  # noqa: BLE001 - 门禁要出报告，不能让单条炸掉整批
        probe.error = f"{type(exc).__name__}: {exc}"
        return probe

    jumps = motion.seam_jumps
    probe.n_frames = int(motion.n_frames)
    probe.n_windows = int(motion.n_windows)
    probe.n_seams = int(jumps.size)
    probe.seam_max_cmps = float(jumps.max()) if jumps.size else 0.0
    probe.seam_mean_cmps = float(jumps.mean()) if jumps.size else 0.0
    probe.total_ms = float(motion.timing_ms.get("total_ms", 0.0))
    probe.ok = probe.n_seams >= 1 and probe.seam_max_cmps <= SEAM_JUMP_MAX_CMPS
    return probe


def verify_seams(
    cfg: TTMConfig,
    *,
    prompts: Sequence[str] | None = None,
    duration_s: float = DEFAULT_SEAM_DURATION_S,
    steps: int | None = None,
    guidance: float | None = None,
    seed: int | None = None,
    generator: MotionGenerator | None = None,
    logger: Any | None = None,
    write_to: str | Path | None = None,
) -> SeamGateReport:
    """训练后跑一遍接缝连续性门禁。

    Args:
        cfg: 生效配置（``prompts`` 为 ``None`` 时用 :data:`SEAM_GATE_PROMPTS`）。
        prompts: 探测指令列表。
        duration_s: 每条指令的时长（默认 8 秒 = 2 窗口 = 1 接缝）。
        steps: DDIM 步数（``None`` → ``cfg.diffusion.ddim_steps``）。
        guidance: CFG 强度（``None`` → ``cfg.runtime.guidance``）。
        seed: 随机种子（``None`` → ``cfg.runtime.seed``）。
        generator: 复用已有的生成器（测试注入用）；``None`` 时按配置加载
            ``runs/<tier>/diffusion/best.pt``。
        logger: 可注入 logger。
        write_to: 若给出，把报告 JSON 写到这里（经写盘守卫）。

    Returns:
        :class:`SeamGateReport`。

    Raises:
        FileNotFoundError: ckpt / norm_stats 缺失（加载阶段，直接上抛）。
    """
    log = logger or get_logger("ttm.infer.seam_gate", script="04_train_diffusion")
    resolved_steps = int(steps if steps is not None else cfg.diffusion.ddim_steps)
    resolved_guidance = float(guidance if guidance is not None else cfg.runtime.guidance)
    resolved_seed = int(seed if seed is not None else cfg.runtime.seed)
    probes = list(prompts) if prompts else list(SEAM_GATE_PROMPTS)

    report = SeamGateReport(
        threshold_cmps=SEAM_JUMP_MAX_CMPS,
        duration_s=float(duration_s),
        steps=resolved_steps,
        guidance=resolved_guidance,
        seed=resolved_seed,
    )

    # 时长必须能撑出至少 2 个窗口，否则接缝数恒为 0、门禁变成空洞通过。
    window_s = float(C.WINDOW) / float(C.FPS)
    stride_s = float(C.WINDOW - C.OVERLAP) / float(C.FPS)
    min_duration_s = window_s + stride_s
    if float(duration_s) < min_duration_s - 1e-9:
        log.warning(
            "接缝门禁时长 %.2fs 撑不出 2 个窗口（至少需 %.2fs = %.1f 帧 + %.1f 帧步长），"
            "本次门禁必然判失败——这是刻意的：没有接缝可验不能算通过",
            float(duration_s),
            min_duration_s,
            float(C.WINDOW),
            float(C.WINDOW - C.OVERLAP),
        )

    active = generator if generator is not None else load_generator(cfg, logger=log)
    for prompt in probes:
        probe = _probe_one(
            active,
            prompt,
            duration_s=float(duration_s),
            steps=resolved_steps,
            guidance=resolved_guidance,
            seed=resolved_seed,
        )
        report.probes.append(probe)
        if probe.error:
            log.warning("接缝门禁: %r 生成失败（%s）", prompt, probe.error)
        else:
            log.info(
                "接缝门禁: %r → %d 帧 / %d 窗 / %d 接缝 / 最大 %.1f cm/s（阈值 %.0f）",
                prompt,
                probe.n_frames,
                probe.n_windows,
                probe.n_seams,
                probe.seam_max_cmps,
                report.threshold_cmps,
            )

    if report.passed:
        worst = report.worst
        log.info(
            "接缝连续性门禁 PASS: %d 条指令全部通过，最差 %.1f cm/s（阈值 %.0f cm/s）",
            len(report.probes),
            worst.seam_max_cmps if worst is not None else 0.0,
            report.threshold_cmps,
        )
    else:
        for reason in report.failures():
            log.error("接缝连续性门禁 FAIL: %s", reason)

    if write_to:
        target = P.resolve_out(write_to, mkparents=True)
        target.write_text(
            json.dumps(report.to_dict(), ensure_ascii=False, indent=2), encoding="utf-8"
        )
        log.info("接缝门禁报告已落盘: %s", target)
    return report
