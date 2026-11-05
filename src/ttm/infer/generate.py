"""端到端生成：中文指令 → 潜变量采样 → VAE 解码 → 融合 → 四元数/根位移 → BVH。

链路（``TTM-ARCH.md`` §7）：

::

    "慢慢走上楼梯", 5.0s
      │ bge(CLS, L2) 512d → TextProjector → cond [1,256]
      │ DDIM(η=0) + CFG(guidance) → K 个潜变量窗口 [1,38,128]
      │ MotionVAE.decode → K 个特征窗口 [150,146]
      │ blend_feature_windows（重叠 30 帧交叉淡化）
      │ denormalize(norm_stats) → to_rotations_root（**全局只积分一次**）
      │ smooth_root_track（runtime.smooth）
      ▼ quats[T,23,4] + root_pos[T,3] → BVH / Motion JSON

设计取舍：所有重活（VAE/DiT/bge）都通过构造参数注入，``generate`` 本身是纯编排，
因此单测可以用随机初始化的小模型跑通形状与连续性断言，不依赖训练产物。
"""

from __future__ import annotations

import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Sequence

import numpy as np
import torch

from ttm.common import constants as C
from ttm.common import paths as P
from ttm.common.config import TTMConfig
from ttm.common.logutil import get_logger
from ttm.data.features import MotionFeature, NormStats
from ttm.infer.blend import (
    blend_feature_windows,
    expected_frames,
    seam_velocity_jumps,
    smooth_root_track,
)
from ttm.infer.sampling import LatentSampler, SamplingConfig, tokens_for_frames
from ttm.models.diffusion import DiffusionTransformer, GaussianDiffusion
from ttm.models.text_encoder import TextEncoder
from ttm.models.vae import MotionVAE

__all__ = [
    "GeneratedMotion",
    "GenerateOptions",
    "MotionGenerator",
    "SEAM_JUMP_MAX_CMPS",
    "ground_origin_offset_cm",
    "load_generator",
    "resolve_frame_budget",
    "root_y_diagnostics",
    "shift_root_y",
    "warmup_generator",
]

#: 接缝根速度台阶的验收上限（cm/s）。
#:
#: 正常人单帧（1/30 秒）由 1 m/s² 加速度产生的速度变化只有 3.3 cm/s，所以
#: 100 cm/s 已是 30 倍余量；超过它不可能用「这个动作本来就快」来解释，只能是
#: 滑窗拼接处的根位移/速度不连续。
SEAM_JUMP_MAX_CMPS: float = 100.0

#: 根高振荡判据（F-17b）：峰峰值 ≥ 该值（cm）**且** dY 变号 ≥ :data:`ROOT_Y_OSC_MIN_SIGN_CHANGES`
#: 次 → 判为纵向振荡（记 KNOWN_LIMITATION，不当 PASS 也不当 FAIL）。
#:
#: 为什么必须有振幅门：GT 行走满窗的 dY 变号中位数就有 32 次（根高在 0.07cm 范围内
#: 抖动），只看「变号次数」会把**完全正常**的行走判成振荡。振幅门是这条判据唯一
#: 的区分度来源（GT 峰峰 p50=0.07cm / max=0.18cm）。
ROOT_Y_OSC_AMPLITUDE_CM: float = 40.0

#: 振荡判据的变号次数门槛（配合 :data:`ROOT_Y_OSC_AMPLITUDE_CM` 使用）。
ROOT_Y_OSC_MIN_SIGN_CHANGES: int = 3

#: 统计「变号」时忽略的逐帧位移噪声（cm）；低于它的 dY 视为静止，不计符号。
ROOT_Y_OSC_NOISE_CM: float = 0.05

#: 诊断里顺带给出的 F-17a 两窗宽度（帧）——与 QA 判据的 f0..f11 / f12..f23 一致。
ROOT_Y_DIAG_WINDOW: int = 12


def resolve_frame_budget(
    duration_s: float, trim_frames: int, *, fps: int = C.FPS
) -> tuple[int, int]:
    """把「用户请求时长 + 裁头帧数」换算成 ``(用户帧数, 申请帧数)``。

    F-17 的帧数不变式就在这里：先按 ``用户帧数 + 裁头帧数`` 采样，导出前丢掉
    开头 ``裁头帧数`` 帧，用户拿到的仍然是 ``用户帧数``——**时长与请求一致**。

    Args:
        duration_s: 用户请求时长（秒）。
        trim_frames: 裁头帧数（``0`` 或负数视为不裁）。
        fps: 帧率。

    Returns:
        ``(n_frames_user, n_frames_target)``，均 ≥ 1。

    Raises:
        ValueError: 时长非正。
    """
    duration = float(duration_s)
    if duration <= 0.0:
        raise ValueError(f"duration_s 必须为正，实际 {duration_s}")
    trim = max(0, int(trim_frames))
    n_frames_user = max(1, int(round(duration * int(fps))))
    return n_frames_user, n_frames_user + trim


def root_y_diagnostics(
    root_pos: np.ndarray,
    *,
    amplitude_cm: float = ROOT_Y_OSC_AMPLITUDE_CM,
    min_sign_changes: int = ROOT_Y_OSC_MIN_SIGN_CHANGES,
) -> dict[str, float]:
    """根高（Y）振荡诊断——F-17b 的口径，供 QA 读 ``meta`` / Motion JSON。

    **为什么要有这条**：F-17a（局部连续性 ≤30cm）会把「动作本身大幅上下」误判成
    启动瞬态——实测 ``在房间里走一圈`` seed43 的 Y 在 45↔147cm 之间往复（峰峰
    103cm，周期约 1.5~2.5s），裁头只是把判据窗口挪到爬坡段。这类片段既不该算
    PASS（它确实不是正常行走），也不该永远算 FAIL（它不是启动瞬态），所以单列
    成 KNOWN_LIMITATION，并把这些数字落进产物，**承认但不掩盖**。

    判据（主理人裁决）：
        ``detrended_peak_to_peak_cm >= amplitude_cm`` **且**
        ``sign_changes >= min_sign_changes`` → ``oscillating = True``

    **为什么量去趋势后的振幅**：直接取峰峰值会把「合法爬升」也算进去（GT 楼梯窗
    根高从 90cm 爬到 344cm，峰峰 254cm 但那是动作本身）。去掉线性趋势后，GT 走/跑
    窗残余振幅 **max=5.2cm**、GT 楼梯 p50=31cm / p90=60cm（z 值 × std 26.12cm 换算），
    而这条 clip 是 **99cm**——比 GT 行走大近 20 倍。

    **本字段是筛查，不是判决**：爬升类去趋势后同样可能 >40cm（GT 楼梯 p90=60cm），
    所以 F-17b 只在 **F-17a 已经失败** 时才参照本字段区分「纵向振荡 vs 启动台阶」；
    楼梯类 F-17a 本来就通过（24.96cm），不会走到这里。

    Args:
        root_pos: ``[T,3]`` 根位移（cm）。
        amplitude_cm: 去趋势振幅门槛（默认 40cm）。
        min_sign_changes: dY 变号次数门槛（默认 3）。

    Returns:
        ``{"peak_to_peak_cm", "detrended_peak_to_peak_cm", "sign_changes",
        "period_s", "local_jump_cm", "oscillating"}``；帧数不足 3 时全部为 0/False。

    ``local_jump_cm`` 就是 F-17a 的读数（``|median(f0..f11) − median(f12..f23)|``），
    一并给出来，QA 就能只用这一份 meta 做门控：
    ``local_jump_cm > 30 and oscillating → KNOWN_LIMITATION``。
    """
    array = np.asarray(root_pos, dtype=np.float64)
    if array.ndim != 2 or array.shape[1] != 3:
        raise ValueError(f"root_pos 形状应为 [T,3]，实际 {array.shape}")
    if array.shape[0] < 3:
        return {
            "peak_to_peak_cm": 0.0,
            "detrended_peak_to_peak_cm": 0.0,
            "sign_changes": 0,
            "period_s": 0.0,
            "local_jump_cm": 0.0,
            "oscillating": False,
        }

    height = array[:, 1]
    peak_to_peak = float(height.max() - height.min())

    # F-17a 的读数（两窗根高中位数差），供 QA 单点门控
    window = int(ROOT_Y_DIAG_WINDOW)
    if height.size >= 2 * window:
        local_jump = abs(
            float(np.median(height[:window])) - float(np.median(height[window : 2 * window]))
        )
    else:
        local_jump = 0.0

    # 去线性趋势：把「合法爬升」从「上下往复」里剥离
    timeline = np.arange(height.size, dtype=np.float64)
    slope, intercept = np.polyfit(timeline, height, 1)
    residual = height - (slope * timeline + intercept)
    detrended_peak_to_peak = float(residual.max() - residual.min())

    # 噪声地板以下的位移不参与符号判定（否则静止片段会被浮点抖动刷出上百次变号）
    delta = np.diff(residual)
    significant = delta[np.abs(delta) >= ROOT_Y_OSC_NOISE_CM]
    if significant.size >= 2:
        sign_changes = int(np.count_nonzero(np.diff(np.sign(significant)) != 0))
    else:
        sign_changes = 0

    # 主周期：去趋势残差去均值后的 FFT 峰值（只作诊断展示，不参与判定）
    centered = residual - residual.mean()
    period_s = 0.0
    if centered.std() > 1e-9:
        spectrum = np.abs(np.fft.rfft(centered))
        freqs = np.fft.rfftfreq(centered.size, d=1.0 / float(C.FPS))
        if spectrum.size > 1 and spectrum[1:].max() > 0.0:
            peak = int(np.argmax(spectrum[1:]) + 1)
            period_s = float(1.0 / freqs[peak]) if freqs[peak] > 0.0 else 0.0

    oscillating = bool(
        detrended_peak_to_peak >= float(amplitude_cm)
        and sign_changes >= int(min_sign_changes)
    )
    return {
        "peak_to_peak_cm": round(peak_to_peak, 3),
        "detrended_peak_to_peak_cm": round(detrended_peak_to_peak, 3),
        "sign_changes": sign_changes,
        "period_s": round(period_s, 3),
        "local_jump_cm": round(local_jump, 3),
        "oscillating": oscillating,
    }


def ground_origin_offset_cm(quats: np.ndarray, root_pos: np.ndarray) -> float:
    """计算「首帧最低足触地点」的世界高度（cm）——ground_origin 的平移量。

    背景（根高排查的下游结论）：特征表示是**绝对根高**，楼梯类生成会从
    数据里继承「世界坐标随楼层累加」的根高（实测终值 403cm）。BVH/预览如果
    原样播放，角色会从 4 米高空开始。正确修法是把整条动作**刚性平移**，
    使首帧最低足触地点 ≈ 0。

    用**首帧足部**最低高度而不是首帧根高：首帧可能正处于腾空（跳跃/上楼），
    按根高平移会把腾空姿态钉在地上。

    足部关节取 :data:`ttm.common.constants.FOOT_CONTACT_JOINTS`（左右脚踝），
    与预处理 ``motion_quality_issue`` 的足高口径一致。

    Args:
        quats: ``[F,23,4]`` 源骨架局部四元数（xyzw）。
        root_pos: ``[F,3]`` 根位移（cm）。

    Returns:
        首帧 ``min(左足, 右足)`` 世界高度（cm）。把 root_pos 的 Y 整体减去
        该值后，首帧足触地点落在 0 附近。

    Raises:
        ValueError: 形状不符。
    """
    world = _world_joint_positions(quats, root_pos)
    feet = world[0, list(C.FOOT_CONTACT_JOINTS), 1]  # [2]
    return float(feet.min())


def _world_joint_positions(quats: np.ndarray, root_pos: np.ndarray) -> np.ndarray:
    """源骨架正运动学：局部四元数 + 根位移 → 全关节世界坐标（cm）。

    与 :meth:`ttm.data.skeleton.SkeletonDef.fk` 同一套递推，只是输入是
    四元数（生成链路的原生表示）而不是欧拉角。

    Args:
        quats: ``[F,23,4]``。
        root_pos: ``[F,3]``。

    Returns:
        ``[F,23,3]`` float64 世界坐标。
    """
    from scipy.spatial.transform import Rotation  # noqa: PLC0415 - 与 vae/dit 同级的重依赖

    array = np.asarray(quats, dtype=np.float64)
    positions = np.asarray(root_pos, dtype=np.float64)
    if array.ndim != 3 or array.shape[1:] != (C.N_JOINTS, 4):
        raise ValueError(f"quats 形状应为 [F,{C.N_JOINTS},4]，实际 {array.shape}")
    if positions.shape != (array.shape[0], 3):
        raise ValueError(f"root_pos 形状应为 [{array.shape[0]},3]，实际 {positions.shape}")

    local = Rotation.from_quat(array.reshape(-1, 4)).as_matrix().reshape(
        array.shape[0], C.N_JOINTS, 3, 3
    )
    offsets = np.asarray(C.DEFAULT_REST_OFFSETS_CM, dtype=np.float64)  # [23,3]
    world_rot = np.empty_like(local)
    world_pos = np.zeros((array.shape[0], C.N_JOINTS, 3), dtype=np.float64)
    world_rot[:, 0] = local[:, 0]
    world_pos[:, 0] = positions
    for idx in range(1, C.N_JOINTS):
        parent = int(C.PARENTS[idx])
        world_rot[:, idx] = world_rot[:, parent] @ local[:, idx]
        world_pos[:, idx] = world_pos[:, parent] + world_rot[:, parent] @ offsets[idx]
    return world_pos


def shift_root_y(root_pos: np.ndarray, offset_cm: float) -> np.ndarray:
    """把根轨迹的 Y 整体平移 ``-offset_cm``（**只许平移，不许缩放**）。

    Args:
        root_pos: ``[F,3]`` 根位移（cm）。
        offset_cm: 平移量（取 :func:`ground_origin_offset_cm` 的返回值）。

    Returns:
        平移后的副本（不修改入参）。
    """
    shifted = np.asarray(root_pos, dtype=np.float64).copy()
    shifted[:, 1] -= float(offset_cm)
    return shifted


@dataclass(frozen=True)
class GenerateOptions:
    """生成参数。"""

    #: 目标时长（秒）；``None`` 时按 5 秒（= 一个训练窗口）。
    duration_s: float | None = 5.0
    #: DDIM 步数。
    steps: int = 50
    #: CFG 强度（``runtime.guidance``）。
    guidance: float = 3.0
    #: 根轨迹平滑强度（``runtime.smooth``）。
    smooth: float = 0.2
    #: 随机种子。
    seed: int = 42
    #: 是否把首帧最低足触地点平移到 0（ground_origin）。只影响 BVH / 预览；
    #: Motion JSON 始终保留原始世界坐标（offset 写进 meta 由消费端决定）。
    ground_origin: bool = True
    #: 启动瞬态裁头帧数（F-17）。生成时多申请同样帧数、导出前丢掉开头，
    #: **用户侧时长不变**；ground_origin 在裁头后的首帧重新计算。
    #: ``0`` = 关闭。裸 ``GenerateOptions()`` 默认不裁（保持单测/既有产物口径），
    #: 生产入口（服务端 / 05_generate / 06_smoke）从 ``cfg.runtime.trim_head_frames``
    #: 注入（配置默认 12）。
    trim_head_frames: int = 0
    #: 是否导出 BVH。
    export_bvh: bool = True
    #: 导出目录（``None`` → ``runs/{run}/samples``）。
    out_dir: str | None = None


@dataclass
class GeneratedMotion:
    """一次生成的结果。"""

    prompt: str
    quats: np.ndarray
    root_pos: np.ndarray
    feature: MotionFeature
    n_windows: int
    seam_jumps: np.ndarray
    timing_ms: dict[str, float] = field(default_factory=dict)
    bvh_path: str = ""
    #: 首帧最低足触地点的世界高度（cm）。
    #: BVH / 预览侧用 ``root_pos.Y - ground_offset_cm`` 做地面锚定；
    #: ``root_pos`` 本身永远保留原始世界坐标。
    ground_offset_cm: float = 0.0
    meta: dict[str, Any] = field(default_factory=dict)

    @property
    def n_frames(self) -> int:
        """帧数。"""
        return int(self.root_pos.shape[0])

    def to_dict(self) -> dict[str, Any]:
        """摘要（写日志/summary 用）。"""
        return {
            "prompt": self.prompt,
            "n_frames": self.n_frames,
            "duration_s": round(self.n_frames / float(C.FPS), 3),
            "n_windows": int(self.n_windows),
            "seam_jump_max_cmps": round(float(self.seam_jumps.max()) if self.seam_jumps.size else 0.0, 3),
            "seam_jump_mean_cmps": round(float(self.seam_jumps.mean()) if self.seam_jumps.size else 0.0, 3),
            "seam_jump_threshold_cmps": float(self.meta.get("seam_threshold_cmps", SEAM_JUMP_MAX_CMPS)),
            "seam_ok": bool(self.meta.get("seam_ok", True)),
            "timing_ms": {key: round(value, 1) for key, value in self.timing_ms.items()},
            "bvh": self.bvh_path,
            "meta": self.meta,
        }


class MotionGenerator:
    """把「中文指令」变成「可播放动作」的编排器。"""

    def __init__(
        self,
        *,
        text_encoder: TextEncoder,
        vae: MotionVAE,
        dit: DiffusionTransformer,
        projector: torch.nn.Module,
        norm_stats: NormStats,
        diffusion: GaussianDiffusion | None = None,
        device: torch.device | str = "cpu",
        logger: Any | None = None,
    ) -> None:
        """初始化。

        Args:
            text_encoder: 冻结的 bge 编码器（提供 ``encode``）。
            vae: 已训练的 MotionVAE（``eval()``）。
            dit: 已训练的 DiT（``eval()``）。
            projector: 已训练文本投影层。
            norm_stats: 预处理阶段落盘的归一化统计。
            diffusion: 调度器；``None`` 时按 ``cfg.diffusion.timesteps`` 默认新建。
            device: 设备。
            logger: 可注入 logger。
        """
        self.text_encoder = text_encoder
        self.device = torch.device(device)
        # **三件套统一搬到 device**：``load_generator`` 里 ``TextEncoder`` 是按
        # ``device="cpu"`` 构造的（bge 冻结编码器本身在 CPU 上跑就够了，省显存），
        # 但它的 ``projector`` 是要参与前向的可训练层——留在 CPU 上就会在
        # ``condition()`` 里抛
        # ``Expected all tensors to be on the same device, but got mat1 is on cuda:0``。
        # 这里统一 ``.to(self.device)``（幂等），比在调用点逐处补更安全。
        self.vae = vae.to(self.device).eval()
        self.dit = dit.to(self.device).eval()
        self.projector = projector.to(self.device).eval()
        self.norm_stats = norm_stats
        self.sampler = LatentSampler(
            self.dit,
            # 不传 diffusion 时也要跟设备：默认 CPU 表在 CUDA 上会索引报错
            diffusion or GaussianDiffusion(device=self.device),
            latent_dim=int(C.LATENT_DIM),
            cond_dim=int(self.projector.proj_dim),
            device=self.device,
        )
        self.log = logger or get_logger("ttm.infer.generate", script="05_generate")

    # -- 条件 ---------------------------------------------------------------
    @torch.no_grad()
    def condition(self, prompt: str) -> torch.Tensor:
        """中文指令 → ``[1, proj_dim]`` 条件向量。

        Args:
            prompt: 中文指令（如「慢慢走上楼梯」）。

        Returns:
            ``[1, proj_dim]`` 张量（在 ``self.device`` 上）。
        """
        vector = self.text_encoder.encode([prompt])
        tensor = torch.from_numpy(np.asarray(vector, dtype=np.float32)).to(self.device)
        return self.projector(tensor)

    # -- 生成 ---------------------------------------------------------------
    @torch.no_grad()
    def generate(self, prompt: str, options: GenerateOptions | None = None) -> GeneratedMotion:
        """生成一段动作（长动作自动滑窗 + 融合）。

        Args:
            prompt: 中文指令。
            options: 生成参数。

        Returns:
            :class:`GeneratedMotion`。

        Raises:
            ValueError: 时长为负或 prompt 为空。
        """
        opts = options or GenerateOptions()
        if not str(prompt).strip():
            raise ValueError("prompt 不能为空")
        duration_s = float(opts.duration_s if opts.duration_s is not None else C.WINDOW / C.FPS)
        if duration_s <= 0:
            raise ValueError(f"duration_s 必须为正，实际 {duration_s}")

        timing: dict[str, float] = {}
        started = time.perf_counter()

        # 1) 文本条件
        mark = time.perf_counter()
        cond = self.condition(prompt)
        timing["text_ms"] = (time.perf_counter() - mark) * 1000.0

        # 2) 潜变量采样（滑窗）
        # trim_head（F-17）：首窗开头的 latent 尚未收敛，前 12 帧根高有 25~41cm
        # 的横向跳变（GT 同口径 p99=0.58cm）。做法是按「用户时长 + 裁头帧数」
        # 申请窗口，导出前丢掉开头——用户侧时长不变。代价是多一个窗口。
        trim = max(0, int(opts.trim_head_frames))
        n_frames_user, n_frames_target = resolve_frame_budget(
            duration_s, trim, fps=C.FPS
        )
        sampling = SamplingConfig(
            steps=int(opts.steps),
            guidance=float(opts.guidance),
            overlap=int(C.OVERLAP),
            window=int(C.WINDOW),
            seed=int(opts.seed),
        )
        n_windows = sampling.windows_for_frames(n_frames_target)
        n_tokens = tokens_for_frames(C.WINDOW)
        mark = time.perf_counter()
        latents = self.sampler.sample_sequence(
            cond, n_windows=n_windows, n_tokens=n_tokens, config=sampling
        )
        timing["sample_ms"] = (time.perf_counter() - mark) * 1000.0

        # 3) 解码 → 特征窗口
        mark = time.perf_counter()
        windows: list[np.ndarray] = []
        for latent in latents:
            recon = self.vae.decode(latent.to(self.device), target_len=int(C.WINDOW))
            feat = recon[0].transpose(0, 1).float().cpu().numpy()  # [150,146]
            windows.append(feat)
        timing["decode_ms"] = (time.perf_counter() - mark) * 1000.0

        # 4) 融合（重叠区交叉淡化）
        mark = time.perf_counter()
        blended = blend_feature_windows(
            windows, overlap=int(C.OVERLAP), window=int(C.WINDOW)
        )
        if blended.shape[0] > n_frames_target:
            blended = blended[:n_frames_target]
        if trim > 0:
            if blended.shape[0] <= trim:
                raise ValueError(
                    f"trim_head_frames={trim} 超过可用帧数 {blended.shape[0]}，无法裁头"
                )
            # 丢掉开头（启动瞬态）。之后所有下游——四元数、根位移、接缝指标、
            # ground_origin 平移量——都基于**裁头后的首帧**，顺序不可反。
            blended = blended[trim:]
        timing["blend_ms"] = (time.perf_counter() - mark) * 1000.0

        # 5) 反归一化 → 四元数/根位移（**全局只积分一次**）
        mark = time.perf_counter()
        feature = MotionFeature(
            feat=blended,
            mask=np.ones(blended.shape[0], dtype=np.uint8),
            fps=C.FPS,
        ).denormalize(self.norm_stats)
        quats, root_pos = feature.to_rotations_root()
        root_pos = smooth_root_track(root_pos, smooth=float(opts.smooth))
        # ground_origin：首帧最低足触地点的世界高度。root_pos 本身**不平移**
        # （JSON 要保留世界坐标），offset 存在结果上，BVH / 预览各自决定是否用。
        ground_offset = (
            ground_origin_offset_cm(quats, root_pos) if opts.ground_origin else 0.0
        )
        if opts.ground_origin and abs(ground_offset) > 1.0:
            self.log.info(
                "ground_origin: 首帧最低足触地点 %.1fcm（BVH/预览将整体平移到地面）",
                ground_offset,
            )
        timing["post_ms"] = (time.perf_counter() - mark) * 1000.0
        timing["total_ms"] = (time.perf_counter() - started) * 1000.0

        jumps = seam_velocity_jumps(
            root_pos,
            n_windows,
            fps=C.FPS,
            window=int(C.WINDOW),
            overlap=int(C.OVERLAP),
            head_trim=int(trim),
        )
        # 接缝连续性断言（主理人 M3 硬性要求）：滑窗拼接出来的长动作，接缝处的
        # 根速度必须连续——不连续就意味着「人物在接缝处瞬移/急停」，用户一眼能看出来。
        # 阈值 100 cm/s：正常人单帧（1/30s）速度变化由 1 m/s² 加速度只有 3.3 cm/s，
        # 100 cm/s 已经是 30 倍余量；超过它不可能用「动作本来就快」解释。
        seam_max = float(jumps.max()) if jumps.size else 0.0
        seam_ok = seam_max <= SEAM_JUMP_MAX_CMPS
        if not seam_ok:
            self.log.warning(
                "接缝速度越界: %.1f cm/s > %.1f cm/s（%d 个接缝，最大 %.1f）",
                seam_max,
                SEAM_JUMP_MAX_CMPS,
                int(jumps.size),
                seam_max,
            )
        result = GeneratedMotion(
            prompt=str(prompt),
            quats=quats,
            root_pos=root_pos,
            feature=feature,
            n_windows=n_windows,
            seam_jumps=jumps,
            timing_ms=timing,
            ground_offset_cm=float(ground_offset),
            meta={
                "layout_version": int(feature.layout_version),
                "guidance": float(opts.guidance),
                "steps": int(opts.steps),
                "seed": int(opts.seed),
                "smooth": float(opts.smooth),
                "n_tokens": int(n_tokens),
                "expected_frames": int(expected_frames(n_windows)),
                "seam_max_cmps": seam_max,
                "seam_threshold_cmps": SEAM_JUMP_MAX_CMPS,
                "seam_ok": bool(seam_ok),
                "ground_origin": bool(opts.ground_origin),
                "ground_origin_offset_cm": float(ground_offset),
                "trim_head_frames": int(trim),
                "frames_before_trim": int(n_frames_target),
                # F-17b：纵向振荡诊断（QA 直接读，用来判 KNOWN_LIMITATION）
                "root_y_diagnostics": root_y_diagnostics(root_pos),
            },
        )
        self.log.info(
            "生成完成: %r → %d 帧 / %d 窗 / 接缝最大跳变 %.1f cm/s / 用时 %.0f ms",
            prompt,
            result.n_frames,
            n_windows,
            float(jumps.max()) if jumps.size else 0.0,
            timing["total_ms"],
        )

        # 6) 可选自动落盘（``export_bvh`` 且给了 ``out_dir`` 才写，默认不写）
        if opts.export_bvh and opts.out_dir:
            mark = time.perf_counter()
            self.export(result, opts.out_dir)
            timing["export_ms"] = (time.perf_counter() - mark) * 1000.0
            timing["total_ms"] = (time.perf_counter() - started) * 1000.0
            result.timing_ms = dict(timing)
        return result

    # -- 导出 ---------------------------------------------------------------
    def export(
        self,
        motion: GeneratedMotion,
        out_dir: str | Path,
        *,
        name: str | None = None,
        ground_origin: bool = True,
    ) -> Path:
        """把生成结果写成 BVH（经写盘守卫）。

        Args:
            motion: 生成结果。
            out_dir: 输出目录。
            name: 文件名（不含扩展名）；``None`` 时按 prompt 的 slug + 时间戳。
            ground_origin: 是否把首帧最低足触地点平移到 0（默认开）。只影响
                BVH——``motion.root_pos`` 不动，JSON 侧仍保留世界坐标。

        Returns:
            写出的 BVH 路径。
        """
        from ttm.export.bvh_io import write_bvh  # noqa: PLC0415 - 避免顶层循环导入

        stem = name or _slug(motion.prompt)
        target = Path(out_dir) / f"{stem}.bvh"
        root_pos = (
            shift_root_y(motion.root_pos, motion.ground_offset_cm)
            if ground_origin
            else motion.root_pos
        )
        path = write_bvh(target, motion.quats, root_pos, fps=C.FPS)
        motion.bvh_path = path.as_posix()
        return path


def _slug(text: str, *, limit: int = 24) -> str:
    """把中文/英文 prompt 变成安全的文件名片段（保留中文，替换空白与斜杠）。"""
    cleaned = "".join(ch if ch.isalnum() or ch in "-_" else "_" for ch in str(text).strip())
    cleaned = "_".join(part for part in cleaned.split("_") if part)
    return (cleaned or "motion")[:limit]


def load_generator(
    cfg: TTMConfig,
    *,
    device: torch.device | str | None = None,
    logger: Any | None = None,
) -> MotionGenerator:
    """按配置加载一整套生成组件（VAE / DiT / 文本投影 / 归一化统计）。

    Args:
        cfg: 生效配置。
        device: 设备（默认取 ``cfg.runtime.device``，CUDA 不可用时回落 CPU）。
        logger: 可注入 logger。

    Returns:
        :class:`MotionGenerator`。

    Raises:
        FileNotFoundError: VAE / DiT ckpt 或 norm_stats 缺失。
    """
    from ttm.train.train_diffusion import _load_diffusion_checkpoint  # noqa: PLC0415
    from ttm.train.train_vae import load_checkpoint  # noqa: PLC0415

    log = logger or get_logger("ttm.infer.generate", script="05_generate")
    resolved = torch.device(
        device
        if device is not None
        else (cfg.runtime.device if torch.cuda.is_available() else "cpu")
    )
    norm_stats = NormStats.load(cfg.norm_stats_path())

    vae_ckpt = Path(cfg.runtime.vae_ckpt) if cfg.runtime.vae_ckpt else cfg.run_dir("vae") / "best.pt"
    dif_ckpt = (
        Path(cfg.runtime.diffusion_ckpt)
        if cfg.runtime.diffusion_ckpt
        else cfg.run_dir("diffusion") / "best.pt"
    )
    for path in (vae_ckpt, dif_ckpt):
        if not path.exists():
            raise FileNotFoundError(f"checkpoint 不存在: {path}")

    vae, _ = load_checkpoint(vae_ckpt, map_location=resolved)
    vae.to(resolved)
    vae.eval()

    dit = DiffusionTransformer(
        latent_dim=cfg.vae.latent_dim,
        d_model=cfg.diffusion.d_model,
        n_layers=cfg.diffusion.n_layers,
        n_heads=cfg.diffusion.n_heads,
        cond_dim=cfg.text.proj_dim,
        dropout=0.0,
    ).to(resolved)

    encoder = TextEncoder(
        cfg.text.encoder,
        model_path=cfg.text.model_path or None,
        proj_dim=cfg.text.proj_dim,
        frozen=cfg.text.frozen,
        max_len=cfg.text.max_len,
        device="cpu",
        local_files_only=True,
    )
    payload = _load_diffusion_checkpoint(
        dif_ckpt, model=dit, projector=encoder.projector, device=resolved
    )
    encoder.projector.eval()
    log.info(
        "生成组件就绪: vae=%s dit=%s text=%s guidance=%.1f steps=%d",
        vae_ckpt.name,
        dif_ckpt.name,
        payload.get("text_model", "?"),
        float(cfg.runtime.guidance),
        int(cfg.diffusion.ddim_steps),
    )
    generator = MotionGenerator(
        text_encoder=encoder,
        vae=vae,
        dit=dit,
        projector=encoder.projector,
        norm_stats=norm_stats,
        # 系数表跟设备走（DDIM 步进要拿 alpha 与 CUDA 张量做运算）
        diffusion=GaussianDiffusion(timesteps=cfg.diffusion.timesteps, device=resolved),
        device=resolved,
        logger=log,
    )
    # 预热文本编码器：首次 encode 的 lazy init 约 3 秒，若算进单条时延会让
    # ``text_ms`` 变成假数字（详见 :func:`warmup_generator`）。
    warm_ms = warmup_generator(generator)
    log.info("文本编码器预热完成: %.0f ms（不计入单条推理时延）", warm_ms)
    return generator


def warmup_generator(generator: MotionGenerator, *, prompt: str = "一个人自然地站着") -> float:
    """预热整条推理链路，返回耗时（ms）。

    为什么必须预热，而且**不只预热文本**：

    * bge 的第一次 ``encode`` 会触发 lazy init（tokenizer + 权重搬进计算图），
      实测约 3 秒。不预热就会让报告里的 ``text_ms`` 变成 3000ms 级的假数字；
    * VAE 解码器的**第一次**调用要建 CUDA kernel，实测 **106.8ms**，而稳态只有
      **2.3ms**（同一份权重、同样形状）——40 倍差。只预热文本的话，时延分解表里
      ``decode`` 那一列会由「谁是第一条指令」决定，而不是由模型决定。

    所以这里跑一次完整的迷你生成（少量 DDIM 步 + 一个窗口），把
    ``text / sample / decode / post`` 四段一次性全部预热。预热耗时单独记成
    ``load_warmup_ms``，不计入单条推理时延。

    Args:
        generator: 已就绪的生成器。
        prompt: 预热用的指令（内容不影响结果，只要包含中文），默认一句无关的占位。

    Returns:
        预热耗时（毫秒）。
    """
    started = time.perf_counter()
    generator.condition(prompt)
    generator.generate(
        prompt,
        GenerateOptions(duration_s=float(C.WINDOW) / float(C.FPS), steps=2, export_bvh=False),
    )
    return (time.perf_counter() - started) * 1000.0
