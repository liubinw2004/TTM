"""T10：FastAPI 推理服务（文本 → 动作 → BVH / Motion JSON）。

职责边界（``TTM-ARCH.md`` §7 T10）：

* **只做编排**：调用 :class:`ttm.infer.generate.MotionGenerator` 生成，再调
  :mod:`ttm.export.retarget` 重定向导出；不包含任何训练/预处理逻辑；
* **重定向在服务端**：输出 UE 骨名的 Motion JSON + BVH，前端只负责蒙皮播放；
* **懒加载 + 单例**：权重很大（VAE 60M + DiT 10M + bge 24M），首次请求时才加载，
  之后复用；加载失败返回 503 而不是抛 500。

启动::

    . D:/TTM/scripts/env.ps1
    uvicorn server.app:app --host 127.0.0.1 --port 8765

    # 或直接跑本文件
    python server/app.py --config configs/tier_s.yaml

接口::

    GET  /health                 → {status, device, weights, retarget}
    GET  /prompts                → 内置示例指令
    POST /generate               → 生成并返回 Motion JSON（内联）+ 资源 URL
    GET  /motions/{name}.json    → 下载 Motion JSON
    GET  /motions/{name}.bvh     → 下载 BVH
    GET  /                       → 前端页面（web/index.html）
"""

from __future__ import annotations

import argparse
import sys
import threading
import time
from pathlib import Path
from typing import Any, Literal

# --- sys.path 引导（允许 `python server/app.py` 直接运行）--------------------
_PROJECT_ROOT = Path(__file__).resolve().parents[1]
_SRC_DIR = _PROJECT_ROOT / "src"
if str(_SRC_DIR) not in sys.path:
    sys.path.insert(0, str(_SRC_DIR))

from fastapi import FastAPI, HTTPException, Query  # noqa: E402
from fastapi.middleware.cors import CORSMiddleware  # noqa: E402
from fastapi.responses import FileResponse, JSONResponse  # noqa: E402
from fastapi.staticfiles import StaticFiles  # noqa: E402
from pydantic import BaseModel, Field  # noqa: E402

from ttm.common import constants as C  # noqa: E402
from ttm.common import paths as P  # noqa: E402
from ttm.common.config import TTMConfig, load_config  # noqa: E402
from ttm.common.logutil import close_all, get_logger  # noqa: E402

__all__ = ["GenerateRequest", "GenerateResponse", "app", "create_app"]

#: 默认配置文件（可用 ``TTM_CONFIG`` 环境变量覆盖）。
DEFAULT_CONFIG = "configs/tier_s.yaml"

#: 前端页面目录。
WEB_DIR = _PROJECT_ROOT / "web"

#: 生成产物目录（BVH / JSON 落盘位置）。
OUTPUT_DIR = _PROJECT_ROOT / "runs" / "api"

#: 前端播放用的角色 FBX（UE 骨架，Python **不解析**，rest pose 由前端自己算）。
CHARACTER_FBX = _PROJECT_ROOT / "SK_Celeste_02.FBX"

#: 内置示例指令（前端下拉框直接用）。
EXAMPLE_PROMPTS: tuple[str, ...] = (
    "慢慢走上楼梯",
    "挥手打招呼然后转身走开",
    "打一套陈式太极拳，动作舒缓连贯",
    "下蹲捡起地上的箱子",
    "原地跳跃两次",
)

ExportFormat = Literal["json", "bvh", "both", "none"]

#: prompt 软长度上限（字）。超过只回 warning 不拒绝；硬上限由
#: :class:`GenerateRequest` 的 ``max_length=200`` 负责（防滥用）。
PROMPT_SOFT_LIMIT: int = 40


def prompt_warnings(prompt: str) -> list[str]:
    """按 prompt 长度组装软警告（不阻断请求）。

    训练集指令都很短（PRD 统计中位数 9 字、95 分位 21 字），过长的指令会把
    文本编码器推到训练分布之外——效果会掉，但不是错误，所以只警告。

    Args:
        prompt: 已 strip 的中文指令。

    Returns:
        警告列表；空列表表示没有需要注意的地方。
    """
    text = str(prompt)
    if len(text) > PROMPT_SOFT_LIMIT:
        return [
            f"prompt 超出训练分布（{len(text)} 字 > {PROMPT_SOFT_LIMIT} 字），效果可能下降"
        ]
    return []


# ---------------------------------------------------------------------------
# 请求 / 响应模型
# ---------------------------------------------------------------------------
class GenerateRequest(BaseModel):
    """生成请求。"""

    # 长度上限 200 防滥用；>40 字超出训练分布（PRD），只回警告不拒绝
    prompt: str = Field(..., min_length=1, max_length=200, description="中文动作指令")
    duration_s: float = Field(5.0, gt=0.0, le=60.0, description="目标时长（秒）")
    steps: int = Field(50, ge=1, le=500, description="DDIM 采样步数（η=0，确定性）")
    guidance: float | None = Field(
        None,
        ge=0.0,
        le=20.0,
        description="CFG 引导强度（1.0 = 关闭）；缺省回落到配置 runtime.guidance",
    )
    smooth: float = Field(0.2, ge=0.0, le=0.95, description="根轨迹 EMA 平滑强度")
    seed: int = Field(42, ge=0, le=2**31 - 1, description="随机种子")
    export: ExportFormat = Field("both", description="导出格式")
    ground_origin: bool = Field(
        True,
        description="把首帧最低足触地点平移到 0（BVH/预览侧）；Motion JSON 始终保留原始世界坐标",
    )


class TimingInfo(BaseModel):
    """时延分解（毫秒）。"""

    text_ms: float = 0.0
    sample_ms: float = 0.0
    decode_ms: float = 0.0
    blend_ms: float = 0.0
    post_ms: float = 0.0
    export_ms: float = 0.0
    total_ms: float = 0.0


class GenerateResponse(BaseModel):
    """生成响应。"""

    prompt: str
    n_frames: int
    n_windows: int
    duration_s: float
    fps: int
    seam_jumps: list[float]
    timing: TimingInfo
    motion: dict[str, Any] | None = Field(None, description="``ttm-motion/v1`` 内容")
    json_url: str = ""
    bvh_url: str = ""
    ground_origin_applied: bool = Field(
        False, description="BVH/预览侧是否做了地面锚定平移"
    )
    ground_origin_offset_cm: float = Field(
        0.0, description="首帧最低足触地点的世界高度（cm）；预览对 root Y 减去它"
    )
    guidance: float = Field(0.0, description="本次实际使用的 CFG 强度（配置回落后的值）")
    warnings: list[str] = Field(default_factory=list, description="软警告（不影响结果）")
    trim_head_frames: int = Field(
        0, description="本次实际生效的启动瞬态裁头帧数（F-17）；0 = 未裁头"
    )
    root_y_diagnostics: dict[str, Any] = Field(
        default_factory=dict,
        description="F-17b 根高振荡诊断：peak_to_peak_cm / sign_changes / period_s / oscillating",
    )


# ---------------------------------------------------------------------------
# 懒加载的生成器持有者
# ---------------------------------------------------------------------------
class _GeneratorHolder:
    """线程安全的懒加载单例（权重加载一次，长期复用）。"""

    def __init__(self, cfg: TTMConfig) -> None:
        self.cfg = cfg
        self._lock = threading.Lock()
        self._generator: Any | None = None
        self._error: str = ""
        self._load_ms: float = 0.0

    @property
    def loaded(self) -> bool:
        """是否已成功加载。"""
        return self._generator is not None

    @property
    def error(self) -> str:
        """最近一次加载失败的原因（空串表示没有失败）。"""
        return self._error

    @property
    def load_ms(self) -> float:
        """加载耗时（毫秒）。"""
        return self._load_ms

    def get(self) -> Any:
        """取生成器（必要时加载）。

        Returns:
            :class:`ttm.infer.generate.MotionGenerator`。

        Raises:
            RuntimeError: 权重缺失或加载失败。
        """
        if self._generator is not None:
            return self._generator
        with self._lock:
            if self._generator is not None:
                return self._generator
            started = time.perf_counter()
            try:
                from ttm.infer.generate import load_generator  # noqa: PLC0415

                self._generator = load_generator(self.cfg)
                self._error = ""
            except Exception as exc:  # noqa: BLE001 - 统一转成 503
                self._error = f"{type(exc).__name__}: {exc}"
                get_logger("ttm.server").error("生成器加载失败: %s", self._error)
                raise RuntimeError(self._error) from exc
            finally:
                self._load_ms = (time.perf_counter() - started) * 1000.0
                get_logger("ttm.server").info("生成器加载完成，用时 %.0f ms", self._load_ms)
            return self._generator

    def reset(self) -> None:
        """释放缓存的生成器（换权重后手动调用）。"""
        with self._lock:
            self._generator = None
            self._error = ""


def create_app(cfg: TTMConfig | None = None) -> FastAPI:
    """构造 FastAPI 应用（便于测试注入配置）。

    Args:
        cfg: 生效配置；``None`` 时按 ``TTM_CONFIG`` 环境变量或默认路径加载。

    Returns:
        配置好的 :class:`fastapi.FastAPI` 实例。
    """
    import os  # noqa: PLC0415 - 仅此处需要

    active = cfg or load_config(os.environ.get("TTM_CONFIG", DEFAULT_CONFIG))
    log = get_logger("ttm.server", script="server")
    holder = _GeneratorHolder(active)
    out_dir = P.safe_makedirs(OUTPUT_DIR)

    app = FastAPI(
        title="TTM 文本驱动动作生成服务",
        version="1.0.0",
        description="中文指令 → 人体动作（BVH / UE 骨架 Motion JSON）",
    )
    app.add_middleware(
        CORSMiddleware,
        allow_origins=["*"],  # 本地演示：允许前端直连
        allow_credentials=False,
        allow_methods=["*"],
        allow_headers=["*"],
    )
    app.state.cfg = active
    app.state.holder = holder
    app.state.output_dir = out_dir

    # -- 基础信息 -----------------------------------------------------------
    @app.get("/health")
    def health() -> JSONResponse:
        """健康检查：报设备、权重是否就位、是否已加载。"""
        payload = {
            "status": "ok" if holder.loaded or not holder.error else "degraded",
            "device": active.runtime.device,
            "precision": active.runtime.precision,
            "retarget": active.retarget.spine_merge,
            "weights": {
                "vae": _weight_status(_vae_ckpt(active)),
                "diffusion": _weight_status(_diffusion_ckpt(active)),
                "norm_stats": _weight_status(active.norm_stats_path()),
                "text_model": _weight_status(Path(active.text.model_path)),
            },
            "generator": {
                "loaded": holder.loaded,
                "load_ms": round(holder.load_ms, 1),
                "error": holder.error,
            },
            "skeleton": {
                "internal_joints": C.N_JOINTS,
                "ue_bones": len(C.UE_BONE_ORDER),
                "feature_dim": C.FEATURE_DIM,
                "window": C.WINDOW,
                "overlap": C.OVERLAP,
                "layout_version": C.MOTION_JSON_FORMAT,
            },
        }
        code = 200 if payload["status"] == "ok" else 200  # degraded 仍返回 200，便于探活
        return JSONResponse(payload, status_code=code)

    @app.get("/prompts")
    def prompts() -> dict[str, Any]:
        """返回内置示例指令与约束。"""
        return {
            "prompts": list(EXAMPLE_PROMPTS),
            "max_duration_s": active.serve.max_duration_s,
            # defaults 跟随配置（别再写死——改了配置前端要跟上的必须是同一个数）
            "defaults": {
                "duration_s": 5.0,
                "steps": int(active.diffusion.ddim_steps),
                "guidance": float(active.runtime.guidance),
                "smooth": float(active.runtime.smooth),
            },
        }

    # -- 生成 ---------------------------------------------------------------
    @app.post("/generate", response_model=GenerateResponse)
    def generate(request: GenerateRequest) -> GenerateResponse:
        """生成动作，可选落盘 BVH / Motion JSON。"""
        prompt = request.prompt.strip()
        if not prompt:
            raise HTTPException(status_code=422, detail="prompt 不能为空")
        if request.duration_s > active.serve.max_duration_s + 1e-6:
            raise HTTPException(
                status_code=422,
                detail=f"duration_s={request.duration_s} 超过上限 {active.serve.max_duration_s}",
            )

        try:
            generator = holder.get()
        except RuntimeError as exc:
            raise HTTPException(
                status_code=503,
                detail=f"模型权重未就绪: {exc}。请先跑 03/04 训练脚本。",
            ) from exc

        from ttm.export.retarget import build_plan, retarget_motion, write_motion_json, write_ue_bvh
        from ttm.infer.generate import GenerateOptions, shift_root_y  # noqa: PLC0415

        started = time.perf_counter()
        # guidance：请求缺省时回落到配置（生产 7.5）。绝不能在这里写死常量——
        # 早先 pydantic 默认 3.0 与配置 7.5 静默不一致，不传参的调用方会偷偷降质。
        guidance = (
            float(request.guidance)
            if request.guidance is not None
            else float(active.runtime.guidance)
        )
        # prompt 长度软拦截：>40 字超出训练分布（PRD 标注），只警告不拒绝请求
        warnings: list[str] = prompt_warnings(prompt)
        options = GenerateOptions(
            duration_s=request.duration_s,
            steps=request.steps,
            guidance=guidance,
            smooth=request.smooth,
            seed=request.seed,
            export_bvh=False,  # 由本层统一导出（要额外写 UE 骨架）
            out_dir=out_dir.as_posix(),
            # F-17：丢弃开头瞬态帧（生成时已多申请同样帧数，用户时长不变）
            trim_head_frames=int(active.runtime.trim_head_frames),
        )
        motion = generator.generate(prompt, options)
        export_started = time.perf_counter()

        plan = build_plan(fold_spine3=active.retarget.spine_merge != "split")
        rotations, positions = retarget_motion(motion.quats, motion.root_pos, plan=plan)
        # ground_origin：BVH 侧整体平移（UE 演示从地面开始）；JSON 保留世界坐标，
        # offset 写进 meta 由前端/UE 自己决定是否应用。retarget 不改根位移，
        # 所以源骨架上算出的 offset 对重定向后的 positions 同样成立。
        ground_applied = bool(request.ground_origin)
        ground_offset = float(motion.ground_offset_cm)
        positions_bvh = shift_root_y(positions, ground_offset) if ground_applied else positions
        stem = f"{_slug(prompt)}_{int(time.time() * 1000) % 100000}"

        json_url = ""
        bvh_url = ""
        payload: dict[str, Any] | None = None
        if request.export in ("json", "both"):
            target = write_motion_json(
                out_dir / f"{stem}.json",
                rotations,
                positions,
                plan=plan,
                fps=C.FPS,
                prompt=prompt,
                extra={
                    "n_windows": motion.n_windows,
                    "seam_jumps": [float(item) for item in motion.seam_jumps],
                    # F-17 裁头留痕：JSON 要能自证「这份产物裁过多少帧」
                    "trim_head_frames": int(motion.meta.get("trim_head_frames", 0)),
                    "frames_before_trim": int(
                        motion.meta.get("frames_before_trim", motion.n_frames)
                    ),
                    # F-17b：根高振荡诊断（KNOWN_LIMITATION 的判定依据，必须随产物落盘）
                    "root_y_diagnostics": motion.meta.get("root_y_diagnostics", {}),
                    "ground_origin": {
                        "applied": ground_applied,
                        "offset_cm": round(ground_offset, 3),
                    },
                },
            )
            json_url = f"/motions/{target.stem}.json"
            if request.export == "json":
                import json as _json  # noqa: PLC0415

                payload = _json.loads(target.read_text(encoding="utf-8"))
        if request.export in ("bvh", "both"):
            target = write_ue_bvh(out_dir / f"{stem}.bvh", rotations, positions_bvh, plan=plan)
            bvh_url = f"/motions/{target.stem}.bvh"

        export_ms = (time.perf_counter() - export_started) * 1000.0
        timing = dict(motion.timing_ms)
        timing["export_ms"] = round(export_ms, 2)
        timing["total_ms"] = round(
            float(timing.get("total_ms", 0.0)) + export_ms, 2
        )
        log.info(
            "生成成功: %r → %d 帧 / %d 窗 / %.0f ms（导出 %.1f ms）",
            prompt,
            motion.n_frames,
            motion.n_windows,
            timing["total_ms"],
            export_ms,
        )
        del started
        return GenerateResponse(
            prompt=prompt,
            n_frames=int(motion.n_frames),
            n_windows=int(motion.n_windows),
            duration_s=round(motion.n_frames / float(C.FPS), 3),
            fps=C.FPS,
            seam_jumps=[float(item) for item in motion.seam_jumps],
            timing=TimingInfo(**{key: float(value) for key, value in timing.items()}),
            motion=payload,
            json_url=json_url,
            bvh_url=bvh_url,
            ground_origin_applied=ground_applied,
            ground_origin_offset_cm=round(ground_offset, 3),
            guidance=guidance,
            warnings=warnings,
            trim_head_frames=int(motion.meta.get("trim_head_frames", 0)),
            root_y_diagnostics=dict(motion.meta.get("root_y_diagnostics") or {}),
        )

    @app.get("/motions/{filename}")
    def download(filename: str) -> FileResponse:
        """下载生成产物（只允许 .json / .bvh，且必须落在输出目录内）。"""
        if "/" in filename or "\\" in filename or ".." in filename:
            raise HTTPException(status_code=400, detail="非法文件名")
        suffix = Path(filename).suffix.lower()
        if suffix not in (".json", ".bvh"):
            raise HTTPException(status_code=400, detail="只允许 .json / .bvh")
        target = out_dir / filename
        if not target.is_file():
            raise HTTPException(status_code=404, detail=f"文件不存在: {filename}")
        media = "application/json" if suffix == ".json" else "text/plain"
        return FileResponse(target, media_type=media, filename=filename)

    @app.get("/motions")
    def list_motions(limit: int = Query(20, ge=1, le=200)) -> dict[str, Any]:
        """列出最近的生成产物。"""
        files = sorted(
            (item for item in out_dir.iterdir() if item.suffix.lower() in (".json", ".bvh")),
            key=lambda item: item.stat().st_mtime,
            reverse=True,
        )[:limit]
        return {
            "files": [
                {
                    "name": item.name,
                    "url": f"/motions/{item.name}",
                    "bytes": item.stat().st_size,
                    "mtime": item.stat().st_mtime,
                }
                for item in files
            ]
        }

    @app.post("/reload")
    def reload_generator() -> dict[str, Any]:
        """清空缓存的生成器（换权重后下一次请求会重新加载）。"""
        holder.reset()
        return {"status": "reset"}

    # -- 角色资源（FBX 原样透传，前端自己解析 rest pose）--------------------
    def _character_manifest() -> dict[str, dict[str, Any]]:
        """读 ``characters.json``（F-18 清单：has_bones / bone_names）；缺省为空表。"""
        import json as _json  # noqa: PLC0415

        manifest_path = _PROJECT_ROOT / "characters.json"
        if not manifest_path.is_file():
            return {}
        try:
            data = _json.loads(manifest_path.read_text(encoding="utf-8"))
            return {
                str(item.get("name")): item
                for item in data.get("characters", [])
                if isinstance(item, dict) and item.get("name")
            }
        except (ValueError, OSError):
            return {}

    @app.get("/characters")
    def characters() -> dict[str, Any]:
        """列出项目根目录下可用的角色 FBX（合并 manifest 的 has_bones/note）。"""
        items = sorted(_PROJECT_ROOT.glob("*.FBX")) + sorted(_PROJECT_ROOT.glob("*.fbx"))
        manifest = _character_manifest()
        enriched: list[dict[str, Any]] = []
        for item in items:
            entry: dict[str, Any] = {
                "name": item.name,
                "url": f"/characters/{item.name}",
                "bytes": item.stat().st_size,
            }
            declared = manifest.get(item.name)
            if declared is not None:
                entry["has_bones"] = bool(declared.get("has_bones", True))
                entry["bone_names"] = list(declared.get("bone_names", []))
                if declared.get("note"):
                    entry["note"] = str(declared["note"])
            else:
                # 无清单条目：按文件名约定兜底（*_no_bones* → 无骨骼）
                entry["has_bones"] = "no_bones" not in item.stem.lower()
            enriched.append(entry)
        return {
            "default": CHARACTER_FBX.name if CHARACTER_FBX.is_file() else "",
            "items": enriched,
        }

    @app.get("/characters/{filename}")
    def character_file(filename: str) -> FileResponse:
        """下载角色 FBX（限制在项目根目录内的 ``*.fbx``）。"""
        if "/" in filename or "\\" in filename or ".." in filename:
            raise HTTPException(status_code=400, detail="非法文件名")
        if Path(filename).suffix.lower() != ".fbx":
            raise HTTPException(status_code=400, detail="只允许 .fbx")
        target = _PROJECT_ROOT / filename
        if not target.is_file():
            raise HTTPException(status_code=404, detail=f"文件不存在: {filename}")
        return FileResponse(target, media_type="application/octet-stream", filename=filename)

    # -- 前端静态页面 -------------------------------------------------------
    if WEB_DIR.is_dir():
        app.mount("/", StaticFiles(directory=str(WEB_DIR), html=True), name="web")
    else:
        log.warning("前端目录不存在，跳过静态挂载: %s", WEB_DIR)

    log.info(
        "服务已就绪: %s:%s | 输出目录 %s | 前端 %s",
        active.serve.host,
        active.serve.port,
        out_dir,
        WEB_DIR if WEB_DIR.is_dir() else "(缺失)",
    )
    return app


# ---------------------------------------------------------------------------
# 工具
# ---------------------------------------------------------------------------
def _vae_ckpt(cfg: TTMConfig) -> Path:
    """VAE checkpoint 路径（空则按档位推导）。"""
    return Path(cfg.runtime.vae_ckpt) if cfg.runtime.vae_ckpt else cfg.run_dir("vae") / "best.pt"


def _diffusion_ckpt(cfg: TTMConfig) -> Path:
    """扩散 checkpoint 路径（空则按档位推导）。"""
    return (
        Path(cfg.runtime.diffusion_ckpt)
        if cfg.runtime.diffusion_ckpt
        else cfg.run_dir("diffusion") / "best.pt"
    )


def _weight_status(path: Path) -> dict[str, Any]:
    """把权重路径转成状态字典。"""
    exists = path.exists()
    size = 0
    if exists and path.is_file():
        size = path.stat().st_size
    return {"path": path.as_posix(), "exists": bool(exists), "bytes": int(size)}


def _slug(text: str, *, limit: int = 16) -> str:
    """把中文指令转成安全文件名片段。"""
    cleaned = "".join(ch if ch.isalnum() else "_" for ch in str(text).strip())
    collapsed = "_".join(part for part in cleaned.split("_") if part)
    return (collapsed or "motion")[:limit]


def main(argv: list[str] | None = None) -> int:
    """命令行入口（等价于 ``uvicorn server.app:app``）。

    Args:
        argv: 参数列表（默认取 ``sys.argv[1:]``）。

    Returns:
        进程退出码。
    """
    parser = argparse.ArgumentParser(
        description="启动 TTM 推理服务", formatter_class=argparse.ArgumentDefaultsHelpFormatter
    )
    parser.add_argument("--config", default=DEFAULT_CONFIG, help="配置文件路径")
    parser.add_argument("--host", default="", help="覆盖 serve.host")
    parser.add_argument("--port", type=int, default=0, help="覆盖 serve.port")
    args = parser.parse_args(argv)

    cfg = load_config(args.config)
    host = args.host or cfg.serve.host
    port = args.port or cfg.serve.port
    application = create_app(cfg)
    try:
        import uvicorn  # noqa: PLC0415
    except Exception as exc:  # noqa: BLE001
        get_logger("ttm.server").error("uvicorn 不可用: %s", exc)
        close_all()
        return 2

    print(f"前端页面: http://{host}:{port}/")
    print(f"接口文档: http://{host}:{port}/docs")
    uvicorn.run(application, host=host, port=port, log_level="info")
    close_all()
    return 0


app = create_app()

if __name__ == "__main__":
    raise SystemExit(main())
