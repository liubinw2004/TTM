"""T13：端到端冒烟 —— 权重 → 生成 → 重定向 → 导出 → 服务 → UE 打包。

一条命令验证整条链路（不碰 ``Z:``，只读 ``D:/TTM``）：:

    . D:/TTM/scripts/env.ps1
    python scripts/06_smoke_e2e.py --config configs/tier_s.yaml
    python scripts/06_smoke_e2e.py --config configs/tier_s.yaml --write-perf

检查项（每项标 PASS / SKIP / FAIL）：

    1. 资产就位      VAE / DiT / norm_stats / bge / FBX
    2. 生成器加载    文本编码 → 采样 → VAE 解码 → 融合
    3. 动作生成      单窗 + 长动作（多窗，验证接缝台阶有界）
    4. BVH 导出      自解析器回读，帧数/骨数/欧拉角一致
    5. 重定向        Spine3 折叠，UE 骨名 BVH 回读（非严格模式）
    6. Motion JSON   结构 + 四元数归一化 + 拓扑序
    7. 服务接口      /health、/prompts、/characters、POST /generate、/motions
    8. UE 打包       tools/ue_import.py 校验与脚本生成

``--write-perf`` 会把实测数字（参数量 / 时延分解 / 峰值显存 / 数据规模）
汇总成 ``deliverables/perf_report.md``。

退出码：全通过（允许 SKIP）0；有 FAIL 1。
"""

from __future__ import annotations

import argparse
import json
import sys
import time
import traceback
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Callable

# --- sys.path 引导 -----------------------------------------------------------
_PROJECT_ROOT = Path(__file__).resolve().parents[1]
_SRC_DIR = _PROJECT_ROOT / "src"
for _candidate in (str(_SRC_DIR), str(_PROJECT_ROOT)):
    if _candidate not in sys.path:
        sys.path.insert(0, _candidate)

from ttm.common import constants as C  # noqa: E402
from ttm.common import paths as P  # noqa: E402
from ttm.common.config import load_config  # noqa: E402
from ttm.common.logutil import close_all, get_logger  # noqa: E402

#: FBX 角色（前端播放用；缺失只影响 Web 演示）。
CHARACTER_FBX = _PROJECT_ROOT / "SK_Celeste_02.FBX"

#: 各类检查的统计容器。
STATUS_PASS = "PASS"
STATUS_SKIP = "SKIP"
STATUS_FAIL = "FAIL"


@dataclass
class CheckResult:
    """单项检查结果。"""

    name: str
    status: str
    detail: str = ""
    elapsed_ms: float = 0.0
    data: dict[str, Any] = field(default_factory=dict)


class SmokeRunner:
    """顺序执行检查项，收集结构化结果。"""

    def __init__(self) -> None:
        self.results: list[CheckResult] = []
        self.log = get_logger("ttm.smoke", script="06_smoke_e2e")

    def run(self, name: str, func: Callable[[], dict[str, Any] | None]) -> CheckResult:
        """执行一项检查。

        Args:
            name: 检查名。
            func: 无参可调用；返回值会并入 ``data``。

        Returns:
            :class:`CheckResult`。
        """
        started = time.perf_counter()
        try:
            data = func() or {}
            elapsed = (time.perf_counter() - started) * 1000.0
            result = CheckResult(name, STATUS_PASS, "", elapsed, dict(data))
        except _Skip as exc:
            elapsed = (time.perf_counter() - started) * 1000.0
            result = CheckResult(name, STATUS_SKIP, str(exc), elapsed)
        except Exception as exc:  # noqa: BLE001 - 冒烟要把所有异常转成 FAIL 而不是崩掉
            elapsed = (time.perf_counter() - started) * 1000.0
            result = CheckResult(
                name,
                STATUS_FAIL,
                f"{type(exc).__name__}: {exc}\n{traceback.format_exc(limit=3)}",
                elapsed,
            )
        self.results.append(result)
        icon = {STATUS_PASS: "[通过]", STATUS_SKIP: "[跳过]", STATUS_FAIL: "[失败]"}[result.status]
        line = f"{icon} {name:<18} {elapsed:>8.0f} ms"
        if result.detail:
            first = result.detail.strip().splitlines()[0]
            line += f"  {first}"
        print(line)
        return result

    @property
    def failed(self) -> list[CheckResult]:
        """所有 FAIL 项。"""
        return [item for item in self.results if item.status == STATUS_FAIL]

    def to_dict(self) -> dict[str, Any]:
        """导出为 JSON 友好结构。"""
        return {
            "generated_at": time.strftime("%Y-%m-%d %H:%M:%S"),
            "total": len(self.results),
            "passed": sum(1 for item in self.results if item.status == STATUS_PASS),
            "skipped": sum(1 for item in self.results if item.status == STATUS_SKIP),
            "failed": len(self.failed),
            "checks": [
                {
                    "name": item.name,
                    "status": item.status,
                    "detail": item.detail,
                    "elapsed_ms": round(item.elapsed_ms, 2),
                    "data": item.data,
                }
                for item in self.results
            ],
        }


class _Skip(Exception):
    """检查项前置条件不满足（不是失败）。"""


# ---------------------------------------------------------------------------
# 检查实现
# ---------------------------------------------------------------------------
def check_assets(cfg) -> dict[str, Any]:
    """1. 资产就位检查。"""
    vae = Path(cfg.runtime.vae_ckpt) if cfg.runtime.vae_ckpt else cfg.run_dir("vae") / "best.pt"
    dit = (
        Path(cfg.runtime.diffusion_ckpt)
        if cfg.runtime.diffusion_ckpt
        else cfg.run_dir("diffusion") / "best.pt"
    )
    stats = cfg.norm_stats_path()
    text_model = Path(cfg.text.model_path)
    assets = {
        "vae_ckpt": vae,
        "diffusion_ckpt": dit,
        "norm_stats": stats,
        "text_model": text_model,
        "character_fbx": CHARACTER_FBX,
    }
    table = {
        name: {"path": path.as_posix(), "exists": path.exists()}
        for name, path in assets.items()
    }
    blocking = [name for name, item in table.items() if not item["exists"] and name != "character_fbx"]
    if blocking:
        raise _Skip(f"缺少关键产物 {blocking}（FBX 只影响 Web 演示）")
    return {"assets": table}


def _build_generator(cfg, log):
    """加载生成器（缓存到函数属性上，供后续检查复用）。"""
    if getattr(_build_generator, "_cached", None) is not None:
        return _build_generator._cached  # type: ignore[attr-defined]
    from ttm.infer.generate import load_generator  # noqa: PLC0415

    started = time.perf_counter()
    generator = load_generator(cfg, logger=log)
    elapsed = (time.perf_counter() - started) * 1000.0
    _build_generator._cached = generator  # type: ignore[attr-defined]
    _build_generator._load_ms = elapsed  # type: ignore[attr-defined]
    return generator


def check_generator(cfg, log) -> dict[str, Any]:
    """2. 生成器加载。"""
    generator = _build_generator(cfg, log)
    vae_params = sum(p.numel() for p in generator.vae.parameters())
    dit_params = sum(p.numel() for p in generator.dit.parameters())
    return {
        "device": str(generator.device),
        "vae_params": int(vae_params),
        "dit_params": int(dit_params),
        "load_ms": round(float(getattr(_build_generator, "_load_ms", 0.0)), 1),
    }


#: 性能测量的重复次数（取中位数）。
#:
#: 为什么要重复：同一份权重、同一条指令，实测总时延在 **2045ms ~ 3330ms** 之间波动
#: （GPU 时钟爬升 + 其他进程占用）。单次测量写进性能报告等于把「这一秒钟的机器状态」
#: 当成了模型指标，换个时间跑差 50%，评审没法用它做判断。
PERF_REPEATS = 3


def _median_timing(records: list[dict[str, float]]) -> dict[str, float]:
    """多轮时延分解逐项取中位数（并保留极值，便于看出波动幅度）。"""
    if not records:
        return {}
    keys = {key for record in records for key in record}
    result: dict[str, float] = {}
    for key in keys:
        values = sorted(float(record[key]) for record in records if key in record)
        if not values:
            continue
        middle = len(values) // 2
        median = values[middle] if len(values) % 2 else (values[middle - 1] + values[middle]) / 2.0
        result[key] = round(float(median), 1)
    return result


def _timing_spread(records: list[dict[str, float]]) -> dict[str, list[float]]:
    """每一段的最快/最慢（写进报告，让人知道中位数背后有多大的抖动）。"""
    if not records:
        return {}
    keys = {key for record in records for key in record}
    return {
        key: [
            round(min(float(record[key]) for record in records if key in record), 1),
            round(max(float(record[key]) for record in records if key in record), 1),
        ]
        for key in keys
        if any(key in record for record in records)
    }


def _startup_root_jump(root_pos, *, window: int = 12) -> float:
    """F-17 启动瞬态指标：前 ``window`` 帧与随后 ``window`` 帧的根 Y 中位数差（cm）。

    ground_origin 是常量平移，不改变中位数差，所以这条指标在裁头前后都可直接用。

    Args:
        root_pos: ``[T,3]`` 根位移（cm）。
        window: 两段的帧数（QA 判据取 12）。

    Returns:
        中位数差绝对值（cm）；帧数不足时返回 0。
    """
    import numpy as np  # noqa: PLC0415

    array = np.asarray(root_pos, dtype=np.float64)
    if array.ndim != 2 or array.shape[0] < 2 * window:
        return 0.0
    head = float(np.median(array[:window, 1]))
    tail = float(np.median(array[window : 2 * window, 1]))
    return abs(tail - head)


def check_generate(cfg, log) -> dict[str, Any]:
    """3. 动作生成（单窗 + 长动作）。"""
    from ttm.infer.generate import GenerateOptions  # noqa: PLC0415

    generator = _build_generator(cfg, log)
    short_runs: list[dict[str, float]] = []
    long_runs: list[dict[str, float]] = []
    short = None
    long_motion = None
    window = int(cfg.data.window)
    long_duration = min(12.0, float(cfg.serve.max_duration_s))
    # guidance / trim_head 一律从配置取：冒烟测的必须是「随配置发货」的那套参数，
    # 在脚本里写死常量就等于测了另一条路径（早先 guidance 写死 3.0 vs 配置 7.5）。
    guidance = float(cfg.runtime.guidance)
    trim_head = int(cfg.runtime.trim_head_frames)
    for _ in range(max(1, PERF_REPEATS)):
        short = generator.generate(
            "慢慢走上楼梯",
            GenerateOptions(
                duration_s=5.0,
                steps=20,
                seed=42,
                guidance=guidance,
                trim_head_frames=trim_head,
                export_bvh=False,
            ),
        )
        short_runs.append(dict(short.timing_ms))
        long_motion = generator.generate(
            "打一套陈式太极拳，动作舒缓连贯",
            GenerateOptions(
                duration_s=long_duration,
                steps=20,
                seed=43,
                guidance=guidance,
                trim_head_frames=trim_head,
                export_bvh=False,
            ),
        )
        long_runs.append(dict(long_motion.timing_ms))
    assert short is not None and long_motion is not None

    # 形状与数值健全性
    import numpy as np  # noqa: PLC0415

    for label, motion in (("short", short), ("long", long_motion)):
        if motion.quats.shape[1:] != (C.N_JOINTS, 4):
            raise AssertionError(f"{label} 四元数形状异常: {motion.quats.shape}")
        norms = np.linalg.norm(motion.quats, axis=-1)
        worst = float(np.abs(norms - 1.0).max())
        if worst > 1e-3:
            raise AssertionError(f"{label} 四元数未归一化，最大偏差 {worst:.4f}")
        if not np.all(np.isfinite(motion.root_pos)):
            raise AssertionError(f"{label} 根位移含 NaN/Inf")

    if long_motion.n_windows < 2:
        raise AssertionError(
            f"长动作（{long_duration}s）未触发滑窗：只有 {long_motion.n_windows} 个窗口"
        )
    # F-17 帧数不变式：裁头只丢开头，**不能改变用户侧时长**
    expected_short = int(round(5.0 * C.FPS))
    if int(short.n_frames) != expected_short:
        raise AssertionError(
            f"裁头后帧数 {short.n_frames} != 请求时长对应帧数 {expected_short}"
        )
    # F-17 启动瞬态：前 12 帧 vs 后 12 帧的根高中位数差（ground_origin 是常量平移，
    # 不影响中位数差）。QA 判据 30cm；GT 同口径 p99=0.58cm。
    startup_jump = _startup_root_jump(short.root_pos)
    if startup_jump > 30.0:
        raise AssertionError(
            f"启动瞬态未消除: |median(f0..f11)-median(f12..f23)| = {startup_jump:.2f}cm > 30cm"
        )
    seam = float(long_motion.seam_jumps.max()) if long_motion.seam_jumps.size else 0.0
    # 随机/欠训练的模型也可能有较大台阶，这里只卡「没有爆炸」
    if seam > 2000.0:
        raise AssertionError(f"接缝台阶失控: {seam:.1f} cm/s")

    return {
        "guidance": guidance,
        "trim_head_frames": trim_head,
        "startup_jump_cm": round(float(startup_jump), 3),
        "short": {
            "n_frames": int(short.n_frames),
            "n_windows": int(short.n_windows),
            "timing_ms": _median_timing(short_runs),
            "timing_spread_ms": _timing_spread(short_runs),
            "repeats": len(short_runs),
        },
        "long": {
            "n_frames": int(long_motion.n_frames),
            "n_windows": int(long_motion.n_windows),
            "seam_max_cmps": round(seam, 3),
            "timing_ms": _median_timing(long_runs),
            "timing_spread_ms": _timing_spread(long_runs),
            "repeats": len(long_runs),
        },
        "window": window,
        "_motions": {"short": short, "long": long_motion},
    }


def check_bvh_export(work: Path) -> dict[str, Any]:
    """4. BVH 导出 + 自解析器回读。"""
    motions = getattr(check_generate, "_motions", None)
    if motions is None:
        raise _Skip("生成步骤未完成")

    import numpy as np  # noqa: PLC0415

    from ttm.data.bvh import BvhParser  # noqa: PLC0415
    from ttm.export.bvh_io import motion_to_euler_deg, write_bvh  # noqa: PLC0415

    motion = motions["long"]
    path = write_bvh(work / "smoke_internal.bvh", motion.quats, motion.root_pos, fps=C.FPS)
    document = BvhParser.parse(path)
    if document.n_frames != motion.n_frames:
        raise AssertionError(f"帧数不一致: {document.n_frames} != {motion.n_frames}")
    if document.n_joints != C.N_JOINTS:
        raise AssertionError(f"关节数不一致: {document.n_joints} != {C.N_JOINTS}")
    expected = motion_to_euler_deg(motion.quats)
    if not np.allclose(document.euler_deg, expected, atol=1e-4):
        raise AssertionError("欧拉角往返不一致")
    if not np.allclose(document.root_pos_cm, motion.root_pos, atol=1e-4):
        raise AssertionError("根位移往返不一致")
    return {
        "path": path.as_posix(),
        "bytes": path.stat().st_size,
        "n_frames": int(document.n_frames),
        "n_joints": int(document.n_joints),
    }


def check_retarget(work: Path, cfg) -> dict[str, Any]:
    """5. 重定向（Spine3 折叠）+ UE 骨名 BVH 回读。"""
    motions = getattr(check_generate, "_motions", None)
    if motions is None:
        raise _Skip("生成步骤未完成")

    from ttm.data.bvh import BvhParser  # noqa: PLC0415
    from ttm.export.retarget import build_plan, retarget_motion, write_ue_bvh  # noqa: PLC0415

    fold = str(getattr(cfg.retarget, "spine_merge", "fold")).lower() != "split"
    plan = build_plan(fold_spine3=fold)
    if plan.spine3_folded != fold:
        raise AssertionError(f"折叠策略不符: {plan.spine3_folded} != {fold}")
    if plan.parents != tuple(C.UE_PARENTS) and fold:
        raise AssertionError("折叠后的父子关系与 UE_PARENTS 不一致")

    motion = motions["long"]
    rotations, positions = retarget_motion(motion.quats, motion.root_pos, plan=plan)
    path = write_ue_bvh(work / "smoke_ue.bvh", rotations, positions, plan=plan)
    document = BvhParser.parse(path, strict=False)
    if document.n_joints != plan.n_bones:
        raise AssertionError(f"UE 骨数不一致: {document.n_joints} != {plan.n_bones}")
    if tuple(document.skeleton.bone_names) != plan.bone_names:
        raise AssertionError("UE 骨名顺序不一致")
    if tuple(int(item) for item in document.skeleton.parents) != plan.parents:
        raise AssertionError("UE 父子关系不一致")
    return {
        "path": path.as_posix(),
        "n_bones": int(plan.n_bones),
        "spine3_folded": bool(plan.spine3_folded),
        "bytes": path.stat().st_size,
    }


def check_motion_json(work: Path) -> dict[str, Any]:
    """6. Motion JSON 结构与归一化。"""
    motions = getattr(check_generate, "_motions", None)
    if motions is None:
        raise _Skip("生成步骤未完成")

    from ttm.export.retarget import build_plan, retarget_motion, write_motion_json  # noqa: PLC0415
    from tools.ue_import import validate_motion  # noqa: PLC0415

    motion = motions["short"]
    plan = build_plan()
    rotations, positions = retarget_motion(motion.quats, motion.root_pos, plan=plan)
    # ground_origin（M4 新增）：JSON 保留世界坐标，offset 写进 meta
    ground_offset = float(motion.ground_offset_cm)
    path = write_motion_json(
        work / "smoke.json",
        rotations,
        positions,
        plan=plan,
        prompt="慢慢走上楼梯",
        extra={
            "seam_jumps": [float(item) for item in motion.seam_jumps],
            "ground_origin": {"applied": True, "offset_cm": round(ground_offset, 3)},
        },
    )
    payload = json.loads(path.read_text(encoding="utf-8"))
    bundle = validate_motion(payload, source=path.as_posix())
    ground_meta = payload.get("ground_origin") or {}
    if ground_meta.get("applied") is not True:
        raise AssertionError("Motion JSON 缺少 ground_origin.applied=true")
    if not isinstance(ground_meta.get("offset_cm"), (int, float)):
        raise AssertionError("Motion JSON 缺少 ground_origin.offset_cm")
    # 世界坐标不变式：JSON 里的首帧根高必须等于原始 root_pos（没被平移过）
    if abs(payload["root_positions_cm"][0][1] - float(motion.root_pos[0, 1])) > 1e-6:
        raise AssertionError("Motion JSON 的 root Y 被平移了（应保留世界坐标）")
    # F-07：半球规约后 qw<0 必须为 0（否则 UE/前端插值会走长弧）
    import numpy as np  # noqa: PLC0415

    negative_w = sum(
        int(np.count_nonzero(np.asarray(track["rotations_xyzw"], dtype=np.float64)[:, 3] < 0.0))
        for track in payload["tracks"]
    )
    if negative_w:
        raise AssertionError(f"Motion JSON 仍有 {negative_w} 个 qw<0（半球规约失效）")
    return {
        "path": path.as_posix(),
        "bytes": path.stat().st_size,
        "n_frames": bundle.n_frames,
        "n_bones": len(bundle.bone_names),
        "ground_origin_offset_cm": round(ground_offset, 2),
        # 与 Motion JSON 内的 meta 同形，方便 QA 直接在冒烟产物里核对是否落盘
        "ground_origin": {"applied": True, "offset_cm": round(ground_offset, 3)},
        "negative_w_count": negative_w,
    }


def check_server(cfg, work: Path) -> dict[str, Any]:
    """7. FastAPI 接口（TestClient，不起真实端口）。"""
    motions = getattr(check_generate, "_motions", None)
    if motions is None:
        raise _Skip("生成步骤未完成")
    try:
        from fastapi.testclient import TestClient  # noqa: PLC0415
    except Exception as exc:  # noqa: BLE001
        raise _Skip(f"TestClient 不可用（缺 httpx）: {exc}") from exc

    from server.app import create_app  # noqa: PLC0415

    app = create_app(cfg)
    with TestClient(app) as client:
        health = client.get("/health")
        if health.status_code != 200:
            raise AssertionError(f"/health 返回 {health.status_code}")
        health_body = health.json()
        prompts = client.get("/prompts")
        if prompts.status_code != 200 or not prompts.json().get("prompts"):
            raise AssertionError("/prompts 异常")
        characters = client.get("/characters")
        if characters.status_code != 200:
            raise AssertionError("/characters 异常")

        response = client.post(
            "/generate",
            json={
                "prompt": "挥手打招呼",
                "duration_s": 5.0,
                "steps": 12,
                # 不写死：冒烟必须测「随配置发货」的 guidance（现在是 sweep 定的 7.5）
                "guidance": float(cfg.runtime.guidance),
                "smooth": 0.2,
                "seed": 7,
                "export": "both",
            },
        )
        if response.status_code != 200:
            raise AssertionError(f"POST /generate 返回 {response.status_code}: {response.text[:300]}")
        body = response.json()
        for key in ("n_frames", "n_windows", "timing", "json_url", "bvh_url"):
            if key not in body:
                raise AssertionError(f"响应缺少字段 {key}")
        # 必修 3：响应必须回显**实际使用**的 guidance（配置回落后的值）
        if abs(float(body.get("guidance", -1.0)) - float(cfg.runtime.guidance)) > 1e-9:
            raise AssertionError(
                f"响应 guidance={body.get('guidance')} 与配置 {cfg.runtime.guidance} 不一致"
            )
        if "warnings" not in body:
            raise AssertionError("响应缺少 warnings 字段")
        # 建议 4：短指令（训练分布内）不该触发软警告
        if body["warnings"]:
            raise AssertionError(f"短指令不应有 warnings: {body['warnings']}")
        # F-17b：振荡诊断必须随响应返回（QA 靠它判 KNOWN_LIMITATION）
        diagnostics = body.get("root_y_diagnostics") or {}
        for key in ("peak_to_peak_cm", "detrended_peak_to_peak_cm", "sign_changes", "local_jump_cm", "oscillating"):
            if key not in diagnostics:
                raise AssertionError(f"响应 root_y_diagnostics 缺少 {key}")
        if int(body["n_frames"]) != int(round(5.0 * C.FPS)):
            raise AssertionError(
                f"裁头后帧数 {body['n_frames']} != 请求时长对应帧数 {int(round(5.0 * C.FPS))}"
            )
        for key in ("text_ms", "sample_ms", "decode_ms", "blend_ms", "post_ms", "export_ms"):
            if key not in body["timing"]:
                raise AssertionError(f"timing 缺少 {key}")

        download = client.get(body["json_url"])
        if download.status_code != 200 or not download.content:
            raise AssertionError(f"下载 {body['json_url']} 失败")
        listing = client.get("/motions")
        if listing.status_code != 200:
            raise AssertionError("/motions 列表异常")
        bad = client.get("/motions/..%2Fsecret.json")
        if bad.status_code == 200:
            raise AssertionError("路径穿越未被拦截")

    return {
        "health_status": health_body.get("status"),
        "device": health_body.get("device"),
        "n_frames": body["n_frames"],
        "n_windows": body["n_windows"],
        "guidance": body.get("guidance"),
        "warnings": body.get("warnings", []),
        "root_y_diagnostics": diagnostics,
        "timing": body["timing"],
        "json_url": body["json_url"],
        "bvh_url": body["bvh_url"],
        "n_files": len(listing.json().get("files", [])),
    }


def check_ue_bundle(work: Path) -> dict[str, Any]:
    """8. UE 打包（tools/ue_import.py 的校验 + 脚本生成）。"""
    from tools.ue_import import (  # noqa: PLC0415
        build_ue_script,
        load_motion_json,
        validate_motion,
        write_bvh_from_json,
    )

    motion_path = work / "smoke.json"
    if not motion_path.is_file():
        raise _Skip("Motion JSON 未生成")
    bundle = validate_motion(load_motion_json(motion_path), source=motion_path.as_posix())
    out_dir = P.safe_makedirs(work / "ue_bundle")
    bvh_path = write_bvh_from_json(bundle, out_dir / f"{motion_path.stem}.bvh")
    script = build_ue_script(bvh_files=[bvh_path.as_posix()])
    script_path = out_dir / "import_unreal.py"
    script_path.write_text(script, encoding="utf-8")
    if "AssetImportTask" not in script:
        raise AssertionError("UE 脚本缺少 AssetImportTask")
    return {
        "bvh": bvh_path.as_posix(),
        "script": script_path.as_posix(),
        "n_bones": len(bundle.bone_names),
        "n_frames": bundle.n_frames,
    }


def check_dataset(cfg) -> dict[str, Any]:
    """附加项：数据规模（分片数 / 窗口数 / 标签数）。"""
    shards_dir = cfg.shards_path()
    if not shards_dir.is_dir():
        raise _Skip(f"分片目录不存在: {shards_dir}")
    shards = sorted(shards_dir.glob("*.npz"))
    if not shards:
        raise _Skip("分片目录为空")

    import numpy as np  # noqa: PLC0415

    n_windows = 0
    n_valid = 0
    for shard in shards:
        with np.load(shard) as data:
            n_windows += int(data["feat"].shape[0])
            n_valid += int(data["mask"].sum())
    labels = 0
    if cfg.labels_path().is_file():
        labels = sum(
            1
            for line in cfg.labels_path().read_text(encoding="utf-8").splitlines()
            if line.strip()
        )
    return {
        "n_shards": len(shards),
        "n_windows": n_windows,
        "n_valid_frames": n_valid,
        "n_labels": labels,
        "bytes": sum(item.stat().st_size for item in shards),
    }


# ---------------------------------------------------------------------------
# 性能报告
# ---------------------------------------------------------------------------
def write_perf_report(runner: SmokeRunner, cfg, path: Path) -> Path:
    """把实测结果写成 Markdown 性能报告。

    Args:
        runner: 已跑完的检查集合。
        cfg: 生效配置。
        path: 输出路径。

    Returns:
        写出的路径。
    """
    by_name = {item.name: item for item in runner.results}
    dataset = by_name.get("数据规模")
    generator = by_name.get("生成器加载")
    generate = by_name.get("动作生成")
    server = by_name.get("服务接口")

    def timing_table(timing: dict[str, Any], spread: dict[str, Any] | None = None) -> str:
        rows = [
            ("文本编码 text", timing.get("text_ms"), "text_ms"),
            ("DDIM 采样 sample", timing.get("sample_ms"), "sample_ms"),
            ("VAE 解码 decode", timing.get("decode_ms"), "decode_ms"),
            ("窗口融合 blend", timing.get("blend_ms"), "blend_ms"),
            ("后处理 post", timing.get("post_ms"), "post_ms"),
            ("导出 export", timing.get("export_ms"), "export_ms"),
            ("**合计**", timing.get("total_ms"), "total_ms"),
        ]
        spread = spread or {}
        lines = ["| 阶段 | 中位数 (ms) | 最快~最慢 (ms) |", "| --- | ---: | ---: |"]
        for label, value, key in rows:
            if value is None:
                continue
            bounds = spread.get(key)
            span = f"{bounds[0]:.1f} ~ {bounds[1]:.1f}" if bounds else "–"
            lines.append(f"| {label} | {float(value):.1f} | {span} |")
        return "\n".join(lines)

    device = (generator.data or {}).get("device", "–") if generator else "–"
    vae_params = (generator.data or {}).get("vae_params", 0) if generator else 0
    dit_params = (generator.data or {}).get("dit_params", 0) if generator else 0
    load_ms = (generator.data or {}).get("load_ms", 0.0) if generator else 0.0

    short_timing = ((generate.data or {}).get("short") or {}).get("timing_ms", {}) if generate else {}
    long_timing = ((generate.data or {}).get("long") or {}).get("timing_ms", {}) if generate else {}
    short_spread = ((generate.data or {}).get("short") or {}).get("timing_spread_ms", {}) if generate else {}
    long_spread = ((generate.data or {}).get("long") or {}).get("timing_spread_ms", {}) if generate else {}
    long_info = (generate.data or {}).get("long") or {}
    server_timing = (server.data or {}).get("timing", {}) if server else {}

    dataset_line = "–"
    if dataset and dataset.data:
        mb = dataset.data.get("bytes", 0) / 1024 / 1024
        dataset_line = (
            f"{dataset.data.get('n_shards', 0)} 个分片 / "
            f"{dataset.data.get('n_windows', 0):,} 个窗口 / "
            f"{dataset.data.get('n_labels', 0):,} 条标签 / {mb:.0f} MB"
        )

    text = f"""# TTM 性能报告

> 由 `scripts/06_smoke_e2e.py --write-perf` 自动生成，数字均为本机实测。
> 生成时间：{time.strftime("%Y-%m-%d %H:%M:%S")}
> 同步产物：`runs/eval/smoke_e2e.json`（is_pass / guidance / 各检查项明细）。
> **核对同步**：本文件与该 JSON 必须是同一次运行的产物（时间戳应一致），
> 若 JSON 早于本文件说明冒烟没重跑，以 JSON 的 `is_pass` 为准。

## 1. 环境

| 项目 | 值 |
| --- | --- |
| 配置文件 | `{cfg.source_path}` |
| 设备 | `{device}` |
| 精度 | `{cfg.runtime.precision}` |
| 目标档位 | Tier-{cfg.data.tier} |
| 数据规模 | {dataset_line} |
| 生成器加载耗时 | {float(load_ms):.0f} ms |

## 2. 模型规模

| 模型 | 参数量 |
| --- | ---: |
| MotionVAE | {int(vae_params):,} |
| DiT（扩散） | {int(dit_params):,} |

骨架：内部 {C.N_JOINTS} 关节 / UE 侧 {len(C.UE_BONE_ORDER)} 骨；
特征 {C.FEATURE_DIM} 维；窗口 {C.WINDOW} 帧 @ {C.FPS}fps；重叠 {C.OVERLAP} 帧；
潜变量 {C.LATENT_DIM} 维 / {C.VAE_TOKEN_LEN} token。

## 3. 单窗推理（5 秒 = {C.WINDOW} 帧）

{timing_table(short_timing, short_spread) if short_timing else "（本次未跑）"}

## 4. 长动作推理（滑窗 {long_info.get("n_windows", "?")} 窗 / {long_info.get("n_frames", "?")} 帧）

{timing_table(long_timing, long_spread) if long_timing else "（本次未跑）"}

接缝根速度台阶（越大越不连续）：**{long_info.get("seam_max_cmps", "–")} cm/s**

## 5. 服务接口（含导出的端到端时延）

{timing_table(server_timing) if server_timing else "（本次未跑）"}

## 6. 冒烟结论

| 检查项 | 状态 | 耗时 (ms) |
| --- | --- | ---: |
""" + "\n".join(
        f"| {item.name} | {item.status} | {item.elapsed_ms:.0f} |" for item in runner.results
    ) + f"""

## 7. 已知取舍

* 采样器是 **DDIM η=0**（确定性），步数默认 50；步数降到 20 时质量略降但时延近似线性下降。
* 长动作按 `stride = window − overlap = 120` 滑窗，重叠区线性交叉淡化后**全局只积分一次**，
  因此根位移在接缝处连续（见第 4 节的台阶值）。
* CFG 训练期条件 dropout 10%，推理 `guidance={float(cfg.runtime.guidance)}`（四点 sweep
  定值，见 `runs/eval/guidance_sweep.json`）；`guidance=1.0` 等价于关闭引导。
* 启动瞬态裁头 `trim_head_frames={int(cfg.runtime.trim_head_frames)}`（F-17）：按「用户时长
  + 裁头帧数」采样、导出前丢掉开头，**用户侧时长不变**（5s 仍是 {C.WINDOW} 帧）；代价是
  5s 请求从 1 窗变 2 窗，采样时延约翻倍。本次实测启动瞬态
  `|median(f0..f11)−median(f12..f23)| = {float((generate.data or {}).get("startup_jump_cm", 0.0)):.2f} cm`
  （判据 ≤30cm）。
"""

    target = P.resolve_out(Path(path), mkparents=True)
    target.write_text(text, encoding="utf-8")
    return target


# ---------------------------------------------------------------------------
# 入口
# ---------------------------------------------------------------------------
def build_argparser() -> argparse.ArgumentParser:
    """构造命令行参数解析器。"""
    parser = argparse.ArgumentParser(
        description="TTM 端到端冒烟（T13）", formatter_class=argparse.ArgumentDefaultsHelpFormatter
    )
    parser.add_argument("--config", default="configs/tier_s.yaml", help="配置文件路径")
    parser.add_argument(
        "--work-dir", default=".cache/smoke", help="临时产物目录（相对项目根）"
    )
    parser.add_argument("--write-perf", action="store_true", help="额外写 deliverables/perf_report.md")
    parser.add_argument(
        "--perf-path", default="deliverables/perf_report.md", help="性能报告输出路径"
    )
    parser.add_argument(
        "--json",
        default="runs/eval/smoke_e2e.json",
        help="冒烟结果 JSON 输出路径（默认必写，避免 perf_report 与 smoke_e2e.json 时间戳脱钩）",
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
    log = get_logger("ttm.smoke", script="06_smoke_e2e")
    work = P.safe_makedirs(Path(args.work_dir))

    print("=" * 88)
    print(f"TTM 端到端冒烟 · 配置 {args.config} · 工作目录 {work.as_posix()}")
    print("=" * 88)

    runner = SmokeRunner()

    def _generate_wrapper() -> dict[str, Any]:
        data = check_generate(cfg, log)
        motions = data.pop("_motions", {})
        # 供后续检查复用（放模块级，避免重复生成）
        check_generate._motions = motions  # type: ignore[attr-defined]
        return data

    runner.run("资产就位", lambda: check_assets(cfg))
    assets_ok = runner.results[0].status == STATUS_PASS
    assets_reason = runner.results[0].detail or "前置资产未就位"

    def _needs_assets(func: Callable[[], Any]) -> Callable[[], Any]:
        """资产没就位时，依赖权重的检查一律跳过（不是失败）。"""

        def _inner() -> Any:
            if not assets_ok:
                raise _Skip(assets_reason)
            return func()

        return _inner

    runner.run("生成器加载", _needs_assets(lambda: check_generator(cfg, log)))
    runner.run("动作生成", _needs_assets(_generate_wrapper))
    runner.run("BVH 导出", lambda: check_bvh_export(work))
    runner.run("重定向", lambda: check_retarget(work, cfg))
    runner.run("Motion JSON", lambda: check_motion_json(work))
    runner.run("服务接口", _needs_assets(lambda: check_server(cfg, work)))
    runner.run("UE 打包", lambda: check_ue_bundle(work))
    runner.run("数据规模", lambda: check_dataset(cfg))

    print("-" * 88)
    summary = runner.to_dict()
    print(
        f"合计 {summary['total']} 项：通过 {summary['passed']} / 跳过 {summary['skipped']}"
        f" / 失败 {summary['failed']}"
    )

    if args.json:
        # QA 直接消费这份 JSON，所以把判定与关键配置一并写进去（自包含）
        summary = {
            **summary,
            "config": args.config,
            "guidance": float(cfg.runtime.guidance),
            "steps": int(cfg.diffusion.ddim_steps),
            "trim_head_frames": int(cfg.runtime.trim_head_frames),
            "is_pass": not runner.failed,
        }
        target = P.resolve_out(Path(args.json), mkparents=True)
        target.write_text(json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8")
        print(f"结果 JSON: {target.as_posix()}")

    if args.write_perf:
        report = write_perf_report(runner, cfg, Path(args.perf_path))
        print(f"性能报告: {report.as_posix()}")

    print("=" * 88)
    if runner.failed:
        print("失败项：")
        for item in runner.failed:
            print(f"  - {item.name}: {item.detail.strip().splitlines()[0]}")
        close_all()
        return 1
    print("IS_PASS: YES")
    close_all()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
