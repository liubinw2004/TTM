"""TTM 验收测试公共支撑：阈值常量 + 报告记录器 + 共享加载器（QA-PLAN v1.3）。

本模块是 35 条验收项（D/M/F/E）的唯一口径集中地：

* 阈值一律引用 :data:`TH`（对应 QA-PLAN §6.4 的 ``tests/acceptance_thresholds.py`` 职责，
  合并成单文件以减少导入层次）；
* :func:`record` 把每条验收项的「id/名称/状态/实测值/判定/路由」增量写入
  ``runs/eval/acceptance_report.json``（中途崩溃也保留已写部分）；
* :func:`finalize` 由 ``scripts/07_acceptance.py`` 在 pytest 结束后调用，把 fast 圈
  没跑到/人工项补成 SKIP，保证 35 条逐条都有记录。

路由约定（QA-PLAN §0.2）：engineer / data / spec / retrain / manual / none。
"""

from __future__ import annotations

import json
import os
import time
from pathlib import Path
from typing import Any

import numpy as np

# ---------------------------------------------------------------------------
# 路径
# ---------------------------------------------------------------------------
PROJECT_ROOT = Path(__file__).resolve().parents[1]
DATA_DIR = PROJECT_ROOT / "data" / "tier_S"
SHARD_DIR = DATA_DIR / "shards"
#: 报告路径。可用环境变量覆盖（07_acceptance.py 用它支持 `--out` 与 fast 圈分文件写，
#: 避免 fast/full 并发或先后覆盖同一份正式报告）。
REPORT_PATH = Path(
    os.environ.get(
        "TTM_ACCEPTANCE_REPORT",
        str(PROJECT_ROOT / "runs" / "eval" / "acceptance_report.json"),
    )
)
E2E_COMMANDS_JSON = PROJECT_ROOT / "deliverables" / "software-company" / "e2e_commands.json"
CONFIG_PATH = "configs/tier_s.yaml"

# ---------------------------------------------------------------------------
# 阈值与基线（改这里必须同步改 QA-PLAN —— 规格修订流程）
# ---------------------------------------------------------------------------
TH: dict[str, Any] = {
    # L1 数据层
    "D01_EXPECT_WINDOWS": 13696,          # 分片窗口总数（= labels 行数）
    "D03_ROUNDTRIP_ATOL": 1e-6,           # 反归一化→再归一化往返误差
    "D03_SELFCONSIST_ATOL": 0.02,         # 归一化后 mean≈0/std≈1 的容差（02d 同口径）
    "D04_DISTINCT_FULL": 179,             # 合并后条件全集
    "D04_DISTINCT_TRAIN": 178,            # train 可学条件
    "D04_EXPECT_DIFF": {"通背拳"},         # 全集 − train 恰为该集合（EX-01 口径例外）
    "D05_WUSHU_CONDITIONS": 24,           # 武术基名条件数
    "D05_WUSHU_MIN_WINDOWS": 3,           # 每个武术条件窗口下限
    "D06_RESIDUAL_COUNT": 21,             # merge_report residual_below_threshold 条数
    # L2 模型层
    "M01_JOINT_MAE_MAX_DEG": 8.0,         # 验收线
    "M01_JOINT_MAE_WARN_DEG": 6.99,       # 回归劣化报警线（基线 6.49 + 0.5）
    "M02_ROOT_MAE_MAX_CM": 3.0,           # 直接通过线
    "M02_ROOT_MAE_DEGRADE_CM": 4.0,       # 降级带上限（>4cm 不通过）
    # M-03 后验健康度（v1.3.1 裁决版，2026-10-29 主理人裁决：修订阈值不重训）
    # 三条同时满足：a) 逐维 mu std 最小值 ≥ 0.01（对齐 README #13 塌缩指纹 mu_std<0.01）
    #               b) 活跃维数（mu std ≥ 0.05 的维）≥ 8
    #               c) 整体 mu std ≥ 0.1
    # b) 的判定线 2026-10-29 由 0.01 改到 0.05（主理人二次裁决）：
    #   按 0.01 算活跃维是 128/128 —— 恒真、零区分力，且与 a) 重复计数。
    #   a) 守「灾难性塌缩」，b) 守「潜空间广度」，两条必须是不同量尺。
    # 已知限制（永久留痕）：实测 dormant 117/128 = 91%，有效潜维约 11 维，KL=583.8。
    # 这是后续若提升多样性的第一嫌疑点，当前不解决但必须记录。
    "M03_MU_DIM_STD_MIN": 0.01,           # a) 单维下限 = README #13 塌缩指纹线
    "M03_ACTIVE_DIM_STD_MIN": 0.05,       # b) 「活跃维」判定线（v1.3.1 二次裁决：0.01→0.05）
    "M03_ACTIVE_DIMS_MIN": 8,             # b) 活跃维数下限
    # 「休眠维 / 有效潜维」的留痕口径同为 0.05（裁决里 117/128 = 91% 就是按这条算的）。
    "M03_DORMANT_DIM_STD_MIN": 0.05,
    "M03_MU_STD_MIN": 0.1,                # c) 后验整体 std 下限
    "M03_KL_BAND": (100.0, 5000.0),       # KL 为 4864 维求和口径（README #13），裁决版 band
    "M03_DORMANT_REFERENCE": {            # 已知限制参考值（不做判据，只留痕对比）
        "dormant_dims": 117,
        "total_dims": 128,
        "dormant_share": 0.9141,
        "kl_per_sample_sum": 583.773,
        "effective_latent_dims": 11,
    },
    "M04_VAR_RATIO_BAND": (0.7, 1.3),     # 帧间方差比（裁决：不许放宽下界）
    "M04_MARGIN_RISK": 0.05,              # 距 band 边界余量小于该值即记风险（实测 0.019）
    "M05_PARAM_COUNT": (8_000_000, 12_000_000),
    "M07_VAR_RATIO_MIN": 0.5,             # (c) 多样性地板
    "M07_ROUGH_BAND": (0.6, 1.5),         # (d) 粗糙比
    "M07_GUIDANCE": 7.5,                  # 生产值（configs/tier_s.yaml 已定案）
    # L3 功能层
    "F03_SEAM_MAX_CMPS": 100.0,           # 接缝根速度台阶上限（README #21 口径）
    "F03_DURATION_S": 12.0,
    "F03_FRAME_TOL": 3,
    "F17_TRANSIENT_WINDOW": 12,           # 局部连续性窗口（帧）
    "F17_TRANSIENT_MAX_CM": 30.0,
    # F-17 拆两支（v1.3.1 裁决）：a) 启动瞬态硬断言（仅对非振荡 clip）；
    # b) 纵向振荡 → KNOWN_LIMITATION（既非 PASS 也非 FAIL，必须进 limitations 段）
    # 振荡判据：root Y 峰峰值 ≥ 40cm 且 dY 变号次数 ≥ 3
    "F17_OSC_PEAK_TO_PEAK_CM": 40.0,
    "F17_OSC_SIGN_FLIPS_MIN": 3,
    "F17_OSC_FPS": 30.0,                  # 周期换算用（服务端固定 30fps）
    "F17_BAND_NON_STAIRS_P50": 90.0,      # GT 非楼梯根高带（软观察）
    "F17_BAND_STAIRS_P50": 179.0,         # GT 楼梯根高带（n=283 全量统计，软观察）
    "F17_BAND_TOL_CM": 80.0,              # 软观察带容差
    "F16_MARGIN_REQUIRED": 0.1,
    "F16_SAMPLE_CONDITIONS": 15,
    "F16_WUSHU_SAMPLE": 8,
    "F18_EXPECT_CHARACTERS": (
        "SK_Celeste_02.FBX",
        "SK_AKANE_inMask_noScroll.FBX",
        "SK_Chinese_Noble_Girl.FBX",
        "SK_Sun_CASUAL_skirt_no_bones.FBX",
    ),
    "E01_FRAME_TOL": 3,
    # 肘/膝「限位」口径（v1.3.1 按 GT 标定修正）：
    # 原 [0,160] 是把**无符号相对旋转角**当成屈曲限位，语义本身不成立；GT 实测
    # （320 窗、同一 retarget 链路）per-joint max = lowerarm_l 160.3 / lowerarm_r 169.2
    # / calf_l 154.6 / calf_r 150.8，且 0.23% 的窗口自己就 >160° —— 原阈值会误杀合法数据。
    # 现改为：硬界 175°（防翻转/数值爆炸，GT max 169.2 留 6° 余量）+ GTmax+10° 软观察。
    "E01_JOINT_HARD_MAX_DEG": 175.0,
    "E01_GT_JOINT_MAX_DEG": {
        "lowerarm_l": 160.3,
        "lowerarm_r": 169.2,
        "calf_l": 154.6,
        "calf_r": 150.8,
    },
    "E01_GT_MARGIN_DEG": 10.0,
    "E01_GUIDANCE": 7.5,                  # 生产值，显式传参
    "E01_SEEDS": (42, 43, 44),
    "E02_TRAIN_VRAM_MAX_MB": 5120.0,
    "E02_INFER_VRAM_MAX_MB": 2560.0,
    "E02_WARM_START_MAX_S": 10.0,
    "GROUND_ORIGIN_FOOT_TOL_CM": 2.0,     # BVH 首帧足踝世界高容差（工程师口径①）
    # 基线锚点（劣化报警用）
    "BASELINE_JOINT_MAE_DEG": 5.9006,     # runs/tier_s/vae/eval_full_m2prime.json
    "BASELINE_ROOT_MAE_CM": 1.8681,
}

#: 35 条验收项登记表：id → (名称, 圈层)。缺一条 finalize 时补 SKIP。
EXPECTED_ITEMS: dict[str, tuple[str, str]] = {
    "D-01": ("窗口总数三方一致（分片/labels/契约）", "fast"),
    "D-02": ("mask 合法性（全 0 窗口=0、前缀连续）", "fast"),
    "D-03": ("norm_stats 合理性（恒等维+自洽+往返）", "fast"),
    "D-04": ("条件数双口径 179/178 + 差集恰为{通背拳}", "fast"),
    "D-05": ("武术 24 条件每类 ≥3 窗", "fast"),
    "D-06": ("label_merge residual 21 项如实上报且无捡垃圾标签", "fast"),
    "D-07": ("split 泄漏检查（take_id 无交集）", "fast"),
    "M-01": ("VAE 关节角重构误差（全量 val）", "fast"),
    "M-02": ("VAE 根位移重构误差（含降级带）", "fast"),
    "M-03": ("VAE 后验健康度（mu_std / KL）", "gpu"),
    "M-04": ("VAE 帧间方差比（防糊）", "gpu"),
    "M-05": ("扩散训练 loss 收敛 + 参数量", "fast"),
    "M-06": ("CFG null-condition 生效性三重断言", "gpu"),
    "M-07": ("多样性地板 (c)/(d) @guidance=7.5", "fast"),
    "F-01": ("采样确定性（同 seed 同指令逐帧一致）", "gpu"),
    "F-02": ("引导强度行为（9.0 探点 (d) 重验）", "gpu"),
    "F-03": ("长动作滑窗接缝连续性（12s）", "gpu"),
    "F-04": ("数值健壮性（无 NaN / 错误路径）", "gpu"),
    "F-05": ("Spine3 折叠正确性（四元数复合手算）", "fast"),
    "F-06": ("UE 骨架名映射完整性 23/23 + root 恒等", "fast"),
    "F-07": ("T-pose 对齐与手性（qw≥0 / 根高口径）", "gpu"),
    "F-08": ("BVH 结构回读 + ground_origin 四断言", "gpu"),
    "F-09": ("Motion JSON schema 校验 + roundtrip", "fast"),
    "F-10": ("导出单位/帧率契约（cm / YXZ / 30fps）", "fast"),
    "F-11": ("/api/generate 正常路径", "gpu"),
    "F-12": ("超时/慢请求路径（12s × 2 串行）", "gpu"),
    "F-13": ("非法输入 422 参数化矩阵", "fast"),
    "F-14": ("路径穿越防护", "fast"),
    "F-15": ("静态资源与接口联通", "fast"),
    "F-16": ("中文条件嵌入可分性（F-16 前置体检）", "slow"),
    "F-17": ("启动瞬态局部连续性（v1.3 判据）", "gpu"),
    "F-18": ("多角色切换契约（manifest / 无骨骼警示）", "fast"),
    "E-01": ("中文指令全链路矩阵（10 条 × 3 seed）", "gpu"),
    "E-02": ("显存与时延实测留证", "fast"),
    "E-03": ("外部工具链验证（UE 空跑 / Blender 人工）", "slow"),
}

#: 仅人工项（QA-PLAN §5.4.1），进报告 manual_items。
MANUAL_ITEMS: list[dict[str, str]] = [
    {
        "id": "MANUAL-盲评",
        "name": "盲评语义正确性（PRD S1，主理人+用户 2 人代理标准）",
        "evidence": "deliverables/blind_eval/（盲评执行后生成 prompts.csv / scores.csv / blind_eval_report.md）",
    },
    {
        "id": "MANUAL-M02b",
        "name": "root_mae ∈(3,4]cm 降级感知检查（当前 1.87cm 未触发）",
        "evidence": "runs/tier_s/vae/eval_full_m2prime.json（root_height_mae_cm=1.87 ≤3cm，直接通过，无需降级裁决）",
    },
    {
        "id": "MANUAL-Blender",
        "name": "Blender 4.x 实际导入 BVH 播放",
        "evidence": "runs/api/*.bvh（供拖入 Blender；E-01 生成产物）",
    },
    {
        "id": "MANUAL-UE",
        "name": "UE 实际导入播放（自动化下限=ue_import.py 空跑，见 E-03）",
        "evidence": "deliverables/ue/（ue_import.py 产物 + import_unreal.py）",
    },
    {
        "id": "MANUAL-前端视觉",
        "name": "前端 4 角色 FBX 套用观感（塌肩/穿模）+ 1080p ≥55fps",
        "evidence": "web/index.html + runs/api/*.json（启动 uvicorn 后人工目检）",
    },
]


# ---------------------------------------------------------------------------
# 报告记录器
# ---------------------------------------------------------------------------
def _read_report() -> dict[str, Any]:
    if REPORT_PATH.is_file():
        try:
            return json.loads(REPORT_PATH.read_text(encoding="utf-8"))
        except ValueError:
            pass
    return {"items": {}, "notes": []}


def reset_report(tier: str) -> None:
    """初始化/清空报告（07_acceptance.py 在 pytest 启动前调用）。"""
    REPORT_PATH.parent.mkdir(parents=True, exist_ok=True)
    payload = {
        "tier": tier,
        "generated_at": time.strftime("%Y-%m-%d %H:%M:%S"),
        "written_at": time.strftime("%Y-%m-%d %H:%M:%S"),
        "report_path": REPORT_PATH.as_posix(),
        "items": {},
        "notes": [],
        "limitations": [],
    }
    REPORT_PATH.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")


def url_to_local(url: str) -> Path:
    """把服务端返回的 ``/motions/<name>`` 下载链接映射回本地磁盘路径。

    服务端产物落盘目录是 ``server.app.OUTPUT_DIR``（= ``runs/api``），
    而对外链接统一是 ``/motions/<name>``，两者不同名，不能直接拼 PROJECT_ROOT。
    """
    name = url.split("/motions/", 1)[-1].lstrip("/")
    return PROJECT_ROOT / "runs" / "api" / name


def add_limitation(
    item_id: str,
    title: str,
    measured: dict[str, Any],
    note: str = "",
    route: str = "engineer",
) -> None:
    """记一条**已知限制**（既不是 PASS 也不是 FAIL）。

    主理人裁决（F-17b）：真实存在、暂不修、但必须永远可见的缺陷走这里。要求：
    ① 进 ``items`` 的状态用 ``KNOWN_LIMITATION``（summary 单列计数）；
    ② 同时进报告顶层的 ``limitations`` 段，带振幅/峰峰值/周期等可复核实测值。

    幂等：同 ``item`` + 同 ``measured`` 的条目已存在时不重复追加（报告是单文件
    「读-改-写」，两个 pytest 进程并发跑同一套验收时会互相 append 出重复条目）。
    """
    report = _read_report()
    bucket = report.setdefault("limitations", [])
    fingerprint = json.dumps([item_id, measured], ensure_ascii=False, sort_keys=True)
    for existing in bucket:
        same = json.dumps(
            [existing.get("item"), existing.get("measured")], ensure_ascii=False, sort_keys=True
        )
        if same == fingerprint:
            return
    bucket.append(
        {
            "item": item_id,
            "title": title,
            "measured": measured,
            "note": note,
            "route": route,
            "recorded_at": time.strftime("%H:%M:%S"),
        }
    )
    REPORT_PATH.write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")


def add_note(message: str, route: str = "none") -> None:
    """往报告追加一条跨项备注（口径差异/发现项）。"""
    report = _read_report()
    report.setdefault("notes", []).append({"note": message, "route": route})
    REPORT_PATH.write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")


def record(
    item_id: str,
    status: str,
    measured: dict[str, Any] | None = None,
    route: str = "none",
    notes: str = "",
) -> None:
    """记录一条验收项结果。status ∈ PASS / FAIL / WARN / SKIP。"""
    name, circle = EXPECTED_ITEMS.get(item_id, (item_id, "?"))
    entry = {
        "id": item_id,
        "name": name,
        "circle": circle,
        "status": status,
        "measured": measured or {},
        "route": route,
        "notes": notes,
        "recorded_at": time.strftime("%H:%M:%S"),
    }
    report = _read_report()
    report.setdefault("items", {})[item_id] = entry
    REPORT_PATH.write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")


def finalize(tier: str) -> dict[str, Any]:
    """pytest 结束后调用：补 SKIP（fast 圈未跑到的 gpu/slow 项）+ 汇总统计。"""
    report = _read_report()
    items: dict[str, Any] = report.setdefault("items", {})
    for item_id, (name, circle) in EXPECTED_ITEMS.items():
        if item_id in items:
            continue
        items[item_id] = {
            "id": item_id,
            "name": name,
            "circle": circle,
            "status": "SKIP",
            "measured": {},
            # fast 圈按契约不跑 gpu/slow 项 → 路由标为「延后到 full 圈」，不是人工项；
            # 只有 circle == "slow" 且确实需要外部工具/人眼的条目才走 manual。
            "route": circle if tier == "fast" else "none",
            "notes": (
                f"fast 圈未执行：该条属 {circle} 圈，请跑 `--tier full` 取实测值"
                if tier == "fast"
                else "未执行（上游失败被跳过）"
            ),
        }
    report["tier"] = tier
    report["written_at"] = time.strftime("%Y-%m-%d %H:%M:%S")
    report["report_path"] = REPORT_PATH.as_posix()
    report["manual_items"] = MANUAL_ITEMS
    report.setdefault("limitations", [])
    counts: dict[str, int] = {}
    for entry in items.values():
        counts[entry["status"]] = counts.get(entry["status"], 0) + 1
    # 已知限制与 PASS/FAIL/SKIP 并列单列（主理人要求：不许只藏在备注里）
    report["summary"] = {
        "total": len(items),
        "PASS": counts.get("PASS", 0),
        "FAIL": counts.get("FAIL", 0),
        "SKIP": counts.get("SKIP", 0),
        "KNOWN_LIMITATION": counts.get("KNOWN_LIMITATION", 0),
        "WARN": counts.get("WARN", 0),
        "limitations_recorded": len(report["limitations"]),
        "exit_hint": "0=无 FAIL" if counts.get("FAIL", 0) == 0 else "1=有 FAIL",
    }
    REPORT_PATH.write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")
    return report


# ---------------------------------------------------------------------------
# 共享数据加载器（L1 / GT 参考）
# ---------------------------------------------------------------------------
def load_labels_merged() -> list[dict[str, Any]]:
    """读 labels_merged.jsonl（窗口级，13696 行）。"""
    import orjson

    path = DATA_DIR / "labels_merged.jsonl"
    return [orjson.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line.strip()]


def load_index() -> list[dict[str, Any]]:
    """读 index.jsonl（clip 级）。"""
    import orjson

    path = DATA_DIR / "index.jsonl"
    return [orjson.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line.strip()]


def iter_shards():
    """按序遍历分片，yield (path, npz)。"""
    for path in sorted(SHARD_DIR.glob("*.npz")):
        yield path, np.load(path, allow_pickle=True)


def shard_window_count() -> int:
    """分片窗口总数（13696）。"""
    total = 0
    for _path, z in iter_shards():
        total += int(z["feat"].shape[0])
    return total


def load_condition_windows(
    condition: str,
    *,
    split: str | None = "train",
    labels: list[dict[str, Any]] | None = None,
    limit: int | None = None,
) -> np.ndarray:
    """从分片取某条件的归一化特征窗口 ``[N,150,146]``。

    Args:
        condition: 合并后中文条件名（如 ``太极拳``）。
        split: 只取该 split（``None`` 取全部）。
        labels: 已加载的 labels_merged 行（缓存复用）。
        limit: 最多取多少窗。

    Returns:
        ``[N,150,146]`` float32（分片本身就是归一化特征）。
    """
    rows = labels if labels is not None else load_labels_merged()
    wanted = {
        row["clip_id"]
        for row in rows
        if row["zh_label"] == condition and (split is None or row["split"] == split)
    }
    chunks: list[np.ndarray] = []
    got = 0
    for _path, z in iter_shards():
        ids = z["clip_id"]
        hit = np.isin(ids, list(wanted)) if wanted else np.zeros(len(ids), dtype=bool)
        if hit.any():
            chunks.append(z["feat"][hit].astype(np.float32))
            got += int(hit.sum())
        if limit is not None and got >= limit:
            break
    if not chunks:
        raise LookupError(f"条件 {condition!r} 在 split={split} 下没有窗口")
    out = np.concatenate(chunks, axis=0)
    return out[:limit] if limit is not None else out


# ---------------------------------------------------------------------------
# 共享数学工具（与 scripts/05b 同口径）
# ---------------------------------------------------------------------------
def dimwise_var(matrix: np.ndarray) -> float:
    """``[N,D]`` 逐维方差再对维取均值（ddof=1）。"""
    if matrix.shape[0] < 2:
        return 0.0
    return float(np.var(matrix, axis=0, ddof=1).mean())


def descriptor_of(mu: np.ndarray) -> np.ndarray:
    """``[B,38,128]`` 潜变量对 token 取均值 → ``[B,128]`` L2 归一化描述子。"""
    matrix = np.asarray(mu)
    while matrix.ndim > 2:
        matrix = matrix.mean(axis=1)
    norms = np.linalg.norm(matrix, axis=1, keepdims=True)
    return matrix / np.clip(norms, 1e-12, None)


def roughness_of(feat_norm: np.ndarray) -> float:
    """帧间一阶差分绝对值均值（在归一化特征的 joints_6d 段上算，05b 同口径）。"""
    from ttm.common import constants as C  # noqa: PLC0415

    joints = feat_norm[:, C.FEAT_LAYOUT["joints_6d"]]
    return float(np.abs(np.diff(joints, axis=0)).mean()) if feat_norm.shape[0] > 1 else 0.0


# ---------------------------------------------------------------------------
# BVH 轻量解析（F-08/F-17：结构回读，不引 Blender 依赖）
# ---------------------------------------------------------------------------
def parse_bvh(path: str | Path) -> dict[str, Any]:
    """解析 BVH 文本（层级 + MOTION），返回结构化字典。

    Returns:
        ``{"joint_names": [层级序], "channels": [每关节通道数], "offsets": [J,3],
        "frame_time": float, "frames": [F, n_channels_total]}``
    """
    lines = Path(path).read_text(encoding="utf-8").splitlines()
    names: list[str] = []
    channels: list[int] = []
    offsets: list[list[float]] = []
    parents: list[int] = []
    stack: list[int] = []
    in_motion = False
    n_frames = 0
    frame_time = 0.0
    rows: list[list[float]] = []
    for raw in lines:
        line = raw.strip()
        if not in_motion:
            if line.startswith(("ROOT", "JOINT")):
                parents.append(stack[-1] if stack else -1)
                names.append(line.split(None, 1)[1])
                channels.append(0)
                offsets.append([0.0, 0.0, 0.0])
            elif line.startswith("End Site"):
                stack.append(-2)
                continue
            elif line.startswith("OFFSET"):
                offsets[len(names) - 1] = [float(v) for v in line.split()[1:4]]
            elif line.startswith("CHANNELS"):
                channels[len(names) - 1] = int(line.split()[1])
            elif line == "{":
                stack.append(len(names) - 1)
            elif line == "}":
                if stack and stack[-1] == -2:
                    stack.pop()
                    continue
                if stack:
                    stack.pop()
            elif line == "MOTION":
                in_motion = True
            continue
        if line.startswith("Frames:"):
            n_frames = int(line.split()[1])
        elif line.startswith("Frame Time:"):
            frame_time = float(line.split(":", 1)[1])
        elif line:
            rows.append([float(v) for v in line.split()])
    return {
        "joint_names": names,
        "channels": channels,
        "offsets": np.asarray(offsets, dtype=np.float64),
        "parents": parents,
        "frame_time": frame_time,
        "n_frames": n_frames,
        "frames": np.asarray(rows, dtype=np.float64),
    }


def bvh_fk_world_positions(parsed: dict[str, Any]) -> np.ndarray:
    """对 BVH 全帧做正运动学（欧拉 YXZ 角度制 + OFFSET），返回 ``[F,J,3]`` 世界坐标。

    这是纯文本口径的 FK（scipy Rotation），与生成链路
    ``generate._world_joint_positions``（四元数口径）互为印证。
    """
    from scipy.spatial.transform import Rotation  # noqa: PLC0415

    names = parsed["joint_names"]
    offsets = parsed["offsets"]
    parents = parsed["parents"]
    channels = parsed["channels"]
    frames = parsed["frames"]  # [F, total_channels]
    n_joints = len(names)
    out = np.zeros((frames.shape[0], n_joints, 3), dtype=np.float64)

    # 每个关节的通道在帧行内的起始偏移
    starts: list[int] = []
    cursor = 0
    for count in channels:
        starts.append(cursor)
        cursor += count

    local_rot = [None] * n_joints
    for idx in range(n_joints):
        base = starts[idx]
        if channels[idx] == 6:  # 根：Xposition Yposition Zposition + Yrotation Xrotation Zrotation
            out[:, idx] = frames[:, base : base + 3]
            euler = frames[:, base + 3 : base + 6]
        else:
            euler = frames[:, base : base + 3]
        local_rot[idx] = Rotation.from_euler("YXZ", euler, degrees=True)

    # 自顶向下递推：world_rot[idx] = world_rot[parent] ∘ local_rot[idx]；
    # world_pos[idx] = world_pos[parent] + R_world[parent] @ OFFSET
    world_rot: list = [None] * n_joints
    for idx in range(n_joints):
        parent = parents[idx]
        if parent < 0:
            world_rot[idx] = local_rot[idx]
        else:
            world_rot[idx] = world_rot[parent] * local_rot[idx]
            out[:, idx] = out[:, parent] + world_rot[parent].apply(offsets[idx])
    return out
