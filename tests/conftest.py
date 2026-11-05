"""pytest 共享 fixtures：程序化合成的微型 BVH（**不依赖 Z 盘**）。

设计目标：让 T03/T04 的全部单测在完全离线、无外部数据的条件下可跑，
同时覆盖真实数据里遇到的各种脏情况：

* 标准 23 关节文件（含 ``End Site`` 块）；
* root-only 轨迹文件（只有 ``ROOT`` + ``CHANNELS 6``）；
* 关节数不足的文件；
* 空 Take 目录；
* 长文件（11,802 帧，演练流式解析）。

所有临时文件都写在 ``D:/TTM/.cache/tmp/pytest`` 之内（写盘硬约束）。
"""

from __future__ import annotations

import shutil
import sys
import tempfile
from pathlib import Path
from typing import Any, Iterable, Sequence

import numpy as np
import pytest

_ROOT = Path(__file__).resolve().parents[1]
_SRC = _ROOT / "src"
if str(_SRC) not in sys.path:  # 兼容未做 editable install 的场景
    sys.path.insert(0, str(_SRC))

from ttm.common import constants as C  # noqa: E402
from ttm.common import paths as P  # noqa: E402

# ---------------------------------------------------------------------------
# 合成 BVH 构造器
# ---------------------------------------------------------------------------
_OFFSETS: tuple[tuple[float, float, float], ...] = (
    (0.0, 0.0, 0.0),
    (0.0, 8.08814, -2.57809),
    (0.0, 13.8624, 0.0),
    (0.0, 13.8624, 0.0),
    (0.0, 13.8624, 0.0),
    (0.0, 6.7103, 3.57809),
    (0.0, 10.1102, 0.0),
    (1.68503, 0.174398, 3.37006),
    (18.9636, 0.0, 0.0),
    (21.1665, 0.0, 0.0),
    (22.9093, 0.0, 0.0),
    (-1.68503, 0.174398, 3.37006),
    (-18.9636, 0.0, 0.0),
    (-21.1665, 0.0, 0.0),
    (-22.9093, 0.0, 0.0),
    (10.1102, -3.37006, 2.0),
    (0.0, -32.9379, 2.35904),
    (0.0, -37.5271, -2.35904),
    (0.0, -6.5059, 15.1653),
    (-10.1102, -3.37006, 2.0),
    (0.0, -32.9379, 2.35904),
    (0.0, -37.5271, -2.35904),
    (0.0, -6.5059, 15.1653),
)


def _children(parents: Sequence[int], idx: int) -> list[int]:
    """返回某关节的子关节索引（保持升序 = BVH 文件中的顺序）。"""
    return [child for child, parent in enumerate(parents) if parent == idx]


def _emit_joint(
    lines: list[str],
    idx: int,
    depth: int,
    *,
    root_name: str,
    offsets: Sequence[Sequence[float]],
    parents: Sequence[int],
    with_end_sites: bool,
) -> None:
    """递归输出一个关节块（标准 BVH 缩进格式）。"""
    pad = "\t" * depth
    keyword = "ROOT" if idx == 0 else "JOINT"
    name = root_name if idx == 0 else C.BONE_ORDER[idx]
    lines.append(f"{pad}{keyword} {name}")
    lines.append(f"{pad}{{")
    offset = offsets[idx]
    lines.append(f"{pad}\tOFFSET {offset[0]:.6g} {offset[1]:.6g} {offset[2]:.6g}")
    if idx == 0:
        lines.append(
            f"{pad}\tCHANNELS 6 Xposition Yposition Zposition Yrotation Xrotation Zrotation"
        )
    else:
        lines.append(f"{pad}\tCHANNELS 3 Yrotation Xrotation Zrotation")
    kids = _children(parents, idx)
    for child in kids:
        _emit_joint(
            lines,
            child,
            depth + 1,
            root_name=root_name,
            offsets=offsets,
            parents=parents,
            with_end_sites=with_end_sites,
        )
    if with_end_sites and not kids:
        lines.append(f"{pad}\tEnd Site")
        lines.append(f"{pad}\t{{")
        lines.append(f"{pad}\t\tOFFSET 0 5 0")
        lines.append(f"{pad}\t}}")
    lines.append(f"{pad}}}")


def build_motion_arrays(
    n_frames: int,
    *,
    fps: float = 120.0,
    start_height: float = 91.7,
    animated: bool = True,
) -> tuple[np.ndarray, np.ndarray]:
    """构造合成动作的 ``(root_pos[F,3], euler_deg[F,23,3])``。

    Args:
        n_frames: 帧数。
        fps: 帧率。
        start_height: Hips 初始高度（cm）。
        animated: ``False`` 时输出恒等旋转（便于精确断言）。

    Returns:
        ``(root_pos_cm, euler_deg)``。
    """
    t = np.arange(n_frames, dtype=np.float64) / float(fps)
    if animated:
        root = np.stack(
            [
                10.0 * np.sin(2 * np.pi * 0.25 * t),
                start_height + 2.0 * np.sin(2 * np.pi * 0.5 * t),
                5.0 * np.cos(2 * np.pi * 0.25 * t),
            ],
            axis=1,
        )
    else:
        root = np.tile(np.array([0.0, start_height, 0.0]), (n_frames, 1))

    euler = np.zeros((n_frames, C.N_JOINTS, 3), dtype=np.float64)
    if animated:
        for joint in range(C.N_JOINTS):
            euler[:, joint, 0] = 6.0 * np.sin(2 * np.pi * 0.5 * t + joint * 0.3)  # Y
            euler[:, joint, 1] = 4.0 * np.cos(2 * np.pi * 0.75 * t + joint * 0.2)  # X
            euler[:, joint, 2] = 2.0 * np.sin(2 * np.pi * 1.0 * t + joint * 0.1)  # Z
    return root, euler


def build_bvh_text(
    *,
    n_frames: int = 48,
    frame_time: float = 1.0 / 120.0,
    root_name: str = "Hips",
    start_height: float = 91.7,
    animated: bool = True,
    with_end_sites: bool = True,
) -> str:
    """生成一份**标准 23 关节**的合成 BVH 文本。

    Args:
        n_frames: 帧数。
        frame_time: Frame Time（秒）。
        root_name: ROOT 名称（用于测试「根名可不同于 Hips」）。
        start_height: Hips 高度（cm）。
        animated: 是否写入非零旋转。
        with_end_sites: 是否输出 ``End Site`` 块（真实数据都有）。

    Returns:
        完整 BVH 文本。
    """
    lines: list[str] = ["HIERARCHY"]
    _emit_joint(
        lines,
        0,
        0,
        root_name=root_name,
        offsets=_OFFSETS,
        parents=C.PARENTS,
        with_end_sites=with_end_sites,
    )
    lines.append("MOTION")
    lines.append(f"Frames: {n_frames}")
    lines.append(f"Frame Time: {frame_time}")

    root, euler = build_motion_arrays(
        n_frames, fps=1.0 / frame_time, start_height=start_height, animated=animated
    )
    for frame in range(n_frames):
        # 每帧通道顺序必须严格对齐 CHANNELS 声明：
        # 根关节 6 通道（X/Y/Z position + Y/X/Z rotation）+ 22 个关节各 3 通道 = 72
        row: list[str] = [f"{value:.6f}" for value in root[frame]]
        row.extend(f"{value:.6f}" for value in euler[frame, 0])
        for joint in range(1, C.N_JOINTS):
            row.extend(f"{value:.6f}" for value in euler[frame, joint])
        lines.append(" ".join(row))
    return "\n".join(lines) + "\n"


def build_root_only_bvh_text(*, n_frames: int = 200, frame_time: float = 1.0 / 120.0) -> str:
    """生成真实的 root-only 轨迹文件内容（6 通道、无任何 JOINT）。"""
    lines = [
        "HIERARCHY",
        "ROOT stair_small_0",
        "{",
        "\tOFFSET 0 0 0",
        "\tCHANNELS 6 Xposition Yposition Zposition Yrotation Xrotation Zrotation",
        "}",
        "MOTION",
        f"Frames: {n_frames}",
        f"Frame Time: {frame_time}",
    ]
    for frame in range(n_frames):
        lines.append(f"{110.0 + frame * 0.01:.6f} 0.0 {158.0 - frame * 0.01:.6f} 0.0 0.0 0.0")
    return "\n".join(lines) + "\n"


def build_few_joints_bvh_text(*, n_frames: int = 10) -> str:
    """生成只有 2 个关节的 BVH（用于「关节数 != 23」的拒绝路径）。"""
    lines = [
        "HIERARCHY",
        "ROOT Hips",
        "{",
        "\tOFFSET 0 0 0",
        "\tCHANNELS 6 Xposition Yposition Zposition Yrotation Xrotation Zrotation",
        "\tJOINT Spine",
        "\t{",
        "\t\tOFFSET 0 10 0",
        "\t\tCHANNELS 3 Yrotation Xrotation Zrotation",
        "\t\tEnd Site",
        "\t\t{",
        "\t\t\tOFFSET 0 5 0",
        "\t\t}",
        "\t}",
        "}",
        "MOTION",
        f"Frames: {n_frames}",
        "Frame Time: 0.0333333",
    ]
    for _ in range(n_frames):
        lines.append("0 91.7 0 0 0 0 0 0 0")
    return "\n".join(lines) + "\n"


def build_bad_channels_bvh_text() -> str:
    """生成通道布局非法的 BVH（非根关节带了位置通道）。"""
    text = build_bvh_text(n_frames=4)
    return text.replace(
        "JOINT Spine\n\t{\n\t\tOFFSET 0 8.08814 -2.57809\n\t\tCHANNELS 3 Yrotation Xrotation Zrotation",
        "JOINT Spine\n\t{\n\t\tOFFSET 0 8.08814 -2.57809\n"
        "\t\tCHANNELS 6 Xposition Yposition Zposition Yrotation Xrotation Zrotation",
        1,
    )


# ---------------------------------------------------------------------------
# fixtures
# ---------------------------------------------------------------------------
@pytest.fixture(scope="session")
def tmp_root() -> Path:
    """本会话的临时根目录（``D:/TTM/.cache/tmp/pytest``）。"""
    return P.safe_makedirs(P.cache_dir() / "tmp" / "pytest")


@pytest.fixture()
def work_dir(tmp_root: Path) -> Path:
    """每个用例独立的临时目录（保证在 D 盘内）。"""
    created = Path(tempfile.mkdtemp(prefix="ttm_", dir=str(tmp_root)))
    try:
        yield created
    finally:
        shutil.rmtree(created, ignore_errors=True)


@pytest.fixture()
def synth_bvh_text() -> str:
    """标准 23 关节合成 BVH 文本（48 帧 @120fps）。"""
    return build_bvh_text(n_frames=48)


@pytest.fixture()
def synth_bvh_file(work_dir: Path) -> Path:
    """把合成 BVH 写成文件，返回路径。"""
    target = P.resolve_out(work_dir / "synth.bvh", mkparents=True)
    target.write_text(build_bvh_text(n_frames=48), encoding="utf-8")
    return target


@pytest.fixture()
def root_only_bvh_file(work_dir: Path) -> Path:
    """root-only 脏文件的路径。"""
    target = P.resolve_out(work_dir / "Take_114_stair_small_0.bvh", mkparents=True)
    target.write_text(build_root_only_bvh_text(n_frames=120), encoding="utf-8")
    return target


@pytest.fixture()
def long_bvh_file(work_dir: Path) -> Path:
    """11,802 帧的长文件（对齐武术套路实测规模，演练流式解析）。"""
    target = P.resolve_out(work_dir / "long.bvh", mkparents=True)
    target.write_text(build_bvh_text(n_frames=11802), encoding="utf-8")
    return target


def write_bvh(path: Path | str, text: str) -> Path:
    """把 BVH 文本写到目标路径（经写盘守卫）。"""
    target = P.resolve_out(path, mkparents=True)
    target.write_text(text, encoding="utf-8")
    return target


def write_config(
    path: Path | str,
    *,
    bvh_roots: Iterable[str],
    tier: str = "S",
    data_dir: str = "data/unit_test",
    runs_dir: str = "runs/unit_test",
    **overrides: Any,
) -> Path:
    """写一份最小可用配置（用于扫描/预处理的离线单测）。

    Args:
        path: 目标 yaml 路径。
        bvh_roots: BVH 根目录列表。
        tier: 数据档位。
        data_dir: 产物目录（必须位于 ``D:/TTM`` 内）。
        runs_dir: 训练产物根目录（各测试模块用不同值，避免 ckpt 互相覆盖；
            实际目录还会再拼一层 ``tier_<档位>``）。
        **overrides: 形如 ``window=150`` 的顶层字段覆盖（写入 ``data`` 小节）。

    Returns:
        配置文件的绝对路径。
    """
    import yaml

    data_section: dict[str, Any] = {
        "tier": tier,
        "bvh_roots": [str(item) for item in bvh_roots],
        "include_categories": None,
        "fps": 30,
        "window": 150,
        "overlap": 30,
        "min_tail_frames": 75,
        "actors_per_take": 1,
        "max_files": 500,
        "val_ratio": 0.1,
        "test_ratio": 0.1,
        "shard_size": 32,
    }
    data_section.update(overrides)
    payload = {
        "project": {
            "root": "D:/TTM",
            "cache_dir": "D:/TTM/.cache",
            "runs_dir": runs_dir,
            "data_dir": data_dir,
        },
        "data": data_section,
        "vae": {"epochs": 2, "batch_size": 4, "hidden": 64},
        "diffusion": {"d_model": 64, "n_layers": 2, "n_heads": 4, "epochs": 1, "batch_size": 4},
        "text": {"proj_dim": 64},
        "runtime": {"device": "cpu", "precision": "fp32"},
    }
    target = P.resolve_out(path, mkparents=True)
    target.write_text(yaml.safe_dump(payload, allow_unicode=True, sort_keys=False), encoding="utf-8")
    return target


# ---------------------------------------------------------------------------
# 测试编排
# ---------------------------------------------------------------------------
def pytest_configure(config: pytest.Config) -> None:
    """注册自定义 marker（补 pyproject 未覆盖的环境相关标记）。"""
    for marker, description in (
        ("requires_torch", "需要 torch 的测试（未安装时 SKIP）"),
        ("requires_zdrive", "需要只读网络盘 Z:/1000hours 的测试（不可达时 SKIP）"),
        ("slow", "耗时较长的测试"),
        ("gpu", "验收：需要 CUDA + 正式 ckpt（fast 圈跳过）"),
        ("manual", "仅人工执行（pytest 永久 skip 并输出指引）"),
    ):
        config.addinivalue_line("markers", f"{marker}: {description}")


# ---------------------------------------------------------------------------
# 验收测试共享 fixtures（QA-PLAN v1.3，供 test_acceptance_*.py 使用；
# fast 圈不收集 gpu/slow 用例，因此这些 fixture 不会被实例化）
# ---------------------------------------------------------------------------
def _cuda_available() -> bool:
    try:
        import torch  # noqa: PLC0415

        return bool(torch.cuda.is_available())
    except Exception:  # noqa: BLE001
        return False


def url_to_local(url: str) -> Path:
    """把服务端返回的 ``/motions/<name>`` 下载链接映射回本地磁盘路径。

    服务端产物落盘目录是 ``server.app.OUTPUT_DIR``（= ``runs/api``），
    而对外链接统一是 ``/motions/<name>``，两者不同名，不能直接拼 PROJECT_ROOT。
    """
    name = url.split("/motions/", 1)[-1].lstrip("/")
    return Path(__file__).resolve().parents[1] / "runs" / "api" / name


@pytest.fixture(scope="session")
def acceptance_cfg():
    """正式档配置（configs/tier_s.yaml）。"""
    from ttm.common.config import load_config

    return load_config("configs/tier_s.yaml")


@pytest.fixture(scope="session")
def acceptance_generator(acceptance_cfg):
    """整条推理链路（VAE + DiT + bge + 融合），session 级单例（加载一次 ~5s）。"""
    pytest.importorskip("torch", reason="torch 未安装")
    if not _cuda_available():
        pytest.skip("SKIP_NO_CUDA: 验收生成类用例需要 CUDA")
    from ttm.infer.generate import load_generator

    return load_generator(acceptance_cfg)


@pytest.fixture(scope="session")
def acceptance_client(acceptance_cfg):
    """FastAPI TestClient（懒加载生成器；首个 /generate 触发权重加载）。"""
    pytest.importorskip("fastapi", reason="fastapi 未安装")
    from fastapi.testclient import TestClient

    from server.app import create_app

    return TestClient(create_app(acceptance_cfg))


@pytest.fixture(scope="session")
def acceptance_e2e_results(acceptance_client, acceptance_generator):
    """E-01 矩阵生成结果缓存（10 条指令 × 3 seed，只跑一次，供 F/E 多条复用）。

    Returns:
        ``{cmd_id: {"cmd", "cat", "results": [GenerateResponse.json() × 3], "error": ""}}``
    """
    import json as _json

    payload = _json.loads(
        (Path(__file__).resolve().parents[1] / "deliverables/software-company/e2e_commands.json")
        .read_text(encoding="utf-8")
    )
    out: dict[str, dict[str, Any]] = {}
    guidance = float(payload.get("test_protocol", {}).get("guidance", 7.5))
    for command in payload["commands"]:
        results = []
        error = ""
        for seed in (42, 43, 44):
            try:
                resp = acceptance_client.post(
                    "/generate",
                    json={
                        "prompt": command["cmd"],
                        "duration_s": 5.0,
                        "seed": seed,
                        "guidance": guidance,
                        "export": "both",
                    },
                )
                body = resp.json()
                entry: dict[str, Any] = {"status": resp.status_code, "body": body, "seed": seed}
                # 服务端只在 export="json" 时内联 payload；这里统一用下载链接取 JSON，
                # 保证 E-01（矩阵）/F-08（BVH 对账）/F-17（BVH 局部连续性）拿到同一份产物。
                if resp.status_code == 200 and body.get("json_url"):
                    download = acceptance_client.get(body["json_url"])
                    entry["payload"] = download.json() if download.status_code == 200 else {}
                    entry["json_path"] = str(url_to_local(body["json_url"]))
                if resp.status_code == 200 and body.get("bvh_url"):
                    entry["bvh_path"] = str(url_to_local(body["bvh_url"]))
                results.append(entry)
            except Exception as exc:  # noqa: BLE001 - 记录后继续，报告里体现
                error = f"{type(exc).__name__}: {exc}"
                results.append({"status": -1, "body": {"error": error}})
        out[command["id"]] = {
            "cmd": command["cmd"],
            "cat": command.get("cat", ""),
            "extra": command.get("extra") or command.get("judgement", ""),
            "expected_frames": command.get("expected_frames", 150),
            "guidance": guidance,
            "results": results,
            "error": error,
        }
    return out


@pytest.fixture(scope="session")
def acceptance_gt_bank(acceptance_cfg, acceptance_generator):
    """GT 参考库（train 窗口 → VAE 潜变量描述子），供 M-06/E-01 自类聚用。"""
    import torch  # noqa: PLC0415

    from ttm.data.features import NormStats

    from acceptance_support import (  # type: ignore[attr-defined]  # tests/ 目录在 sys.path
        load_condition_windows,
        load_labels_merged,
    )

    labels = load_labels_merged()
    stats = NormStats.load(acceptance_cfg.norm_stats_path())
    device = next(acceptance_generator.vae.parameters()).device

    def encode_condition(condition: str, *, split: str = "train", limit: int | None = None):
        feats = load_condition_windows(condition, split=split, labels=labels, limit=limit)
        mu_list = []
        with torch.no_grad():
            for start in range(0, feats.shape[0], 64):
                chunk = torch.from_numpy(feats[start : start + 64].transpose(0, 2, 1)).to(device)
                mu, _ = acceptance_generator.vae.encode(chunk)
                mu_list.append(mu.mean(dim=1).float().cpu().numpy())
        out = np.concatenate(mu_list, axis=0)
        norms = np.linalg.norm(out, axis=1, keepdims=True)
        return out / np.clip(norms, 1e-12, None), stats

    return {"labels": labels, "stats": stats, "encode_condition": encode_condition}


def _torch_available() -> bool:
    """torch 是否可用（延迟导入，避免影响无 torch 环境）。"""
    try:
        import torch  # noqa: PLC0415

        return True
    except Exception:  # noqa: BLE001
        return False


def _zdrive_available() -> bool:
    """只读网络盘是否可达。"""
    return Path("Z:/1000hours").is_dir()


def pytest_collection_modifyitems(config: pytest.Config, items: list[pytest.Item]) -> None:
    """按环境自动 SKIP 相关用例。"""
    torch_ok = _torch_available()
    zdrive_ok = _zdrive_available()
    skip_torch = pytest.mark.skip(reason="SKIP_IF_NO_TORCH: torch 未安装（等待安装后回归）")
    skip_z = pytest.mark.skip(reason="Z:/1000hours 不可达（离线环境）")
    for item in items:
        if "requires_torch" in item.keywords and not torch_ok:
            item.add_marker(skip_torch)
        if "requires_zdrive" in item.keywords and not zdrive_ok:
            item.add_marker(skip_z)
