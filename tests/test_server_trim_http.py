"""服务端层必须真正生效的契约：裁头（F-17）+ ground_origin（M4）+ guidance 回落。

**为什么单独建这个文件**（M4 返工的真实事故）：``generate.py`` 里裁头写好了、单测也
过了，但 ``server/app.py`` 的 ``GenerateOptions`` 没注入 ``trim_head_frames``——于是
「本地测通、服务端没生效」，QA 走 HTTP 拿到的仍是未裁头产物（F-17 = 40.67cm）。

同理 ``guidance`` 曾经在 pydantic 里写死 3.0 而配置是 7.5。

所以这里**只测 HTTP 层**：走 ``POST /generate`` 拿真实产物，断言
① 响应回显的裁头/引导值等于配置；② 帧数不变式（用户时长不变）；
③ 裁头后的首帧最低足踝仍在地面（独立 FK，不看实现内部）；④ 接缝/瞬态指标不退化。
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import numpy as np
import pytest
from scipy.spatial.transform import Rotation

from acceptance_support import parse_bvh, url_to_local  # type: ignore[attr-defined]


def _lowest_foot_y_cm(payload: dict[str, Any], frame: int = 0) -> float:
    """用 JSON 里的 skeleton（parents + rest_offset）做**独立 FK**，返回该帧最低足 Y。

    刻意不复用导出侧代码：只有独立算一遍，才能证明「导出的数不是自己解释自己」。

    Args:
        payload: ``ttm-motion/v1`` 内容。
        frame: 帧号。

    Returns:
        左右脚踝在世界 Y 上的较低者（cm）。
    """
    entries = payload["skeleton"]["bones"]
    names = [item["name"] for item in entries]
    parents = [int(item["parent"]) for item in entries]
    offsets = np.asarray([item["rest_offset_cm"] for item in entries], dtype=np.float64)
    tracks = {
        item["bone"]: np.asarray(item["rotations_xyzw"], dtype=np.float64)
        for item in payload["tracks"]
    }
    roots = np.asarray(payload["root_positions_cm"], dtype=np.float64)

    world_rot: dict[int, np.ndarray] = {}
    world_pos: dict[int, np.ndarray] = {}
    for index, name in enumerate(names):
        local = Rotation.from_quat(tracks[name][frame]).as_matrix()
        parent = parents[index]
        if parent < 0:
            world_rot[index] = local
            world_pos[index] = roots[frame].copy()
        else:
            world_rot[index] = world_rot[parent] @ local
            world_pos[index] = world_pos[parent] + world_rot[parent] @ offsets[index]

    index_of = {name: i for i, name in enumerate(names)}
    feet = [world_pos[index_of["foot_l"]][1], world_pos[index_of["foot_r"]][1]]
    return float(min(feet))


def _startup_jump_cm(root_y: np.ndarray, window: int = 12) -> float:
    """F-17 判据：``|median(f0..f11) − median(f12..f23)|``（cm）。"""
    return abs(float(np.median(root_y[:window])) - float(np.median(root_y[window : 2 * window])))


@pytest.fixture(scope="module")
def stair_clip(acceptance_client, acceptance_cfg) -> dict[str, Any]:
    """走一次真实 HTTP：楼梯指令，返回响应体 / JSON / BVH 三件套。"""
    response = acceptance_client.post(
        "/generate",
        json={
            "prompt": "慢慢走上楼梯",
            "duration_s": 5.0,
            "steps": 50,
            "seed": 43,
            "export": "both",
        },
    )
    assert response.status_code == 200, response.text[:300]
    body = response.json()
    payload = json.loads(
        Path(url_to_local(body["json_url"])).read_text(encoding="utf-8")
    )
    bvh = parse_bvh(url_to_local(body["bvh_url"]))
    return {"body": body, "payload": payload, "root_y": bvh["frames"][:, 1], "cfg": acceptance_cfg}


def test_server_injects_trim_head(stair_clip: dict[str, Any]) -> None:
    """服务端必须把配置里的裁头帧数注入进去，并在响应里回显。"""
    body = stair_clip["body"]
    expected = int(stair_clip["cfg"].runtime.trim_head_frames)
    assert expected > 0, "配置的 trim_head_frames 为 0，本用例失去意义"
    assert body["trim_head_frames"] == expected
    # JSON 侧也要自证：meta 记录裁头帧数与裁头前的帧数
    assert int(stair_clip["payload"]["trim_head_frames"]) == expected
    assert int(stair_clip["payload"]["frames_before_trim"]) == 150 + expected


def test_trim_keeps_user_duration(stair_clip: dict[str, Any]) -> None:
    """裁头**只丢开头**：5s 请求仍返回 150 帧（多申请的那 12 帧不进产物）。"""
    body = stair_clip["body"]
    assert body["n_frames"] == 150
    assert int(stair_clip["payload"]["n_frames"]) == 150
    # 裁头的代价：5s 从 1 窗变 2 窗（stride=120）
    assert body["n_windows"] == 2


def test_server_echoes_guidance_from_config(stair_clip: dict[str, Any]) -> None:
    """不显式传 guidance 时，响应回显配置值（不是 pydantic 默认 3.0）。"""
    body = stair_clip["body"]
    cfg = stair_clip["cfg"]
    assert abs(float(body["guidance"]) - float(cfg.runtime.guidance)) < 1e-9
    assert body["warnings"] == [], f"短指令不应有 warnings: {body['warnings']}"


def test_first_frame_foot_on_ground_after_shift(stair_clip: dict[str, Any]) -> None:
    """ground_origin 平移后，首帧最低足踝 ∈[-2,+2]cm（独立 FK 交叉验证）。"""
    body = stair_clip["body"]
    offset = float(body["ground_origin_offset_cm"])
    before = _lowest_foot_y_cm(stair_clip["payload"], 0)
    # 响应里的 offset 四舍五入到 3 位小数，容差按量化误差给（不是放宽语义）
    assert abs(before - offset) <= 1e-2, "offset 与独立 FK 算出的首帧最低足高不一致"
    assert abs(before - offset) <= 2.0, f"平移后首帧足高 {before - offset:.2f}cm 超 ±2cm"


def test_bvh_root_matches_json_minus_offset(stair_clip: dict[str, Any]) -> None:
    """BVH 只做刚性平移：``bvh_root_y[0] == json_root_y[0] − offset``。"""
    body = stair_clip["body"]
    payload = stair_clip["payload"]
    expected = float(payload["root_positions_cm"][0][1]) - float(body["ground_origin_offset_cm"])
    assert abs(float(stair_clip["root_y"][0]) - expected) <= 1e-3  # BVH 只写 4 位小数


def test_startup_transient_within_criterion(stair_clip: dict[str, Any]) -> None:
    """楼梯类（原本最差 38.08cm）裁头后 F-17 判据通过。"""
    jump = _startup_jump_cm(stair_clip["root_y"])
    assert jump <= 30.0, f"楼梯类启动瞬态 {jump:.2f}cm > 30cm"
