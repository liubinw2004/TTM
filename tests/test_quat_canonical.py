"""F-07：四元数半球规约（``q`` 与 ``-q`` 表示同一旋转，插值却会走长弧）。

背景：QA 实测导出 JSON 里 ``qw < 0`` 占 4.2%（4344/103500，集中在 pelvis /
thigh_r / thigh_l）。BVH 走欧拉角所以完全看不到，属**静默错误**：UE 导入与
前端 slerp 会在异号四元数之间翻转。

本文件锁三件事：
1. :func:`canonicalize_quats` 的语义（翻转 → 归一化，旋转矩阵不变）；
2. ``write_motion_json`` 写出物里 ``qw < 0`` 计数必须为 0（导出层兜底）；
3. 端到端产物（QA E-01 矩阵）里同样为 0。
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import numpy as np
import pytest
from scipy.spatial.transform import Rotation

from ttm.export.retarget import build_plan, canonicalize_quats, write_motion_json

#: 旋转矩阵逐元素一致性的容差（纯粹是浮点噪声）。
MATRIX_TOL = 1e-6

#: JSON 里四元数保留 6 位小数，矩阵比对容差要留出量化误差。
JSON_MATRIX_TOL = 1e-5


def _random_quats(rng: np.random.Generator, frames: int, bones: int) -> np.ndarray:
    """生成 ``[F,B,4]`` 单位四元数（w 随机正负，模拟真实导出）。"""
    quats = rng.normal(size=(frames, bones, 4))
    quats /= np.linalg.norm(quats, axis=-1, keepdims=True)
    return quats


def _rotation_matrices(quats: np.ndarray) -> np.ndarray:
    """``[F,B,4]`` → ``[F,B,3,3]`` 旋转矩阵。"""
    flat = np.asarray(quats, dtype=np.float64).reshape(-1, 4)
    return Rotation.from_quat(flat).as_matrix().reshape(*np.shape(quats)[:-1], 3, 3)


def test_canonicalize_flips_negative_w_and_normalizes() -> None:
    """``qw < 0`` 全部取反，且结果逐帧单位范数。"""
    rng = np.random.default_rng(2024)
    quats = _random_quats(rng, frames=64, bones=23)
    # 强制一半为负半球（真实数据里几乎必然出现）
    quats[::2] *= -1.0
    assert np.count_nonzero(quats[..., 3] < 0.0) > 0

    fixed = canonicalize_quats(quats)

    assert np.count_nonzero(fixed[..., 3] < 0.0) == 0
    assert np.allclose(np.linalg.norm(fixed, axis=-1), 1.0, atol=1e-12)


def test_canonicalize_preserves_rotation_matrix() -> None:
    """翻转前后是同一个旋转：旋转矩阵逐元素一致（≤1e-6）。"""
    rng = np.random.default_rng(7)
    quats = _random_quats(rng, frames=32, bones=23)
    quats[1::3] *= -1.0
    # 混入未归一化输入，验证「先翻转后归一化」不会把误差带进旋转
    quats[5] *= 1.0003

    fixed = canonicalize_quats(quats)
    before = _rotation_matrices(quats)
    after = _rotation_matrices(fixed)

    assert np.abs(before - after).max() <= MATRIX_TOL


def test_canonicalize_does_not_mutate_input() -> None:
    """返回新数组，不修改入参（调用方可能还要用原始数组）。"""
    rng = np.random.default_rng(11)
    quats = _random_quats(rng, frames=8, bones=4)
    quats[0] *= -1.0
    snapshot = quats.copy()

    _ = canonicalize_quats(quats)

    assert np.array_equal(quats, snapshot)


def test_canonicalize_rejects_bad_shape() -> None:
    """形状不是 ``[F,B,4]`` 时立刻报错，不做静默兜底。"""
    with pytest.raises(ValueError, match=r"\[F,B,4\]"):
        canonicalize_quats(np.zeros((10, 3), dtype=np.float64))


def test_write_motion_json_has_no_negative_w(work_dir: Path) -> None:
    """导出物里 ``qw < 0`` 计数为 0，且旋转矩阵与原始一致。"""
    rng = np.random.default_rng(99)
    plan = build_plan()
    frames = 24
    rotations = _random_quats(rng, frames=frames, bones=plan.n_bones)
    rotations[::2] *= -1.0  # 一半负半球
    positions = np.cumsum(rng.normal(scale=1.5, size=(frames, 3)), axis=0)

    target = write_motion_json(
        work_dir / "canon.json",
        rotations,
        positions,
        plan=plan,
        prompt="四元数半球规约",
    )
    payload: dict[str, Any] = json.loads(target.read_text(encoding="utf-8"))

    negative = 0
    for track in payload["tracks"]:
        array = np.asarray(track["rotations_xyzw"], dtype=np.float64)
        negative += int(np.count_nonzero(array[:, 3] < 0.0))
        bone = int(plan.bone_names.index(track["bone"]))
        assert np.abs(_rotation_matrices(array) - _rotation_matrices(rotations[:, bone])).max() <= (
            JSON_MATRIX_TOL
        ), f"骨 {track['bone']} 的旋转在半球规约后发生变化"

    assert negative == 0, f"导出 JSON 仍有 {negative} 个 qw<0"
    # 世界坐标不动（半球规约只碰旋转）
    assert np.allclose(
        np.asarray(payload["root_positions_cm"], dtype=np.float64), positions, atol=1e-6
    )


def test_export_matrix_has_no_negative_w(acceptance_e2e_results: dict[str, dict[str, Any]]) -> None:
    """端到端产物（E-01 矩阵：10 指令 × 3 seed）里 ``qw < 0`` 全为 0。"""
    checked_payloads = 0
    total_quats = 0
    negative = 0

    for entry in acceptance_e2e_results.values():
        for result in entry["results"]:
            payload = result.get("payload") or {}
            if not payload.get("tracks"):
                continue
            checked_payloads += 1
            for track in payload["tracks"]:
                array = np.asarray(track["rotations_xyzw"], dtype=np.float64)
                total_quats += int(array.shape[0])
                negative += int(np.count_nonzero(array[:, 3] < 0.0))

    assert checked_payloads > 0, "E-01 矩阵没有可用产物，无法核对半球一致性"
    assert total_quats > 0
    assert negative == 0, (
        f"{negative}/{total_quats} 个四元数落在负半球"
        f"（{checked_payloads} 份 Motion JSON）"
    )
