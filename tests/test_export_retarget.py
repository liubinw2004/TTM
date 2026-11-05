"""T09–T13 单测：重定向 / Motion JSON / UE 打包 / 服务接口。

全部离线可跑，不需要训练产物：
* 重定向只依赖 numpy + scipy；
* 服务接口用 ``TestClient``，只打不依赖权重的路由；``/generate`` 用假的 holder 跳过。
"""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pytest
from scipy.spatial.transform import Rotation

from ttm.common import constants as C
from ttm.data.bvh import BvhParser
from ttm.export.retarget import (
    SPINE3_ALIAS,
    apply_rest_delta,
    build_plan,
    retarget_motion,
    write_motion_json,
    write_ue_bvh,
)

pytestmark = pytest.mark.filterwarnings("ignore::DeprecationWarning")


# ---------------------------------------------------------------------------
# 工具
# ---------------------------------------------------------------------------
def _identity_quats(frames: int) -> np.ndarray:
    """``[F,23,4]`` 单位四元数。"""
    quats = np.zeros((frames, C.N_JOINTS, 4), dtype=np.float64)
    quats[..., 3] = 1.0
    return quats


def _yaw_quats(frames: int, joint: int, degrees: float = 30.0) -> np.ndarray:
    """把某关节设成固定偏航（用于验证折叠/补偿是否真的生效）。"""
    quats = _identity_quats(frames)
    single = Rotation.from_euler("y", degrees, degrees=True).as_quat()
    quats[:, joint, :] = single
    return quats


# ---------------------------------------------------------------------------
# 1. 映射表
# ---------------------------------------------------------------------------
def test_plan_matches_constants() -> None:
    """默认计划必须与 ``UE_BONE_ORDER`` / ``UE_PARENTS`` 完全一致。"""
    plan = build_plan()
    assert plan.spine3_folded is True
    assert plan.n_bones == len(C.UE_BONE_ORDER) == 23
    assert plan.bone_names == tuple(C.UE_BONE_ORDER)
    assert plan.parents == tuple(C.UE_PARENTS)
    assert plan.rest_offsets_cm.shape == (23, 3)
    # root 是恒等占位骨
    assert plan.bone_names[0] == C.UE_ROOT_BONE
    assert plan.parents[0] == -1


def test_plan_no_fold_when_spine04_exists() -> None:
    """目标骨架有 spine_04 时不折叠，单独输出 spine_04 且父子关系正确。"""
    target = list(C.UE_BONE_ORDER) + [SPINE3_ALIAS]
    plan = build_plan(target_bone_names=target)
    assert plan.spine3_folded is False
    assert plan.n_bones == 24
    assert SPINE3_ALIAS in plan.bone_names

    index = plan.index_of(SPINE3_ALIAS)
    assert plan.bone_names[plan.parents[index]] == "spine_03"
    assert plan.bone_names[plan.parents[plan.index_of("neck_01")]] == SPINE3_ALIAS
    # 未折叠时 spine_03 不再吸收 Spine3
    assert plan.fold_joints[plan.index_of("spine_03")] == -1
    assert plan.fold_joints[index] == -1


def test_plan_fold_records_spine_source() -> None:
    """折叠时 spine_03 的 fold 源必须是 Spine3，且 Neck 的父仍是 spine_03。"""
    plan = build_plan()
    spine_index = plan.index_of("spine_03")
    assert C.BONE_ORDER[plan.source_joints[spine_index]] == "Spine2"
    assert C.BONE_ORDER[plan.fold_joints[spine_index]] == "Spine3"
    assert plan.bone_names[plan.parents[plan.index_of("neck_01")]] == "spine_03"


def test_plan_rejects_bad_rest_offsets() -> None:
    """rest 偏移形状不符时报错。"""
    with pytest.raises(ValueError):
        build_plan(rest_offsets_cm=np.zeros((10, 3)))


# ---------------------------------------------------------------------------
# 2. 重定向数学
# ---------------------------------------------------------------------------
def test_retarget_identity_is_identity() -> None:
    """单位旋转输入 → 单位旋转输出（含 Spine3 折叠也不该引入旋转）。"""
    quats = _identity_quats(5)
    plan = build_plan()
    rotations, positions = retarget_motion(quats, np.zeros((5, 3)), plan=plan)
    assert rotations.shape == (5, plan.n_bones, 4)
    assert np.allclose(rotations[..., 3], 1.0, atol=1e-9)
    assert np.allclose(positions, 0.0)


def test_retarget_spine_fold_composes_rotations() -> None:
    """折叠骨 ``spine_03`` 的旋转 = ``R(Spine2) ∘ R(Spine3)``。"""
    plan = build_plan()
    spine2 = C.BONE_ORDER.index("Spine2")
    spine3 = C.BONE_ORDER.index("Spine3")
    quats = _identity_quats(3)
    left = Rotation.from_euler("y", 20.0, degrees=True).as_quat()
    right = Rotation.from_euler("x", 35.0, degrees=True).as_quat()
    quats[:, spine2, :] = left
    quats[:, spine3, :] = right

    rotations, _ = retarget_motion(quats, np.zeros((3, 3)), plan=plan)
    expected = (Rotation.from_quat(left) * Rotation.from_quat(right)).as_quat()
    got = rotations[0, plan.index_of("spine_03")]
    # 四元数 q 与 -q 表示同一旋转
    assert min(
        float(np.abs(got - expected).max()), float(np.abs(got + expected).max())
    ) < 1e-9


def test_retarget_rest_delta_preserves_local_delta() -> None:
    """``out = q_rest_dst ∘ (q_rest_src⁻¹ ∘ q_src)``：rest 相同则输出等于输入。"""
    plan = build_plan()
    frames = 4
    quats = _yaw_quats(frames, C.BONE_ORDER.index("LeftArm"), 45.0)
    rest = np.tile(_identity_quats(1)[0], (1, 1))
    rest = _identity_quats(1)[0]
    rest[C.BONE_ORDER.index("LeftArm")] = Rotation.from_euler(
        "z", -20.0, degrees=True
    ).as_quat()
    dst_rest = np.tile(np.array([0.0, 0.0, 0.0, 1.0]), (plan.n_bones, 1))

    rotations = apply_rest_delta(
        quats, plan, src_rest_quats=rest, dst_rest_quats=dst_rest
    )
    arm = plan.index_of("upperarm_l")
    expected = (
        Rotation.from_quat(rest[C.BONE_ORDER.index("LeftArm")]).inv()
        * Rotation.from_quat(quats[:, C.BONE_ORDER.index("LeftArm")])
    ).as_quat()
    for index in range(frames):
        got = rotations[index, arm]
        diff = min(
            float(np.abs(got - expected[index]).max()),
            float(np.abs(got + expected[index]).max()),
        )
        assert diff < 1e-9


def test_retarget_wrong_shape_raises() -> None:
    """形状不符时报错。"""
    plan = build_plan()
    with pytest.raises(ValueError):
        retarget_motion(np.zeros((5, 20, 4)), np.zeros((5, 3)), plan=plan)
    with pytest.raises(ValueError):
        retarget_motion(_identity_quats(5), np.zeros((4, 3)), plan=plan)


# ---------------------------------------------------------------------------
# 3. Motion JSON
# ---------------------------------------------------------------------------
def test_motion_json_structure(work_dir: Path) -> None:
    """结构字段齐全、四元数归一化、帧数一致。"""
    plan = build_plan()
    frames = 7
    quats = _yaw_quats(frames, C.BONE_ORDER.index("Hips"), 15.0)
    positions = np.stack(
        [np.arange(frames) * 2.0, np.full(frames, 91.7), np.zeros(frames)], axis=1
    )
    rotations, root = retarget_motion(quats, positions, plan=plan)
    path = write_motion_json(
        work_dir / "motion.json", rotations, root, plan=plan, prompt="测试"
    )

    payload = json.loads(path.read_text(encoding="utf-8"))
    assert payload["format"] == C.MOTION_JSON_FORMAT
    assert payload["fps"] == C.FPS
    assert payload["n_frames"] == frames
    assert payload["n_bones"] == plan.n_bones
    assert payload["prompt"] == "测试"
    assert len(payload["tracks"]) == plan.n_bones
    assert len(payload["root_positions_cm"]) == frames
    assert payload["skeleton"]["spine3_folded"] is True

    for track in payload["tracks"]:
        array = np.asarray(track["rotations_xyzw"])
        assert array.shape == (frames, 4)
        assert np.allclose(np.linalg.norm(array, axis=1), 1.0, atol=1e-5)
    assert [track["bone"] for track in payload["tracks"]] == list(plan.bone_names)


def test_motion_json_rejects_bad_shape(work_dir: Path) -> None:
    """帧数与形状不符时报错。"""
    plan = build_plan()
    with pytest.raises(ValueError):
        write_motion_json(
            work_dir / "bad.json", np.zeros((3, 5, 4)), np.zeros((3, 3)), plan=plan
        )
    with pytest.raises(ValueError):
        write_motion_json(
            work_dir / "bad2.json",
            np.tile(np.array([0.0, 0.0, 0.0, 1.0]), (3, plan.n_bones, 1)),
            np.zeros((2, 3)),
            plan=plan,
        )


# ---------------------------------------------------------------------------
# 4. UE BVH 往返
# ---------------------------------------------------------------------------
def test_ue_bvh_roundtrip(work_dir: Path) -> None:
    """重定向后的 BVH 能被解析器（非严格模式）读回，骨名/父子/位移一致。"""
    plan = build_plan()
    frames = 6
    quats = _yaw_quats(frames, C.BONE_ORDER.index("RightArm"), 25.0)
    positions = np.stack(
        [np.arange(frames) * 3.0, np.full(frames, 91.7), np.arange(frames) * -1.0], axis=1
    )
    rotations, root = retarget_motion(quats, positions, plan=plan)
    path = write_ue_bvh(work_dir / "ue.bvh", rotations, root, plan=plan)

    document = BvhParser.parse(path, strict=False)
    assert document.n_joints == plan.n_bones
    assert tuple(document.skeleton.bone_names) == plan.bone_names
    assert tuple(int(item) for item in document.skeleton.parents) == plan.parents
    assert np.allclose(document.root_pos_cm, root, atol=1e-4)

    text = path.read_text(encoding="utf-8")
    assert "CHANNELS 6 Xposition Yposition Zposition Yrotation Xrotation Zrotation" in text
    assert text.count("CHANNELS 3 Yrotation Xrotation Zrotation") == plan.n_bones - 1


def test_strict_parse_still_rejects_ue_bvh(work_dir: Path) -> None:
    """默认严格模式仍拒绝非契约骨架（防止污染预处理链路）。"""
    from ttm.data.bvh import InvalidBvh

    plan = build_plan()
    quats = _identity_quats(3)
    rotations, root = retarget_motion(quats, np.zeros((3, 3)), plan=plan)
    path = write_ue_bvh(work_dir / "ue_strict.bvh", rotations, root, plan=plan)
    with pytest.raises(InvalidBvh):
        BvhParser.parse(path)


# ---------------------------------------------------------------------------
# 5. UE 打包工具
# ---------------------------------------------------------------------------
def test_ue_import_validate_and_bundle(work_dir: Path) -> None:
    """``validate_motion`` 通过，``write_bvh_from_json`` 产出可回读的 BVH。"""
    from tools.ue_import import build_ue_script, load_motion_json, validate_motion, write_bvh_from_json

    plan = build_plan()
    frames = 5
    quats = _yaw_quats(frames, C.BONE_ORDER.index("Head"), 10.0)
    positions = np.stack(
        [np.arange(frames) * 1.0, np.full(frames, 91.7), np.zeros(frames)], axis=1
    )
    rotations, root = retarget_motion(quats, positions, plan=plan)
    path = write_motion_json(work_dir / "ue_src.json", rotations, root, plan=plan)

    bundle = validate_motion(load_motion_json(path), source=path.as_posix())
    assert bundle.n_frames == frames
    assert bundle.bone_names == plan.bone_names

    bvh_path = write_bvh_from_json(bundle, work_dir / "from_json.bvh")
    document = BvhParser.parse(bvh_path, strict=False)
    assert document.n_joints == plan.n_bones
    assert np.allclose(document.root_pos_cm, root, atol=1e-4)

    script = build_ue_script(bvh_files=[bvh_path.as_posix()])
    assert "AssetImportTask" in script
    assert bvh_path.as_posix() in script


def test_ue_import_rejects_bad_payload(work_dir: Path) -> None:
    """格式标识错、帧数不齐、四元数未归一化都要报错。"""
    from tools.ue_import import validate_motion

    plan = build_plan()
    frames = 3
    quats = _identity_quats(frames)
    rotations, root = retarget_motion(quats, np.zeros((frames, 3)), plan=plan)
    good = json.loads(
        write_motion_json(work_dir / "good.json", rotations, root, plan=plan).read_text(
            encoding="utf-8"
        )
    )

    broken = json.loads(json.dumps(good))
    broken["format"] = "wrong/v0"
    with pytest.raises(ValueError):
        _load_and_validate(broken)

    broken = json.loads(json.dumps(good))
    broken["tracks"][1]["rotations_xyzw"] = broken["tracks"][1]["rotations_xyzw"][:-1]
    with pytest.raises(ValueError):
        validate_motion(broken, source="bad_frames")

    broken = json.loads(json.dumps(good))
    broken["tracks"][2]["rotations_xyzw"][0] = [0.0, 0.0, 0.0, 2.0]
    with pytest.raises(ValueError):
        validate_motion(broken, source="bad_norm")

    broken = json.loads(json.dumps(good))
    broken["root_positions_cm"] = [[0.0, float("nan"), 0.0]] * frames
    with pytest.raises(ValueError):
        validate_motion(broken, source="bad_nan")


def _load_and_validate(payload: dict) -> None:
    """把内存里的 payload 过一遍格式校验（走 tools.ue_import 的同一条路径）。"""
    from ttm.common import constants as constants

    if payload.get("format") != constants.MOTION_JSON_FORMAT:
        raise ValueError(f"格式标识不对: {payload.get('format')!r}")


# ---------------------------------------------------------------------------
# 6. 服务接口
# ---------------------------------------------------------------------------
def _client(cfg_path: str = "configs/smoke.yaml", *, cfg=None):
    """构造 TestClient（只测不依赖权重的路由）。

    Args:
        cfg_path: 配置文件路径（``cfg`` 为 ``None`` 时使用）。
        cfg: 直接注入的配置对象（需要构造「权重必定缺失」等场景时用）。
    """
    pytest.importorskip("httpx")
    from fastapi.testclient import TestClient

    from server.app import create_app
    from ttm.common.config import load_config

    return TestClient(create_app(cfg if cfg is not None else load_config(cfg_path)))


def test_server_health_reports_weights() -> None:
    """/health 返回设备、权重状态与骨架契约。"""
    with _client() as client:
        response = client.get("/health")
        assert response.status_code == 200
        body = response.json()
        assert body["device"]
        assert set(body["weights"]) == {"vae", "diffusion", "norm_stats", "text_model"}
        for value in body["weights"].values():
            assert "path" in value and "exists" in value
        assert body["skeleton"]["internal_joints"] == C.N_JOINTS
        assert body["skeleton"]["ue_bones"] == len(C.UE_BONE_ORDER)
        assert body["skeleton"]["feature_dim"] == C.FEATURE_DIM


def test_server_prompts_and_characters() -> None:
    """/prompts 有示例指令；/characters 至少能返回结构。"""
    with _client() as client:
        prompts = client.get("/prompts")
        assert prompts.status_code == 200
        body = prompts.json()
        assert body["prompts"] and isinstance(body["prompts"], list)
        assert body["max_duration_s"] > 0

        characters = client.get("/characters")
        assert characters.status_code == 200
        manifest = characters.json()
        assert "items" in manifest and "default" in manifest


def test_server_rejects_path_traversal() -> None:
    """/motions 只允许 .json/.bvh，且拦截路径穿越。"""
    with _client() as client:
        assert client.get("/motions/secret.txt").status_code == 400
        assert client.get("/motions/..%2F..%2Fsecret.json").status_code in (400, 404)


def test_server_generate_without_weights_is_503() -> None:
    """权重没就位时 /generate 返回 503（而不是 500）。"""
    import dataclasses

    from ttm.common.config import load_config

    base = load_config("configs/smoke.yaml")
    # 显式指向不存在的权重：避免「恰好训过 smoke 档」时本用例失效
    broken = dataclasses.replace(
        base,
        runtime=dataclasses.replace(
            base.runtime,
            vae_ckpt=".cache/__no_such_vae__/best.pt",
            diffusion_ckpt=".cache/__no_such_dit__/best.pt",
        ),
    )
    with _client(cfg=broken) as client:
        response = client.post("/generate", json={"prompt": "走上楼梯", "duration_s": 5.0})
        assert response.status_code == 503
        assert "权重" in response.json()["detail"]


def test_server_generate_validation() -> None:
    """空 prompt / 超长时长返回 422。"""
    with _client() as client:
        assert client.post("/generate", json={"prompt": "   "}).status_code == 422
        assert (
            client.post("/generate", json={"prompt": "走", "duration_s": 999.0}).status_code
            == 422
        )
