"""T03 单测：BVH 解析 / 脏文件拒绝 / 重采样 / 扫描索引与切分。

全部离线可跑：合成 BVH 由 ``tests/conftest.py`` 程序化生成，不依赖 Z 盘。
"""

from __future__ import annotations

import time
from pathlib import Path

import numpy as np
import pytest
from conftest import (
    build_bad_channels_bvh_text,
    build_bvh_text,
    build_few_joints_bvh_text,
    build_motion_arrays,
    write_bvh,
    write_config,
)

from ttm.common import constants as C
from ttm.common import paths as P
from ttm.common.config import load_config
from ttm.data.bvh import BvhParser, InvalidBvh, load_bvh, peek_bvh
from ttm.data.scan import ScanEntry, Scanner, count_windows, read_index, split_for_take

# ---------------------------------------------------------------------------
# 1. 正常解析
# ---------------------------------------------------------------------------


def test_parse_standard_bvh(synth_bvh_file: Path) -> None:
    """标准 23 关节文件：形状/数值/元信息全部正确。"""
    document = BvhParser.parse(synth_bvh_file)

    assert document.n_frames == 48
    assert document.n_joints == 23
    assert document.euler_deg.shape == (48, 23, 3)
    assert document.root_pos_cm.shape == (48, 3)
    assert document.euler_order == "YXZ"
    assert document.root_name == "Hips"
    assert document.fps == pytest.approx(120.0)
    assert document.duration_s == pytest.approx(47 / 120.0)
    assert document.skeleton.n_joints == 23

    root_expected, euler_expected = build_motion_arrays(48, fps=120.0)
    assert np.allclose(document.root_pos_cm, root_expected, atol=1e-4)
    assert np.allclose(document.euler_deg[:, 1:], euler_expected[:, 1:], atol=1e-4)
    # 根关节的旋转通道也应写进 euler_deg[:,0]
    assert np.allclose(document.euler_deg[:, 0], euler_expected[:, 0], atol=1e-4)


def test_parse_rest_offsets_from_file(synth_bvh_file: Path) -> None:
    """rest 偏移必须来自文件（不是默认表）。"""
    document = BvhParser.parse(synth_bvh_file)
    offsets = document.skeleton.rest_offsets_cm
    assert offsets.shape == (23, 3)
    # 合成文件里 Spine 的偏移是 (0, 8.08814, -2.57809)
    assert offsets[1] == pytest.approx([0.0, 8.08814, -2.57809], abs=1e-6)
    assert offsets[0] == pytest.approx([0.0, 0.0, 0.0], abs=1e-12)


def test_parse_omits_end_sites(synth_bvh_file: Path) -> None:
    """``End Site`` 块不得被计为关节（否则 n_joints 会变成 28）。"""
    header = peek_bvh(synth_bvh_file)
    assert header.n_joints == 23
    assert header.joint_names[0] == "Hips"
    assert header.joint_names[-1] == "RightToeBase"
    assert header.total_channels == 72


def test_parse_tolerates_custom_root_name(work_dir: Path) -> None:
    """ROOT 名字不是 Hips 时按位置对齐（真实数据里的 root-only 文件就是这样）。"""
    path = write_bvh(work_dir / "custom_root.bvh", build_bvh_text(n_frames=8, root_name="ActorRoot"))
    document = BvhParser.parse(path)
    assert document.root_name == "ActorRoot"
    assert document.n_joints == 23
    assert document.skeleton.bone_names == C.BONE_ORDER


def test_parse_without_end_sites(work_dir: Path) -> None:
    """没有 End Site 的文件同样能解析。"""
    path = write_bvh(
        work_dir / "no_end_site.bvh", build_bvh_text(n_frames=8, with_end_sites=False)
    )
    document = BvhParser.parse(path)
    assert document.n_joints == 23


def test_to_quaternions_and_world(synth_bvh_file: Path) -> None:
    """四元数与 FK 输出形状/取值合法。"""
    document = BvhParser.parse(synth_bvh_file)
    quats = document.to_quaternions()
    assert quats.shape == (48, 23, 4)
    assert np.allclose(np.linalg.norm(quats, axis=-1), 1.0, atol=1e-6)

    world = document.world_positions()
    assert world.shape == (48, 23, 3)
    assert np.all(np.isfinite(world))
    # Hips 世界坐标必须等于根位移
    assert np.allclose(world[:, 0], document.root_pos_cm, atol=1e-6)


def test_parse_static_motion_exact(work_dir: Path) -> None:
    """静态动作（零旋转）下 FK 结果可精确预期。"""
    path = write_bvh(work_dir / "static.bvh", build_bvh_text(n_frames=4, animated=False))
    document = BvhParser.parse(path)
    assert np.allclose(document.euler_deg, 0.0)
    world = document.world_positions()
    offsets = document.skeleton.rest_offsets_cm
    expected_hips = np.array([0.0, 91.7, 0.0])
    assert np.allclose(world[0, 0], expected_hips, atol=1e-6)
    # 头在髋之上、脚在髋之下
    assert world[0, C.bone_index("Head")][1] > expected_hips[1]
    assert world[0, C.bone_index("LeftFoot")][1] < expected_hips[1]
    assert offsets[0].tolist() == [0.0, 0.0, 0.0]


def test_long_file_streaming(long_bvh_file: Path) -> None:
    """11,802 帧（武术套路实测规模）能完整解析，且不需要二次读文件。"""
    started = time.perf_counter()
    document = BvhParser.parse(long_bvh_file)
    elapsed = time.perf_counter() - started
    assert document.n_frames == 11802
    assert document.euler_deg.shape == (11802, 23, 3)
    assert np.all(np.isfinite(document.euler_deg))
    # 本地盘解析 11802 帧应远快于 10 秒（网络盘另算）
    assert elapsed < 10.0, f"解析耗时 {elapsed:.2f}s 异常"


# ---------------------------------------------------------------------------
# 2. 脏文件拒绝
# ---------------------------------------------------------------------------


def test_root_only_file_rejected(root_only_bvh_file: Path) -> None:
    """实测过的 root-only 轨迹文件（6 通道、无 JOINT）必须被拒绝并给出原因。"""
    with pytest.raises(InvalidBvh) as excinfo:
        BvhParser.parse(root_only_bvh_file)
    assert "n_joints" in excinfo.value.reason
    assert "1" in excinfo.value.reason

    header = BvhParser.peek(root_only_bvh_file)
    assert header.n_joints == 1
    assert header.is_usable is False
    assert "n_joints" in header.rejection_reason()


def test_few_joints_rejected(work_dir: Path) -> None:
    """只有 2 个关节的文件必须被拒绝。"""
    path = write_bvh(work_dir / "two_joints.bvh", build_few_joints_bvh_text())
    with pytest.raises(InvalidBvh) as excinfo:
        BvhParser.parse(path)
    assert "2 != 23" in excinfo.value.reason or "n_joints" in excinfo.value.reason


def test_bad_channels_rejected(work_dir: Path) -> None:
    """非根关节带位置通道必须被拒绝（通道语义无法解释）。"""
    path = write_bvh(work_dir / "bad_channels.bvh", build_bad_channels_bvh_text())
    with pytest.raises(InvalidBvh) as excinfo:
        BvhParser.parse(path)
    assert "通道" in excinfo.value.reason


def test_missing_motion_rejected(work_dir: Path) -> None:
    """缺少 MOTION 小节必须被拒绝。"""
    path = write_bvh(work_dir / "no_motion.bvh", "HIERARCHY\nROOT Hips\n{\n\tOFFSET 0 0 0\n}\n")
    with pytest.raises(InvalidBvh) as excinfo:
        BvhParser.parse(path)
    assert "MOTION" in excinfo.value.reason


def test_empty_motion_rejected(work_dir: Path) -> None:
    """MOTION 声明了帧数但没有数据必须被拒绝。"""
    text = build_bvh_text(n_frames=4).split("MOTION")[0]
    text += "MOTION\nFrames: 4\nFrame Time: 0.00833333\n"
    path = write_bvh(work_dir / "empty_motion.bvh", text)
    with pytest.raises(InvalidBvh):
        BvhParser.parse(path)


def test_nonexistent_file_rejected(work_dir: Path) -> None:
    """不存在的文件抛出 InvalidBvh（而不是 FileNotFoundError）。"""
    with pytest.raises(InvalidBvh):
        BvhParser.parse(work_dir / "does_not_exist.bvh")


# ---------------------------------------------------------------------------
# 3. 重采样（120 → 30）
# ---------------------------------------------------------------------------


def test_resample_to_30fps(work_dir: Path) -> None:
    """120fps 源 → 30fps：帧数为 1/4，数值落在源采样点上。"""
    path = write_bvh(work_dir / "resample.bvh", build_bvh_text(n_frames=481, animated=False))
    document = BvhParser.parse(path)
    assert document.n_frames == 481

    euler_30, root_30 = document.resample(30)
    assert euler_30.shape == (121, 23, 3)
    assert root_30.shape == (121, 3)
    # 静态动作重采样后仍是同一个静止姿态
    assert np.allclose(root_30, np.array([0.0, 91.7, 0.0]), atol=1e-4)
    assert np.allclose(euler_30, 0.0, atol=1e-6)


def test_resample_preserves_signal(work_dir: Path) -> None:
    """重采样必须精确抽取每 4 帧（4× 整除），误差 < 1e-3 度。"""
    path = write_bvh(work_dir / "resample2.bvh", build_bvh_text(n_frames=241))
    document = BvhParser.parse(path)
    euler_30, _ = document.resample(30)
    expected = document.euler_deg[::4]
    assert euler_30.shape[0] == expected.shape[0]
    assert np.abs(euler_30 - expected).max() < 1e-3


# ---------------------------------------------------------------------------
# 4. 窗口计数与切分
# ---------------------------------------------------------------------------


@pytest.mark.parametrize(
    ("n_frames", "expected"),
    [
        (0, 0),
        (74, 0),
        (75, 1),  # 不足 1 整窗，但 >= min_tail → pad+mask 尾窗
        (149, 1),
        (150, 1),  # 恰好 1 整窗
        (224, 1),  # 1 整窗 + 74 帧余量（< 75 丢弃）
        (225, 2),  # 1 整窗 + 75 帧尾窗
        (300, 2),  # 2 整窗 + 0 余量
        (360, 2),  # 2 整窗 + 60 帧余量（< 75 丢弃）
        (450, 3),  # 3 整窗 + 0 余量
    ],
)
def test_count_windows(n_frames: int, expected: int) -> None:
    """§5.5 窗口切分规则：步长 = 窗口（非重叠），尾窗 ≥75 帧保留。"""
    assert count_windows(n_frames, C.WINDOW, C.MIN_TAIL_FRAMES) == expected


def test_split_for_take_is_deterministic_and_disjoint() -> None:
    """同一 take_id 永远落在同一集；大量 take 的三集比例接近 8:1:1。"""
    take_ids = [f"BVH_Body/Take_{idx:05d}" for idx in range(2000)]
    assignments = {take_id: split_for_take(take_id, 0.1, 0.1) for take_id in take_ids}
    # 幂等
    for take_id in take_ids[:100]:
        assert split_for_take(take_id, 0.1, 0.1) == assignments[take_id]
    counts = {name: 0 for name in ("train", "val", "test")}
    for split in assignments.values():
        counts[split] += 1
    total = len(take_ids)
    assert counts["train"] / total == pytest.approx(0.8, abs=0.05)
    assert counts["val"] / total == pytest.approx(0.1, abs=0.04)
    assert counts["test"] / total == pytest.approx(0.1, abs=0.04)
    assert sum(counts.values()) == total


# ---------------------------------------------------------------------------
# 5. 扫描器（离线合成目录树）
# ---------------------------------------------------------------------------


def _build_dataset_tree(root: Path) -> dict[str, int]:
    """构造一棵仿真的数据目录树，返回各类文件的期望数量。

    结构（对齐真实数据的层级）::

        <root>/1.7.Daily_Life_Behavior/1.7.3.Walking_Movement_With_Posture/
              1.7.3.1.Basic_Gait/20251211/Take_001/{a0.bvh,a1.bvh}
              1.7.3.1.Basic_Gait/20251211/Take_002/{a0.bvh}
              1.7.3.1.Basic_Gait/20251211/Take_003/            （空目录）
              1.7.3.3.Stairs_Up_Down/20260326/Take_114/{stair_small_0.bvh}   （root-only）
              1.7.3.3.Stairs_Up_Down/20260326/Take_115/{stair_0.bvh}          （正常）
    """
    base = (
        root
        / "1.7.Daily_Life_Behavior"
        / "1.7.3.Walking_Movement_With_Posture"
    )
    gait = base / "1.7.3.1.Basic_Gait" / "20251211"
    stairs = base / "1.7.3.3.Stairs_Up_Down" / "20260326"

    write_bvh(gait / "Take_001" / "Take_001_actor_0.bvh", build_bvh_text(n_frames=481))
    write_bvh(gait / "Take_001" / "Take_001_actor_1.bvh", build_bvh_text(n_frames=481))
    write_bvh(gait / "Take_002" / "Take_002_actor_0.bvh", build_bvh_text(n_frames=601))
    (gait / "Take_003").mkdir(parents=True, exist_ok=True)  # 空 Take 目录
    write_bvh(
        stairs / "Take_114" / "Take_114_stair_small_0.bvh",
        "HIERARCHY\nROOT stair_small_0\n{\n\tOFFSET 0 0 0\n"
        "\tCHANNELS 6 Xposition Yposition Zposition Yrotation Xrotation Zrotation\n}\n"
        "MOTION\nFrames: 120\nFrame Time: 0.00833333\n" + "0 0 0 0 0 0\n" * 120,
    )
    write_bvh(stairs / "Take_115" / "Take_115_stair_0.bvh", build_bvh_text(n_frames=1201))
    return {"valid_takes": 3, "valid_files": 4, "bad_files": 1, "empty_takes": 1}


def test_scanner_end_to_end(work_dir: Path) -> None:
    """扫描器：Take 去重生效、坏文件入 failed、空目录跳过、三集 take 无交集。"""
    dataset_root = work_dir / "BVH_Body"
    expected = _build_dataset_tree(dataset_root)

    config_path = write_config(
        work_dir / "scan_cfg.yaml",
        bvh_roots=[dataset_root.as_posix()],
        data_dir="data/unit_test_scan",
        max_files=100,
    )
    cfg = load_config(config_path)
    scanner = Scanner(cfg, workers=2)
    result = scanner.run()

    # 3 个有效 Take（Take_001/002/115），每 Take 只留 1 个演员；Take_114 是坏文件
    assert len(result.entries) == expected["valid_takes"]
    assert len(result.failed) == expected["bad_files"]
    assert result.stats["takes_empty"] == expected["empty_takes"]
    assert result.stats["takes_deduped"] >= 1
    assert result.stats["clips_rejected_header"] == 1

    # 每个 take 只出现一次（去重）
    take_ids = [entry.take_id for entry in result.entries]
    assert len(take_ids) == len(set(take_ids))

    # 40 秒 clip 应有 2 个窗口（1201 帧 @120fps → 301 帧 @30fps → 301//150=2，尾 1 帧丢弃）
    by_actor = {entry.actor: entry for entry in result.entries}
    assert by_actor["Take_115_stair_0"].n_frames == 301
    assert by_actor["Take_115_stair_0"].n_windows == 2
    assert by_actor["Take_001_actor_0"].n_frames == 121
    assert by_actor["Take_001_actor_0"].n_windows == 1

    # 坏文件原因可读
    assert "n_joints" in result.failed[0]["error"]

    # 三集 take_id 无交集
    train = result.take_ids("train")
    val = result.take_ids("val")
    test = result.take_ids("test")
    assert not (train & val)
    assert not (train & test)
    assert not (val & test)
    assert len(train | val | test) == len(take_ids)

    # 落盘并被读回
    index_path, failed_path = scanner.write(result)
    assert index_path.exists() and failed_path.exists()
    records = read_index(index_path)
    assert len(records) == len(result.entries)
    required = {
        "clip_id",
        "path",
        "take_id",
        "split",
        "zh_label",
        "category_l1",
        "category_l2",
        "category_l3",
        "n_frames",
        "n_windows",
        "source",
    }
    assert required <= set(records[0])
    assert records[0]["split"] in {"train", "val", "test"}
    assert records[0]["n_frames"] > 0
    assert records[0]["category_l1"] == "1.7.Daily_Life_Behavior"
    assert records[0]["category_l3"] in {
        "1.7.3.1.Basic_Gait",
        "1.7.3.3.Stairs_Up_Down",
    }

    # 清理产物，避免影响其它用例
    import shutil

    shutil.rmtree(P.project_root() / "data" / "unit_test_scan", ignore_errors=True)


def test_scanner_respects_include_categories(work_dir: Path) -> None:
    """类别白名单只放行命中的子树。"""
    dataset_root = work_dir / "BVH_Body"
    _build_dataset_tree(dataset_root)

    config_path = write_config(
        work_dir / "scan_cfg2.yaml",
        bvh_roots=[dataset_root.as_posix()],
        data_dir="data/unit_test_scan2",
        include_categories=["1.7.Daily_Life_Behavior/1.7.3.Walking_Movement_With_Posture/1.7.3.3.Stairs_Up_Down"],
        max_files=100,
    )
    cfg = load_config(config_path)
    result = Scanner(cfg, workers=1).run()
    assert len(result.entries) == 1
    assert result.entries[0].category_l3 == "1.7.3.3.Stairs_Up_Down"

    import shutil

    shutil.rmtree(P.project_root() / "data" / "unit_test_scan2", ignore_errors=True)


def test_scanner_limit(work_dir: Path) -> None:
    """``--limit`` 语义：最多采集 N 个 clip。"""
    dataset_root = work_dir / "BVH_Body"
    _build_dataset_tree(dataset_root)
    config_path = write_config(
        work_dir / "scan_cfg3.yaml",
        bvh_roots=[dataset_root.as_posix()],
        data_dir="data/unit_test_scan3",
        max_files=100,
    )
    cfg = load_config(config_path)
    result = Scanner(cfg, limit=2, workers=1).run()
    assert len(result.entries) == 2

    import shutil

    shutil.rmtree(P.project_root() / "data" / "unit_test_scan3", ignore_errors=True)


def test_scanner_missing_root_is_tolerated(work_dir: Path) -> None:
    """BVH 根目录不存在时只告警，不崩溃。"""
    config_path = write_config(
        work_dir / "scan_cfg4.yaml",
        bvh_roots=[(work_dir / "nonexistent_root").as_posix()],
        data_dir="data/unit_test_scan4",
    )
    cfg = load_config(config_path)
    result = Scanner(cfg, workers=1).run()
    assert result.entries == []
    assert result.failed == []
    assert result.stats["clips_selected"] == 0


def test_scan_entry_rejects_unknown_field() -> None:
    """ScanEntry 是 frozen dataclass，字段改动必须显式。"""
    entry = ScanEntry(
        clip_id="a/b",
        path="p",
        take_id="t",
        split="train",
        zh_label="走",
        category_l1="l1",
        category_l2="l2",
        category_l3="l3",
        n_frames=150,
        n_windows=1,
        source="s",
    )
    assert entry.to_dict()["split"] == "train"
    with pytest.raises(Exception):
        entry.split = "val"  # type: ignore[misc]


# ---------------------------------------------------------------------------
# 6. 真实数据（Z 盘可达时才跑）
# ---------------------------------------------------------------------------


@pytest.mark.requires_zdrive
def test_parse_real_wushu_file() -> None:
    """实测武术文件：23 关节、11,802 帧（T03 验收标准 ①）。"""
    path = "Z:/1000hours/4.wushu/4.3/Bajiquan/wuda_017_Skeleton1.bvh"
    document = load_bvh(path)
    assert document.n_joints == C.N_JOINTS
    assert document.n_frames == 11802
    assert document.euler_deg.shape == (11802, 23, 3)
    assert document.root_pos_cm.shape == (11802, 3)
    assert document.fps == pytest.approx(120.0, abs=1e-3)


@pytest.mark.requires_zdrive
def test_reject_real_root_only_file() -> None:
    """实测 root-only 文件必须被拒绝（T03 验收标准 ②）。"""
    path = (
        "Z:/1000hours/BVH_Body/1.7.Daily_Life_Behavior/1.7.3.Walking_Movement_With_Posture/"
        "1.7.3.3.Stairs_Up_Down/20260326/Take_114/Take_114_stair_small_0.bvh"
    )
    if not Path(path).is_file():
        pytest.skip("样本文件不可达")
    with pytest.raises(InvalidBvh) as excinfo:
        load_bvh(path)
    assert excinfo.value.reason
