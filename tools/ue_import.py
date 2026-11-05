"""T12：把生成的 ``ttm-motion/v1`` 打包成「可直接导进 UE」的产物。

本工具**不依赖 UE**（不需要 unreal 模块），它做三件事：

1. **校验** Motion JSON（格式标识、骨名、四元数归一化、帧数连续）；
2. **转出 BVH**（UE 骨名，供 UE 的 BVH → AnimSequence 导入器吃）；
3. **生成 UE 侧导入脚本**（``import_unreal.py``），在 UE 的 Python 控制台里跑：

   .. code-block:: text

       # UE 编辑器 Output Log → 切到 Python → 执行
       py "D:/TTM/deliverables/ue/import_unreal.py"

产出一份 ``manifest.json`` 记录骨架、帧数、接缝台阶等元数据，便于事后追溯。

用法::

    . D:/TTM/scripts/env.ps1
    # 单个 Motion JSON
    python tools/ue_import.py --motion deliverables/generated/00_xxx.json
    # 批量：整个目录里的 *.json
    python tools/ue_import.py --dir deliverables/generated
    # 只校验不落盘
    python tools/ue_import.py --dir deliverables/generated --check-only

退出码：全部成功 0；有文件校验失败 1；参数/路径问题 2。
"""

from __future__ import annotations

import argparse
import json
import sys
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Sequence

# --- sys.path 引导 -----------------------------------------------------------
_PROJECT_ROOT = Path(__file__).resolve().parents[1]
_SRC_DIR = _PROJECT_ROOT / "src"
if str(_SRC_DIR) not in sys.path:
    sys.path.insert(0, str(_SRC_DIR))

from ttm.common import constants as C  # noqa: E402
from ttm.common import paths as P  # noqa: E402
from ttm.common.logutil import close_all, get_logger  # noqa: E402

__all__ = [
    "MotionBundle",
    "build_ue_script",
    "load_motion_json",
    "main",
    "validate_motion",
    "write_bvh_from_json",
]

#: UE 侧导入脚本的文件名。
UE_SCRIPT_NAME = "import_unreal.py"

#: 四元数归一化容差。
QUAT_TOL = 1e-3


@dataclass
class MotionBundle:
    """一份校验通过的 Motion JSON（已转成 numpy 友好的扁平结构）。"""

    path: Path
    prompt: str
    fps: int
    n_frames: int
    bone_names: tuple[str, ...]
    parents: tuple[int, ...]
    rotations: Any  # np.ndarray [F,B,4]
    root_pos_cm: Any  # np.ndarray [F,3]
    seam_jumps: list[float] = field(default_factory=list)
    meta: dict[str, Any] = field(default_factory=dict)


def load_motion_json(path: str | Path) -> dict[str, Any]:
    """读取并做基础结构校验。

    Args:
        path: Motion JSON 路径。

    Returns:
        原始 dict。

    Raises:
        FileNotFoundError: 文件不存在。
        ValueError: JSON 解析失败或格式标识不对。
    """
    target = Path(path)
    if not target.is_file():
        raise FileNotFoundError(f"Motion JSON 不存在: {target}")
    try:
        payload = json.loads(target.read_text(encoding="utf-8"))
    except json.JSONDecodeError as exc:
        raise ValueError(f"{target.name} 不是合法 JSON: {exc}") from exc
    if not isinstance(payload, dict):
        raise ValueError(f"{target.name} 顶层应为对象，实际 {type(payload).__name__}")
    fmt = payload.get("format")
    if fmt != C.MOTION_JSON_FORMAT:
        raise ValueError(
            f"{target.name} 格式标识为 {fmt!r}，期望 {C.MOTION_JSON_FORMAT!r}"
        )
    return payload


def validate_motion(payload: dict[str, Any], *, source: str = "<memory>") -> MotionBundle:
    """把 Motion JSON 校验并转成 :class:`MotionBundle`。

    校验项：

    * ``tracks`` 里每根骨的帧数一致且等于 ``n_frames``；
    * 四元数模长在 ``1 ± 1e-3``；
    * ``root_positions_cm`` 帧数一致；
    * 骨名不重复。

    Args:
        payload: :func:`load_motion_json` 的结果。
        source: 报错时显示的名字。

    Returns:
        :class:`MotionBundle`。

    Raises:
        ValueError: 任一项校验失败。
    """
    import numpy as np  # noqa: PLC0415 - 仅校验时需要

    n_frames = int(payload.get("n_frames") or 0)
    if n_frames <= 0:
        raise ValueError(f"{source}: n_frames={n_frames} 非法")
    fps = int(payload.get("fps") or C.FPS)
    if fps <= 0:
        raise ValueError(f"{source}: fps={fps} 非法")

    tracks = payload.get("tracks")
    if not isinstance(tracks, list) or not tracks:
        raise ValueError(f"{source}: tracks 缺失或为空")

    names: list[str] = []
    rotations: list[Any] = []
    for index, track in enumerate(tracks):
        if not isinstance(track, dict):
            raise ValueError(f"{source}: tracks[{index}] 不是对象")
        name = str(track.get("bone") or "").strip()
        if not name:
            raise ValueError(f"{source}: tracks[{index}] 缺少 bone")
        if name in names:
            raise ValueError(f"{source}: 骨名重复 {name!r}")
        quats = track.get("rotations_xyzw")
        if not isinstance(quats, list) or len(quats) != n_frames:
            got = len(quats) if isinstance(quats, list) else "非数组"
            raise ValueError(
                f"{source}: 骨 {name} 的帧数 {got} != n_frames {n_frames}"
            )
        array = np.asarray(quats, dtype=np.float64)
        if array.shape != (n_frames, 4):
            raise ValueError(f"{source}: 骨 {name} 的四元数形状 {array.shape} != ({n_frames},4)")
        norms = np.linalg.norm(array, axis=1)
        worst = float(np.abs(norms - 1.0).max())
        if worst > QUAT_TOL:
            raise ValueError(f"{source}: 骨 {name} 的四元数未归一化（最大偏差 {worst:.4f}）")
        names.append(name)
        rotations.append(array)

    positions = payload.get("root_positions_cm")
    if not isinstance(positions, list) or len(positions) != n_frames:
        got = len(positions) if isinstance(positions, list) else "非数组"
        raise ValueError(f"{source}: root_positions_cm 帧数 {got} != {n_frames}")
    root_pos = np.asarray(positions, dtype=np.float64)
    if root_pos.shape != (n_frames, 3):
        raise ValueError(f"{source}: root_positions_cm 形状 {root_pos.shape} != ({n_frames},3)")
    if not np.all(np.isfinite(root_pos)):
        raise ValueError(f"{source}: root_positions_cm 含 NaN/Inf")

    skeleton = payload.get("skeleton") or {}
    bones_meta = skeleton.get("bones") or []
    if len(bones_meta) == len(names):
        parents = tuple(int(item.get("parent", -1)) for item in bones_meta)
        # 拓扑序：父索引必须 < 子索引，否则 UE/BVH 导入会错位
        for index, parent in enumerate(parents):
            if index == 0:
                if parent != -1:
                    raise ValueError(f"{source}: 根骨 parent 应为 -1，实际 {parent}")
                continue
            if not 0 <= parent < index:
                raise ValueError(
                    f"{source}: 骨 {names[index]} 的 parent={parent} 违反拓扑序（应 < {index}）"
                )
    else:
        parents = tuple(-1 if index == 0 else index - 1 for index in range(len(names)))

    return MotionBundle(
        path=Path(source) if source != "<memory>" else Path("."),
        prompt=str(payload.get("prompt") or ""),
        fps=fps,
        n_frames=n_frames,
        bone_names=tuple(names),
        parents=parents,
        rotations=np.stack(rotations, axis=1),  # [F, B, 4]
        root_pos_cm=root_pos,
        seam_jumps=[float(item) for item in payload.get("seam_jumps") or []],
        meta=dict(skeleton),
    )


def write_bvh_from_json(bundle: MotionBundle, path: str | Path) -> Path:
    """把 bundle 写成 UE 骨名的 BVH。

    Args:
        bundle: 校验后的动作。
        path: 输出路径。

    Returns:
        写出的路径。
    """
    import numpy as np  # noqa: PLC0415

    from ttm.export.bvh_io import write_bvh  # noqa: PLC0415

    # 复用 retarget 的 offsets（同名骨 → 同样的 rest 偏移语义）
    offsets = None
    rest_meta = bundle.meta.get("bones")
    if isinstance(rest_meta, list) and len(rest_meta) == len(bundle.bone_names):
        candidates = [item.get("rest_offset_cm") for item in rest_meta]
        if all(isinstance(item, list) and len(item) == 3 for item in candidates):
            offsets = np.asarray(candidates, dtype=np.float64)

    return write_bvh(
        path,
        bundle.rotations,
        bundle.root_pos_cm,
        fps=bundle.fps,
        bone_names=bundle.bone_names,
        parents=bundle.parents,
        offsets_cm=offsets,
        root_name=bundle.bone_names[0] if bundle.bone_names else C.UE_ROOT_BONE,
    )


def build_ue_script(
    *,
    bvh_files: Sequence[str],
    target_skeleton: str = "/Game/Characters/Celeste/SK_Celeste_Skeleton",
    destination: str = "/Game/TTM/Animations",
    fps: int = C.FPS,
) -> str:
    """生成 UE 编辑器里可直接执行的 Python 导入脚本。

    Args:
        bvh_files: 要导入的 BVH 绝对路径列表。
        target_skeleton: 目标 Skeleton 资产路径。
        destination: 导入后 AnimSequence 的落点目录。
        fps: 期望帧率（仅用于校验提示）。

    Returns:
        脚本文本。
    """
    quoted = ",\n    ".join(f"r'{path}'" for path in bvh_files)
    return f'''"""由 tools/ue_import.py 生成：把 TTM 生成的 BVH 批量导入 UE 为 AnimSequence。

在 UE 编辑器里执行：

    py "deliverables/ue/{UE_SCRIPT_NAME}"

前置：目标 Skeleton 资产存在（默认 {target_skeleton}）。
若路径不同，改下面的 TARGET_SKELETON 即可。
"""

import unreal

TARGET_SKELETON = r"{target_skeleton}"
DESTINATION = r"{destination}"
FPS = {fps}

BVH_FILES = [
    {quoted},
]


def import_one(bvh_path):
    """导入单个 BVH，返回导入任务成功与否。"""
    task = unreal.AssetImportTask()
    task.set_editor_property("filename", bvh_path)
    task.set_editor_property("destination_path", DESTINATION)
    task.set_editor_property("automated", True)
    task.set_editor_property("replace_existing", True)
    task.set_editor_property("save", True)

    options = unreal.ImportAssetOptions()
    task.set_editor_property("options", options)

    unreal.AssetToolsHelpers.get_asset_tools().import_asset_tasks([task])
    created = list(task.get_editor_property("imported_object_paths") or [])
    if not created:
        unreal.log_error("导入失败（无产物）: %s" % bvh_path)
        return False

    for asset_path in created:
        asset = unreal.load_asset(asset_path)
        if isinstance(asset, unreal.AnimSequence) and TARGET_SKELETON:
            skeleton = unreal.load_asset(TARGET_SKELETON)
            if skeleton:
                try:
                    asset.set_editor_property("skeleton", skeleton)
                except Exception as error:  # noqa: BLE001
                    unreal.log_warning("设置 Skeleton 失败（%s）: %s" % (asset_path, error))
        unreal.log("已导入: %s" % asset_path)
    return True


def main():
    """批量导入入口。"""
    if not BVH_FILES:
        unreal.log_error("BVH_FILES 为空：先用 tools/ue_import.py 生成清单")
        return 1
    failed = 0
    for path in BVH_FILES:
        try:
            if not import_one(path):
                failed += 1
        except Exception as error:  # noqa: BLE001
            failed += 1
            unreal.log_error("导入异常 %s: %s" % (path, error))
    unreal.log("导入完成：成功 %d / 失败 %d" % (len(BVH_FILES) - failed, failed))
    return 0 if failed == 0 else 1


if __name__ == "__main__":
    main()
'''


def _collect_motion_files(args: argparse.Namespace) -> list[Path]:
    """按命令行参数收集要处理的 Motion JSON。"""
    if args.motion:
        return [Path(item) for item in args.motion]
    roots: list[Path] = []
    if args.dir:
        roots.extend(Path(item) for item in args.dir)
    if not roots:
        return []
    files: list[Path] = []
    for root in roots:
        if root.is_dir():
            files.extend(sorted(root.glob("*.json")))
        elif root.is_file() and root.suffix.lower() == ".json":
            files.append(root)
    return files


def _is_motion_json(path: Path) -> bool:
    """粗判一个 JSON 是不是 ``ttm-motion/v1``（目录模式下用来过滤同目录的报表）。

    目录模式（``--dir``）会把目录下所有 ``*.json`` 都捞进来，而生成目录里通常还
    躺着 ``generate_report.json`` 这种报表。把它们记成「失败」会让整次打包看起来
    有问题（退出码非 0、CI 变红），而实际上 Motion JSON 一个都没坏。所以目录模式下
    先看 ``format`` 字段，不是 ``ttm-motion/v1`` 的直接**跳过并说明**；
    显式 ``--motion`` 指名道姓给的文件仍然按失败处理——那是用户明确要说的话。

    Args:
        path: JSON 路径。

    Returns:
        是 Motion JSON 返回 ``True``。
    """
    try:
        payload = json.loads(Path(path).read_text(encoding="utf-8"))
    except (OSError, ValueError):
        return False
    return isinstance(payload, dict) and payload.get("format") == C.MOTION_JSON_FORMAT


def build_argparser() -> argparse.ArgumentParser:
    """构造命令行参数解析器。"""
    parser = argparse.ArgumentParser(
        description="把 ttm-motion/v1 打包成 UE 可导入的产物（TTM T12）",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument("--motion", action="append", default=None, help="Motion JSON 路径，可重复")
    parser.add_argument("--dir", action="append", default=None, help="目录（取其下 *.json），可重复")
    parser.add_argument("--out", default="deliverables/ue", help="输出目录")
    parser.add_argument(
        "--target-skeleton",
        default="/Game/Characters/Celeste/SK_Celeste_Skeleton",
        help="UE 目标 Skeleton 资产路径（写进导入脚本）",
    )
    parser.add_argument(
        "--destination", default="/Game/TTM/Animations", help="UE 导入落点目录"
    )
    parser.add_argument("--check-only", action="store_true", help="只校验，不写任何文件")
    return parser


def main(argv: list[str] | None = None) -> int:
    """命令行入口。

    Args:
        argv: 参数列表（默认取 ``sys.argv[1:]``）。

    Returns:
        进程退出码。
    """
    args = build_argparser().parse_args(argv)
    log = get_logger("ttm.tools.ue_import", script="ue_import")

    files = _collect_motion_files(args)
    if not files:
        log.error("没有待处理的文件：用 --motion 或 --dir 指定")
        close_all()
        return 2

    out_dir = Path(args.out) if args.check_only else P.safe_makedirs(Path(args.out))
    bundles: list[MotionBundle] = []
    failures: list[tuple[Path, str]] = []
    bvh_paths: list[Path] = []

    print("=" * 88)
    print("TTM → UE 打包")
    print("=" * 88)
    skipped: list[Path] = []
    for path in files:
        # 目录模式下跳过同目录的报表 JSON（见 _is_motion_json）；显式 --motion 不跳
        if not args.motion and not _is_motion_json(path):
            skipped.append(path)
            print(f"[跳过] {path.name}: 不是 {C.MOTION_JSON_FORMAT}（同目录的报表文件）")
            continue
        try:
            payload = load_motion_json(path)
            bundle = validate_motion(payload, source=path.as_posix())
        except (FileNotFoundError, ValueError) as exc:
            failures.append((path, str(exc)))
            print(f"[失败] {path.name}: {exc}")
            continue
        bundles.append(bundle)
        seam = max(bundle.seam_jumps) if bundle.seam_jumps else 0.0
        print(
            f"[通过] {path.name}: {bundle.n_frames} 帧 × {len(bundle.bone_names)} 骨 "
            f"@ {bundle.fps}fps，接缝最大 {seam:.1f} cm/s"
        )
        if not args.check_only:
            bvh_path = write_bvh_from_json(bundle, out_dir / f"{path.stem}.bvh")
            bvh_paths.append(bvh_path)

    if args.check_only:
        print("-" * 88)
        print(
            f"校验完成：通过 {len(bundles)} / 失败 {len(failures)}"
            + (f" / 跳过 {len(skipped)}" if skipped else "")
        )
        print("=" * 88)
        close_all()
        return 0 if not failures else 1

    script_path = out_dir / UE_SCRIPT_NAME
    script_path.write_text(
        build_ue_script(
            bvh_files=[item.as_posix() for item in bvh_paths],
            target_skeleton=args.target_skeleton,
            destination=args.destination,
        ),
        encoding="utf-8",
    )

    manifest = {
        "format": "ttm-ue-bundle/v1",
        "target_skeleton": args.target_skeleton,
        "destination": args.destination,
        "ue_script": script_path.as_posix(),
        "motions": [
            {
                "source": bundle.path.as_posix(),
                "bvh": (out_dir / f"{bundle.path.stem}.bvh").as_posix(),
                "prompt": bundle.prompt,
                "fps": bundle.fps,
                "n_frames": bundle.n_frames,
                "n_bones": len(bundle.bone_names),
                "seam_jumps": bundle.seam_jumps,
                "duration_s": round(bundle.n_frames / float(bundle.fps), 3),
            }
            for bundle in bundles
        ],
        "failures": [{"source": path.as_posix(), "reason": reason} for path, reason in failures],
    }
    manifest_path = P.resolve_out(out_dir / "manifest.json", mkparents=True)
    manifest_path.write_text(json.dumps(manifest, ensure_ascii=False, indent=2), encoding="utf-8")

    print("-" * 88)
    print(f"通过 {len(bundles)} / 失败 {len(failures)}" + (f" / 跳过 {len(skipped)}" if skipped else ""))
    print(f"BVH      : {out_dir.as_posix()}")
    print(f"UE 脚本  : {script_path.as_posix()}")
    print(f"清单     : {manifest_path.as_posix()}")
    print(f"在 UE 里执行: py \"{script_path.as_posix()}\"")
    print("=" * 88)
    close_all()
    return 0 if not failures else 1


if __name__ == "__main__":
    raise SystemExit(main())
