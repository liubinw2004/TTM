"""由 tools/ue_import.py 生成：把 TTM 生成的 BVH 批量导入 UE 为 AnimSequence。

在 UE 编辑器里执行：

    py "deliverables/ue/import_unreal.py"

前置：目标 Skeleton 资产存在（默认 /Game/Characters/Celeste/SK_Celeste_Skeleton）。
若路径不同，改下面的 TARGET_SKELETON 即可。
"""

import unreal

TARGET_SKELETON = r"/Game/Characters/Celeste/SK_Celeste_Skeleton"
DESTINATION = r"/Game/TTM/Animations"
FPS = 30

BVH_FILES = [
    r'D:/TTM/deliverables/ue/00_打一套长拳第七段.bvh',
    r'D:/TTM/deliverables/ue/01_打一套太极拳.bvh',
    r'D:/TTM/deliverables/ue/02_慢慢走上楼梯.bvh',
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
