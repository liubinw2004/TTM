"""中文标签生成：目录层级 → 具体中文动作标签 → 模板扩写出多样化中文描述。

三层结构（``TTM-ARCH.md`` §5.6、PRD §3.4）：

1. **细类精确表** :data:`L3_EXACT_ZH` —— 覆盖 ``BVH_Body`` 实测全部 160 个 l3。
   同名 l3 出现在多个 l2 下时（如 ``Pass_Ball`` 同时存在于篮球/排球），值退化为
   ``{l2 归一化键: 中文}`` 字典按 l2 消歧。
2. **武术基名 + 序号展开** :data:`WUSHU_BASE_ZH` / :data:`WUSHU_PLAIN_ZH` ——
   ``4.wushu/4.3`` 实测 191 个类别里 **148 个是「基名+数字」**、只来自 21 个基名，
   剩下的 43 个是无序号单件。因此只需手写 64 条中文基名，序号用中文数字自动展开
   （``Changquan7`` → ``长拳第七段``、``Tai_Chi18`` → ``太极拳第十八式``）。
3. **关键词兜底** :data:`KEYWORD_RULES` —— 应对未来新增的未见细类。

**禁止对已知 l3 落 l1 大类名兜底**：旧版按 ``l3 → l2 → l1`` 顺序取第一个命中，
导致 ``1.12.7.1.Front_Desk_Process`` 之类未收录的 l3 直接变成大类名
「复杂行为动作序列」，实测该类吞掉 9851/22682 = 39% 的窗口，把 40 个细类压成 19 个。
现在解析链是 ``l3 精确 → 武术基名+序号 → l3 关键词 → l2 精确 → l2 关键词 → 机械转写``，
**l1 只用于「l3/l2 全空」的退化情形**，不再参与兜底。

标签与动作一一对应且可追溯：``zh_label`` 是类中心，``zh_texts`` 是同一语义的
多种表述（PRD R1 明确指出这只是「同一动作的不同说法」，不是细粒度标注）。
"""

from __future__ import annotations

import re
from dataclasses import dataclass
from typing import Final, Iterable, Sequence

__all__ = [
    "CATEGORY_ZH",
    "KEYWORD_RULES",
    "L2_ZH",
    "L3_EXACT_ZH",
    "LabelGenerator",
    "SYNONYMS",
    "WUSHU_BASE_ZH",
    "WUSHU_PLAIN_ZH",
    "all_known_labels",
    "clean_segment",
    "cn_number",
    "coverage_check",
    "normalize_key",
    "split_index",
    "zh_label_for_directory",
    "zh_label_for_l3",
]

#: 目标生成的中文变体条数区间（PRD §3.4 / ARCH §5.6：每类 8–15 条）。
MIN_TEXTS: Final[int] = 8

#: 无论如何都保证的最少变体数（未映射类也满足）。
MAX_TEXTS: Final[int] = 15

#: 极端情况下仍保证的下限。
HARD_MIN_TEXTS: Final[int] = 3

#: 零宽 / 不换行空格 / 双向控制符等不可见字符。实测 ``Penetrating_Arm_Fist\u200b``
#: 末尾带 U+200B，不清洗会把不可见字符带进中文标签，下游按标签建条件库时会
#: 「看起来一样但不相等」，静默丢样本。
#:
#: 覆盖范围按主理人要求取 ``\u200b-\u200f``（零宽空格/零宽连接/零宽非连接/
#: LRM/RLM）+ ``\ufeff``（BOM）+ ``\u00a0``（不换行空格）+ ``\u00ad``（软连字符）。
#: LRM/RLM 尤其阴险：它们在终端里完全不可见，但会让 ``"Pass_Ball"`` 与
#: ``"Pass_Ball\u200f"`` 变成两个不同的类，且 pandas 的 groupby 也分得开。
_INVISIBLE_CHARS: Final[str] = (
    "\u200b\u200c\u200d\u200e\u200f\u2060\ufeff\u00a0\u00ad"
)

#: 形如 ``1.7.3.3.`` 的编号前缀。
#:
#: 必须用 ``(?:\\d+\\.)+`` 而不是 ``[0-9.]+``：后者对
#: ``2.3.1.15.360_Turn_Jump`` 会把 ``.360`` 一并吃掉，剩下 ``_Turn_Jump``。
_NUMBER_PREFIX: Final[re.Pattern[str]] = re.compile(r"^(?:\d+\.)+")

#: 结尾序号（``Changquan12`` → ``Changquan`` + ``12``）。
_TRAILING_INDEX: Final[re.Pattern[str]] = re.compile(r"^(?P<base>.*?)(?P<index>\d+)$")

#: 中文数字（0–9）。
_CN_DIGITS: Final[tuple[str, ...]] = (
    "零", "一", "二", "三", "四", "五", "六", "七", "八", "九",
)

#: 大类（l1）中文名。**只用于「l3 与 l2 都为空」的退化情形**，不参与兜底。
CATEGORY_ZH: Final[dict[str, str]] = {
    "daily_life_behavior": "日常行为",
    "special_worker_actions": "特殊职业动作",
    "action_sequence_complex_behavior": "复杂行为动作序列",
    "object_carrying_and_manipulation": "持物与操作",
    "basic_environment_interaction_class": "基础环境交互类",
    "complex_terrain_interaction": "复杂地形交互",
    "extreme_env_interaction": "极端环境交互",
    "adversarial_interaction": "对抗交互",
    "object_carrying": "持物",
    "environment_operation": "环境操作",
    "ball_interaction": "球类交互",
    "box_ing": "拳击",
    "equipment_exercises": "器械健身",
    "pop_dance": "流行舞蹈",
    "wushu": "武术",
}

#: 子类（l2）中文名。用于 l3 为 ``Other`` 时的组合（``其他坐姿``）与 l3 缺失时兜底。
L2_ZH: Final[dict[str, str]] = {
    # 1.12 复杂行为动作序列
    "cleaning_workflow_task": "清洁流程",
    "cooking_prep_process": "备餐流程",
    "carrying_and_warehousing": "搬运仓储",
    "human_robot_interaction_flow": "人机交互流程",
    "industrial_production_flow": "工业生产流程",
    "vehicle_enter_exit_flow": "上下车流程",
    "service_sop_flow": "服务标准流程",
    "complex_household_task_flow": "家务综合流程",
    "dressing_and_grooming": "穿衣梳洗",
    "long_chain_task": "长链综合任务",
    # 1.9 / 1.13 / 1.7
    "catering_service": "餐饮服务",
    "multi_step_complex_operation": "多步复合操作",
    "walking_movement_with_posture": "行走姿态",
    "basic_gait": "基本步态",
    # 2.x
    "stairs_up_down": "上下楼梯",
    "various_sitting_postures": "各种坐姿",
    "uneven_ground_walking": "不平地面行走",
    "parkour": "跑酷",
    "box_ing": "拳击",
    "fencing": "击剑",
    "two_person_balance": "双人对抗平衡",
    # 3.2
    "open_door_operation": "开关门操作",
    "valve_turn": "阀门旋拧",
    "open_close_furniture": "开关家具",
    "pick_and_place": "取放操作",
    # 3.3 球类
    "basketball": "篮球",
    "badminton": "羽毛球",
    "soccer": "足球",
    "volley_ball": "排球",
    "table_tennis": "乒乓球",
    "tennis": "网球",
    # 武术
    "4_3": "武术套路",
}

#: 细类（l3）精确中文标签，键为 :func:`normalize_key` 之后的 l3 名。
#:
#: 值为 ``str`` 表示全局唯一；值为 ``dict[str, str]`` 表示同名 l3 出现在多个 l2 下，
#: 按 l2 归一化键消歧（缺失 l2 命中时取 ``""`` 键或首个值）。
L3_EXACT_ZH: Final[dict[str, str | dict[str, str]]] = {
    # --- 1.12.1 清洁流程 ---
    "floor_sweeping": "扫地",
    "mopping": "拖地",
    "wipe_table_flow": "擦桌子",
    "trash_disposal_flow": "倒垃圾",
    "tidy_room_flow": "整理房间",
    "quilt_cleaning": "叠被子",
    "clothes_cleaning": "整理衣物",
    # --- 1.12.2 备餐流程 ---
    "chopping_process": "切菜",
    "boiling_water_process": "烧水",
    "stir_fry_process": "炒菜",
    "plating_process": "装盘",
    "tidy_kitchen_flow": "收拾厨房",
    "washing_ingredients": "洗菜",
    # --- 1.12.3 搬运仓储 ---
    "carrying_box": "搬箱子",
    "open_box_process": "拆箱开箱",
    "stacking_process": "码垛堆叠",
    "rack_mounting_flow": "货架上架",
    "packing": "打包封箱",
    "carrying": "徒手搬运",
    "sorting": "分拣归类",
    # --- 1.12.4 人机交互 ---
    "greeting_process": "打招呼问候",
    "handover_item_process": "递接物品",
    "pointing_flow": "手指引路",
    "companion_flow": "陪同行走",
    "emotional_comfort_flow": "安抚安慰",
    # --- 1.12.5 工业生产 ---
    "pick_part_place_screw": "取件拧螺丝",
    "check_mark_flow": "打勾确认",
    "spray_simulation_process": "喷涂作业",
    "quality_check": "质量检验",
    "hammering": "挥锤敲击",
    # --- 1.12.6 上下车 ---
    "vehicle_boarding": "上车",
    "getting_off_vehicle": "下车",
    "in_vehicle_adjust": "车内调整坐姿",
    "in_vehicle_move": "车内挪动",
    # --- 1.12.7 服务流程 ---
    "front_desk_process": "前台接待",
    "restaurant_operation": "餐厅运营",
    "restaurant_waiter_flow": "传菜上菜",
    "hotel_room_tidying_flow": "客房整理",
    "hospital_guidance_flow": "医院导诊",
    "cashier_sop": "收银结账",
    "luggage_carrying_process": "提搬行李",
    "coffee_shop_sop": "咖啡店出单",
    # --- 1.12.8 家务综合 ---
    "laundry_process": "洗衣流程",
    "clothes_hanging": "晾挂衣服",
    "tidy_clean_clothes": "整理干净衣物",
    "kitchen_cleaning_full_flow": "厨房深度清洁",
    "bathroom_cleaning_flow": "卫生间清洁",
    "tidy_bedroom_flow": "整理卧室",
    "furniture_move": "挪动家具",
    "open_box_shelving_flow": "开箱上架",
    "tidy_living_room_flow": "整理客厅",
    # --- 1.12.9 穿衣梳洗 ---
    "dressing_sequence": "穿衣流程",
    "put_on_shoes": "穿鞋",
    "backpack_wear": "背书包",
    "washing_routine": "洗漱",
    "combing_hair": "梳头",
    "wear_hat_glasses": "戴帽子眼镜",
    # --- 1.12.10 长链任务 ---
    "cooking_a_meal": "做一顿饭",
    "full_room_clean": "全屋大扫除",
    "full_visitor_reception": "完整接待访客",
    "full_carrying_task": "完整搬运任务",
    "laundry_hanging_full_flow": "洗衣晾衣全流程",
    "plant_tree": "种树",
    # --- 1.13 多步复合 ---
    "pick_rotate_place": "取物转身放置",
    # --- 1.7 / 1.9 日常 ---
    "stairs_up_down": {"stairs_up_down": "上下楼梯", "": "上下楼梯"},
    "basic_gait": "基本步态",
    "walking_movement_with_posture": "行走姿态",
    "walking_movement": "行走",
    "pouring_water": "倒水",
    "wipe_table_action": "擦拭台面",
    # --- 2.1 基础环境交互 ---
    "stairs_up": "上楼梯",
    "stairs_down": "下楼梯",
    "standard_sitting": "端正坐姿",
    "forward_lean_sitting": "前倾坐",
    "lean_back_sitting": "后靠坐",
    "lean_back_relaxed_sitting": "后靠放松坐",
    "cross_legged_sit": "盘腿坐",
    "crossed_leg_sitting": "翘二郎腿坐",
    "side_sitting": "侧身坐",
    "backrest_sit": "靠背坐",
    "half_reclined_sitting": "半躺坐",
    "one_side_leaning": "单侧倚靠坐",
    # --- 2.2 复杂地形 ---
    "break_stone": "碎石地行走",
    "cobblestone": "鹅卵石路行走",
    "pothole": "坑洼地行走",
    # --- 2.3 极端环境 / 跑酷 ---
    "precision_jump": "精准定点跳跃",
    "descent_speed_control": "下降控速",
    "side_flip": "侧空翻",
    "backflip": "后空翻",
    "front_flip": "前空翻",
    "wall_climb": "攀墙",
    "360_turn_jump": "转身跳三百六",
    "butterfly_spin": "蝴蝶旋转体",
    "aerial_no_hands": "无手空翻",
    "forward_flip_with_backflip": "前空翻接后空翻",
    "koi_flip": "鲤鱼打挺",
    "hurdle_crossing": "跨栏",
    "speed_crossing": "快速翻越",
    "lazy_roll": "懒人翻滚",
    "two_hands_climb_over": "双手撑越",
    "crawl_through": "匍匐钻越",
    "roll_and_land": "翻滚落地",
    "run_up_wall_mount": "助跑上墙",
    "balance_walk": "平衡木行走",
    # --- 2.4 对抗交互 ---
    "straight_punch": "直拳",
    "hook_punch": "勾拳",
    "swing_punch": "摆拳",
    "straight_thrust": "击剑直刺",
    "chopping_strike": "击剑劈击",
    "footwork": "击剑步法",
    "push_and_shove": "推搡",
    "tugging": "拉扯",
    "dodge": "闪躲避让",
    # --- 3.2 环境操作 ---
    "revolving_door": "推旋转门",
    "push_pull_door": "推拉门",
    "folding_door": "开关折叠门",
    "small_valve": "拧小阀门",
    "medium_valve": "拧中阀门",
    "large_valve": "拧大阀门",
    "faucet": "拧水龙头",
    "drawer_pull": "拉开抽屉",
    "cabinet_door": "开柜门",
    "refrigerator": "开冰箱门",
    "bend_grab_place": "弯腰取放",
    "kneel_grab_place": "跪地取放",
    "one_knee_grab_place": "单膝跪取放",
    "overhead_grab_place": "过顶取放",
    # --- 3.3 球类（同名按 l2 消歧） ---
    "dribbling_ball": "篮球运球",
    "basketball_shot": "篮球投篮",
    "rebound": "抢篮板",
    "pass_ball": {"basketball": "篮球传球", "volley_ball": "排球传球"},
    "serve_ball": {"badminton": "羽毛球发球", "table_tennis": "乒乓球发球", "tennis": "网球发球"},
    "smash": "羽毛球扣杀",
    "drop_shot": "羽毛球吊球",
    "short_pass": "足球短传",
    "long_pass": "足球长传",
    "shooting_goal": "足球射门",
    "stop_ball": "足球停球",
    "ball_toss": "排球抛球",
    "spike": "排球扣球",
    "block_net": "排球拦网",
    "forehand_attack": "乒乓球正手攻",
    "backhand_push": "乒乓球反手推",
    "loop_drive": "乒乓球弧圈球",
    "forehand_strike": "网球正手抽",
    "backhand_hit": "网球反手击",
    "volley": "网球截击",
    "move_and_catch": "网球移动接球",
    # 以下三条是**实测漏网**（M3 前的文本条件体检抓出来的）：这三个 l3 目录
    # （``4.1.Box_Ing`` / ``3.1.4.Cooperative_Transport`` / ``4.3.Equipment_Exercises``）
    # 当时既不在 l3 精确表里，也过不了关键词（``box_ing`` 里有下划线、``boxing``
    # 匹配不上），于是落到机械转写，产出 ``Box Ing`` 这种英文标签 + 中英混排的
    # 扩写（「慢慢地Box Ing」）。英文条件对中文指令模型是**永远触发不到**的，
    # 必须在这里补上，否则下一次全量预处理会重新产出同样的坏标签。
    "box_ing": "拳击",
    "cooperative_transport": "合作搬运",
    "equipment_exercises": "器械健身",
    "block": {"box_ing": "拳击格挡", "fencing": "击剑格挡"},
    # ``Other`` 由 l2 组合成「其他{子类}」，见 :func:`zh_label_for_l3`。
    "other": {"various_sitting_postures": "其他坐姿", "parkour": "其他跑酷动作",
              "box_ing": "其他拳击动作", "basketball": "其他篮球动作",
              "badminton": "其他羽毛球动作", "soccer": "其他足球动作",
              "volley_ball": "其他排球动作", "table_tennis": "其他乒乓动作",
              "tennis": "其他网球动作"},
}

#: 武术「基名 + 序号」表：``基名(小写) → (中文基名, 序号后缀模板)``。
#:
#: 实测 ``4.wushu/4.3`` 的 148 个带序号类别只来自这 21 个基名。模板里的 ``{n}``
#: 会替换成中文数字（``第{n}段`` → ``第七段``）。
WUSHU_BASE_ZH: Final[dict[str, tuple[str, str]]] = {
    "changquan": ("长拳", "第{n}段"),
    "nanquan": ("南拳", "第{n}路"),
    "tai_chi": ("太极拳", "第{n}式"),
    "chen_style_tai_chi": ("陈式太极拳", "第{n}路"),
    "crane_fist": ("鹤拳", "第{n}路"),
    "dragon_fist": ("龙拳", "第{n}路"),
    "snake_fist": ("蛇拳", "第{n}路"),
    "tiger_fist": ("虎拳", "第{n}路"),
    "leopard_fist": ("豹拳", "第{n}路"),
    "wing_chun": ("咏春", "第{n}段"),
    "optional_broadsword": ("自选刀术", "第{n}段"),
    "optional_staff": ("自选棍术", "第{n}段"),
    "optional_straight_sword": ("自选剑术", "第{n}段"),
    "da_dao": ("大刀", "第{n}路"),
    "eagle_claw_kung_fu": ("鹰爪功", "第{n}路"),
    "five_element_fist": ("五行拳", "第{n}路"),
    "five_stance_fist": ("五形拳", "第{n}路"),
    "low_balance_with_leg_inserted_behind": ("插腿低式平衡", "第{n}式"),
    "martial_arts_marching": ("武术行进", "第{n}段"),
    "praying_mantis_kung_fu": ("螳螂拳", "第{n}路"),
    "sword_vs_broadsword_sparring": ("剑对刀对练", "第{n}段"),
}

#: 武术无序号单件（43 个）。
WUSHU_PLAIN_ZH: Final[dict[str, str]] = {
    "bajiquan": "八极拳",
    "basic_broadsword": "基本刀术",
    "basic_cudgel": "基本棍术",
    "basic_fist": "基本拳",
    "basic_spear": "基本枪术",
    "basic_sword": "基本剑术",
    "competition_broadsword": "竞赛刀术",
    "double_hammers": "双锤",
    "double_swords": "双剑",
    "dual_routine": "对练套路",
    "elbow_strike_in_horse_stance": "马步顶肘",
    "horizontal_row": "横扫",
    "hua_quan": "华拳",
    "international_compulsory_boxing_routine_i": "国际规定拳术一路",
    "international_compulsory_broadsword_routine_i": "国际规定刀术一路",
    "international_compulsory_broadsword_routine_iii": "国际规定刀术三路",
    "international_compulsory_cudgel_routine_i": "国际规定棍术一路",
    "international_compulsory_long_fist_routine_iii": "国际规定长拳三路",
    "international_compulsory_spear_routine_ii": "国际规定枪术二路",
    "international_compulsory_spear_routine_iii": "国际规定枪术三路",
    "international_compulsory_sword_routine_i": "国际规定剑术一路",
    "international_compulsory_sword_routine_iii": "国际规定剑术三路",
    "juvenile_compulsory_boxing_routine": "少年规定拳",
    "lark_sword": "云雀剑",
    "longsword": "长剑",
    "muay_thai": "泰拳",
    "offensive_sword_techniques": "进攻剑技",
    "optional_spear": "自选枪术",
    "parting_the_wild_horse_mane": "野马分鬃",
    "penetrating_arm_fist": "通背拳",
    "pigua_quan": "劈挂拳",
    "plank_backbend": "平板后弯",
    "pudao": "朴刀",
    "self_created_staff_technique": "自编棍术",
    "slow_swing": "慢摆",
    "sword_dance": "剑舞",
    "tai_chi_sword": "太极剑",
    "tang_dao": "唐刀",
    "two_handed_sword": "双手剑",
    "wild_horse_leaps_over_the_stream": "野马跳涧",
    "wind_fire_wheel": "风火轮",
    "wudang_sword": "武当剑",
    "xingyi_quan": "形意拳",
}

#: 关键词规则原始定义：``(子串, 优先级, 中文)``，优先级数字小的先匹配。
#:
#: **不能只按长度排序**：``Walking_On_Stairs_Slowly`` 里 ``walking``(7 字符) 比
#: ``stairs``(6) 长，纯长度排序会把「上下楼梯」误标成「行走」。所以用显式优先级
#: 分三层：0 = 楼梯方向/武术器械/武术流派（最具体），1 = 复合动作，2 = 泛化动词。
_RULE_DEFS: Final[tuple[tuple[str, int, str], ...]] = (
    # --- 优先级 0：楼梯方向 ---
    ("stairs_up", 0, "上楼梯"),
    ("stair_up", 0, "上楼梯"),
    ("up_stairs", 0, "上楼梯"),
    ("up_stair", 0, "上楼梯"),
    ("stairs_down", 0, "下楼梯"),
    ("stair_down", 0, "下楼梯"),
    ("down_stairs", 0, "下楼梯"),
    ("down_stair", 0, "下楼梯"),
    ("stair", 0, "上下楼梯"),
    # --- 优先级 0：武术器械（长 token 优先，避免 sword 抢掉 broadsword）---
    ("optional_straight_sword", 0, "自选剑术"),
    ("optional_broadsword", 0, "自选刀术"),
    ("optional_staff", 0, "自选棍术"),
    ("international_compulsory", 0, "国际规定套路"),
    ("broadsword", 0, "刀术"),
    ("longsword", 0, "长剑"),
    ("cudgel", 0, "棍术"),
    ("spear", 0, "枪术"),
    ("sword", 0, "剑术"),
    ("pudao", 0, "朴刀"),
    # --- 优先级 0：武术流派 ---
    ("tai_chi", 0, "太极拳"),
    ("taichi", 0, "太极拳"),
    ("changquan", 0, "长拳"),
    ("nanquan", 0, "南拳"),
    ("bajiquan", 0, "八极拳"),
    ("crane_fist", 0, "鹤拳"),
    ("dragon_fist", 0, "龙拳"),
    ("snake_fist", 0, "蛇拳"),
    ("tiger_fist", 0, "虎拳"),
    ("leopard_fist", 0, "豹拳"),
    ("mantis", 0, "螳螂拳"),
    ("wing_chun", 0, "咏春"),
    ("pigua", 0, "劈挂拳"),
    ("xingyi", 0, "形意拳"),
    ("hua_quan", 0, "华拳"),
    ("muay_thai", 0, "泰拳"),
    ("double_hammers", 0, "双锤"),
    ("double_swords", 0, "双剑"),
    ("five_element_fist", 0, "五行拳"),
    ("five_stance_fist", 0, "五形拳"),
    ("eagle_claw", 0, "鹰爪功"),
    ("elbow_strike", 0, "顶肘"),
    # --- 优先级 1：复合日常动作 ---
    ("floor_sweeping", 1, "扫地"),
    ("sweeping", 1, "扫地"),
    ("sweep", 1, "扫地"),
    ("mopping", 1, "拖地"),
    ("mop", 1, "拖地"),
    ("wipe_table", 1, "擦桌子"),
    ("wipe", 1, "擦拭"),
    ("washing", 1, "洗涤"),
    ("wash", 1, "洗漱"),
    ("brushing", 1, "刷牙"),
    ("brush", 1, "刷牙"),
    ("combing", 1, "梳头"),
    ("comb", 1, "梳头"),
    ("hair", 1, "梳头"),
    ("drinking", 1, "喝水"),
    ("drink", 1, "喝水"),
    ("eating", 1, "吃东西"),
    ("eat", 1, "吃东西"),
    ("cooking", 1, "做饭"),
    ("cook", 1, "做饭"),
    ("chopping", 1, "切菜"),
    ("cut", 1, "切割"),
    ("carrying", 1, "搬运"),
    ("carry", 1, "搬运"),
    ("lifting", 1, "举重"),
    ("lift", 1, "举起"),
    ("pushing", 1, "推"),
    ("push", 1, "推"),
    ("pulling", 1, "拉"),
    ("pull", 1, "拉"),
    ("open", 1, "开门"),
    ("close", 1, "关门"),
    ("picking", 1, "捡起"),
    ("pick", 1, "捡起"),
    ("placing", 1, "放置"),
    ("place", 1, "放置"),
    ("throwing", 1, "投掷"),
    ("throw", 1, "投掷"),
    ("kicking", 1, "踢"),
    ("kick", 1, "踢"),
    ("catching", 1, "接住"),
    ("catch", 1, "接住"),
    ("dribbling", 1, "运球"),
    ("dribble", 1, "运球"),
    ("waving", 1, "挥手"),
    ("wave", 1, "挥手"),
    ("handshake", 1, "握手"),
    ("dancing", 1, "跳舞"),
    ("dance", 1, "跳舞"),
    ("boxing", 1, "打拳"),
    ("punch", 1, "出拳"),
    ("weight", 1, "举重"),
    ("ball", 1, "球类动作"),
    ("climbing", 1, "攀爬"),
    ("climb", 1, "攀爬"),
    ("crawling", 1, "爬行"),
    ("crawl", 1, "爬行"),
    ("falling", 1, "摔倒"),
    ("fall", 1, "摔倒"),
    ("squatting", 1, "下蹲"),
    ("squat", 1, "下蹲"),
    ("jumping", 1, "跳跃"),
    ("jump", 1, "跳跃"),
    ("turning", 1, "转身"),
    ("turn", 1, "转身"),
    ("lying", 1, "躺下"),
    ("lie", 1, "躺下"),
    # --- 优先级 2：泛化动词（最后才用）---
    ("running", 2, "跑步"),
    ("run", 2, "跑步"),
    ("walking", 2, "行走"),
    ("walk", 2, "行走"),
    ("sitting", 2, "坐下"),
    ("sit", 2, "坐下"),
    ("standing", 2, "站立"),
    ("stand", 2, "站立"),
    ("gait", 2, "基本步态"),
)

#: 关键词兜底规则：``(子串, 中文标签)``。
#:
#: 顺序 = **(优先级升序, 同优先级内长度降序)**。优先级保证「楼梯」不会被
#: 「行走」抢占，长度保证 ``broadsword`` 先于 ``sword`` 命中（否则刀被标成剑）。
KEYWORD_RULES: Final[tuple[tuple[str, str], ...]] = tuple(
    (token, label)
    for token, _priority, label in sorted(_RULE_DEFS, key=lambda item: (item[1], -len(item[0])))
)

#: 同义词/近义表述。键为**中文标签**，值为该标签的其他自然说法。
SYNONYMS: Final[dict[str, tuple[str, ...]]] = {
    "扫地": ("扫地板", "用扫帚扫地", "打扫地面", "清扫地面"),
    "拖地": ("用拖把拖地", "擦地", "拖地板"),
    "擦桌子": ("擦干净桌面", "抹桌子"),
    "倒垃圾": ("扔垃圾", "清倒垃圾"),
    "整理房间": ("收拾房间", "把房间整理干净"),
    "叠被子": ("把被子叠好", "整理床铺"),
    "洗菜": ("冲洗食材", "清洗蔬菜"),
    "切菜": ("把菜切好", "切配食材"),
    "炒菜": ("翻炒菜肴", "下锅炒菜"),
    "烧水": ("接水烧开", "把水烧上"),
    "装盘": ("把菜盛盘", "摆盘出菜"),
    "做一顿饭": ("下厨做饭", "烹饪一餐", "煮一顿饭"),
    "搬箱子": ("搬运箱子", "抱起箱子搬走", "抬着箱子走"),
    "上架": ("把货物上架", "摆上货架"),
    "码垛堆叠": ("把箱子码起来", "堆叠货物"),
    "分拣归类": ("分拣物品", "把东西分类"),
    "打包封箱": ("封箱打包", "把东西装进箱子"),
    "打招呼问候": ("向人打招呼", "挥手问候"),
    "递接物品": ("把东西递过去", "接过物品"),
    "手指引路": ("伸手指方向", "用手指示路线"),
    "上车": ("打开车门上车", "坐进车里"),
    "下车": ("从车上下来", "开门下车"),
    "上楼梯": ("走上楼梯", "上楼", "登楼梯", "一步一步上台阶"),
    "下楼梯": ("走下楼梯", "下楼", "一级一级下台阶"),
    "上下楼梯": ("来回上下楼", "上下台阶"),
    "弯腰取放": ("弯腰捡起再放下", "弯身拿放东西"),
    "跪地取放": ("跪下来取放物品", "双膝跪地拿放"),
    "拧水龙头": ("打开水龙头", "拧开水龙头"),
    "开冰箱门": ("打开冰箱取东西", "拉开冰箱"),
    "拉开抽屉": ("抽出抽屉", "把抽屉拉开"),
    "推拉门": ("推开推拉门", "拉开移门"),
    "盘腿坐": ("盘膝而坐", "双腿盘起来坐"),
    "翘二郎腿坐": ("翘着腿坐", "叠腿坐着"),
    "侧身坐": ("侧着身子坐", "偏身坐下"),
    "喝水": ("喝一口水", "端起水杯喝水", "饮水"),
    "洗漱": ("洗脸刷牙", "做洗漱"),
    "梳头": ("梳头发", "整理头发", "用梳子梳头"),
    "穿鞋": ("把鞋穿上", "弯腰穿鞋"),
    "穿衣流程": ("一件件穿衣服", "把衣服穿上"),
    "跑步": ("快跑", "奔跑", "跑起来"),
    "行走": ("走路", "步行", "往前走"),
    "跳跃": ("起跳", "蹦跳", "跳起来"),
    "蹲下": ("蹲下来", "屈膝下蹲"),
    "转身": ("回头转身", "转过身子"),
    "摔倒": ("跌倒", "倒在地上"),
    "挥手": ("挥挥左手", "摆动手臂打招呼", "举手挥动"),
    "跳舞": ("跳一段舞", "起舞", "跟着节拍跳舞"),
    "篮球投篮": ("投篮球", "把球投进篮筐"),
    "篮球传球": ("把球传给队友", "胸前传球"),
    "篮球运球": ("拍球前进", "运着球走"),
    "足球射门": ("起脚射门", "把球射向球门"),
    "羽毛球扣杀": ("跳起扣杀", "大力杀球"),
    "排球扣球": ("起跳扣球", "大力扣杀"),
    "乒乓球正手攻": ("正手抽击", "正手进攻"),
    "网球发球": ("抛球发球", "发一记球"),
    "击剑直刺": ("向前直刺", "剑尖直取"),
    "直拳": ("打直拳", "直线出拳"),
    "勾拳": ("打出勾拳", "由下往上勾"),
    "摆拳": ("横摆出拳", "抡拳横击"),
    "格挡": ("抬手格挡", "挡住来拳"),
    "闪躲避让": ("侧身闪开", "躲开来势"),
    "推搡": ("互相推搡", "推开对方"),
    "前空翻": ("向前翻一个跟头", "前滚翻腾空"),
    "后空翻": ("向后翻腾", "后空翻一周"),
    "侧空翻": ("侧向翻腾", "侧手翻腾空"),
    "鲤鱼打挺": ("从地上鲤鱼打挺起身", "仰卧弹起"),
    "跨栏": ("跨过栏架", "抬腿过栏"),
    "平衡木行走": ("在窄处保持平衡行走", "走平衡木"),
    "攀墙": ("攀上墙面", "手脚并用爬墙"),
    "八极拳": ("打一套八极拳", "练八极拳", "演练八极拳"),
    "太极拳": ("打太极", "练太极拳"),
    "太极剑": ("练太极剑", "舞太极剑"),
    "形意拳": ("练形意拳", "打形意"),
    "通背拳": ("练通背拳", "打通背"),
    "劈挂拳": ("练劈挂拳", "打劈挂"),
    "咏春": ("练咏春", "打木人桩式的咏春"),
    "泰拳": ("练泰拳", "打泰拳"),
    "南拳": ("练南拳", "打南拳"),
    "长拳": ("打一套长拳", "练一段长拳", "演练长拳套路"),
    "基本拳": ("练基本拳法", "打基础拳"),
    "基本剑术": ("练基本剑法", "舞剑"),
    "基本刀术": ("练基本刀法", "舞刀"),
    "基本棍术": ("练基本棍法", "舞棍"),
    "基本枪术": ("练基本枪法", "抖枪"),
    "竞赛刀术": ("练竞赛刀", "打竞赛刀术"),
    "双剑": ("舞双剑", "双手各持一剑"),
    "双锤": ("舞双锤", "双手抡锤"),
    "剑舞": ("舞一套剑", "持剑起舞"),
    "野马分鬃": ("做野马分鬃", "野马分鬃式"),
    "风火轮": ("抡臂如风火轮", "转臂风火轮"),
}

#: 程度副词（插入「地」字后与动词组合）。
ADVERBS: Final[tuple[str, ...]] = (
    "慢慢", "缓缓", "快速", "轻快", "稳稳", "轻松", "从容", "小心",
)

#: 句式模板：``{base}`` 是中文标签，``{adv}`` 是副词。
TEMPLATES: Final[tuple[str, ...]] = (
    "{base}",
    "{adv}地{base}",
    "做一个{base}的动作",
    "角色正在{base}",
    "{base}的动作",
    "连续地{base}",
    "自然地{base}",
    "演示一段{base}",
    "开始{base}",
    "完成{base}的动作",
)


def cn_number(value: int) -> str:
    """把 1–99 的整数转成中文数字。

    Args:
        value: 正整数。

    Returns:
        中文数字（``1`` → ``一``、``10`` → ``十``、``21`` → ``二十一``）。
        超出 ``[1, 99]`` 或非正数时直接返回阿拉伯数字字符串。
    """
    if value <= 0 or value > 99:
        return str(value)
    if value < 10:
        return _CN_DIGITS[value]
    tens, ones = divmod(value, 10)
    head = "" if tens == 1 else _CN_DIGITS[tens]
    return f"{head}十{_CN_DIGITS[ones] if ones else ''}"


def clean_segment(text: str) -> str:
    """清掉不可见字符与首尾空白。

    实测 ``4.wushu`` 里 ``Penetrating_Arm_Fist\\u200b`` 带零宽空格，
    不清洗会让「看起来相同」的标签互不相等，条件库会静默丢样本。

    Args:
        text: 原始目录名。

    Returns:
        清洗后的字符串。
    """
    cleaned = str(text)
    for char in _INVISIBLE_CHARS:
        cleaned = cleaned.replace(char, "")
    return cleaned.strip()


def normalize_key(directory: str) -> str:
    """把目录名归一化为查表键。

    规则：清不可见字符 → 去掉 ``1.7.3.3.`` 这类编号前缀 → 去尾部序号 →
    小写 → 空格/连字符/下划线统一为下划线 → 折叠重复下划线。

    Args:
        directory: 原始目录名（如 ``"1.7.3.3.Stairs_Up_Down"`` / ``"Changquan1"``）。

    Returns:
        归一化键（如 ``"stairs_up_down"`` / ``"changquan"``）。
    """
    text = clean_segment(directory)
    if not text:
        return ""
    text = _NUMBER_PREFIX.sub("", text)
    lowered = text.lower()
    match = _TRAILING_INDEX.match(lowered)
    if match and match.group("base").strip("_- "):
        lowered = match.group("base")
    lowered = re.sub(r"[\s\-.]+", "_", lowered)
    lowered = re.sub(r"_+", "_", lowered)
    return lowered.strip("_")


def split_index(directory: str) -> tuple[str, int | None]:
    """把目录名拆成「基名 + 尾部序号」。

    与 :func:`normalize_key` 的区别是**保留序号**：``Changquan7`` 得到
    ``("changquan", 7)``，而不是把 7 丢掉（丢掉的话 12 套长拳会塌成同一个标签）。

    Args:
        directory: 目录名。

    Returns:
        ``(基名归一化键, 序号或 None)``。
    """
    text = clean_segment(directory)
    if not text:
        return "", None
    text = _NUMBER_PREFIX.sub("", text)
    text = re.sub(r"[\s\-.]+", "_", text.lower())
    text = re.sub(r"_+", "_", text).strip("_")
    match = _TRAILING_INDEX.match(text)
    if not match:
        return text, None
    base = match.group("base").strip("_")
    if not base:
        return text, None
    return base, int(match.group("index"))


def _resolve_from_table(
    table: dict[str, str | dict[str, str]], key: str, l2_key: str
) -> str:
    """在「可能是 str 也可能是 dict」的精确表里取值。"""
    value = table.get(key)
    if value is None:
        return ""
    if isinstance(value, str):
        return value
    if l2_key and l2_key in value:
        return value[l2_key]
    return value.get("", "")


def zh_label_for_l3(l3: str, l2: str = "") -> str:
    """由细类（和可选的子类）得到具体中文标签。

    解析链：``l3 精确表 → 武术基名+序号 → 武术单件 → l3 关键词 → l2 精确/关键词``。
    **不含 l1 兜底**。

    Args:
        l3: 细类目录名。
        l2: 子类目录名（同名 l3 消歧用，如 ``Pass_Ball`` 在篮球/排球下）。

    Returns:
        具体中文标签；全部未命中时返回空串（调用方再决定退化方式）。
    """
    key = normalize_key(l3)
    l2_key = normalize_key(l2)
    if not key:
        return ""

    # 1) 精确表（支持按 l2 消歧）
    label = _resolve_from_table(L3_EXACT_ZH, key, l2_key)
    if label:
        return label

    # 2) 武术：基名 + 序号
    base, index = split_index(l3)
    if index is not None:
        spec = WUSHU_BASE_ZH.get(base)
        if spec is None:
            # 基名本身可能是无序号词条（如 Five_Element_Fist 之外的写法）
            spec = WUSHU_BASE_ZH.get(key)
        if spec is not None:
            name, style = spec
            return f"{name}{style.format(n=cn_number(index))}"

    # 3) 武术无序号单件
    if key in WUSHU_PLAIN_ZH:
        return WUSHU_PLAIN_ZH[key]

    # 4) 关键词兜底（长度降序，broadsword 不会被 sword 抢走）
    for token, word in KEYWORD_RULES:
        if token in key:
            return word

    # 5) 子类兜底：l3 认不出时，退到「子类名 + 该 l3 的关键词」，仍不含 l1
    if l2_key:
        l2_label = L2_ZH.get(l2_key, "")
        for token, word in KEYWORD_RULES:
            if token in l2_key:
                # 例如 l3 未知、l2 = two_person_balance → 用关键词「balance」
                return f"{l2_label or word}{word}"
        if l2_label:
            return l2_label

    return ""


def zh_label_for_directory(directory: str) -> str:
    """兼容旧签名：只给目录名时查中文标签（未命中返回空串）。

    注意顺序：**先查大类精确表**（``1.13.Object_Carrying_And_Manipulation``
    这类 l1/l2 目录名会被关键词 ``carrying`` 抢成「搬运」），再走 l3 解析链。

    Args:
        directory: 目录名（``category_l3`` 或任意层级名）。

    Returns:
        中文标签，或空串。
    """
    key = normalize_key(directory)
    if not key:
        return ""
    exact = CATEGORY_ZH.get(key, "") or L2_ZH.get(key, "")
    if exact:
        return exact
    return zh_label_for_l3(directory, "")


@dataclass(frozen=True)
class LabelGenerator:
    """中文标签与多样化中文描述的生成器。"""

    min_texts: int = MIN_TEXTS
    max_texts: int = MAX_TEXTS

    def generate(
        self,
        *,
        category_l1: str = "",
        category_l2: str = "",
        category_l3: str = "",
        stem: str = "",
    ) -> tuple[str, list[str]]:
        """为一个 clip 生成中文标签与描述列表。

        Args:
            category_l1: 大类目录名（仅作最后退化，不参与兜底）。
            category_l2: 子类目录名。
            category_l3: 细类目录名（最优先）。
            stem: 文件名主干（全部未命中时的兜底）。

        Returns:
            ``(zh_label, zh_texts)``；``zh_texts`` 保证长度 ∈
            ``[HARD_MIN_TEXTS, max_texts]``，元素互不重复。
        """
        label, mapped = self._resolve_label(category_l3, category_l2, category_l1, stem)
        texts = self._expand(label, mapped)
        return label, texts

    def expand(self, label: str) -> list[str]:
        """为**已知标签**扩写出 8–15 条中文描述（合并重标签用）。

        :meth:`generate` 是「目录名 → 标签 + 描述」；本方法是「标签 → 描述」。
        少样本合并（:mod:`ttm.data.label_merge`）把「长拳第七段」并成「长拳」之后，
        描述必须同步换掉，否则条件文本指向一个没有样本的语义点。

        Args:
            label: 中文标签（来自映射表时可用同义词）。

        Returns:
            去重后的描述列表，首元素恒为 ``label`` 本身。
        """
        return self._expand(label, True)

    # -- 内部 ---------------------------------------------------------------
    @staticmethod
    def _resolve_label(l3: str, l2: str, l1: str, stem: str) -> tuple[str, bool]:
        """按细类 → 子类 → 机械转写的顺序解析中文标签。

        **不再对 l1 大类名兜底**：l1 只在 l3/l2 全空时使用。

        Returns:
            ``(label, mapped)``；``mapped=False`` 表示走了机械转写。
        """
        label = zh_label_for_l3(l3, l2)
        if label:
            return label, True
        if l2:
            l2_key = normalize_key(l2)
            if l2_key in L2_ZH:
                return L2_ZH[l2_key], True
        if not (l3 or l2):
            l1_label = CATEGORY_ZH.get(normalize_key(l1), "")
            if l1_label:
                return l1_label, True
        fallback_source = l3 or l2 or l1 or stem
        cleaned = clean_segment(fallback_source) or str(fallback_source)
        cleaned = _NUMBER_PREFIX.sub("", cleaned)
        cleaned = re.sub(r"[\s\-]+", "_", cleaned).strip("_")
        return (cleaned.replace("_", " ") or "未知动作"), False

    def _expand(self, label: str, mapped: bool) -> list[str]:
        """用同义词 + 副词 + 模板扩写出多样化描述。

        Args:
            label: 中文标签。
            mapped: 是否来自映射表（决定可用同义词）。

        Returns:
            去重后的中文描述列表（长度 ∈ ``[3, max_texts]``）。
        """
        texts: list[str] = []
        seen: set[str] = set()

        def _add(text: str) -> None:
            """加入一条（去空白、去重、受上限约束）。"""
            value = text.strip()
            if not value or value in seen:
                return
            seen.add(value)
            texts.append(value)

        _add(label)

        synonyms = SYNONYMS.get(label, ()) if mapped else ()
        for synonym in synonyms:
            if len(texts) >= self.max_texts:
                break
            _add(synonym)

        adverb_cursor = 0
        template_cursor = 0
        while len(texts) < self.min_texts and template_cursor < len(TEMPLATES) * len(ADVERBS):
            template = TEMPLATES[template_cursor % len(TEMPLATES)]
            adverb = ADVERBS[adverb_cursor % len(ADVERBS)]
            template_cursor += 1
            if "{adv}" in template:
                adverb_cursor += 1
            rendered = template.format(base=label, adv=adverb)
            if rendered in seen:
                continue
            _add(rendered)

        suffixes = ("", "，动作自然流畅", "，节奏均匀", "，连贯完成", "，保持平衡")
        suffix_cursor = 0
        while len(texts) < HARD_MIN_TEXTS:
            suffix = suffixes[suffix_cursor % len(suffixes)]
            suffix_cursor += 1
            _add(f"{label}{suffix}")
            if suffix_cursor > 32:  # 理论不可达，防死循环
                break

        return texts[: self.max_texts]


def generate_labels(
    entries: Iterable[tuple[str, str, str, str]],
    *,
    min_texts: int = MIN_TEXTS,
    max_texts: int = MAX_TEXTS,
) -> list[tuple[str, list[str]]]:
    """批量生成标签（便捷函数）。

    Args:
        entries: ``(category_l1, category_l2, category_l3, stem)`` 的可迭代对象。
        min_texts: 最少变体数。
        max_texts: 最多变体数。

    Returns:
        与输入一一对应的 ``(zh_label, zh_texts)`` 列表。
    """
    generator = LabelGenerator(min_texts=min_texts, max_texts=max_texts)
    return [
        generator.generate(
            category_l1=l1, category_l2=l2, category_l3=l3, stem=stem
        )
        for l1, l2, l3, stem in entries
    ]


def all_known_labels() -> list[str]:
    """返回映射表中出现过的全部中文标签（去重，按首次出现顺序）。"""
    labels: list[str] = []
    seen: set[str] = set()

    def _push(value: str) -> None:
        if value and value not in seen:
            seen.add(value)
            labels.append(value)

    for value in L3_EXACT_ZH.values():
        if isinstance(value, str):
            _push(value)
        else:
            for item in value.values():
                _push(item)
    for name, style in WUSHU_BASE_ZH.values():
        _push(f"{name}{style.format(n='三')}")
    for value in WUSHU_PLAIN_ZH.values():
        _push(value)
    for value in L2_ZH.values():
        _push(value)
    for value in CATEGORY_ZH.values():
        _push(value)
    return labels


def coverage_check(directories: Sequence[str]) -> tuple[int, int]:
    """统计一批目录名的映射覆盖率（调试用）。

    Args:
        directories: 目录名序列。

    Returns:
        ``(命中数, 总数)``。
    """
    hit = sum(1 for item in directories if zh_label_for_directory(item))
    return hit, len(directories)
