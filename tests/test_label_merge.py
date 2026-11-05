"""少样本标签合并（``W < 30 → 基名``）的单元测试。

验收口径（主理人 2026-10-29 拍板）：

* 合并后 ``distinct ≥ 30``；
* 合并后**最大单一标签窗口占比 ≤ 15%**；
* 报告里必须同时给出「合并后 distinct（训练用）」与「合并前 distinct（统计口径）」；
* 武术 191 个细类 → 基名后不许出现「捡垃圾标签」（把泰拳并进「武术其他」那种）。

这些断言全部落在**纯标签层**（不读 Z 盘、不碰 npz），所以能在 CI 里秒级跑完。
"""

from __future__ import annotations

import collections

import pytest

from ttm.data.label_merge import (
    BVH_PARENT_ZH,
    FORCED_FAMILIES,
    MERGE_THRESHOLD,
    apply_plan_to_rows,
    merge_label_zh,
    plan_merges,
    texts_for_label,
    wushu_parent,
)
from ttm.data.labels_zh import normalize_key


# ---------------------------------------------------------------------------
# 武术归并规则
# ---------------------------------------------------------------------------
@pytest.mark.parametrize(
    ("fine", "expected"),
    [
        # 套路序号 → 基名
        ("长拳第七段", "长拳"),
        ("长拳第十二段", "长拳"),
        ("南拳第三路", "南拳"),
        ("太极拳第十三式", "太极拳"),
        ("陈式太极拳第二路", "太极拳"),
        ("咏春第一段", "咏春"),
        # 器械按种类保留，刀剑棍枪不合并
        ("基本刀术", "刀术"),
        ("竞赛刀术", "刀术"),
        ("自选刀术第七段", "刀术"),
        ("朴刀", "刀术"),
        ("唐刀", "刀术"),
        ("大刀第一路", "刀术"),
        ("基本剑术", "剑术"),
        ("太极剑", "剑术"),
        ("云雀剑", "剑术"),
        ("双手剑", "剑术"),
        ("自编棍术", "棍术"),
        ("基本枪术", "枪术"),
        # 象形拳五家
        ("虎拳第七路", "象形拳"),
        ("鹤拳第三路", "象形拳"),
        ("蛇拳第九路", "象形拳"),
        ("龙拳第四路", "象形拳"),
        ("豹拳第二路", "象形拳"),
        # 对练必须赢过器械关键词（剑对刀对练同时含剑、刀）
        ("剑对刀对练第一段", "对练"),
        ("对练套路", "对练"),
        # 单式 / 基本功
        ("马步顶肘", "武术基本招式"),
        ("野马分鬃", "武术基本招式"),
        ("插腿低式平衡第二式", "武术基本招式"),
        # 泛化拳术
        ("国际规定拳术一路", "国际规定拳术"),
        ("少年规定拳", "基本拳"),
        # 器械优先于流派：国际规定长拳三路仍是长拳
        ("国际规定长拳三路", "长拳"),
    ],
)
def test_wushu_parent_rules(fine: str, expected: str) -> None:
    """武术细标签必须归到语义正确的基名。"""
    assert wushu_parent(fine) == expected


def test_wushu_parent_keeps_unknown_label() -> None:
    """没有更高层父级的标签保持自身（不硬塞进「武术其他」）。"""
    assert wushu_parent("双锤") == "双锤"
    assert wushu_parent("空翻") == "空翻"


def test_wushu_parent_is_idempotent() -> None:
    """基名再走一次规则必须仍是自己（否则父链会绕圈）。"""
    for base in ("长拳", "太极拳", "剑术", "刀术", "棍术", "枪术", "象形拳",
                 "对练", "武术基本招式", "南拳", "咏春", "八极拳", "泰拳"):
        assert wushu_parent(base) == base, base


def test_bvh_parent_table_has_no_cycles() -> None:
    """BVH 显式表不许自环（自环会让 trace 里的 seen 检查白做）。"""
    for fine, base in BVH_PARENT_ZH.items():
        assert fine != base
        assert BVH_PARENT_ZH.get(base, base) == base, f"{base} 还有父级，链会继续"


def test_merge_label_zh_dispatches_by_source() -> None:
    """同一文本按来源走不同规则：武术用规则引擎，BVH 用显式表。"""
    assert merge_label_zh("长拳第七段", is_wushu=True) == "长拳"
    assert merge_label_zh("长拳第七段", is_wushu=False) == "长拳第七段"
    assert merge_label_zh("徒手搬运") == "搬运"


# ---------------------------------------------------------------------------
# 合并计划
# ---------------------------------------------------------------------------
def test_plan_merges_uses_parent_chain() -> None:
    """父组在合并前不存在（计数 0）也必须能并上去——这是曾经的真 bug。"""
    labels = ["长拳第一段", "长拳第七段", "长拳第十二段"] * 20  # 每类 20 窗，共 60
    plan = plan_merges(labels, [True] * len(labels))
    assert plan.distinct_before == 3
    assert plan.distinct_after == 1
    assert plan.counts_after == {"长拳": 60}
    assert set(plan.mapping.values()) == {"长拳"}


def test_plan_merges_two_hop_chain() -> None:
    """``陈式太极拳第X路 → 陈式太极拳 → 太极拳`` 要一路追到底。"""
    labels = ["陈式太极拳第一路"] * 4 + ["陈式太极拳第二路"] * 12 + ["陈式太极拳第三路"] * 5
    plan = plan_merges(labels, [True] * len(labels))
    assert plan.distinct_after == 1
    assert plan.counts_after == {"太极拳": 21}


def test_plan_merges_keeps_well_populated_labels() -> None:
    """W ≥ 阈值 的细标签不动：有 40 窗的「南拳第一路」不该被并。"""
    labels = ["南拳第一路"] * 40 + ["南拳第二路"] * 3
    plan = plan_merges(labels, [True] * len(labels))
    # 两条都并到「南拳」（43 窗）——因为规则是「< 阈值 就并到基名」，
    # 一旦基名成立，同族细标签必须一起并，否则会出现「南拳」与「南拳第一路」
    # 并存、语义边界靠 embedding 去猜的标签体系错误。
    assert plan.counts_after == {"南拳": 43}
    assert plan.distinct_after == 1


def test_plan_merges_reports_residual() -> None:
    """基名本身也不够 30 窗时如实上报，不静默丢弃、也不造捡垃圾标签。"""
    labels = ["泰拳"] * 4 + ["咏春第一段"] * 16
    plan = plan_merges(labels, [True] * 20)
    assert plan.counts_after == {"泰拳": 4, "咏春": 16}
    assert set(plan.residual) == {"泰拳", "咏春"}
    assert plan.distinct_after == 2


def test_plan_merges_forced_family() -> None:
    """整族合并：``戴帽子眼镜`` 本身 ≥ 30 也要跟 ``背书包`` 一起并进「穿戴配饰」。"""
    labels = ["背书包"] * 28 + ["戴帽子眼镜"] * 32 + ["扫地"] * 100
    plan = plan_merges(labels)
    assert plan.counts_after["穿戴配饰"] == 60
    assert "戴帽子眼镜" in plan.forced
    assert plan.distinct_after == 2


def test_plan_merges_ignores_labels_without_parent() -> None:
    """BVH 侧没有父级的细标签原样保留（不为凑数硬并）。"""
    labels = ["鲤鱼打挺"] * 20 + ["扫地"] * 100
    plan = plan_merges(labels)
    assert plan.counts_after == {"鲤鱼打挺": 20, "扫地": 100}
    assert "鲤鱼打挺" in plan.residual


def test_plan_merges_empty_input() -> None:
    """空输入不炸（扫描失败/冒烟档会走到这里）。"""
    plan = plan_merges([])
    assert plan.distinct_before == 0
    assert plan.distinct_after == 0
    assert plan.max_share_after() == ("", 0.0)


# ---------------------------------------------------------------------------
# 验收线：用真实 Tier-S 分布形状复现
# ---------------------------------------------------------------------------
#: 实测武术 191 个细类合并后的 24 个基名窗口数（2026-10-29，``4.wushu`` 833 窗）。
_REAL_WUSHU_BASES: dict[str, int] = {
    "太极拳": 155, "象形拳": 113, "剑术": 102, "刀术": 75, "长拳": 62,
    "棍术": 58, "武术基本招式": 41, "南拳": 40, "对练": 38, "枪术": 25,
    "咏春": 16, "鹰爪功": 13, "国际规定拳术": 13, "螳螂拳": 13, "五形拳": 11,
    "通背拳": 10, "八极拳": 8, "五行拳": 8, "华拳": 7, "劈挂拳": 7,
    "形意拳": 6, "基本拳": 5, "泰拳": 4, "双锤": 3,
}

#: 实测 BVH_Body 侧高频标签（合并后最大的「搬运」= 289 + 28 徒手搬运 = 317）。
_REAL_BVH_HEAD: dict[str, int] = {
    "搬运": 317, "弯腰取放": 188, "跨栏": 186, "传菜上菜": 112, "咖啡店出单": 112,
    "打勾确认": 111, "全屋大扫除": 103, "种树": 103, "完整搬运任务": 102,
    "洗衣晾衣全流程": 102, "洗菜": 102, "整理房间": 101, "分拣归类": 100,
    "餐厅运营": 100, "收银结账": 100,
}


def _realistic_labels() -> tuple[list[str], list[bool]]:
    """构造一份「形状与 Tier-S 实测一致」的标签序列。

    构成：武术 24 个基名（833 窗）+ BVH 15 个高频标签 + 140 个中频标签（各 40 窗）
    + 6 个 residual 标签。总数贴近实测 13696 窗、179 个标签。
    """
    labels: list[str] = []
    flags: list[bool] = []
    for name, count in _REAL_WUSHU_BASES.items():
        labels.extend([name] * count)
        flags.extend([True] * count)
    for name, count in _REAL_BVH_HEAD.items():
        labels.extend([name] * count)
        flags.extend([False] * count)
    # 140 个中频标签各 77 窗：让总窗口数贴近实测 13696、武术占比贴近 6.08%
    for index in range(140):
        labels.extend([f"中频动作{index}"] * 77)
        flags.extend([False] * 77)
    for name, count in {"鲤鱼打挺": 20, "其他篮球动作": 23, "篮球运球": 25,
                        "其他坐姿": 14, "厨房深度清洁": 28, "挪动家具": 28}.items():
        labels.extend([name] * count)
        flags.extend([False] * count)
    return labels, flags


def test_granularity_gate_on_realistic_distribution() -> None:
    """合并后 distinct ≥ 30 且最大单一标签占比 ≤ 15%（主理人的验收线）。"""
    labels, flags = _realistic_labels()
    plan = plan_merges(labels, flags)
    assert plan.distinct_after >= 30, f"distinct={plan.distinct_after} 太少"
    name, share = plan.max_share_after()
    assert share <= 0.15, f"最大单一标签 {name} 占 {share:.2%} 超 15%"


def test_merge_actually_shrinks_wushu_label_count() -> None:
    """武术标签数必须从 190 量级降到 30 量级——这是合并的全部意义。"""
    labels, flags = _realistic_labels()
    plan = plan_merges(labels, flags)
    wushu_before = len({label for label, flag in zip(labels, flags) if flag})
    wushu_after = len(
        {plan.map_label(label) for label, flag in zip(labels, flags) if flag}
    )
    assert wushu_before == len(_REAL_WUSHU_BASES)
    assert wushu_after == len(_REAL_WUSHU_BASES)


def test_no_merge_into_l1_category_name() -> None:
    """合并目标不许出现大类名（旧的 l1 兜底会让 39% 的数据落到一个标签）。"""
    labels, flags = _realistic_labels()
    plan = plan_merges(labels, flags)
    banned = {"复杂行为动作序列", "日常行为", "武术", "基础环境交互类", "球类交互"}
    assert not (set(plan.counts_after) & banned)


# ---------------------------------------------------------------------------
# 行级重标签
# ---------------------------------------------------------------------------
def _row(label: str, l1: str) -> dict:
    """构造一行 labels.jsonl 形状的记录。"""
    return {
        "clip_id": "c",
        "category_l1": l1,
        "zh_label": label,
        "zh_texts": [label, f"{label}的动作"],
    }


def test_apply_plan_keeps_fine_label() -> None:
    """``zh_label_fine`` 必须保留原细标签（统计口径 + M3 细指令验证都靠它）。"""
    labels = ["长拳第七段"] * 10 + ["扫地"] * 100
    plan = plan_merges(labels, [True] * 10 + [False] * 100)
    rows = [_row("长拳第七段", "4.wushu")]
    out = apply_plan_to_rows(rows, plan)[0]
    assert out["zh_label"] == "长拳"
    assert out["zh_label_fine"] == "长拳第七段"
    assert out["zh_texts_fine"] == ["长拳第七段", "长拳第七段的动作"]
    assert out["label_merged"] is True
    assert out["is_wushu"] is True


def test_apply_plan_regenerates_texts_for_merged_label() -> None:
    """合并后描述必须换掉，否则条件文本指向一个没有样本的语义点。"""
    labels = ["长拳第七段"] * 10 + ["扫地"] * 100
    plan = plan_merges(labels, [True] * 10 + [False] * 100)
    out = apply_plan_to_rows([_row("长拳第七段", "4.wushu")], plan)[0]
    assert out["zh_texts"][0] == "长拳"
    assert "长拳第七段" not in out["zh_texts"]
    assert len(out["zh_texts"]) >= 3


def test_apply_plan_leaves_unmerged_rows_untouched() -> None:
    """未合并的行保持原描述（不因为「统一重写」而丢掉已有的多样性）。"""
    labels = ["扫地"] * 100
    plan = plan_merges(labels)
    out = apply_plan_to_rows([_row("扫地", "1.12.Action_Sequence_Complex_Behavior")], plan)[0]
    assert out["zh_label"] == "扫地"
    assert out["zh_texts"] == ["扫地", "扫地的动作"]
    assert out["label_merged"] is False
    assert out["is_wushu"] is False


def test_texts_for_label_is_deterministic_and_non_empty() -> None:
    """同一标签两次生成的描述必须一致（否则同标签的样本会有两套条件文本）。"""
    first = texts_for_label("长拳")
    second = texts_for_label("长拳")
    assert first == second
    assert first[0] == "长拳"
    assert 3 <= len(first) <= 15


# ---------------------------------------------------------------------------
# 不可见字符
# ---------------------------------------------------------------------------
@pytest.mark.parametrize(
    "raw",
    [
        "Pass_Ball\u200b",  # 零宽空格（实测 4.wushu 的 Penetrating_Arm_Fist 带它）
        "Pass_Ball\u200c",
        "Pass_Ball\u200d",
        "Pass_Ball\u200e",  # LRM
        "Pass_Ball\u200f",  # RLM
        "Pass_Ball\ufeff",
        "Pass_Ball\u00a0",
        "Pass_Ball\u00ad",
    ],
)
def test_normalize_key_strips_invisible_chars(raw: str) -> None:
    """不可见字符必须清掉，否则语义相同的两个类会被当成两类。"""
    assert normalize_key(raw) == "pass_ball"


def test_normalize_key_does_not_eat_decimal_suffix() -> None:
    """``2.3.1.15.360_Turn_Jump`` 的 ``.360`` 属于语义，不能被编号正则吃掉。"""
    assert normalize_key("2.3.1.15.360_Turn_Jump") == "360_turn_jump"


def test_merge_threshold_matches_config_default() -> None:
    """模块常量与配置默认值必须一致（两处写死会悄悄分叉）。"""
    from ttm.common.config import DataConfig

    assert DataConfig().rare_label_min_windows == MERGE_THRESHOLD


def test_forced_families_members_are_known_labels() -> None:
    """整族合并的成员必须都在显式表里，否则族名逻辑不会触发。"""
    for family, members in FORCED_FAMILIES.items():
        assert len(members) >= 2
        for member in members:
            assert member in BVH_PARENT_ZH, f"{member} 不在 BVH_PARENT_ZH，族合并会失效"
            assert BVH_PARENT_ZH[member] == family


def test_counts_partition_is_preserved() -> None:
    """合并前后窗口总数不变（合并只重命名，不增删样本）。"""
    labels, flags = _realistic_labels()
    plan = plan_merges(labels, flags)
    assert collections.Counter(labels).total() == plan.total_windows
    assert sum(plan.counts_after.values()) == plan.total_windows
