"""少样本标签合并：``W < threshold`` 的细标签并到「可学」的基名标签。

**为什么必须有这一层**（主理人 2026-10-29 追加的硬规则）：

Tier-S 实测 350 个细标签里有 **191 个是武术**，而武术总共只有 833 个窗口
（191 类 / 220 文件 / 每类 1.15 个文件），其中 184 个不到 20 窗、中位数 4 窗。
如果直接拿 350 个标签当条件，等于给模型 191 个「各只有 4 个样本」的条件：

* 模型只会把这些条件拟合到噪声上（4 个样本连条件向量都估不出来）；
* 更糟的是**把本来能学的共性语义冲掉**——「这是长拳」本来学得到，
  硬塞进「长拳第七段 / 长拳第十一段」之后，长拳的共性被 12 个稀疏条件切碎。

这比标签塌陷（l3 落 l1 兜底、40 个细类塌成 19 个）更难查：塌陷是**语义太粗**，
一眼能看出来；条件不可学是**语义看着很细、实际全是噪声**，要等生成质量崩了才发现。

规则（逐字实现主理人的口径，仅一处工程细化，见 ``plan_merges`` 的 ``parent`` 判定）::

    对每个候选标签统计其窗口数 W：
      W >= threshold  → 保持细分标签（如「长拳第七段」）
      W <  threshold  → 合并到基名标签（如「长拳」/「太极拳」/「象形拳」/「刀术」）

**唯一细化**：合并后的父组自身也必须 ``>= threshold`` 才真正执行合并。
若父组本身就不够（如「咏春」16 窗、「八极拳」8 窗——它们已经是基名，没有更粗的
语义父级），则**保留细标签并显式记为 residual**，不为了凑数把它们塞进
「武术其他」这种捡垃圾标签里。理由：把一个 4 窗的「泰拳」并进「武术其他」
并不能让条件变得可学，只会让「武术其他」这个标签本身的语义边界消失，
下游生成会得到一个什么都像、什么都不像的条件。宁可少一个条件，不要一个假条件。

⚠️ **口径裁决（D-06，v1.4）**：residual 项（实测 21 个）**保留为条件**参与训练，
不并入「武术其他」。以 ``README.md`` #16 为唯一契约；曾有一版 QA-PLAN 写成
「residual 不进条件」，与实现相反，**以本段与 README #16 为准**。

层级结构（武术，从细到粗）：

1. 细标签：``长拳第七段`` / ``陈式太极拳第二路`` / ``虎拳第三路``；
2. 基名：``长拳`` / ``太极拳`` / ``南拳`` / ``咏春`` / ``象形拳`` / ``刀术`` /
   ``剑术`` / ``棍术`` / ``枪术`` / ``对练`` / ``武术基本招式``；
3. 器械按**器械种类**保留（刀 / 剑 / 棍 / 枪 各自成组），不塌成一个「器械」——
   「这是刀术」和「这是剑术」是最容易混、也最需要显式区分的两类（实测旧词表把
   ``Broadsword`` 错标成「剑法」，两条刀剑指令全错还查不出原因）。

BVH_Body 侧同规则，但只有 7 个标签需要合并（实测 160 个标签里仅 12 个 < 30 窗），
用显式表 :data:`BVH_PARENT_ZH` 而不是规则引擎——条目少，显式表更可审计。
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Final, Sequence

from ttm.data.labels_zh import LabelGenerator, clean_segment

__all__ = [
    "BVH_PARENT_ZH",
    "FORCED_FAMILIES",
    "MERGE_THRESHOLD",
    "MergePlan",
    "apply_plan_to_rows",
    "merge_label_zh",
    "plan_merges",
    "texts_for_label",
    "wushu_parent",
]

#: 条件可学的最小窗口数。低于它就把细标签并到基名。
MERGE_THRESHOLD: Final[int] = 30

#: 武术细标签 → 基名的**有序**子串规则，先命中先赢。
#:
#: 顺序不变量（改这里之前先看这条）：
#:
#: 1. ``对练`` 必须在所有器械规则之前：``剑对刀对练`` 同时含「剑」「刀」，
#:    但它是对练套路，不是剑术也不是刀术；
#: 2. 器械（刀/剑/棍/枪）在流派之前：``太极剑`` 是剑术、``大刀第一路`` 是刀术，
#:    器械是**动作形态**、流派是**风格归属**，前者对生成更有判别力；
#: 3. 具体流派在泛化拳术之前：``国际规定长拳三路`` 归「长拳」而不是「国际规定拳术」；
#: 4. 单式/基本功放最后：``马步顶肘`` / ``横扫`` / ``野马分鬃`` 这类只有 2–9 窗，
#:    单靠任何一条都不可学，合并成「武术基本招式」（实测合计 41 窗）才立得住。
_WUSHU_RULES: Final[tuple[tuple[str, str], ...]] = (
    # --- 1. 对练（必须最先） ---
    ("对练", "对练"),
    ("sparring", "对练"),
    # --- 2. 器械按种类 ---
    ("刀", "刀术"),
    ("剑", "剑术"),
    ("棍", "棍术"),
    ("枪", "枪术"),
    # --- 3. 象形拳五家 ---
    ("虎拳", "象形拳"),
    ("鹤拳", "象形拳"),
    ("蛇拳", "象形拳"),
    ("龙拳", "象形拳"),
    ("豹拳", "象形拳"),
    # --- 4. 太极（陈式归太极拳）---
    ("陈式太极拳", "太极拳"),
    ("太极", "太极拳"),
    # --- 5. 具体流派 ---
    ("长拳", "长拳"),
    ("南拳", "南拳"),
    ("咏春", "咏春"),
    ("八极拳", "八极拳"),
    ("形意拳", "形意拳"),
    ("泰拳", "泰拳"),
    ("华拳", "华拳"),
    ("劈挂拳", "劈挂拳"),
    ("通背拳", "通背拳"),
    ("螳螂拳", "螳螂拳"),
    ("鹰爪功", "鹰爪功"),
    ("五行拳", "五行拳"),
    ("五形拳", "五形拳"),
    # --- 6. 泛化拳术 ---
    ("国际规定拳术", "国际规定拳术"),
    ("基本拳", "基本拳"),
    ("少年规定拳", "基本拳"),
    # --- 7. 单式与基本功（兜底）---
    ("马步顶肘", "武术基本招式"),
    ("横扫", "武术基本招式"),
    ("慢摆", "武术基本招式"),
    ("风火轮", "武术基本招式"),
    ("野马分鬃", "武术基本招式"),
    ("野马跳涧", "武术基本招式"),
    ("平板后弯", "武术基本招式"),
    ("插腿低式平衡", "武术基本招式"),
    ("武术行进", "武术基本招式"),
    # 注意：``剑舞`` 不在这里——它在上面就被「剑」抓到并归了「剑术」。
    # 剑舞确实是持剑套路，归剑术比归「基本招式」更准确，所以这条不用改。
)

#: BVH_Body 侧「细标签 → 基名」显式表（实测只有这些需要合并）。
#:
#: * 空翻家族：``后空翻`` / ``无手空翻`` / ``前空翻接后空翻`` 都 < 30 窗，
#:   并成「空翻」（合计 58 窗）。``前空翻``(33) / ``侧空翻``(94) 样本够，
#:   保持细分——「空翻」在语义上是它们的上位词，embedding 里天然邻近，
#:   并存不会互相干扰；
#: * ``徒手搬运``(28) → 「搬运」(289)：本来就是同一个动作的两种说法；
#: * ``车内挪动``(26) → 「车内调整坐姿」(34)：都在车内小范围变换重心；
#: * ``背书包``(28) + ``戴帽子眼镜``(32) → 「穿戴配饰」(60)：配饰穿戴是一族，
#:   合并后 60 窗可学，不合并则两个都不够。
BVH_PARENT_ZH: Final[dict[str, str]] = {
    "后空翻": "空翻",
    "无手空翻": "空翻",
    "前空翻接后空翻": "空翻",
    "徒手搬运": "搬运",
    "车内挪动": "车内调整坐姿",
    "背书包": "穿戴配饰",
    "戴帽子眼镜": "穿戴配饰",
}

#: 需要**整族**一起合并的标签族：``族名 → 成员``。
#:
#: 用途：``戴帽子眼镜``(32) 本来够 30 窗、按规则不必合并，但若只把 28 窗的
#: ``背书包`` 挤进一个新组「穿戴配饰」，就会出现「穿戴配饰」和「戴帽子眼镜」
#: 两个语义重叠的标签（一个泛指、一个特指同一批动作），模型要靠 embedding
#: 去猜两者边界——这是标签体系设计错误，不是数据问题。所以族内要么全合、
#: 要么全不合。
FORCED_FAMILIES: Final[dict[str, tuple[str, ...]]] = {
    "穿戴配饰": ("背书包", "戴帽子眼镜"),
}


def wushu_parent(label: str) -> str:
    """武术细标签的**一级**父标签（无父级时返回自身）。

    Args:
        label: 中文标签（如 ``"长拳第七段"``）。

    Returns:
        基名标签（``"长拳"``）；已无可归并的父级时返回原标签。
    """
    text = clean_segment(label)
    if not text:
        return text
    for token, parent in _WUSHU_RULES:
        if token in text:
            return parent
    return text


def merge_label_zh(label: str, *, is_wushu: bool = False) -> str:
    """细标签的一级归并目标（不分来源，统一入口）。

    Args:
        label: 中文标签。
        is_wushu: 是否来自 ``4.wushu``（决定用武术规则还是 BVH 显式表）。

    Returns:
        归并后的标签；无父级时返回原标签。
    """
    text = clean_segment(label)
    if is_wushu:
        return wushu_parent(text)
    return BVH_PARENT_ZH.get(text, text)


def texts_for_label(label: str, *, generator: LabelGenerator | None = None) -> list[str]:
    """为一个（合并后的）标签重新生成 8–15 条中文描述。

    合并后必须换描述，否则条件文本还是「长拳第七段」，与训练标签不一致——
    bge 编码出来的条件向量会指向一个没有样本的语义点。

    Args:
        label: 中文标签。
        generator: 可选复用 :class:`LabelGenerator`（避免重复构造）。

    Returns:
        非空描述列表，首元素恒为标签本身。
    """
    maker = generator or LabelGenerator()
    texts = maker.expand(label)
    if not texts or texts[0] != label:
        texts = [label, *[item for item in texts if item != label]]
    return texts


@dataclass
class MergePlan:
    """一次标签合并的完整决策记录。

    Attributes:
        mapping: ``细标签 → 合并后标签``（只含真正发生变化的条目）。
        counts_before: 合并前 ``标签 → 窗口数``。
        counts_after: 合并后 ``标签 → 窗口数``。
        residual: 合并后仍 ``< threshold`` 的标签（无更粗父级，按实际值上报）。
        forced: 因 :data:`FORCED_FAMILIES` 整族合并、且本身 ``>= threshold`` 的标签。
        passes: 迭代轮数（父级链条长度）。
        total_windows: 总窗口数。
        threshold: 生效阈值。
    """

    mapping: dict[str, str] = field(default_factory=dict)
    counts_before: dict[str, int] = field(default_factory=dict)
    counts_after: dict[str, int] = field(default_factory=dict)
    residual: tuple[str, ...] = ()
    forced: tuple[str, ...] = ()
    passes: int = 0
    total_windows: int = 0
    threshold: int = MERGE_THRESHOLD

    # -- 基本属性 -----------------------------------------------------------
    @property
    def distinct_before(self) -> int:
        """合并前 distinct 标签数（统计口径）。"""
        return len(self.counts_before)

    @property
    def distinct_after(self) -> int:
        """合并后 distinct 标签数（**训练实际使用的条件数**）。"""
        return len(self.counts_after)

    def map_label(self, label: str) -> str:
        """按本计划把细标签映射到合并后标签。"""
        return self.mapping.get(label, label)

    def max_share_after(self) -> tuple[str, float]:
        """合并后窗口数最多的标签及其占比。

        Returns:
            ``(标签, 占比)``；无数据时返回 ``("", 0.0)``。
        """
        if not self.counts_after or self.total_windows <= 0:
            return "", 0.0
        label, count = max(self.counts_after.items(), key=lambda item: item[1])
        return label, count / float(self.total_windows)

    def top_labels(self, limit: int = 15) -> list[tuple[str, int, float]]:
        """按窗口数降序返回前 ``limit`` 个标签。

        Returns:
            ``[(标签, 窗口数, 占比), ...]``。
        """
        total = float(self.total_windows) or 1.0
        ordered = sorted(self.counts_after.items(), key=lambda item: (-item[1], item[0]))
        return [(name, count, count / total) for name, count in ordered[:limit]]

    def as_dict(self) -> dict[str, object]:
        """序列化为可直接写 JSON 的字典（报告用）。"""
        name, share = self.max_share_after()
        return {
            "threshold": self.threshold,
            "passes": self.passes,
            "total_windows": self.total_windows,
            "distinct_before": self.distinct_before,
            "distinct_after": self.distinct_after,
            "merged_labels": len(self.mapping),
            "max_label_after": name,
            "max_share_after": share,
            "residual_below_threshold": list(self.residual),
            "forced_family_merges": list(self.forced),
            "counts_after": dict(
                sorted(self.counts_after.items(), key=lambda item: (-item[1], item[0]))
            ),
        }


def _trace(label: str, *, is_wushu: bool) -> str:
    """沿父链一直追到不动点（``陈式太极拳第二路 → 陈式太极拳 → 太极拳``）。

    Args:
        label: 起始标签。
        is_wushu: 是否武术（决定用规则引擎还是 BVH 显式表）。

    Returns:
        最终归并目标；无父级时返回 ``label`` 本身。
    """
    current = label
    seen = {current}
    for _hop in range(8):
        nxt = wushu_parent(current) if is_wushu else BVH_PARENT_ZH.get(current, current)
        if nxt == current or nxt in seen:
            break
        seen.add(nxt)
        current = nxt
    return current


def plan_merges(
    labels: Sequence[str],
    flags_wushu: Sequence[bool] | None = None,
    *,
    threshold: int = MERGE_THRESHOLD,
) -> MergePlan:
    """按「窗口数 < threshold → 并到基名」的规则计算合并计划。

    做法是**先把每个细标签沿父链追到不动点，再统计每一组的窗口数**——
    而不是「一轮一轮地看谁的计数够了再并」。后者会漏掉一个关键情形：
    父组名（如「长拳」）在合并前根本不存在于标签集合里，它的计数是 0，
    于是「长拳第七段(5 窗) → 长拳(0 窗)」的增益被算成 5，永远够不到阈值 30，
    结果 190 个武术标签一个都并不上去（这个 bug 实测出现过一次）。
    先追父链再计数就自然正确：12 套长拳的窗口在追到「长拳」之后自动汇总成 52。

    Args:
        labels: 逐窗口的中文标签（长度 = 窗口数）。
        flags_wushu: 逐窗口的「是否武术」标记；``None`` 时视为全部非武术。
        threshold: 可学阈值，默认 :data:`MERGE_THRESHOLD`。

    Returns:
        :class:`MergePlan`。

    Example:
        >>> plan = plan_merges(["长拳第七段", "长拳第八段"] * 20)
        >>> plan.distinct_after
        1
    """
    counts_before: dict[str, int] = {}
    kind: dict[str, bool] = {}
    for index, raw in enumerate(labels):
        label = clean_segment(str(raw)) or "未标注"
        counts_before[label] = counts_before.get(label, 0) + 1
        is_wushu = bool(flags_wushu[index]) if flags_wushu is not None else False
        # 同一标签只要有一处来自武术就按武术规则处理（标签文本本身带流派信息）
        kind[label] = kind.get(label, False) or is_wushu

    targets: dict[str, str] = {}
    depth: dict[str, int] = {}
    for label in counts_before:
        target = _trace(label, is_wushu=kind.get(label, False))
        targets[label] = target
        depth[label] = 0 if target == label else 1

    # 整族合并（族内要么全合、要么全不合，见 FORCED_FAMILIES 文档）
    forced: list[str] = []
    for family, members in FORCED_FAMILIES.items():
        present = [item for item in members if item in targets]
        if len(present) < 2:
            continue
        for item in present:
            if targets[item] != family:
                targets[item] = family
                depth[item] = 1
            if counts_before[item] >= threshold:
                forced.append(item)

    counts_after: dict[str, int] = {}
    for label, count in counts_before.items():
        target = targets[label]
        counts_after[target] = counts_after.get(target, 0) + count

    mapping = {label: target for label, target in targets.items() if target != label}
    residual = tuple(
        sorted(name for name, count in counts_after.items() if count < threshold)
    )
    return MergePlan(
        mapping=mapping,
        counts_before=counts_before,
        counts_after=counts_after,
        residual=residual,
        forced=tuple(sorted(forced)),
        passes=max(depth.values()) if depth else 0,
        total_windows=sum(counts_before.values()),
        threshold=int(threshold),
    )


def apply_plan_to_rows(rows: Sequence[dict], plan: MergePlan, *, generator: LabelGenerator | None = None) -> list[dict]:
    """把合并计划作用到 ``labels.jsonl`` 的行上（纯标签层，不碰特征）。

    **细标签必须留档**：``zh_label_fine`` / ``zh_texts_fine`` 保留原值，
    报告里的「合并前 distinct 细类数」和 M3 的细指令对比验证都靠它。
    训练只读 ``zh_label`` / ``zh_texts``。

    Args:
        rows: ``labels.jsonl`` 的行（每个窗口一行）。
        plan: :func:`plan_merges` 的结果。
        generator: 可选复用 :class:`LabelGenerator`。

    Returns:
        新的行列表（不修改入参）。
    """
    maker = generator or LabelGenerator()
    cache: dict[str, list[str]] = {}
    output: list[dict] = []
    for row in rows:
        fine = clean_segment(str(row.get("zh_label", ""))) or "未标注"
        merged = plan.map_label(fine)
        if merged not in cache:
            cache[merged] = texts_for_label(merged, generator=maker) if merged != fine else list(
                row.get("zh_texts") or [merged]
            )
        item = dict(row)
        item["zh_label"] = merged
        item["zh_texts"] = cache[merged]
        item["zh_label_fine"] = fine
        item["zh_texts_fine"] = list(row.get("zh_texts") or [])
        item["label_merged"] = merged != fine
        item["is_wushu"] = bool(row.get("is_wushu", False)) or "wushu" in str(
            row.get("category_l1", "")
        ).lower()
        output.append(item)
    return output
