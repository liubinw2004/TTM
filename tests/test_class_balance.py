"""类别均衡采样（T-均衡）的单元测试。

验收口径（主理人 M1' 要求）：加一个单测断言采样后各类**期望比例偏差 < 20%**。

这里用「解析期望」而不是「真跑随机采样」来验证：加权采样的期望占比
``p_c = freq_c * w_c / Σ(freq * w)`` 是闭式的，直接算它比跑 10 万次抽样更严格
（没有采样噪声），也更快。
"""

from __future__ import annotations

import numpy as np
import pytest

from ttm.data.dataset import (
    DEFAULT_BALANCE_POWER,
    build_sample_weights,
    make_balanced_sampler,
)


def _expected_shares(labels: list[str], weights: np.ndarray) -> dict[str, float]:
    """给定逐样本权重，算出各类别的**期望采样占比**。

    Args:
        labels: 逐样本标签。
        weights: 逐样本权重。

    Returns:
        ``{标签: 期望占比}``，占比之和为 1。
    """
    counts: dict[str, int] = {}
    for label in labels:
        counts[label] = counts.get(label, 0) + 1
    mass: dict[str, float] = {}
    for label, weight in zip(labels, weights):
        mass[label] = mass.get(label, 0.0) + float(weight)
    total = sum(mass.values())
    return {label: mass[label] / total for label in counts}


def test_weights_mean_is_one() -> None:
    """权重整体归一：均值恒为 1（便于跨数据集比较与记录到日志）。"""
    labels = ["a"] * 100 + ["b"] * 10 + ["c"] * 1
    weights = build_sample_weights(labels)
    assert weights.shape == (111,)
    assert weights.mean() == pytest.approx(1.0, rel=1e-9)
    assert np.all(weights > 0)


def test_rare_class_gets_higher_weight() -> None:
    """长尾类的单样本权重必须显著高于高频类。"""
    labels = ["head"] * 1000 + ["tail"] * 4
    weights = build_sample_weights(labels, power=0.5)
    head_weight = float(weights[0])
    tail_weight = float(weights[-1])
    assert tail_weight > head_weight
    # 1/sqrt(freq) 之比应为 sqrt(1000/4) ≈ 15.8（未触顶时）
    assert tail_weight / head_weight == pytest.approx((1000 / 4) ** 0.5, rel=0.2)


def test_full_balance_within_20_percent() -> None:
    """``power=1.0`` 时各类期望占比与均匀分布偏差必须 < 20%。"""
    labels = ["a"] * 500 + ["b"] * 260 + ["c"] * 120 + ["d"] * 40 + ["e"] * 12
    weights = build_sample_weights(labels, power=1.0)
    shares = _expected_shares(labels, weights)
    uniform = 1.0 / len(shares)
    for label, share in shares.items():
        deviation = abs(share - uniform) / uniform
        assert deviation < 0.20, f"{label} 期望占比 {share:.3f} 偏离均匀 {uniform:.3f} 达 {deviation:.1%}"


def test_sqrt_balance_reduces_dominance() -> None:
    """``1/sqrt(freq)`` 在**真实形状**的类别分布上把最大类压到 15% 以内。

    注意用 3 个类别是不够的：只有 3 类时 ``1/sqrt`` 的最优结果就是
    ``sqrt(f)/(Σsqrt)`` = 67%，怎么调都下不来。真实数据有 349 个标签
    （其中 191 个是每类约 4 个窗口的武术细分），分母被长尾撑大，头部占比
    自然落到 12% 左右。这条测试用实测的 Top10 频次 + 191 个各 4 窗口的
    武术类来复现真实形状。
    """
    top10 = [9851, 2919, 2755, 1534, 1185, 902, 736, 582, 507, 481]
    labels: list[str] = []
    for index, frequency in enumerate(top10):
        labels.extend([f"top{index}"] * frequency)
    for index in range(191):
        labels.extend([f"wushu{index}"] * 4)

    raw_shares = _expected_shares(labels, np.ones(len(labels)))
    assert raw_shares["top0"] > 0.40, "原始分布头部应占 40% 以上"

    weights = build_sample_weights(labels, power=DEFAULT_BALANCE_POWER)
    shares = _expected_shares(labels, weights)
    assert shares["top0"] < 0.15, f"1/sqrt 后最大类仍占 {shares['top0']:.1%}"
    assert shares["top0"] > shares["wushu0"], "应保留频次次序（不是完全拉平）"
    assert shares["wushu0"] / shares["top0"] > raw_shares["wushu0"] / raw_shares["top0"] * 3


def test_three_class_case_is_documented() -> None:
    """只有 3 个类别时 ``1/sqrt`` 只能压到 ~67%——这是数学上限，不是 bug。

    因此均衡采样的验收必须建立在**真实标签数**上（见上一条测试），
    而不能用 2–3 个类别的玩具分布。
    """
    labels = ["head"] * 19193 + ["mid"] * 2919 + ["tail"] * 177
    weights = build_sample_weights(labels, power=0.5, max_weight_ratio=1000.0)
    shares = _expected_shares(labels, weights)
    expected = (19193 ** 0.5) / sum(f ** 0.5 for f in (19193, 2919, 177))
    assert shares["head"] == pytest.approx(expected, rel=0.01)


def test_max_weight_ratio_clamps_extreme_tail() -> None:
    """极端长尾（1 vs 10000）必须被 ``max_weight_ratio`` 截断成比值上限。"""
    labels = ["head"] * 10000 + ["tail"] * 1
    weights = build_sample_weights(labels, power=1.0, max_weight_ratio=5.0)
    spread = float(weights.max() / weights.min())
    assert spread <= 5.0 + 1e-9, f"最大/最小权重比 {spread:.3f} 超过上限 5"
    assert weights.mean() == pytest.approx(1.0, rel=1e-9)


def test_empty_labels() -> None:
    """空输入返回空数组，不抛异常。"""
    weights = build_sample_weights([])
    assert weights.shape == (0,)


def test_blank_label_is_its_own_class() -> None:
    """空标签视为独立类别「未标注」，不会把所有无标签样本当成同一个高频类。"""
    labels = ["扫地"] * 3 + [""] * 3
    weights = build_sample_weights(labels, power=1.0)
    np.testing.assert_allclose(weights, np.ones(6), rtol=1e-9)


def test_balanced_sampler_builds() -> None:
    """``make_balanced_sampler`` 产出可迭代的采样器。"""
    torch = pytest.importorskip("torch")
    labels = ["a"] * 50 + ["b"] * 5
    sampler = make_balanced_sampler(build_sample_weights(labels, power=1.0), num_samples=20)
    indices = list(iter(sampler))
    assert len(indices) == 20
    assert all(0 <= index < len(labels) for index in indices)
    assert isinstance(sampler, torch.utils.data.WeightedRandomSampler)


def test_dataset_sampler_only_when_enabled(tmp_path) -> None:
    """``class_balanced=False``（验证/测试集）时 ``sampler()`` 返回 None。"""
    from ttm.data.dataset import MotionDataset

    shard_dir = tmp_path / "shards"
    shard_dir.mkdir()
    count = 4
    np.savez(
        shard_dir / "shard_0000.npz",
        feat=np.zeros((count, 150, 146), dtype=np.float32),
        mask=np.ones((count, 150), dtype=np.uint8),
        clip_id=np.array([f"clip{index}" for index in range(count)]),
        win_idx=np.zeros(count, dtype=np.int16),
        layout_version=np.array([1], dtype=np.int16),
        fps=np.array([30], dtype=np.int16),
    )
    dataset = MotionDataset(shard_dir, with_torch=False)
    assert dataset.sampler() is None, "未启用 class_balanced 时不应返回采样器"

    balanced = MotionDataset(shard_dir, with_torch=False, class_balanced=True)
    sampler = balanced.sampler()
    assert sampler is not None
    assert sampler.num_samples == count


# ---------------------------------------------------------------------------
# 组占比调整：把武术的训练期采样占比抬到 15%
# ---------------------------------------------------------------------------
#: 实测 Tier-S（合并后）的武术窗口分布：24 个基名 / 833 窗（占总 13696 的 6.08%）。
_REAL_WUSHU_BASES: dict[str, int] = {
    "太极拳": 155, "象形拳": 113, "剑术": 102, "刀术": 75, "长拳": 62,
    "棍术": 58, "武术基本招式": 41, "南拳": 40, "对练": 38, "枪术": 25,
    "咏春": 16, "鹰爪功": 13, "国际规定拳术": 13, "螳螂拳": 13, "五形拳": 11,
    "通背拳": 10, "八极拳": 8, "五行拳": 8, "华拳": 7, "劈挂拳": 7,
    "形意拳": 6, "基本拳": 5, "泰拳": 4, "双锤": 3,
}

#: 实测 BVH_Body 侧高频标签（头部「搬运」317 窗）。
_REAL_BVH_HEAD: dict[str, int] = {
    "搬运": 317, "弯腰取放": 188, "跨栏": 186, "传菜上菜": 112, "咖啡店出单": 112,
    "打勾确认": 111, "全屋大扫除": 103, "种树": 103, "完整搬运任务": 102,
    "洗衣晾衣全流程": 102, "洗菜": 102, "整理房间": 101, "分拣归类": 100,
    "餐厅运营": 100, "收银结账": 100,
}


def _realistic_dataset() -> tuple[list[str], np.ndarray]:
    """构造形状贴近实测的 ``(标签列表, 武术掩码)``。"""
    labels: list[str] = []
    wushu: list[bool] = []
    for name, count in _REAL_WUSHU_BASES.items():
        labels.extend([name] * count)
        wushu.extend([True] * count)
    for name, count in _REAL_BVH_HEAD.items():
        labels.extend([name] * count)
        wushu.extend([False] * count)
    # 140 个中频标签各 77 窗：让总部窗口数贴近实测 13696、武术占比贴近 6.08%
    # （833 / 13690 = 6.08%）。数字是反推出来的，不是随手写的。
    for index in range(140):
        labels.extend([f"中频动作{index}"] * 77)
        wushu.extend([False] * 77)
    return labels, np.asarray(wushu, dtype=bool)


def test_resolve_group_multiplier_closed_form() -> None:
    """倍率闭式解必须让期望占比精确命中目标（不是「调个 0.5 试试」）。"""
    from ttm.data.dataset import expected_group_share, resolve_group_multiplier

    labels, mask = _realistic_dataset()
    base = build_sample_weights(labels, power=0.5)
    multiplier = resolve_group_multiplier(base, mask, 0.15)
    tuned = base.copy()
    tuned[mask] *= multiplier
    assert expected_group_share(tuned, mask) == pytest.approx(0.15, rel=1e-6)
    assert multiplier > 1.0, "数据占比 6% 要抬到 15%，倍率必然 > 1"


def test_wushu_sampling_share_lands_in_12_to_18_percent() -> None:
    """主理人验收线：训练期武术**期望采样占比**必须落在 12%~18%。"""
    from ttm.data.dataset import expected_group_share

    labels, mask = _realistic_dataset()
    weights = build_sample_weights(
        labels, power=0.5, group_mask=mask, target_group_share=0.15
    )
    share = expected_group_share(weights, mask)
    assert 0.12 <= share <= 0.18, f"武术采样占比 {share:.2%} 不在 12%~18% 内"
    # 数据侧占比仍是 6% 量级：抬的是采样，不是数据本身
    assert abs(float(mask.mean()) - 0.0608) < 0.01


def test_power_alone_cannot_hit_wushu_target() -> None:
    """只调 ``balance_power`` 到不了 15%，所以必须另有组倍率。

    实测：power=0.5 只把武术从 6.08% 抬到 8.09%；要把 power 拉到能到 15% 的量级，
    会把所有长尾一起抬起来，头部标签反而被压到看不清——这是组倍率存在的理由。
    """
    from ttm.data.dataset import expected_group_share

    labels, mask = _realistic_dataset()
    for power in (0.3, 0.5, 0.7):
        share = expected_group_share(build_sample_weights(labels, power=power), mask)
        assert share < 0.15, f"power={power} 竟然已经到 {share:.2%}，组倍率的必要性要复核"


def test_wushu_target_ratio_zero_is_noop() -> None:
    """``target_group_share=0`` 时权重与不传掩码完全一致（冒烟/无武术档位）。"""
    labels, mask = _realistic_dataset()
    plain = build_sample_weights(labels, power=0.5)
    zero = build_sample_weights(labels, power=0.5, group_mask=mask, target_group_share=0.0)
    np.testing.assert_allclose(plain, zero, rtol=1e-12)


def test_group_mask_shape_mismatch_is_ignored() -> None:
    """掩码长度不对时忽略调整而不是静默错位（错位会把权重乘到别的样本上）。"""
    labels = ["a"] * 10 + ["b"] * 10
    bogus = np.ones(3, dtype=bool)
    plain = build_sample_weights(labels, power=0.5)
    guarded = build_sample_weights(labels, power=0.5, group_mask=bogus, target_group_share=0.5)
    np.testing.assert_allclose(plain, guarded, rtol=1e-12)


def test_weights_stay_normalized_after_group_adjust() -> None:
    """加组倍率后权重均值仍为 1（日志与跨数据集比较都依赖这条）。"""
    labels, mask = _realistic_dataset()
    weights = build_sample_weights(
        labels, power=0.5, group_mask=mask, target_group_share=0.15
    )
    assert weights.mean() == pytest.approx(1.0, rel=1e-9)


def test_dataset_wushu_mask_from_labels_field(tmp_path) -> None:
    """``is_wushu`` 字段优先于 ``category_l1`` 字符串判断。"""
    import orjson

    from ttm.data.dataset import MotionDataset

    shard_dir = tmp_path / "shards"
    shard_dir.mkdir()
    count = 3
    np.savez(
        shard_dir / "shard_0000.npz",
        feat=np.zeros((count, 150, 146), dtype=np.float32),
        mask=np.ones((count, 150), dtype=np.uint8),
        clip_id=np.array(["a", "b", "c"]),
        win_idx=np.zeros(count, dtype=np.int16),
        layout_version=np.array([1], dtype=np.int16),
        fps=np.array([30], dtype=np.int16),
    )
    labels_path = tmp_path / "labels.jsonl"
    rows = [
        {"clip_id": "a", "zh_label": "长拳", "zh_texts": ["长拳"], "is_wushu": True,
         "category_l1": "1.7.Daily_Life_Behavior"},
        # 没有 is_wushu 字段 → 退化为看 category_l1
        {"clip_id": "b", "zh_label": "扫地", "zh_texts": ["扫地"],
         "category_l1": "4.wushu"},
        {"clip_id": "c", "zh_label": "拳击格挡", "zh_texts": ["拳击格挡"],
         "category_l1": "4.1.Box_Ing"},
    ]
    with labels_path.open("wb") as handle:
        for row in rows:
            handle.write(orjson.dumps(row) + b"\n")

    dataset = MotionDataset(shard_dir, labels_path=labels_path, with_torch=False)
    np.testing.assert_array_equal(dataset.wushu_mask(), np.array([True, True, False]))
    assert dataset.data_group_share(dataset.wushu_mask()) == pytest.approx(2 / 3)
