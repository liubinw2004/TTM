"""标签层两件体检的单测：英文标签回填（02f）+ 文本条件体检（02e）。

这两件都是「M3 开跑前的闸门」，共同点是**判据错了会安静地放过坏数据**：

* 回填脚本如果只改 ``zh_label`` 不改 ``zh_texts``，条件中心变了而扩写还指向旧语义，
  训练时会得到「标签是中文、文本是英文」的错配；
* 体检脚本如果只报数字不给判定，等于没闸门；如果判定方向反了（``sim_intra`` 越
  高越算好），「扩写几乎同句」这种坏味道会被判成完美。

用例都不依赖真实数据：喂合成行即可。
"""

from __future__ import annotations

import importlib.util
import sys
from pathlib import Path

import numpy as np
import pytest

from ttm.data.labels_zh import LabelGenerator


def _load(name: str, filename: str):  # noqa: ANN202 - 测试辅助
    """把 ``scripts/`` 下的脚本作为模块导入（它们不是包的一部分）。"""
    module_name = f"ttm_script_{name}"
    if module_name in sys.modules:
        return sys.modules[module_name]
    path = Path(__file__).resolve().parents[1] / "scripts" / filename
    spec = importlib.util.spec_from_file_location(module_name, path)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    sys.modules[module_name] = module
    spec.loader.exec_module(module)
    return module


# ---------------------------------------------------------------------------
# 02f：英文标签回填
# ---------------------------------------------------------------------------
def _row(label: str, *, l3: str, l2: str = "", texts: list[str] | None = None) -> dict:
    """造一行标签记录。"""
    return {
        "clip_id": f"x/{label}/win00",
        "zh_label": label,
        "zh_texts": texts or [label],
        "category_l1": "4.1.Box_Ing",
        "category_l2": l2,
        "category_l3": l3,
        "zh_label_fine": label,
        "zh_texts_fine": list(texts or [label]),
    }


def test_has_cjk_detects_chinese_only() -> None:
    """``has_cjk`` 只看汉字，英文/数字/下划线都不算。"""
    module = _load("02f", "02f_relabel_english.py")
    assert module.has_cjk("拳击") is True
    assert module.has_cjk("慢慢地Box Ing") is True  # 混排也算中文
    assert module.has_cjk("Box Ing") is False
    assert module.has_cjk("Equipment_Exercises") is False
    assert module.has_cjk("") is False


def test_plan_relabels_maps_english_to_chinese_and_regenerates_texts() -> None:
    """英文标签应解析出中文标签，且扩写整体换成中文（不能留中英混排）。"""
    module = _load("02f", "02f_relabel_english.py")
    rows = [
        _row("Box Ing", l3="4.1.Box_Ing", texts=["Box Ing", "慢慢地Box Ing"]),
        _row("Box Ing", l3="4.1.Box_Ing", texts=["Box Ing"]),
        _row("拳击格挡", l3="4.1.2.Block"),
    ]
    relabels, unresolved = module.plan_relabels(rows)

    assert unresolved == []
    assert len(relabels) == 1
    record = relabels[0]
    assert record.old == "Box Ing"
    assert record.new == "拳击"
    assert record.n_windows == 2
    assert record.variants[0] == "拳击"
    assert all(module.has_cjk(text) for text in record.variants), "扩写里残留英文"


def test_plan_relabels_reports_unmapped_instead_of_guessing() -> None:
    """解析不出中文时必须报出来，不能瞎猜一个标签。"""
    module = _load("02f", "02f_relabel_english.py")
    rows = [_row("Totally Unknown Thing", l3="9.9.Totally_Unknown_Thing")]
    relabels, unresolved = module.plan_relabels(rows)

    assert relabels == []
    assert unresolved == ["Totally Unknown Thing"]


def test_apply_relabels_updates_label_and_texts_but_keeps_real_fine_label() -> None:
    """回填要同时改 ``zh_label`` / ``zh_texts``；细标签只在与旧值相同时才跟着改。

    最后一条是关键：合并过的行 ``zh_label_fine`` 是**另一个**值（真正的细标签），
    跟着改会把审计留档毁掉。
    """
    module = _load("02f", "02f_relabel_english.py")
    merged_like = _row("Box Ing", l3="4.1.Box_Ing")
    merged_like["zh_label_fine"] = "Box Ing 第2段"  # 与 zh_label 不同 → 不应被覆盖
    rows = [_row("Box Ing", l3="4.1.Box_Ing", texts=["Box Ing"]), merged_like]
    relabels, _ = module.plan_relabels(rows)

    changed = module.apply_relabels(rows, relabels)

    assert changed == 2
    assert rows[0]["zh_label"] == "拳击"
    assert rows[0]["zh_texts"][0] == "拳击"
    assert rows[0]["zh_label_fine"] == "拳击"  # 原本等于旧值 → 跟着改
    assert rows[1]["zh_label"] == "拳击"
    assert rows[1]["zh_label_fine"] == "Box Ing 第2段"  # 原本不同 → 保持


# ---------------------------------------------------------------------------
# 02e：文本条件体检
# ---------------------------------------------------------------------------
def _fake_encoder(vectors: dict[str, list[float]]):  # noqa: ANN202 - 测试辅助
    """按文本查表的假编码器（未登记的词给一个正交方向）。"""

    class _Encoder:
        def encode(self, texts, batch_size: int = 64):  # noqa: ANN001, ARG002
            rows = []
            for index, text in enumerate(texts):
                if text in vectors:
                    rows.append(vectors[text])
                else:
                    # 没登记的用一条独有的正交向量，避免误聚
                    base = np.zeros(8, dtype=np.float64)
                    base[index % 8] = 1.0
                    rows.append(base.tolist())
            return np.asarray(rows, dtype=np.float64)

    return _Encoder()


def test_compute_similarities_ranks_intra_above_inter() -> None:
    """同一条件的扩写互相靠近时，``sim_intra`` 必须高于 ``sim_inter``。"""
    module = _load("02e", "02e_text_sanity.py")
    stats = [
        module.ConditionStat(label="拳击", category_l1="4.1", is_wushu=False, n_windows=10),
        module.ConditionStat(label="扫地", category_l1="1.12", is_wushu=False, n_windows=10),
    ]
    # 拳击三个变体挤在 x 轴附近；扫地三个变体挤在 y 轴附近 → 组间近似正交
    encoder = _fake_encoder(
        {
            "拳击": [1.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0],
            "打拳": [0.99, 0.1, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0],
            "挥拳": [0.98, -0.1, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0],
            "扫地": [0.0, 1.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0],
            "扫地板": [0.1, 0.99, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0],
            "用扫帚扫地": [-0.1, 0.98, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0],
        }
    )
    stats[0].variants = ["拳击", "打拳", "挥拳"]
    stats[1].variants = ["扫地", "扫地板", "用扫帚扫地"]

    module.compute_similarities(stats, encoder)

    for item in stats:
        assert item.sim_intra > 0.9
        assert item.sim_intra > item.sim_inter + 0.1
        assert item.margin > 0.1


def test_compute_similarities_skips_intra_for_single_variant() -> None:
    """只有 1 条变体算不出 ``sim_intra``，必须是 NaN 而不是 0（0 会误判成「最差」）。"""
    module = _load("02e", "02e_text_sanity.py")
    stat = module.ConditionStat(label="独苗", category_l1="x", is_wushu=False, n_windows=5)
    stat.variants = ["独苗"]
    module.compute_similarities([stat], _fake_encoder({}))
    assert np.isnan(stat.sim_intra)
    assert np.isnan(stat.margin)


def test_build_report_flags_over_similar_expansions() -> None:
    """超过半数条件 ``sim_intra > 0.9`` 时判 FAIL（扩写几乎同句 = 没做增强）。"""
    module = _load("02e", "02e_text_sanity.py")
    stats = []
    for index in range(20):
        item = module.ConditionStat(
            label=f"动作{index}", category_l1="x", is_wushu=False, n_windows=40
        )
        item.variants = ["a", "b", "c"]
        item.sim_intra = 0.97  # 全部过度相似
        item.sim_inter = 0.30
        stats.append(item)

    report = module.build_report(stats, stats[:15], margin_required=0.1, labels_path="x.jsonl")

    assert report["passed"] is False
    assert report["all_conditions"]["over_similar_share"] == pytest.approx(1.0)
    assert any("扩写几乎同句" in reason for reason in report["failures"])


def test_build_report_fails_when_margin_too_small() -> None:
    """margin 不够就是聚不住，哪怕 intra 本身不低。"""
    module = _load("02e", "02e_text_sanity.py")
    item = module.ConditionStat(label="拳击", category_l1="4.1", is_wushu=False, n_windows=40)
    item.variants = ["a", "b", "c"]
    item.sim_intra = 0.80
    item.sim_inter = 0.75  # margin 只有 0.05

    report = module.build_report([item], [item], margin_required=0.1, labels_path="x.jsonl")

    assert report["passed"] is False
    assert report["sample"]["passed"] == 0
    # 失败原因里既有「样本不够」也有「聚不住」，要能看到具体是哪个条件
    assert any("拳击" in reason for reason in report["failures"])


def test_pick_sample_honours_wushu_quota() -> None:
    """抽样必须先满足武术下限，再跨类别铺开；结果确定（同输入同输出）。"""
    module = _load("02e", "02e_text_sanity.py")
    stats = []
    for index in range(30):
        is_wushu = index < 12
        item = module.ConditionStat(
            label=f"{'武' if is_wushu else '日'}{index}",
            category_l1="4.wushu" if is_wushu else f"c{index % 5}",
            is_wushu=is_wushu,
            n_windows=40 + index,
        )
        item.variants = ["a", "b", "c"]
        item.sim_intra = 0.8
        item.sim_inter = 0.4
        stats.append(item)

    first = module.pick_sample(stats, sample=15, wushu_sample=8)
    second = module.pick_sample(stats, sample=15, wushu_sample=8)

    assert len(first) == 15
    assert sum(1 for item in first if item.is_wushu) >= 8
    assert [item.label for item in first] == [item.label for item in second]
    # 非武术部分应跨多个类别（不是全从一个类别里抓）
    assert len({item.category_l1 for item in first if not item.is_wushu}) >= 3


def test_pick_sample_returns_everything_when_pool_is_small() -> None:
    """候选比样本量少时返回全部，不能硬凑出重复项。"""
    module = _load("02e", "02e_text_sanity.py")
    stats = []
    for index in range(3):
        item = module.ConditionStat(
            label=f"动作{index}", category_l1="x", is_wushu=False, n_windows=40
        )
        item.variants = ["a", "b", "c"]
        stats.append(item)
    picked = module.pick_sample(stats, sample=15, wushu_sample=8)
    assert len(picked) == 3


def test_group_conditions_collects_variants_and_flags_wushu() -> None:
    """聚合要按 ``zh_label`` 归并、去重变体、并把主标签放在首位。"""
    module = _load("02e", "02e_text_sanity.py")
    rows = [
        {"zh_label": "拳击", "zh_texts": ["打拳", "拳击"], "category_l1": "4.1.Box_Ing"},
        {"zh_label": "拳击", "zh_texts": ["打拳", "挥拳"], "category_l1": "4.1.Box_Ing"},
        {"zh_label": "长拳", "zh_texts": ["长拳"], "category_l1": "4.wushu", "is_wushu": True},
    ]
    stats = module.group_conditions(rows)

    by_label = {item.label: item for item in stats}
    assert by_label["拳击"].n_windows == 2
    assert by_label["拳击"].variants[0] == "拳击"
    assert set(by_label["拳击"].variants) == {"拳击", "打拳", "挥拳"}
    assert by_label["长拳"].is_wushu is True
    assert by_label["拳击"].is_wushu is False


def test_label_generator_expand_is_all_chinese_for_relabelled_conditions() -> None:
    """回填后的标签必须能扩出整串中文（防止中英混排流回训练数据）。"""
    generator = LabelGenerator()
    for label in ("拳击", "合作搬运", "器械健身"):
        texts = generator.expand(label)
        assert texts[0] == label
        assert all(
            any("\u4e00" <= ch <= "\u9fff" for ch in text) for text in texts
        ), f"{label} 的扩写里有非中文: {texts}"
