"""05b 多样性地板指标 (c)(d) 的单测。

这两个指标是 sweep 选 guidance 的**必要条件**（QA M-07 防线）：判据写反或
分母取错，会把「CFG 过强压多样性」的那一档判成最优，直接写进
``runtime.guidance`` 里污染全链路。所以判定方向必须锁死：

* (c) ``var_ratio`` = 同指令跨 seed 特征方差 ÷ GT 同条件方差，``>= 0.5`` 过；
* (d) ``rough_ratio`` = 生成帧间粗糙度 ÷ GT 同条件，``∈[0.6, 1.5]`` 过。

全部用合成数据，不依赖真实权重与数据集。
"""

from __future__ import annotations

import importlib.util
import sys
from pathlib import Path

import numpy as np
import pytest


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


@pytest.fixture(scope="module")
def ws():  # noqa: ANN202
    """导入 05b 脚本模块。"""
    return _load("05b", "05b_wushu_special.py")


def _sample(module, prompt: str, seed: int, mu, roughness: float, *, nearest: str = "长拳"):
    """造一个最小可用的 :class:`SampleResult`。"""
    return module.SampleResult(
        group="武术",
        prompt=prompt,
        expected="4.wushu",
        seed=seed,
        descriptor=np.zeros(4, dtype=np.float64),
        mu=np.asarray(mu, dtype=np.float64),
        roughness=roughness,
        nearest_label=nearest,
        hit=True,
    )


def _bank(module, cond: str, mus, roughs):
    """造一个只含一个条件、带逐窗统计的最小 :class:`ReferenceBank`。"""
    return module.ReferenceBank(
        centroids=np.zeros((1, 4), dtype=np.float64),
        labels=[cond],
        categories=["4.wushu"],
        is_wushu=[True],
        n_windows=[len(mus)],
        total_windows=len(mus),
        cond_mu={cond: [np.asarray(item, dtype=np.float64) for item in mus]},
        cond_rough={cond: list(roughs)},
    )


# ---------------------------------------------------------------------------
# (c) 跨 seed 方差比
# ---------------------------------------------------------------------------
def test_var_ratio_zero_when_seeds_identical(ws) -> None:
    """同指令各 seed 的 mu 完全一样 → 方差比 0，(c) 必须不过（僵直防线）。"""
    bank = _bank(ws, "长拳", mus=[[1.0, 0.0], [-1.0, 0.0]], roughs=[0.5])
    samples = [
        _sample(ws, "打一套长拳第七段", 0, mu=[0.3, 0.1], roughness=0.5),
        _sample(ws, "打一套长拳第七段", 1, mu=[0.3, 0.1], roughness=0.5),
        _sample(ws, "打一套长拳第七段", 2, mu=[0.3, 0.1], roughness=0.5),
    ]
    result = ws.diversity_metrics(samples, bank)
    assert result["var_ratio_mean"] == 0.0
    assert result["var_ratio_pass"] is False


def test_var_ratio_against_gt_variance(ws) -> None:
    """方差比 = 生成跨 seed 方差 ÷ GT 逐窗方差（逐维算再平均，ddof=1）。

    GT 两窗 ``[1,0] / [-1,0]``：dim0 样本方差 = 2.0、dim1 = 0 → 均值 1.0；
    生成两 seed ``[3,0] / [-3,0]``：均值 9.0 → 比值 9.0。
    """
    bank = _bank(ws, "长拳", mus=[[1.0, 0.0], [-1.0, 0.0]], roughs=[0.5])
    samples = [
        _sample(ws, "打一套长拳第七段", 0, mu=[3.0, 0.0], roughness=0.5),
        _sample(ws, "打一套长拳第七段", 1, mu=[-3.0, 0.0], roughness=0.5),
    ]
    result = ws.diversity_metrics(samples, bank)
    assert result["var_ratio_mean"] == pytest.approx(9.0, rel=1e-6)
    assert result["var_ratio_pass"] is True


def test_var_ratio_needs_two_seeds(ws) -> None:
    """单 seed 算不出跨 seed 方差：置 None 且不过，绝不静默当成好。"""
    bank = _bank(ws, "长拳", mus=[[1.0, 0.0], [-1.0, 0.0]], roughs=[0.5])
    samples = [_sample(ws, "打一套长拳第七段", 0, mu=[1.0, 0.0], roughness=0.5)]
    result = ws.diversity_metrics(samples, bank)
    assert result["var_ratio_mean"] is None
    assert result["var_ratio_pass"] is False


def test_reference_condition_by_majority_vote(ws) -> None:
    """GT 分母条件取各 seed 最近条件的多数票（不是随便取第一个 seed 的）。"""
    bank = _bank(ws, "长拳", mus=[[1.0], [-1.0]], roughs=[0.5])
    samples = [
        _sample(ws, "打一套长拳第七段", 0, mu=[0.1], roughness=0.5, nearest="太极拳"),
        _sample(ws, "打一套长拳第七段", 1, mu=[0.2], roughness=0.5, nearest="长拳"),
        _sample(ws, "打一套长拳第七段", 2, mu=[0.3], roughness=0.5, nearest="长拳"),
    ]
    result = ws.diversity_metrics(samples, bank)
    assert result["by_prompt"]["打一套长拳第七段"]["reference_condition"] == "长拳"


def test_missing_reference_condition_is_none_not_zero(ws) -> None:
    """多数票条件不在参考库里（理论上不该发生）：指标置 None，不伪造通过。"""
    bank = _bank(ws, "长拳", mus=[[1.0], [-1.0]], roughs=[0.5])
    samples = [
        _sample(ws, "打一套长拳第七段", 0, mu=[0.1], roughness=0.5, nearest="不存在的条件"),
        _sample(ws, "打一套长拳第七段", 1, mu=[0.2], roughness=0.5, nearest="不存在的条件"),
    ]
    result = ws.diversity_metrics(samples, bank)
    detail = result["by_prompt"]["打一套长拳第七段"]
    assert detail["var_ratio"] is None
    assert detail["rough_ratio"] is None
    assert result["var_ratio_pass"] is False
    assert result["rough_ratio_pass"] is False


# ---------------------------------------------------------------------------
# (d) 帧间粗糙度比
# ---------------------------------------------------------------------------
def test_rough_ratio_band_pass(ws) -> None:
    """生成粗糙度 = GT 同条件 → 比值 1.0，落在带内，过。"""
    bank = _bank(ws, "长拳", mus=[[1.0], [-1.0]], roughs=[0.5, 0.5])
    samples = [
        _sample(ws, "打一套长拳第七段", 0, mu=[0.1], roughness=0.5),
        _sample(ws, "打一套长拳第七段", 1, mu=[0.2], roughness=0.5),
    ]
    result = ws.diversity_metrics(samples, bank)
    assert result["rough_ratio_mean"] == pytest.approx(1.0)
    assert result["rough_ratio_pass"] is True


def test_rough_ratio_too_low_is_stiff(ws) -> None:
    """生成粗糙度只有 GT 的 0.2 倍 → 低于 0.6 下限，判僵直，不过。"""
    bank = _bank(ws, "长拳", mus=[[1.0], [-1.0]], roughs=[0.5])
    samples = [
        _sample(ws, "打一套长拳第七段", 0, mu=[0.1], roughness=0.1),
        _sample(ws, "打一套长拳第七段", 1, mu=[0.2], roughness=0.1),
    ]
    result = ws.diversity_metrics(samples, bank)
    assert result["rough_ratio_mean"] == pytest.approx(0.2)
    assert result["rough_ratio_pass"] is False


def test_rough_ratio_too_high_is_jittery(ws) -> None:
    """生成粗糙度是 GT 的 2 倍 → 高于 1.5 上限，判抖动，不过。"""
    bank = _bank(ws, "长拳", mus=[[1.0], [-1.0]], roughs=[0.5])
    samples = [
        _sample(ws, "打一套长拳第七段", 0, mu=[0.1], roughness=1.0),
        _sample(ws, "打一套长拳第七段", 1, mu=[0.2], roughness=1.0),
    ]
    result = ws.diversity_metrics(samples, bank)
    assert result["rough_ratio_mean"] == pytest.approx(2.0)
    assert result["rough_ratio_pass"] is False


# ---------------------------------------------------------------------------
# 选值规则
# ---------------------------------------------------------------------------
def _sweep_row(module, guidance: float, hit_rate: float, margin: float, *, c_pass: bool, d_pass: bool) -> dict:
    """造一行 sweep 汇总（结构必须与 ``_run_sweep`` 的产出一致）。"""
    return {
        "guidance": guidance,
        "overall": {"hit_rate": hit_rate, "margin_mean": margin, "seed_stability_mean": 0.7},
        "summary_by_group": {},
        "diversity": {
            "var_ratio_mean": 0.8 if c_pass else 0.2,
            "rough_ratio_mean": 1.0 if d_pass else 2.0,
            "var_ratio_pass": c_pass,
            "rough_ratio_pass": d_pass,
        },
    }


def test_pick_guidance_requires_c_and_d(ws) -> None:
    """hit 率再高，(c)(d) 不过就没有资格——只从资格档里选。"""
    rows = [
        _sweep_row(ws, 3.0, 0.60, 0.30, c_pass=True, d_pass=True),
        _sweep_row(ws, 7.5, 1.00, 0.40, c_pass=False, d_pass=False),
    ]
    best = ws.pick_guidance(rows)
    assert best is not None
    assert best["guidance"] == 3.0
    assert best["recommended"] is True


def test_pick_guidance_prefers_higher_hit_among_qualified(ws) -> None:
    """资格档之间选整体 hit 率最高者。"""
    rows = [
        _sweep_row(ws, 3.0, 0.60, 0.30, c_pass=True, d_pass=True),
        _sweep_row(ws, 4.5, 0.78, 0.35, c_pass=True, d_pass=True),
        _sweep_row(ws, 6.0, 0.90, 0.25, c_pass=True, d_pass=True),
    ]
    best = ws.pick_guidance(rows)
    assert best is not None and best["guidance"] == 6.0


def test_pick_guidance_tiebreak_margin_then_lower(ws) -> None:
    """hit 率并列 → margin 高者胜；再并列 → 更小的 guidance（别过度压多样性）。"""
    rows = [
        _sweep_row(ws, 6.0, 0.80, 0.20, c_pass=True, d_pass=True),
        _sweep_row(ws, 4.5, 0.80, 0.35, c_pass=True, d_pass=True),
        _sweep_row(ws, 3.0, 0.80, 0.35, c_pass=True, d_pass=True),
    ]
    best = ws.pick_guidance(rows)
    assert best is not None and best["guidance"] == 3.0


def test_pick_guidance_none_when_all_fail(ws) -> None:
    """没有任何档同时过 (c)(d) → 返回 None（不能硬选一档写进配置）。"""
    rows = [
        _sweep_row(ws, 3.0, 0.60, 0.30, c_pass=True, d_pass=False),
        _sweep_row(ws, 7.5, 1.00, 0.40, c_pass=False, d_pass=True),
    ]
    assert ws.pick_guidance(rows) is None


# ---------------------------------------------------------------------------
# 辅助函数
# ---------------------------------------------------------------------------
def test_majority_tie_takes_first_occurrence(ws) -> None:
    """平票取先出现者，保证确定性（同输入必同输出）。"""
    assert ws._majority(["甲", "乙", "甲", "乙"]) == "甲"
    assert ws._majority(["乙", "甲", "甲", "乙"]) == "乙"
    assert ws._majority(["丙"]) == "丙"


def test_dimwise_var_matches_manual_ddof1(ws) -> None:
    """``_dimwise_var`` 用样本方差（ddof=1）：两行 [[0,2],[2,0]] → (2+2)/2=2。"""
    matrix = np.asarray([[0.0, 2.0], [2.0, 0.0]])
    assert ws._dimwise_var(matrix) == pytest.approx(2.0)
    assert ws._dimwise_var(matrix[:1]) == 0.0  # 单行没有方差可言


def test_sample_dict_includes_roughness(ws) -> None:
    """``to_dict`` 必须带 roughness（报告里 (d) 的分子可追溯），mu 不落盘。"""
    item = _sample(ws, "打一套长拳第七段", 0, mu=[1.0, 2.0], roughness=0.25)
    dumped = item.to_dict()
    assert dumped["roughness"] == 0.25
    assert "mu" not in dumped
