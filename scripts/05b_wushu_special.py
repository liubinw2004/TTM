"""武术专项：生成结果与训练集做「运动特征自类聚」，并与球类/日常并列对照。

为什么要单独做这一件：M2' 全量口径里 ``4.wushu`` 是**全类别最差**的一档
（`root 5.98cm / joint 13.20°`，全局是 `1.87cm / 5.90°`）。这不意外——武术只有
833 窗 / 24 个条件，动作幅度又大——但「最差」到底差在**重建精度**还是差在
**条件可控性**，是两个完全不同的问题：

* 重建差 → VAE 侧问题，M2' 的逐类数字已经答了；
* 条件不可控（说「打长拳」却生成出扫地）→ 扩散 + CFG 侧问题，必须在这里量化。

所以本脚本不看 loss、不看像素，只回答一句：**生成出来的动作，在运动特征空间里
落不落在它该落的那个条件附近？**

做法（自类聚 / self-clustering）：

1. 用**冻结 VAE** 把所有训练窗口编码成潜变量，对 token 维取均值 → 128 维描述子，
   L2 归一化；按条件（179 个）算类中心；
2. 生成样本走**完全相同**的路径（生成特征 → 再归一化 → VAE 编码 → 描述子），
   保证两边可比。注意生成侧特征在 :meth:`MotionGenerator.generate` 里已被
   ``denormalize`` 成真实单位，所以必须先 ``normalize`` 回去，否则 VAE 看到的是
   分布外的输入，描述子毫无意义；
3. 每个生成样本找最近的条件中心，统计「落在预期大类里的比例」（hit rate）、
   「到正确类的 margin」、「同 prompt 不同 seed 的稳定性」。

三个组并列，才能判断「武术差」是武术特有的，还是普遍现象：
``4.wushu`` / ``3.3.Ball_Interaction`` / 日常类。

用法::

    python scripts/05b_wushu_special.py --config configs/tier_s.yaml
    python scripts/05b_wushu_special.py --config configs/tier_s.yaml --seeds 3 --duration 5

退出码：0 正常出报告；3 权重/数据缺失；2 torch 不可用。
"""

from __future__ import annotations

import argparse
import itertools
import json
import sys
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Sequence

# --- sys.path 引导 -----------------------------------------------------------
_PROJECT_ROOT = Path(__file__).resolve().parents[1]
_SRC_DIR = _PROJECT_ROOT / "src"
if str(_SRC_DIR) not in sys.path:
    sys.path.insert(0, str(_SRC_DIR))

import numpy as np  # noqa: E402

from ttm.common import paths as P  # noqa: E402
from ttm.common.config import load_config  # noqa: E402
from ttm.common.logutil import close_all, get_logger  # noqa: E402

#: 三组对照指令 + 各自预期的大类（``""`` = 不预设，只报最近条件）。
#:
#: 武术组刻意选三类差异大的：长拳（舒展、位移大）、太极拳（慢、重心低）、
#: 剑术（上肢为主）——如果连这三条都分不开，说明条件空间里武术这一块的几何
#: 是糊的，不是「某一条指令写得不好」。
GROUPS: tuple[tuple[str, tuple[tuple[str, str], ...]], ...] = (
    (
        "武术",
        (
            ("打一套长拳第七段", "4.wushu"),
            ("打一套太极拳", "4.wushu"),
            ("练一套剑术", "4.wushu"),
        ),
    ),
    (
        "球类",
        (
            ("打篮球运球", "3.3.Ball_Interaction"),
            ("踢足球", "3.3.Ball_Interaction"),
            ("打羽毛球", "3.3.Ball_Interaction"),
        ),
    ),
    (
        "日常",
        (
            ("扫地", "1.12.Action_Sequence_Complex_Behavior"),
            # ``上下楼梯`` 这个条件挂在 2.1（基础环境交互）下，不是 2.2（复杂地形）。
            # 第一版把预期写成 2.2 是我的错判：实测生成出「上下楼梯」本该算命中，
            # 却被记成 miss —— 预期标错会系统性地低估可控性，必须按数据改正。
            ("慢慢走上楼梯", "2.1.Basic_Environment_Interaction_Class"),
            ("拿起桌上的杯子放到右边", ""),
        ),
    ),
)


@dataclass
class ReferenceBank:
    """训练集条件中心（VAE 潜变量描述子）+ 逐窗运动统计（多样性地板的分母）。

    ``cond_mu`` / ``cond_rough`` 是 (c)(d) 两个多样性指标的分母：

    * ``cond_mu[label]``：该条件下每个窗口的**时间均值特征** ``[146]``——
      跨窗口的方差就是「同一条件在真实数据里的多样性」；
    * ``cond_rough[label]``：该条件下每个窗口的**时间粗糙度**（帧间一阶差分绝对值的
      均值，在归一化特征上算）——生成动作比它低太多是「僵直」，高太多是「抖动」。
    """

    centroids: np.ndarray = field(default_factory=lambda: np.zeros((0, 128)))
    labels: list[str] = field(default_factory=list)
    categories: list[str] = field(default_factory=list)
    is_wushu: list[bool] = field(default_factory=list)
    n_windows: list[int] = field(default_factory=list)
    total_windows: int = 0
    cond_mu: dict[str, list[np.ndarray]] = field(default_factory=dict)
    cond_rough: dict[str, list[float]] = field(default_factory=dict)


def _l2(matrix: np.ndarray) -> np.ndarray:
    """按行 L2 归一化。"""
    norms = np.linalg.norm(matrix, axis=1, keepdims=True)
    return matrix / np.clip(norms, 1e-12, None)


def encode_descriptors(vae: Any, feats: np.ndarray, *, device: Any, batch_size: int = 64) -> np.ndarray:
    """把 ``[N,146,T]`` 特征批量编码成 ``[N,128]`` 描述子（潜变量对 token 取均值）。

    Args:
        vae: 冻结的 MotionVAE。
        feats: ``[N,146,T]`` 归一化特征。
        device: 设备。
        batch_size: 批大小。

    Returns:
        ``[N,128]`` L2 归一化描述子。
    """
    import torch  # noqa: PLC0415 - 与脚本主体解耦，便于无 torch 环境导入本模块

    out: list[np.ndarray] = []
    with torch.no_grad():
        for start in range(0, feats.shape[0], batch_size):
            chunk = torch.from_numpy(feats[start : start + batch_size]).to(device)
            mu, _logvar = vae.encode(chunk)
            descriptor = mu.mean(dim=1).float().cpu().numpy()  # [B,128]
            out.append(descriptor)
    return _l2(np.concatenate(out, axis=0))


def build_reference_bank(
    cfg: Any,
    vae: Any,
    *,
    device: Any,
    split: str = "train",
    batch_size: int = 64,
    logger: Any | None = None,
) -> ReferenceBank:
    """用训练窗口建条件中心。

    Args:
        cfg: 配置。
        vae: 冻结 VAE。
        device: 设备。
        split: 用哪个 split 建库（应该用 train，和训练同分布）。
        batch_size: 编码批大小。
        logger: 可注入 logger。

    Returns:
        :class:`ReferenceBank`。
    """
    from ttm.data.dataset import MotionDataset, auto_cache_size, collate_samples  # noqa: PLC0415

    log = logger or get_logger("ttm.eval.wushu_special", script="05b_wushu_special")
    dataset = MotionDataset(
        cfg.shards_path(),
        cache_size=auto_cache_size(cfg.shards_path()),
        window=cfg.data.window,
        feature_dim=cfg.motion.feature_dim,
        labels_path=cfg.train_labels_path(),
        split=split,
        with_torch=True,
    )
    log.info("参考库: %s split 共 %d 个窗口", split, len(dataset))

    feats: list[np.ndarray] = []
    labels: list[str] = []
    categories: list[str] = []
    for start in range(0, len(dataset), batch_size):
        items = [dataset[index] for index in range(start, min(start + batch_size, len(dataset)))]
        batch = collate_samples(items)
        # dataset __getitem__ 已是 channel-first [146,150]，collate 后即 [B,146,T]，
        # **不要**再 transpose（转完反而把时间/特征两轴对调——描述子不受影响是
        # 因为 MotionVAE.encode 对两种布局都自动转正，但逐窗统计会算错轴）
        feats.append(batch["feat"].numpy())  # [B,146,T]
        # 标签与特征必须**同序**：都从 batch["clip_id"] 推出来，不要另起一轮
        # 重新遍历数据集（那样一旦批顺序变化，特征和标签就错位了）
        for clip_id in batch["clip_id"]:
            info = dataset.labels.get(str(clip_id)) or {}
            labels.append(str(info.get("zh_label") or ""))
            categories.append(str(info.get("category_l1") or ""))

    descriptors = encode_descriptors(vae, np.concatenate(feats, axis=0), device=device)
    bank = ReferenceBank(total_windows=len(labels))
    all_feats = np.concatenate(feats, axis=0)  # [N,146,T]
    # 逐窗时间均值与时间粗糙度：(c)(d) 两个多样性指标的分母，一次遍历顺手算完
    window_mu = all_feats.mean(axis=2)  # [N,146]
    window_rough = np.abs(np.diff(all_feats, axis=2)).mean(axis=(1, 2))  # [N]
    by_label: dict[str, list[int]] = {}
    for index, label in enumerate(labels):
        by_label.setdefault(label, []).append(index)
    for label in sorted(by_label):
        rows = np.asarray(by_label[label], dtype=np.int64)
        centroid = _l2(descriptors[rows].mean(axis=0, keepdims=True))
        bank.centroids = np.vstack([bank.centroids, centroid]) if bank.centroids.size else centroid
        bank.labels.append(label)
        # 同一条件的窗口必然同大类，取第一条非空即可（空说明标签层缺 category_l1）
        bank.categories.append(next((categories[i] for i in rows if categories[i]), ""))
        bank.is_wushu.append("wushu" in bank.categories[-1].lower())
        bank.n_windows.append(int(rows.size))
        bank.cond_mu[label] = [window_mu[i] for i in rows]
        bank.cond_rough[label] = [float(window_rough[i]) for i in rows]
    log.info("条件中心就绪: %d 个条件（武术 %d 个）", len(bank.labels), sum(bank.is_wushu))
    return bank


@dataclass
class SampleResult:
    """单个生成样本的自类聚结果。"""

    group: str
    prompt: str
    expected: str
    seed: int
    descriptor: np.ndarray
    mu: np.ndarray = field(default_factory=lambda: np.zeros(0))
    roughness: float = 0.0
    nearest_label: str = ""
    nearest_category: str = ""
    sim_to_nearest: float = 0.0
    sim_to_expected: float = 0.0
    margin: float = 0.0
    hit: bool = False
    seam_max_cmps: float = 0.0
    total_ms: float = 0.0

    def to_dict(self) -> dict[str, Any]:
        """序列化（描述子不落盘，太大且无解释价值）。"""
        return {
            "group": self.group,
            "prompt": self.prompt,
            "expected": self.expected,
            "seed": int(self.seed),
            "nearest_label": self.nearest_label,
            "nearest_category": self.nearest_category,
            "is_wushu_nearest": "wushu" in self.nearest_category.lower(),
            "sim_to_nearest": round(float(self.sim_to_nearest), 4),
            "sim_to_expected": round(float(self.sim_to_expected), 4),
            "margin": round(float(self.margin), 4),
            "hit": bool(self.hit),
            "roughness": round(float(self.roughness), 4),
            "seam_max_cmps": round(float(self.seam_max_cmps), 3),
            "total_ms": round(float(self.total_ms), 1),
        }


def classify(
    descriptor: np.ndarray, bank: ReferenceBank, *, expected: str
) -> tuple[str, str, float, float, float]:
    """找最近的条件中心，并算出到预期类的 margin。

    ``margin`` 的定义（统一口径，两种情形都同向：越大越好）::

        有预期类：margin = 预期类里最像的那个 sim − 其它类里最像的那个 sim
        无预期类：margin = 最近 sim − 次近 sim

    Args:
        descriptor: ``[128]`` 已归一化描述子。
        bank: 条件中心库。
        expected: 预期大类（``""`` 表示不预设）。

    Returns:
        ``(最近条件名, 最近条件所属大类, 到最近条件的相似度, 到预期类的相似度, margin)``。
    """
    sims = bank.centroids @ descriptor
    best = int(np.argmax(sims))
    sim_nearest = float(sims[best])
    if not expected:
        others = np.delete(sims, best)
        second = float(others.max()) if others.size else 0.0
        return bank.labels[best], bank.categories[best], sim_nearest, sim_nearest, sim_nearest - second

    in_expected = np.asarray([item == expected for item in bank.categories], dtype=bool)
    if not in_expected.any():
        # 预期类在参考库里不存在：只能报最近条件，margin 置 0（不伪造结论）
        return bank.labels[best], bank.categories[best], sim_nearest, 0.0, 0.0
    sim_expected = float(sims[in_expected].max())
    sim_other = float(sims[~in_expected].max()) if (~in_expected).any() else 0.0
    return bank.labels[best], bank.categories[best], sim_nearest, sim_expected, sim_expected - sim_other


def summarize(samples: Sequence[SampleResult]) -> dict[str, Any]:
    """按组汇总（hit rate / margin / seed 稳定性）。"""
    report: dict[str, Any] = {}
    for group in dict.fromkeys(item.group for item in samples):
        rows = [item for item in samples if item.group == group]
        judged = [item for item in rows if item.expected]
        hits = [item for item in judged if item.hit]
        # seed 稳定性：同一 prompt 不同 seed 的描述子两两余弦
        stability: list[float] = []
        by_prompt: dict[str, list[np.ndarray]] = {}
        for item in rows:
            by_prompt.setdefault(item.prompt, []).append(item.descriptor)
        for vectors in by_prompt.values():
            for a, b in itertools.combinations(vectors, 2):
                stability.append(float(a @ b))
        report[group] = {
            "n_samples": len(rows),
            "n_judged": len(judged),
            "hits": len(hits),
            "hit_rate": round(len(hits) / len(judged), 4) if judged else None,
            "margin_mean": round(float(np.mean([item.margin for item in judged])), 4) if judged else None,
            "sim_to_nearest_mean": round(float(np.mean([item.sim_to_nearest for item in rows])), 4),
            "seed_stability_mean": round(float(np.mean(stability)), 4) if stability else None,
            "nearest_labels": [item.nearest_label for item in rows],
            "seam_max_cmps": round(float(max(item.seam_max_cmps for item in rows)), 3),
        }
    return report


def _majority(labels: Sequence[str]) -> str:
    """取多数票（平票取先出现者，确定性）。

    用途：同一条指令的多个 seed 各自会聚到某个最近条件，取多数票作为
    该指令的 **GT 参考条件**——(c)(d) 两个指标的分母都从这个条件取。
    """
    counts: dict[str, int] = {}
    for item in labels:
        counts[item] = counts.get(item, 0) + 1
    return max(counts, key=lambda k: (counts[k], -labels.index(k)))


#: (c) 的达标下限：跨 seed 特征方差至少要有 GT 同条件方差的一半，
#: 否则说明 CFG 把同一条指令压成了同一个动作（多样性地板被击穿）。
VAR_RATIO_MIN = 0.5

#: (d) 的达标带：生成帧间粗糙度落在 GT 同条件的 0.6~1.5 倍之间。
#: 低于 0.6 是「僵直」（CFG 过强的典型症状），高于 1.5 是「抖动」。
ROUGH_RATIO_LO = 0.6
ROUGH_RATIO_HI = 1.5


def _dimwise_var(matrix: np.ndarray) -> float:
    """``[N,D]`` 逐维方差再对维取均值（ddof=1，样本方差）。"""
    if matrix.shape[0] < 2:
        return 0.0
    return float(np.var(matrix, axis=0, ddof=1).mean())


def diversity_metrics(samples: Sequence[SampleResult], bank: ReferenceBank) -> dict[str, Any]:
    """多样性地板指标（QA M-07 防线，sweep 选档的必要条件）。

    两个指标都以「GT 同条件」为分母（参考库编码时顺手存下的逐窗统计）：

    * **(c) ``var_ratio``** = 同指令跨 seed 的特征时间均值方差 ÷ GT 同条件
      逐窗时间均值方差。衡量「同指令不同 seed 能不能长出不一样的动作」。
      ``>= :data:`VAR_RATIO_MIN` 才算过；趋零说明 CFG 过强把同指令压成同一动作。
    * **(d) ``rough_ratio``** = 生成帧间一阶差分绝对值均值 ÷ GT 同条件。
      必须落在 ``[ROUGH_RATIO_LO, ROUGH_RATIO_HI]``：低于下限是僵直
      （动作没了细节），高于上限是抖动（高频噪声）。

    Args:
        samples: 本档 guidance 下的全部生成样本（每条指令 ≥2 个 seed 才能
            算 (c)；单 seed 的指令 (c) 置 None，不参与均值）。
        bank: 含 ``cond_mu`` / ``cond_rough`` 的参考库。

    Returns:
        ``{"by_prompt": {...}, "var_ratio_mean": ..., "rough_ratio_mean": ...,
        "var_ratio_pass": ..., "rough_ratio_pass": ...}``。
    """
    by_prompt: dict[str, list[SampleResult]] = {}
    for item in samples:
        by_prompt.setdefault(item.prompt, []).append(item)

    details: dict[str, Any] = {}
    var_ratios: list[float] = []
    rough_ratios: list[float] = []
    for prompt, rows in by_prompt.items():
        # GT 参考条件：各 seed 最近条件的多数票（预期大类太粗，不落到具体条件）
        reference = _majority([item.nearest_label for item in rows if item.nearest_label])
        gt_mus = bank.cond_mu.get(reference, [])
        gt_rough = bank.cond_rough.get(reference, [])
        gt_var = _dimwise_var(np.stack(gt_mus, axis=0)) if len(gt_mus) >= 2 else None
        gt_rough_mean = float(np.mean(gt_rough)) if gt_rough else None

        # (c) 跨 seed 特征均值方差 ÷ GT 同条件逐窗方差
        var_ratio: float | None = None
        if len(rows) >= 2 and gt_var and gt_var > 0.0:
            gen_var = _dimwise_var(np.stack([item.mu for item in rows], axis=0))
            var_ratio = gen_var / gt_var
            var_ratios.append(var_ratio)

        # (d) 生成帧间粗糙度 ÷ GT 同条件
        rough_ratio: float | None = None
        if gt_rough_mean and gt_rough_mean > 0.0:
            rough_ratio = float(np.mean([item.roughness for item in rows])) / gt_rough_mean
            rough_ratios.append(rough_ratio)

        details[prompt] = {
            "reference_condition": reference,
            "n_gt_windows": len(gt_mus),
            "gt_var": round(gt_var, 8) if gt_var is not None else None,
            "gen_var": round(_dimwise_var(np.stack([item.mu for item in rows], axis=0)), 8)
            if len(rows) >= 2
            else None,
            "var_ratio": round(var_ratio, 4) if var_ratio is not None else None,
            "gt_roughness": round(gt_rough_mean, 6) if gt_rough_mean is not None else None,
            "gen_roughness": round(float(np.mean([item.roughness for item in rows])), 6),
            "rough_ratio": round(rough_ratio, 4) if rough_ratio is not None else None,
        }

    var_mean = float(np.mean(var_ratios)) if var_ratios else None
    rough_mean = float(np.mean(rough_ratios)) if rough_ratios else None
    return {
        "by_prompt": details,
        "var_ratio_mean": round(var_mean, 4) if var_mean is not None else None,
        "rough_ratio_mean": round(rough_mean, 4) if rough_mean is not None else None,
        "var_ratio_pass": bool(var_mean is not None and var_mean >= VAR_RATIO_MIN),
        "rough_ratio_pass": bool(
            rough_mean is not None and ROUGH_RATIO_LO <= rough_mean <= ROUGH_RATIO_HI
        ),
    }


def pick_guidance(rows: Sequence[dict[str, Any]]) -> dict[str, Any] | None:
    """按选值规则从 sweep 各档里挑最终 guidance。

    规则（主理人裁决 1）：在 **(c)≥0.5 且 (d)∈[0.6,1.5]** 的档里选**整体
    hit 率最高**的；hit 率并列时 margin 高者优先，再并列取更小的 guidance
    （过强的 CFG 只会继续压多样性）。

    Args:
        rows: sweep 各档汇总（含 ``guidance`` / ``overall_hit_rate`` /
            ``overall_margin_mean`` / ``diversity``）。

    Returns:
        达标的最佳档（原 dict 的浅拷贝，附 ``recommended: true``）；
        没有任何档同时过 (c)(d) 时返回 ``None``。
    """
    candidates = [
        row
        for row in rows
        if row.get("diversity", {}).get("var_ratio_pass") and row.get("diversity", {}).get("rough_ratio_pass")
    ]
    if not candidates:
        return None
    best = min(
        candidates,
        key=lambda row: (
            -float(row["overall"]["hit_rate"]),
            -float(row["overall"]["margin_mean"] or 0.0),
            float(row["guidance"]),
        ),
    )
    result = dict(best)
    result["recommended"] = True
    return result


def _short_category(category: str) -> str:
    """把 ``4.wushu`` / ``3.3.Ball_Interaction`` 压成短标签（表格里能对齐的宽度）。

    只取前两段：``3.3`` / ``4.wushu`` / ``1.12``。只取第一段会让 ``3.3`` 与 ``3.1``
    看起来一样，反而看不出差别。
    """
    parts = str(category).split(".")
    return ".".join(parts[:2]) if len(parts) > 1 else str(category)


def run_generation(
    generator: Any,
    bank: ReferenceBank,
    cfg: Any,
    *,
    guidance: float,
    steps: int,
    duration_s: float,
    seeds: int,
    device: Any,
    log: Any,
) -> list[SampleResult]:
    """跑全部指令 × seed 的生成与自类聚（不写盘，供单档与 sweep 两种模式复用）。

    Args:
        generator: 已加载的 :class:`MotionGenerator`。
        bank: 条件中心参考库。
        cfg: 配置（取基础 seed）。
        guidance: CFG 强度。
        steps: DDIM 步数。
        duration_s: 每条生成时长（秒）。
        seeds: 每条指令的 seed 数。
        device: 设备。
        log: logger。

    Returns:
        样本结果列表（长度 = 指令数 × seeds）。
    """
    from ttm.infer.generate import GenerateOptions  # noqa: PLC0415 - 保持模块可无 torch 导入

    samples: list[SampleResult] = []
    for group, prompts in GROUPS:
        for prompt, expected in prompts:
            for seed in range(int(seeds)):
                motion = generator.generate(
                    prompt,
                    GenerateOptions(
                        duration_s=float(duration_s),
                        steps=steps,
                        guidance=guidance,
                        seed=int(cfg.runtime.seed) + seed,
                        export_bvh=False,
                    ),
                )
                # 生成侧特征是真实单位，必须先归一化回 VAE 的输入分布
                normalized = motion.feature.normalize(generator.norm_stats)
                feat = np.asarray(normalized.feat, dtype=np.float32).T[None]  # [1,146,T]
                descriptor = encode_descriptors(generator.vae, feat, device=device)[0]
                # (c) 的分子：同指令跨 seed 的特征时间均值；(d) 的分子：帧间粗糙度
                # （GT 侧的对应分母在 build_reference_bank 里逐窗顺手存好）
                sample_mu = feat[0].mean(axis=1)  # [146] 时间均值
                sample_rough = float(np.abs(np.diff(feat, axis=2)).mean())
                nearest_label, nearest_cat, sim_nearest, sim_expected, margin = classify(
                    descriptor, bank, expected=expected
                )
                # 命中判定：武术类的大类目录是 ``4.wushu``（与 ``4.1`` / ``4.3`` 前缀撞名，
                # 只能按子串判）；其它类按 category_l1 精确匹配
                if expected == "4.wushu":
                    hit = "wushu" in nearest_cat.lower()
                elif expected:
                    hit = nearest_cat == expected
                else:
                    hit = False
                samples.append(
                    SampleResult(
                        group=group,
                        prompt=prompt,
                        expected=expected,
                        seed=int(cfg.runtime.seed) + seed,
                        descriptor=descriptor,
                        mu=sample_mu,
                        roughness=sample_rough,
                        nearest_label=nearest_label,
                        nearest_category=nearest_cat,
                        sim_to_nearest=sim_nearest,
                        sim_to_expected=sim_expected,
                        margin=margin,
                        hit=hit,
                        seam_max_cmps=float(motion.seam_jumps.max()) if motion.seam_jumps.size else 0.0,
                        total_ms=float(motion.timing_ms.get("total_ms", 0.0)),
                    )
                )
                log.info(
                    "%s | %r seed=%d → 最近条件 %r（%s）sim=%.3f %s",
                    group,
                    prompt,
                    seed,
                    nearest_label,
                    _short_category(nearest_cat),
                    sim_nearest,
                    "HIT" if hit else "",
                )
    return samples


def _overall(samples: Sequence[SampleResult]) -> dict[str, Any]:
    """全体样本（有预期类的）整体 hit 率 / margin / seed 稳定性。"""
    judged = [item for item in samples if item.expected]
    stability: list[float] = []
    by_prompt: dict[str, list[np.ndarray]] = {}
    for item in samples:
        by_prompt.setdefault(item.prompt, []).append(item.descriptor)
    for vectors in by_prompt.values():
        for a, b in itertools.combinations(vectors, 2):
            stability.append(float(a @ b))
    return {
        "n_samples": len(samples),
        "n_judged": len(judged),
        "hit_rate": round(len([item for item in judged if item.hit]) / len(judged), 4) if judged else None,
        "margin_mean": round(float(np.mean([item.margin for item in judged])), 4) if judged else None,
        "seed_stability_mean": round(float(np.mean(stability)), 4) if stability else None,
    }


def _fmt(value: Any) -> str:
    """表格单元格式化（None 显示为 ``-``）。"""
    return "-" if value is None else (f"{value:.4f}" if isinstance(value, float) else str(value))


def _print_sweep_table(rows: Sequence[dict[str, Any]]) -> None:
    """打印 sweep 四点对照表：hit / margin / seed 稳 / (c) / (d) / 判定。"""
    print("-" * 96)
    print(
        f"  {'guid':>5}{'整体hit':>9}{'margin':>9}{'seed稳':>9}"
        f"{'(c)方差比':>10}{'(d)粗糙比':>10}{'分组hit率(武/球/日)':>20}  判定"
    )
    for row in rows:
        diversity = row["diversity"]
        groups = row["summary_by_group"]
        group_hits = "/".join(
            _fmt(groups.get(name, {}).get("hit_rate")) if groups.get(name, {}).get("hit_rate") is not None else "-"
            for name in ("武术", "球类", "日常")
        )
        flags = []
        if not diversity["var_ratio_pass"]:
            flags.append("c✗")
        if not diversity["rough_ratio_pass"]:
            flags.append("d✗")
        verdict = "PASS" if not flags else " ".join(flags)
        marker = " ←推荐" if row.get("recommended") else ""
        print(
            f"  {row['guidance']:>5.1f}{_fmt(row['overall']['hit_rate']):>9}"
            f"{_fmt(row['overall']['margin_mean']):>9}{_fmt(row['overall']['seed_stability_mean']):>9}"
            f"{_fmt(diversity['var_ratio_mean']):>10}{_fmt(diversity['rough_ratio_mean']):>10}"
            f"{group_hits:>20}  {verdict}{marker}"
        )
    print("-" * 96)


def build_argparser() -> argparse.ArgumentParser:
    """构造命令行参数解析器。"""
    parser = argparse.ArgumentParser(
        description="武术专项自类聚（生成 vs 训练条件的运动特征对照）",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument("--config", default="configs/default.yaml", help="配置文件路径")
    parser.add_argument("--seeds", type=int, default=3, help="每条指令的 seed 数")
    parser.add_argument("--duration", type=float, default=5.0, help="每条生成时长（秒）")
    parser.add_argument("--steps", type=int, default=0, help="DDIM 步数（0 = 用配置）")
    parser.add_argument("--guidance", type=float, default=0.0, help="CFG 强度（0 = 用配置）")
    parser.add_argument(
        "--guidance-sweep",
        default="",
        help="sweep 模式：逗号分隔的多个 CFG 强度（如 3.0,4.5,6.0,7.5），逐档生成并对比 (c)(d) 多样性指标",
    )
    parser.add_argument("--out", default="runs/eval/wushu_special.json", help="报告输出路径")
    return parser


def main(argv: list[str] | None = None) -> int:
    """命令行入口。

    Args:
        argv: 参数列表（默认取 ``sys.argv[1:]``）。

    Returns:
        进程退出码。
    """
    args = build_argparser().parse_args(argv)
    cfg = load_config(args.config)
    log = get_logger("ttm.eval.wushu_special", script="05b_wushu_special")

    try:
        import torch  # noqa: PLC0415
    except Exception as exc:  # noqa: BLE001
        log.error("torch 不可用（%s）", exc)
        close_all()
        return 2

    from ttm.infer.generate import GenerateOptions, load_generator  # noqa: PLC0415

    device = torch.device(cfg.runtime.device if torch.cuda.is_available() else "cpu")
    try:
        generator = load_generator(cfg, device=device, logger=log)
    except FileNotFoundError as exc:
        log.error("%s", exc)
        close_all()
        return 3

    bank = build_reference_bank(cfg, generator.vae, device=device, split="train", logger=log)

    steps = int(args.steps if args.steps > 0 else cfg.diffusion.ddim_steps)
    sweep = [float(item) for item in args.guidance_sweep.split(",") if item.strip()] if args.guidance_sweep else []

    if sweep:
        return _run_sweep(
            args,
            cfg,
            generator,
            bank,
            sweep=sweep,
            steps=steps,
            device=device,
            log=log,
        )

    guidance = float(args.guidance if args.guidance > 0 else cfg.runtime.guidance)
    started = time.perf_counter()
    samples = run_generation(
        generator,
        bank,
        cfg,
        guidance=guidance,
        steps=steps,
        duration_s=float(args.duration),
        seeds=int(args.seeds),
        device=device,
        log=log,
    )
    elapsed = time.perf_counter() - started

    report = {
        "config": args.config,
        "duration_s": float(args.duration),
        "steps": steps,
        "guidance": guidance,
        "seeds_per_prompt": int(args.seeds),
        "elapsed_s": round(elapsed, 1),
        "reference": {
            "split": "train",
            "n_conditions": len(bank.labels),
            "n_wushu_conditions": int(sum(bank.is_wushu)),
            "n_windows": int(bank.total_windows),
        },
        "overall": _overall(samples),
        "summary_by_group": summarize(samples),
        "diversity": diversity_metrics(samples, bank),
        "samples": [item.to_dict() for item in samples],
    }

    out = P.resolve_out(args.out, mkparents=True)
    out.write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")

    print("=" * 88)
    print("武术专项：生成结果 × 训练条件自类聚")
    print("=" * 88)
    print(f"参考库            : {bank.total_windows} 个 train 窗口 / {len(bank.labels)} 个条件（武术 {sum(bank.is_wushu)}）")
    print(f"生成设置          : {args.duration:.0f}s / {steps} 步 DDIM / guidance {guidance} / 每指令 {args.seeds} seed")
    print("-" * 88)
    print(f"  {'组':<6}{'样本':>6}{'判定':>6}{'命中':>6}{'hit率':>8}{'margin':>9}{'最近sim':>9}{'seed稳':>8}{'接缝max':>9}")
    for group, stats in report["summary_by_group"].items():
        print(
            f"  {group:<6}{stats['n_samples']:>6}{stats['n_judged']:>6}{stats['hits']:>6}"
            f"{_fmt(stats['hit_rate']):>8}{_fmt(stats['margin_mean']):>9}"
            f"{stats['sim_to_nearest_mean']:>9.4f}{_fmt(stats['seed_stability_mean']):>8}"
            f"{stats['seam_max_cmps']:>9.1f}"
        )
    print("-" * 88)
    diversity = report["diversity"]
    print(
        f"多样性地板        : (c) 跨seed方差/GT方差 = {_fmt(diversity['var_ratio_mean'])}"
        f"（≥{VAR_RATIO_MIN} {'过' if diversity['var_ratio_pass'] else '不过'}）"
    )
    print(
        f"                    (d) 帧间粗糙度/GT = {_fmt(diversity['rough_ratio_mean'])}"
        f"（∈[{ROUGH_RATIO_LO},{ROUGH_RATIO_HI}] {'过' if diversity['rough_ratio_pass'] else '不过'}）"
    )
    print("-" * 88)
    print("逐条最近条件:")
    for item in samples:
        print(
            f"  {item.group:<4}{item.prompt:<22}seed={item.seed:<3} → {item.nearest_label:<12}"
            f"({_short_category(item.nearest_category):<8}) sim={item.sim_to_nearest:.3f}"
            f" {'HIT' if item.hit else '   '}"
        )
    print("=" * 88)
    print(f"报告文件: {out.as_posix()}")
    close_all()
    return 0


def _run_sweep(
    args: argparse.Namespace,
    cfg: Any,
    generator: Any,
    bank: ReferenceBank,
    *,
    sweep: Sequence[float],
    steps: int,
    device: Any,
    log: Any,
) -> int:
    """sweep 模式：逐档 guidance 生成 + (c)(d) 多样性地板，出四点对照表。

    选值规则（主理人裁决 1）：只有 **(c)≥0.5 且 (d)∈[0.6,1.5]** 的档才有资格
    写进 ``runtime.guidance``；资格档里整体 hit 率最高者胜。

    Returns:
        进程退出码（0 = 出表；2 = 没有任何档过 (c)(d)）。
    """
    rows: list[dict[str, Any]] = []
    for guidance in sweep:
        log.info("sweep 开始: guidance=%.2f", guidance)
        started = time.perf_counter()
        samples = run_generation(
            generator,
            bank,
            cfg,
            guidance=guidance,
            steps=steps,
            duration_s=float(args.duration),
            seeds=int(args.seeds),
            device=device,
            log=log,
        )
        elapsed = time.perf_counter() - started
        row = {
            "guidance": guidance,
            "duration_s": float(args.duration),
            "steps": steps,
            "seeds_per_prompt": int(args.seeds),
            "elapsed_s": round(elapsed, 1),
            "overall": _overall(samples),
            "summary_by_group": summarize(samples),
            "diversity": diversity_metrics(samples, bank),
            "samples": [item.to_dict() for item in samples],
        }
        rows.append(row)
        log.info(
            "sweep 完成: guidance=%.2f hit=%.4f (c)=%s (d)=%s (%.1fs)",
            guidance,
            row["overall"]["hit_rate"] if row["overall"]["hit_rate"] is not None else -1.0,
            row["diversity"]["var_ratio_mean"],
            row["diversity"]["rough_ratio_mean"],
            elapsed,
        )

    recommended = pick_guidance(rows)

    report = {
        "config": args.config,
        "mode": "guidance_sweep",
        "duration_s": float(args.duration),
        "steps": steps,
        "seeds_per_prompt": int(args.seeds),
        "reference": {
            "split": "train",
            "n_conditions": len(bank.labels),
            "n_wushu_conditions": int(sum(bank.is_wushu)),
            "n_windows": int(bank.total_windows),
        },
        "var_ratio_min": VAR_RATIO_MIN,
        "rough_ratio_band": [ROUGH_RATIO_LO, ROUGH_RATIO_HI],
        "sweep": rows,
        "recommended_guidance": recommended["guidance"] if recommended else None,
    }
    out = P.resolve_out(args.out, mkparents=True)
    out.write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")

    print("=" * 96)
    print("guidance sweep：可控性 × 多样性地板（(c) 跨seed方差/GT方差 ≥0.5，(d) 粗糙度/GT ∈[0.6,1.5]）")
    print("=" * 96)
    print(f"参考库: {bank.total_windows} 个 train 窗口 / {len(bank.labels)} 个条件")
    print(f"设置  : {args.duration:.0f}s / {steps} 步 DDIM / 每指令 {args.seeds} seed")
    _print_sweep_table(rows)
    if recommended is not None:
        print(
            f"推荐 guidance = {recommended['guidance']:.1f}"
            f"（hit {_fmt(recommended['overall']['hit_rate'])} / (c) {_fmt(recommended['diversity']['var_ratio_mean'])}"
            f" / (d) {_fmt(recommended['diversity']['rough_ratio_mean'])}）"
        )
    else:
        print("没有任何档同时过 (c)(d) 多样性地板——需要回看 CFG 上限或改判定，不能直接定值。")
    print(f"报告文件: {out.as_posix()}")
    close_all()
    return 0 if recommended is not None else 2


if __name__ == "__main__":
    raise SystemExit(main())
