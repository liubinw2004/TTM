"""数据层：BVH 解析、骨架运动学、特征提取、标签生成、预处理与数据集。

对外契约（见 ``TTM-ARCH.md`` §5）：

* :mod:`ttm.data.skeleton` — :class:`SkeletonDef`，纯运动学工具（FK / 重采样 /
  足接触），不含文件解析；
* :mod:`ttm.data.bvh` — :class:`BvhParser`，纯文本解析 + 关节数校验；
* :mod:`ttm.data.scan` — :class:`Scanner`，目录枚举 / Take 去重 / 8:1:1 切分；
* :mod:`ttm.data.features` — :class:`FeatureExtractor`，146 维特征 + 归一化；
* :mod:`ttm.data.labels_zh` — :class:`LabelGenerator`，目录 → 中文 → 模板扩写；
* :mod:`ttm.data.preprocess` — :class:`PreprocessPipeline`，单趟流式 + 分片落盘；
* :mod:`ttm.data.dataset` — :class:`MotionDataset`，npz 分片随机读取。
"""

from __future__ import annotations

__all__ = ["skeleton"]
