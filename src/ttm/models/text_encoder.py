"""中文文本编码器：``BAAI/bge-small-zh-v1.5``（24M，冻结）+ MLP 投影到 256 维。

结构（``TTM-ARCH.md`` §1.2 / §5.8）：

::

    中文指令 "慢慢走上楼梯"
      │  bge-small-zh-v1.5（512d，冻结，完全离线本地加载）
      │  **CLS pooling（第 0 位）+ L2 归一化**（bge 官方用法）
      ▼  vec512 [B, 512]
      │  TextProjector: Linear(512→256) + GELU + Linear(256→256) + LayerNorm  （可训练）
      ▼  cond [B, 256]   → 扩散 Transformer 的交叉注意力条件

设计取舍：

* **backbone 冻结**，只有 :class:`TextProjector` 可训练 —— 24M 编码器在 12 GB 卡上
  用 fp16/bf16 跑前向即可，且避免小数据集上把语义先验训崩（PRD R6）；
* **完全离线**：默认从项目内 ``D:/TTM/models/bge-small-zh-v1.5`` 加载，并在进程内
  设置 ``TRANSFORMERS_OFFLINE=1`` / ``HF_HUB_OFFLINE=1``（禁止任何联网探测）；
  权重缺失时抛 :class:`TextEncoderUnavailable`，由上层决定是否降级为 null condition；
* 编码结果既是 numpy（``encode``）也是 torch（``forward``），前者便于落盘/缓存。
"""

from __future__ import annotations

import os
from pathlib import Path
from typing import Any, Sequence

import numpy as np
import torch
import torch.nn as nn

__all__ = [
    "BGE_HIDDEN_DIM",
    "DEFAULT_MODEL_PATH",
    "TextEncoder",
    "TextEncoderUnavailable",
    "TextProjector",
    "cosine_similarity",
    "null_condition",
]

#: bge-small-zh-v1.5 的句向量维度。
BGE_HIDDEN_DIM = 512

#: 项目内置的 bge 权重目录（由主理人用 ModelScope 源落盘，**禁止联网**）。
DEFAULT_MODEL_PATH = "D:/TTM/models/bge-small-zh-v1.5"


class TextEncoderUnavailable(RuntimeError):
    """文本编码器不可用（模型未下载 / transformers 缺失 / 离线不可解析）。"""


def null_condition(proj_dim: int, batch_size: int = 1, *, device: torch.device | str = "cpu") -> torch.Tensor:
    """构造 CFG 的「无条件」向量（零向量）。

    Args:
        proj_dim: 条件维数（256）。
        batch_size: batch 大小。
        device: 设备。

    Returns:
        ``[B, proj_dim]`` 全零张量。
    """
    return torch.zeros(batch_size, proj_dim, device=torch.device(device))


class TextProjector(nn.Module):
    """512d 句向量 → proj_dim（默认 256）的 MLP 投影（可训练）。"""

    def __init__(self, in_dim: int = BGE_HIDDEN_DIM, proj_dim: int = 256, *, dropout: float = 0.1) -> None:
        """初始化。

        Args:
            in_dim: 输入维数（bge 的 512）。
            proj_dim: 输出维数（与扩散模型 d_model 对齐）。
            dropout: dropout 比例。
        """
        super().__init__()
        self.in_dim = int(in_dim)
        self.proj_dim = int(proj_dim)
        self.net = nn.Sequential(
            nn.Linear(self.in_dim, self.proj_dim),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(self.proj_dim, self.proj_dim),
            nn.LayerNorm(self.proj_dim),
        )

    def forward(self, vec: torch.Tensor) -> torch.Tensor:
        """前向。

        Args:
            vec: ``[B, in_dim]`` 或 ``[B, *, in_dim]``。

        Returns:
            同前导维度的 ``[..., proj_dim]``。
        """
        if vec.shape[-1] != self.in_dim:
            raise ValueError(f"投影输入最后一维应为 {self.in_dim}，实际 {tuple(vec.shape)}")
        return self.net(vec)


class TextEncoder:
    """冻结的中文句编码器 + 可训练投影层。

    Attributes:
        model_name: HF 模型名。
        proj_dim: 投影维数。
        max_len: 分词最大长度。
        device: 计算设备。
    """

    def __init__(
        self,
        model_name: str = "BAAI/bge-small-zh-v1.5",
        *,
        model_path: str | os.PathLike[str] | None = None,
        proj_dim: int = 256,
        frozen: bool = True,
        max_len: int = 32,
        device: torch.device | str = "cpu",
        cache_dir: str | os.PathLike[str] | None = None,
        local_files_only: bool = True,
        dropout: float = 0.1,
    ) -> None:
        """初始化（**不**加载权重，首次使用时惰性加载）。

        Args:
            model_name: HF 模型名（仅用于日志/元信息，**不会**联网拉取）。
            model_path: 本地权重目录；``None`` 时用 :data:`DEFAULT_MODEL_PATH`
                （项目内 ``D:/TTM/models/bge-small-zh-v1.5``）。目录存在即走本地加载。
            proj_dim: 投影维数（与 ``diffusion.d_model`` 对齐）。
            frozen: 是否冻结 backbone。
            max_len: 分词最大长度。
            device: 设备。
            cache_dir: HF 缓存目录（仅当回退到 HF 缓存时才用）。
            local_files_only: 是否只允许本地文件（默认 True，禁止联网）。
            dropout: 投影层 dropout。
        """
        self.model_name = str(model_name)
        self.local_files_only = bool(local_files_only)
        # 本地权重优先；显式先设离线环境变量，避免 transformers 发起任何网络探测
        candidate = Path(model_path) if model_path else Path(DEFAULT_MODEL_PATH)
        if candidate.is_dir():
            self.model_source = str(candidate)
            self._force_offline_env()
        else:
            self.model_source = self.model_name
        self.proj_dim = int(proj_dim)
        self.frozen = bool(frozen)
        self.max_len = int(max_len)
        self.device = torch.device(device)
        self.cache_dir = (
            Path(cache_dir)
            if cache_dir
            else Path(os.environ.get("HF_HOME", str(Path("D:/TTM/.cache/hf"))))
        )

        self.projector = TextProjector(BGE_HIDDEN_DIM, self.proj_dim, dropout=dropout).to(self.device)
        self._tokenizer: Any | None = None
        self._backbone: nn.Module | None = None
        self._hidden_dim = BGE_HIDDEN_DIM

    # -- 惰性加载 -----------------------------------------------------------
    @staticmethod
    def _force_offline_env() -> None:
        """在进程内显式禁用 transformers / huggingface_hub 的联网行为。"""
        os.environ.setdefault("TRANSFORMERS_OFFLINE", "1")
        os.environ.setdefault("HF_HUB_OFFLINE", "1")
        os.environ.setdefault("HF_HUB_DISABLE_TELEMETRY", "1")

    def _ensure_backbone(self) -> None:
        """加载 tokenizer 与 backbone（失败时抛 :class:`TextEncoderUnavailable`）。

        Raises:
            TextEncoderUnavailable: transformers 缺失或模型不可用。
        """
        if self._backbone is not None and self._tokenizer is not None:
            return
        try:
            from transformers import AutoModel, AutoTokenizer  # noqa: PLC0415
        except Exception as exc:  # noqa: BLE001
            raise TextEncoderUnavailable(f"transformers 不可用: {exc}") from exc

        if self.local_files_only:
            self._force_offline_env()
        kwargs: dict[str, Any] = {
            "cache_dir": str(self.cache_dir),
            "local_files_only": self.local_files_only,
        }
        try:
            self._tokenizer = AutoTokenizer.from_pretrained(self.model_source, **kwargs)
            backbone = AutoModel.from_pretrained(self.model_source, **kwargs)
        except Exception as exc:  # noqa: BLE001
            raise TextEncoderUnavailable(
                f"无法加载文本编码器 {self.model_source}（cache_dir={self.cache_dir}, "
                f"local_files_only={self.local_files_only}）: {exc}"
            ) from exc

        backbone.eval()
        if self.frozen:
            for parameter in backbone.parameters():
                parameter.requires_grad = False
        self._backbone = backbone.to(self.device)
        self._hidden_dim = int(getattr(backbone.config, "hidden_size", BGE_HIDDEN_DIM))
        if self._hidden_dim != self.projector.in_dim:
            # 兜底：万一换了别的 backbone，重建投影层以匹配维度
            self.projector = TextProjector(self._hidden_dim, self.proj_dim).to(self.device)

    @property
    def is_loaded(self) -> bool:
        """backbone 是否已加载。"""
        return self._backbone is not None

    def is_available(self) -> bool:
        """模型是否可用（尝试加载，失败返回 False，不抛异常）。"""
        try:
            self._ensure_backbone()
            return True
        except TextEncoderUnavailable:
            return False

    # -- 编码 ---------------------------------------------------------------
    @torch.no_grad()
    def encode(self, texts: Sequence[str], *, batch_size: int = 16) -> np.ndarray:
        """把中文文本编码成 512d 句向量（**CLS pooling** + L2 归一化）。

        Args:
            texts: 文本序列。
            batch_size: 内部分批大小。

        Returns:
            ``[N, 512]`` float32（每行单位范数）。

        Raises:
            TextEncoderUnavailable: 模型不可用。
        """
        if not texts:
            return np.zeros((0, self._hidden_dim), dtype=np.float32)
        self._ensure_backbone()
        assert self._tokenizer is not None and self._backbone is not None

        outputs: list[np.ndarray] = []
        for start in range(0, len(texts), max(1, batch_size)):
            chunk = [str(item) for item in texts[start : start + batch_size]]
            batch = self._tokenizer(
                chunk,
                padding=True,
                truncation=True,
                max_length=self.max_len,
                return_tensors="pt",
            )
            batch = {key: value.to(self.device) for key, value in batch.items()}
            with torch.autocast(device_type=self.device.type, enabled=False):
                hidden = self._backbone(**batch).last_hidden_state  # [B,T,H]
            # bge 官方池化 = CLS（第 0 位 token），不是 mean pooling
            pooled = hidden[:, 0]
            pooled = torch.nn.functional.normalize(pooled.float(), p=2, dim=-1)
            outputs.append(pooled.cpu().numpy())
        return np.concatenate(outputs, axis=0).astype(np.float32)

    def project(self, vec: np.ndarray | torch.Tensor) -> np.ndarray:
        """把 512d 向量投影到 ``proj_dim``（推理路径，无梯度）。

        Args:
            vec: ``[N, 512]``（numpy 或 torch）。

        Returns:
            ``[N, proj_dim]`` float32 numpy。
        """
        if isinstance(vec, torch.Tensor):
            tensor = vec.detach().to(self.device).float()
        else:
            tensor = torch.as_tensor(np.asarray(vec, dtype=np.float32), device=self.device)
        if tensor.dim() == 1:
            tensor = tensor.unsqueeze(0)
        was_training = self.projector.training
        self.projector.eval()
        with torch.no_grad():
            projected = self.projector(tensor)
        if was_training:
            self.projector.train()
        return projected.float().cpu().numpy()

    def forward(self, texts: Sequence[str]) -> torch.Tensor:
        """端到端：文本 → 条件向量（投影层保留梯度，用于训练扩散模型）。

        Args:
            texts: 文本序列。

        Returns:
            ``[N, proj_dim]`` 张量（在 ``self.device`` 上）。
        """
        vectors = self.encode(texts)
        tensor = torch.from_numpy(vectors).to(self.device)
        return self.projector(tensor)

    def __call__(self, texts: Sequence[str]) -> torch.Tensor:
        """``encoder(texts)`` 等价于 :meth:`forward`（便于当模块用）。"""
        return self.forward(texts)

    # -- 设备/状态 -----------------------------------------------------------
    def to(self, device: torch.device | str) -> "TextEncoder":
        """切换设备（backbone 已加载时一并迁移）。"""
        self.device = torch.device(device)
        self.projector.to(self.device)
        if self._backbone is not None:
            self._backbone.to(self.device)
        return self

    def train(self, mode: bool = True) -> "TextEncoder":
        """训练/评估模式切换（backbone 永远保持 eval）。"""
        self.projector.train(mode)
        if self._backbone is not None:
            self._backbone.eval()
        return self

    def eval(self) -> "TextEncoder":
        """切到评估模式。"""
        return self.train(False)

    def parameters(self) -> Any:
        """可训练参数（仅投影层）。"""
        return self.projector.parameters()

    def state_dict(self) -> dict[str, Any]:
        """投影层权重（用于 ckpt）。"""
        return {"projector": self.projector.state_dict()}

    def load_state_dict(self, payload: dict[str, Any]) -> None:
        """加载投影层权重。"""
        state = payload.get("projector", payload)
        self.projector.load_state_dict(state)

    def describe(self) -> dict[str, Any]:
        """结构摘要。"""
        return {
            "model_name": self.model_name,
            "model_source": self.model_source,
            "hidden_dim": self._hidden_dim,
            "proj_dim": self.proj_dim,
            "frozen": self.frozen,
            "max_len": self.max_len,
            "device": str(self.device),
            "cache_dir": str(self.cache_dir),
            "projector_params": int(
                sum(p.numel() for p in self.projector.parameters() if p.requires_grad)
            ),
        }

    def __repr__(self) -> str:  # pragma: no cover - 调试用
        return (
            f"TextEncoder(model={self.model_name!r}, proj_dim={self.proj_dim}, "
            f"frozen={self.frozen}, device={self.device})"
        )


def cosine_similarity(a: np.ndarray, b: np.ndarray) -> float:
    """两个向量的余弦相似度（验收测试用）。

    Args:
        a: ``[D]`` 或 ``[1,D]``。
        b: 同上。

    Returns:
        余弦相似度（float）。
    """
    va = np.asarray(a, dtype=np.float64).reshape(-1)
    vb = np.asarray(b, dtype=np.float64).reshape(-1)
    if va.shape != vb.shape:
        raise ValueError(f"维度不一致: {va.shape} vs {vb.shape}")
    denominator = float(np.linalg.norm(va) * np.linalg.norm(vb))
    if denominator == 0.0:
        return 0.0
    return float(np.dot(va, vb) / denominator)
