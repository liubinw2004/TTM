"""配置系统：YAML 读取 → schema 校验 → dataclass 暴露 → 生效配置落盘。

字段表严格对齐 ``TTM-ARCH.md`` §5.8。设计要点：

* **强校验**：类型、枚举、区间、互斥、未知键（拼写错误）全部报错，
  错误信息带完整字段路径（如 ``data.val_ratio``），便于定位；
* **契约常量只读**：``motion.n_joints`` 必须等于 23、``motion.feature_dim`` 必须等于
  146，否则直接报错（防止有人改了 config 却忘了改模型）；
* **可复现**：任何脚本启动时调 :func:`dump_resolved` 把最终生效配置写进
  ``runs/<name>/config_resolved.yaml``。

本模块只依赖 ``pyyaml`` 与 :mod:`ttm.common.constants`，不导入 torch。
"""

from __future__ import annotations

import dataclasses
import os
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Callable, Iterable, Sequence

import yaml

from ttm import LAYOUT_VERSION
from ttm.common import constants as C
from ttm.common import paths as P

__all__ = [
    "ConfigError",
    "DataConfig",
    "DiffusionConfig",
    "MotionConfig",
    "ProjectConfig",
    "RetargetConfig",
    "RuntimeConfig",
    "ServeConfig",
    "TTMConfig",
    "TextConfig",
    "VAEConfig",
    "default_config",
    "dump_resolved",
    "load_config",
    "resolve_config_path",
    "to_dict",
]

#: 允许的数据档位。
VALID_TIERS: tuple[str, ...] = ("S", "M", "L")

#: 允许的数值精度。
VALID_PRECISIONS: tuple[str, ...] = ("fp32", "bf16", "fp16")

#: 允许的计算设备。
VALID_DEVICES: tuple[str, ...] = ("cuda", "cpu")

#: 允许的 Spine3 合并策略。
VALID_SPINE_MERGE: tuple[str, ...] = ("fold", "split")


class ConfigError(ValueError):
    """配置非法时抛出。

    Attributes:
        field: 出错的字段路径（如 ``"data.val_ratio"``）。
        message: 原始错误说明。
    """

    def __init__(self, field: str, message: str) -> None:
        self.field = field
        self.message = message
        super().__init__(f"配置错误 [{field}]: {message}")


# ---------------------------------------------------------------------------
# 校验辅助
# ---------------------------------------------------------------------------
_MISSING: Any = object()


class _Section:
    """带类型/枚举/区间校验的配置小节读取器。

    用法::

        sec = _Section("data", raw["data"], required=True)
        fps = sec.get("fps", 30, kind=int, min_value=1)
        ...
        sec.finish()      # 检查未知键（拼写错误）
    """

    def __init__(self, name: str, raw: Any, *, required: bool = False) -> None:
        self.name = name
        if raw is _MISSING or raw is None:
            if required:
                raise ConfigError(name, "缺少必需的小节")
            raw = {}
        if not isinstance(raw, dict):
            raise ConfigError(name, f"应为映射/dict，实际 {type(raw).__name__}")
        self.raw: dict[str, Any] = dict(raw)
        self.used: set[str] = set()

    # -- 取值 ---------------------------------------------------------------
    def get(
        self,
        key: str,
        default: Any = _MISSING,
        *,
        kind: type | tuple[type, ...] = object,
        choices: Sequence[Any] | None = None,
        min_value: float | None = None,
        max_value: float | None = None,
        allow_none: bool = False,
        validator: Callable[[Any], Any] | None = None,
    ) -> Any:
        """读取并校验一个字段。

        Args:
            key: 字段名。
            default: 缺省值；``_MISSING`` 表示必填。
            kind: 期望的 Python 类型（可传元组）。
            choices: 允许的取值集合。
            min_value: 数值下界（含）。
            max_value: 数值上界（含）。
            allow_none: 是否允许显式 ``null``。
            validator: 自定义转换/校验函数，返回规范化后的值。

        Returns:
            校验后的值。

        Raises:
            ConfigError: 字段缺失/类型不符/越界。
        """
        path = f"{self.name}.{key}"
        if key not in self.raw:
            if default is _MISSING:
                raise ConfigError(path, "缺少必需字段")
            value = default
        else:
            self.used.add(key)
            value = self.raw[key]

        if value is None:
            if allow_none:
                return None
            raise ConfigError(path, "不允许为 null")

        expected = kind if isinstance(kind, tuple) else (kind,)
        # bool 是 int 的子类，需要单独排除，避免 True 被当成 1
        if bool not in expected and isinstance(value, bool):
            raise ConfigError(path, f"期望 {self._type_names(expected)}，实际 bool")
        if not isinstance(value, expected):
            raise ConfigError(
                path, f"期望 {self._type_names(expected)}，实际 {type(value).__name__}（值 {value!r}）"
            )

        if isinstance(value, (int, float)) and not isinstance(value, bool):
            if min_value is not None and value < min_value:
                raise ConfigError(path, f"应 >= {min_value}，实际 {value}")
            if max_value is not None and value > max_value:
                raise ConfigError(path, f"应 <= {max_value}，实际 {value}")

        if choices is not None and value not in choices:
            raise ConfigError(path, f"应为 {list(choices)} 之一，实际 {value!r}")

        if validator is not None:
            try:
                value = validator(value)
            except ConfigError:
                raise
            except Exception as exc:  # noqa: BLE001
                raise ConfigError(path, f"校验失败: {exc}") from exc
        return value

    @staticmethod
    def _type_names(expected: Iterable[type]) -> str:
        """把类型元组渲染成可读文本。"""
        return "/".join(item.__name__ for item in expected)

    # -- 未知键 -------------------------------------------------------------
    def finish(self) -> None:
        """检查是否存在未消费的键（通常是拼写错误）。

        Raises:
            ConfigError: 存在未知键。
        """
        unknown = sorted(set(self.raw) - self.used)
        if unknown:
            raise ConfigError(
                self.name,
                f"存在未知字段 {unknown}；请对照 TTM-ARCH.md §5.8 的字段表",
            )


def _str_list(value: Any) -> list[str]:
    """校验「非空字符串列表」。"""
    if not isinstance(value, list):
        raise ConfigError("", f"应为字符串列表，实际 {type(value).__name__}")
    result: list[str] = []
    for item in value:
        if not isinstance(item, str) or not item.strip():
            raise ConfigError("", f"列表元素应为非空字符串，实际 {item!r}")
        result.append(item.strip())
    return result


# ---------------------------------------------------------------------------
# dataclass 定义（对应 §5.8 的六个小节 + 三个附加小节）
# ---------------------------------------------------------------------------
@dataclass(frozen=True, slots=True)
class ProjectConfig:
    """项目路径配置。所有路径都以 :attr:`root` 为基准。"""

    root: str = "D:/TTM"
    cache_dir: str = "D:/TTM/.cache"
    runs_dir: str = "runs"
    data_dir: str = "data"

    def runs_path(self) -> Path:
        """返回 ``runs/`` 的绝对路径。"""
        return Path(self.root) / self.runs_dir

    def data_path(self) -> Path:
        """返回 ``data/`` 的绝对路径。"""
        return Path(self.root) / self.data_dir


@dataclass(frozen=True, slots=True)
class DataConfig:
    """数据集 / 扫描 / 窗口配置。"""

    tier: str = "S"
    bvh_roots: tuple[str, ...] = ("Z:/1000hours/BVH_Body",)
    include_categories: tuple[str, ...] | None = None
    fps: int = 30
    window: int = 150
    overlap: int = 30
    min_tail_frames: int = 75
    actors_per_take: int = 1
    max_files: int = 3000
    #: 总窗口预算（**按窗口而不是按文件**截断）。旧的 ``max_files`` 按文件数截断，
    #: 但实测文件长度相差近 10 倍（1.12 的 clip 有 2850 帧、3.2 的只有几百帧），
    #: 等于配额完全失控：1.12 一个类吃掉 19193/22682 = 85% 的窗口。
    max_windows: int = 30000
    #: 单个细类的窗口硬上限（保险丝，防长套路吃光预算）。
    max_windows_per_category: int = 4000
    #: 单个细类的窗口占比软上限，超出即停止给该类继续发配额。
    category_window_share_cap: float = 0.2
    #: 每个细类至少凑到多少条 clip（素材不够的类别按实际值上报，不静默丢弃）。
    min_clips_per_leaf: int = 12
    #: 超长 clip（> ``long_clip_stride_frames`` 帧）的窗口抽样步长，避免长套路
    #: 吃光预算；``0`` 表示不降密度。
    long_clip_stride_frames: int = 450
    val_ratio: float = 0.1
    test_ratio: float = 0.1
    shard_size: int = 512
    max_failure_rate: float = 0.01
    #: 训练集是否启用「按标签频次 1/sqrt(freq) 加权」的均衡采样。
    #: 只影响训练集的取样本顺序，不改数据本身；验证/测试集恒为 False
    #: （重加权会让 metric 不可比）。
    class_balanced: bool = True
    #: 均衡采样的幂次：``1.0`` 完全均衡，``0.5`` 按 ``1/sqrt(freq)`` 折中，
    #: ``0.0`` 等权。实测类别窗口数从 2 到 19193 跨度极大（近 4 个数量级），
    #: 用 0.5 既抬长尾又不至于让只有 4 个窗口的类别被过采样到失真的程度。
    balance_power: float = 0.5
    #: 单个样本权重相对均值的最大倍数，防止极端长尾类被过采样到爆炸。
    balance_max_ratio: float = 60.0
    #: 训练期武术的目标**期望采样占比**（``0`` = 不调整）。
    #:
    #: 数据侧武术只占 6.1%（``4.wushu`` 220 个文件 / 191 类 / 每类 1.15 个文件，
    #: 是 Z 盘的物理上限，配额策略改不动），而「武术动作」是这个项目的核心诉求，
    #: 所以只能在训练期加权补。倍率不是拍出来的：由
    #: :func:`ttm.data.dataset.resolve_group_multiplier` 闭式解出
    #: ``m = target·Wo / ((1-target)·Wg)``，再用 ``expected_group_share`` 复核。
    balanced_target_wushu_ratio: float = 0.15
    #: 是否启用「少样本标签合并」（``W < threshold`` 的细标签并到基名）。
    #: 关掉会让 191 个武术细类各带 4 个样本直接进条件库——条件不可学。
    merge_rare_labels: bool = True
    #: 少样本合并的窗口数阈值（见 :data:`ttm.data.label_merge.MERGE_THRESHOLD`）。
    rare_label_min_windows: int = 30
    #: 重标签后处理的产物文件名（在 ``data/tier_<X>/`` 下）。
    #: 训练优先读它，不存在时退化为 ``labels.jsonl``（向后兼容旧产物）。
    labels_merged_file: str = "labels_merged.jsonl"


@dataclass(frozen=True, slots=True)
class MotionConfig:
    """运动特征契约（只读校验：必须与 :mod:`ttm.common.constants` 一致）。"""

    n_joints: int = 23
    feature_dim: int = 146
    contact_height_cm: float = 5.0
    contact_speed_cmps: float = 15.0


@dataclass(frozen=True, slots=True)
class VAEConfig:
    """MotionVAE 结构与训练超参。"""

    latent_dim: int = 128
    hidden: int = 512
    downsample: int = 4
    epochs: int = 200
    batch_size: int = 64
    lr: float = 3e-4
    #: KL 权重。注意 :func:`ttm.models.vae.kl_divergence` 是对
    #: ``[token_len × latent_dim] = 38×128 = 4864`` 维**求和**再取 batch 均值，
    #: 不是逐维平均——所以 ``1e-4`` 的实际等效权重 ≈ 0.49，比重建项（L1≈0.108）
    #: 还大，KL 会主导并把后验推到 ``(mu=0, logvar=0)`` 的驻点上（后验塌缩，
    #: 实测 mu_std=1e-4、重建帧间方差只有真值的 1/17、joint_mae 卡在 19.4°）。
    kl_weight: float = 1e-5
    #: ``logvar`` 的硬上下界。**KL 对 ``exp(logvar)`` 是线性的**，而它的梯度只有
    #: ``kl_weight=1e-5`` 量级，所以个别样本的 ``logvar`` 会一路漂到 +6（std≈400）：
    #: 单样本 KL 就有 ``0.5*(e^6-1-6)*4864 ≈ 7.2e5``，实测把 val loss 从 0.039
    #: 顶到 112，200 个 epoch 里出现 8 次尖峰，``best.pt`` 的选择被污染。
    #: 夹到 ``2.0`` 后单样本 logvar 侧 KL 上界降到 10674（×1e-5 = 0.107）。
    #: ``mu`` 那一侧**不夹**：mu 大是模型在用潜变量，不是数值问题。
    logvar_min: float = -10.0
    logvar_max: float = 2.0
    #: KL 权重线性 warmup 的 epoch 数（0 = 直接满权重）。
    #: 不加 warmup 时实测会**后验塌缩**：mu 标准差掉到 1e-4，解码器退化成输出
    #: 一个常驻姿态（帧间方差只有真实数据的 1/17），joint_mae 卡在 19°。
    #: 原因是死锁——解码器一旦不依赖 z，mu 就只剩 KL 一项梯度，而 KL 只把 mu 往 0 推。
    kl_warmup_epochs: int = 20
    #: 保留最近多少个 epoch 快照（``ckpt_XXXX.pt``）。
    #:
    #: 500 epoch × 120MB = **60GB** 写盘，而 ``D:`` 已用到 93%——这不是洁癖问题，
    #: 是「写满盘会让训练在中途崩掉、且 best.pt 也跟着写坏」的实际风险。
    #: ``best.pt`` 永远单独保留，不受本项影响。
    save_keep_last: int = 3


@dataclass(frozen=True, slots=True)
class DiffusionConfig:
    """潜空间扩散 Transformer 结构与训练超参。"""

    d_model: int = 256
    n_layers: int = 8
    n_heads: int = 8
    timesteps: int = 1000
    ddim_steps: int = 50
    epochs: int = 100
    batch_size: int = 64
    lr: float = 1e-4
    cfg_dropout: float = 0.1


@dataclass(frozen=True, slots=True)
class TextConfig:
    """中文文本编码器配置。"""

    encoder: str = "BAAI/bge-small-zh-v1.5"
    #: 项目内离线权重目录（禁止联网；由 ``TextEncoder`` 以 local_files_only 加载）。
    model_path: str = "D:/TTM/models/bge-small-zh-v1.5"
    frozen: bool = True
    max_len: int = 32
    proj_dim: int = 256


@dataclass(frozen=True, slots=True)
class RuntimeConfig:
    """运行期（推理/训练共用）配置。"""

    seed: int = 42
    precision: str = "bf16"
    device: str = "cuda"
    guidance: float = 3.0
    smooth: float = 0.2
    #: 启动瞬态裁头帧数（F-17）。生成时多申请同样帧数、导出前丢掉开头，
    #: 用户侧时长不变；ground_origin 在裁头后的首帧重新计算。``0`` = 关闭。
    trim_head_frames: int = 12
    vae_ckpt: str = ""
    diffusion_ckpt: str = ""


@dataclass(frozen=True, slots=True)
class ServeConfig:
    """FastAPI 服务配置。"""

    host: str = "127.0.0.1"
    port: int = 8765
    max_duration_s: float = 12.0


@dataclass(frozen=True, slots=True)
class RetargetConfig:
    """重定向配置（Spine3 降级策略开关）。"""

    spine_merge: str = "fold"


@dataclass(frozen=True, slots=True)
class TTMConfig:
    """TTM 的完整生效配置。"""

    project: ProjectConfig = field(default_factory=ProjectConfig)
    data: DataConfig = field(default_factory=DataConfig)
    motion: MotionConfig = field(default_factory=MotionConfig)
    vae: VAEConfig = field(default_factory=VAEConfig)
    diffusion: DiffusionConfig = field(default_factory=DiffusionConfig)
    text: TextConfig = field(default_factory=TextConfig)
    runtime: RuntimeConfig = field(default_factory=RuntimeConfig)
    serve: ServeConfig = field(default_factory=ServeConfig)
    retarget: RetargetConfig = field(default_factory=RetargetConfig)
    #: 配置来源文件路径（用于日志与 ``config_resolved.yaml`` 的可追溯性）。
    source_path: str = ""

    # -- 派生路径 -----------------------------------------------------------
    def tier_dir(self) -> Path:
        """返回本档位的数据产物目录 ``<data_dir>/tier_{tier}``。

        注意：``data_dir`` 来自配置（可被 smoke/unit-test 覆盖），
        因此这里必须显式传入基目录，不能依赖 ``paths.data_dir()`` 的默认值。
        """
        return P.tier_data_dir(self.data.tier, base_data_dir=self.project.data_path())

    def tier_name(self) -> str:
        """返回档位名 ``tier_{tier}``（大写，与目录名一致）。"""
        return P.tier_filename(self.data.tier)

    def index_path(self) -> Path:
        """返回 ``index.jsonl`` 路径。"""
        return self.tier_dir() / "index.jsonl"

    def labels_path(self) -> Path:
        """返回原始 ``labels.jsonl`` 路径（窗口级统计口径，**合并前**的细标签）。"""
        return self.tier_dir() / "labels.jsonl"

    def train_labels_path(self) -> Path:
        """返回训练实际读取的标签文件路径。

        优先 ``labels_merged.jsonl``（少样本合并后的条件库），不存在时退回
        ``labels.jsonl``。这样：
        * 旧产物（没跑过 :file:`scripts/02b_merge_labels.py`）照常能训；
        * 新产物用合并后的标签训，而 ``labels.jsonl`` 仍是「合并前细类数」的
          统计口径，两个数字可以同时报出来。
        """
        merged = self.tier_dir() / self.data.labels_merged_file
        return merged if merged.is_file() else self.labels_path()

    def failed_path(self) -> Path:
        """返回 ``failed.jsonl`` 路径。"""
        return self.tier_dir() / "failed.jsonl"

    def shards_path(self) -> Path:
        """返回 ``shards/`` 目录路径。"""
        return self.tier_dir() / "shards"

    def norm_stats_path(self) -> Path:
        """返回 ``norm_stats.npz`` 路径。"""
        return self.tier_dir() / "norm_stats.npz"

    def run_dir(self, name: str) -> Path:
        """返回某次运行（``vae`` / ``diffusion`` / 自定义）的产物目录。

        产物**按档位隔离**：``<runs_dir>/tier_<tier 小写>/<name>``，例如
        ``runs/tier_s/vae``。这样用 ``configs/smoke.yaml`` 冒烟时不会覆盖正式档权重
        （早前 ``runs/vae`` 的写法会在两个档位之间互相踩）。
        """
        tier = str(self.data.tier or "").strip().lower()
        base = self.project.runs_path()
        if not tier:
            return base / name
        return base / f"tier_{tier}" / name

    def summary(self) -> str:
        """返回单行配置摘要（日志用）。"""
        return (
            f"tier={self.data.tier} roots={list(self.data.bvh_roots)} "
            f"fps={self.data.fps} window={self.data.window} "
            f"feat={self.motion.feature_dim} latent={self.vae.latent_dim} "
            f"d_model={self.diffusion.d_model}x{self.diffusion.n_layers} "
            f"device={self.runtime.device}/{self.runtime.precision} seed={self.runtime.seed}"
        )


def default_config() -> TTMConfig:
    """返回全部使用默认值（即 ``configs/default.yaml`` 语义）的配置对象。"""
    return TTMConfig()


# ---------------------------------------------------------------------------
# 解析
# ---------------------------------------------------------------------------
def resolve_config_path(path: str | os.PathLike[str]) -> Path:
    """把配置路径解析为绝对路径。

    相对路径优先按当前工作目录解析；不存在时再按项目根解析
    （兼容从任意目录调用脚本）。

    Args:
        path: 配置路径（如 ``configs/tier_s.yaml``）。

    Returns:
        存在的配置文件绝对路径。

    Raises:
        ConfigError: 文件不存在。
    """
    raw = Path(os.path.expanduser(str(path)))
    candidates: list[Path] = []
    if raw.is_absolute():
        candidates.append(raw)
    else:
        candidates.append(Path.cwd() / raw)
        candidates.append(Path(P.project_root()) / raw)
    for candidate in candidates:
        if candidate.is_file():
            return candidate
    tried = ", ".join(str(item) for item in candidates)
    raise ConfigError("config", f"配置文件不存在（已尝试: {tried}）")


def _load_yaml(path: Path) -> dict[str, Any]:
    """读取 YAML 文件为 dict。"""
    try:
        text = path.read_text(encoding="utf-8")
    except OSError as exc:
        raise ConfigError("config", f"无法读取 {path}: {exc}") from exc
    try:
        raw = yaml.safe_load(text)
    except yaml.YAMLError as exc:
        raise ConfigError("config", f"YAML 解析失败 {path}: {exc}") from exc
    if raw is None:
        raw = {}
    if not isinstance(raw, dict):
        raise ConfigError("config", f"顶层应为映射/dict，实际 {type(raw).__name__}")
    return raw


def _parse_project(raw: dict[str, Any]) -> ProjectConfig:
    """解析 ``project`` 小节。"""
    sec = _Section("project", raw.get("project", _MISSING))
    result = ProjectConfig(
        root=sec.get("root", P.DEFAULT_PROJECT_ROOT, kind=str),
        cache_dir=sec.get("cache_dir", "D:/TTM/.cache", kind=str),
        runs_dir=sec.get("runs_dir", "runs", kind=str),
        data_dir=sec.get("data_dir", "data", kind=str),
    )
    sec.finish()
    # 白名单根必须与写盘守卫一致，否则后续所有写盘都会炸；这里提前拦截
    if Path(result.root).as_posix().rstrip("/").lower() != Path(P.DEFAULT_PROJECT_ROOT).as_posix().lower():
        raise ConfigError(
            "project.root",
            f"必须为 {P.DEFAULT_PROJECT_ROOT}（写盘白名单根，PRD §8 硬约束），实际 {result.root!r}",
        )
    return result


def _parse_data(raw: dict[str, Any]) -> DataConfig:
    """解析 ``data`` 小节并做交叉校验。"""
    sec = _Section("data", raw.get("data", _MISSING), required=True)
    tier = sec.get("tier", "S", kind=str, validator=lambda v: v.strip().upper())
    if tier not in VALID_TIERS:
        raise ConfigError("data.tier", f"应为 {list(VALID_TIERS)} 之一，实际 {tier!r}")
    bvh_roots = sec.get("bvh_roots", ["Z:/1000hours/BVH_Body"], kind=list, validator=_str_list)
    include = sec.get("include_categories", None, kind=list, allow_none=True, validator=_str_list)
    fps = sec.get("fps", 30, kind=int, min_value=1, max_value=240)
    window = sec.get("window", 150, kind=int, min_value=1, max_value=6000)
    overlap = sec.get("overlap", 30, kind=int, min_value=0, max_value=6000)
    min_tail = sec.get("min_tail_frames", 75, kind=int, min_value=1, max_value=6000)
    actors = sec.get("actors_per_take", 1, kind=int, min_value=1, max_value=16)
    max_files = sec.get("max_files", 3000, kind=int, min_value=1)
    max_windows = sec.get("max_windows", 30000, kind=int, min_value=1)
    max_windows_per_category = sec.get("max_windows_per_category", 4000, kind=int, min_value=1)
    share_cap = sec.get("category_window_share_cap", 0.2, kind=(int, float), min_value=0.0)
    min_clips_per_leaf = sec.get("min_clips_per_leaf", 12, kind=int, min_value=1)
    long_clip_stride = sec.get("long_clip_stride_frames", 450, kind=int, min_value=0)

    # val_ratio / test_ratio 支持整数百分比写法（10 表示 10%）
    def _ratio(value: Any) -> float:
        ratio = float(value)
        if ratio > 1.0:
            if ratio > 100.0:
                raise ConfigError("", f"比例应 <= 1.0 或百分比 <= 100，实际 {ratio}")
            ratio = ratio / 100.0
        if not 0.0 <= ratio < 1.0:
            raise ConfigError("", f"比例应在 [0,1) 内，实际 {ratio}")
        return ratio

    val_ratio = sec.get("val_ratio", 0.1, kind=(int, float), validator=_ratio)
    test_ratio = sec.get("test_ratio", 0.1, kind=(int, float), validator=_ratio)
    shard_size = sec.get("shard_size", 512, kind=int, min_value=1)
    max_failure_rate = sec.get("max_failure_rate", 0.01, kind=(int, float), min_value=0.0)
    class_balanced = sec.get("class_balanced", True, kind=bool)
    balance_power = sec.get("balance_power", 0.5, kind=(int, float), min_value=0.0)
    balance_max_ratio = sec.get("balance_max_ratio", 60.0, kind=(int, float), min_value=1.0)
    target_wushu = sec.get(
        "balanced_target_wushu_ratio", 0.15, kind=(int, float), min_value=0.0, max_value=1.0
    )
    merge_rare = sec.get("merge_rare_labels", True, kind=bool)
    rare_min_windows = sec.get("rare_label_min_windows", 30, kind=int, min_value=1)
    labels_merged_file = sec.get("labels_merged_file", "labels_merged.jsonl", kind=str)
    sec.finish()

    if overlap >= window:
        raise ConfigError("data.overlap", f"应小于 window={window}，实际 {overlap}")
    if min_tail > window:
        raise ConfigError("data.min_tail_frames", f"应 <= window={window}，实际 {min_tail}")
    if val_ratio + test_ratio >= 1.0:
        raise ConfigError(
            "data.val_ratio", f"val_ratio+test_ratio 应 < 1.0，实际 {val_ratio + test_ratio}"
        )
    if not bvh_roots:
        raise ConfigError("data.bvh_roots", "至少需要一个 BVH 根目录")

    return DataConfig(
        tier=tier,
        bvh_roots=tuple(bvh_roots),
        include_categories=tuple(include) if include else None,
        fps=fps,
        window=window,
        overlap=overlap,
        min_tail_frames=min_tail,
        actors_per_take=actors,
        max_files=max_files,
        max_windows=max_windows,
        max_windows_per_category=max_windows_per_category,
        category_window_share_cap=float(share_cap),
        min_clips_per_leaf=min_clips_per_leaf,
        long_clip_stride_frames=long_clip_stride,
        val_ratio=val_ratio,
        test_ratio=test_ratio,
        shard_size=shard_size,
        max_failure_rate=float(max_failure_rate),
        class_balanced=bool(class_balanced),
        balance_power=float(balance_power),
        balance_max_ratio=float(balance_max_ratio),
        balanced_target_wushu_ratio=float(target_wushu),
        merge_rare_labels=bool(merge_rare),
        rare_label_min_windows=int(rare_min_windows),
        labels_merged_file=str(labels_merged_file),
    )


def _parse_motion(raw: dict[str, Any]) -> MotionConfig:
    """解析 ``motion`` 小节，并强校验契约常量。"""
    sec = _Section("motion", raw.get("motion", _MISSING))
    n_joints = sec.get("n_joints", C.N_JOINTS, kind=int, min_value=1)
    feature_dim = sec.get("feature_dim", C.FEATURE_DIM, kind=int, min_value=1)
    contact_height = sec.get("contact_height_cm", C.CONTACT_HEIGHT_CM, kind=(int, float), min_value=0.0)
    contact_speed = sec.get(
        "contact_speed_cmps", C.CONTACT_SPEED_CMPS, kind=(int, float), min_value=0.0
    )
    sec.finish()

    if n_joints != C.N_JOINTS:
        raise ConfigError(
            "motion.n_joints", f"契约常量固定为 {C.N_JOINTS}，实际 {n_joints}（改这里必须同步改骨架代码）"
        )
    if feature_dim != C.FEATURE_DIM:
        raise ConfigError(
            "motion.feature_dim",
            f"契约常量固定为 {C.FEATURE_DIM}，实际 {feature_dim}（改布局必须升 layout_version）",
        )
    return MotionConfig(
        n_joints=n_joints,
        feature_dim=feature_dim,
        contact_height_cm=float(contact_height),
        contact_speed_cmps=float(contact_speed),
    )


def _parse_vae(raw: dict[str, Any]) -> VAEConfig:
    """解析 ``vae`` 小节。"""
    sec = _Section("vae", raw.get("vae", _MISSING))
    downsample = sec.get("downsample", C.DOWNSAMPLE, kind=int, min_value=1, max_value=16)
    result = VAEConfig(
        latent_dim=sec.get("latent_dim", C.LATENT_DIM, kind=int, min_value=8, max_value=1024),
        hidden=sec.get("hidden", 512, kind=int, min_value=32, max_value=4096),
        downsample=downsample,
        epochs=sec.get("epochs", 200, kind=int, min_value=1),
        batch_size=sec.get("batch_size", 64, kind=int, min_value=1),
        lr=sec.get("lr", 2e-4, kind=(int, float), min_value=0.0),
        kl_weight=sec.get("kl_weight", 1e-4, kind=(int, float), min_value=0.0),
        logvar_min=sec.get("logvar_min", -10.0, kind=(int, float)),
        logvar_max=sec.get("logvar_max", 2.0, kind=(int, float)),
        kl_warmup_epochs=sec.get("kl_warmup_epochs", 30, kind=int, min_value=0),
        save_keep_last=sec.get("save_keep_last", 3, kind=int, min_value=1),
    )
    sec.finish()
    if result.logvar_min >= result.logvar_max:
        raise ConfigError(
            "vae.logvar_min", f"应小于 logvar_max={result.logvar_max}，实际 {result.logvar_min}"
        )
    if C.token_len_for(C.WINDOW, result.downsample) < 1:
        raise ConfigError(
            "vae.downsample",
            f"window={C.WINDOW} 与 downsample={result.downsample} 组合得到非法 token 数",
        )
    return result


def _parse_diffusion(raw: dict[str, Any]) -> DiffusionConfig:
    """解析 ``diffusion`` 小节。"""
    sec = _Section("diffusion", raw.get("diffusion", _MISSING))
    result = DiffusionConfig(
        d_model=sec.get("d_model", 256, kind=int, min_value=32, max_value=2048),
        n_layers=sec.get("n_layers", 8, kind=int, min_value=1, max_value=64),
        n_heads=sec.get("n_heads", 8, kind=int, min_value=1, max_value=64),
        timesteps=sec.get("timesteps", 1000, kind=int, min_value=10, max_value=10000),
        ddim_steps=sec.get("ddim_steps", 50, kind=int, min_value=1, max_value=1000),
        epochs=sec.get("epochs", 100, kind=int, min_value=1),
        batch_size=sec.get("batch_size", 64, kind=int, min_value=1),
        lr=sec.get("lr", 1e-4, kind=(int, float), min_value=0.0),
        cfg_dropout=sec.get("cfg_dropout", 0.1, kind=(int, float), min_value=0.0, max_value=1.0),
    )
    sec.finish()
    if result.d_model % result.n_heads != 0:
        raise ConfigError(
            "diffusion.d_model",
            f"d_model({result.d_model}) 必须能被 n_heads({result.n_heads}) 整除",
        )
    if result.ddim_steps > result.timesteps:
        raise ConfigError(
            "diffusion.ddim_steps",
            f"DDIM 步数应 <= timesteps({result.timesteps})，实际 {result.ddim_steps}",
        )
    return result


def _parse_text(raw: dict[str, Any]) -> TextConfig:
    """解析 ``text`` 小节。"""
    sec = _Section("text", raw.get("text", _MISSING))
    result = TextConfig(
        encoder=sec.get("encoder", "BAAI/bge-small-zh-v1.5", kind=str),
        model_path=sec.get("model_path", "D:/TTM/models/bge-small-zh-v1.5", kind=str),
        frozen=sec.get("frozen", True, kind=bool),
        max_len=sec.get("max_len", 32, kind=int, min_value=1, max_value=512),
        proj_dim=sec.get("proj_dim", 256, kind=int, min_value=8, max_value=2048),
    )
    sec.finish()
    if not result.encoder.strip():
        raise ConfigError("text.encoder", "不能为空")
    return result


def _parse_runtime(raw: dict[str, Any]) -> RuntimeConfig:
    """解析 ``runtime`` 小节。"""
    sec = _Section("runtime", raw.get("runtime", _MISSING))
    precision = sec.get("precision", "bf16", kind=str, validator=lambda v: v.strip().lower())
    if precision not in VALID_PRECISIONS:
        raise ConfigError("runtime.precision", f"应为 {list(VALID_PRECISIONS)} 之一，实际 {precision!r}")
    device = sec.get("device", "cuda", kind=str, validator=lambda v: v.strip().lower())
    if device not in VALID_DEVICES:
        raise ConfigError("runtime.device", f"应为 {list(VALID_DEVICES)} 之一，实际 {device!r}")
    result = RuntimeConfig(
        seed=sec.get("seed", 42, kind=int, min_value=0),
        precision=precision,
        device=device,
        guidance=sec.get("guidance", 3.0, kind=(int, float), min_value=0.0, max_value=50.0),
        smooth=sec.get("smooth", 0.2, kind=(int, float), min_value=0.0, max_value=1.0),
        vae_ckpt=sec.get("vae_ckpt", "", kind=str),
        diffusion_ckpt=sec.get("diffusion_ckpt", "", kind=str),
        # F-17：启动瞬态裁头帧数（0 = 关闭）。上限 150（一个窗口）——超过就没有
        # 可用帧了，属于配置错误而不是可调项。
        trim_head_frames=sec.get(
            "trim_head_frames", 12, kind=int, min_value=0, max_value=150
        ),
    )
    sec.finish()
    return result


def _parse_serve(raw: dict[str, Any]) -> ServeConfig:
    """解析 ``serve`` 小节。"""
    sec = _Section("serve", raw.get("serve", _MISSING))
    result = ServeConfig(
        host=sec.get("host", "127.0.0.1", kind=str),
        port=sec.get("port", 8765, kind=int, min_value=1, max_value=65535),
        max_duration_s=sec.get("max_duration_s", 12.0, kind=(int, float), min_value=1.0, max_value=120.0),
    )
    sec.finish()
    return result


def _parse_retarget(raw: dict[str, Any]) -> RetargetConfig:
    """解析 ``retarget`` 小节。"""
    sec = _Section("retarget", raw.get("retarget", _MISSING))
    spine_merge = sec.get(
        "spine_merge", "fold", kind=str, choices=VALID_SPINE_MERGE, validator=lambda v: v.strip().lower()
    )
    sec.finish()
    return RetargetConfig(spine_merge=spine_merge)


#: 已知的顶层小节名（用于拦截拼写错误的顶层键）。
KNOWN_SECTIONS: tuple[str, ...] = (
    "project",
    "data",
    "motion",
    "vae",
    "diffusion",
    "text",
    "runtime",
    "serve",
    "retarget",
)


def load_config(path: str | os.PathLike[str] | None = None) -> TTMConfig:
    """加载并校验配置文件。

    Args:
        path: 配置文件路径；``None`` 时使用 ``configs/default.yaml``。

    Returns:
        校验通过的 :class:`TTMConfig`。

    Raises:
        ConfigError: 文件缺失、YAML 语法错误、字段类型/枚举/区间/未知键不合法。
    """
    target = resolve_config_path("configs/default.yaml" if path is None else path)
    raw = _load_yaml(target)

    unknown = sorted(set(raw) - set(KNOWN_SECTIONS))
    if unknown:
        raise ConfigError(
            "config", f"存在未知顶层小节 {unknown}；已知小节: {list(KNOWN_SECTIONS)}"
        )

    config = TTMConfig(
        project=_parse_project(raw),
        data=_parse_data(raw),
        motion=_parse_motion(raw),
        vae=_parse_vae(raw),
        diffusion=_parse_diffusion(raw),
        text=_parse_text(raw),
        runtime=_parse_runtime(raw),
        serve=_parse_serve(raw),
        retarget=_parse_retarget(raw),
        source_path=str(target),
    )
    return config


# ---------------------------------------------------------------------------
# 序列化 / 落盘
# ---------------------------------------------------------------------------
def to_dict(cfg: TTMConfig) -> dict[str, Any]:
    """把配置对象转成可直接 ``yaml.safe_dump`` 的 dict（tuple → list）。"""

    def _convert(value: Any) -> Any:
        """递归转换 dataclass / tuple / Path。"""
        if dataclasses.is_dataclass(value) and not isinstance(value, type):
            return {f.name: _convert(getattr(value, f.name)) for f in dataclasses.fields(value)}
        if isinstance(value, tuple):
            return [_convert(item) for item in value]
        if isinstance(value, list):
            return [_convert(item) for item in value]
        if isinstance(value, Path):
            return value.as_posix()
        return value

    return _convert(cfg)


def dump_resolved(cfg: TTMConfig, run_name: str) -> Path:
    """把最终生效配置写进 ``runs/<run_name>/config_resolved.yaml``。

    经过写盘守卫（:func:`ttm.common.paths.resolve_out`）。

    Args:
        cfg: 生效配置。
        run_name: 运行名（如 ``"vae"`` / ``"diffusion"``）。

    Returns:
        实际写出的文件路径。
    """
    target = P.resolve_out(cfg.run_dir(run_name) / "config_resolved.yaml", mkparents=True)
    payload = to_dict(cfg)
    payload["_meta"] = {
        "source_path": cfg.source_path,
        "layout_version": LAYOUT_VERSION,
        "n_joints": C.N_JOINTS,
        "feature_dim": C.FEATURE_DIM,
        "vae_token_len": C.VAE_TOKEN_LEN,
        "window": C.WINDOW,
        "fps": C.FPS,
    }
    text = yaml.safe_dump(payload, allow_unicode=True, sort_keys=False, width=120)
    target.write_text(text, encoding="utf-8")
    return target
