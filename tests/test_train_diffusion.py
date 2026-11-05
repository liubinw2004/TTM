"""T07 单测：扩散训练全链路（冻结 VAE + 条件库 + CFG dropout + ckpt 往返）。

这是 T07 唯一的自动化护栏——训练要跑十几分钟到一小时，绝不能靠肉眼判断是否
真的在学。用例覆盖：

1. ``frames_to_token_mask`` 的几何（150 帧 → 38 token，尾窗 padding 不参与 loss）；
2. ``TextConditionBank`` 的建库/取词/缺词兜底；
3. ``train_diffusion`` 端到端：loss 下降、``best.pt`` 落盘、ckpt 存的是**真实** VAE
   路径（不是空字符串）；
4. CFG：``cfg_dropout`` 必须真的丢条件（``n_dropped > 0``），否则推理期
   ``guidance > 1`` 无效；
5. ``load_vae_for_diffusion`` 的冻结语义（eval + requires_grad=False）。
"""

from __future__ import annotations

import json
import shutil
from pathlib import Path

import numpy as np
import pytest

torch = pytest.importorskip("torch", reason="SKIP_IF_NO_TORCH: 需要 torch")

from conftest import build_bvh_text, write_bvh, write_config  # noqa: E402

from ttm.common import constants as C  # noqa: E402
from ttm.common import paths as P  # noqa: E402
from ttm.common.config import load_config  # noqa: E402
from ttm.data.preprocess import PreprocessPipeline  # noqa: E402
from ttm.data.scan import Scanner  # noqa: E402
from ttm.models.diffusion import DiffusionTransformer, GaussianDiffusion  # noqa: E402
from ttm.train.train_diffusion import (  # noqa: E402
    TextConditionBank,
    frames_to_token_mask,
    load_vae_for_diffusion,
    resolve_vae_checkpoint,
    train_diffusion,
)
from ttm.train.train_vae import train_vae  # noqa: E402

pytestmark = pytest.mark.requires_torch

#: 本模块独占的产物目录（与其它测试模块隔离，避免 ckpt 互相覆盖）。
_DATA_DIR = "data/unit_test_t07"
_RUNS_DIR = "runs/unit_test_t07"


# ---------------------------------------------------------------------------
# 合成数据集 + 先训一个小 VAE（扩散训练需要可用的 VAE ckpt）
# ---------------------------------------------------------------------------
def _build_tree(root: Path) -> None:
    """6 个长 clip（各 2401 帧）→ 每 epoch 有多个 step，loss 才看得出趋势。

    单样本/单 step 的扩散 MSE 噪声极大（实测 1.3224 → 1.3270 完全可能是噪声），
    所以这里刻意把数据量做够：batch=4 时每 epoch 有 4 个 step。
    """
    base = root / "1.7.Daily_Life_Behavior" / "1.7.3.Walking_Movement_With_Posture"
    plans = [
        ("1.7.3.1.Basic_Gait", "20251211", "Take_001", "actor"),
        ("1.7.3.1.Basic_Gait", "20251212", "Take_002", "actor"),
        ("1.7.3.1.Basic_Gait", "20251213", "Take_003", "actor"),
        ("1.7.3.3.Stairs_Up_Down", "20260326", "Take_115", "stair"),
        ("1.7.3.3.Stairs_Up_Down", "20260327", "Take_116", "stair"),
        ("1.7.3.3.Stairs_Up_Down", "20260328", "Take_117", "stair"),
    ]
    for category, date, take, suffix in plans:
        write_bvh(
            base / category / date / take / f"{take}_{suffix}_0.bvh",
            build_bvh_text(n_frames=2401),
        )


@pytest.fixture(scope="module")
def prepared(tmp_path_factory: pytest.TempPathFactory) -> dict:
    """构建分片 + labels，并训出一个可加载的小 VAE。"""
    work_dir = P.resolve_out(
        P.cache_dir() / "tmp" / "pytest" / f"t07_{abs(hash(str(tmp_path_factory)))}",
        mkparents=True,
    )
    shutil.rmtree(work_dir, ignore_errors=True)
    work_dir.mkdir(parents=True, exist_ok=True)
    data_root = work_dir / "BVH_Body"
    _build_tree(data_root)

    config_path = write_config(
        work_dir / "cfg.yaml",
        bvh_roots=[data_root.as_posix()],
        data_dir=_DATA_DIR,
        runs_dir=_RUNS_DIR,
        max_files=50,
        shard_size=4,
        val_ratio=0.0,   # 合成数据太少，全部进 train，val 退化用 train loss
        test_ratio=0.0,
    )
    cfg = load_config(config_path)
    scan = Scanner(cfg, workers=1).run()
    Scanner(cfg, workers=1).write(scan)
    PreprocessPipeline(cfg, workers=2).run()

    # 训一个小 VAE：扩散训练只把它当冻结编码器，2 个 epoch 足够
    vae_result = train_vae(cfg, max_steps_per_epoch=2, max_epochs=2)
    assert Path(vae_result.checkpoint).is_file()

    yield {"cfg": cfg, "work_dir": work_dir, "vae": vae_result}

    shutil.rmtree(P.project_root() / _DATA_DIR, ignore_errors=True)
    shutil.rmtree(P.project_root() / _RUNS_DIR, ignore_errors=True)
    shutil.rmtree(work_dir, ignore_errors=True)


# ---------------------------------------------------------------------------
# 1. token 掩码几何
# ---------------------------------------------------------------------------
def test_token_mask_geometry() -> None:
    """150 帧（全有效）→ 38 token 全 1；尾窗 padding 映射成 0。"""
    full = torch.ones(2, C.WINDOW)
    mask = frames_to_token_mask(full)
    assert mask.shape == (2, C.VAE_TOKEN_LEN) == (2, 38)
    assert torch.all(mask == 1.0)

    # 只有前 100 帧有效 → ceil(100/4) = 25 个 token 有效
    partial = torch.zeros(1, C.WINDOW)
    partial[:, :100] = 1.0
    partial_mask = frames_to_token_mask(partial)
    assert int(partial_mask.sum().item()) == 25
    assert torch.all(partial_mask[0, :25] == 1.0)
    assert torch.all(partial_mask[0, 25:] == 0.0)

    # 感受野里只要有一帧有效，该 token 就算有效
    sparse = torch.zeros(1, 16)
    sparse[0, 3] = 1.0
    assert int(frames_to_token_mask(sparse).sum().item()) == 1


def test_token_mask_rejects_bad_shape() -> None:
    """非 [B,T] 输入报错。"""
    with pytest.raises(ValueError):
        frames_to_token_mask(torch.zeros(150))


# ---------------------------------------------------------------------------
# 2. 条件库
# ---------------------------------------------------------------------------
class _StubEncoder:
    """固定维度的假编码器（避免测试里反复加载 bge）。"""

    def __init__(self, dim: int = 16) -> None:
        self.dim = dim
        self.seen: list[str] = []

    def encode(self, texts, *, batch_size: int = 16):  # noqa: ANN001
        vectors = []
        for text in texts:
            self.seen.append(text)
            # 用字符和做确定性投影，保证同一文本恒得同一向量
            base = sum(ord(ch) for ch in text) % 97
            vectors.append(np.full(self.dim, base / 97.0, dtype=np.float32))
        return np.stack(vectors)


def test_condition_bank_build_and_gather() -> None:
    """建库去重、候选列表保留同义句、缺词时置零兜底。"""
    labels = {
        "clipA_win00": {"zh_label": "走上楼梯", "zh_texts": ["上楼梯", "爬楼梯", "登楼"]},
        "clipB_win00": {"zh_label": "打拳", "zh_texts": ["打拳"]},
    }
    bank = TextConditionBank(_StubEncoder(16), device=torch.device("cpu"))  # type: ignore[arg-type]
    bank.build(labels)
    # 去重后：走上楼梯 / 上楼梯 / 爬楼梯 / 登楼 / 打拳 = 5 条
    # （clipB 的 label "打拳" 与其 zh_texts[0] 相同，只在同一条 clip 内去重）
    assert bank.size == 5
    assert bank.dim == 16

    generator = torch.Generator().manual_seed(0)
    vectors = bank.gather(["clipA_win00", "clipB_win00"], generator=generator)
    assert vectors.shape == (2, 16)
    assert torch.isfinite(vectors).all()
    # clipB 只有一个候选 → 多次取样结果必须一致
    again = bank.gather(["clipB_win00"], generator=generator)
    assert torch.allclose(again[0], vectors[1])

    # 未知 clip_id → 全零（等价 null condition）
    missing = bank.gather(["nope_win99"], generator=generator)
    assert torch.all(missing == 0.0)


def test_condition_bank_null_vector() -> None:
    """``null()`` 返回指定形状的全零向量。"""
    bank = TextConditionBank(_StubEncoder(8), device=torch.device("cpu"))  # type: ignore[arg-type]
    null = bank.null(32, 3)
    assert null.shape == (3, 32)
    assert torch.all(null == 0.0)


def test_condition_bank_requires_build() -> None:
    """未 build 就 gather 报错。"""
    bank = TextConditionBank(_StubEncoder(8), device=torch.device("cpu"))  # type: ignore[arg-type]
    with pytest.raises(RuntimeError):
        bank.gather(["a"], generator=None)


# ---------------------------------------------------------------------------
# 3. CFG dropout（能否真的丢条件）
# ---------------------------------------------------------------------------
def test_cfg_dropout_actually_drops() -> None:
    """``cfg_dropout=1.0`` 时整批条件都被置空，``n_dropped == B``。"""
    torch.manual_seed(0)
    model = DiffusionTransformer(
        latent_dim=C.LATENT_DIM, d_model=32, n_layers=1, n_heads=4, cond_dim=16
    )
    diffusion = GaussianDiffusion(timesteps=50)
    latents = torch.randn(4, C.VAE_TOKEN_LEN, C.LATENT_DIM)
    cond = torch.randn(4, 16)

    always = diffusion.p_losses(
        model, latents, cond, cfg_dropout=1.0, generator=torch.Generator().manual_seed(0)
    )
    assert int(always.n_dropped) == 4

    never = diffusion.p_losses(
        model, latents, cond, cfg_dropout=0.0, generator=torch.Generator().manual_seed(0)
    )
    assert int(never.n_dropped) == 0
    assert torch.isfinite(never.loss)


def test_cfg_guidance_changes_output() -> None:
    """条件被 dropout 训练过，guidance 才可能改变采样结果（这里验证接口与形状）。"""
    torch.manual_seed(0)
    model = DiffusionTransformer(
        latent_dim=C.LATENT_DIM, d_model=32, n_layers=1, n_heads=4, cond_dim=16
    ).eval()
    diffusion = GaussianDiffusion(timesteps=50)
    cond = torch.randn(2, 16)
    shape = (2, C.VAE_TOKEN_LEN, C.LATENT_DIM)
    result = diffusion.ddim_sample(
        model, shape, cond, steps=3, guidance=3.0, generator=torch.Generator().manual_seed(0)
    )
    assert result.latent.shape == shape
    assert torch.isfinite(result.latent).all()

    # guidance=1.0 等价关闭 CFG，结果应当不同（同 seed）
    plain = diffusion.ddim_sample(
        model, shape, cond, steps=3, guidance=1.0, generator=torch.Generator().manual_seed(0)
    )
    assert not torch.allclose(result.latent, plain.latent)


# ---------------------------------------------------------------------------
# 4. 端到端训练
# ---------------------------------------------------------------------------
def test_resolve_vae_checkpoint_is_not_empty(prepared: dict) -> None:
    """空的 ``runtime.vae_ckpt`` 必须解析成真实路径（防止 ckpt 里存空串）。"""
    cfg = prepared["cfg"]
    assert cfg.runtime.vae_ckpt == ""
    resolved = resolve_vae_checkpoint(cfg)
    assert resolved.is_file()
    assert "vae" in resolved.as_posix()


def test_load_vae_is_frozen(prepared: dict) -> None:
    """加载后的 VAE 必须是 eval 且所有参数 requires_grad=False。"""
    vae = load_vae_for_diffusion(prepared["cfg"], torch.device("cpu"))
    assert vae.training is False
    assert all(not parameter.requires_grad for parameter in vae.parameters())


def test_train_diffusion_end_to_end(prepared: dict) -> None:
    """完整跑通：loss 下降、best.pt 存在、ckpt 元信息正确、loss.jsonl 逐 step 落盘。"""
    cfg = prepared["cfg"]
    result = train_diffusion(cfg, max_steps_per_epoch=4, max_epochs=6)

    losses = result.history.train_loss
    assert len(losses) == 6, f"应有 6 个 epoch，实际 {len(losses)}"
    assert all(np.isfinite(value) for value in losses)
    # 单 step 的扩散 MSE 噪声很大，比较「前两轮均值」与「后两轮均值」
    head = float(np.mean(losses[:2]))
    tail = float(np.mean(losses[-2:]))
    assert tail < head, f"loss 未下降: 前两轮均值 {head:.5f} → 后两轮均值 {tail:.5f}"
    assert result.param_count > 0
    assert np.isfinite(result.best_val)

    best = Path(result.ckpt_path)
    assert best.is_file()

    run_dir = Path(result.run_dir)
    assert (run_dir / "last.pt").is_file()
    assert (run_dir / "summary.json").is_file()

    payload = torch.load(best, map_location="cpu", weights_only=False)
    for key in ("model", "projector", "optimizer", "vae_ckpt", "text_model", "model_config"):
        assert key in payload, f"ckpt 缺少 {key}"
    # 关键回归：必须存「解析后的真实路径」，不能是空字符串
    assert payload["vae_ckpt"], "ckpt 里的 vae_ckpt 为空，推理端会找不到权重"
    assert Path(payload["vae_ckpt"]).is_file()

    loss_path = run_dir / "loss.jsonl"
    lines = [line for line in loss_path.read_text(encoding="utf-8").splitlines() if line.strip()]
    assert len(lines) == 6 * 4  # 6 epoch × 4 step
    for line in lines:
        record = json.loads(line)
        assert set(record) >= {"epoch", "step", "loss"}
        assert np.isfinite(record["loss"])


def test_train_diffusion_respects_step_budget(prepared: dict) -> None:
    """``max_minutes`` 极小值时，跑完第一个 epoch 就停（但必须留下 ckpt）。"""
    cfg = prepared["cfg"]
    result = train_diffusion(cfg, max_steps_per_epoch=1, max_epochs=50, max_minutes=1e-6)
    assert len(result.history.val_loss) == 1
    assert Path(result.run_dir, "best.pt").is_file()
