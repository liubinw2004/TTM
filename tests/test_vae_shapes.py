"""T05 单测：Dataset 取样 / VAE 编解码 shape / mask 生效 / 训练收敛。

需要 torch：torch 未安装时整个模块 SKIP（``SKIP_IF_NO_TORCH``），
torch 装好后由主理人安排回归。
"""

from __future__ import annotations

import shutil
from pathlib import Path

import numpy as np
import pytest

torch = pytest.importorskip("torch", reason="SKIP_IF_NO_TORCH: 需要 torch")

from conftest import build_bvh_text, write_bvh, write_config  # noqa: E402

from ttm import LAYOUT_VERSION  # noqa: E402
from ttm.common import constants as C  # noqa: E402
from ttm.common import paths as P  # noqa: E402
from ttm.common.config import load_config  # noqa: E402
from ttm.data.dataset import MotionDataset, collate_samples  # noqa: E402
from ttm.data.preprocess import PreprocessPipeline  # noqa: E402
from ttm.data.scan import Scanner  # noqa: E402
from ttm.models.vae import MotionVAE, kl_divergence, resample_length_after_conv  # noqa: E402
from ttm.train.train_vae import (  # noqa: E402
    joint_angle_mae_deg,
    masked_l1,
    train_vae,
)

pytestmark = pytest.mark.requires_torch


# ---------------------------------------------------------------------------
# fixtures：离线合成数据集（走完整 scan → preprocess 链路）
# ---------------------------------------------------------------------------
def _build_tree(root: Path) -> None:
    """构造足够训练的小数据集（3 个 clip，均为 5s 以上）。"""
    base = root / "1.7.Daily_Life_Behavior" / "1.7.3.Walking_Movement_With_Posture"
    write_bvh(
        base / "1.7.3.1.Basic_Gait" / "20251211" / "Take_001" / "Take_001_actor_0.bvh",
        build_bvh_text(n_frames=601),
    )
    write_bvh(
        base / "1.7.3.1.Basic_Gait" / "20251212" / "Take_002" / "Take_002_actor_0.bvh",
        build_bvh_text(n_frames=601),
    )
    write_bvh(
        base / "1.7.3.3.Stairs_Up_Down" / "20260326" / "Take_115" / "Take_115_stair_0.bvh",
        build_bvh_text(n_frames=1201),
    )


@pytest.fixture(scope="module")
def prepared_dataset(tmp_path_factory: pytest.TempPathFactory) -> dict:
    """一次性构建「分片 + labels」数据集，供本模块所有用例复用。"""
    work_dir = P.resolve_out(
        P.cache_dir() / "tmp" / "pytest" / f"t05_{abs(hash(str(tmp_path_factory)))}", mkparents=True
    )
    shutil.rmtree(work_dir, ignore_errors=True)
    work_dir.mkdir(parents=True, exist_ok=True)

    dataset_root = work_dir / "BVH_Body"
    _build_tree(dataset_root)
    config_path = write_config(
        work_dir / "cfg.yaml",
        bvh_roots=[dataset_root.as_posix()],
        data_dir="data/unit_test_t05",
        max_files=50,
        shard_size=4,
        val_ratio=0.0,
        test_ratio=0.0,
    )
    cfg = load_config(config_path)
    scan = Scanner(cfg, workers=1).run()
    Scanner(cfg, workers=1).write(scan)
    pipeline = PreprocessPipeline(cfg, workers=2).run()

    yield {"cfg": cfg, "work_dir": work_dir, "result": pipeline}

    shutil.rmtree(P.project_root() / "data" / "unit_test_t05", ignore_errors=True)
    shutil.rmtree(work_dir, ignore_errors=True)


# ---------------------------------------------------------------------------
# 1. Dataset
# ---------------------------------------------------------------------------
def test_dataset_shapes_and_mask(prepared_dataset: dict) -> None:
    """Dataset 取值：feat [146,150] channel-first、mask [150]、字段齐备。"""
    cfg = prepared_dataset["cfg"]
    dataset = MotionDataset(
        cfg.shards_path(), window=150, feature_dim=146, labels_path=cfg.labels_path()
    )
    assert len(dataset) == 4  # 601→151 帧(1 窗) ×2 + 1201→301 帧(2 窗)
    sample = dataset[0]
    assert sample["feat"].shape == (146, 150)
    assert sample["feat"].dtype == torch.float32
    assert sample["mask"].shape == (150,)
    assert sample["mask"].dtype == torch.float32
    assert isinstance(sample["clip_id"], str) and sample["clip_id"].endswith("_win00")
    assert sample["zh_label"]
    assert sample["text"]
    assert len(sample["zh_texts"]) >= 3


def test_dataset_batch_shapes(prepared_dataset: dict) -> None:
    """collate 后 batch 形状为 [B,146,150] / [B,150]。"""
    cfg = prepared_dataset["cfg"]
    dataset = MotionDataset(
        cfg.shards_path(), labels_path=cfg.labels_path(), with_torch=False
    )
    batch = collate_samples([dataset[index] for index in range(min(3, len(dataset)))])
    assert batch["feat"].shape == (min(3, len(dataset)), 146, 150)
    assert batch["mask"].shape == (min(3, len(dataset)), 150)
    assert len(batch["text"]) == batch["feat"].shape[0]


def test_dataset_split_filter(prepared_dataset: dict) -> None:
    """split 过滤：val 比例 0 时 val 集应为空（无泄漏），而不是报错。"""
    cfg = prepared_dataset["cfg"]
    train = MotionDataset(cfg.shards_path(), labels_path=cfg.labels_path(), split="train")
    assert len(train) == 4
    val = MotionDataset(cfg.shards_path(), labels_path=cfg.labels_path(), split="val")
    assert len(val) == 0
    test = MotionDataset(cfg.shards_path(), labels_path=cfg.labels_path(), split="test")
    assert len(test) == 0


def test_dataset_rejects_missing_dir(work_dir: Path) -> None:
    """分片目录不存在时抛 FileNotFoundError。"""
    with pytest.raises(FileNotFoundError):
        MotionDataset(work_dir / "not_a_dir")


# ---------------------------------------------------------------------------
# 2. MotionVAE 形状
# ---------------------------------------------------------------------------
def test_vae_encode_decode_shapes() -> None:
    """encode → (B,38,128)；decode → (B,146,150)（T05 验收标准 ②）。"""
    model = MotionVAE(feature_dim=146, latent_dim=128, hidden=64, downsample=4)
    model.eval()
    feat = torch.randn(2, 146, 150)
    mu, logvar = model.encode(feat)
    assert mu.shape == (2, 38, 128)
    assert logvar.shape == (2, 38, 128)
    recon = model.decode(mu)
    assert recon.shape == (2, 146, 150)

    output = model(feat, sample=False)
    assert output.recon.shape == (2, 146, 150)
    assert output.z.shape == (2, 38, 128)
    assert torch.isfinite(output.recon).all()


def test_vae_token_length_matches_contract() -> None:
    """150 帧 / 4× 下采样 = 38 tokens。"""
    assert resample_length_after_conv(150, 4, 4, 1) == 38
    assert C.token_len_for(150, 4) == C.VAE_TOKEN_LEN == 38
    model = MotionVAE(hidden=32)
    assert model.token_length == 38
    # 不同输入长度也能编解码（decode 用 target_len 对齐）
    feat = torch.randn(1, 146, 120)
    output = model(feat)
    assert output.recon.shape[-1] == 120
    assert output.z.shape[1] == resample_length_after_conv(120, 4, 4, 1)


def test_vae_accepts_channel_last() -> None:
    """``[B,T,146]`` 与 ``[B,146,T]`` 两种布局都接受且等价。"""
    model = MotionVAE(hidden=32)
    model.eval()
    feat = torch.randn(1, 146, 150)
    with torch.no_grad():
        first = model.decode(model.encode(feat)[0])
        second = model.decode(model.encode(feat.transpose(1, 2))[0])
    assert torch.allclose(first, second, atol=1e-5)


def test_vae_param_count_reasonable() -> None:
    """默认结构参数量应在合理区间（同量级的卷积网络）。"""
    model = MotionVAE(feature_dim=146, latent_dim=128, hidden=512, downsample=4)
    count = model.param_count()
    assert 1_000_000 < count < 12_000_000, f"VAE 参数量 {count:,} 超出合理区间"
    assert model.describe()["token_length"] == 38


def test_vae_reparameterize_deterministic() -> None:
    """``sample=False`` 时输出确定（用于推理与评估）。"""
    model = MotionVAE(hidden=32)
    model.eval()
    feat = torch.randn(1, 146, 150)
    with torch.no_grad():
        first = model(feat, sample=False).recon
        second = model(feat, sample=False).recon
    assert torch.equal(first, second)


def test_kl_divergence_nonnegative() -> None:
    """KL 散度必须 >= 0。"""
    mu = torch.zeros(4, 38, 128)
    logvar = torch.zeros(4, 38, 128)
    assert float(kl_divergence(mu, logvar)) == pytest.approx(0.0, abs=1e-6)
    assert float(kl_divergence(mu + 1.0, logvar)) > 0.0


# ---------------------------------------------------------------------------
# 3. mask 感知 loss
# ---------------------------------------------------------------------------
def test_masked_l1_ignores_padded_frames() -> None:
    """pad 帧（mask=0）不得影响 loss（T05 验收标准 ①）。"""
    pred = torch.zeros(2, 146, 10)
    target = torch.zeros(2, 146, 10)
    target[:, :, 5:] = 100.0  # 后 5 帧差异巨大
    mask = torch.ones(2, 10)
    mask[:, 5:] = 0.0  # 但这些帧是 padding

    loss_ignoring = masked_l1(pred, target, mask)
    assert float(loss_ignoring) == pytest.approx(0.0, abs=1e-6)

    mask_full = torch.ones(2, 10)
    loss_full = masked_l1(pred, target, mask_full)
    assert float(loss_full) > 1.0


def test_masked_l1_all_zero_mask() -> None:
    """全 0 mask 不产生 NaN。"""
    pred = torch.zeros(1, 146, 5)
    target = torch.ones(1, 146, 5)
    mask = torch.zeros(1, 5)
    loss = masked_l1(pred, target, mask)
    assert torch.isfinite(loss)
    assert float(loss) == pytest.approx(0.0)


def test_joint_angle_mae() -> None:
    """角度误差：相同四元数为 0°，反向四元数（同旋转）也为 0°。"""
    quat = np.tile(np.array([0.0, 0.0, 0.0, 1.0]), (4, 1))
    assert joint_angle_mae_deg(quat, quat) == pytest.approx(0.0, abs=1e-6)
    assert joint_angle_mae_deg(quat, -quat) == pytest.approx(0.0, abs=1e-6)
    other = np.tile(np.array([0.0, np.sin(np.pi / 4), 0.0, np.cos(np.pi / 4)]), (4, 1))
    assert joint_angle_mae_deg(quat, other) == pytest.approx(90.0, abs=1e-3)


# ---------------------------------------------------------------------------
# 4. 训练（小规模、少步数）
# ---------------------------------------------------------------------------
def test_train_vae_runs_and_loss_decreases(prepared_dataset: dict) -> None:
    """小样本训练：loss 有下降、ckpt 落盘、loss.jsonl 无 NaN。"""
    cfg = prepared_dataset["cfg"]
    from ttm.common.config import TTMConfig, VAEConfig

    override = TTMConfig(
        project=cfg.project,
        data=cfg.data,
        motion=cfg.motion,
        vae=VAEConfig(
            latent_dim=cfg.vae.latent_dim,
            hidden=64,
            downsample=cfg.vae.downsample,
            epochs=6,
            batch_size=2,
            lr=5e-3,
            kl_weight=1e-4,
        ),
        diffusion=cfg.diffusion,
        text=cfg.text,
        runtime=cfg.runtime,
        serve=cfg.serve,
        retarget=cfg.retarget,
        source_path=cfg.source_path,
    )
    result = train_vae(override, max_steps_per_epoch=2)

    assert result.epochs_done == 6
    assert result.params > 0
    assert result.best_val_loss < float("inf")
    assert result.checkpoint and Path(result.checkpoint).is_file()

    loss_path = override.run_dir("vae") / "loss.jsonl"
    assert loss_path.is_file()
    lines = [line for line in loss_path.read_text(encoding="utf-8").splitlines() if line.strip()]
    assert len(lines) == 12  # 6 epoch × 2 step
    import json

    values = [json.loads(line)["loss"] for line in lines]
    assert all(np.isfinite(values))
    assert values[-1] < values[0], f"loss 未下降: {values[0]:.4f} -> {values[-1]:.4f}"

    shutil.rmtree(P.project_root() / "runs" / "unit_test", ignore_errors=True)


def test_train_vae_requires_shards(work_dir: Path) -> None:
    """没有分片时报错（提示先跑预处理）。"""
    config_path = write_config(
        work_dir / "cfg.yaml",
        bvh_roots=[(work_dir / "nowhere").as_posix()],
        data_dir="data/unit_test_t05_empty",
    )
    cfg = load_config(config_path)
    with pytest.raises((FileNotFoundError, ValueError)):
        train_vae(cfg, max_steps_per_epoch=1, max_epochs=1)


# ---------------------------------------------------------------------------
# 8. 后验塌缩的预防与诊断（M2 返工项）
# ---------------------------------------------------------------------------
def test_vae_logvar_bias_init() -> None:
    """``logvar`` 分支的初始偏置必须是负的（否则初始 std=1，采样全是噪声）。"""
    from ttm.models.vae import MotionVAE

    model = MotionVAE(feature_dim=146, latent_dim=16, hidden=32, downsample=4, logvar_bias=-2.0)
    bias = model.enc_out.bias.detach()
    mu_bias = bias[: model.latent_dim]
    logvar_bias = bias[model.latent_dim :]
    # logvar 分支被显式设成 -2.0；mu 分支保持 Conv1d 的默认随机初始化
    assert float((logvar_bias - (-2.0)).abs().max()) < 1e-6
    assert float(mu_bias.abs().max()) > 0.0
    assert not torch.allclose(mu_bias, logvar_bias)
    # 初始后验 std = exp(-2/2) ≈ 0.368，远小于默认的 1.0
    assert float((logvar_bias * 0.5).exp().mean()) < 0.5
    assert model.describe()["logvar_bias"] == -2.0


def test_posterior_diagnostics_reports_collapse_fingerprint(prepared_dataset: dict) -> None:
    """诊断函数必须返回塌缩指纹所需的全部字段。"""
    from ttm.data.dataset import MotionDataset
    from ttm.models.vae import MotionVAE
    from ttm.train.train_vae import posterior_diagnostics

    cfg = prepared_dataset["cfg"]
    dataset = MotionDataset(
        cfg.shards_path(), window=cfg.data.window, split=None, cache_size=4, with_torch=True
    )
    model = MotionVAE(feature_dim=146, latent_dim=16, hidden=32, downsample=4)
    metrics = posterior_diagnostics(model, dataset, torch.device("cpu"), max_batches=1)

    for key in ("mu_std", "logvar_mean", "z_std", "recon_frame_var", "target_frame_var", "var_ratio"):
        assert key in metrics, f"缺少诊断字段 {key}"
        assert np.isfinite(metrics[key]), f"{key} 非有限值: {metrics[key]}"
    assert metrics["target_frame_var"] > 0, "真值序列应有帧间变化"
    # 未训练模型的重建不应比真值更「活跃」太多
    assert metrics["var_ratio"] < 10.0


def test_kl_warmup_ramps_linearly(prepared_dataset: dict) -> None:
    """KL 权重必须按 epoch 线性上升，并在 warmup 结束后等于配置值。"""
    import json

    import shutil

    from ttm.common import paths as P
    from ttm.common.config import TTMConfig, VAEConfig
    from ttm.train.train_vae import train_vae

    cfg = prepared_dataset["cfg"]
    override = TTMConfig(
        project=cfg.project,
        data=cfg.data,
        motion=cfg.motion,
        vae=VAEConfig(
            latent_dim=cfg.vae.latent_dim,
            hidden=64,
            downsample=cfg.vae.downsample,
            epochs=4,
            batch_size=2,
            lr=5e-3,
            kl_weight=1e-3,
            kl_warmup_epochs=4,
        ),
        diffusion=cfg.diffusion,
        text=cfg.text,
        runtime=cfg.runtime,
        serve=cfg.serve,
        retarget=cfg.retarget,
        source_path=cfg.source_path,
    )
    train_vae(override, max_steps_per_epoch=1)

    loss_path = override.run_dir("vae") / "loss.jsonl"
    rows = [
        json.loads(line)
        for line in loss_path.read_text(encoding="utf-8").splitlines()
        if line.strip()
    ]
    weights = [row["kl_weight"] for row in rows]
    assert weights == sorted(weights), f"KL 权重未按 epoch 单调上升: {weights}"
    assert weights[0] == pytest.approx(1e-3 / 4), f"首轮应为满权重的 1/4，实得 {weights[0]}"
    assert weights[-1] == pytest.approx(1e-3), f"末轮应到满权重，实得 {weights[-1]}"

    shutil.rmtree(P.project_root() / "runs" / "unit_test", ignore_errors=True)


# ---------------------------------------------------------------------------
# logvar 上界（M2 实测：val loss 每 20 个 epoch 就跳出一次 6~112 的尖峰）
# ---------------------------------------------------------------------------
def test_logvar_is_clamped() -> None:
    """``logvar`` 必须被夹住：KL 对 ``exp(logvar)`` 线性，没有上界就会漂到 +6。"""
    torch = pytest.importorskip("torch")
    from ttm.models.vae import MotionVAE

    model = MotionVAE(feature_dim=146, latent_dim=128, hidden=64, logvar_clamp=(-10.0, 2.0))
    model.eval()
    feat = torch.randn(3, 146, 150) * 5.0
    with torch.no_grad():
        mu, logvar = model.encode(feat)
    assert float(logvar.max()) <= 2.0 + 1e-6
    assert float(logvar.min()) >= -10.0 - 1e-6
    assert torch.isfinite(mu).all()


def test_logvar_clamp_bounds_the_logvar_side_of_kl() -> None:
    """夹住之后 KL 的 **logvar 侧**有上界：``0.5*(e^hi - 1 - hi)*4864``。

    注意 ``mu`` 那一侧**没有**上界，也不该有——``mu`` 大意味着后验信息量大，
    KL 里 ``mu²`` 项变大是正常的（它不是数值问题，是模型在用潜变量）。
    M2 的 val 尖峰是 ``logvar`` 侧失控：``logvar≈6`` 时单样本 logvar 项就有
    ``0.5*(e^6-1-6)*4864 ≈ 7.2e5``，×kl_weight=1e-5 → 单批次 loss 直接 +7，
    实测最高把 val 顶到 112。夹到 2.0 之后这个上界降到 10674。
    """
    torch = pytest.importorskip("torch")
    from ttm.models.vae import kl_divergence

    shape = (2, 38, 128)
    mu_zero = torch.zeros(shape)
    logvar_hi = torch.full(shape, 2.0)
    logvar_side = float(kl_divergence(mu_zero, logvar_hi)) - float(
        kl_divergence(mu_zero, torch.full(shape, 0.0))
    )
    upper = 0.5 * (float(np.exp(2.0)) - 1.0 - 2.0) * 38 * 128
    assert logvar_side == pytest.approx(upper, rel=1e-5)


def test_logvar_clamp_rejects_inverted_bounds() -> None:
    """上下界写反要报错，不能静默夹成一个点。"""
    pytest.importorskip("torch")
    from ttm.models.vae import MotionVAE

    with pytest.raises(ValueError):
        MotionVAE(feature_dim=146, latent_dim=128, hidden=64, logvar_clamp=(2.0, -10.0))


def test_prune_checkpoints_keeps_only_recent_snapshots(work_dir) -> None:
    """``_prune_checkpoints`` 只留最近 N 个快照，且**绝不删** ``best.pt``。

    500 epoch × 120MB = 60GB，而 ``D:`` 已用 93%——把盘写满是会让训练中途崩、
    连 ``best.pt`` 一起写坏的实打实风险，所以这条行为要有测试兜住。
    """
    pytest.importorskip("torch")
    from ttm.train.train_vae import _prune_checkpoints

    for epoch in range(5):
        (work_dir / f"ckpt_{epoch:04d}.pt").write_bytes(b"x")
    (work_dir / "best.pt").write_bytes(b"x")
    (work_dir / "last.pt").write_bytes(b"x")

    removed = _prune_checkpoints(work_dir, keep=2)

    assert removed == 3
    assert sorted(path.name for path in work_dir.glob("ckpt_*.pt")) == [
        "ckpt_0003.pt",
        "ckpt_0004.pt",
    ]
    assert (work_dir / "best.pt").is_file()
    assert (work_dir / "last.pt").is_file()


def test_prune_checkpoints_is_noop_when_under_limit(work_dir) -> None:
    """快照数不超上限时一个都不删。"""
    pytest.importorskip("torch")
    from ttm.train.train_vae import _prune_checkpoints

    (work_dir / "ckpt_0000.pt").write_bytes(b"x")
    assert _prune_checkpoints(work_dir, keep=3) == 0
    assert (work_dir / "ckpt_0000.pt").is_file()
