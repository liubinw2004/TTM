"""T06 单测：文本投影/编码器、DiT 前向 shape、参数量、确定性、DDPM/DDIM。

需要 torch：torch 未安装时整个模块 SKIP。
bge 模型未下载时，仅「真实模型」相关用例 SKIP（其余全部照跑）。
"""

from __future__ import annotations

import numpy as np
import pytest

torch = pytest.importorskip("torch", reason="SKIP_IF_NO_TORCH: 需要 torch")

from ttm.common import constants as C  # noqa: E402
from ttm.models.diffusion import (  # noqa: E402
    DiffusionTransformer,
    GaussianDiffusion,
    sinusoidal_embedding,
)
from ttm.models.text_encoder import (  # noqa: E402
    BGE_HIDDEN_DIM,
    DEFAULT_MODEL_PATH,
    TextEncoder,
    TextEncoderUnavailable,
    TextProjector,
    cosine_similarity,
    null_condition,
)

pytestmark = pytest.mark.requires_torch


# ---------------------------------------------------------------------------
# 1. 文本投影层
# ---------------------------------------------------------------------------
def test_text_projector_shapes() -> None:
    """512d → 256d；也支持自定义维度。"""
    projector = TextProjector(512, 256)
    out = projector(torch.randn(4, 512))
    assert out.shape == (4, 256)
    assert torch.isfinite(out).all()

    small = TextProjector(512, 64)
    assert small(torch.randn(2, 512)).shape == (2, 64)

    with pytest.raises(ValueError):
        projector(torch.randn(4, 128))


def test_text_projector_deterministic_in_eval() -> None:
    """eval 模式下同输入两次前向输出一致。"""
    projector = TextProjector(512, 256)
    projector.eval()
    vec = torch.randn(3, 512)
    with torch.no_grad():
        first = projector(vec)
        second = projector(vec)
    assert torch.equal(first, second)


def test_text_encoder_project_numpy() -> None:
    """``TextEncoder.project`` 接受 numpy 512d 向量并返回 [N,256]。"""
    encoder = TextEncoder(proj_dim=256, device="cpu", local_files_only=True)
    vecs = np.random.default_rng(0).normal(size=(5, BGE_HIDDEN_DIM)).astype(np.float32)
    projected = encoder.project(vecs)
    assert projected.shape == (5, 256)
    assert projected.dtype == np.float32
    assert np.isfinite(projected).all()


def test_text_encoder_describe_without_backbone() -> None:
    """未加载 backbone 时结构信息仍可读，且参数量很小。"""
    encoder = TextEncoder(proj_dim=256, local_files_only=True)
    info = encoder.describe()
    assert info["proj_dim"] == 256
    assert info["hidden_dim"] == BGE_HIDDEN_DIM
    assert info["frozen"] is True
    assert 0 < info["projector_params"] < 1_000_000
    assert encoder.is_loaded is False


def test_text_encoder_unavailable_is_graceful() -> None:
    """权重路径不存在时 ``is_available()`` 返回 False 而不是抛异常（离线降级）。"""
    encoder = TextEncoder(
        model_path="D:/TTM/models/__not_exist__",
        model_name="not-a-real-model-xyz",
        local_files_only=True,
    )
    assert encoder.is_available() is False
    with pytest.raises(TextEncoderUnavailable):
        encoder.encode(["上楼梯"])


def test_null_condition_shape() -> None:
    """CFG 的无条件向量为零向量。"""
    cond = null_condition(256, 3)
    assert cond.shape == (3, 256)
    assert float(cond.abs().sum()) == pytest.approx(0.0)


def test_cosine_similarity_helper() -> None:
    """余弦相似度辅助函数行为正确。"""
    assert cosine_similarity(np.array([1.0, 0.0]), np.array([1.0, 0.0])) == pytest.approx(1.0)
    assert cosine_similarity(np.array([1.0, 0.0]), np.array([0.0, 1.0])) == pytest.approx(0.0)
    assert cosine_similarity(np.array([1.0, 0.0]), np.array([-1.0, 0.0])) == pytest.approx(-1.0)
    assert cosine_similarity(np.zeros(2), np.array([1.0, 0.0])) == pytest.approx(0.0)


# ---------------------------------------------------------------------------
# 2. 真实 bge 编码器（项目内离线权重；权重缺失时才 skip）
# ---------------------------------------------------------------------------
def _encoder_or_skip() -> TextEncoder:
    """构造真实编码器（项目内离线权重 ``D:/TTM/models/bge-small-zh-v1.5``）。"""
    encoder = TextEncoder(proj_dim=256, frozen=True, max_len=32, local_files_only=True)
    if not encoder.is_available():
        pytest.skip(f"bge 权重缺失（{DEFAULT_MODEL_PATH} 不存在）")
    assert encoder.model_source.startswith("D:")
    return encoder


def test_bge_encode_semantics() -> None:
    """同义句 cos > 0.7，异义句 cos < 0.4（T06 验收标准 ①）。"""
    encoder = _encoder_or_skip()
    texts = ["上楼梯", "上楼", "打拳", "扫地", "喝水"]
    vectors = encoder.encode(texts)
    assert vectors.shape == (len(texts), BGE_HIDDEN_DIM)
    assert np.allclose(np.linalg.norm(vectors, axis=1), 1.0, atol=1e-5)

    same = cosine_similarity(vectors[0], vectors[1])
    different = cosine_similarity(vectors[0], vectors[2])
    assert same > 0.7, f"「上楼梯」vs「上楼」相似度 {same:.3f} 应 > 0.7"
    assert different < 0.4, f"「上楼梯」vs「打拳」相似度 {different:.3f} 应 < 0.4"


def test_bge_forward_projects_to_256() -> None:
    """端到端：文本 → [B,256] 条件向量，且投影层有梯度。"""
    encoder = _encoder_or_skip()
    encoder.train(True)
    cond = encoder(["慢慢走上楼梯", "打一套陈式太极"])
    assert cond.shape == (2, 256)
    loss = cond.pow(2).mean()
    loss.backward()
    grads = [p.grad for p in encoder.parameters() if p.grad is not None]
    assert grads and all(torch.isfinite(g).all() for g in grads)


# ---------------------------------------------------------------------------
# 3. DiT
# ---------------------------------------------------------------------------
def test_sinusoidal_embedding() -> None:
    """时间步编码形状正确，且 t=0 与 t=1 不同。"""
    t = torch.tensor([0, 1, 999])
    emb = sinusoidal_embedding(t, 256)
    assert emb.shape == (3, 256)
    assert torch.isfinite(emb).all()
    assert not torch.allclose(emb[0], emb[1])
    with pytest.raises(ValueError):
        sinusoidal_embedding(t, 255)


def test_dit_forward_shape() -> None:
    """[B,38,128] → [B,38,128]（T06 验收标准 ②）。"""
    model = DiffusionTransformer(latent_dim=128, d_model=256, n_layers=8, n_heads=8, cond_dim=256)
    model.eval()
    x = torch.randn(2, C.VAE_TOKEN_LEN, 128)
    t = torch.tensor([0, 500])
    cond = torch.randn(2, 256)
    with torch.no_grad():
        eps = model(x, t, cond)
    assert eps.shape == (2, C.VAE_TOKEN_LEN, 128)
    assert torch.isfinite(eps).all()


def test_dit_param_count_in_range() -> None:
    """默认结构参数量 ∈ [8M, 12M]（ARCH §1.2 / §10 T06）。"""
    model = DiffusionTransformer(latent_dim=128, d_model=256, n_layers=8, n_heads=8, cond_dim=256)
    count = model.param_count()
    assert 8_000_000 <= count <= 12_000_000, f"DiT 参数量 {count:,} 超出 [8M,12M]"


def test_dit_deterministic_without_dropout() -> None:
    """dropout 关闭时同输入两次前向输出完全一致（T06 验收标准 ③）。"""
    model = DiffusionTransformer(dropout=0.0)
    model.eval()
    x = torch.randn(2, 38, 128)
    t = torch.tensor([10, 700])
    cond = torch.randn(2, 256)
    with torch.no_grad():
        first = model(x, t, cond)
        second = model(x, t, cond)
    assert torch.equal(first, second)


def test_dit_accepts_longer_sequences() -> None:
    """长动作（360 帧 → 90 tokens）超过 max_tokens=64 时靠位置编码插值支持。"""
    model = DiffusionTransformer(max_tokens=64)
    model.eval()
    x = torch.randn(1, 90, 128)
    with torch.no_grad():
        eps = model(x, torch.tensor([5]), torch.randn(1, 256))
    assert eps.shape == (1, 90, 128)
    assert model.tokens_for_frames(360) == 90
    assert model.tokens_for_frames(150) == 38


def test_dit_rejects_bad_inputs() -> None:
    """输入维度错误必须报错。"""
    model = DiffusionTransformer()
    with pytest.raises(ValueError):
        model(torch.randn(2, 38, 64), torch.tensor([0, 1]), torch.randn(2, 256))
    with pytest.raises(ValueError):
        model(torch.randn(2, 38, 128), torch.tensor([0]), torch.randn(2, 256))
    with pytest.raises(ValueError):
        model(torch.randn(2, 38, 128), torch.tensor([0, 1]), torch.randn(2, 128))


def test_dit_cond_affects_output() -> None:
    """不同文本条件必须产生不同噪声预测（否则文本条件是死的）。"""
    model = DiffusionTransformer(dropout=0.0)
    model.eval()
    x = torch.randn(1, 38, 128)
    t = torch.tensor([100])
    generator = torch.Generator().manual_seed(7)
    cond_a = torch.randn(1, 256, generator=generator)
    cond_b = torch.randn(1, 256, generator=generator)
    with torch.no_grad():
        out_a = model(x, t, cond_a)
        out_b = model(x, t, cond_b)
    assert not torch.allclose(out_a, out_b)
    # 零条件（CFG 的无条件分支）也必须与真实条件不同
    with torch.no_grad():
        out_null = model(x, t, torch.zeros(1, 256))
    assert not torch.allclose(out_a, out_null)


# ---------------------------------------------------------------------------
# 4. DDPM / DDIM
# ---------------------------------------------------------------------------
def test_q_sample_extremes() -> None:
    """t=0 时 x_t ≈ x0；t 越大噪声占比越高（统计意义上 std 趋近 1）。"""
    diffusion = GaussianDiffusion(timesteps=1000)
    x0 = torch.ones(64, 50)
    generator = torch.Generator().manual_seed(0)

    x_t0, noise0 = diffusion.q_sample(x0, torch.tensor([0]), generator=generator)
    sqrt_alpha0 = float(torch.sqrt(diffusion.alphas_cumprod[0]))
    assert torch.allclose(
        x_t0, x0 * sqrt_alpha0 + noise0 * (1 - sqrt_alpha0**2) ** 0.5, atol=1e-5
    )
    # t=0 时几乎不含噪声：与 x0 的偏差远小于 1
    assert float((x_t0 - x0).abs().mean()) < 0.02

    # 噪声占比随 t 单调上升（用大样本的 |x_t − x0| 均值衡量）
    deviations = []
    for step in (100, 500, 999):
        sample, _ = diffusion.q_sample(
            x0, torch.full((64,), step), generator=torch.Generator().manual_seed(step)
        )
        deviations.append(float((sample - x0).abs().mean()))
        assert torch.isfinite(sample).all()
    assert deviations[0] < deviations[1] < deviations[2]

    # 最末时间步近似纯噪声：均值≈0、std≈1
    x_late, _ = diffusion.q_sample(
        x0, torch.tensor([999]), generator=torch.Generator().manual_seed(999)
    )
    assert abs(float(x_late.mean())) < 0.1
    assert abs(float(x_late.std()) - 1.0) < 0.1


def test_p_losses_finite_and_cfg_dropout() -> None:
    """损失有限；``cfg_dropout=1.0`` 时所有样本都被置空。"""
    diffusion = GaussianDiffusion(timesteps=100)
    model = DiffusionTransformer(n_layers=2, d_model=64, n_heads=4, cond_dim=64, max_tokens=16)
    x0 = torch.randn(4, 8, 128)
    cond = torch.randn(4, 64)

    output = diffusion.p_losses(model, x0, cond, cfg_dropout=0.0, generator=torch.Generator().manual_seed(1))
    assert torch.isfinite(output.loss)
    assert output.n_dropped == 0
    assert output.eps_pred.shape == x0.shape

    dropped = diffusion.p_losses(
        model, x0, cond, cfg_dropout=1.0, generator=torch.Generator().manual_seed(2)
    )
    assert dropped.n_dropped == 4
    assert torch.isfinite(dropped.loss)


def test_p_losses_with_mask() -> None:
    """token 掩码生效：全 0 掩码时 loss 为 0 且不产生 NaN。"""
    diffusion = GaussianDiffusion(timesteps=50)
    model = DiffusionTransformer(n_layers=1, d_model=32, n_heads=2, cond_dim=32, max_tokens=16)
    x0 = torch.randn(2, 8, 128)
    cond = torch.randn(2, 32)
    mask = torch.zeros(2, 8)
    output = diffusion.p_losses(model, x0, cond, cfg_dropout=0.0, mask=mask)
    assert torch.isfinite(output.loss)
    assert float(output.loss.detach()) == pytest.approx(0.0)


def test_ddim_sample_shape_and_determinism() -> None:
    """DDIM 输出形状正确；同 seed 两次采样逐元素一致（T08 可复现性的基础）。"""
    diffusion = GaussianDiffusion(timesteps=100)
    model = DiffusionTransformer(n_layers=1, d_model=64, n_heads=4, cond_dim=64, max_tokens=16)
    model.eval()
    cond = torch.randn(1, 64)
    shape = (1, C.VAE_TOKEN_LEN, 128)

    first = diffusion.ddim_sample(
        model, shape, cond, steps=5, guidance=3.0, generator=torch.Generator().manual_seed(7)
    )
    second = diffusion.ddim_sample(
        model, shape, cond, steps=5, guidance=3.0, generator=torch.Generator().manual_seed(7)
    )
    assert first.latent.shape == shape
    assert first.steps == 5
    assert torch.equal(first.latent, second.latent)
    assert torch.isfinite(first.latent).all()


def test_ddim_without_guidance() -> None:
    """``guidance=1.0`` 走单次前向分支，输出同样合法。"""
    diffusion = GaussianDiffusion(timesteps=50)
    model = DiffusionTransformer(n_layers=1, d_model=32, n_heads=2, cond_dim=32, max_tokens=16)
    model.eval()
    result = diffusion.ddim_sample(
        model,
        (2, 10, 128),
        torch.randn(2, 32),
        steps=3,
        guidance=1.0,
        generator=torch.Generator().manual_seed(3),
    )
    assert result.latent.shape == (2, 10, 128)
    assert torch.isfinite(result.latent).all()


def test_diffusion_coefficients_sane() -> None:
    """扩散系数单调递减、边界合理。"""
    diffusion = GaussianDiffusion(timesteps=1000)
    cumulative = diffusion.alphas_cumprod
    assert torch.all(cumulative[:-1] >= cumulative[1:])
    assert 0.0 < float(cumulative[-1]) < 0.1
    assert float(cumulative[0]) > 0.99
