import pytest

torch = pytest.importorskip("torch")

from puppetmaster.adapters import IdentityProjector, IlluminationAdaIN, PoseExpressionAdapter  # noqa: E402


def test_identity_projector_shape_dtype_and_norm():
    torch.manual_seed(0)
    m = IdentityProjector(arcface_dim=16, clip_id_dim=8, cross_attn_dim=12, num_tokens=3)
    out = m(torch.randn(2, 16), torch.randn(2, 8), lambda_id=2.0)
    assert out.shape == (2, 3, 12)
    assert out.dtype == torch.float32
    torch.testing.assert_close(out.norm(dim=-1), torch.full((2, 3), 2.0))


def test_identity_projector_default_dims():
    out = IdentityProjector()(torch.randn(1, 512), torch.randn(1, 128))
    assert out.shape == (1, 2, 768)


def test_pose_expr_adapter_shapes():
    m = PoseExpressionAdapter()
    out = m(torch.randn(2, 6), torch.randn(2, 40), torch.randn(2, 10))
    assert out["heatmap_feat"].shape == (2, 512)
    assert [r.shape for r in out["residuals"]] == [(2, c) for c in m.LATENT_CHANNELS]
    assert all(r.abs().max() <= 1 for r in out["residuals"])  # tanh heads


def test_illumination_adain_identity_at_init():
    m = IlluminationAdaIN(sh_dim=27)
    fmap = torch.randn(2, 320, 4, 4)
    out = m(torch.randn(2, 27), fmap, res_idx=2, lambda_light=0.0)
    assert out.shape == fmap.shape and out.dtype == fmap.dtype
    torch.testing.assert_close(out, fmap)
    # with lambda=1 and zero-init heads, output is the instance-normalised map
    out1 = m(torch.randn(2, 27), fmap, res_idx=2, lambda_light=1.0)
    assert out1.shape == fmap.shape
    assert out1.mean(dim=[2, 3]).abs().max() < 1e-5


def test_float64_supported():
    m = PoseExpressionAdapter().double()
    out = m(torch.randn(1, 6, dtype=torch.float64), torch.randn(1, 40, dtype=torch.float64),
            torch.randn(1, 10, dtype=torch.float64))
    assert out["heatmap_feat"].dtype == torch.float64
