import numpy as np
import pytest

torch = pytest.importorskip("torch")

from puppetmaster.pipeline.blend import blend_latents, mask_to_latent  # noqa: E402
from puppetmaster.pipeline.denoise import build_cond_hidden_states, pose_residual_term  # noqa: E402


def test_mask_to_latent_shape_and_range():
    mask = np.zeros((64, 64), dtype=bool)
    mask[16:48, 16:48] = True
    m = mask_to_latent(mask, (8, 8))
    assert m.shape == (1, 1, 8, 8) and m.dtype == torch.float32
    assert 0 <= m.min() and m.max() <= 1
    assert m[0, 0, 4, 4] == 1 and m[0, 0, 0, 0] == 0


def test_mask_to_latent_uint8_255():
    m = mask_to_latent(np.full((16, 16), 255, np.uint8), (2, 2))
    torch.testing.assert_close(m, torch.ones(1, 1, 2, 2))


def test_blend_latents_extremes():
    zs, zt = torch.ones(1, 4, 8, 8), torch.zeros(1, 4, 8, 8)
    torch.testing.assert_close(blend_latents(zs, zt, np.ones((64, 64))), zs)
    torch.testing.assert_close(blend_latents(zs, zt, np.zeros((64, 64))), zt)


def test_cond_hidden_states():
    hs = build_cond_hidden_states(torch.randn(1, 77, 16), torch.randn(3, 2, 16))
    assert hs.shape == (3, 79, 16)


def test_pose_residual_term():
    like = torch.zeros(2, 4, 8, 8)
    term = pose_residual_term([torch.full((2, 1280), 10.0)], like, scale=0.1, clamp=0.5)
    assert term.shape == like.shape
    torch.testing.assert_close(term, torch.full_like(like, 0.05))
