import numpy as np

from puppetmaster.extractors.bfm import N_EXPR, N_SHAPE, estimate_illumination, fit_bfm_approx


def _lmks():
    rng = np.random.default_rng(0)
    return rng.uniform(20, 44, size=(68, 2)).astype(np.float32)


def test_fit_bfm_approx_shapes():
    sp, ep = fit_bfm_approx(_lmks(), (64, 64))
    assert sp.shape == (N_SHAPE,) and ep.shape == (N_EXPR,)
    assert sp.dtype == np.float32 and ep.dtype == np.float32
    assert np.abs(sp).max() <= 1


def test_estimate_illumination():
    img = np.full((64, 64, 3), 128, np.uint8)
    sh = estimate_illumination(img, _lmks())
    assert sh.shape == (27,) and sh.dtype == np.float32
    np.testing.assert_allclose(sh[:3], 128 / 255, rtol=1e-5)
