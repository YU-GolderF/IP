import numpy as np

from algorithms.rhlt.config import RHLTConfig
from algorithms.rhlt.core import build_rhlt_psf
from algorithms.rhlt.pipeline import process_fingerprint


def synthetic_fingerprint(size=192):
    y, x = np.mgrid[0:size, 0:size]
    cx = cy = size / 2
    r = np.hypot(x - cx, y - cy)
    theta = np.arctan2(y - cy, x - cx)
    pattern = np.sin(0.20 * r + 2.0 * np.sin(theta))
    image = 128 + 80 * pattern
    rng = np.random.default_rng(7)
    image += rng.normal(0, 12, image.shape)
    return np.clip(image, 0, 255).astype(np.uint8)


def test_psf_shape_and_finiteness():
    psf = build_rhlt_psf(65, 1, 0.9, 0.0)
    assert psf.shape == (65, 65)
    assert np.isfinite(psf.real).all()
    assert np.isfinite(psf.imag).all()


def test_full_pipeline_runs():
    image = synthetic_fingerprint()
    result = process_fingerprint(image, RHLTConfig())
    assert result["rhlt_stretched"].shape == image.shape
    assert result["ridge_enhanced"].dtype == np.uint8
    assert result["mask"].dtype == bool
    assert result["metrics"]["processing_time_ms"] >= 0
