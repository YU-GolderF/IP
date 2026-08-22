from __future__ import annotations

import numpy as np
import pytest

from algorithms.rhlt import RHLTConfig, run_rhlt
from algorithms.rhlt.orientation import estimate_orientation_field, smooth_orientation_field
from algorithms.rhlt.ridge_filter import enhance_ridges_with_gabor
from algorithms.rhlt.segmentation import segment_fingerprint
from core.preprocessing import preprocess_fingerprint


def synthetic_fingerprint(height: int = 96, width: int = 96) -> np.ndarray:
    y, x = np.mgrid[0:height, 0:width]
    radius = np.hypot(x - width / 2, y - height / 2)
    angle = np.arctan2(y - height / 2, x - width / 2)
    ridges = 128 + 75 * np.sin(0.35 * radius + 1.8 * np.sin(angle))
    return np.clip(ridges, 0, 255).astype(np.uint8)


def test_blank_image_returns_safe_warning_and_empty_foreground():
    result = run_rhlt(np.zeros((64, 64), dtype=np.uint8), small_config())

    assert result["status"] == "warning"
    assert not result["foreground_mask"].any()
    assert result["ridge_restored"].shape == (64, 64)


@pytest.mark.parametrize(
    "image",
    [
        np.full((64, 64), 128, dtype=np.uint8),
        synthetic_fingerprint(),
        np.dstack([synthetic_fingerprint()] * 3),
        synthetic_fingerprint(8, 10),
    ],
)
def test_rhlt_handles_uniform_grayscale_colour_and_small_images(image):
    result = run_rhlt(image, small_config())
    expected_shape = image.shape[:2]

    assert result["ridge_restored"].shape == expected_shape
    assert result["ridge_binary"].shape == expected_shape
    assert result["foreground_mask"].shape == expected_shape
    assert np.isfinite(result["orientation_field"]).all()


def test_run_rhlt_returns_expected_result_keys():
    result = run_rhlt(synthetic_fingerprint(), small_config())
    expected = {
        "preprocessed",
        "foreground_mask",
        "orientation_field",
        "orientation_visualisation",
        "ridge_restored",
        "ridge_binary",
        "enhanced_image",
        "metrics",
        "processing_time_ms",
    }
    assert expected.issubset(result)


def test_orientation_and_gabor_outputs_are_valid():
    preprocessed = preprocess_fingerprint(synthetic_fingerprint())
    mask, _ = segment_fingerprint(preprocessed, block_size=8)
    orientation, valid, _ = estimate_orientation_field(
        preprocessed, mask, block_size=8
    )
    smoothed = smooth_orientation_field(orientation, valid, sigma=1.0)
    config = small_config()
    restored = enhance_ridges_with_gabor(
        preprocessed, smoothed, mask, config, valid_blocks=valid
    )

    assert orientation.shape == valid.shape
    assert np.isfinite(smoothed).all()
    assert valid.any()
    assert restored.shape == preprocessed.shape
    assert restored.dtype == np.uint8


def small_config() -> RHLTConfig:
    return RHLTConfig(
        psf_size=33,
        block_size=8,
        orientation_bins=8,
        gabor_kernel_size=15,
        gabor_sigma=3.0,
        gabor_lambda=8.0,
    )

