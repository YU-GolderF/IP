"""Compatibility imports for code written before shared preprocessing existed."""

from __future__ import annotations

import numpy as np

from core.preprocessing import (
    ensure_uint8,
    gaussian_denoise as _shared_gaussian_denoise,
    percentile_normalise,
    to_grayscale,
)

from .segmentation import fingerprint_foreground_mask as _block_foreground_mask


def gaussian_denoise(gray: np.ndarray, sigma: float) -> np.ndarray:
    """Compatibility wrapper using the shared conservative Gaussian filter."""
    return _shared_gaussian_denoise(gray, kernel_size=5, sigma=sigma)


def fingerprint_foreground_mask(gray: np.ndarray, sigma: float = 9.0) -> np.ndarray:
    """Compatibility wrapper for the new block-based segmentation."""
    _ = sigma
    return _block_foreground_mask(gray)


__all__ = [
    "ensure_uint8",
    "fingerprint_foreground_mask",
    "gaussian_denoise",
    "percentile_normalise",
    "to_grayscale",
]
