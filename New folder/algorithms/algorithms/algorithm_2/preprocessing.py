"""Algorithm 2 adapter for the shared preprocessing interface.

The shared preprocessing implementation remains the single owner of grayscale,
denoising, percentile normalisation and CLAHE. This module only adds the
DCT-specific ROI mask needed to exclude background blocks.
"""
from __future__ import annotations

import cv2
import numpy as np

from core.preprocessing import PreprocessingConfig, preprocess_with_stages


def prepare_dct_input(
    image: np.ndarray,
    config: PreprocessingConfig | None = None,
) -> tuple[dict[str, np.ndarray], np.ndarray]:
    """Run standard shared stages and return its final image plus a DCT ROI."""
    stages = preprocess_with_stages(image, config)
    roi = estimate_fingerprint_roi(stages["normalised"], block_size=32)
    return stages, roi


def estimate_fingerprint_roi(gray: np.ndarray, block_size: int) -> np.ndarray:
    """DCT-only valid-region mask; this does not alter shared preprocessing."""
    image = gray.astype(np.float32)
    mean = cv2.boxFilter(image, -1, (block_size, block_size), normalize=True)
    mean_sq = cv2.boxFilter(image * image, -1, (block_size, block_size), normalize=True)
    deviation = np.sqrt(np.maximum(mean_sq - mean * mean, 0.0))
    threshold = max(float(np.percentile(deviation, 45)), 4.0)
    mask = (deviation >= threshold).astype(np.uint8)
    kernel = np.ones((5, 5), np.uint8)
    mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, kernel)
    mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, kernel)
    return mask.astype(bool)
