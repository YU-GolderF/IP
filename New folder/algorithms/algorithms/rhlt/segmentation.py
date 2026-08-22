"""Block-based fingerprint foreground segmentation."""

from __future__ import annotations

import cv2
import numpy as np

from core.preprocessing import to_grayscale


def _largest_component(mask: np.ndarray) -> np.ndarray:
    if not np.any(mask):
        return np.zeros_like(mask, dtype=bool)
    count, labels, stats, _ = cv2.connectedComponentsWithStats(mask.astype(np.uint8), connectivity=8)
    if count <= 1:
        return mask.astype(bool)
    label = 1 + int(np.argmax(stats[1:, cv2.CC_STAT_AREA]))
    return labels == label


def segment_fingerprint(
    image: np.ndarray,
    block_size: int = 16,
    min_block_std: float = 6.0,
    threshold_scale: float = 0.55,
) -> tuple[np.ndarray, np.ndarray]:
    """
    Segment ridge-bearing blocks using local intensity standard deviation.

    Uniform or blank inputs return an empty mask. The function deliberately
    avoids replacing a failed segmentation with an all-foreground debug mask.
    """
    if block_size < 4:
        raise ValueError("block_size must be at least 4")
    if min_block_std < 0 or threshold_scale <= 0:
        raise ValueError("segmentation thresholds must be non-negative")

    gray = to_grayscale(image)
    height, width = gray.shape
    block_rows = max(1, int(np.ceil(height / block_size)))
    block_columns = max(1, int(np.ceil(width / block_size)))
    local_std = np.zeros((block_rows, block_columns), dtype=np.float32)

    for block_y in range(block_rows):
        y_start = block_y * block_size
        y_end = min(height, y_start + block_size)
        for block_x in range(block_columns):
            x_start = block_x * block_size
            x_end = min(width, x_start + block_size)
            local_std[block_y, block_x] = float(np.std(gray[y_start:y_end, x_start:x_end]))

    informative = local_std[local_std > 1e-6]
    if informative.size == 0:
        block_mask = np.zeros_like(local_std, dtype=bool)
    else:
        adaptive_threshold = max(min_block_std, float(np.percentile(informative, 70)) * threshold_scale)
        block_mask = local_std >= adaptive_threshold
        if block_mask.size >= 4:
            kernel = np.ones((3, 3), dtype=np.uint8)
            block_mask = cv2.morphologyEx(block_mask.astype(np.uint8), cv2.MORPH_CLOSE, kernel) > 0
        block_mask = _largest_component(block_mask)

    pixel_mask = np.repeat(np.repeat(block_mask, block_size, axis=0), block_size, axis=1)
    pixel_mask = pixel_mask[:height, :width]
    return pixel_mask.astype(bool), block_mask.astype(bool)


def fingerprint_foreground_mask(
    image: np.ndarray,
    block_size: int = 16,
    min_block_std: float = 6.0,
    threshold_scale: float = 0.55,
) -> np.ndarray:
    """Return only the full-resolution mask for callers that do not need blocks."""
    return segment_fingerprint(image, block_size, min_block_std, threshold_scale)[0]

