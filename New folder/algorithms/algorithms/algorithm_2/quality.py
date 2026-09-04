from __future__ import annotations

import numpy as np


def compute_block_quality(block: np.ndarray, roi_fraction: float, orientation_confidence: float, frequency_confidence: float) -> float:
    """ROI/contrast quality; proposed confidence is applied separately."""
    _ = orientation_confidence, frequency_confidence
    contrast = float(np.std(block.astype(np.float32)) / 64.0)
    return float(np.clip(roi_fraction * min(contrast, 1.0), 0.0, 1.0))
