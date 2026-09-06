"""
Core utility functions for the Hybrid Filter
"""

import numpy as np


def robust_rescale(
    image: np.ndarray,
    low_percentile: float = 1.0,
    high_percentile: float = 99.0,
    mask: np.ndarray | None = None,
) -> np.ndarray:
    """
    Rescale image using percentiles to handle outliers.
    """
    img = np.asarray(image, dtype=np.float32)
    
    if mask is not None:
        valid_pixels = img[mask]
        if valid_pixels.size == 0:
            return np.clip(img, 0, 255).astype(np.uint8)
    else:
        valid_pixels = img
        
    low = np.percentile(valid_pixels, low_percentile)
    high = np.percentile(valid_pixels, high_percentile)
    
    if high - low < 1e-6:
        return np.clip(img, 0, 255).astype(np.uint8)
    
    rescaled = (img - low) * (255.0 / (high - low))
    return np.clip(rescaled, 0, 255).astype(np.uint8)