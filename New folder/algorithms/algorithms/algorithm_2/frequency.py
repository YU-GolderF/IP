from __future__ import annotations

import cv2
import numpy as np


def estimate_local_frequency(block: np.ndarray, orientation: float, low: float, high: float) -> tuple[float, float]:
    coefficient = cv2.dct(block.astype(np.float32) / 255.0)
    size = block.shape[0]; yy, xx = np.mgrid[:size, :size]
    radial = np.sqrt(xx * xx + yy * yy) / max(size, 1)
    angle = np.arctan2(yy, xx); normal = (orientation - np.pi / 2.0) % np.pi
    angular = np.abs(np.cos(angle - normal))
    valid = (radial >= low) & (radial <= high) & (angular >= np.cos(np.deg2rad(25.0)))
    energy = coefficient * coefficient * valid; total = float(energy.sum())
    if total <= 1e-12:
        return (low + high) / 2.0, 0.0
    frequency = float((radial * energy).sum() / total)
    confidence = float(total / max(float((coefficient * coefficient).sum()), 1e-12))
    return frequency, float(np.clip(confidence, 0.0, 1.0))
