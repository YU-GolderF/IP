from __future__ import annotations

import cv2
import numpy as np


def estimate_local_orientation_for_dct(block: np.ndarray) -> tuple[float, float]:
    image = block.astype(np.float32)
    gx = cv2.Sobel(image, cv2.CV_32F, 1, 0, ksize=3)
    gy = cv2.Sobel(image, cv2.CV_32F, 0, 1, ksize=3)
    gxx, gyy, gxy = float(np.mean(gx * gx)), float(np.mean(gy * gy)), float(np.mean(gx * gy))
    orientation = 0.5 * np.arctan2(2.0 * gxy, gxx - gyy) + np.pi / 2.0
    coherence = np.sqrt((gxx - gyy) ** 2 + 4.0 * gxy * gxy) / max(gxx + gyy, 1e-9)
    return float(orientation % np.pi), float(np.clip(coherence, 0.0, 1.0))
