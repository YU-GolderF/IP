from __future__ import annotations

import cv2
import numpy as np


def information_entropy(gray: np.ndarray, mask: np.ndarray | None = None) -> float:
    values = gray[mask] if mask is not None and np.any(mask) else gray.ravel()
    hist = np.bincount(values.astype(np.uint8), minlength=256).astype(np.float64)
    p = hist / max(hist.sum(), 1.0)
    p = p[p > 0]
    return float(-(p * np.log2(p)).sum())


def average_gradient(gray: np.ndarray, mask: np.ndarray | None = None) -> float:
    f = gray.astype(np.float32)
    gx = cv2.Sobel(f, cv2.CV_32F, 1, 0, ksize=3)
    gy = cv2.Sobel(f, cv2.CV_32F, 0, 1, ksize=3)
    mag = np.sqrt(gx * gx + gy * gy)
    values = mag[mask] if mask is not None and np.any(mask) else mag.ravel()
    return float(np.mean(values))


def local_contrast_score(gray: np.ndarray, mask: np.ndarray | None = None, sigma: float = 3.0) -> float:
    """Mean local standard deviation, normalised to approximately 0..1."""
    f = gray.astype(np.float32) / 255.0
    mean = cv2.GaussianBlur(f, (0, 0), sigmaX=sigma, sigmaY=sigma, borderType=cv2.BORDER_REFLECT)
    mean2 = cv2.GaussianBlur(f * f, (0, 0), sigmaX=sigma, sigmaY=sigma, borderType=cv2.BORDER_REFLECT)
    std = np.sqrt(np.maximum(mean2 - mean * mean, 0.0))
    values = std[mask] if mask is not None and np.any(mask) else std.ravel()
    return float(np.mean(values))


def ridge_sharpness_score(gray: np.ndarray, mask: np.ndarray | None = None) -> float:
    """Normalised average Sobel gradient used as a reproducible ridge sharpness proxy."""
    return average_gradient(gray, mask) / 255.0


def metric_bundle(
    gray: np.ndarray,
    mask: np.ndarray,
    endings_count: int,
    bifurcations_count: int,
    processing_ms: float,
) -> dict[str, float | int]:
    return {
        "local_contrast": local_contrast_score(gray, mask),
        "ridge_sharpness": ridge_sharpness_score(gray, mask),
        "information_entropy": information_entropy(gray, mask),
        "average_gradient": average_gradient(gray, mask),
        "ridge_endings": int(endings_count),
        "bifurcations": int(bifurcations_count),
        "minutiae_total": int(endings_count + bifurcations_count),
        "processing_time_ms": float(processing_ms),
        "foreground_ratio": float(mask.mean()),
    }
