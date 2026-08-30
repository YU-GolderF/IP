"""Local ridge-quality analysis and defect-aware fusion weights."""

from __future__ import annotations

import cv2
import numpy as np

from core.preprocessing import to_grayscale

from .config import RHLTConfig
from .frequency import expand_block_map


def local_ridge_quality_maps(
    image: np.ndarray,
    foreground_mask: np.ndarray,
    *,
    sigma: float = 3.0,
    target_contrast: float = 28.0,
    clear_region_protection: float = 1.5,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Return quality, weak-ridge and local-contrast maps in [0, 1]."""
    gray = to_grayscale(image).astype(np.float32)
    foreground = np.asarray(foreground_mask, dtype=bool)
    if foreground.shape != gray.shape:
        raise ValueError("foreground_mask dimensions must match the image")
    mean = cv2.GaussianBlur(gray, (0, 0), sigmaX=sigma, sigmaY=sigma, borderType=cv2.BORDER_REFLECT)
    mean_square = cv2.GaussianBlur(
        gray * gray, (0, 0), sigmaX=sigma, sigmaY=sigma, borderType=cv2.BORDER_REFLECT
    )
    local_std = np.sqrt(np.maximum(mean_square - mean * mean, 0.0))
    contrast = np.clip(local_std / max(float(target_contrast), 1e-6), 0.0, 1.0)
    quality = np.where(foreground, contrast, 0.0).astype(np.float32)
    weakness = np.power(np.clip(1.0 - contrast, 0.0, 1.0), clear_region_protection)
    weakness = np.where(foreground, weakness, 0.0).astype(np.float32)
    return quality, weakness, local_std.astype(np.float32)


def defect_aware_fusion_weight(
    weak_ridge_map: np.ndarray,
    foreground_mask: np.ndarray,
    orientation_coherence: np.ndarray,
    valid_orientation_blocks: np.ndarray,
    rhlt_edge: np.ndarray,
    settings: RHLTConfig,
) -> np.ndarray:
    """Build bounded Gabor support weights without suppressing weak ridges.

    weight = foreground * max_weight * weakness * coherence^gamma
             * (evidence_floor + (1-evidence_floor)*RHLT_edge^edge_gamma)

    The evidence floor lets coherent weak ridges receive support even when their
    RHLT magnitude is weak, while the RHLT term still modulates every weight.
    """
    weak = np.asarray(weak_ridge_map, dtype=np.float32)
    foreground = np.asarray(foreground_mask, dtype=bool)
    valid = np.asarray(valid_orientation_blocks, dtype=bool)
    coherence = np.asarray(orientation_coherence, dtype=np.float32)
    if weak.shape != foreground.shape or np.asarray(rhlt_edge).shape != foreground.shape:
        raise ValueError("pixel-level quality, RHLT and foreground maps must have equal shape")
    if coherence.shape != valid.shape:
        raise ValueError("orientation coherence and validity maps must have equal shape")

    reliable = valid & (coherence >= settings.minimum_orientation_coherence)
    coherence_blocks = np.where(reliable, coherence, 0.0)
    coherence_pixels = expand_block_map(coherence_blocks, settings.block_size, foreground.shape)
    direction = np.power(np.clip(coherence_pixels, 0.0, 1.0), settings.rhlt_support_gamma)
    edge = np.power(
        np.clip(np.asarray(rhlt_edge, dtype=np.float32) / 255.0, 0.0, 1.0),
        settings.rhlt_edge_gamma,
    )
    evidence = settings.rhlt_evidence_floor + (1.0 - settings.rhlt_evidence_floor) * edge
    defect = np.clip(weak * settings.weak_ridge_strength, 0.0, 1.0)
    weight = settings.hybrid_gabor_max_weight * defect * direction * evidence
    return np.where(foreground, np.clip(weight, 0.0, settings.hybrid_gabor_max_weight), 0.0).astype(np.float32)
