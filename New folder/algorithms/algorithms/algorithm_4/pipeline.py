"""Median Filter + Unsharp Mask fingerprint enhancement — placeholder for Member 4.

This implements a classic spatial-domain approach: median filtering for noise
removal followed by unsharp masking for ridge sharpening. Replace this file
with your own implementation while keeping the same run_algorithm() interface.
"""

from __future__ import annotations

from time import perf_counter

import cv2
import numpy as np

from core.metrics import calculate_image_metrics
from core.preprocessing import PreprocessingConfig, ensure_uint8, preprocess_with_stages, to_grayscale

from algorithms.rhlt.orientation import (
    estimate_orientation_field,
    smooth_orientation_field,
    visualise_orientation_field,
)
from algorithms.rhlt.postprocess import (
    binarise_dark_ridges,
    clean_binary,
    crossing_number_minutiae,
    make_skeleton,
    minutiae_overlay,
)
from algorithms.rhlt.segmentation import segment_fingerprint
from algorithms.rhlt.metrics import metric_bundle

PIPELINE_BUILD = "median-unsharp-mask-v1"


def _median_unsharp_enhance(gray: np.ndarray, foreground: np.ndarray) -> np.ndarray:
    """Apply median denoising followed by aggressive unsharp masking."""
    fg = foreground.astype(bool)

    # Stage 1: Median filter for salt-and-pepper noise removal
    denoised = cv2.medianBlur(gray, 5)

    # Stage 2: Bilateral filter to preserve edges while smoothing
    smoothed = cv2.bilateralFilter(denoised, d=7, sigmaColor=50, sigmaSpace=50)

    # Stage 3: Unsharp mask (strong)
    blurred = cv2.GaussianBlur(smoothed.astype(np.float32), (0, 0), 2.0)
    sharpened = cv2.addWeighted(
        smoothed.astype(np.float32), 2.0,
        blurred, -1.0,
        0,
    )
    output = np.clip(sharpened, 0, 255).astype(np.uint8)

    # Stage 4: Contrast stretch on foreground only
    fg_vals = output[fg]
    if fg_vals.size > 0:
        p2, p98 = float(np.percentile(fg_vals, 2)), float(np.percentile(fg_vals, 98))
        if p98 > p2 + 1.0:
            stretched = (output.astype(np.float32) - p2) / (p98 - p2) * 255.0
            output = np.clip(stretched, 0, 255).astype(np.uint8)

    result = gray.copy()
    result[fg] = output[fg]
    return result


def run_algorithm(
    image: np.ndarray,
    *,
    preprocessing_config: PreprocessingConfig | None = None,
) -> dict:
    """
    Enhance a fingerprint using Median Filter + Unsharp Mask.

    Returns a dict with the standard algorithm result structure.
    """
    started = perf_counter()
    original = ensure_uint8(image)
    config = preprocessing_config or PreprocessingConfig()
    config.validate()

    preprocessing_stages = preprocess_with_stages(original, config)
    preprocessed = preprocessing_stages["enhanced"]

    # Segmentation
    foreground_mask, segmentation_blocks = segment_fingerprint(
        preprocessed, block_size=16, min_block_std=5.0,
    )

    # Orientation field (for consistent comparison metrics)
    raw_orientation, valid_blocks, orientation_coherence = estimate_orientation_field(
        preprocessed, foreground_mask, block_size=16,
    )
    orientation_field = smooth_orientation_field(raw_orientation, valid_blocks, sigma=1.0)
    orientation_vis = visualise_orientation_field(
        preprocessed, orientation_field, valid_blocks, block_size=16,
    )

    # Enhancement
    warnings: list[str] = []
    gray = to_grayscale(original)
    enhanced = _median_unsharp_enhance(preprocessed, foreground_mask)

    # Post-processing
    ridge_binary = binarise_dark_ridges(enhanced, foreground_mask)
    ridge_binary = clean_binary(ridge_binary, min_area=10)
    skeleton = make_skeleton(ridge_binary)
    endings, bifurcations = crossing_number_minutiae(skeleton, foreground_mask, border=10, min_distance=8)
    overlay = minutiae_overlay(enhanced, endings, bifurcations)

    elapsed_ms = (perf_counter() - started) * 1000.0
    metrics = calculate_image_metrics(original, enhanced, foreground_mask=foreground_mask)
    metrics.update(metric_bundle(enhanced, foreground_mask, len(endings), len(bifurcations), elapsed_ms))
    metrics["foreground_coverage_percent"] = float(foreground_mask.mean() * 100.0)
    metrics["valid_orientation_blocks"] = int(valid_blocks.sum())
    if np.any(valid_blocks):
        metrics["mean_orientation_coherence"] = float(orientation_coherence[valid_blocks].mean())
    else:
        metrics["mean_orientation_coherence"] = 0.0

    return {
        "pipeline_build": PIPELINE_BUILD,
        "algorithm_name": "Median + Unsharp Mask",
        "status": "warning" if warnings else "ok",
        "warnings": warnings,
        "original": original,
        "grayscale": preprocessing_stages["grayscale"],
        "preprocessing_stages": preprocessing_stages,
        "preprocessed": preprocessed,
        "normalised": preprocessing_stages["normalised"],
        "denoised": preprocessing_stages["denoised"],
        "foreground_mask": foreground_mask,
        "mask": foreground_mask,
        "orientation_field": orientation_field,
        "orientation_block_mask": valid_blocks,
        "orientation_coherence": orientation_coherence,
        "orientation_visualisation": orientation_vis,
        "ridge_restored": enhanced,
        "enhanced_image": enhanced,
        "ridge_enhanced": enhanced,
        "ridge_binary": ridge_binary,
        "binary": ridge_binary,
        "skeleton": skeleton,
        "endings": endings,
        "bifurcations": bifurcations,
        "minutiae_overlay": overlay,
        "metrics": metrics,
        "processing_time_ms": float(elapsed_ms),
    }
