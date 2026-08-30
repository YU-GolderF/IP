"""Gabor Filter Bank fingerprint enhancement — placeholder for Member 2.

This implements a classic orientation-adaptive Gabor filter bank approach
(inspired by Hong, Wan & Jain 1998) for fingerprint ridge enhancement.
Replace this file with your own implementation while keeping the same
run_algorithm() interface and return dict structure.
"""

from __future__ import annotations

from time import perf_counter

import cv2
import numpy as np

from core.metrics import calculate_image_metrics
from core.preprocessing import PreprocessingConfig, ensure_uint8, preprocess_with_stages, to_grayscale

# Borrow shared utilities from the RHLT package for segmentation, orientation,
# skeleton extraction and minutiae detection. These are general-purpose
# fingerprint analysis functions that any algorithm can reuse.
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

PIPELINE_BUILD = "gabor-filter-bank-v1"


def _gabor_enhance(
    gray: np.ndarray,
    orientation: np.ndarray,
    valid_blocks: np.ndarray,
    foreground: np.ndarray,
    block_size: int = 16,
    num_orientations: int = 8,
    kernel_size: int = 11,
    sigma: float = 4.0,
    wavelength: float = 8.0,
    gamma: float = 0.5,
) -> np.ndarray:
    """Apply a bank of Gabor filters steered by the local orientation field."""
    # Adapt parameters for small images to avoid over-blurring
    short_side = min(gray.shape[:2])
    if short_side < 160:
        kernel_size = min(kernel_size, 7)
        sigma = min(sigma, 1.5)
        wavelength = min(wavelength, 5.0)

    # Build the kernel bank
    kernels = []
    for i in range(num_orientations):
        theta = i * np.pi / num_orientations
        kernel = cv2.getGaborKernel(
            (kernel_size, kernel_size), sigma, theta + np.pi / 2.0,
            wavelength, gamma, 0, ktype=cv2.CV_32F,
        )
        kernel -= kernel.mean()
        norm = np.sum(np.abs(kernel))
        if norm > 1e-12:
            kernel /= norm
        kernels.append(kernel)

    # Quantise block orientations to the nearest kernel index
    height, width = gray.shape
    float_img = gray.astype(np.float32)
    result = np.zeros_like(float_img)
    pixel_bins = np.full(gray.shape, -1, dtype=np.int16)

    for by in range(valid_blocks.shape[0]):
        y0 = by * block_size
        y1 = min(height, y0 + block_size)
        for bx in range(valid_blocks.shape[1]):
            if not valid_blocks[by, bx]:
                continue
            x0 = bx * block_size
            x1 = min(width, x0 + block_size)
            angle = orientation[by, bx]
            idx = int(round((angle % np.pi) / np.pi * num_orientations)) % num_orientations
            pixel_bins[y0:y1, x0:x1] = idx

    # Filter and composite
    fg = foreground.astype(bool)
    for i, kernel in enumerate(kernels):
        mask = (pixel_bins == i) & fg
        if not np.any(mask):
            continue
        response = cv2.filter2D(float_img, cv2.CV_32F, kernel, borderType=cv2.BORDER_REFLECT)
        result[mask] = response[mask]

    # Normalise the Gabor response to produce the enhanced image
    fg_vals = result[fg]
    if fg_vals.size > 0:
        p2, p98 = float(np.percentile(fg_vals, 2)), float(np.percentile(fg_vals, 98))
        if p98 > p2 + 1.0:
            result = (result - p2) / (p98 - p2) * 255.0
    output = np.clip(result, 0, 255).astype(np.uint8)
    output[~fg] = gray[~fg]
    return output


def run_algorithm(
    image: np.ndarray,
    *,
    preprocessing_config: PreprocessingConfig | None = None,
) -> dict:
    """
    Enhance a fingerprint using a classic Gabor Filter Bank approach.

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

    # Orientation field
    raw_orientation, valid_blocks, orientation_coherence = estimate_orientation_field(
        preprocessed, foreground_mask, block_size=16,
    )
    orientation_field = smooth_orientation_field(raw_orientation, valid_blocks, sigma=1.0)
    orientation_vis = visualise_orientation_field(
        preprocessed, orientation_field, valid_blocks, block_size=16,
    )

    # Gabor enhancement
    warnings: list[str] = []
    gray = to_grayscale(original)
    if np.any(valid_blocks):
        enhanced = _gabor_enhance(
            preprocessed, orientation_field, valid_blocks, foreground_mask,
        )
    else:
        enhanced = gray.copy()
        warnings.append("No valid orientation blocks found; Gabor enhancement skipped.")

    # Post-processing: binarise, skeleton, minutiae
    ridge_binary = binarise_dark_ridges(enhanced, foreground_mask)
    ridge_binary = clean_binary(ridge_binary, min_component_area=10)
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
        "algorithm_name": "Gabor Filter Bank",
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
