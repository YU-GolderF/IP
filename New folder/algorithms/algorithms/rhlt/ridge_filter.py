"""Orientation-adaptive Gabor ridge restoration and mild ridge isolation."""

from __future__ import annotations

from functools import lru_cache

import cv2
import numpy as np

from core.preprocessing import to_grayscale

from .config import RHLTConfig
from .postprocess import binarise_dark_ridges, clean_binary


@lru_cache(maxsize=32)
def _gabor_kernel_bank(
    orientation_bins: int,
    kernel_size: int,
    sigma: float,
    wavelength: float,
    gamma: float,
    phase: float,
) -> tuple[np.ndarray, ...]:
    """Cache one conservative Gabor kernel per ridge-orientation bin."""
    kernels: list[np.ndarray] = []
    for index in range(orientation_bins):
        ridge_angle = index * np.pi / orientation_bins
        # OpenCV theta is the sinusoid normal, so add pi/2 to a ridge direction.
        kernel = cv2.getGaborKernel(
            (kernel_size, kernel_size),
            sigma,
            ridge_angle + np.pi / 2.0,
            wavelength,
            gamma,
            phase,
            ktype=cv2.CV_32F,
        )
        kernel -= float(kernel.mean())
        normaliser = float(np.sum(np.abs(kernel)))
        if normaliser > 1e-12:
            kernel /= normaliser
        kernels.append(kernel)
    return tuple(kernels)


def _block_orientation_bins(
    orientation: np.ndarray,
    valid_blocks: np.ndarray,
    orientation_bins: int,
) -> np.ndarray:
    indices = np.rint((np.mod(orientation, np.pi) / np.pi) * orientation_bins).astype(np.int32)
    indices %= orientation_bins
    return np.where(valid_blocks, indices, -1)


def enhance_ridges_with_gabor(
    image: np.ndarray,
    orientation: np.ndarray,
    mask: np.ndarray,
    config: RHLTConfig,
    valid_blocks: np.ndarray | None = None,
    base_image: np.ndarray | None = None,
) -> np.ndarray:
    """Restore ridge evidence while retaining detail from an unblurred base image."""
    gray = to_grayscale(image)
    base = gray if base_image is None else to_grayscale(base_image)
    if base.shape != gray.shape:
        raise ValueError("base_image dimensions must match the processed image")
    foreground = np.asarray(mask, dtype=bool)
    if foreground.shape != gray.shape:
        raise ValueError("mask dimensions must match the image")
    angles = np.asarray(orientation, dtype=np.float32)
    if valid_blocks is None:
        block_mask = np.zeros_like(angles, dtype=bool)
        for block_y in range(angles.shape[0]):
            y_start = block_y * config.block_size
            y_end = min(gray.shape[0], y_start + config.block_size)
            for block_x in range(angles.shape[1]):
                x_start = block_x * config.block_size
                x_end = min(gray.shape[1], x_start + config.block_size)
                local_mask = foreground[y_start:y_end, x_start:x_end]
                block_mask[block_y, block_x] = (
                    local_mask.size > 0 and float(local_mask.mean()) >= 0.20
                )
    else:
        block_mask = np.asarray(valid_blocks, dtype=bool)
    if angles.shape != block_mask.shape:
        raise ValueError("orientation and valid_blocks must have equal dimensions")
    if not np.any(foreground) or not np.any(block_mask):
        return base.copy()

    kernels = _gabor_kernel_bank(
        config.orientation_bins,
        config.gabor_kernel_size,
        config.gabor_sigma,
        config.gabor_lambda,
        config.gabor_gamma,
        config.gabor_psi,
    )
    block_bins = _block_orientation_bins(angles, block_mask, config.orientation_bins)
    height, width = gray.shape
    pixel_bins = np.full(gray.shape, -1, dtype=np.int16)
    for block_y in range(block_bins.shape[0]):
        y_start = block_y * config.block_size
        y_end = min(height, y_start + config.block_size)
        for block_x in range(block_bins.shape[1]):
            selected_bin = int(block_bins[block_y, block_x])
            if selected_bin < 0:
                continue
            x_start = block_x * config.block_size
            x_end = min(width, x_start + config.block_size)
            pixel_bins[y_start:y_end, x_start:x_end] = selected_bin

    analysis_image = gray.astype(np.float32)
    base_float = base.astype(np.float32)
    # Scale local-mean sigma proportionally to the image's short side so that
    # the subtraction does not over-blur small fingerprint images.
    short_side = min(gray.shape[:2])
    max_local_sigma = max(0.8, short_side / 40.0)
    local_sigma = min(max(1.0, config.gabor_lambda / 2.0), max_local_sigma)
    local_mean = cv2.GaussianBlur(
        analysis_image,
        (0, 0),
        sigmaX=local_sigma,
        sigmaY=local_sigma,
        borderType=cv2.BORDER_REFLECT,
    )
    centred = analysis_image - local_mean
    selected_response = np.zeros_like(analysis_image)
    for index, kernel in enumerate(kernels):
        selected_pixels = (pixel_bins == index) & foreground
        if not np.any(selected_pixels):
            continue
        response = cv2.filter2D(centred, cv2.CV_32F, kernel, borderType=cv2.BORDER_REFLECT)
        selected_response[selected_pixels] = response[selected_pixels]

    values = np.abs(selected_response[foreground])
    # Use p95 (not p99) so the ridge_delta values are larger and more impactful.
    response_scale = float(np.percentile(values, 95.0)) if values.size else 0.0
    if response_scale <= 1e-6:
        return base.copy()
    ridge_delta = np.clip(selected_response / response_scale, -1.0, 1.0)

    # ---- Three-stage aggressive enhancement ----
    # Stage 1: Strong unsharp mask for edge/ridge sharpening.
    usm_sigma = min(0.5, max(0.3, short_side / 300.0))
    usm_blur = cv2.GaussianBlur(
        base_float, (0, 0), sigmaX=usm_sigma, sigmaY=usm_sigma, borderType=cv2.BORDER_REFLECT
    )
    usm_amount = getattr(config, 'gabor_strength', 1.2) * 2.5
    sharpened = np.clip(base_float + usm_amount * (base_float - usm_blur), 0.0, 255.0)

    # Stage 2: Directional sigmoid push based on Gabor orientation response.
    # ridge_delta < 0  =>  ridge pixel (push toward black=0)
    # ridge_delta > 0  =>  valley pixel (push toward white=255)
    target = np.where(ridge_delta < 0, 0.0, 255.0)
    push_weight = np.abs(ridge_delta)  # confidence in direction (0..1)
    blend = getattr(config, 'gabor_blend_strength', 48.0)
    # Normalise blend to a 0..1 gain that scales the directional push.
    push_gain = min(1.0, blend / 100.0)
    pushed = sharpened + push_gain * push_weight * (target - sharpened)
    pushed = np.clip(pushed, 0.0, 255.0)

    # Stage 3: Per-foreground contrast stretch so the full 0-255 range is used.
    fg_vals = pushed[foreground]
    p2, p98 = float(np.percentile(fg_vals, 2.0)), float(np.percentile(fg_vals, 98.0))
    if p98 > p2 + 1.0:
        stretched = (pushed - p2) / (p98 - p2) * 255.0
        pushed = np.clip(stretched, 0.0, 255.0)

    output = pushed.astype(np.uint8)
    output[~foreground] = base[~foreground]
    return output


def detail_preserving_sharpen(
    image: np.ndarray,
    mask: np.ndarray,
    amount: float = 0.6,
) -> np.ndarray:
    """Apply mild unsharp masking without smoothing away fine ridge lines."""
    gray = to_grayscale(image)
    foreground = np.asarray(mask, dtype=bool)
    if foreground.shape != gray.shape:
        raise ValueError("mask dimensions must match the image")

    source = gray.astype(np.float32)
    local_average = cv2.GaussianBlur(
        source,
        (0, 0),
        sigmaX=0.5,
        sigmaY=0.5,
        borderType=cv2.BORDER_REFLECT,
    )
    detail = source - local_average
    sharpened = np.clip(source + amount * detail, 0, 255).astype(np.uint8)
    sharpened[~foreground] = gray[~foreground]
    return sharpened


def isolate_ridges(
    enhanced: np.ndarray,
    mask: np.ndarray,
    min_component_area: int = 20,
) -> np.ndarray:
    """Return a separate binary result with only mild connected-component cleanup."""
    binary = binarise_dark_ridges(to_grayscale(enhanced), np.asarray(mask, dtype=bool))
    return clean_binary(binary, min_component_area)
