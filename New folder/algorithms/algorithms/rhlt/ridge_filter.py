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
) -> np.ndarray:
    """Restore weak ridge evidence using the nearest local Gabor orientation."""
    gray = to_grayscale(image)
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
        return gray.copy()

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

    float_image = gray.astype(np.float32)
    local_mean = cv2.GaussianBlur(
        float_image,
        (0, 0),
        sigmaX=max(1.0, config.gabor_lambda / 2.0),
        sigmaY=max(1.0, config.gabor_lambda / 2.0),
        borderType=cv2.BORDER_REFLECT,
    )
    centred = float_image - local_mean
    selected_response = np.zeros_like(float_image)
    for index, kernel in enumerate(kernels):
        selected_pixels = (pixel_bins == index) & foreground
        if not np.any(selected_pixels):
            continue
        response = cv2.filter2D(centred, cv2.CV_32F, kernel, borderType=cv2.BORDER_REFLECT)
        selected_response[selected_pixels] = response[selected_pixels]

    values = np.abs(selected_response[foreground])
    response_scale = float(np.percentile(values, 99.0)) if values.size else 0.0
    if response_scale <= 1e-6:
        return gray.copy()
    ridge_delta = np.clip(selected_response / response_scale, -1.0, 1.0)
    restored = float_image + config.gabor_strength * 32.0 * ridge_delta
    output = np.clip(restored, 0, 255).astype(np.uint8)
    output[~foreground] = gray[~foreground]
    return output


def isolate_ridges(
    enhanced: np.ndarray,
    mask: np.ndarray,
    min_component_area: int = 20,
) -> np.ndarray:
    """Return a separate binary result with only mild connected-component cleanup."""
    binary = binarise_dark_ridges(to_grayscale(enhanced), np.asarray(mask, dtype=bool))
    return clean_binary(binary, min_component_area)
