"""Reusable preprocessing that preserves fingerprint ridge structure."""

from __future__ import annotations

from dataclasses import dataclass

import cv2
import numpy as np


@dataclass(frozen=True)
class PreprocessingConfig:
    """Conservative defaults suitable for later ridge-orientation estimation."""

    gaussian_kernel_size: int = 5
    gaussian_sigma: float = 0.6
    use_median_filter: bool = False
    median_kernel_size: int = 3
    normalisation_low_percentile: float = 1.0
    normalisation_high_percentile: float = 99.0
    clahe_clip_limit: float = 1.5
    clahe_grid_size: int = 8

    def validate(self) -> None:
        if self.gaussian_kernel_size < 1 or self.gaussian_kernel_size % 2 == 0:
            raise ValueError("gaussian_kernel_size must be a positive odd integer")
        if self.gaussian_sigma < 0:
            raise ValueError("gaussian_sigma must be non-negative")
        if self.median_kernel_size < 1 or self.median_kernel_size % 2 == 0:
            raise ValueError("median_kernel_size must be a positive odd integer")
        if not 0 <= self.normalisation_low_percentile < self.normalisation_high_percentile <= 100:
            raise ValueError("normalisation percentiles must satisfy 0 <= low < high <= 100")
        if self.clahe_clip_limit <= 0:
            raise ValueError("clahe_clip_limit must be positive")
        if self.clahe_grid_size < 1:
            raise ValueError("clahe_grid_size must be positive")


def ensure_uint8(image: np.ndarray) -> np.ndarray:
    """Convert a finite image array to uint8 while preserving relative contrast."""
    array = np.asarray(image)
    if array.size == 0:
        raise ValueError("image must not be empty")
    if array.ndim not in (2, 3):
        raise ValueError("image must be a 2-D grayscale or 3-D colour array")
    if array.ndim == 3 and array.shape[2] not in (1, 3, 4):
        raise ValueError("colour image must have 1, 3, or 4 channels")
    if not np.isfinite(array).all():
        raise ValueError("image contains NaN or infinite values")
    if array.dtype == np.uint8:
        return array.copy()

    values = array.astype(np.float32)
    low = float(values.min())
    high = float(values.max())
    if low >= 0.0 and high <= 1.0:
        return np.clip(values * 255.0, 0, 255).astype(np.uint8)
    if high <= low:
        return np.clip(values, 0, 255).astype(np.uint8)
    scaled = (values - low) * (255.0 / (high - low))
    return np.clip(scaled, 0, 255).astype(np.uint8)


def to_grayscale(image: np.ndarray) -> np.ndarray:
    """Convert a shared RGB/RGBA image representation to uint8 grayscale."""
    array = ensure_uint8(image)
    if array.ndim == 2:
        return array
    if array.shape[2] == 1:
        return array[:, :, 0]
    if array.shape[2] == 4:
        return cv2.cvtColor(array, cv2.COLOR_RGBA2GRAY)
    return cv2.cvtColor(array, cv2.COLOR_RGB2GRAY)


def percentile_normalise(gray: np.ndarray, low: float = 1.0, high: float = 99.0) -> np.ndarray:
    """Stretch robust intensity percentiles without amplifying extreme outliers."""
    image = to_grayscale(gray).astype(np.float32)
    lower, upper = np.percentile(image, [low, high])
    if upper <= lower + 1e-6:
        return image.astype(np.uint8)
    normalised = (image - lower) * (255.0 / (upper - lower))
    return np.clip(normalised, 0, 255).astype(np.uint8)


def gaussian_denoise(gray: np.ndarray, kernel_size: int = 5, sigma: float = 1.0) -> np.ndarray:
    """Apply mild Gaussian noise removal; sigma zero disables the operation."""
    image = to_grayscale(gray)
    if sigma <= 0:
        return image.copy()
    return cv2.GaussianBlur(
        image,
        (kernel_size, kernel_size),
        sigmaX=sigma,
        sigmaY=sigma,
        borderType=cv2.BORDER_REFLECT,
    )


def preprocess_with_stages(
    image: np.ndarray,
    config: PreprocessingConfig | None = None,
) -> dict[str, np.ndarray]:
    """Run shared preprocessing and return both the final image and useful stages."""
    settings = config or PreprocessingConfig()
    settings.validate()

    grayscale = to_grayscale(image)
    gaussian = gaussian_denoise(
        grayscale,
        settings.gaussian_kernel_size,
        settings.gaussian_sigma,
    )
    if settings.use_median_filter and settings.median_kernel_size > 1:
        denoised = cv2.medianBlur(gaussian, settings.median_kernel_size)
    else:
        denoised = gaussian.copy()
    normalised = percentile_normalise(
        denoised,
        settings.normalisation_low_percentile,
        settings.normalisation_high_percentile,
    )
    clahe = cv2.createCLAHE(
        clipLimit=settings.clahe_clip_limit,
        tileGridSize=(settings.clahe_grid_size, settings.clahe_grid_size),
    )
    enhanced = clahe.apply(normalised)
    return {
        "grayscale": grayscale,
        "gaussian_denoised": gaussian,
        "denoised": denoised,
        "normalised": normalised,
        "enhanced": enhanced,
    }


def preprocess_fingerprint(
    image: np.ndarray,
    config: PreprocessingConfig | None = None,
) -> np.ndarray:
    """Return the final shared-preprocessing image for any team algorithm."""
    return preprocess_with_stages(image, config)["enhanced"]

