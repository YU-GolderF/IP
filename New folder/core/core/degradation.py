"""Deterministic in-memory fingerprint degradation for reference experiments."""

from __future__ import annotations

from dataclasses import dataclass

import cv2
import numpy as np

from .preprocessing import to_grayscale


@dataclass(frozen=True)
class DegradationConfig:
    blur_sigma: float
    contrast_factor: float
    noise_std: float
    seed: int = 7

    def validate(self) -> None:
        if self.blur_sigma < 0 or self.noise_std < 0:
            raise ValueError("blur_sigma and noise_std must be non-negative")
        if not 0 < self.contrast_factor <= 1.0:
            raise ValueError("contrast_factor must be in (0, 1]")


DEGRADATION_PRESETS = {
    "Mild": DegradationConfig(0.8, 0.80, 5.0),
    "Medium": DegradationConfig(1.5, 0.60, 10.0),
    "Severe": DegradationConfig(2.2, 0.45, 15.0),
}


def degrade_fingerprint(
    image: np.ndarray,
    level: str = "Medium",
    *,
    seed: int | None = None,
) -> np.ndarray:
    """Create reproducible blur/contrast/noise degradation without file writes."""
    if level not in DEGRADATION_PRESETS:
        raise ValueError(f"unknown degradation level: {level}")
    preset = DEGRADATION_PRESETS[level]
    settings = DegradationConfig(
        preset.blur_sigma,
        preset.contrast_factor,
        preset.noise_std,
        preset.seed if seed is None else int(seed),
    )
    settings.validate()
    gray = to_grayscale(image).astype(np.float32)
    if settings.blur_sigma > 0:
        gray = cv2.GaussianBlur(
            gray, (0, 0), sigmaX=settings.blur_sigma, sigmaY=settings.blur_sigma,
            borderType=cv2.BORDER_REFLECT,
        )
    centre = float(np.mean(gray))
    degraded = centre + settings.contrast_factor * (gray - centre)
    rng = np.random.default_rng(settings.seed)
    degraded += rng.normal(0.0, settings.noise_std, degraded.shape).astype(np.float32)
    return np.clip(degraded, 0, 255).astype(np.uint8)
