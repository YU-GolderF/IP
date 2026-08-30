"""Explainable local fingerprint ridge-wavelength estimation."""

from __future__ import annotations

import cv2
import numpy as np

from core.preprocessing import to_grayscale


def _projection_wavelength(
    patch: np.ndarray,
    ridge_angle: float,
    minimum_wavelength: float,
    maximum_wavelength: float,
) -> tuple[float, float]:
    """Estimate spacing from the autocorrelation of a ridge-normal projection."""
    height, width = patch.shape
    centre = (width / 2.0, height / 2.0)
    # Align ridge flow vertically; the column profile then varies across ridges.
    rotation = 90.0 - float(np.degrees(ridge_angle))
    matrix = cv2.getRotationMatrix2D(centre, rotation, 1.0)
    aligned = cv2.warpAffine(
        patch.astype(np.float32),
        matrix,
        (width, height),
        flags=cv2.INTER_LINEAR,
        borderMode=cv2.BORDER_REFLECT,
    )
    margin = max(1, min(height, width) // 8)
    core = aligned[margin : height - margin, margin : width - margin]
    if min(core.shape, default=0) < 4:
        return 0.0, 0.0
    profile = np.mean(core, axis=0)
    profile -= float(np.mean(profile))
    profile_std = float(np.std(profile))
    if profile_std < 1.5:
        return 0.0, 0.0
    profile /= profile_std
    autocorrelation = np.correlate(profile, profile, mode="full")[profile.size - 1 :]
    if autocorrelation[0] <= 1e-6:
        return 0.0, 0.0
    autocorrelation /= autocorrelation[0]

    minimum_lag = max(2, int(np.ceil(minimum_wavelength)))
    maximum_lag = min(int(np.floor(maximum_wavelength)), autocorrelation.size - 2)
    if maximum_lag < minimum_lag:
        return 0.0, 0.0
    candidates: list[int] = []
    for lag in range(minimum_lag, maximum_lag + 1):
        if autocorrelation[lag] >= autocorrelation[lag - 1] and autocorrelation[lag] > autocorrelation[lag + 1]:
            candidates.append(lag)
    if not candidates:
        return 0.0, 0.0
    best_lag = max(candidates, key=lambda lag: float(autocorrelation[lag]))
    confidence = float(autocorrelation[best_lag])
    if confidence < 0.15:
        return 0.0, confidence
    return float(best_lag), float(np.clip(confidence, 0.0, 1.0))


def estimate_local_ridge_wavelength(
    image: np.ndarray,
    orientation: np.ndarray,
    valid_orientation_blocks: np.ndarray,
    foreground_mask: np.ndarray,
    *,
    orientation_block_size: int = 16,
    analysis_block_size: int = 24,
    minimum_wavelength: float = 3.0,
    maximum_wavelength: float = 14.0,
    smoothing_size: int = 3,
    fallback_wavelength: float = 10.0,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Return block wavelength, validity and confidence maps.

    Each orientation block is analysed with a larger centred window. Invalid or
    unstable estimates are explicitly marked false and receive the documented
    fallback wavelength; no random value is generated.
    """
    gray = to_grayscale(image)
    angles = np.asarray(orientation, dtype=np.float32)
    valid_orientation = np.asarray(valid_orientation_blocks, dtype=bool)
    foreground = np.asarray(foreground_mask, dtype=bool)
    if angles.shape != valid_orientation.shape:
        raise ValueError("orientation and valid_orientation_blocks must have equal shape")
    if foreground.shape != gray.shape:
        raise ValueError("foreground_mask dimensions must match the image")
    if analysis_block_size < 8 or orientation_block_size < 4:
        raise ValueError("frequency analysis windows are too small")
    if not 2.0 <= minimum_wavelength < maximum_wavelength:
        raise ValueError("ridge wavelength limits are invalid")

    rows, columns = angles.shape
    wavelength = np.full((rows, columns), float(fallback_wavelength), dtype=np.float32)
    valid = np.zeros((rows, columns), dtype=bool)
    confidence = np.zeros((rows, columns), dtype=np.float32)
    radius = analysis_block_size // 2
    height, width = gray.shape
    # Short images safely support periods down to 2 px and no more than half
    # the analysis window; the configured scientific range remains the cap.
    local_min = max(2.0, min(float(minimum_wavelength), analysis_block_size / 3.0))
    local_max = min(float(maximum_wavelength), max(local_min + 1.0, analysis_block_size / 2.0))

    for by, bx in np.argwhere(valid_orientation):
        centre_y = min(height - 1, int(by * orientation_block_size + orientation_block_size / 2))
        centre_x = min(width - 1, int(bx * orientation_block_size + orientation_block_size / 2))
        y0, y1 = max(0, centre_y - radius), min(height, centre_y + radius)
        x0, x1 = max(0, centre_x - radius), min(width, centre_x + radius)
        local_mask = foreground[y0:y1, x0:x1]
        if local_mask.size == 0 or float(local_mask.mean()) < 0.45:
            continue
        patch = gray[y0:y1, x0:x1]
        if min(patch.shape) < 8:
            continue
        estimate, score = _projection_wavelength(
            patch, float(angles[by, bx]), local_min, local_max
        )
        confidence[by, bx] = score
        if estimate > 0:
            wavelength[by, bx] = estimate
            valid[by, bx] = True

    if smoothing_size > 1 and np.any(valid):
        radius_blocks = smoothing_size // 2
        smoothed = wavelength.copy()
        for by, bx in np.argwhere(valid):
            y0, y1 = max(0, by - radius_blocks), min(rows, by + radius_blocks + 1)
            x0, x1 = max(0, bx - radius_blocks), min(columns, bx + radius_blocks + 1)
            neighbours = wavelength[y0:y1, x0:x1][valid[y0:y1, x0:x1]]
            if neighbours.size:
                smoothed[by, bx] = float(np.median(neighbours))
        wavelength = smoothed
    return wavelength, valid, confidence


def expand_block_map(block_map: np.ndarray, block_size: int, image_shape: tuple[int, int]) -> np.ndarray:
    """Expand block values to an exact full-resolution pixel map."""
    expanded = np.repeat(np.repeat(np.asarray(block_map), block_size, axis=0), block_size, axis=1)
    return expanded[: image_shape[0], : image_shape[1]]
