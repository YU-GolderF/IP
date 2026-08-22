"""Local ridge-flow orientation estimation and doubled-angle smoothing."""

from __future__ import annotations

import cv2
import numpy as np

from core.preprocessing import to_grayscale


def estimate_orientation_field(
    image: np.ndarray,
    mask: np.ndarray,
    block_size: int = 16,
    minimum_mask_coverage: float = 0.20,
    minimum_gradient_energy: float = 10.0,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """
    Estimate ridge direction per block with the Sobel structure tensor.

    Sobel gradients point across a ridge. The accumulated 2*Gx*Gy and
    Gx^2-Gy^2 terms estimate that gradient direction; pi/2 converts it to the
    direction running along the ridge flow.
    """
    if block_size < 4:
        raise ValueError("block_size must be at least 4")
    gray = to_grayscale(image)
    foreground = np.asarray(mask, dtype=bool)
    if foreground.shape != gray.shape:
        raise ValueError("mask dimensions must match the image")

    float_image = gray.astype(np.float32) / 255.0
    gradient_x = cv2.Sobel(float_image, cv2.CV_32F, 1, 0, ksize=3, borderType=cv2.BORDER_REFLECT)
    gradient_y = cv2.Sobel(float_image, cv2.CV_32F, 0, 1, ksize=3, borderType=cv2.BORDER_REFLECT)

    height, width = gray.shape
    rows = max(1, int(np.ceil(height / block_size)))
    columns = max(1, int(np.ceil(width / block_size)))
    orientation = np.zeros((rows, columns), dtype=np.float32)
    valid = np.zeros((rows, columns), dtype=bool)
    coherence = np.zeros((rows, columns), dtype=np.float32)

    for block_y in range(rows):
        y_start = block_y * block_size
        y_end = min(height, y_start + block_size)
        for block_x in range(columns):
            x_start = block_x * block_size
            x_end = min(width, x_start + block_size)
            block_mask = foreground[y_start:y_end, x_start:x_end]
            if block_mask.size == 0 or float(block_mask.mean()) < minimum_mask_coverage:
                continue

            gx = gradient_x[y_start:y_end, x_start:x_end][block_mask]
            gy = gradient_y[y_start:y_end, x_start:x_end][block_mask]
            energy = float(np.mean(gx * gx + gy * gy))
            if not np.isfinite(energy) or energy < minimum_gradient_energy / (255.0 * 255.0):
                continue

            tensor_x = float(np.sum(gx * gx - gy * gy))
            tensor_y = float(2.0 * np.sum(gx * gy))
            tensor_strength = float(np.hypot(tensor_x, tensor_y))
            tensor_total = float(np.sum(gx * gx + gy * gy))
            if tensor_total <= 1e-12 or tensor_strength <= 1e-12:
                continue

            gradient_direction = 0.5 * np.arctan2(tensor_y, tensor_x)
            orientation[block_y, block_x] = float((gradient_direction + np.pi / 2.0) % np.pi)
            coherence[block_y, block_x] = float(np.clip(tensor_strength / tensor_total, 0.0, 1.0))
            valid[block_y, block_x] = True

    return orientation, valid, coherence


def smooth_orientation_field(
    orientation: np.ndarray,
    mask: np.ndarray,
    sigma: float = 1.0,
) -> np.ndarray:
    """Smooth pi-periodic ridge directions through sin(2*theta)/cos(2*theta)."""
    angles = np.asarray(orientation, dtype=np.float32)
    valid = np.asarray(mask, dtype=bool)
    if angles.shape != valid.shape:
        raise ValueError("orientation and mask must have equal dimensions")
    if sigma <= 0 or not np.any(valid):
        return np.where(valid, angles % np.pi, 0.0).astype(np.float32)

    weights = valid.astype(np.float32)
    cosine = np.cos(2.0 * angles) * weights
    sine = np.sin(2.0 * angles) * weights
    smooth_weights = cv2.GaussianBlur(weights, (0, 0), sigmaX=sigma, sigmaY=sigma)
    smooth_cosine = cv2.GaussianBlur(cosine, (0, 0), sigmaX=sigma, sigmaY=sigma)
    smooth_sine = cv2.GaussianBlur(sine, (0, 0), sigmaX=sigma, sigmaY=sigma)
    safe_weights = np.maximum(smooth_weights, 1e-6)
    smooth_cosine /= safe_weights
    smooth_sine /= safe_weights
    smoothed = 0.5 * np.arctan2(smooth_sine, smooth_cosine)
    smoothed = np.mod(smoothed, np.pi)
    return np.where(valid, smoothed, 0.0).astype(np.float32)


def visualise_orientation_field(
    image: np.ndarray,
    orientation: np.ndarray,
    mask: np.ndarray,
    block_size: int = 16,
    line_scale: float = 0.70,
) -> np.ndarray:
    """Draw ridge-direction line segments without changing the orientation data."""
    gray = to_grayscale(image)
    angles = np.asarray(orientation, dtype=np.float32)
    valid = np.asarray(mask, dtype=bool)
    if angles.shape != valid.shape:
        raise ValueError("orientation and mask must have equal dimensions")

    visualisation = cv2.cvtColor(gray, cv2.COLOR_GRAY2RGB)
    half_length = max(2.0, block_size * line_scale / 2.0)
    height, width = gray.shape
    for block_y, block_x in np.argwhere(valid):
        centre_x = min(width - 1, int(block_x * block_size + block_size / 2))
        centre_y = min(height - 1, int(block_y * block_size + block_size / 2))
        angle = float(angles[block_y, block_x])
        delta_x = int(round(np.cos(angle) * half_length))
        delta_y = int(round(np.sin(angle) * half_length))
        start = (centre_x - delta_x, centre_y - delta_y)
        end = (centre_x + delta_x, centre_y + delta_y)
        cv2.line(visualisation, start, end, (255, 64, 64), 1, cv2.LINE_AA)
    return visualisation

