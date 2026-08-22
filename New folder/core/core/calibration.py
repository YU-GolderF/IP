"""Simple, explainable image calibration without invented physical scale."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Sequence

import cv2
import numpy as np

from .preprocessing import ensure_uint8


@dataclass(frozen=True)
class CalibrationConfig:
    max_width: int | None = 1024
    max_height: int | None = 1024
    allow_upscale: bool = False
    rotation_degrees: float = 0.0
    rectification_points: tuple[tuple[float, float], ...] | None = None
    rectified_size: tuple[int, int] | None = None

    def validate(self) -> None:
        for name, value in (("max_width", self.max_width), ("max_height", self.max_height)):
            if value is not None and value < 1:
                raise ValueError(f"{name} must be positive when provided")
        if self.rectification_points is not None and len(self.rectification_points) != 4:
            raise ValueError("rectification_points must contain four corner points")
        if self.rectified_size is not None and min(self.rectified_size) < 1:
            raise ValueError("rectified_size values must be positive")


def resize_preserve_aspect_ratio(
    image: np.ndarray,
    max_width: int | None = None,
    max_height: int | None = None,
    allow_upscale: bool = False,
) -> np.ndarray:
    """Resize within a bounding box while preserving the original aspect ratio."""
    array = ensure_uint8(image)
    height, width = array.shape[:2]
    width_scale = max_width / width if max_width is not None else float("inf")
    height_scale = max_height / height if max_height is not None else float("inf")
    scale = min(width_scale, height_scale)
    if not allow_upscale:
        scale = min(scale, 1.0)
    if not np.isfinite(scale) or abs(scale - 1.0) < 1e-9:
        return array.copy()
    new_width = max(1, int(round(width * scale)))
    new_height = max(1, int(round(height * scale)))
    interpolation = cv2.INTER_CUBIC if scale > 1.0 else cv2.INTER_AREA
    return cv2.resize(array, (new_width, new_height), interpolation=interpolation)


def rotate_image(image: np.ndarray, angle_degrees: float, expand: bool = False) -> np.ndarray:
    """Rotate around the image centre; rotation is user-controlled, not guessed."""
    array = ensure_uint8(image)
    if abs(angle_degrees) < 1e-9:
        return array.copy()
    height, width = array.shape[:2]
    centre = (width / 2.0, height / 2.0)
    matrix = cv2.getRotationMatrix2D(centre, angle_degrees, 1.0)
    output_width, output_height = width, height
    if expand:
        cosine = abs(matrix[0, 0])
        sine = abs(matrix[0, 1])
        output_width = int(np.ceil(height * sine + width * cosine))
        output_height = int(np.ceil(height * cosine + width * sine))
        matrix[0, 2] += output_width / 2.0 - centre[0]
        matrix[1, 2] += output_height / 2.0 - centre[1]
    return cv2.warpAffine(
        array,
        matrix,
        (output_width, output_height),
        flags=cv2.INTER_LINEAR,
        borderMode=cv2.BORDER_REFLECT,
    )


def rectify_image(
    image: np.ndarray,
    source_points: Sequence[Sequence[float]],
    output_size: tuple[int, int] | None = None,
) -> np.ndarray:
    """Apply optional four-corner perspective rectification."""
    array = ensure_uint8(image)
    points = np.asarray(source_points, dtype=np.float32)
    if points.shape != (4, 2):
        raise ValueError("source_points must have shape (4, 2): TL, TR, BR, BL")
    if output_size is None:
        top = np.linalg.norm(points[1] - points[0])
        bottom = np.linalg.norm(points[2] - points[3])
        left = np.linalg.norm(points[3] - points[0])
        right = np.linalg.norm(points[2] - points[1])
        output_size = (max(1, int(round(max(top, bottom)))), max(1, int(round(max(left, right)))))
    width, height = output_size
    destination = np.array(
        [[0, 0], [width - 1, 0], [width - 1, height - 1], [0, height - 1]],
        dtype=np.float32,
    )
    transform = cv2.getPerspectiveTransform(points, destination)
    return cv2.warpPerspective(array, transform, (width, height), borderMode=cv2.BORDER_REFLECT)


def calibrate_image(
    image: np.ndarray,
    config: CalibrationConfig | None = None,
) -> dict[str, object]:
    """Calibrate dimensions and optional geometry while recording what changed."""
    settings = config or CalibrationConfig()
    settings.validate()
    original = ensure_uint8(image)
    calibrated = original.copy()
    if settings.rectification_points is not None:
        calibrated = rectify_image(calibrated, settings.rectification_points, settings.rectified_size)
    calibrated = rotate_image(calibrated, settings.rotation_degrees)
    calibrated = resize_preserve_aspect_ratio(
        calibrated,
        settings.max_width,
        settings.max_height,
        settings.allow_upscale,
    )
    return {
        "image": calibrated,
        "original_dimensions": (int(original.shape[1]), int(original.shape[0])),
        "processed_dimensions": (int(calibrated.shape[1]), int(calibrated.shape[0])),
        "rotation_degrees": float(settings.rotation_degrees),
        "rectified": settings.rectification_points is not None,
    }

