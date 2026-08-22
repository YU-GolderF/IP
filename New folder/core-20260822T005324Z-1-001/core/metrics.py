"""Algorithm-independent image-quality metrics for fair comparisons."""

from __future__ import annotations

import cv2
import numpy as np

from .preprocessing import to_grayscale


def image_contrast(image: np.ndarray) -> float:
    """RMS contrast (standard deviation) normalised to the 0..1 intensity range."""
    return float(np.std(to_grayscale(image).astype(np.float32) / 255.0))


def image_variance(image: np.ndarray) -> float:
    return float(np.var(to_grayscale(image).astype(np.float32)))


def laplacian_sharpness(image: np.ndarray) -> float:
    """Variance of the Laplacian, used as a focus/ridge-sharpness proxy."""
    gray = to_grayscale(image)
    return float(cv2.Laplacian(gray, cv2.CV_32F).var())


def edge_clarity(image: np.ndarray) -> float:
    """Mean Sobel gradient magnitude; this is not fingerprint matching accuracy."""
    gray = to_grayscale(image).astype(np.float32)
    gradient_x = cv2.Sobel(gray, cv2.CV_32F, 1, 0, ksize=3)
    gradient_y = cv2.Sobel(gray, cv2.CV_32F, 0, 1, ksize=3)
    return float(np.mean(cv2.magnitude(gradient_x, gradient_y)))


def structural_similarity(reference: np.ndarray, candidate: np.ndarray) -> float:
    """Calculate standard local SSIM for equal-sized reference and candidate images."""
    first = to_grayscale(reference).astype(np.float32)
    second = to_grayscale(candidate).astype(np.float32)
    if first.shape != second.shape:
        raise ValueError("SSIM requires reference and candidate images with equal dimensions")
    c1 = (0.01 * 255.0) ** 2
    c2 = (0.03 * 255.0) ** 2
    mu_first = cv2.GaussianBlur(first, (11, 11), 1.5)
    mu_second = cv2.GaussianBlur(second, (11, 11), 1.5)
    sigma_first = cv2.GaussianBlur(first * first, (11, 11), 1.5) - mu_first * mu_first
    sigma_second = cv2.GaussianBlur(second * second, (11, 11), 1.5) - mu_second * mu_second
    covariance = cv2.GaussianBlur(first * second, (11, 11), 1.5) - mu_first * mu_second
    numerator = (2 * mu_first * mu_second + c1) * (2 * covariance + c2)
    denominator = (mu_first * mu_first + mu_second * mu_second + c1) * (
        sigma_first + sigma_second + c2
    )
    return float(np.mean(numerator / np.maximum(denominator, 1e-12)))


def calculate_image_metrics(
    original: np.ndarray,
    processed: np.ndarray,
    reference: np.ndarray | None = None,
) -> dict[str, float | None]:
    """Compare input and enhanced quality; SSIM is only used with a real reference."""
    metrics: dict[str, float | None] = {
        "original_contrast": image_contrast(original),
        "processed_contrast": image_contrast(processed),
        "original_standard_deviation": float(np.std(to_grayscale(original))),
        "processed_standard_deviation": float(np.std(to_grayscale(processed))),
        "original_variance": image_variance(original),
        "processed_variance": image_variance(processed),
        "original_sharpness": laplacian_sharpness(original),
        "processed_sharpness": laplacian_sharpness(processed),
        "original_edge_clarity": edge_clarity(original),
        "processed_edge_clarity": edge_clarity(processed),
        "ssim": None,
    }
    if reference is not None:
        metrics["ssim"] = structural_similarity(reference, processed)
    return metrics

