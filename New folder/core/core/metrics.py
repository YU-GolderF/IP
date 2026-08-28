"""Algorithm-independent image-quality metrics for fair comparisons."""

from __future__ import annotations

import cv2
import numpy as np

from .preprocessing import to_grayscale


def image_contrast(image: np.ndarray) -> float:
    """RMS contrast (standard deviation) normalised to the 0..1 intensity range."""
    return float(np.std(to_grayscale(image).astype(np.float32) / 255.0))


def image_entropy(image: np.ndarray) -> float:
    """Shannon entropy of the 8-bit grayscale histogram, in bits."""
    gray = to_grayscale(image)
    hist = np.bincount(gray.ravel().astype(np.uint8), minlength=256).astype(np.float64)
    probabilities = hist / max(hist.sum(), 1.0)
    probabilities = probabilities[probabilities > 0]
    return float(-(probabilities * np.log2(probabilities)).sum())


def _same_size_gray(reference: np.ndarray, candidate: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    first = to_grayscale(reference)
    second = to_grayscale(candidate)
    if first.shape != second.shape:
        second = cv2.resize(second, (first.shape[1], first.shape[0]), interpolation=cv2.INTER_AREA)
    return first, second


def mean_squared_error(reference: np.ndarray, candidate: np.ndarray) -> float:
    first, second = _same_size_gray(reference, candidate)
    return float(np.mean((first.astype(np.float64) - second.astype(np.float64)) ** 2))


def peak_signal_to_noise_ratio(reference: np.ndarray, candidate: np.ndarray) -> float:
    mse = mean_squared_error(reference, candidate)
    if mse <= 1e-12:
        return float("inf")
    return float(10.0 * np.log10((255.0**2) / mse))


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
    """Calculate standard local SSIM for reference and candidate images."""
    first, second = _same_size_gray(reference, candidate)
    first = first.astype(np.float32)
    second = second.astype(np.float32)
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


def ridge_valley_clarity(image: np.ndarray, mask: np.ndarray | None = None) -> float:
    """
    Ridge-Valley Clarity (RVC): measures how cleanly separated ridges and valleys are.

    Computed as the variance of the Laplacian response restricted to the fingerprint
    foreground. Higher values indicate sharper, better-separated ridge/valley transitions.
    """
    gray = to_grayscale(image).astype(np.float32)
    lap = cv2.Laplacian(gray, cv2.CV_32F)
    if mask is not None:
        fg = np.asarray(mask, dtype=bool)
        if fg.shape == gray.shape and np.any(fg):
            return float(np.var(lap[fg]))
    return float(np.var(lap))


def calculate_image_metrics(
    original: np.ndarray,
    processed: np.ndarray,
    reference: np.ndarray | None = None,
    foreground_mask: np.ndarray | None = None,
) -> dict[str, float | None]:
    """
    Compare original and enhanced quality with fingerprint-relevant derived metrics.

    Includes CII (Contrast Improvement Index), improvement percentages for sharpness
    and edge clarity, SSIM, PSNR, MSE, and Ridge-Valley Clarity.
    """
    orig_contrast = image_contrast(original)
    proc_contrast = image_contrast(processed)
    orig_sharpness = laplacian_sharpness(original)
    proc_sharpness = laplacian_sharpness(processed)
    orig_edge = edge_clarity(original)
    proc_edge = edge_clarity(processed)

    # Contrast Improvement Index: > 1.0 means contrast improved.
    cii = proc_contrast / max(orig_contrast, 1e-9)

    # Percentage improvements (positive = better).
    sharpness_improvement_pct = (proc_sharpness - orig_sharpness) / max(orig_sharpness, 1e-6) * 100.0
    edge_improvement_pct = (proc_edge - orig_edge) / max(orig_edge, 1e-6) * 100.0

    ssim_vs_original = structural_similarity(original, processed)

    metrics: dict[str, float | None] = {
        # ---- Contrast ----
        "original_contrast": orig_contrast,
        "processed_contrast": proc_contrast,
        "cii": float(cii),
        # ---- Sharpness / Edge ----
        "original_sharpness": orig_sharpness,
        "processed_sharpness": proc_sharpness,
        "sharpness_improvement_pct": float(sharpness_improvement_pct),
        "original_edge_clarity": orig_edge,
        "processed_edge_clarity": proc_edge,
        "edge_improvement_pct": float(edge_improvement_pct),
        # ---- Structural fidelity ----
        "ssim": ssim_vs_original,
        # ---- Full-image fidelity ----
        "mse": mean_squared_error(original, processed),
        "psnr": peak_signal_to_noise_ratio(original, processed),
        # ---- Fingerprint-specific ----
        "original_ridge_valley_clarity": ridge_valley_clarity(original, foreground_mask),
        "processed_ridge_valley_clarity": ridge_valley_clarity(processed, foreground_mask),
    }
    if reference is not None:
        metrics["ssim_reference"] = structural_similarity(reference, processed)
    return metrics

