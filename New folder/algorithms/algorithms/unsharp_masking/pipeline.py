"""Unsharp Masking fingerprint enhancement — Member 4.

Implements conventional linear Unsharp Masking as the baseline method,
with Adaptive and Nonlinear Polynomial Unsharp Masking methods evaluated
as enhancement candidates.
"""

from __future__ import annotations

from time import perf_counter

import cv2
import numpy as np

from core.metrics import calculate_image_metrics
from core.preprocessing import (
    PreprocessingConfig,
    ensure_uint8,
    preprocess_with_stages,
)

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

PIPELINE_BUILD = "polynomial-unsharp-mask-v1"


# ============================================================
# Conventional Unsharp Masking
# ============================================================


def _conventional_unsharp_enhance(
    gray: np.ndarray,
    foreground: np.ndarray,
    lambda_value: float = 1.0,
) -> np.ndarray:
    """
    Apply conventional linear Unsharp Masking.

    Based on Polesel et al. (2000):

        y(n,m) = x(n,m) + lambda * z(n,m)

    where:

        z(n,m) = 4x(n,m)
                 - x(n-1,m)
                 - x(n+1,m)
                 - x(n,m-1)
                 - x(n,m+1)
    """

    fg = foreground.astype(bool)
    x = gray.astype(np.float32)

    high_pass_kernel = np.array(
        [
            [0, -1, 0],
            [-1, 4, -1],
            [0, -1, 0],
        ],
        dtype=np.float32,
    )

    z = cv2.filter2D(
        x,
        ddepth=cv2.CV_32F,
        kernel=high_pass_kernel,
        borderType=cv2.BORDER_REFLECT,
    )

    y = x + lambda_value * z
    y = np.clip(y, 0, 255).astype(np.uint8)

    result = gray.copy()
    result[fg] = y[fg]

    return result


# ============================================================
# Adaptive Unsharp Masking
# ============================================================


def _adaptive_unsharp_enhance(
    gray: np.ndarray,
    foreground: np.ndarray,
    tau1: float = 60.0,
    tau2: float = 200.0,
    alpha_dl: float = 3.0,
    alpha_dh: float = 4.0,
    mu: float = 0.1,
    beta: float = 0.5,
) -> np.ndarray:
    """
    Apply Adaptive Directional Unsharp Masking.

    Based on Polesel et al. (2000).
    This function implements the first enhancement candidate.
    """

    fg = foreground.astype(bool)
    x = gray.astype(np.float32)

    # Equation (3)
    horizontal_kernel = np.array(
        [
            [0, 0, 0],
            [-1, 2, -1],
            [0, 0, 0],
        ],
        dtype=np.float32,
    )

    # Equation (4)
    vertical_kernel = np.array(
        [
            [0, -1, 0],
            [0, 2, 0],
            [0, -1, 0],
        ],
        dtype=np.float32,
    )

    z_x = cv2.filter2D(
        x,
        ddepth=cv2.CV_32F,
        kernel=horizontal_kernel,
        borderType=cv2.BORDER_REFLECT,
    )

    z_y = cv2.filter2D(
        x,
        ddepth=cv2.CV_32F,
        kernel=vertical_kernel,
        borderType=cv2.BORDER_REFLECT,
    )

    local_mean = cv2.boxFilter(
        x,
        ddepth=cv2.CV_32F,
        ksize=(3, 3),
        normalize=True,
        borderType=cv2.BORDER_REFLECT,
    )

    local_mean_squared = cv2.boxFilter(
        x * x,
        ddepth=cv2.CV_32F,
        ksize=(3, 3),
        normalize=True,
        borderType=cv2.BORDER_REFLECT,
    )

    local_variance = local_mean_squared - (local_mean * local_mean)
    local_variance = np.maximum(local_variance, 0.0)

    medium_region = (local_variance >= tau1) & (local_variance < tau2) & fg

    high_region = (local_variance >= tau2) & fg

    # Equation (12)
    alpha = np.ones_like(x, dtype=np.float32)

    alpha[medium_region] = alpha_dh
    alpha[high_region] = alpha_dl

    # Figure 2 high-pass operator
    dynamics_kernel = np.array(
        [
            [-1, -1, -1],
            [-1, 8, -1],
            [-1, -1, -1],
        ],
        dtype=np.float32,
    )

    g_x = cv2.filter2D(
        x,
        ddepth=cv2.CV_32F,
        kernel=dynamics_kernel,
        borderType=cv2.BORDER_REFLECT,
    )

    g_zx = cv2.filter2D(
        z_x,
        ddepth=cv2.CV_32F,
        kernel=dynamics_kernel,
        borderType=cv2.BORDER_REFLECT,
    )

    g_zy = cv2.filter2D(
        z_y,
        ddepth=cv2.CV_32F,
        kernel=dynamics_kernel,
        borderType=cv2.BORDER_REFLECT,
    )

    # Equation (11)
    g_d = alpha * g_x

    height, width = x.shape

    adaptive_output = x.copy()

    lambda_vector = np.zeros(2, dtype=np.float32)

    # Stable initial autocorrelation matrix
    R = np.eye(2, dtype=np.float32)

    epsilon = 1e-6

    for n in range(height):
        for m in range(width):

            # Equation (15)
            G = np.array(
                [
                    g_zx[n, m],
                    g_zy[n, m],
                ],
                dtype=np.float32,
            )

            # Equation (14)
            g_y = g_x[n, m] + float(np.dot(lambda_vector, G))

            e = g_d[n, m] - g_y

            # Equation (5)
            adaptive_output[n, m] = (
                x[n, m] + lambda_vector[0] * z_x[n, m] + lambda_vector[1] * z_y[n, m]
            )

            # Equation (17)
            R = (1.0 - beta) * R + beta * np.outer(G, G)

            # Numerical implementation of Equation (16)
            regularised_R = R.astype(np.float64) + epsilon * np.eye(
                2,
                dtype=np.float64,
            )

            update_direction = (
                np.linalg.pinv(regularised_R) @ G.astype(np.float64)
            ).astype(np.float32)

            lambda_vector = lambda_vector + 2.0 * mu * e * update_direction

    adaptive_output = np.clip(
        adaptive_output,
        0,
        255,
    ).astype(np.uint8)

    result = gray.copy()
    result[fg] = adaptive_output[fg]

    return result


# ============================================================
# Nonlinear Polynomial Unsharp Masking
# ============================================================


def _nonlinear_polynomial_unsharp_enhance(
    gray: np.ndarray,
    foreground: np.ndarray,
    lambda_value: float = 0.00085,
    k: float = 400.0,
) -> np.ndarray:
    """
    Apply nonlinear polynomial Unsharp Masking (Type 1A-Pk).

    Based on Ramponi et al. (1996), Equation (24).
    """

    fg = foreground.astype(bool)
    x = gray.astype(np.float32)

    vertical_edge_kernel = np.array(
        [
            [0, 1, 0],
            [0, 0, 0],
            [0, -1, 0],
        ],
        dtype=np.float32,
    )

    vertical_highpass_kernel = np.array(
        [
            [0, -1, 0],
            [0, 2, 0],
            [0, -1, 0],
        ],
        dtype=np.float32,
    )

    horizontal_edge_kernel = np.array(
        [
            [0, 0, 0],
            [1, 0, -1],
            [0, 0, 0],
        ],
        dtype=np.float32,
    )

    horizontal_highpass_kernel = np.array(
        [
            [0, 0, 0],
            [-1, 2, -1],
            [0, 0, 0],
        ],
        dtype=np.float32,
    )

    vertical_edge = cv2.filter2D(
        x,
        cv2.CV_32F,
        vertical_edge_kernel,
        borderType=cv2.BORDER_REFLECT,
    )

    vertical_highpass = cv2.filter2D(
        x,
        cv2.CV_32F,
        vertical_highpass_kernel,
        borderType=cv2.BORDER_REFLECT,
    )

    horizontal_edge = cv2.filter2D(
        x,
        cv2.CV_32F,
        horizontal_edge_kernel,
        borderType=cv2.BORDER_REFLECT,
    )

    horizontal_highpass = cv2.filter2D(
        x,
        cv2.CV_32F,
        horizontal_highpass_kernel,
        borderType=cv2.BORDER_REFLECT,
    )

    # Equation (24)
    z = ((vertical_edge**2) + k) * vertical_highpass + (
        (horizontal_edge**2) + k
    ) * horizontal_highpass

    y = x + lambda_value * z

    y = np.clip(
        y,
        0,
        255,
    ).astype(np.uint8)

    result = gray.copy()
    result[fg] = y[fg]

    return result


# ============================================================
# Sobel-based similar algorithm comparison
# ============================================================


def _sobel_sharpen_enhance(
    gray: np.ndarray,
    foreground: np.ndarray,
    gain: float = 0.5,
) -> np.ndarray:
    """
    Apply Sobel-based sharpening as a similar spatial-domain
    sharpening method for comparison with Conventional UM.
    """

    fg = foreground.astype(bool)
    x = gray.astype(np.float32)

    sobel_x = cv2.Sobel(
        x,
        cv2.CV_32F,
        1,
        0,
        ksize=3,
        borderType=cv2.BORDER_REFLECT,
    )

    sobel_y = cv2.Sobel(
        x,
        cv2.CV_32F,
        0,
        1,
        ksize=3,
        borderType=cv2.BORDER_REFLECT,
    )

    gradient_magnitude = cv2.magnitude(
        sobel_x,
        sobel_y,
    )

    y = x + gain * gradient_magnitude

    y = np.clip(
        y,
        0,
        255,
    ).astype(np.uint8)

    result = gray.copy()
    result[fg] = y[fg]

    return result


# ============================================================
# Candidate minutiae evaluation
# ============================================================


def _count_candidate_minutiae(
    image: np.ndarray,
    foreground: np.ndarray,
) -> tuple[int, int]:
    """
    Detect ridge endings and bifurcations using the same
    post-processing settings for every candidate.
    """

    ridge_binary = binarise_dark_ridges(
        image,
        foreground,
    )

    ridge_binary = clean_binary(
        ridge_binary,
        min_component_area=10,
    )

    skeleton = make_skeleton(ridge_binary)

    endings, bifurcations = crossing_number_minutiae(
        skeleton,
        foreground,
        border=10,
        min_distance=8,
    )

    return len(endings), len(bifurcations)


# ============================================================
# Main Unsharp Masking pipeline
# ============================================================


def run_algorithm(
    image: np.ndarray,
    *,
    preprocessing_config: PreprocessingConfig | None = None,
    final_only: bool = False,
) -> dict:
    """
    Enhance a fingerprint using Unsharp Masking.

    final_only=False:
        Runs the full research workflow:
        Conventional UM, Adaptive UM, Polynomial UM and Sobel comparison.

    final_only=True:
        Runs only the final selected Nonlinear Polynomial UM pipeline.
        This mode is used for the four-member Algorithm Comparison.
    """

    started = perf_counter()

    original = ensure_uint8(image)

    config = preprocessing_config or PreprocessingConfig()

    config.validate()

    # --------------------------------------------------------
    # Shared preprocessing
    # --------------------------------------------------------

    preprocessing_stages = preprocess_with_stages(
        original,
        config,
    )

    preprocessed = preprocessing_stages["enhanced"]

    # --------------------------------------------------------
    # Fingerprint segmentation
    # --------------------------------------------------------

    foreground_mask, segmentation_blocks = segment_fingerprint(
        preprocessed,
        block_size=16,
        min_block_std=5.0,
    )

    # --------------------------------------------------------
    # Orientation field
    # --------------------------------------------------------

    (
        raw_orientation,
        valid_blocks,
        orientation_coherence,
    ) = estimate_orientation_field(
        preprocessed,
        foreground_mask,
        block_size=16,
    )

    orientation_field = smooth_orientation_field(
        raw_orientation,
        valid_blocks,
        sigma=1.0,
    )

    orientation_vis = visualise_orientation_field(
        preprocessed,
        orientation_field,
        valid_blocks,
        block_size=16,
    )

    warnings: list[str] = []

    # ========================================================
    # FINAL-ONLY MODE
    #
    # Used only by the four-member Algorithm Comparison.
    #
    # This skips:
    #   - Conventional UM
    #   - Adaptive UM
    #   - Sobel comparison
    #
    # and executes only the final selected Polynomial UM.
    # ========================================================

    if final_only:

        polynomial_started = perf_counter()

        polynomial_enhanced = _nonlinear_polynomial_unsharp_enhance(
            preprocessed,
            foreground_mask,
            lambda_value=0.00085,
            k=400.0,
        )

        polynomial_time_ms = (perf_counter() - polynomial_started) * 1000.0

        enhanced = polynomial_enhanced

        # ----------------------------------------------------
        # Post-processing
        # ----------------------------------------------------

        ridge_binary = binarise_dark_ridges(
            enhanced,
            foreground_mask,
        )

        ridge_binary = clean_binary(
            ridge_binary,
            min_component_area=10,
        )

        skeleton = make_skeleton(ridge_binary)

        endings, bifurcations = crossing_number_minutiae(
            skeleton,
            foreground_mask,
            border=10,
            min_distance=8,
        )

        overlay = minutiae_overlay(
            enhanced,
            endings,
            bifurcations,
        )

        # ----------------------------------------------------
        # Metrics
        # ----------------------------------------------------

        elapsed_ms = (perf_counter() - started) * 1000.0

        metrics = calculate_image_metrics(
            original,
            enhanced,
            foreground_mask=foreground_mask,
        )

        metrics.update(
            metric_bundle(
                enhanced,
                foreground_mask,
                len(endings),
                len(bifurcations),
                elapsed_ms,
            )
        )

        metrics["foreground_coverage_percent"] = float(foreground_mask.mean() * 100.0)

        metrics["valid_orientation_blocks"] = int(valid_blocks.sum())

        if np.any(valid_blocks):
            metrics["mean_orientation_coherence"] = float(
                orientation_coherence[valid_blocks].mean()
            )
        else:
            metrics["mean_orientation_coherence"] = 0.0

        return {
            "pipeline_build": PIPELINE_BUILD,
            "algorithm_name": "Unsharp Masking",
            "status": ("warning" if warnings else "ok"),
            "warnings": warnings,
            "original": original,
            "grayscale": preprocessing_stages["grayscale"],
            "preprocessing_stages": (preprocessing_stages),
            "preprocessed": preprocessed,
            "normalised": preprocessing_stages["normalised"],
            "denoised": preprocessing_stages["denoised"],
            "foreground_mask": (foreground_mask),
            "mask": foreground_mask,
            "orientation_field": (orientation_field),
            "orientation_block_mask": (valid_blocks),
            "orientation_coherence": (orientation_coherence),
            "orientation_visualisation": (orientation_vis),
            "polynomial_unsharp": (polynomial_enhanced),
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
            "polynomial_unsharp_time_ms": float(polynomial_time_ms),
            "polynomial_unsharp_minutiae": int(len(endings) + len(bifurcations)),
        }

    # ========================================================
    # NORMAL FULL RESEARCH MODE
    #
    # Everything below here runs when final_only=False.
    # ========================================================

    # --------------------------------------------------------
    # Conventional UM
    # --------------------------------------------------------

    conventional_started = perf_counter()

    conventional_enhanced = _conventional_unsharp_enhance(
        preprocessed,
        foreground_mask,
        lambda_value=1.0,
    )

    conventional_time_ms = (perf_counter() - conventional_started) * 1000.0

    # --------------------------------------------------------
    # Adaptive UM
    # --------------------------------------------------------

    adaptive_started = perf_counter()

    adaptive_enhanced = _adaptive_unsharp_enhance(
        preprocessed,
        foreground_mask,
    )

    adaptive_time_ms = (perf_counter() - adaptive_started) * 1000.0

    # --------------------------------------------------------
    # Nonlinear Polynomial UM
    # --------------------------------------------------------

    polynomial_started = perf_counter()

    polynomial_enhanced = _nonlinear_polynomial_unsharp_enhance(
        preprocessed,
        foreground_mask,
        lambda_value=0.00085,
        k=400.0,
    )

    polynomial_time_ms = (perf_counter() - polynomial_started) * 1000.0

    # --------------------------------------------------------
    # Sobel-based comparison
    # --------------------------------------------------------

    sobel_started = perf_counter()

    sobel_enhanced = _sobel_sharpen_enhance(
        preprocessed,
        foreground_mask,
        gain=0.5,
    )

    sobel_time_ms = (perf_counter() - sobel_started) * 1000.0

    # --------------------------------------------------------
    # Candidate image-quality metrics
    # --------------------------------------------------------

    conventional_metrics = calculate_image_metrics(
        original,
        conventional_enhanced,
        foreground_mask=foreground_mask,
    )

    adaptive_metrics = calculate_image_metrics(
        original,
        adaptive_enhanced,
        foreground_mask=foreground_mask,
    )

    polynomial_metrics = calculate_image_metrics(
        original,
        polynomial_enhanced,
        foreground_mask=foreground_mask,
    )

    sobel_metrics = calculate_image_metrics(
        original,
        sobel_enhanced,
        foreground_mask=foreground_mask,
    )

    # --------------------------------------------------------
    # Candidate minutiae counts
    # --------------------------------------------------------

    (
        original_endings,
        original_bifurcations,
    ) = _count_candidate_minutiae(
        preprocessed,
        foreground_mask,
    )

    (
        conventional_endings,
        conventional_bifurcations,
    ) = _count_candidate_minutiae(
        conventional_enhanced,
        foreground_mask,
    )

    (
        adaptive_endings,
        adaptive_bifurcations,
    ) = _count_candidate_minutiae(
        adaptive_enhanced,
        foreground_mask,
    )

    (
        polynomial_endings,
        polynomial_bifurcations,
    ) = _count_candidate_minutiae(
        polynomial_enhanced,
        foreground_mask,
    )

    (
        sobel_endings,
        sobel_bifurcations,
    ) = _count_candidate_minutiae(
        sobel_enhanced,
        foreground_mask,
    )

    # --------------------------------------------------------
    # Final selected enhancement
    # --------------------------------------------------------

    # Nonlinear Polynomial UM is the selected final output.
    enhanced = polynomial_enhanced

    # --------------------------------------------------------
    # Final post-processing
    # --------------------------------------------------------

    ridge_binary = binarise_dark_ridges(
        enhanced,
        foreground_mask,
    )

    ridge_binary = clean_binary(
        ridge_binary,
        min_component_area=10,
    )

    skeleton = make_skeleton(ridge_binary)

    endings, bifurcations = crossing_number_minutiae(
        skeleton,
        foreground_mask,
        border=10,
        min_distance=8,
    )

    overlay = minutiae_overlay(
        enhanced,
        endings,
        bifurcations,
    )

    # --------------------------------------------------------
    # Final metrics
    # --------------------------------------------------------

    elapsed_ms = (perf_counter() - started) * 1000.0

    metrics = calculate_image_metrics(
        original,
        enhanced,
        foreground_mask=foreground_mask,
    )

    metrics.update(
        metric_bundle(
            enhanced,
            foreground_mask,
            len(endings),
            len(bifurcations),
            elapsed_ms,
        )
    )

    metrics["foreground_coverage_percent"] = float(foreground_mask.mean() * 100.0)

    metrics["valid_orientation_blocks"] = int(valid_blocks.sum())

    if np.any(valid_blocks):
        metrics["mean_orientation_coherence"] = float(
            orientation_coherence[valid_blocks].mean()
        )
    else:
        metrics["mean_orientation_coherence"] = 0.0

    # --------------------------------------------------------
    # Return full research result
    # --------------------------------------------------------

    return {
        "pipeline_build": PIPELINE_BUILD,
        "algorithm_name": "Unsharp Masking",
        "status": ("warning" if warnings else "ok"),
        "warnings": warnings,
        # Input / preprocessing
        "original": original,
        "grayscale": preprocessing_stages["grayscale"],
        "preprocessing_stages": (preprocessing_stages),
        "preprocessed": preprocessed,
        "normalised": preprocessing_stages["normalised"],
        "denoised": preprocessing_stages["denoised"],
        # UM candidates
        "conventional_unsharp": (conventional_enhanced),
        "adaptive_unsharp": (adaptive_enhanced),
        "polynomial_unsharp": (polynomial_enhanced),
        # Similar-algorithm comparator
        "sobel_sharpening": (sobel_enhanced),
        # Candidate metrics
        "conventional_unsharp_metrics": (conventional_metrics),
        "adaptive_unsharp_metrics": (adaptive_metrics),
        "polynomial_unsharp_metrics": (polynomial_metrics),
        "sobel_sharpening_metrics": (sobel_metrics),
        # Segmentation / orientation
        "foreground_mask": (foreground_mask),
        "mask": foreground_mask,
        "orientation_field": (orientation_field),
        "orientation_block_mask": (valid_blocks),
        "orientation_coherence": (orientation_coherence),
        "orientation_visualisation": (orientation_vis),
        # Final Polynomial UM output
        "ridge_restored": enhanced,
        "enhanced_image": enhanced,
        "ridge_enhanced": enhanced,
        # Post-processing
        "ridge_binary": ridge_binary,
        "binary": ridge_binary,
        "skeleton": skeleton,
        "endings": endings,
        "bifurcations": bifurcations,
        "minutiae_overlay": overlay,
        # Final metrics
        "metrics": metrics,
        "processing_time_ms": float(elapsed_ms),
        # Candidate processing times
        "conventional_unsharp_time_ms": float(conventional_time_ms),
        "adaptive_unsharp_time_ms": float(adaptive_time_ms),
        "polynomial_unsharp_time_ms": float(polynomial_time_ms),
        "sobel_sharpening_time_ms": float(sobel_time_ms),
        # Candidate minutiae
        "original_minutiae": int(original_endings + original_bifurcations),
        "conventional_unsharp_minutiae": int(
            conventional_endings + conventional_bifurcations
        ),
        "adaptive_unsharp_minutiae": int(adaptive_endings + adaptive_bifurcations),
        "polynomial_unsharp_minutiae": int(
            polynomial_endings + polynomial_bifurcations
        ),
        "sobel_sharpening_minutiae": int(sobel_endings + sobel_bifurcations),
    }
