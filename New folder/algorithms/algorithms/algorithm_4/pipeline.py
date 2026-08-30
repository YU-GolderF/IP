"""Unsharp Masking fingerprint enhancement — Member 4.

Implements conventional linear Unsharp Masking as the baseline method,
with an enhanced adaptive Unsharp Masking method for comparison.
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
    to_grayscale,
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

PIPELINE_BUILD = "adaptive-unsharp-mask-v1"


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

    # Convert to float so negative high-pass values are preserved.
    x = gray.astype(np.float32)

    # High-pass correction signal z(n,m)
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

    # Conventional Unsharp Masking:
    # y(n,m) = x(n,m) + lambda * z(n,m)
    y = x + lambda_value * z

    # Convert result back to a valid 8-bit image.
    y = np.clip(y, 0, 255).astype(np.uint8)

    # Only enhance the detected fingerprint foreground.
    result = gray.copy()
    result[fg] = y[fg]

    return result


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
    This function will implement the proposed enhanced method.
    """

    fg = foreground.astype(bool)

    # Convert to float so negative directional responses are preserved.
    x = gray.astype(np.float32)

    # Equation (3) from Polesel et al. (2000):
    # z_x(n,m) = 2x(n,m) - x(n,m-1) - x(n,m+1)
    horizontal_kernel = np.array(
        [
            [0, 0, 0],
            [-1, 2, -1],
            [0, 0, 0],
        ],
        dtype=np.float32,
    )

    # Equation (4) from Polesel et al. (2000):
    # z_y(n,m) = 2x(n,m) - x(n-1,m) - x(n+1,m)
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

    # v_i(n,m) = mean(x^2) - mean(x)^2
    local_variance = local_mean_squared - (local_mean * local_mean)

    # Protect against very small negative values caused by floating-point rounding.
    local_variance = np.maximum(local_variance, 0.0)

    # Region classification based on local variance:
    # smooth region:        v_i < tau1
    # medium-contrast:      tau1 <= v_i < tau2
    # high-contrast region: v_i >= tau2

    smooth_region = (local_variance < tau1) & fg

    medium_region = (local_variance >= tau1) & (local_variance < tau2) & fg

    high_region = (local_variance >= tau2) & fg

    # Equation (12) from Polesel et al. (2000):
    # Desired activity gain alpha(n,m).
    #
    # Smooth regions: no activity increase.
    # Medium-contrast regions: maximum enhancement.
    # High-contrast regions: moderate enhancement.

    alpha = np.ones_like(x, dtype=np.float32)

    alpha[medium_region] = alpha_dh
    alpha[high_region] = alpha_dl

    # Figure 2 from Polesel et al. (2000):
    # 3 x 3 high-pass operator g(.) used to measure local image dynamics.
    dynamics_kernel = np.array(
        [
            [-1, -1, -1],
            [-1, 8, -1],
            [-1, -1, -1],
        ],
        dtype=np.float32,
    )

    # g_x(n,m): local dynamics of the input image.
    g_x = cv2.filter2D(
        x,
        ddepth=cv2.CV_32F,
        kernel=dynamics_kernel,
        borderType=cv2.BORDER_REFLECT,
    )

    # Local dynamics of the horizontal directional correction z_x.
    g_zx = cv2.filter2D(
        z_x,
        ddepth=cv2.CV_32F,
        kernel=dynamics_kernel,
        borderType=cv2.BORDER_REFLECT,
    )

    # Local dynamics of the vertical directional correction z_y.
    g_zy = cv2.filter2D(
        z_y,
        ddepth=cv2.CV_32F,
        kernel=dynamics_kernel,
        borderType=cv2.BORDER_REFLECT,
    )

    # Equation (11):
    # g_d(n,m) = alpha(n,m) * g_x(n,m)
    # This represents the desired local dynamics of the output image.
    g_d = alpha * g_x

    # Equations (14)-(17) from Polesel et al. (2000):
    # Gauss-Newton adaptation of the directional scaling factors.

    height, width = x.shape

    # Output image before conversion back to uint8.
    adaptive_output = x.copy()

    # Scaling vector:
    # Lambda(n,m) = [lambda_x(n,m), lambda_y(n,m)]^T
    lambda_vector = np.zeros(2, dtype=np.float32)

    # Initial autocorrelation estimate.
    #
    # The paper defines the recursive update of R but does not specify
    # an explicit initial matrix. An identity matrix is used here to
    # provide a stable, invertible starting condition.
    R = np.eye(2, dtype=np.float32)

    # Small numerical regularisation used only to keep the matrix
    # inversion stable when R is close to singular.
    epsilon = 1e-6
    identity = np.eye(2, dtype=np.float32)

    # Adapt the scaling factors sequentially along image rows.
    for n in range(height):
        for m in range(width):

            # Equation (15):
            # G(n,m) = [g_zx(n,m), g_zy(n,m)]^T
            G = np.array(
                [g_zx[n, m], g_zy[n, m]],
                dtype=np.float32,
            )

            # Equation (14):
            # g_y(n,m) = g_x(n,m) + Lambda^T(n,m) G(n,m)
            g_y = g_x[n, m] + float(np.dot(lambda_vector, G))

            # Error between desired and current local dynamics.
            e = g_d[n, m] - g_y

            # Equation (5):
            # y(n,m) = x(n,m)
            #          + lambda_x(n,m) * z_x(n,m)
            #          + lambda_y(n,m) * z_y(n,m)
            adaptive_output[n, m] = (
                x[n, m] + lambda_vector[0] * z_x[n, m] + lambda_vector[1] * z_y[n, m]
            )

            # Equation (17):
            # R(n,m) = (1-beta)R(n,m-1)
            #          + beta G(n,m)G^T(n,m)
            R = (1.0 - beta) * R + beta * np.outer(G, G)

            # Equation (16):
            # Lambda(n,m+1) =
            # Lambda(n,m) + 2*mu*e*R^-1*G

            # Numerical implementation of R^-1 G.
            # The pseudo-inverse is used to remain stable when R is singular
            # or nearly singular for low-activity image regions.
            regularised_R = R.astype(np.float64) + epsilon * np.eye(2, dtype=np.float64)

            update_direction = (
                np.linalg.pinv(regularised_R) @ G.astype(np.float64)
            ).astype(np.float32)

            lambda_vector = lambda_vector + 2.0 * mu * e * update_direction

    # Keep pixel intensities within the valid 8-bit range.
    adaptive_output = np.clip(
        adaptive_output,
        0,
        255,
    ).astype(np.uint8)

    # Apply the adaptive result only within the detected fingerprint.
    result = gray.copy()
    result[fg] = adaptive_output[fg]

    return result


def _nonlinear_polynomial_unsharp_enhance(
    gray: np.ndarray,
    foreground: np.ndarray,
    lambda_value: float = 0.00085,
    k: float = 400.0,
) -> np.ndarray:
    """
    Apply nonlinear polynomial Unsharp Masking (Type 1A-Pk).

    Based on Ramponi et al. (1996), Equation (24).
    The nonlinear edge sensor strengthens meaningful edges while
    reducing the uniform amplification produced by linear UM.
    """

    fg = foreground.astype(bool)
    x = gray.astype(np.float32)

    # Equation (24) - vertical edge sensor:
    # x(m-1,n) - x(m+1,n)
    vertical_edge_kernel = np.array(
        [
            [0, 1, 0],
            [0, 0, 0],
            [0, -1, 0],
        ],
        dtype=np.float32,
    )

    # Vertical high-pass component:
    # 2x(m,n) - x(m-1,n) - x(m+1,n)
    vertical_highpass_kernel = np.array(
        [
            [0, -1, 0],
            [0, 2, 0],
            [0, -1, 0],
        ],
        dtype=np.float32,
    )

    # Equation (24) - horizontal edge sensor:
    # x(m,n-1) - x(m,n+1)
    horizontal_edge_kernel = np.array(
        [
            [0, 0, 0],
            [1, 0, -1],
            [0, 0, 0],
        ],
        dtype=np.float32,
    )

    # Horizontal high-pass component:
    # 2x(m,n) - x(m,n-1) - x(m,n+1)
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

    # Equation (24) from Ramponi et al. (1996).
    z = ((vertical_edge**2) + k) * vertical_highpass + (
        (horizontal_edge**2) + k
    ) * horizontal_highpass

    # Standard Unsharp Masking combination:
    # y = x + lambda * z
    y = x + lambda_value * z

    y = np.clip(y, 0, 255).astype(np.uint8)

    result = gray.copy()
    result[fg] = y[fg]

    return result


def _count_candidate_minutiae(
    image: np.ndarray,
    foreground: np.ndarray,
) -> tuple[int, int]:
    """
    Detect ridge endings and bifurcations using the same post-processing
    settings for each Unsharp Masking candidate.
    """

    ridge_binary = binarise_dark_ridges(image, foreground)
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


def run_algorithm(
    image: np.ndarray,
    *,
    preprocessing_config: PreprocessingConfig | None = None,
) -> dict:
    """
    Enhance a fingerprint using Unsharp Masking.

    Returns a dict with the standard algorithm result structure.
    """
    started = perf_counter()
    original = ensure_uint8(image)
    config = preprocessing_config or PreprocessingConfig()
    config.validate()

    preprocessing_stages = preprocess_with_stages(original, config)
    preprocessed = preprocessing_stages["enhanced"]

    # Segmentation
    foreground_mask, segmentation_blocks = segment_fingerprint(
        preprocessed,
        block_size=16,
        min_block_std=5.0,
    )

    # Orientation field (for consistent comparison metrics)
    raw_orientation, valid_blocks, orientation_coherence = estimate_orientation_field(
        preprocessed,
        foreground_mask,
        block_size=16,
    )
    orientation_field = smooth_orientation_field(
        raw_orientation, valid_blocks, sigma=1.0
    )
    orientation_vis = visualise_orientation_field(
        preprocessed,
        orientation_field,
        valid_blocks,
        block_size=16,
    )

    # Enhancement
    warnings: list[str] = []

    conventional_started = perf_counter()

    conventional_enhanced = _conventional_unsharp_enhance(
        preprocessed,
        foreground_mask,
        lambda_value=1.0,
    )

    conventional_time_ms = (perf_counter() - conventional_started) * 1000.0

    adaptive_started = perf_counter()

    adaptive_enhanced = _adaptive_unsharp_enhance(
        preprocessed,
        foreground_mask,
    )

    adaptive_time_ms = (perf_counter() - adaptive_started) * 1000.0

    polynomial_started = perf_counter()

    polynomial_enhanced = _nonlinear_polynomial_unsharp_enhance(
        preprocessed,
        foreground_mask,
        lambda_value=0.00085,
        k=400.0,
    )

    polynomial_time_ms = (perf_counter() - polynomial_started) * 1000.0

    # Separate image-quality metrics for baseline and proposed enhancement.
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

    conventional_endings, conventional_bifurcations = _count_candidate_minutiae(
        conventional_enhanced,
        foreground_mask,
    )

    adaptive_endings, adaptive_bifurcations = _count_candidate_minutiae(
        adaptive_enhanced,
        foreground_mask,
    )

    polynomial_endings, polynomial_bifurcations = _count_candidate_minutiae(
        polynomial_enhanced,
        foreground_mask,
    )

    # Use Adaptive Unsharp Masking as the final enhanced output.
    enhanced = adaptive_enhanced

    # Post-processing
    ridge_binary = binarise_dark_ridges(enhanced, foreground_mask)
    ridge_binary = clean_binary(ridge_binary, min_component_area=10)
    skeleton = make_skeleton(ridge_binary)
    endings, bifurcations = crossing_number_minutiae(
        skeleton, foreground_mask, border=10, min_distance=8
    )
    overlay = minutiae_overlay(enhanced, endings, bifurcations)

    elapsed_ms = (perf_counter() - started) * 1000.0
    metrics = calculate_image_metrics(
        original, enhanced, foreground_mask=foreground_mask
    )
    metrics.update(
        metric_bundle(
            enhanced, foreground_mask, len(endings), len(bifurcations), elapsed_ms
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
        "status": "warning" if warnings else "ok",
        "warnings": warnings,
        "original": original,
        "grayscale": preprocessing_stages["grayscale"],
        "preprocessing_stages": preprocessing_stages,
        "preprocessed": preprocessed,
        "normalised": preprocessing_stages["normalised"],
        "denoised": preprocessing_stages["denoised"],
        "conventional_unsharp": conventional_enhanced,
        "adaptive_unsharp": adaptive_enhanced,
        "polynomial_unsharp": polynomial_enhanced,
        "conventional_unsharp_metrics": conventional_metrics,
        "adaptive_unsharp_metrics": adaptive_metrics,
        "polynomial_unsharp_metrics": polynomial_metrics,
        "foreground_mask": foreground_mask,
        "mask": foreground_mask,
        "orientation_field": orientation_field,
        "orientation_block_mask": valid_blocks,
        "orientation_coherence": orientation_coherence,
        "orientation_visualisation": orientation_vis,
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
        "conventional_unsharp_time_ms": float(conventional_time_ms),
        "adaptive_unsharp_time_ms": float(adaptive_time_ms),
        "polynomial_unsharp_time_ms": float(polynomial_time_ms),
        "conventional_unsharp_minutiae": int(
            conventional_endings + conventional_bifurcations
        ),
        "adaptive_unsharp_minutiae": int(
            adaptive_endings + adaptive_bifurcations
        ),
        "polynomial_unsharp_minutiae": int(
            polynomial_endings + polynomial_bifurcations
        ),
    }
