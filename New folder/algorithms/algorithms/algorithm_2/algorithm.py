"""Independent DCT-based contextual fingerprint enhancement core."""
from __future__ import annotations

from time import perf_counter

import cv2
import numpy as np

from .blocks import block_positions, raised_cosine_window
from .config import DCTContextualConfig
from .dct_filter import filter_dct_block
from .frequency import estimate_local_frequency
from .orientation import estimate_local_orientation_for_dct
from .preprocessing import prepare_dct_input
from .quality import compute_block_quality


def run_dct_contextual_enhancement(
    image: np.ndarray,
    config: DCTContextualConfig | None = None,
    variant: str = "proposed",
    preprocessing_config=None,
) -> dict:
    """
    Run DCT-based contextual fingerprint enhancement.

    Variants:
        - basic_dct
        - adaptive_frequency
        - confidence_aware
        - proposed
    """

    # ---------------------------------------------------------
    # 1. Configuration
    # ---------------------------------------------------------
    config = config or DCTContextualConfig()
    config.validate()

    started = perf_counter()

    # ---------------------------------------------------------
    # 2. Shared preprocessing
    # ---------------------------------------------------------
    shared_stages, roi = prepare_dct_input(
        image,
        preprocessing_config,
    )

    grayscale = shared_stages["grayscale"]

    # Keep the current source used by your benchmark.
    source_image = shared_stages["grayscale"]

    source = source_image.astype(np.float32) / 255.0

    # ---------------------------------------------------------
    # 3. Padding
    # ---------------------------------------------------------
    padded = cv2.copyMakeBorder(
        source,
        0,
        max(config.block_size - source.shape[0], 0),
        0,
        max(config.block_size - source.shape[1], 0),
        cv2.BORDER_REFLECT,
    )

    padded_roi = cv2.copyMakeBorder(
        roi.astype(np.uint8),
        0,
        padded.shape[0] - roi.shape[0],
        0,
        padded.shape[1] - roi.shape[1],
        cv2.BORDER_REFLECT,
    ).astype(bool)

    # ---------------------------------------------------------
    # 4. Accumulation buffers
    # ---------------------------------------------------------
    accumulation = np.zeros_like(padded)
    weights = np.zeros_like(padded)

    orientation_map = np.zeros_like(padded)
    frequency_map = np.zeros_like(padded)
    confidence_map = np.zeros_like(padded)
    quality_map = np.zeros_like(padded)

    # ---------------------------------------------------------
    # 5. Overlap-add window
    # ---------------------------------------------------------
    window = raised_cosine_window(
        config.block_size
    )

    # ---------------------------------------------------------
    # 6. Block processing
    # ---------------------------------------------------------
    for y in block_positions(
        padded.shape[0],
        config.block_size,
        config.stride,
    ):

        for x in block_positions(
            padded.shape[1],
            config.block_size,
            config.stride,
        ):

            # -------------------------------------------------
            # 6.1 Extract block
            # -------------------------------------------------
            block = padded[
                y:y + config.block_size,
                x:x + config.block_size,
            ]

            block_uint8 = np.rint(
                block * 255.0
            ).astype(np.uint8)

            # -------------------------------------------------
            # 6.2 ROI fraction
            # -------------------------------------------------
            roi_fraction = float(
                padded_roi[
                    y:y + config.block_size,
                    x:x + config.block_size,
                ].mean()
            )

            # -------------------------------------------------
            # 6.3 Orientation estimation
            # -------------------------------------------------
            (
                orientation,
                orientation_confidence,
            ) = estimate_local_orientation_for_dct(
                block_uint8
            )

            # -------------------------------------------------
            # 6.4 Frequency estimation
            # -------------------------------------------------
            (
                frequency,
                frequency_confidence,
            ) = estimate_local_frequency(
                block_uint8,
                orientation,
                config.minimum_frequency,
                config.maximum_frequency,
            )

            # -------------------------------------------------
            # 6.5 Block quality
            # -------------------------------------------------
            quality = compute_block_quality(
                block_uint8,
                roi_fraction,
                orientation_confidence,
                frequency_confidence,
            )

            # -------------------------------------------------
            # 6.6 Nominal frequency
            # -------------------------------------------------
            nominal_frequency = (
                config.minimum_frequency
                + config.maximum_frequency
            ) / 2.0

            # -------------------------------------------------
            # 7. Variant selection
            # -------------------------------------------------
            if variant == "basic_dct":

                frequency = nominal_frequency

                strength = config.base_strength

            elif variant == "adaptive_frequency":

                strength = config.base_strength

            elif variant == "confidence_aware":

                frequency = nominal_frequency

                confidence = min(
                    orientation_confidence,
                    frequency_confidence,
                )

                strength = (
                    config.base_strength
                    * (
                        0.25
                        + 0.75 * confidence
                    )
                )

            else:
                # ---------------------------------------------
                # Proposed adaptive strength
                # ---------------------------------------------
                confidence = min(
                    orientation_confidence,
                    frequency_confidence,
                )

                adaptive = (
                    quality
                    ** config.proposed_quality_exponent
                ) * (
                    confidence
                    ** config.proposed_confidence_exponent
                )

                strength = (
                    config.base_strength
                    * (
                        config.proposed_min_strength_ratio
                        + (
                            1.0
                            - config.proposed_min_strength_ratio
                        )
                        * adaptive
                    )
                )

            # -------------------------------------------------
            # 8. Contextual gain parameters
            #
            # ONLY Proposed is increased.
            # Other variants remain unchanged.
            # -------------------------------------------------
            if variant == "proposed":

                ridge_gain_scale = (
                    config.proposed_ridge_gain
                    * 1.15
                )

                detail_gain_scale = (
                    config.proposed_detail_gain
                    * 1.20
                )

            else:

                ridge_gain_scale = 0.85

                detail_gain_scale = 0.0

            # -------------------------------------------------
            # 9. DCT contextual filtering
            # -------------------------------------------------
            reconstructed = filter_dct_block(
                block_uint8,
                orientation,
                frequency,
                strength,
                config,
                ridge_gain_scale=ridge_gain_scale,
                detail_gain_scale=detail_gain_scale,
                clip_output=True,
            )

            # -------------------------------------------------
            # 10. Weighted overlap-add reconstruction
            # -------------------------------------------------
            accumulation[
                y:y + config.block_size,
                x:x + config.block_size,
            ] += reconstructed * window

            weights[
                y:y + config.block_size,
                x:x + config.block_size,
            ] += window

            # -------------------------------------------------
            # 11. Diagnostic maps
            # -------------------------------------------------
            orientation_map[
                y:y + config.block_size,
                x:x + config.block_size,
            ] += orientation * window

            frequency_map[
                y:y + config.block_size,
                x:x + config.block_size,
            ] += frequency * window

            confidence_map[
                y:y + config.block_size,
                x:x + config.block_size,
            ] += (
                orientation_confidence
                * frequency_confidence
            ) * window

            quality_map[
                y:y + config.block_size,
                x:x + config.block_size,
            ] += quality * window

    # ---------------------------------------------------------
    # 12. Normalize overlap-add reconstruction
    # ---------------------------------------------------------
    reconstructed = (
        accumulation
        / np.maximum(
            weights,
            1e-8,
        )
    )

    # ---------------------------------------------------------
    # 13. Final enhanced image
    #
    # Keep current behaviour unchanged.
    # ---------------------------------------------------------
    enhanced = reconstructed

    # ---------------------------------------------------------
    # 14. Crop to original image size
    # ---------------------------------------------------------
    h, w = grayscale.shape

    crop = lambda value: value[:h, :w]

    denominator = np.maximum(
        weights,
        1e-8,
    )

    # ---------------------------------------------------------
    # 15. Output
    # ---------------------------------------------------------
    return {
        "image": np.rint(
            np.clip(
                crop(enhanced),
                0.0,
                1.0,
            )
            * 255.0
        ).astype(np.uint8),

        "grayscale": grayscale,

        "normalised": shared_stages["normalised"],

        "shared_preprocessing_stages": shared_stages,

        "foreground": crop(
            padded_roi
        ),

        "dct_contextual": np.rint(
            np.clip(
                crop(reconstructed),
                0.0,
                1.0,
            )
            * 255.0
        ).astype(np.uint8),

        "dct_orientation_map": crop(
            orientation_map / denominator
        ),

        "frequency_map": crop(
            frequency_map / denominator
        ),

        "confidence_map": crop(
            confidence_map / denominator
        ),

        "quality_map": crop(
            quality_map / denominator
        ),

        "processing_time_ms": (
            perf_counter() - started
        ) * 1000.0,
    }