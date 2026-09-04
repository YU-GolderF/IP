from __future__ import annotations

import cv2
import numpy as np

from .config import DCTContextualConfig


def build_dct_contextual_filter(
    size: int,
    orientation: float,
    frequency: float,
    strength: float,
    config: DCTContextualConfig,
    ridge_gain_scale: float = 0.85,
    detail_gain_scale: float = 0.0,
) -> np.ndarray:

    yy, xx = np.mgrid[:size, :size]

    # ---------------------------------------------------------
    # Normalised DCT frequency coordinates
    # ---------------------------------------------------------
    radial = np.sqrt(
        xx.astype(np.float32) ** 2
        + yy.astype(np.float32) ** 2
    ) / float(size)

    angle = np.arctan2(
        yy.astype(np.float32),
        xx.astype(np.float32),
    )

    # ---------------------------------------------------------
    # Ridge-normal direction
    # ---------------------------------------------------------
    normal = (
        orientation - np.pi / 2.0
    ) % np.pi

    # ---------------------------------------------------------
    # Orientation selectivity
    #
    # DCT coefficients aligned with the local ridge structure
    # receive stronger contextual response.
    # ---------------------------------------------------------
    cosine_alignment = np.abs(
        np.cos(angle - normal)
    )

    angular_distance = np.arccos(
        np.clip(
            cosine_alignment,
            0.0,
            1.0,
        )
    )

    angular_sigma = np.deg2rad(
        config.orientation_sigma_degrees
    )

    angular_response = np.exp(
        -0.5
        * (
            angular_distance
            / max(angular_sigma, 1e-6)
        ) ** 2
    )

    # ---------------------------------------------------------
    # Fundamental ridge-frequency response
    # ---------------------------------------------------------
    frequency_sigma = max(
        config.frequency_sigma,
        1e-6,
    )

    fundamental_response = np.exp(
        -0.5
        * (
            (radial - frequency)
            / frequency_sigma
        ) ** 2
    )

    # ---------------------------------------------------------
    # Harmonic support
    # ---------------------------------------------------------
    harmonic_frequency = min(
        frequency * 2.0,
        config.maximum_frequency,
    )

    harmonic_response = np.exp(
        -0.5
        * (
            (radial - harmonic_frequency)
            / (frequency_sigma * 1.4)
        ) ** 2
    )

    # ---------------------------------------------------------
    # Combined frequency response
    # ---------------------------------------------------------
    frequency_response = np.maximum(
        fundamental_response,
        0.35 * harmonic_response,
    )

    # ---------------------------------------------------------
    # Contextual response
    # ---------------------------------------------------------
    contextual = (
        frequency_response
        * angular_response
    )

    contextual = np.clip(
        contextual,
        0.0,
        1.0,
    )

    # ---------------------------------------------------------
    # Ridge-band enhancement
    #
    # Non-target coefficients remain unchanged.
    # Target coefficients receive contextual amplification.
    # ---------------------------------------------------------
    baseline_gain = 1.0

    ridge_band_gain = (
        1.0
        + ridge_gain_scale * strength
    )

    gain = (
        baseline_gain
        + (
            ridge_band_gain
            - baseline_gain
        )
        * contextual
    )

    # ---------------------------------------------------------
    # Directional detail enhancement
    #
    # The response is concentrated toward higher frequencies
    # while remaining orientation-aware.
    #
    # tanh provides a soft saturation so that detail enhancement
    # does not grow excessively in strong-texture regions.
    # ---------------------------------------------------------
    detail_scale = max(
        frequency * 1.5,
        1e-6,
    )

    high_frequency = (
        1.0
        - np.exp(
            -0.5
            * (radial / detail_scale) ** 2
        )
    )

    directional_detail = (
        high_frequency
        * (
            0.25
            + 0.75 * angular_response
        )
    )

    # Soft saturation of detail response.
    directional_detail = np.tanh(
        directional_detail * 1.5
    )

    detail_gain = (
        detail_gain_scale
        * strength
        * directional_detail
    )

    gain += detail_gain

    # ---------------------------------------------------------
    # Protect low-frequency / DC region
    #
    # The DC coefficient represents the local mean intensity
    # and must not be enhanced.
    # ---------------------------------------------------------
    gain[0, 0] = 1.0

    # ---------------------------------------------------------
    # Numerical safety
    # ---------------------------------------------------------
    gain = np.nan_to_num(
        gain,
        nan=1.0,
        posinf=1.0,
        neginf=1.0,
    )

    return gain.astype(
        np.float32
    )


def filter_dct_block(
    block: np.ndarray,
    orientation: float,
    frequency: float,
    strength: float,
    config: DCTContextualConfig,
    ridge_gain_scale: float = 0.85,
    detail_gain_scale: float = 0.0,
    clip_output: bool = True,
) -> np.ndarray:

    # ---------------------------------------------------------
    # Forward DCT
    # ---------------------------------------------------------
    dct = cv2.dct(
        block.astype(np.float32)
        / 255.0
    )

    # ---------------------------------------------------------
    # Build contextual DCT filter
    # ---------------------------------------------------------
    gain = build_dct_contextual_filter(
        size=block.shape[0],
        orientation=orientation,
        frequency=frequency,
        strength=strength,
        config=config,
        ridge_gain_scale=ridge_gain_scale,
        detail_gain_scale=detail_gain_scale,
    )

    # ---------------------------------------------------------
    # Contextual filtering in DCT domain
    # ---------------------------------------------------------
    filtered = dct * gain

    # ---------------------------------------------------------
    # Inverse DCT
    # ---------------------------------------------------------
    reconstructed = cv2.idct(
        filtered
    )

    if clip_output:
        return np.clip(
            reconstructed,
            0.0,
            1.0,
        )

    return reconstructed