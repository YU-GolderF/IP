from __future__ import annotations

from dataclasses import asdict
from time import perf_counter

import numpy as np

from core.metrics import calculate_image_metrics
from core.preprocessing import (
    PreprocessingConfig,
    ensure_uint8 as shared_ensure_uint8,
    preprocess_with_stages,
    to_grayscale as shared_to_grayscale,
)

from .config import RHLTConfig
from .core import apply_rhlt, build_rhlt_psf, psf_visualisation
from .metrics import metric_bundle
from .orientation import (
    estimate_orientation_field,
    smooth_orientation_field,
    visualise_orientation_field,
)
from .postprocess import (
    binarise_dark_ridges,
    clean_binary,
    crossing_number_minutiae,
    linear_grayscale_stretch,
    make_skeleton,
    minutiae_overlay,
    rhlt_edge_guided_sharpen,
)
from .preprocess import fingerprint_foreground_mask, gaussian_denoise, percentile_normalise, to_grayscale
from .ridge_filter import enhance_ridges_with_gabor, isolate_ridges
from .segmentation import segment_fingerprint


def run_rhlt(
    image: np.ndarray,
    config: RHLTConfig | None = None,
    *,
    preprocessed_image: np.ndarray | None = None,
    preprocessing_config: PreprocessingConfig | None = None,
) -> dict:
    """
    Restore fingerprint ridge flow using shared preprocessing and local orientation.

    When preprocessed_image is supplied by the central app, preprocessing is not
    repeated. The legacy spiral-phase RHLT edge response is retained as a separate
    diagnostic output; orientation-adaptive Gabor filtering performs ridge restoration.
    """
    settings = config or RHLTConfig()
    settings.validate()
    started = perf_counter()
    original = shared_ensure_uint8(image)

    if preprocessed_image is None:
        shared_config = preprocessing_config or PreprocessingConfig(
            gaussian_sigma=settings.gaussian_sigma
        )
        preprocessing_stages = preprocess_with_stages(original, shared_config)
        preprocessed = preprocessing_stages["enhanced"]
    else:
        preprocessed = shared_to_grayscale(preprocessed_image)
        grayscale = shared_to_grayscale(original)
        preprocessing_stages = {
            "grayscale": grayscale,
            "gaussian_denoised": preprocessed,
            "denoised": preprocessed,
            "normalised": preprocessed,
            "enhanced": preprocessed,
        }

    foreground_mask, segmentation_blocks = segment_fingerprint(
        preprocessed,
        block_size=settings.block_size,
        min_block_std=settings.segmentation_min_std,
        threshold_scale=settings.segmentation_threshold_scale,
    )
    raw_orientation, valid_orientation_blocks, orientation_coherence = estimate_orientation_field(
        preprocessed,
        foreground_mask,
        block_size=settings.block_size,
    )
    orientation_field = smooth_orientation_field(
        raw_orientation,
        valid_orientation_blocks,
        sigma=settings.orientation_smoothing_sigma,
    )
    orientation_visualisation = visualise_orientation_field(
        preprocessed,
        orientation_field,
        valid_orientation_blocks,
        block_size=settings.block_size,
    )
    ridge_restored = enhance_ridges_with_gabor(
        preprocessed,
        orientation_field,
        foreground_mask,
        settings,
        valid_blocks=valid_orientation_blocks,
    )
    ridge_binary = isolate_ridges(
        ridge_restored,
        foreground_mask,
        min_component_area=settings.min_component_area,
    )

    # Keep the original spiral-phase RHLT result as an explainable diagnostic.
    psf = build_rhlt_psf(
        size=settings.psf_size,
        topological_charge=settings.topological_charge,
        aperture_ratio=settings.aperture_ratio,
        apodisation=settings.apodisation,
    )
    if min(preprocessed.shape) < 2:
        complex_response = np.zeros(preprocessed.shape, dtype=np.complex128)
        rhlt_edge = np.zeros(preprocessed.shape, dtype=np.uint8)
    else:
        complex_response, rhlt_edge = apply_rhlt(preprocessed, psf, foreground_mask)
    rhlt_stretched = linear_grayscale_stretch(
        rhlt_edge,
        foreground_mask,
        settings.stretch_low_percentile,
        settings.stretch_high_percentile,
    )

    skeleton = make_skeleton(ridge_binary)
    endings, bifurcations = crossing_number_minutiae(
        skeleton,
        foreground_mask,
        settings.minutiae_border,
        settings.minutiae_min_distance,
    )
    overlay = minutiae_overlay(ridge_restored, endings, bifurcations)

    warnings: list[str] = []
    if not np.any(foreground_mask):
        warnings.append("Too little ridge variation was found; no foreground was enhanced.")
    elif not np.any(valid_orientation_blocks):
        warnings.append("Foreground was found, but ridge orientation was not reliable enough for Gabor enhancement.")

    elapsed_ms = (perf_counter() - started) * 1000.0
    metrics = calculate_image_metrics(original, ridge_restored)
    metrics.update(
        metric_bundle(
            ridge_restored,
            foreground_mask,
            len(endings),
            len(bifurcations),
            elapsed_ms,
        )
    )
    metrics["foreground_coverage_percent"] = float(foreground_mask.mean() * 100.0)
    metrics["valid_orientation_blocks"] = int(valid_orientation_blocks.sum())

    return {
        "algorithm_name": "RHLT Ridge Flow Restoration",
        "status": "warning" if warnings else "ok",
        "warnings": warnings,
        "config": asdict(settings),
        "original": original,
        "grayscale": preprocessing_stages["grayscale"],
        "preprocessing_stages": preprocessing_stages,
        "preprocessed": preprocessed,
        "normalised": preprocessing_stages["normalised"],
        "denoised": preprocessing_stages["denoised"],
        "foreground_mask": foreground_mask,
        "mask": foreground_mask,
        "segmentation_blocks": segmentation_blocks,
        "raw_orientation_field": raw_orientation,
        "orientation_field": orientation_field,
        "orientation_block_mask": valid_orientation_blocks,
        "orientation_coherence": orientation_coherence,
        "orientation_visualisation": orientation_visualisation,
        "ridge_restored": ridge_restored,
        "enhanced_image": ridge_restored,
        "ridge_enhanced": ridge_restored,
        "ridge_binary": ridge_binary,
        "binary": ridge_binary,
        "skeleton": skeleton,
        "endings": endings,
        "bifurcations": bifurcations,
        "minutiae_overlay": overlay,
        "psf": psf,
        "psf_visualisation": psf_visualisation(psf),
        "complex_response": complex_response,
        "rhlt_raw": rhlt_edge,
        "rhlt_stretched": rhlt_stretched,
        "metrics": metrics,
        "processing_time_ms": float(elapsed_ms),
    }


def process_fingerprint(image: np.ndarray, config: RHLTConfig) -> dict:
    """Run the complete, reproducible RHLT fingerprint processing pipeline."""
    config.validate()
    started = perf_counter()

    original = np.asarray(image)
    grayscale = to_grayscale(original)
    normalised = percentile_normalise(grayscale, 1.0, 99.0)
    denoised = gaussian_denoise(normalised, config.gaussian_sigma)
    mask = fingerprint_foreground_mask(denoised, config.segmentation_sigma)

    psf = build_rhlt_psf(
        size=config.psf_size,
        topological_charge=config.topological_charge,
        aperture_ratio=config.aperture_ratio,
        apodisation=config.apodisation,
    )
    complex_response, rhlt_raw = apply_rhlt(denoised, psf, mask)
    rhlt_stretched = linear_grayscale_stretch(
        rhlt_raw,
        mask,
        config.stretch_low_percentile,
        config.stretch_high_percentile,
    )

    # Optional project enhancement: preserve a ridge-like grey image while using RHLT
    # edges to strengthen local ridge/valley separation.
    ridge_enhanced = rhlt_edge_guided_sharpen(denoised, rhlt_stretched, mask, config.edge_gain)

    binary = binarise_dark_ridges(ridge_enhanced, mask)
    binary = clean_binary(binary, config.min_component_area)
    skeleton = make_skeleton(binary)
    endings, bifurcations = crossing_number_minutiae(
        skeleton,
        mask,
        config.minutiae_border,
        config.minutiae_min_distance,
    )
    overlay = minutiae_overlay(ridge_enhanced, endings, bifurcations)

    elapsed_ms = (perf_counter() - started) * 1000.0
    metrics = metric_bundle(
        ridge_enhanced,
        mask,
        len(endings),
        len(bifurcations),
        elapsed_ms,
    )

    return {
        "config": asdict(config),
        "original": original,
        "grayscale": grayscale,
        "normalised": normalised,
        "denoised": denoised,
        "mask": mask,
        "psf": psf,
        "psf_visualisation": psf_visualisation(psf),
        "complex_response": complex_response,
        "rhlt_raw": rhlt_raw,
        "rhlt_stretched": rhlt_stretched,
        "ridge_enhanced": ridge_enhanced,
        "binary": binary,
        "skeleton": skeleton,
        "endings": endings,
        "bifurcations": bifurcations,
        "minutiae_overlay": overlay,
        "metrics": metrics,
    }


def run_ablation(image: np.ndarray, base: RHLTConfig) -> list[dict]:
    """
    Small parameter study for Mode A experimental evaluation.

    Baseline uses no apodisation. Windowed variants test sidelobe suppression, which is
    motivated by the relief/sidelobe limitation reported for RHLT fingerprint filtering.
    """
    settings = [
        ("Baseline RHLT", 0.00, base.psf_size),
        ("Windowed RHLT (mild)", 0.25, base.psf_size),
        ("Windowed RHLT (strong)", 0.50, base.psf_size),
        ("Larger PSF", base.apodisation, min(129, base.psf_size + 32 if (base.psf_size + 32) % 2 == 1 else base.psf_size + 33)),
    ]
    rows = []
    for label, apo, size in settings:
        cfg = RHLTConfig(**{**asdict(base), "apodisation": apo, "psf_size": size})
        result = process_fingerprint(image, cfg)
        rows.append({"variant": label, "apodisation": apo, "psf_size": size, **result["metrics"]})
    return rows
