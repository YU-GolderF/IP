from __future__ import annotations

from dataclasses import asdict, replace
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
from .ridge_filter import detail_preserving_sharpen, enhance_ridges_with_gabor, isolate_ridges
from .segmentation import segment_fingerprint

PIPELINE_BUILD = "sigmoid-push-directional-v6"


def _adapt_settings_to_image(
    settings: RHLTConfig,
    preprocessing: PreprocessingConfig,
    image_shape: tuple[int, ...],
) -> tuple[RHLTConfig, PreprocessingConfig]:
    """Scale filter windows down for very small fingerprint images."""
    short_side = min(image_shape[:2])
    if short_side >= 160:
        return settings, preprocessing

    # The bundled fingerprint samples are roughly 100 px high. Applying the
    # desktop-size filters to them merges neighbouring ridge lines. Use much
    # tighter parameters so the Gabor filter resolves individual ridges.
    tuned_settings = replace(
        settings,
        gaussian_sigma=min(settings.gaussian_sigma, 0.25),
        block_size=min(settings.block_size, 8),
        gabor_kernel_size=min(settings.gabor_kernel_size, 7),
        gabor_sigma=min(settings.gabor_sigma, 1.2),
        gabor_lambda=min(settings.gabor_lambda, 4.0),
        gabor_blend_strength=max(settings.gabor_blend_strength, 80.0),
        gabor_strength=max(settings.gabor_strength, 1.5),
        min_component_area=min(settings.min_component_area, 6),
    )
    tuned_preprocessing = replace(
        preprocessing,
        gaussian_kernel_size=3,
        gaussian_sigma=min(preprocessing.gaussian_sigma, 0.25),
        clahe_clip_limit=min(preprocessing.clahe_clip_limit, 1.2),
        clahe_grid_size=min(preprocessing.clahe_grid_size, 4),
    )
    return tuned_settings, tuned_preprocessing


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
    started = perf_counter()
    original = shared_ensure_uint8(image)
    shared_config = preprocessing_config or PreprocessingConfig(
        gaussian_sigma=settings.gaussian_sigma
    )
    settings, shared_config = _adapt_settings_to_image(
        settings, shared_config, original.shape
    )
    settings.validate()
    shared_config.validate()

    if preprocessed_image is None:
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
    warnings: list[str] = []
    ridge_candidate = enhance_ridges_with_gabor(
        preprocessed,
        orientation_field,
        foreground_mask,
        settings,
        valid_blocks=valid_orientation_blocks,
        base_image=original,
    )
    original_gray = shared_to_grayscale(original)
    detail_candidate = detail_preserving_sharpen(original_gray, foreground_mask)
    candidates = [ridge_candidate, detail_candidate]
    baseline = calculate_image_metrics(original_gray, original_gray)

    def candidate_score(candidate: np.ndarray) -> tuple[float, dict]:
        candidate_metrics = calculate_image_metrics(original_gray, candidate)
        contrast_ratio = candidate_metrics["processed_contrast"] / max(
            baseline["original_contrast"], 1e-6
        )
        sharpness_ratio = candidate_metrics["processed_sharpness"] / max(
            baseline["original_sharpness"], 1e-6
        )
        edge_ratio = candidate_metrics["processed_edge_clarity"] / max(
            baseline["original_edge_clarity"], 1e-6
        )
        ssim_value = candidate_metrics.get("ssim")
        ssim_is_acceptable = ssim_value is None or ssim_value >= 0.80
        preserves_quality = (
            contrast_ratio >= 0.90
            and sharpness_ratio >= 0.90
            and edge_ratio >= 0.90
            and ssim_is_acceptable
        )
        if not preserves_quality:
            return 0.0, candidate_metrics
        score = (
            0.20 * min(contrast_ratio, 1.5)
            + 0.40 * min(sharpness_ratio, 1.5)
            + 0.40 * min(edge_ratio, 1.5)
        )
        return float(score), candidate_metrics

    scored_candidates = [(candidate_score(candidate)[0], candidate) for candidate in candidates]
    best_score, best_candidate = max(scored_candidates, key=lambda item: item[0])
    if best_score > 0.98:
        ridge_restored = best_candidate
    else:
        ridge_restored = original_gray
        warnings.append(
            "No enhancement candidate improved the measurable ridge detail; "
            "the already-sharp original was preserved."
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

    if not np.any(foreground_mask):
        warnings.append("Too little ridge variation was found; no foreground was enhanced.")
    elif not np.any(valid_orientation_blocks):
        warnings.append("Foreground was found, but ridge orientation was not reliable enough for Gabor enhancement.")

    elapsed_ms = (perf_counter() - started) * 1000.0
    metrics = calculate_image_metrics(original, ridge_restored, foreground_mask=foreground_mask)
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
    # Mean coherence over valid blocks: 0 = chaotic, 1 = perfectly parallel ridges.
    if np.any(valid_orientation_blocks):
        metrics["mean_orientation_coherence"] = float(
            orientation_coherence[valid_orientation_blocks].mean()
        )
    else:
        metrics["mean_orientation_coherence"] = 0.0

    return {
        "pipeline_build": PIPELINE_BUILD,
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
