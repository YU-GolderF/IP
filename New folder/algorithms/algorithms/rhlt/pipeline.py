from __future__ import annotations

from dataclasses import asdict, replace
from time import perf_counter

import cv2
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
from .frequency import estimate_local_ridge_wavelength, expand_block_map
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
from .quality import defect_aware_fusion_weight, local_ridge_quality_maps
from .ridge_filter import enhance_ridges_with_gabor, isolate_ridges
from .segmentation import segment_fingerprint

PIPELINE_BUILD = "rhlt-primary-quality-adaptive-v3.2"


def _adapt_settings_to_image(
    settings: RHLTConfig,
    preprocessing: PreprocessingConfig,
    image_shape: tuple[int, ...],
) -> tuple[RHLTConfig, PreprocessingConfig]:
    """Scale filter windows down for very small fingerprint images."""
    short_side = min(image_shape[:2])
    if short_side >= 160:
        return settings, preprocessing

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
        frequency_block_size=min(settings.frequency_block_size, 16),
        minimum_ridge_wavelength=min(settings.minimum_ridge_wavelength, 2.5),
        maximum_ridge_wavelength=min(settings.maximum_ridge_wavelength, 7.0),
    )
    tuned_preprocessing = replace(
        preprocessing,
        gaussian_kernel_size=3,
        gaussian_sigma=min(preprocessing.gaussian_sigma, 0.25),
        clahe_clip_limit=min(preprocessing.clahe_clip_limit, 1.2),
        clahe_grid_size=min(preprocessing.clahe_grid_size, 4),
    )
    return tuned_settings, tuned_preprocessing


def _orientation_only_weight(
    foreground: np.ndarray,
    orientation_coherence: np.ndarray,
    valid_blocks: np.ndarray,
    rhlt_edge: np.ndarray,
    settings: RHLTConfig,
) -> np.ndarray:
    """Retain the previous orientation-only weighting for ablation."""
    reliable = np.asarray(valid_blocks, dtype=bool) & (
        np.asarray(orientation_coherence) >= settings.minimum_orientation_coherence
    )
    coherence_blocks = np.where(reliable, orientation_coherence, 0.0)
    coherence_map = expand_block_map(coherence_blocks, settings.block_size, foreground.shape)
    edge = np.asarray(rhlt_edge, dtype=np.float32) / 255.0
    weight = settings.hybrid_gabor_max_weight * np.power(
        np.clip(coherence_map, 0.0, 1.0), settings.rhlt_support_gamma
    ) * edge
    return np.where(
        foreground,
        np.clip(weight, 0.0, settings.hybrid_gabor_max_weight),
        0.0,
    ).astype(np.float32)


def _fuse_rhlt_gabor(
    rhlt_baseline: np.ndarray,
    gabor_support: np.ndarray,
    foreground: np.ndarray,
    support_weight: np.ndarray,
) -> np.ndarray:
    """Blend bounded Gabor support into the RHLT baseline in foreground only."""
    fg = np.asarray(foreground, dtype=bool)
    weight = np.asarray(support_weight, dtype=np.float32)
    if weight.shape != fg.shape:
        raise ValueError("support_weight dimensions must match the image")
    base_f = rhlt_baseline.astype(np.float32)
    gabor_f = gabor_support.astype(np.float32)
    blended = base_f * (1.0 - weight) + gabor_f * weight
    result = rhlt_baseline.copy()
    result[fg] = np.clip(blended[fg], 0, 255).astype(np.uint8)
    return result


def _structure_diagnostics(image: np.ndarray, mask: np.ndarray, settings: RHLTConfig) -> dict:
    binary = isolate_ridges(image, mask, min_component_area=settings.min_component_area)
    skeleton = make_skeleton(binary)
    endings, bifurcations = crossing_number_minutiae(
        skeleton, mask, settings.minutiae_border, settings.minutiae_min_distance
    )
    count, _ = cv2.connectedComponents(binary.astype(np.uint8), connectivity=8)
    foreground_pixels = max(int(np.asarray(mask, dtype=bool).sum()), 1)
    return {
        "binary": binary,
        "skeleton": skeleton,
        "endings": endings,
        "bifurcations": bifurcations,
        "minutiae_total": len(endings) + len(bifurcations),
        "components_per_10k": float(max(0, count - 1) * 10000.0 / foreground_pixels),
    }


def _candidate_quality_score(
    metrics: dict,
    structure: dict,
    original_structure: dict,
    settings: RHLTConfig,
) -> tuple[float, dict[str, float | bool]]:
    """Transparent foreground-first score with fragmentation/noise penalties."""
    def bounded_ratio(processed_key: str, original_key: str) -> float:
        ratio = float(metrics[processed_key]) / max(float(metrics[original_key]), 1e-6)
        return float(np.clip(ratio, 0.0, 1.25) / 1.25)

    contrast = bounded_ratio("foreground_processed_contrast", "foreground_original_contrast")
    ridge_valley = bounded_ratio(
        "processed_ridge_valley_clarity", "original_ridge_valley_clarity"
    )
    edge = bounded_ratio(
        "foreground_processed_edge_clarity", "foreground_original_edge_clarity"
    )
    structural = float(np.clip(metrics.get("foreground_ssim", metrics.get("ssim", 0.0)), 0.0, 1.0))
    continuity = float(1.0 / (1.0 + structure["components_per_10k"]))
    original_minutiae = max(int(original_structure["minutiae_total"]), 1)
    excess = max(0.0, (structure["minutiae_total"] - original_minutiae) / original_minutiae)
    minutiae_penalty = float(np.clip(excess, 0.0, 1.0))
    score = (
        0.20 * contrast
        + 0.25 * ridge_valley
        + 0.20 * edge
        + 0.25 * structural
        + 0.10 * continuity
        - 0.10 * minutiae_penalty
    )
    safe = bool(
        structural >= settings.selector_ssim_floor
        and float(metrics["foreground_processed_edge_clarity"])
        >= 0.75 * float(metrics["foreground_original_edge_clarity"])
        and float(metrics["foreground_processed_contrast"])
        >= 0.65 * float(metrics["foreground_original_contrast"])
    )
    return float(score), {
        "contrast": contrast,
        "ridge_valley_clarity": ridge_valley,
        "edge_clarity": edge,
        "structural_preservation": structural,
        "ridge_continuity": continuity,
        "excess_minutiae_penalty": minutiae_penalty,
        "safe": safe,
    }


def run_rhlt(
    image: np.ndarray,
    config: RHLTConfig | None = None,
    *,
    preprocessed_image: np.ndarray | None = None,
    preprocessing_config: PreprocessingConfig | None = None,
    reference_image: np.ndarray | None = None,
) -> dict:
    """
    Restore fingerprint ridge flow with RHLT as the primary enhancement foundation.

    Data flow:
        Calibrated image
        → shared preprocessing
        → foreground segmentation
        → spiral-phase RHLT convolution  (primary)
        → RHLT edge-guided baseline image
        → ridge orientation estimation
        → doubled-angle orientation smoothing
        → orientation-adaptive Gabor support  (secondary)
        → bounded RHLT+Gabor fusion → improved RHLT
        → quality-preservation guard (RHLT candidates only)
        → binarisation → skeleton → minutiae → metrics
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

    # ── Shared preprocessing ──────────────────────────────────────────────────
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

    original_gray = shared_to_grayscale(original)

    # ── Foreground segmentation ───────────────────────────────────────────────
    foreground_mask, segmentation_blocks = segment_fingerprint(
        preprocessed,
        block_size=settings.block_size,
        min_block_std=settings.segmentation_min_std,
        threshold_scale=settings.segmentation_threshold_scale,
    )

    # ── Spiral-phase RHLT convolution (primary) ───────────────────────────────
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

    # ── Traditional RHLT baseline (RHLT-dependent) ────────────────────────────
    # rhlt_edge_guided_sharpen fuses the denoised image with RHLT edge magnitude,
    # producing a visually interpretable fingerprint that depends on the RHLT response.
    traditional_rhlt_baseline = rhlt_edge_guided_sharpen(
        preprocessed, rhlt_stretched, foreground_mask, settings.edge_gain
    )
    # Enhancement is foreground-only; calibrated input background is preserved.
    traditional_rhlt_baseline[~foreground_mask] = original_gray[~foreground_mask]

    # ── Ridge orientation estimation ──────────────────────────────────────────
    raw_orientation, valid_orientation_blocks, orientation_coherence = estimate_orientation_field(
        preprocessing_stages["denoised"],
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

    # ── Orientation-adaptive Gabor support (secondary) ────────────────────────
    # Gabor is used only as bounded directional support; it is never the final output.
    local_wavelength_field, frequency_valid_blocks, frequency_confidence = (
        estimate_local_ridge_wavelength(
            preprocessing_stages["denoised"],
            orientation_field,
            valid_orientation_blocks,
            foreground_mask,
            orientation_block_size=settings.block_size,
            analysis_block_size=settings.frequency_block_size,
            minimum_wavelength=settings.minimum_ridge_wavelength,
            maximum_wavelength=settings.maximum_ridge_wavelength,
            smoothing_size=settings.frequency_smoothing_size,
            fallback_wavelength=settings.gabor_lambda,
        )
    )
    wavelength_pixels = expand_block_map(
        local_wavelength_field, settings.block_size, foreground_mask.shape
    ).astype(np.float32)
    frequency_valid_pixels = expand_block_map(
        frequency_valid_blocks, settings.block_size, foreground_mask.shape
    ).astype(bool)
    local_frequency_map = np.zeros(foreground_mask.shape, dtype=np.float32)
    local_frequency_map[frequency_valid_pixels] = 1.0 / np.maximum(
        wavelength_pixels[frequency_valid_pixels], 1e-6
    )
    local_quality_map, weak_ridge_map, local_contrast_map = local_ridge_quality_maps(
        preprocessing_stages["denoised"],
        foreground_mask,
        sigma=settings.local_quality_sigma,
        target_contrast=settings.weak_ridge_target_contrast,
        clear_region_protection=settings.clear_region_protection,
    )

    warnings: list[str] = []
    if np.any(valid_orientation_blocks):
        gabor_support = enhance_ridges_with_gabor(
            preprocessed,
            orientation_field,
            foreground_mask,
            settings,
            valid_blocks=valid_orientation_blocks,
            base_image=original,
            wavelength_field=local_wavelength_field,
            frequency_valid_blocks=frequency_valid_blocks,
        )
    else:
        gabor_support = traditional_rhlt_baseline.copy()
        warnings.append(
            "No reliable orientation blocks found; Gabor support was not applied."
        )

    # ── Bounded RHLT+Gabor fusion → Improved RHLT ────────────────────────────
    if settings.use_quality_adaptive_fusion:
        fusion_weight_map = defect_aware_fusion_weight(
            weak_ridge_map,
            foreground_mask,
            orientation_coherence,
            valid_orientation_blocks,
            rhlt_stretched,
            settings,
        )
    else:
        fusion_weight_map = _orientation_only_weight(
            foreground_mask,
            orientation_coherence,
            valid_orientation_blocks,
            rhlt_stretched,
            settings,
        )
    improved_rhlt = _fuse_rhlt_gabor(
        traditional_rhlt_baseline, gabor_support, foreground_mask, fusion_weight_map
    )

    # ── Quality-preservation guard (RHLT candidates only) ─────────────────────
    # Only traditional_rhlt_baseline and improved_rhlt are eligible.
    # Pure Gabor and pure detail-sharpening are NOT eligible.
    clean_reference = None if reference_image is None else shared_to_grayscale(reference_image)
    baseline_metrics = calculate_image_metrics(
        original_gray,
        traditional_rhlt_baseline,
        reference=clean_reference,
        foreground_mask=foreground_mask,
    )
    improved_metrics = calculate_image_metrics(
        original_gray,
        improved_rhlt,
        reference=clean_reference,
        foreground_mask=foreground_mask,
    )

    original_structure = _structure_diagnostics(original_gray, foreground_mask, settings)
    baseline_structure = _structure_diagnostics(
        traditional_rhlt_baseline, foreground_mask, settings
    )
    improved_structure = _structure_diagnostics(improved_rhlt, foreground_mask, settings)
    traditional_quality_score, traditional_components = _candidate_quality_score(
        baseline_metrics, baseline_structure, original_structure, settings
    )
    improved_quality_score, improved_components = _candidate_quality_score(
        improved_metrics, improved_structure, original_structure, settings
    )

    # getattr keeps a long-running Streamlit session compatible with a config
    # object constructed before selector_regression_tolerance was introduced.
    regression_tolerance = float(
        getattr(settings, "selector_regression_tolerance", 0.010)
    )
    if not np.any(foreground_mask):
        ridge_restored = original_gray
        selected_output = "original_quality_fallback"
        selected_structure = original_structure
        selection_reason = "No reliable fingerprint foreground was available for enhancement."
    elif bool(improved_components["safe"]) and (
        not bool(traditional_components["safe"])
        or improved_quality_score >= traditional_quality_score - regression_tolerance
    ):
        ridge_restored = improved_rhlt
        selected_output = "improved_rhlt"
        selected_structure = improved_structure
        score_delta = improved_quality_score - traditional_quality_score
        selection_reason = (
            "Proposed Improved RHLT was selected as the primary method: it passed "
            f"all structural safety checks and its score difference was {score_delta:+.4f} "
            f"(allowed regression tolerance {regression_tolerance:.4f})."
        )
    elif bool(traditional_components["safe"]):
        ridge_restored = traditional_rhlt_baseline
        selected_output = "traditional_rhlt_baseline"
        selected_structure = baseline_structure
        selection_reason = (
            "Safety fallback to Traditional RHLT: Proposed Improved RHLT either failed "
            "a structural safety check or scored materially below Traditional RHLT "
            f"by more than {regression_tolerance:.4f}."
        )
    else:
        ridge_restored = original_gray
        selected_output = "original_quality_fallback"
        selected_structure = original_structure
        selection_reason = (
            "Neither RHLT candidate satisfied the foreground structural and quality safety bounds."
        )
    if selected_output != "improved_rhlt":
        warnings.append(selection_reason)

    # ── Post-processing ───────────────────────────────────────────────────────
    ridge_binary = selected_structure["binary"]
    skeleton = selected_structure["skeleton"]
    endings = selected_structure["endings"]
    bifurcations = selected_structure["bifurcations"]
    overlay = minutiae_overlay(ridge_restored, endings, bifurcations)

    # ── Warnings ─────────────────────────────────────────────────────────────
    if not np.any(foreground_mask):
        warnings.append("Too little ridge variation found; no foreground was segmented.")
    elif not np.any(valid_orientation_blocks):
        warnings.append(
            "Foreground found but ridge orientation was not reliable enough for Gabor support."
        )

    # ── Final metrics ─────────────────────────────────────────────────────────
    elapsed_ms = (perf_counter() - started) * 1000.0
    metrics = calculate_image_metrics(
        original,
        ridge_restored,
        reference=clean_reference,
        foreground_mask=foreground_mask,
    )
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
    if np.any(valid_orientation_blocks):
        metrics["mean_orientation_coherence"] = float(
            orientation_coherence[valid_orientation_blocks].mean()
        )
    else:
        metrics["mean_orientation_coherence"] = 0.0
    foreground_weights = fusion_weight_map[foreground_mask]
    mean_fusion_weight = float(np.mean(foreground_weights)) if foreground_weights.size else 0.0
    maximum_fusion_weight = float(np.max(foreground_weights)) if foreground_weights.size else 0.0
    enhanced_foreground_percent = (
        float(np.mean(foreground_weights > 0.01 * max(settings.hybrid_gabor_max_weight, 1e-6)) * 100.0)
        if foreground_weights.size
        else 0.0
    )
    valid_frequency_blocks = int(frequency_valid_blocks.sum())
    metrics.update(
        {
            "mean_fusion_weight": mean_fusion_weight,
            "maximum_fusion_weight": maximum_fusion_weight,
            "enhanced_foreground_percent": enhanced_foreground_percent,
            "valid_frequency_blocks": valid_frequency_blocks,
            "traditional_quality_score": traditional_quality_score,
            "improved_quality_score": improved_quality_score,
        }
    )

    return {
        # ── Identity ─────────────────────────────────────────────────────────
        "pipeline_build": PIPELINE_BUILD,
        "algorithm_name": "RHLT Ridge Flow Restoration",
        "status": "warning" if warnings else "ok",
        "warnings": warnings,
        "config": asdict(settings),
        # ── Input / preprocessing ─────────────────────────────────────────────
        "original": original,
        "grayscale": preprocessing_stages["grayscale"],
        "preprocessing_stages": preprocessing_stages,
        "preprocessed": preprocessed,
        "normalised": preprocessing_stages["normalised"],
        "denoised": preprocessing_stages["denoised"],
        # ── Segmentation / orientation ────────────────────────────────────────
        "foreground_mask": foreground_mask,
        "mask": foreground_mask,
        "segmentation_blocks": segmentation_blocks,
        "raw_orientation_field": raw_orientation,
        "orientation_field": orientation_field,
        "orientation_block_mask": valid_orientation_blocks,
        "orientation_coherence": orientation_coherence,
        "orientation_visualisation": orientation_visualisation,
        "local_wavelength_field": local_wavelength_field,
        "local_frequency_map": local_frequency_map,
        "frequency_valid_blocks": frequency_valid_blocks,
        "frequency_confidence": frequency_confidence,
        # ── RHLT diagnostic outputs ───────────────────────────────────────────
        "psf": psf,
        "psf_visualisation": psf_visualisation(psf),
        "complex_response": complex_response,
        "rhlt_raw": rhlt_edge,
        "rhlt_stretched": rhlt_stretched,
        # ── Algorithm stages (new) ────────────────────────────────────────────
        "traditional_rhlt_baseline": traditional_rhlt_baseline,
        "gabor_support": gabor_support,
        "improved_rhlt": improved_rhlt,
        "local_quality_map": local_quality_map,
        "weak_ridge_map": weak_ridge_map,
        "local_contrast_map": local_contrast_map,
        "fusion_weight_map": fusion_weight_map,
        "selected_output": selected_output,
        "selection_reason": selection_reason,
        "traditional_quality_score": traditional_quality_score,
        "improved_quality_score": improved_quality_score,
        "quality_score_components": {
            "traditional_rhlt": traditional_components,
            "improved_rhlt": improved_components,
        },
        "mean_fusion_weight": mean_fusion_weight,
        "maximum_fusion_weight": maximum_fusion_weight,
        "enhanced_foreground_percent": enhanced_foreground_percent,
        "valid_frequency_blocks": valid_frequency_blocks,
        # ── Per-candidate metrics (new) ───────────────────────────────────────
        "traditional_rhlt_metrics": baseline_metrics,
        "improved_rhlt_metrics": improved_metrics,
        # ── Final selected output (backward-compatible keys) ──────────────────
        "ridge_restored": ridge_restored,
        "enhanced_image": ridge_restored,
        "ridge_enhanced": ridge_restored,
        "ridge_binary": ridge_binary,
        "binary": ridge_binary,
        "skeleton": skeleton,
        "endings": endings,
        "bifurcations": bifurcations,
        "minutiae_overlay": overlay,
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
    mask = fingerprint_foreground_mask(denoised, config.block_size)

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
    Parameter study comparing RHLT variants.

    Variants separate apodisation, orientation-only support and the proposed
    quality-and-frequency-adaptive method. Candidate metrics are reported
    directly, rather than metrics from a possible selector fallback.
    """
    base_no_apo = replace(base, apodisation=0.00, hybrid_gabor_max_weight=0.0)
    windowed = replace(base, apodisation=0.25, hybrid_gabor_max_weight=0.0)
    orientation_only = replace(
        base, apodisation=0.00, use_local_frequency=False,
        use_quality_adaptive_fusion=False,
    )
    proposed = replace(
        base, apodisation=0.00, use_local_frequency=True,
        use_quality_adaptive_fusion=True,
    )

    variants = [
        ("Traditional RHLT baseline", base_no_apo, "traditional"),
        ("RHLT with apodisation", windowed, "traditional"),
        ("RHLT + orientation-only Gabor support", orientation_only, "improved"),
        ("Proposed quality/frequency-adaptive RHLT", proposed, "improved"),
    ]
    rows = []
    for label, cfg, candidate in variants:
        result = run_rhlt(image, cfg)
        m = (
            result["traditional_rhlt_metrics"]
            if candidate == "traditional"
            else result["improved_rhlt_metrics"]
        )
        rows.append({
            "variant": label,
            "apodisation": cfg.apodisation,
            "hybrid_gabor_max_weight": cfg.hybrid_gabor_max_weight,
            "local_frequency": cfg.use_local_frequency,
            "quality_adaptive": cfg.use_quality_adaptive_fusion,
            "selected_output": result["selected_output"],
            "candidate_quality_score": (
                result["traditional_quality_score"]
                if candidate == "traditional"
                else result["improved_quality_score"]
            ),
            **m,
        })
    return rows
