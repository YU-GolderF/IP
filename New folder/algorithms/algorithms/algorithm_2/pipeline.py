"""Algorithm 2 interface: independent DCT contextual filtering plus downstream evaluation."""
from __future__ import annotations

from time import perf_counter

import numpy as np

from algorithms.rhlt.metrics import metric_bundle
from algorithms.rhlt.orientation import estimate_orientation_field, smooth_orientation_field, visualise_orientation_field
from algorithms.rhlt.postprocess import binarise_dark_ridges, clean_binary, crossing_number_minutiae, make_skeleton, minutiae_overlay
from core.metrics import calculate_image_metrics
from core.preprocessing import ensure_uint8

from .algorithm import run_dct_contextual_enhancement
from .config import DCTContextualConfig

PIPELINE_BUILD = "dct-contextual-filtering-v1"
ALGORITHM_NAME = "DCT-based Contextual Filtering"


def _evaluate(original: np.ndarray, candidate: dict) -> dict:
    enhanced, foreground = candidate["image"], candidate["foreground"]
    raw, valid, coherence = estimate_orientation_field(enhanced, foreground, block_size=16)
    orientation = smooth_orientation_field(raw, valid, sigma=1.0)
    binary = clean_binary(binarise_dark_ridges(enhanced, foreground), min_component_area=10)
    skeleton = make_skeleton(binary)
    endings, bifurcations = crossing_number_minutiae(skeleton, foreground, border=10, min_distance=8)
    metrics = calculate_image_metrics(original, enhanced, foreground_mask=foreground)
    metrics.update(metric_bundle(enhanced, foreground, len(endings), len(bifurcations), candidate["processing_time_ms"]))
    metrics.update({"foreground_coverage_percent": float(foreground.mean() * 100.0), "valid_orientation_blocks": int(valid.sum()), "mean_orientation_coherence": float(coherence[valid].mean()) if np.any(valid) else 0.0})
    candidate.update({"metrics": metrics, "orientation_field": orientation, "orientation_block_mask": valid, "orientation_coherence": coherence, "orientation_visualisation": visualise_orientation_field(enhanced, orientation, valid, block_size=16), "ridge_binary": binary, "skeleton": skeleton, "endings": endings, "bifurcations": bifurcations, "minutiae_overlay": minutiae_overlay(enhanced, endings, bifurcations)})
    return candidate


def run_algorithm(image: np.ndarray, *, preprocessing_config=None, config: DCTContextualConfig | None = None) -> dict:
    started = perf_counter(); original = ensure_uint8(image); settings = config or DCTContextualConfig()
    variants = ("basic_dct", "adaptive_frequency", "confidence_aware", "proposed")
    results = {name: _evaluate(original, run_dct_contextual_enhancement(original, settings, name, preprocessing_config)) for name in variants}
    proposed = results["proposed"]
    baseline = {"image": proposed["grayscale"], "metrics": calculate_image_metrics(proposed["grayscale"], proposed["grayscale"], foreground_mask=proposed["foreground"]), "processing_time_ms": 0.0}
    results = {"baseline": baseline, **results}; enhanced = proposed["image"]
    shared = proposed["shared_preprocessing_stages"]
    stages = {"grayscale": shared["grayscale"], "gaussian_denoised": shared["gaussian_denoised"], "denoised": shared["denoised"], "normalised": shared["normalised"], "shared_enhanced": shared["enhanced"], "foreground": proposed["foreground"].astype(np.uint8) * 255, "dct_contextual": proposed["dct_contextual"], "enhanced": enhanced, "wavelet_branch": proposed["dct_contextual"], "ridgelet_branch": proposed["dct_contextual"], "fusion_pre_wiener": proposed["dct_contextual"]}
    return {"pipeline_build": PIPELINE_BUILD, "algorithm_name": ALGORITHM_NAME, "status": "ok", "warnings": ["Shared orientation, skeleton and minutiae are downstream evaluation only; they never modify the DCT result."], "original": original, "grayscale": proposed["grayscale"], "preprocessing_stages": stages, "preprocessed": proposed["normalised"], "normalised": proposed["normalised"], "denoised": enhanced, "foreground_mask": proposed["foreground"], "mask": proposed["foreground"], "dct_orientation_map": proposed["dct_orientation_map"], "frequency_map": proposed["frequency_map"], "confidence_map": proposed["confidence_map"], "quality_map": proposed["quality_map"], "orientation_field": proposed["orientation_field"], "orientation_block_mask": proposed["orientation_block_mask"], "orientation_coherence": proposed["orientation_coherence"], "orientation_visualisation": proposed["orientation_visualisation"], "enhanced_image": enhanced, "ridge_enhanced": enhanced, "ridge_restored": enhanced, "ridge_binary": proposed["ridge_binary"], "binary": proposed["ridge_binary"], "skeleton": proposed["skeleton"], "endings": proposed["endings"], "bifurcations": proposed["bifurcations"], "minutiae_overlay": proposed["minutiae_overlay"], "metrics": proposed["metrics"], "processing_time_ms": (perf_counter() - started) * 1000.0, "method_results": results}
