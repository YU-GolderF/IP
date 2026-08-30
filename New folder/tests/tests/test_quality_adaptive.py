"""Scientific-behaviour tests for quality/frequency-adaptive Improved RHLT."""

from __future__ import annotations

from dataclasses import replace

import numpy as np

from algorithms.rhlt.config import RHLTConfig
from algorithms.rhlt.frequency import estimate_local_ridge_wavelength
from algorithms.rhlt.pipeline import run_rhlt
from algorithms.rhlt.quality import defect_aware_fusion_weight, local_ridge_quality_maps
from algorithms.rhlt.ridge_filter import enhance_ridges_with_gabor
from core.degradation import degrade_fingerprint
from core.metrics import calculate_image_metrics


def sinusoidal_ridges(
    height: int = 96,
    width: int = 96,
    wavelength: float = 6.0,
    amplitude: float = 70.0,
) -> np.ndarray:
    y, _ = np.mgrid[0:height, 0:width]
    image = 128.0 + amplitude * np.sin(2.0 * np.pi * y / wavelength)
    return np.clip(image, 0, 255).astype(np.uint8)


def compact_config(**changes) -> RHLTConfig:
    base = RHLTConfig(
        psf_size=33,
        block_size=8,
        orientation_bins=8,
        gabor_kernel_size=7,
        gabor_sigma=1.2,
        gabor_lambda=4.0,
        frequency_block_size=16,
        minimum_ridge_wavelength=2.5,
        maximum_ridge_wavelength=7.0,
        frequency_bins=4,
        min_component_area=4,
    )
    return replace(base, **changes)


def test_local_frequency_estimation_recovers_known_wavelength():
    image = sinusoidal_ridges(wavelength=6.0)
    orientation = np.zeros((6, 6), dtype=np.float32)  # horizontal ridge flow
    valid_orientation = np.ones_like(orientation, dtype=bool)
    foreground = np.ones(image.shape, dtype=bool)

    wavelength, valid, confidence = estimate_local_ridge_wavelength(
        image,
        orientation,
        valid_orientation,
        foreground,
        orientation_block_size=16,
        analysis_block_size=32,
        minimum_wavelength=3.0,
        maximum_wavelength=12.0,
        smoothing_size=3,
        fallback_wavelength=10.0,
    )

    assert valid.any()
    assert abs(float(np.median(wavelength[valid])) - 6.0) <= 1.0
    assert np.all((confidence >= 0.0) & (confidence <= 1.0))


def test_weak_ridge_map_increases_as_local_contrast_decreases():
    foreground = np.ones((96, 96), dtype=bool)
    clear = sinusoidal_ridges(amplitude=80.0)
    weak = sinusoidal_ridges(amplitude=12.0)
    _, clear_weakness, _ = local_ridge_quality_maps(clear, foreground)
    _, weak_weakness, _ = local_ridge_quality_maps(weak, foreground)
    assert float(np.mean(weak_weakness)) > float(np.mean(clear_weakness))


def test_defect_weight_is_bounded_monotonic_and_respects_reliability():
    settings = compact_config(hybrid_gabor_max_weight=0.4)
    shape = (32, 32)
    foreground = np.ones(shape, dtype=bool)
    foreground[:, :4] = False
    coherence = np.full((4, 4), 0.8, dtype=np.float32)
    valid = np.ones((4, 4), dtype=bool)
    rhlt = np.full(shape, 64, dtype=np.uint8)

    clear_weight = defect_aware_fusion_weight(
        np.full(shape, 0.2, dtype=np.float32), foreground, coherence, valid, rhlt, settings
    )
    weak_weight = defect_aware_fusion_weight(
        np.full(shape, 0.8, dtype=np.float32), foreground, coherence, valid, rhlt, settings
    )
    unreliable_weight = defect_aware_fusion_weight(
        np.full(shape, 0.8, dtype=np.float32), foreground, coherence, np.zeros_like(valid), rhlt, settings
    )

    assert float(np.mean(weak_weight[foreground])) > float(np.mean(clear_weight[foreground]))
    assert np.all(weak_weight[~foreground] == 0.0)
    assert np.all(unreliable_weight == 0.0)
    assert float(weak_weight.min()) >= 0.0
    assert float(weak_weight.max()) <= settings.hybrid_gabor_max_weight + 1e-7


def test_adaptive_gabor_preserves_shape_dtype_and_background():
    image = sinusoidal_ridges(64, 64, wavelength=5.0)
    foreground = np.zeros(image.shape, dtype=bool)
    foreground[8:56, 8:56] = True
    orientation = np.zeros((8, 8), dtype=np.float32)
    valid = np.ones((8, 8), dtype=bool)
    wavelengths = np.full((8, 8), 5.0, dtype=np.float32)
    frequency_valid = np.ones((8, 8), dtype=bool)
    output = enhance_ridges_with_gabor(
        image,
        orientation,
        foreground,
        compact_config(),
        valid_blocks=valid,
        base_image=image,
        wavelength_field=wavelengths,
        frequency_valid_blocks=frequency_valid,
    )
    assert output.shape == image.shape
    assert output.dtype == np.uint8
    assert np.array_equal(output[~foreground], image[~foreground])
    assert int(output.min()) >= 0 and int(output.max()) <= 255


def test_local_wavelength_changes_adaptive_gabor_kernel_selection():
    image = sinusoidal_ridges(64, 64, wavelength=5.0)
    foreground = np.ones(image.shape, dtype=bool)
    orientation = np.zeros((8, 8), dtype=np.float32)
    valid = np.ones((8, 8), dtype=bool)
    settings = compact_config()
    short_period = enhance_ridges_with_gabor(
        image,
        orientation,
        foreground,
        settings,
        valid_blocks=valid,
        wavelength_field=np.full((8, 8), 3.0, dtype=np.float32),
        frequency_valid_blocks=valid,
    )
    long_period = enhance_ridges_with_gabor(
        image,
        orientation,
        foreground,
        settings,
        valid_blocks=valid,
        wavelength_field=np.full((8, 8), 7.0, dtype=np.float32),
        frequency_valid_blocks=valid,
    )
    assert not np.array_equal(short_period, long_period)


def test_controlled_degradation_is_deterministic_and_does_not_touch_input():
    clean = sinusoidal_ridges()
    before = clean.copy()
    first = degrade_fingerprint(clean, "Medium", seed=19)
    second = degrade_fingerprint(clean, "Medium", seed=19)
    different_seed = degrade_fingerprint(clean, "Medium", seed=20)
    assert np.array_equal(first, second)
    assert not np.array_equal(first, different_seed)
    assert np.array_equal(clean, before)


def test_reference_metrics_are_separate_and_correct():
    clean = sinusoidal_ridges()
    degraded = degrade_fingerprint(clean, "Mild", seed=7)
    identical = calculate_image_metrics(degraded, clean, reference=clean)
    assert identical["ssim_reference"] > 0.999
    assert identical["mse_reference"] == 0.0
    assert identical["psnr_reference"] == float("inf")
    assert "foreground_ssim_reference" in identical


def test_selector_falls_back_when_improved_has_material_score_regression(monkeypatch):
    image = sinusoidal_ridges(64, 64, wavelength=5.0)
    import algorithms.rhlt.pipeline as pipeline_module

    scores = iter((0.80, 0.60))

    def controlled_score(metrics, structure, original_structure, settings):
        return next(scores), {"safe": True}

    monkeypatch.setattr(pipeline_module, "_candidate_quality_score", controlled_score)
    result = run_rhlt(image, compact_config(selector_regression_tolerance=0.05))
    assert result["selected_output"] == "traditional_rhlt_baseline"


def test_selector_uses_safe_improved_when_scores_are_effectively_tied(monkeypatch):
    image = sinusoidal_ridges(64, 64, wavelength=5.0)
    import algorithms.rhlt.pipeline as pipeline_module

    scores = iter((0.50, 0.49))

    def controlled_score(metrics, structure, original_structure, settings):
        return next(scores), {"safe": True}

    monkeypatch.setattr(pipeline_module, "_candidate_quality_score", controlled_score)
    result = run_rhlt(image, compact_config(selector_regression_tolerance=0.05))
    assert result["selected_output"] == "improved_rhlt"


def test_quality_adaptive_result_contract_and_rhlt_dependency(monkeypatch):
    image = sinusoidal_ridges(64, 64, wavelength=5.0)
    import algorithms.rhlt.pipeline as pipeline_module

    def controlled_apply(scale: int):
        def fake_apply(gray, psf, mask):
            gradient = np.tile(np.linspace(0, scale, gray.shape[1]), (gray.shape[0], 1))
            edge = np.clip(gradient, 0, 255).astype(np.uint8)
            edge[~mask] = 0
            return edge.astype(np.complex128), edge
        return fake_apply

    monkeypatch.setattr(pipeline_module, "apply_rhlt", controlled_apply(40))
    low = run_rhlt(image, compact_config())
    monkeypatch.setattr(pipeline_module, "apply_rhlt", controlled_apply(220))
    high = run_rhlt(image, compact_config())

    required = {
        "local_quality_map",
        "weak_ridge_map",
        "fusion_weight_map",
        "local_frequency_map",
        "frequency_valid_blocks",
        "selection_reason",
        "quality_score_components",
    }
    assert required.issubset(high)
    assert not np.array_equal(low["traditional_rhlt_baseline"], high["traditional_rhlt_baseline"])
    assert not np.array_equal(low["improved_rhlt"], high["improved_rhlt"])
