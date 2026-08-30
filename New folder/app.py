from __future__ import annotations

import importlib
import re
import sys
from io import BytesIO
from pathlib import Path
from zipfile import ZIP_DEFLATED, ZipFile

# Packages live one directory deeper (e.g. algorithms/algorithms).
_ROOT = Path(__file__).resolve().parent
for _pkg in ("algorithms", "core", "reporting"):
    _path = str(_ROOT / _pkg)
    if _path not in sys.path:
        sys.path.insert(0, _path)

import pandas as pd
import numpy as np
import streamlit as st

from algorithms import ALGORITHM_STATUS, get_algorithm_runner
from algorithms.rhlt import config as rhlt_config_module
from algorithms.rhlt import pipeline as rhlt_pipeline_module
from algorithms.rhlt import ridge_filter as rhlt_ridge_filter_module
from core import metrics as core_metrics_module

# Streamlit can rerun app.py while retaining imported project modules. Reload the
# implementation modules in dependency order so code changes cannot leave the UI
# on a new build while run_rhlt still points at an older in-memory function.
core_metrics_module = importlib.reload(core_metrics_module)
# Configuration must be reloaded before consumers. Otherwise a Streamlit hot
# rerun can construct an old RHLTConfig object and pass it to a new pipeline.
rhlt_config_module = importlib.reload(rhlt_config_module)
importlib.reload(rhlt_ridge_filter_module)
rhlt_pipeline_module = importlib.reload(rhlt_pipeline_module)
RHLTConfig = rhlt_config_module.RHLTConfig
run_rhlt = rhlt_pipeline_module.run_rhlt
run_ablation = rhlt_pipeline_module.run_ablation
from core import (
    CalibrationConfig,
    DEGRADATION_PRESETS,
    PreprocessingConfig,
    calibrate_image,
    degrade_fingerprint,
    load_images_from_folder,
    load_multiple_images,
    process_batch,
)
from reporting import build_pdf_report, encode_png

st.set_page_config(
    page_title="Fingerprint Enhancement System", page_icon="🔬", layout="wide"
)
st.title("Fingerprint Enhancement System")
APP_BUILD = "rhlt-primary-quality-adaptive-v3.2-2026-08-29"
st.caption(
    "Shared preprocessing, calibration, batch ingestion and quality metrics with "
    "pluggable team algorithms. RHLT Ridge Flow Restoration is currently available."
)
st.caption(f"Build: `{APP_BUILD}`")
st.caption(f"Algorithm source: `{Path(rhlt_pipeline_module.__file__).resolve()}`")

available_algorithms = [item["name"] for item in ALGORITHM_STATUS if item["available"]]

with st.sidebar:
    st.header("Algorithm")
    selected_algorithm = st.selectbox("Select algorithm", available_algorithms)
    for algorithm in ALGORITHM_STATUS:
        status = "Available" if algorithm["available"] else "Reserved"
        st.caption(f"{algorithm['name']} · {algorithm['owner']} · {status}")

    st.header("1. Fingerprint input")
    uploads = st.file_uploader(
        "Upload additional images (Optional)",
        type=["png", "jpg", "jpeg", "bmp", "tif", "tiff"],
        accept_multiple_files=True,
        help="A corrupted image is reported and skipped without stopping the batch.",
    )

# Algorithm parameters are deliberately code-owned. End users select an algorithm
# and provide images; they are not expected to tune its internal image-processing
# pipeline for every upload.
preprocessing_config = PreprocessingConfig()
calibration_config = CalibrationConfig()
rhlt_config = RHLTConfig()


@st.cache_data(show_spinner=False)
def load_uploaded_batch(payloads: tuple[tuple[str, bytes], ...]):
    return load_multiple_images(payloads)


@st.cache_data(show_spinner=False)
def load_folder_batch(path: str, directory_signature: tuple):
    _ = directory_signature
    return load_images_from_folder(path)


def folder_signature(path: Path) -> tuple[tuple[str, int, int], ...]:
    """Invalidate the folder cache when an image is added, replaced or removed."""
    supported = {".png", ".jpg", ".jpeg", ".bmp", ".tif", ".tiff"}
    return tuple(
        (item.name, item.stat().st_size, item.stat().st_mtime_ns)
        for item in sorted(path.iterdir(), key=lambda value: value.name.lower())
        if item.is_file() and item.suffix.lower() in supported
    )


def is_primary_experiment_image(filename: str) -> bool:
    """Select the 13 original BMP samples and deliberately exclude 000*.png."""
    return (
        re.fullmatch(
            r"(?:left[1-5]|right[1-5]|special[12]|spectial3)\.bmp",
            filename,
            flags=re.IGNORECASE,
        )
        is not None
    )


def diagnostic_map_to_uint8(
    values: np.ndarray, mask: np.ndarray | None = None
) -> np.ndarray:
    array = np.asarray(values, dtype=np.float32)
    valid = np.asarray(mask, dtype=bool) if mask is not None else np.isfinite(array)
    valid &= np.isfinite(array)
    output = np.zeros(array.shape, dtype=np.uint8)
    sample = array[valid]
    if sample.size:
        low, high = float(np.min(sample)), float(np.max(sample))
        if high > low + 1e-9:
            output[valid] = np.clip(
                (array[valid] - low) * 255.0 / (high - low), 0, 255
            ).astype(np.uint8)
    return output


def fingerprint_region(
    image: np.ndarray, mask: np.ndarray, padding: int = 6
) -> np.ndarray:
    foreground = np.asarray(mask, dtype=bool)
    if foreground.shape != np.asarray(image).shape[:2] or not np.any(foreground):
        return np.asarray(image)
    rows, columns = np.where(foreground)
    y0, y1 = max(0, int(rows.min()) - padding), min(
        foreground.shape[0], int(rows.max()) + padding + 1
    )
    x0, x1 = max(0, int(columns.min()) - padding), min(
        foreground.shape[1], int(columns.max()) + padding + 1
    )
    return np.asarray(image)[y0:y1, x0:x1]


def amplified_difference(
    first: np.ndarray, second: np.ndarray, gain: float
) -> np.ndarray:
    a = np.asarray(first, dtype=np.float32)
    b = np.asarray(second, dtype=np.float32)
    return np.clip(np.abs(a - b) * float(gain), 0, 255).astype(np.uint8)


def reference_restoration_score(metrics: dict) -> float:
    """Transparent clean-reference score used only for research conclusions."""
    ssim = float(
        metrics.get("foreground_ssim_reference", metrics.get("ssim_reference", 0.0))
    )
    psnr = float(
        metrics.get("foreground_psnr_reference", metrics.get("psnr_reference", 0.0))
    )
    mse = float(
        metrics.get("foreground_mse_reference", metrics.get("mse_reference", 65025.0))
    )
    return float(
        0.50 * np.clip(ssim, 0.0, 1.0)
        + 0.30 * np.clip(psnr / 40.0, 0.0, 1.0)
        + 0.20 * (1.0 - np.clip(mse / 65025.0, 0.0, 1.0))
    )


def selected_output_label(result: dict) -> str:
    """Return a short, user-facing label for the actual selected image."""
    return {
        "improved_rhlt": "Proposed Improved RHLT",
        "traditional_rhlt_baseline": "Traditional RHLT (safety fallback)",
        "original_quality_fallback": "Original (quality fallback)",
    }.get(
        result.get("selected_output"),
        str(result.get("algorithm_name", "Enhanced output")),
    )


def comparison_quality_score(metrics: dict) -> float:
    """Balanced no-reference score for the multi-algorithm UI (0..100)."""
    original_rvc = max(float(metrics.get("original_ridge_valley_clarity", 0.0)), 1e-6)
    processed_rvc = float(metrics.get("processed_ridge_valley_clarity", 0.0))
    original_edge = max(float(metrics.get("original_edge_clarity", 0.0)), 1e-6)
    processed_edge = float(metrics.get("processed_edge_clarity", 0.0))
    contrast = np.clip(float(metrics.get("cii", 1.0)) / 1.5, 0.0, 1.0)
    rvc = np.clip((processed_rvc / original_rvc) / 1.5, 0.0, 1.0)
    edge = np.clip((processed_edge / original_edge) / 1.5, 0.0, 1.0)
    structural = np.clip(float(metrics.get("ssim", 0.0)), 0.0, 1.0)
    return float(
        100.0 * (0.15 * contrast + 0.25 * rvc + 0.20 * edge + 0.40 * structural)
    )


@st.cache_data(show_spinner=False)
def run_controlled_experiment(
    payloads: tuple[tuple[str, bytes], ...],
    levels: tuple[str, ...],
    seed: int,
    algorithm_config: RHLTConfig,
    shared_config: PreprocessingConfig,
) -> tuple[list[dict], list[dict[str, str]]]:
    clean_images, load_errors = load_multiple_images(payloads)
    records: list[dict] = []
    errors = list(load_errors)
    for clean in clean_images:
        for level in levels:
            try:
                degraded = degrade_fingerprint(clean.image, level, seed=seed)
                result = run_rhlt(
                    degraded,
                    algorithm_config,
                    preprocessing_config=shared_config,
                    reference_image=clean.image,
                )
                traditional_metrics = result["traditional_rhlt_metrics"]
                improved_metrics = result["improved_rhlt_metrics"]
                records.append(
                    {
                        "filename": clean.filename,
                        "level": level,
                        "clean": clean.image,
                        "degraded": degraded,
                        "result": result,
                        "traditional_reference_score": reference_restoration_score(
                            traditional_metrics
                        ),
                        "improved_reference_score": reference_restoration_score(
                            improved_metrics
                        ),
                    }
                )
            except Exception as exc:
                errors.append({"filename": clean.filename, "error": f"{level}: {exc}"})
    return records, errors


uploaded_payloads = tuple((uploaded.name, uploaded.getvalue()) for uploaded in uploads)
loaded_images, input_errors = load_uploaded_batch(uploaded_payloads)
loaded_images = list(loaded_images)
input_errors = list(input_errors)

default_image_folder = (Path(__file__).resolve().parents[1] / "image").resolve()
if default_image_folder.exists() and default_image_folder.is_dir():
    try:
        folder_images, folder_errors = load_folder_batch(
            str(default_image_folder), folder_signature(default_image_folder)
        )
        loaded_images.extend(folder_images)
        input_errors.extend(folder_errors)
    except (OSError, ValueError) as exc:
        st.error(f"Folder input: {exc}")

if input_errors:
    st.warning(f"{len(input_errors)} input file(s) were skipped.")
    st.dataframe(pd.DataFrame(input_errors), width="stretch", hide_index=True)

if not loaded_images:
    st.info(
        "Upload one or more fingerprint images, or enter a local Windows folder "
        "containing supported image files."
    )
    st.stop()


def _run_selected_algorithm(
    calibrated,
    algo_name,
    *,
    final_only=False,
):
    """Dispatch to the correct algorithm runner based on the selected name."""

    if algo_name == "RHLT":
        return run_rhlt(
            calibrated,
            rhlt_config,
            preprocessing_config=preprocessing_config,
        )

    runner = get_algorithm_runner(algo_name)

    # The four-member comparison should run only the final
    # selected Polynomial UM pipeline.
    if algo_name == "Unsharp Masking":
        return runner(
            calibrated,
            preprocessing_config=preprocessing_config,
            final_only=final_only,
        )

    return runner(
        calibrated,
        preprocessing_config=preprocessing_config,
    )


def process_loaded_image(item):
    calibration = calibrate_image(item.image, calibration_config)
    calibrated = calibration["image"]
    result = _run_selected_algorithm(calibrated, selected_algorithm)
    preprocessing_stages = result["preprocessing_stages"]

    result["source_original"] = item.image
    result["preprocessing_stages"] = preprocessing_stages
    result["grayscale"] = preprocessing_stages["grayscale"]
    result["denoised"] = preprocessing_stages["denoised"]
    result["normalised"] = preprocessing_stages["normalised"]
    result["preprocessed"] = preprocessing_stages["enhanced"]
    result["calibration"] = {
        key: value for key, value in calibration.items() if key != "image"
    }
    return result


with st.spinner(f"Processing {len(loaded_images)} fingerprint image(s)..."):
    records, processing_errors = process_batch(
        loaded_images, selected_algorithm, process_loaded_image
    )

if processing_errors:
    st.warning(f"{len(processing_errors)} image(s) failed during processing.")
    st.dataframe(pd.DataFrame(processing_errors), width="stretch", hide_index=True)

if not records:
    st.error("No fingerprint image could be processed.")
    st.stop()

names = [record["filename"] for record in records]
selected_name = st.selectbox("Selected fingerprint", names)
selected_record = records[names.index(selected_name)]
selected = selected_record["result"]
st.caption(f"Active pipeline: `{selected.get('pipeline_build', 'legacy/unknown')}`")

for warning in selected["warnings"]:
    st.warning(warning)

metrics = selected["metrics"]

summary_rows = []
for record in records:
    result = record["result"]
    result_metrics = result["metrics"]
    calibration = result["calibration"]
    summary_rows.append(
        {
            "filename": record["filename"],
            "algorithm": record["algorithm"],
            "original_dimensions": f"{calibration['original_dimensions'][0]} × {calibration['original_dimensions'][1]}",
            "processed_dimensions": f"{calibration['processed_dimensions'][0]} × {calibration['processed_dimensions'][1]}",
            "processing_time_ms": record["processing_time_ms"],
            "original_contrast": result_metrics["original_contrast"],
            "enhanced_contrast": result_metrics["processed_contrast"],
            "original_sharpness": result_metrics["original_sharpness"],
            "enhanced_sharpness": result_metrics["processed_sharpness"],
            "foreground_coverage_percent": result_metrics.get(
                "foreground_coverage_percent", 0.0
            ),
            "valid_orientation_blocks": result_metrics.get(
                "valid_orientation_blocks", 0
            ),
        }
    )
summary = pd.DataFrame(summary_rows)

tabs = st.tabs(
    [
        "📊 Overview",
        "🔬 Ridge Orientation",
        "⚙️ Pipeline Internals",
        "🧪 RHLT Algorithm Internals",
        "📁 Data Dashboard",
        "🔭 Research Experiment",
        "🏆 Algorithm Comparison",
    ]
)

# ── Tab 0: Overview ─────────────────────────────────────────────────────────
with tabs[0]:
    # --- KPI headline cards ---
    cii_val = float(metrics.get("cii", 1.0))
    ssim_val = float(metrics.get("ssim", 0.0))
    sharpness_pct = float(metrics.get("sharpness_improvement_pct", 0.0))
    edge_pct = float(metrics.get("edge_improvement_pct", 0.0))
    rvc_orig = float(metrics.get("original_ridge_valley_clarity", 0.0))
    rvc_proc = float(metrics.get("processed_ridge_valley_clarity", 0.0))
    rvc_pct = (rvc_proc - rvc_orig) / max(rvc_orig, 1e-6) * 100.0
    coherence_val = float(metrics.get("mean_orientation_coherence", 0.0))
    minutiae_total = int(metrics.get("minutiae_total", 0))
    endings_count = int(metrics.get("ridge_endings", 0))
    bifurcations_count = int(metrics.get("bifurcations", 0))

    if selected_algorithm == "RHLT":
        st.success(
            f"Recommended output: **{selected_output_label(selected)}** — "
            f"{selected.get('selection_reason', 'quality checks completed.')}"
        )

    kpi_cols = st.columns(4)
    kpi_cols[0].metric(
        "Contrast Improvement (CII)",
        f"{cii_val:.2f}×",
        delta=f"{(cii_val - 1.0) * 100:+.1f}%" if cii_val != 1.0 else None,
        help="Contrast Improvement Index: processed / original contrast. >1.0 = improved.",
    )
    kpi_cols[1].metric(
        "Sharpness Improvement",
        f"{sharpness_pct:+.1f}%",
        delta=f"{sharpness_pct:+.1f}%",
        delta_color="normal",
        help="Laplacian sharpness improvement relative to original.",
    )
    kpi_cols[2].metric(
        "SSIM (Structural Similarity)",
        f"{ssim_val:.3f}",
        help="SSIM vs original. Near 1.0 = structure well preserved. <0.80 may indicate distortion.",
        delta=None,
    )
    kpi_cols[3].metric(
        "Ridge-Valley Clarity",
        f"{rvc_pct:+.1f}%",
        delta=f"{rvc_pct:+.1f}%",
        delta_color="normal",
        help="Foreground Laplacian variance improvement — fingerprint-specific sharpness.",
    )

    st.divider()

    # --- Side-by-side image comparison ---
    st.subheader("Visual comparison")
    if selected_algorithm == "RHLT" and "traditional_rhlt_baseline" in selected:
        selected_output = selected.get("selected_output", "original_quality_fallback")
        fallback_selected = selected_output != "improved_rhlt"
        columns = st.columns(4 if fallback_selected else 3)
        col_orig, col_trad, col_imp = columns[:3]
        col_orig.image(
            selected["source_original"],
            caption="Original",
            use_container_width=True,
        )
        col_trad.image(
            selected["traditional_rhlt_baseline"],
            caption="Traditional RHLT Baseline",
            clamp=True,
            use_container_width=True,
        )
        col_imp.image(
            selected["improved_rhlt"],
            caption=(
                "Proposed Improved RHLT"
                if fallback_selected
                else "Final Selected: Proposed Improved RHLT"
            ),
            clamp=True,
            use_container_width=True,
        )
        if fallback_selected:
            fallback_labels = {
                "traditional_rhlt_baseline": "Traditional RHLT Baseline",
                "original_quality_fallback": "Original Quality Fallback",
            }
            columns[3].image(
                selected["enhanced_image"],
                caption=f"Final Safety Fallback: {fallback_labels.get(selected_output, selected_output)}",
                clamp=True,
                use_container_width=True,
            )

    elif selected_algorithm == "Unsharp Masking":
        original_column, conventional_column, adaptive_column, polynomial_column = (
            st.columns(4)
        )

        original_column.image(
            selected["source_original"],
            caption="Original Fingerprint",
            use_container_width=True,
        )

        conventional_column.image(
            selected["conventional_unsharp"],
            caption="Conventional Unsharp Masking",
            clamp=True,
            use_container_width=True,
        )

        adaptive_column.image(
            selected["adaptive_unsharp"],
            caption="Enhancement 1: Adaptive UM",
            clamp=True,
            use_container_width=True,
        )

        polynomial_column.image(
            selected["polynomial_unsharp"],
            caption="Final Selected: Nonlinear Polynomial UM",
            clamp=True,
            use_container_width=True,
        )

        st.markdown("### Unsharp Masking Enhancement Comparison")

        conventional_metrics = selected["conventional_unsharp_metrics"]
        adaptive_metrics = selected["adaptive_unsharp_metrics"]
        polynomial_metrics = selected["polynomial_unsharp_metrics"]

        def rvc_change(candidate_metrics):
            original_rvc = max(
                float(candidate_metrics.get("original_ridge_valley_clarity", 0.0)),
                1e-6,
            )
            processed_rvc = float(
                candidate_metrics.get("processed_ridge_valley_clarity", 0.0)
            )
            return (processed_rvc / original_rvc - 1.0) * 100.0

        comparison_data = {
            "Metric": [
                "CII",
                "Sharpness Improvement",
                "Ridge-Valley Clarity Improvement",
                "Edge Clarity Improvement",
                "SSIM",
                "Detected Minutiae",
                "Processing Time",
            ],
            "Conventional UM": [
                f"{conventional_metrics.get('cii', 1.0):.3f}×",
                f"{conventional_metrics.get('sharpness_improvement_pct', 0.0):+.1f}%",
                f"{rvc_change(conventional_metrics):+.1f}%",
                f"{conventional_metrics.get('edge_improvement_pct', 0.0):+.1f}%",
                f"{conventional_metrics.get('ssim', 0.0):.3f}",
                f"{selected.get('conventional_unsharp_minutiae', 0)}",
                f"{selected.get('conventional_unsharp_time_ms', 0.0):.2f} ms",
            ],
            "Adaptive UM": [
                f"{adaptive_metrics.get('cii', 1.0):.3f}×",
                f"{adaptive_metrics.get('sharpness_improvement_pct', 0.0):+.1f}%",
                f"{rvc_change(adaptive_metrics):+.1f}%",
                f"{adaptive_metrics.get('edge_improvement_pct', 0.0):+.1f}%",
                f"{adaptive_metrics.get('ssim', 0.0):.3f}",
                f"{selected.get('adaptive_unsharp_minutiae', 0)}",
                f"{selected.get('adaptive_unsharp_time_ms', 0.0):.2f} ms",
            ],
            "Polynomial UM": [
                f"{polynomial_metrics.get('cii', 1.0):.3f}×",
                f"{polynomial_metrics.get('sharpness_improvement_pct', 0.0):+.1f}%",
                f"{rvc_change(polynomial_metrics):+.1f}%",
                f"{polynomial_metrics.get('edge_improvement_pct', 0.0):+.1f}%",
                f"{polynomial_metrics.get('ssim', 0.0):.3f}",
                f"{selected.get('polynomial_unsharp_minutiae', 0)}",
                f"{selected.get('polynomial_unsharp_time_ms', 0.0):.2f} ms",
            ],
        }

        st.dataframe(
            comparison_data,
            use_container_width=True,
            hide_index=True,
        )

        # Batch-average comparison across all processed fingerprints.
        batch_candidate_rows = []

        for record in records:
            result = record["result"]

            batch_candidate_rows.append(
                {
                    "Method": "Preprocessed Input (Before UM)",
                    "CII": 1.0,
                    "Sharpness Improvement (%)": 0.0,
                    "RVC Improvement (%)": 0.0,
                    "Edge Improvement (%)": 0.0,
                    "SSIM": 1.0,
                    "Detected Minutiae": int(result.get("original_minutiae", 0)),
                    "Processing Time (ms)": np.nan,
                }
            )

            candidates = [
                (
                    "Conventional UM",
                    result["conventional_unsharp_metrics"],
                    result.get("conventional_unsharp_minutiae", 0),
                    result.get("conventional_unsharp_time_ms", 0.0),
                ),
                (
                    "Adaptive UM",
                    result["adaptive_unsharp_metrics"],
                    result.get("adaptive_unsharp_minutiae", 0),
                    result.get("adaptive_unsharp_time_ms", 0.0),
                ),
                (
                    "Polynomial UM",
                    result["polynomial_unsharp_metrics"],
                    result.get("polynomial_unsharp_minutiae", 0),
                    result.get("polynomial_unsharp_time_ms", 0.0),
                ),
            ]

            for method, candidate_metrics, minutiae, time_ms in candidates:
                original_rvc = max(
                    float(
                        candidate_metrics.get(
                            "original_ridge_valley_clarity",
                            0.0,
                        )
                    ),
                    1e-6,
                )

                processed_rvc = float(
                    candidate_metrics.get(
                        "processed_ridge_valley_clarity",
                        0.0,
                    )
                )

                rvc_improvement = (processed_rvc / original_rvc - 1.0) * 100.0

                batch_candidate_rows.append(
                    {
                        "Method": method,
                        "CII": float(candidate_metrics.get("cii", 1.0)),
                        "Sharpness Improvement (%)": float(
                            candidate_metrics.get(
                                "sharpness_improvement_pct",
                                0.0,
                            )
                        ),
                        "RVC Improvement (%)": rvc_improvement,
                        "Edge Improvement (%)": float(
                            candidate_metrics.get(
                                "edge_improvement_pct",
                                0.0,
                            )
                        ),
                        "SSIM": float(candidate_metrics.get("ssim", 0.0)),
                        "Detected Minutiae": int(minutiae),
                        "Processing Time (ms)": float(time_ms),
                    }
                )

        batch_candidate_frame = pd.DataFrame(batch_candidate_rows)

        batch_average = (
            batch_candidate_frame.groupby("Method", sort=False)
            .mean(numeric_only=True)
            .reset_index()
        )

        st.markdown("### Batch Average Comparison")
        st.caption(
            f"Average results across {len(records)} processed fingerprint images."
        )

        batch_display = batch_average.copy()

        batch_display["CII"] = batch_display["CII"].map(lambda value: f"{value:.3f}×")

        batch_display["Sharpness Improvement (%)"] = batch_display[
            "Sharpness Improvement (%)"
        ].map(lambda value: f"{value:+.1f}%")

        batch_display["RVC Improvement (%)"] = batch_display["RVC Improvement (%)"].map(
            lambda value: f"{value:+.1f}%"
        )

        batch_display["Edge Improvement (%)"] = batch_display[
            "Edge Improvement (%)"
        ].map(lambda value: f"{value:+.1f}%")

        batch_display["SSIM"] = batch_display["SSIM"].map(lambda value: f"{value:.3f}")

        batch_display["Detected Minutiae"] = batch_display["Detected Minutiae"].map(
            lambda value: f"{value:.1f}"
        )

        batch_display["Processing Time (ms)"] = batch_display[
            "Processing Time (ms)"
        ].map(lambda value: "N/A" if pd.isna(value) else f"{value:.2f} ms")

        st.dataframe(
            batch_display,
            use_container_width=True,
            hide_index=True,
        )

        # ============================================================
        # Similar algorithm comparison:
        # Sobel-based Sharpening vs Conventional Unsharp Masking
        # ============================================================
        if "sobel_sharpening_metrics" in selected:
            st.markdown("### Similar Algorithm Comparison")
            st.caption(
                "Comparison between Sobel-based Sharpening and "
                "Conventional Unsharp Masking on the selected fingerprint."
            )

            sobel_metrics = selected["sobel_sharpening_metrics"]

            similar_comparison = {
                "Metric": [
                    "CII",
                    "Sharpness Improvement",
                    "Ridge-Valley Clarity Improvement",
                    "Edge Clarity Improvement",
                    "SSIM",
                    "Detected Minutiae",
                    "Processing Time",
                ],
                "Sobel-based Sharpening": [
                    f"{sobel_metrics.get('cii', 1.0):.3f}×",
                    f"{sobel_metrics.get('sharpness_improvement_pct', 0.0):+.1f}%",
                    f"{rvc_change(sobel_metrics):+.1f}%",
                    f"{sobel_metrics.get('edge_improvement_pct', 0.0):+.1f}%",
                    f"{sobel_metrics.get('ssim', 0.0):.3f}",
                    f"{selected.get('sobel_sharpening_minutiae', 0)}",
                    f"{selected.get('sobel_sharpening_time_ms', 0.0):.2f} ms",
                ],
                "Conventional UM": [
                    f"{conventional_metrics.get('cii', 1.0):.3f}×",
                    f"{conventional_metrics.get('sharpness_improvement_pct', 0.0):+.1f}%",
                    f"{rvc_change(conventional_metrics):+.1f}%",
                    f"{conventional_metrics.get('edge_improvement_pct', 0.0):+.1f}%",
                    f"{conventional_metrics.get('ssim', 0.0):.3f}",
                    f"{selected.get('conventional_unsharp_minutiae', 0)}",
                    f"{selected.get('conventional_unsharp_time_ms', 0.0):.2f} ms",
                ],
            }

            st.dataframe(
                similar_comparison,
                use_container_width=True,
                hide_index=True,
            )

    else:
        original_column, enhanced_column = st.columns(2)
        original_column.image(
            selected["source_original"],
            caption="Original fingerprint",
            use_container_width=True,
        )
        enhanced_column.image(
            selected["enhanced_image"],
            caption=f"Enhanced · {selected_algorithm}",
            clamp=True,
            use_container_width=True,
        )

    st.divider()

    # --- Metrics comparison table ---
    evaluated_output = (
        selected_output_label(selected)
        if selected_algorithm == "RHLT"
        else selected_algorithm
    )
    overview_details = st.expander(
        f"Detailed metrics · Original vs {evaluated_output}", expanded=False
    )
    overview_details.caption(
        "In this table, Selected output refers to the exact image returned as enhanced_image, "
        "not automatically to every candidate shown above."
    )
    comparison_metrics = [
        ("Contrast", "original_contrast", "processed_contrast", 4),
        (
            "Ridge-Valley Clarity (RVC)",
            "original_ridge_valley_clarity",
            "processed_ridge_valley_clarity",
            1,
        ),
        ("Sharpness (Laplacian)", "original_sharpness", "processed_sharpness", 1),
        ("Edge Clarity (Sobel)", "original_edge_clarity", "processed_edge_clarity", 2),
    ]
    comparison_rows = []
    for label, original_key, enhanced_key, decimals in comparison_metrics:
        original_value = metrics.get(original_key)
        enhanced_value = metrics.get(enhanced_key)
        if original_value is None or enhanced_value is None:
            continue
        original_value = float(original_value)
        enhanced_value = float(enhanced_value)
        pct_change = (
            (enhanced_value - original_value) / max(abs(original_value), 1e-9) * 100.0
        )
        status = "✅" if enhanced_value >= original_value else "⚠️"
        comparison_rows.append(
            {
                "Metric": label,
                "Original": f"{original_value:.{decimals}f}",
                "Selected output": f"{enhanced_value:.{decimals}f}",
                "Change": f"{enhanced_value - original_value:+.{decimals}f}",
                "Change %": f"{pct_change:+.1f}%",
                "Status": status,
            }
        )
    # Add SSIM row separately
    ssim_status = "Preserved" if ssim_val >= 0.80 else "Distortion risk"
    comparison_rows.append(
        {
            "Metric": "SSIM (Structural Similarity)",
            "Original": "1.000 (reference)",
            "Selected output": f"{ssim_val:.3f}",
            "Change": f"{ssim_val - 1.0:+.3f}",
            "Change %": f"{(ssim_val - 1.0) * 100:+.1f}%",
            "Status": ssim_status,
        }
    )
    overview_details.dataframe(
        pd.DataFrame(comparison_rows), width="stretch", hide_index=True
    )

    # --- Fingerprint structure stats ---
    overview_details.subheader("Fingerprint structure analysis")
    struct_cols = overview_details.columns(4)
    struct_cols[0].metric(
        "Mean Orientation Coherence",
        f"{coherence_val:.3f}",
        help="0 = chaotic gradients (noisy), 1 = perfectly parallel ridges. Higher is better.",
    )
    struct_cols[1].metric(
        "Total Minutiae",
        minutiae_total,
        help="Ridge endings + bifurcations detected on the skeleton (Crossing Number method).",
    )
    struct_cols[2].metric(
        "Ridge Endings",
        endings_count,
        help="Locations where a ridge terminates (endpoint minutiae).",
    )
    struct_cols[3].metric(
        "Bifurcations",
        bifurcations_count,
        help="Locations where a ridge splits into two (branch minutiae).",
    )
    st.caption(
        f"Algorithm: **{selected_algorithm}** · "
        f"Foreground coverage: **{metrics.get('foreground_coverage_percent', 0.0):.1f}%** · "
        f"Valid orientation blocks: **{int(metrics.get('valid_orientation_blocks', 0))}** · "
        f"Processing time: **{selected.get('processing_time_ms', 0.0):.1f} ms**"
    )

# ── Tab 2: Pipeline Internals ─────────────────────────────────────────────────
with tabs[2]:
    st.caption(
        "These are the intermediate processing stages of the shared preprocessing pipeline. "
        "They are intended for technical inspection and debugging — not for evaluating enhancement quality."
    )
    stages = selected["preprocessing_stages"]
    stage_images = [
        ("Grayscale", stages["grayscale"]),
        ("Gaussian denoised", stages["gaussian_denoised"]),
        ("Median / denoised", stages["denoised"]),
        ("Intensity normalised", stages["normalised"]),
        ("CLAHE enhanced", stages["enhanced"]),
        (
            "Fingerprint foreground mask",
            (selected["foreground_mask"].astype("uint8") * 255),
        ),
        ("Ridge flow restored", selected["ridge_restored"]),
        (
            f"Skeleton (minutiae: {int(metrics.get('minutiae_total', 0))})",
            (selected["skeleton"].astype("uint8") * 255),
        ),
    ]
    for start in range(0, len(stage_images), 4):
        columns = st.columns(4)
        for column, (label, image) in zip(columns, stage_images[start : start + 4]):
            column.image(image, caption=label, clamp=True, use_container_width=True)
    if "minutiae_overlay" in selected:
        st.subheader("Minutiae overlay")
        st.caption("Green dots = ridge endings · Red dots = bifurcations")
        st.image(selected["minutiae_overlay"], clamp=True, use_container_width=True)

# ── Tab 1: Ridge Orientation ─────────────────────────────────────────────────
with tabs[1]:
    st.caption(
        "The orientation field estimates the local ridge flow direction across the fingerprint. "
        "It drives the direction-adaptive Gabor filter used for ridge enhancement."
    )
    if "orientation_visualisation" in selected:
        left, right = st.columns([2, 1])
        left.image(
            selected["orientation_visualisation"],
            caption="Smoothed local ridge directions (red lines = estimated ridge flow)",
            use_container_width=True,
        )
        right.metric(
            "Valid Orientation Blocks",
            int(metrics.get("valid_orientation_blocks", 0)),
            help="Number of image blocks where the orientation field was reliably estimated.",
        )
        right.metric(
            "Foreground Coverage",
            f"{metrics.get('foreground_coverage_percent', 0.0):.1f}%",
            help="Fraction of the image identified as fingerprint foreground.",
        )
        right.metric(
            "Mean Orientation Coherence",
            f"{float(metrics.get('mean_orientation_coherence', 0.0)):.3f}",
            help="Structure tensor coherence averaged over valid blocks. 0 = chaotic, 1 = perfectly parallel ridges.",
        )
        right.markdown("""
            **How it works:**
            Sobel gradients estimate the direction *across* each ridge.
            The system rotates that direction by 90° to obtain the ridge *flow*,
            smooths it using doubled angles (to handle the 180° ambiguity),
            and selects the nearest cached Gabor orientation for each valid block.
            """)
    else:
        st.info(f"Ridge orientation data is not provided by {selected_algorithm}.")

# ── Tab 3: RHLT Algorithm Internals ──────────────────────────────────────────
with tabs[3]:
    st.caption(
        "This page shows all stages of the RHLT-primary pipeline: the spiral-phase edge response, "
        "the RHLT-based traditional baseline, the orientation-guided Gabor support component, "
        "and the final proposed improved RHLT image."
    )
    if selected_algorithm == "RHLT" and "rhlt_stretched" in selected:
        st.success(
            f"Recommended: **{selected_output_label(selected)}** — "
            f"{selected.get('selection_reason', 'candidate comparison completed.')}"
        )
        decision_cols = st.columns(3)
        decision_cols[0].metric(
            "Traditional score", f"{selected.get('traditional_quality_score', 0.0):.4f}"
        )
        decision_cols[1].metric(
            "Improved score", f"{selected.get('improved_quality_score', 0.0):.4f}"
        )
        decision_cols[2].metric(
            "Improved fusion applied",
            f"{selected.get('mean_fusion_weight', 0.0) * 100.0:.1f}% avg",
        )

        # Row 1: RHLT diagnostics
        rhlt_diagnostics = st.expander(
            "Technical RHLT response and configuration", expanded=False
        )
        rhlt_diagnostics.subheader("Spiral-phase RHLT diagnostic")
        r1_left, r1_mid, r1_right = rhlt_diagnostics.columns(3)
        r1_left.image(
            selected["rhlt_stretched"],
            caption="Spiral-phase RHLT edge response (isotropic edge magnitude — diagnostic only)",
            clamp=True,
            use_container_width=True,
        )
        r1_mid.image(
            selected["psf_visualisation"],
            caption="RHLT PSF magnitude (donut shape, topological charge l=1)",
            clamp=True,
            use_container_width=True,
        )
        r1_right.markdown(f"""
            **RHLT configuration**

            | Parameter | Value |
            |---|---|
            | PSF size | **{rhlt_config.psf_size} × {rhlt_config.psf_size}** |
            | Aperture ratio | **{rhlt_config.aperture_ratio:.2f}** |
            | Apodisation | **{rhlt_config.apodisation:.2f}** |
            | Topological charge | **{rhlt_config.topological_charge}** |
            | Edge gain | **{rhlt_config.edge_gain:.2f}** |
            | Gabor max weight | **{rhlt_config.hybrid_gabor_max_weight:.2f}** |

            The RHLT edge magnitude drives both the traditional baseline image
            and the per-pixel support weight for the improved RHLT fusion.
            """)
        rhlt_diagnostics.divider()
        # Row 2: Algorithm stages
        st.subheader("Enhancement stages")
        r2_cols = st.columns(3)
        if "traditional_rhlt_baseline" in selected:
            r2_cols[0].image(
                selected["traditional_rhlt_baseline"],
                caption="Traditional RHLT baseline (RHLT edge-guided sharpening)",
                clamp=True,
                use_container_width=True,
            )
        if "gabor_support" in selected:
            r2_cols[1].image(
                selected["gabor_support"],
                caption="Orientation-guided Gabor support (bounded contribution, not final output)",
                clamp=True,
                use_container_width=True,
            )
        if "improved_rhlt" in selected:
            sel_label = selected.get("selected_output", "?")
            r2_cols[2].image(
                selected["improved_rhlt"],
                caption=f"Proposed improved RHLT (RHLT + Gabor fusion) · selected: {sel_label}",
                clamp=True,
                use_container_width=True,
            )
        # Per-candidate metrics
        if (
            "traditional_rhlt_metrics" in selected
            and "improved_rhlt_metrics" in selected
        ):
            candidate_details = st.expander(
                "Detailed Traditional vs Improved metrics", expanded=False
            )
            candidate_details.caption(
                selected.get("selection_reason", "No selection reason was recorded.")
            )
            detail_cols = candidate_details.columns(2)
            detail_cols[0].metric(
                "Maximum fusion weight",
                f"{selected.get('maximum_fusion_weight', 0.0):.3f}",
            )
            detail_cols[1].metric(
                "Valid frequency blocks", int(selected.get("valid_frequency_blocks", 0))
            )
            tm = selected["traditional_rhlt_metrics"]
            im = selected["improved_rhlt_metrics"]
            cand_data = {
                "Metric": ["CII", "Sharpness Δ%", "Edge Clarity Δ%", "SSIM"],
                "Traditional RHLT baseline": [
                    f"{tm.get('cii', 1.0):.3f}×",
                    f"{tm.get('sharpness_improvement_pct', 0.0):+.1f}%",
                    f"{tm.get('edge_improvement_pct', 0.0):+.1f}%",
                    f"{tm.get('ssim', 0.0):.3f}",
                ],
                "Proposed improved RHLT": [
                    f"{im.get('cii', 1.0):.3f}×",
                    f"{im.get('sharpness_improvement_pct', 0.0):+.1f}%",
                    f"{im.get('edge_improvement_pct', 0.0):+.1f}%",
                    f"{im.get('ssim', 0.0):.3f}",
                ],
            }
            candidate_details.dataframe(
                pd.DataFrame(cand_data), width="stretch", hide_index=True
            )
    else:
        st.info("RHLT Algorithm Internals is specific to the RHLT algorithm.")

# ── Tab 4: Data Dashboard ─────────────────────────────────────────────────────
with tabs[4]:
    st.subheader("Batch enhancement summary")
    st.caption(
        "One row per image processed in this session. All key quality indicators are included."
    )
    # Build enriched summary with the new metrics
    summary_rows = []
    for record in records:
        result = record["result"]
        result_metrics = result["metrics"]
        calibration = result["calibration"]
        summary_rows.append(
            {
                "Filename": record["filename"],
                "Algorithm": record["algorithm"],
                "Dimensions (orig)": f"{calibration['original_dimensions'][0]} × {calibration['original_dimensions'][1]}",
                "Dimensions (out)": f"{calibration['processed_dimensions'][0]} × {calibration['processed_dimensions'][1]}",
                "Selected output": selected_output_label(result),
                "CII (contrast)": f"{result_metrics.get('cii', 1.0):.2f}×",
                "Sharpness Δ%": f"{result_metrics.get('sharpness_improvement_pct', 0.0):+.1f}%",
                "Edge Clarity Δ%": f"{result_metrics.get('edge_improvement_pct', 0.0):+.1f}%",
                "SSIM": f"{result_metrics.get('ssim', 0.0):.3f}",
                "Orientation Coherence": f"{result_metrics.get('mean_orientation_coherence', 0.0):.3f}",
                "Minutiae Total": int(result_metrics.get("minutiae_total", 0)),
                "Foreground %": f"{result_metrics.get('foreground_coverage_percent', 0.0):.1f}%",
                "Time (ms)": f"{record['processing_time_ms']:.1f}",
            }
        )
    enriched_summary = pd.DataFrame(summary_rows)
    batch_ssim = [
        float(record["result"]["metrics"].get("ssim", 0.0)) for record in records
    ]
    batch_rvc = [
        (
            float(
                record["result"]["metrics"].get("processed_ridge_valley_clarity", 0.0)
            )
            / max(
                float(
                    record["result"]["metrics"].get(
                        "original_ridge_valley_clarity", 0.0
                    )
                ),
                1e-6,
            )
            - 1.0
        )
        * 100.0
        for record in records
    ]
    batch_time = [float(record["processing_time_ms"]) for record in records]
    batch_cards = st.columns(4)
    batch_cards[0].metric("Processed", len(records))
    batch_cards[1].metric("Average SSIM", f"{np.mean(batch_ssim):.3f}")
    batch_cards[2].metric("Average RVC change", f"{np.mean(batch_rvc):+.1f}%")
    batch_cards[3].metric("Average time", f"{np.mean(batch_time):.1f} ms")

    concise_columns = [
        "Filename",
        "Selected output",
        "CII (contrast)",
        "Sharpness Δ%",
        "SSIM",
        "Time (ms)",
    ]
    st.dataframe(enriched_summary[concise_columns], width="stretch", hide_index=True)
    batch_details = st.expander("View complete batch metrics", expanded=False)
    batch_details.dataframe(enriched_summary, width="stretch", hide_index=True)
    csv_bytes = enriched_summary.to_csv(index=False).encode("utf-8")
    st.download_button(
        "⬇️ Download metrics CSV", csv_bytes, "fingerprint_metrics.csv", "text/csv"
    )

    zip_buffer = BytesIO()
    with ZipFile(zip_buffer, "w", ZIP_DEFLATED) as archive:
        for record in records:
            result = record["result"]
            stem = record["filename"].rsplit(".", 1)[0]
            archive.writestr(
                f"{stem}_ridge_restored.png", encode_png(result["ridge_restored"])
            )
            archive.writestr(
                f"{stem}_ridge_binary.png", encode_png(result["ridge_binary"])
            )
            archive.writestr(
                f"{stem}_orientation.png",
                encode_png(result["orientation_visualisation"]),
            )
            archive.writestr(
                f"{stem}_foreground_mask.png",
                encode_png(result["foreground_mask"]),
            )
        archive.writestr("fingerprint_metrics.csv", csv_bytes)
    st.download_button(
        "⬇️ Download all outputs (ZIP)",
        zip_buffer.getvalue(),
        "fingerprint_outputs.zip",
        "application/zip",
    )

with tabs[5]:
    st.subheader("Controlled degradation with clean ground truth")
    overview_details.caption(
        "The 13 original BMP samples are treated as clean references. Blur, contrast "
        "reduction and Gaussian noise are generated deterministically in memory; files "
        "in the dataset are never modified."
    )
    displayed_level = st.selectbox("Displayed degradation", list(DEGRADATION_PRESETS))
    experiment_settings = st.expander("Experiment settings", expanded=False)
    experiment_controls = experiment_settings.columns(2)
    experiment_seed = int(
        experiment_controls[0].number_input(
            "Deterministic seed", min_value=0, value=7, step=1
        )
    )
    difference_gain = float(
        experiment_controls[1].slider("Difference-map gain", 1.0, 10.0, 4.0, 0.5)
    )
    run_controlled = st.checkbox(
        "Run the 13 original images at Mild, Medium and Severe levels",
        value=False,
    )
    if selected_algorithm == "RHLT" and run_controlled:
        experiment_paths = sorted(
            (
                path
                for path in default_image_folder.iterdir()
                if path.is_file() and is_primary_experiment_image(path.name)
            ),
            key=lambda path: path.name.lower(),
        )
        experiment_payloads = tuple(
            (path.name, path.read_bytes()) for path in experiment_paths
        )
        with st.spinner(
            f"Running {len(experiment_payloads)} images × 3 deterministic degradation levels..."
        ):
            experiment_records, experiment_errors = run_controlled_experiment(
                experiment_payloads,
                tuple(DEGRADATION_PRESETS),
                experiment_seed,
                rhlt_config,
                preprocessing_config,
            )
        if experiment_errors:
            st.warning(f"{len(experiment_errors)} experiment item(s) failed.")
            st.dataframe(pd.DataFrame(experiment_errors), hide_index=True)
        if experiment_records:
            research_rows = []
            selection_rows = []
            improved_wins = 0
            for experiment_record in experiment_records:
                experiment_result = experiment_record["result"]
                traditional_metrics = experiment_result["traditional_rhlt_metrics"]
                proposed_metrics = experiment_result["improved_rhlt_metrics"]
                if (
                    experiment_record["improved_reference_score"]
                    > experiment_record["traditional_reference_score"]
                ):
                    improved_wins += 1
                for candidate_name, candidate_metrics, reference_score in (
                    (
                        "Traditional RHLT",
                        traditional_metrics,
                        experiment_record["traditional_reference_score"],
                    ),
                    (
                        "Proposed Improved RHLT",
                        proposed_metrics,
                        experiment_record["improved_reference_score"],
                    ),
                ):
                    research_rows.append(
                        {
                            "Level": experiment_record["level"],
                            "Filename": experiment_record["filename"],
                            "Candidate": candidate_name,
                            "SSIM reference": candidate_metrics.get(
                                "foreground_ssim_reference", 0.0
                            ),
                            "PSNR reference": candidate_metrics.get(
                                "foreground_psnr_reference", 0.0
                            ),
                            "MSE reference": candidate_metrics.get(
                                "foreground_mse_reference", 0.0
                            ),
                            "Ridge-valley clarity": candidate_metrics.get(
                                "processed_ridge_valley_clarity", 0.0
                            ),
                            "Reference score": reference_score,
                            "Orientation coherence": experiment_result["metrics"].get(
                                "mean_orientation_coherence", 0.0
                            ),
                            "Processing time (ms)": experiment_result[
                                "processing_time_ms"
                            ],
                        }
                    )
                selection_rows.append(
                    {
                        "Level": experiment_record["level"],
                        "Filename": experiment_record["filename"],
                        "Selected output": experiment_result["selected_output"],
                        "Selection reason": experiment_result["selection_reason"],
                        "Orientation coherence": experiment_result["metrics"].get(
                            "mean_orientation_coherence", 0.0
                        ),
                        "Processing time (ms)": experiment_result["processing_time_ms"],
                    }
                )

            research_frame = pd.DataFrame(research_rows)
            grouped = research_frame.groupby(["Level", "Candidate"], sort=False)
            aggregate = grouped.agg(
                {
                    "SSIM reference": ["mean", "std"],
                    "PSNR reference": ["mean", "std"],
                    "MSE reference": ["mean", "std"],
                    "Ridge-valley clarity": ["mean", "std"],
                    "Reference score": ["mean", "std"],
                    "Orientation coherence": ["mean", "std"],
                    "Processing time (ms)": ["mean", "std"],
                }
            ).reset_index()
            aggregate.columns = [
                (
                    " ".join(str(part) for part in column if part).strip()
                    if isinstance(column, tuple)
                    else column
                )
                for column in aggregate.columns
            ]
            overall = (
                research_frame.groupby("Candidate", sort=False)
                .agg(
                    {
                        "SSIM reference": ["mean", "std"],
                        "PSNR reference": ["mean", "std"],
                        "MSE reference": ["mean", "std"],
                        "Ridge-valley clarity": ["mean", "std"],
                        "Reference score": ["mean", "std"],
                        "Orientation coherence": ["mean", "std"],
                        "Processing time (ms)": ["mean", "std"],
                    }
                )
                .reset_index()
            )
            overall.insert(0, "Level", "Overall")
            overall.columns = aggregate.columns
            aggregate = pd.concat([aggregate, overall], ignore_index=True)

            overall_means = research_frame.groupby("Candidate", sort=False).mean(
                numeric_only=True
            )
            traditional_overall = overall_means.loc["Traditional RHLT"]
            improved_overall = overall_means.loc["Proposed Improved RHLT"]
            total_experiments = len(experiment_records)
            score_gain = float(
                improved_overall["Reference score"]
                - traditional_overall["Reference score"]
            )
            experiment_winner = (
                "Proposed Improved RHLT" if score_gain >= 0.0 else "Traditional RHLT"
            )
            st.success(
                f"Best overall: **{experiment_winner}** — won {improved_wins}/{total_experiments} "
                f"samples; clean-reference score difference {score_gain:+.4f}."
            )
            evidence_cards = st.columns(4)
            evidence_cards[0].metric(
                "Win rate", f"{improved_wins / max(total_experiments, 1) * 100.0:.0f}%"
            )
            evidence_cards[1].metric(
                "Reference score",
                f"{improved_overall['Reference score']:.4f}",
                delta=f"{score_gain:+.4f} vs Traditional",
            )
            evidence_cards[2].metric(
                "Foreground SSIM",
                f"{improved_overall['SSIM reference']:.4f}",
                delta=f"{improved_overall['SSIM reference'] - traditional_overall['SSIM reference']:+.4f}",
            )
            evidence_cards[3].metric(
                "Foreground MSE",
                f"{improved_overall['MSE reference']:.1f}",
                delta=f"{improved_overall['MSE reference'] - traditional_overall['MSE reference']:+.1f}",
                delta_color="inverse",
            )

            level_means = research_frame.groupby(
                ["Level", "Candidate"], sort=False
            ).mean(numeric_only=True)
            level_summary = []
            for level in DEGRADATION_PRESETS:
                traditional_level = level_means.loc[(level, "Traditional RHLT")]
                improved_level = level_means.loc[(level, "Proposed Improved RHLT")]
                level_delta = float(
                    improved_level["Reference score"]
                    - traditional_level["Reference score"]
                )
                level_summary.append(
                    {
                        "Degradation": level,
                        "Best": "Improved" if level_delta >= 0.0 else "Traditional",
                        "Traditional score": f"{traditional_level['Reference score']:.4f}",
                        "Improved score": f"{improved_level['Reference score']:.4f}",
                        "Score advantage": f"{level_delta:+.4f}",
                    }
                )
            st.dataframe(pd.DataFrame(level_summary), width="stretch", hide_index=True)

            research_details = st.expander(
                "View complete experiment statistics", expanded=False
            )
            research_details.dataframe(aggregate, width="stretch", hide_index=True)
            st.download_button(
                "Download controlled-experiment CSV",
                research_frame.to_csv(index=False).encode("utf-8"),
                "rhlt_controlled_degradation_results.csv",
                "text/csv",
            )

            level_records = [
                item for item in experiment_records if item["level"] == displayed_level
            ]
            st.subheader(f"Inspect one sample · {displayed_level}")
            selection_frame = pd.DataFrame(selection_rows)
            research_details.dataframe(
                selection_frame[selection_frame["Level"] == displayed_level],
                width="stretch",
                hide_index=True,
            )
            displayed_name = st.selectbox(
                "Displayed experiment image",
                [item["filename"] for item in level_records],
                key="controlled_experiment_image",
            )
            shown = next(
                item for item in level_records if item["filename"] == displayed_name
            )
            shown_result = shown["result"]
            sample_score_delta = (
                shown["improved_reference_score"] - shown["traditional_reference_score"]
            )
            sample_winner = (
                "Proposed Improved RHLT"
                if sample_score_delta >= 0.0
                else "Traditional RHLT"
            )
            st.info(
                f"This sample's best restoration: **{sample_winner}** · "
                f"reference-score difference {sample_score_delta:+.4f}."
            )
            shown_images = [
                ("Clean Ground Truth", shown["clean"]),
                ("Degraded Input", shown["degraded"]),
                ("Traditional RHLT", shown_result["traditional_rhlt_baseline"]),
                ("Proposed Improved RHLT", shown_result["improved_rhlt"]),
            ]
            for column, (caption, image) in zip(st.columns(4), shown_images):
                column.image(
                    image, caption=caption, clamp=True, use_container_width=True
                )

            visual_details = st.expander(
                "View crops, difference maps and diagnostic maps", expanded=False
            )
            visual_details.subheader("Magnified fingerprint-region crop")
            shown_mask = shown_result["foreground_mask"]
            for column, (caption, image) in zip(
                visual_details.columns(4), shown_images
            ):
                column.image(
                    fingerprint_region(image, shown_mask),
                    caption=caption,
                    clamp=True,
                    use_container_width=True,
                )

            difference_columns = visual_details.columns(2)
            difference_columns[0].image(
                amplified_difference(
                    shown_result["traditional_rhlt_baseline"],
                    shown["degraded"],
                    difference_gain,
                ),
                caption=f"|Traditional − degraded| × {difference_gain:g}",
                clamp=True,
                use_container_width=True,
            )
            difference_columns[1].image(
                amplified_difference(
                    shown_result["improved_rhlt"], shown["degraded"], difference_gain
                ),
                caption=f"|Improved − degraded| × {difference_gain:g}",
                clamp=True,
                use_container_width=True,
            )

            visual_details.subheader("Quality-adaptive diagnostic maps")
            diagnostic_images = [
                ("RHLT response", shown_result["rhlt_stretched"]),
                ("Orientation field", shown_result["orientation_visualisation"]),
                (
                    "Local frequency",
                    diagnostic_map_to_uint8(
                        shown_result["local_frequency_map"], shown_mask
                    ),
                ),
                (
                    "Weak-ridge map",
                    diagnostic_map_to_uint8(shown_result["weak_ridge_map"], shown_mask),
                ),
                (
                    "Fusion-weight map",
                    diagnostic_map_to_uint8(
                        shown_result["fusion_weight_map"], shown_mask
                    ),
                ),
                ("Gabor support", shown_result["gabor_support"]),
            ]
            for column, (caption, image) in zip(
                visual_details.columns(6), diagnostic_images
            ):
                column.image(
                    image, caption=caption, clamp=True, use_container_width=True
                )
        else:
            st.error(
                "No original BMP sample was available for the controlled experiment."
            )

    st.divider()
    st.subheader("RHLT component ablation")
    st.caption(
        "This ablation compares Traditional RHLT, apodised RHLT, orientation-only "
        "Gabor support, and the proposed quality/frequency-adaptive RHLT. It does "
        "**not** change the main result."
    )
    if selected_algorithm == "RHLT":
        run_study = st.checkbox(
            "Run RHLT ablation study for the selected image",
            value=False,
            help="Reruns the selected image with four RHLT configurations. Research diagnostic only.",
        )
        if run_study:
            st.info(
                "Comparing: Traditional baseline · RHLT with apodisation · "
                "orientation-only support · quality/frequency-adaptive RHLT."
            )
            ablation = pd.DataFrame(
                run_ablation(selected["source_original"], rhlt_config)
            )
            ablation_winner_index = (
                ablation["candidate_quality_score"].astype(float).idxmax()
            )
            ablation_winner = str(ablation.loc[ablation_winner_index, "variant"])
            st.success(f"Best component combination: **{ablation_winner}**")
            ablation_summary = pd.DataFrame(
                {
                    "Variant": ablation["variant"],
                    "Recommended": [
                        "✓" if index == ablation_winner_index else ""
                        for index in ablation.index
                    ],
                    "Quality score": ablation["candidate_quality_score"].map(
                        lambda value: f"{float(value):.4f}"
                    ),
                    "SSIM": ablation["ssim"].map(lambda value: f"{float(value):.3f}"),
                    "RVC change": (
                        (
                            ablation["processed_ridge_valley_clarity"]
                            / ablation["original_ridge_valley_clarity"]
                            - 1.0
                        )
                        * 100.0
                    ).map(lambda value: f"{float(value):+.1f}%"),
                }
            )
            st.dataframe(ablation_summary, width="stretch", hide_index=True)
            ablation_details = st.expander(
                "View complete ablation metrics", expanded=False
            )
            ablation_details.dataframe(ablation, width="stretch", hide_index=True)
            st.download_button(
                "⬇️ Download ablation CSV",
                ablation.to_csv(index=False).encode("utf-8"),
                f"{selected_name}_rhlt_ablation.csv",
                "text/csv",
            )
        else:
            st.info(
                "Enable the study above only when you need the parameter comparison."
            )
    else:
        st.info("The ablation study experiment is currently only configured for RHLT.")

# ── Tab 6: Algorithm Comparison ───────────────────────────────────────────────
with tabs[6]:
    st.caption(
        "Run **all available algorithms** on the selected fingerprint image and compare "
        "their enhancement quality side by side. This satisfies the Technical Requirement "
        "of benchmarking multiple techniques."
    )

    run_comparison = st.checkbox(
        "Run multi-algorithm comparison for the selected image",
        value=False,
        help="This runs every available algorithm on the same image. May take a few seconds.",
    )

    if run_comparison:
        all_algo_names = [a["name"] for a in ALGORITHM_STATUS if a["available"]]
        # Get the source image from the currently selected record
        source_image = selected["source_original"]
        calibration = calibrate_image(source_image, calibration_config)
        calibrated = calibration["image"]

        comparison_results = {}
        with st.spinner(f"Running {len(all_algo_names)} algorithms..."):
            for algo_name in all_algo_names:
                try:
                    result = _run_selected_algorithm(
                        calibrated,
                        algo_name,
                        final_only=True,
                    )
                    result["source_original"] = source_image
                    comparison_results[algo_name] = result
                except Exception as exc:
                    st.error(f"{algo_name} failed: {exc}")

        if comparison_results:
            quality_scores = {
                name: comparison_quality_score(result["metrics"])
                for name, result in comparison_results.items()
            }
            winner_name = max(quality_scores, key=quality_scores.get)
            winner_result = comparison_results[winner_name]
            winner_metrics = winner_result["metrics"]
            winner_rvc_change = (
                float(winner_metrics.get("processed_ridge_valley_clarity", 0.0))
                / max(
                    float(winner_metrics.get("original_ridge_valley_clarity", 0.0)),
                    1e-6,
                )
                - 1.0
            ) * 100.0
            st.success(
                f"Recommended algorithm: **{winner_name}** · balanced quality score "
                f"**{quality_scores[winner_name]:.1f}/100**."
            )
            winner_cards = st.columns(4)
            winner_cards[0].metric(
                "Quality score", f"{quality_scores[winner_name]:.1f}/100"
            )
            winner_cards[1].metric("SSIM", f"{winner_metrics.get('ssim', 0.0):.3f}")
            winner_cards[2].metric("RVC change", f"{winner_rvc_change:+.1f}%")
            winner_cards[3].metric(
                "Processing time",
                f"{winner_result.get('processing_time_ms', 0.0):.1f} ms",
            )

            # --- Visual comparison: Original + all enhanced ---
            st.subheader("Visual comparison")
            vis_cols = st.columns(len(comparison_results) + 1)
            vis_cols[0].image(
                source_image, caption="Original", use_container_width=True
            )
            for idx, (algo_name, result) in enumerate(comparison_results.items(), 1):
                vis_cols[idx].image(
                    result["enhanced_image"],
                    caption=f"{algo_name}{' · Recommended' if algo_name == winner_name else ''}",
                    clamp=True,
                    use_container_width=True,
                )

            st.divider()

            # --- Concise leaderboard ---
            st.subheader("Algorithm ranking")
            leaderboard_rows = []
            for algo_name, result in comparison_results.items():
                m = result["metrics"]
                original_rvc = max(
                    float(m.get("original_ridge_valley_clarity", 0.0)), 1e-6
                )
                rvc_change = (
                    float(m.get("processed_ridge_valley_clarity", 0.0)) / original_rvc
                    - 1.0
                ) * 100.0
                leaderboard_rows.append(
                    {
                        "Algorithm": algo_name,
                        "Recommended": "✓" if algo_name == winner_name else "",
                        "Quality score": quality_scores[algo_name],
                        "SSIM": f"{m.get('ssim', 0.0):.3f}",
                        "RVC change": f"{rvc_change:+.1f}%",
                        "Time (ms)": f"{result.get('processing_time_ms', 0.0):.1f}",
                    }
                )
            lb_df = pd.DataFrame(leaderboard_rows).sort_values(
                "Quality score", ascending=False
            )
            lb_df["Quality score"] = lb_df["Quality score"].map(
                lambda value: f"{value:.1f}/100"
            )
            st.dataframe(lb_df, width="stretch", hide_index=True)

            comparison_details = st.expander(
                "View complete algorithm metrics", expanded=False
            )
            complete_rows = []
            for algo_name, result in comparison_results.items():
                complete_rows.append({"Algorithm": algo_name, **result["metrics"]})
            comparison_details.dataframe(
                pd.DataFrame(complete_rows), width="stretch", hide_index=True
            )

            st.divider()

            # --- Download comparison PDF ---
            from reporting import build_comparison_report

            comparison_pdf = build_comparison_report(
                selected_name,
                source_image,
                comparison_results,
            )
            st.download_button(
                "⬇️ Download Comparison PDF Report",
                comparison_pdf,
                f"{selected_name.rsplit('.', 1)[0]}_comparison_report.pdf",
                "application/pdf",
            )
    else:
        st.info(
            "Enable the comparison above to benchmark all algorithms on the selected image."
        )

st.divider()
download_columns = st.columns(2)
download_columns[0].download_button(
    "⬇️ Download enhanced PNG",
    encode_png(selected["ridge_restored"]),
    f"{selected_name.rsplit('.', 1)[0]}_ridge_restored.png",
    "image/png",
)
download_columns[1].download_button(
    "⬇️ Download PDF Report",
    build_pdf_report(selected_name, selected, selected_algorithm),
    f"{selected_name.rsplit('.', 1)[0]}_report.pdf",
    "application/pdf",
)
