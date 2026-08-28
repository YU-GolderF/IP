from __future__ import annotations

import importlib
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
import streamlit as st

from algorithms import ALGORITHM_STATUS
from algorithms.rhlt import pipeline as rhlt_pipeline_module
from algorithms.rhlt import ridge_filter as rhlt_ridge_filter_module
from algorithms.rhlt.config import RHLTConfig
from core import metrics as core_metrics_module

# Streamlit can rerun app.py while retaining imported project modules. Reload the
# implementation modules in dependency order so code changes cannot leave the UI
# on a new build while run_rhlt still points at an older in-memory function.
core_metrics_module = importlib.reload(core_metrics_module)
importlib.reload(rhlt_ridge_filter_module)
rhlt_pipeline_module = importlib.reload(rhlt_pipeline_module)
run_rhlt = rhlt_pipeline_module.run_rhlt
run_ablation = rhlt_pipeline_module.run_ablation
calculate_image_metrics = core_metrics_module.calculate_image_metrics
from core import (
    CalibrationConfig,
    PreprocessingConfig,
    calibrate_image,
    load_images_from_folder,
    load_multiple_images,
    process_batch,
)
from reporting import build_pdf_report, encode_png


st.set_page_config(page_title="Fingerprint Enhancement System", page_icon="🔬", layout="wide")
st.title("Fingerprint Enhancement System")
APP_BUILD = "sigmoid-push-directional-2026-08-28-v6"
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
def load_folder_batch(path: str):
    return load_images_from_folder(path)


uploaded_payloads = tuple((uploaded.name, uploaded.getvalue()) for uploaded in uploads)
loaded_images, input_errors = load_uploaded_batch(uploaded_payloads)
loaded_images = list(loaded_images)
input_errors = list(input_errors)

default_image_folder = (Path(__file__).resolve().parents[1] / "image").resolve()
if default_image_folder.exists() and default_image_folder.is_dir():
    try:
        folder_images, folder_errors = load_folder_batch(str(default_image_folder))
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


def process_loaded_image(item):
    calibration = calibrate_image(item.image, calibration_config)
    calibrated = calibration["image"]
    result = run_rhlt(
        calibrated,
        rhlt_config,
        preprocessing_config=preprocessing_config,
    )
    preprocessing_stages = result["preprocessing_stages"]

    # Never publish a candidate that severely damages measurable ridge detail.
    # This second guard also protects a rerun if Streamlit retained an older
    # imported algorithm module in its long-running process.
    result_metrics = result["metrics"]
    sharpness_retained = result_metrics["processed_sharpness"] >= (
        result_metrics["original_sharpness"] * 0.90
    )
    edge_clarity_retained = result_metrics["processed_edge_clarity"] >= (
        result_metrics["original_edge_clarity"] * 0.90
    )
    if not (sharpness_retained and edge_clarity_retained):
        preserved = preprocessing_stages["grayscale"]
        result["ridge_restored"] = preserved
        result["ridge_enhanced"] = preserved
        result["enhanced_image"] = preserved
        result["metrics"].update(calculate_image_metrics(calibrated, preserved))
        result["warnings"].append(
            "A degraded enhancement result was blocked by the dashboard quality guard."
        )

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
            "foreground_coverage_percent": result_metrics.get("foreground_coverage_percent", 0.0),
            "valid_orientation_blocks": result_metrics.get("valid_orientation_blocks", 0),
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
    original_column, enhanced_column = st.columns(2)
    original_column.image(
        selected["source_original"], caption="Original fingerprint", use_container_width=True
    )
    enhanced_column.image(
        selected["enhanced_image"],
        caption=f"Enhanced · {selected_algorithm}",
        clamp=True,
        use_container_width=True,
    )

    st.divider()

    # --- Metrics comparison table ---
    st.subheader("Quality metrics: original vs enhanced")
    comparison_metrics = [
        ("Contrast", "original_contrast", "processed_contrast", 4),
        ("Ridge-Valley Clarity (RVC)", "original_ridge_valley_clarity", "processed_ridge_valley_clarity", 1),
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
        pct_change = (enhanced_value - original_value) / max(abs(original_value), 1e-9) * 100.0
        status = "✅" if enhanced_value >= original_value else "⚠️"
        comparison_rows.append(
            {
                "Metric": label,
                "Original": f"{original_value:.{decimals}f}",
                "Enhanced": f"{enhanced_value:.{decimals}f}",
                "Change": f"{enhanced_value - original_value:+.{decimals}f}",
                "Change %": f"{pct_change:+.1f}%",
                "Status": status,
            }
        )
    # Add SSIM row separately
    ssim_status = "✅" if ssim_val >= 0.80 else "⚠️"
    comparison_rows.append(
        {
            "Metric": "SSIM (Structural Similarity)",
            "Original": "1.000 (reference)",
            "Enhanced": f"{ssim_val:.3f}",
            "Change": f"{ssim_val - 1.0:+.3f}",
            "Change %": f"{(ssim_val - 1.0) * 100:+.1f}%",
            "Status": ssim_status,
        }
    )
    st.dataframe(pd.DataFrame(comparison_rows), use_container_width=True, hide_index=True)

    st.divider()

    # --- Fingerprint structure stats ---
    st.subheader("Fingerprint structure analysis")
    struct_cols = st.columns(4)
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
        ("Fingerprint foreground mask", (selected["foreground_mask"].astype("uint8") * 255)),
        ("Ridge flow restored", selected["ridge_restored"]),
        (f"Skeleton (minutiae: {int(metrics.get('minutiae_total', 0))})", (selected["skeleton"].astype("uint8") * 255)),
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
        right.markdown(
            """
            **How it works:**
            Sobel gradients estimate the direction *across* each ridge.
            The system rotates that direction by 90° to obtain the ridge *flow*,
            smooths it using doubled angles (to handle the 180° ambiguity),
            and selects the nearest cached Gabor orientation for each valid block.
            """
        )
    else:
        st.info(f"Ridge orientation data is not provided by {selected_algorithm}.")

# ── Tab 3: RHLT Algorithm Internals ──────────────────────────────────────────
with tabs[3]:
    st.caption(
        "This page shows the visualisation of the spiral-phase RHLT transfer function and "
        "its spatial PSF. It illustrates the mathematical foundation of the RHLT algorithm "
        "for technical review — it does not directly show the enhancement quality."
    )
    if selected_algorithm == "RHLT" and "rhlt_stretched" in selected:
        left, middle, right = st.columns(3)
        left.image(
            selected["rhlt_stretched"],
            caption="Spiral-phase RHLT edge response (isotropic edge magnitude)",
            clamp=True,
            use_container_width=True,
        )
        middle.image(
            selected["psf_visualisation"],
            caption="RHLT PSF magnitude (donut = topological charge l=1)",
            clamp=True,
            use_container_width=True,
        )
        right.markdown(
            f"""
            **Legacy diagnostic settings**

            | Parameter | Value |
            |---|---|
            | PSF size | **{rhlt_config.psf_size} × {rhlt_config.psf_size}** |
            | Aperture ratio | **{rhlt_config.aperture_ratio:.2f}** |
            | Apodisation | **{rhlt_config.apodisation:.2f}** |
            | Topological charge | **{rhlt_config.topological_charge}** |

            Apodisation 0.0 is the baseline spiral-phase RHLT (no cosine taper).
            The RHLT edge response is used as a diagnostic; ridge restoration is
            performed by the direction-adaptive Gabor stage.
            """
        )
    else:
        st.info("RHLT Algorithm Internals is specific to the RHLT algorithm.")

# ── Tab 4: Data Dashboard ─────────────────────────────────────────────────────
with tabs[4]:
    st.subheader("Batch enhancement summary")
    st.caption("One row per image processed in this session. All key quality indicators are included.")
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
    st.dataframe(enriched_summary, use_container_width=True, hide_index=True)
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
    st.caption(
        "This is a research-only diagnostic page. It reruns the selected image with "
        "four different RHLT parameter configurations to study the effect of apodisation "
        "and PSF size on enhancement quality. It does **not** change the main result."
    )
    if selected_algorithm == "RHLT":
        run_study = st.checkbox(
            "Run four-setting RHLT ablation for the selected image",
            value=False,
            help="This is a research diagnostic and does not change the enhanced output.",
        )
        if run_study:
            st.warning(
                "This ablation reruns the selected image four times with different RHLT settings. "
                "Use the same dataset and metrics for every team algorithm."
            )
            ablation = pd.DataFrame(run_ablation(selected["source_original"], rhlt_config))
            st.dataframe(ablation, use_container_width=True, hide_index=True)
            st.download_button(
                "⬇️ Download ablation CSV",
                ablation.to_csv(index=False).encode("utf-8"),
                f"{selected_name}_rhlt_ablation.csv",
                "text/csv",
            )
        else:
            st.info("Enable the study above only when you need the parameter comparison.")
    else:
        st.info("The ablation study experiment is currently only configured for RHLT.")

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
