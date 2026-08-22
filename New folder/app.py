from __future__ import annotations

from io import BytesIO
from zipfile import ZIP_DEFLATED, ZipFile

import pandas as pd
import streamlit as st

from algorithms import ALGORITHM_STATUS
from algorithms.rhlt import RHLTConfig, run_ablation, run_rhlt
from core import (
    CalibrationConfig,
    PreprocessingConfig,
    calibrate_image,
    load_images_from_folder,
    load_multiple_images,
    preprocess_with_stages,
    process_batch,
)
from reporting import build_pdf_report, encode_png


st.set_page_config(page_title="Fingerprint Enhancement System", page_icon="🔬", layout="wide")
st.title("Fingerprint Enhancement System")
st.caption(
    "Shared preprocessing, calibration, batch ingestion and quality metrics with "
    "pluggable team algorithms. RHLT Ridge Flow Restoration is currently available."
)

available_algorithms = [item["name"] for item in ALGORITHM_STATUS if item["available"]]

with st.sidebar:
    st.header("Algorithm")
    selected_algorithm = st.selectbox("Select algorithm", available_algorithms)
    for algorithm in ALGORITHM_STATUS:
        status = "Available" if algorithm["available"] else "Reserved"
        st.caption(f"{algorithm['name']} · {algorithm['owner']} · {status}")

    st.header("1. Fingerprint input")
    uploads = st.file_uploader(
        "Upload one or multiple images",
        type=["png", "jpg", "jpeg", "bmp", "tif", "tiff"],
        accept_multiple_files=True,
        help="A corrupted image is reported and skipped without stopping the batch.",
    )
    folder_path = st.text_input(
        "Optional local Windows folder",
        help="Used only when Streamlit runs locally. All supported images in this folder are loaded.",
    )

    with st.expander("2. Image calibration", expanded=False):
        max_dimension = st.select_slider(
            "Maximum width/height",
            options=[512, 768, 1024, 1280, 1600, 2048],
            value=1024,
        )
        rotation_degrees = st.slider(
            "Manual rotation correction (degrees)", -15.0, 15.0, 0.0, 0.5
        )

    with st.expander("3. Shared preprocessing", expanded=True):
        gaussian_sigma = st.slider("Gaussian sigma", 0.0, 2.5, 1.0, 0.1)
        use_median_filter = st.checkbox("Use optional median filter", value=False)
        median_kernel_size = st.select_slider("Median kernel", options=[3, 5], value=3)
        clahe_clip_limit = st.slider("CLAHE clip limit", 1.0, 4.0, 2.0, 0.1)

    with st.expander("4. RHLT ridge-flow settings", expanded=True):
        block_size = st.select_slider(
            "Orientation block size", options=[8, 12, 16, 24, 32], value=16
        )
        orientation_smoothing_sigma = st.slider(
            "Orientation smoothing", 0.0, 3.0, 1.0, 0.1
        )
        orientation_bins = st.select_slider(
            "Gabor orientation bins", options=[8, 12, 16, 18], value=12
        )
        gabor_lambda = st.slider("Gabor wavelength", 6.0, 16.0, 10.0, 0.5)
        gabor_strength = st.slider(
            "Ridge restoration strength", 0.0, 2.0, 0.75, 0.05
        )

    with st.expander("5. Legacy RHLT diagnostic", expanded=False):
        psf_size = st.select_slider(
            "PSF size", options=[33, 49, 65, 81, 97, 129], value=65
        )
        aperture_ratio = st.slider(
            "Circular aperture ratio", 0.50, 1.00, 0.90, 0.01
        )
        apodisation = st.slider("PSF apodisation", 0.0, 0.8, 0.0, 0.05)
        run_study = st.checkbox("Enable four-setting RHLT ablation", value=False)

preprocessing_config = PreprocessingConfig(
    gaussian_kernel_size=5,
    gaussian_sigma=gaussian_sigma,
    use_median_filter=use_median_filter,
    median_kernel_size=median_kernel_size,
    clahe_clip_limit=clahe_clip_limit,
)
calibration_config = CalibrationConfig(
    max_width=max_dimension,
    max_height=max_dimension,
    rotation_degrees=rotation_degrees,
)
rhlt_config = RHLTConfig(
    gaussian_sigma=gaussian_sigma,
    psf_size=int(psf_size),
    aperture_ratio=aperture_ratio,
    apodisation=apodisation,
    block_size=int(block_size),
    orientation_smoothing_sigma=orientation_smoothing_sigma,
    orientation_bins=int(orientation_bins),
    gabor_lambda=gabor_lambda,
    gabor_strength=gabor_strength,
)


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

if folder_path.strip():
    try:
        folder_images, folder_errors = load_folder_batch(folder_path.strip())
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
    preprocessing_stages = preprocess_with_stages(calibrated, preprocessing_config)
    result = run_rhlt(
        calibrated,
        rhlt_config,
        preprocessed_image=preprocessing_stages["enhanced"],
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

for warning in selected["warnings"]:
    st.warning(warning)

metrics = selected["metrics"]
metric_columns = st.columns(6)
metric_columns[0].metric("Original contrast", f"{metrics['original_contrast']:.4f}")
metric_columns[1].metric("Enhanced contrast", f"{metrics['processed_contrast']:.4f}")
metric_columns[2].metric("Original sharpness", f"{metrics['original_sharpness']:.1f}")
metric_columns[3].metric("Enhanced sharpness", f"{metrics['processed_sharpness']:.1f}")
metric_columns[4].metric("Foreground", f"{metrics['foreground_coverage_percent']:.1f}%")
metric_columns[5].metric("Processing time", f"{selected['processing_time_ms']:.1f} ms")

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
            "foreground_coverage_percent": result_metrics["foreground_coverage_percent"],
            "valid_orientation_blocks": result_metrics["valid_orientation_blocks"],
        }
    )
summary = pd.DataFrame(summary_rows)

tabs = st.tabs(
    [
        "Overview",
        "Shared preprocessing",
        "Ridge orientation",
        "Legacy RHLT diagnostic",
        "Data dashboard",
        "Experiment",
    ]
)

with tabs[0]:
    columns = st.columns(4)
    columns[0].image(
        selected["source_original"], caption="Original fingerprint", width="stretch"
    )
    columns[1].image(
        selected["preprocessed"], caption="Shared preprocessed", clamp=True, width="stretch"
    )
    columns[2].image(
        selected["ridge_restored"], caption="Ridge flow restored", clamp=True, width="stretch"
    )
    columns[3].image(
        selected["ridge_binary"], caption="Optional ridge isolation", clamp=True, width="stretch"
    )
    st.image(
        selected["minutiae_overlay"],
        caption="Diagnostic minutiae overlay (not matching accuracy)",
        width="stretch",
    )

with tabs[1]:
    stages = selected["preprocessing_stages"]
    stage_images = [
        ("Grayscale", stages["grayscale"]),
        ("Gaussian denoised", stages["gaussian_denoised"]),
        ("Median/denoised", stages["denoised"]),
        ("Intensity normalised", stages["normalised"]),
        ("CLAHE enhanced", stages["enhanced"]),
        ("Fingerprint foreground", selected["foreground_mask"]),
        ("Ridge flow restored", selected["ridge_restored"]),
        ("Skeleton", selected["skeleton"]),
    ]
    for start in range(0, len(stage_images), 4):
        columns = st.columns(4)
        for column, (label, image) in zip(columns, stage_images[start : start + 4]):
            column.image(image, caption=label, clamp=True, width="stretch")

with tabs[2]:
    left, right = st.columns([2, 1])
    left.image(
        selected["orientation_visualisation"],
        caption="Smoothed local ridge directions",
        width="stretch",
    )
    right.metric("Valid orientation blocks", int(metrics["valid_orientation_blocks"]))
    right.metric("Foreground coverage", f"{metrics['foreground_coverage_percent']:.1f}%")
    right.markdown(
        """
        Sobel gradients estimate the direction across each ridge. The system rotates
        that direction by 90° to obtain ridge flow, smooths it with doubled angles,
        and selects the nearest cached Gabor orientation for each valid block.
        """
    )

with tabs[3]:
    left, middle, right = st.columns(3)
    left.image(
        selected["rhlt_stretched"],
        caption="Spiral-phase RHLT edge response",
        clamp=True,
        width="stretch",
    )
    middle.image(
        selected["psf_visualisation"],
        caption="RHLT PSF magnitude",
        clamp=True,
        width="stretch",
    )
    right.markdown(
        f"""
        **Legacy diagnostic settings**

        - PSF size: **{rhlt_config.psf_size} × {rhlt_config.psf_size}**
        - Aperture ratio: **{rhlt_config.aperture_ratio:.2f}**
        - Apodisation: **{rhlt_config.apodisation:.2f}**
        - Topological charge: **{rhlt_config.topological_charge}**

        Apodisation 0.0 remains the baseline spiral-phase RHLT setting.
        """
    )

with tabs[4]:
    st.subheader("Batch enhancement summary")
    st.dataframe(summary, width="stretch", hide_index=True)
    numeric_summary = summary.select_dtypes(include="number")
    st.subheader("Batch mean")
    st.dataframe(
        numeric_summary.mean().to_frame("mean").T, width="stretch", hide_index=True
    )
    csv_bytes = summary.to_csv(index=False).encode("utf-8")
    st.download_button(
        "Download metrics CSV", csv_bytes, "fingerprint_metrics.csv", "text/csv"
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
        "Download all outputs (ZIP)",
        zip_buffer.getvalue(),
        "fingerprint_outputs.zip",
        "application/zip",
    )

with tabs[5]:
    if run_study:
        st.warning(
            "This legacy ablation reruns the selected image four times. "
            "Use the same dataset and metrics for every team algorithm."
        )
        ablation = pd.DataFrame(run_ablation(selected["source_original"], rhlt_config))
        st.dataframe(ablation, width="stretch", hide_index=True)
        st.download_button(
            "Download ablation CSV",
            ablation.to_csv(index=False).encode("utf-8"),
            f"{selected_name}_rhlt_ablation.csv",
            "text/csv",
        )
    else:
        st.info("Enable the legacy RHLT ablation study in the sidebar when needed.")

st.divider()
download_columns = st.columns(2)
download_columns[0].download_button(
    "Download selected enhanced PNG",
    encode_png(selected["ridge_restored"]),
    f"{selected_name.rsplit('.', 1)[0]}_ridge_restored.png",
    "image/png",
)
download_columns[1].download_button(
    "Download selected PDF report",
    build_pdf_report(selected_name, selected, selected_algorithm),
    f"{selected_name.rsplit('.', 1)[0]}_report.pdf",
    "application/pdf",
)
