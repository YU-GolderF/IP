"""Reusable, lightweight PDF reporting for single fingerprint results."""

from __future__ import annotations

from datetime import datetime
from io import BytesIO
from pathlib import Path
from tempfile import TemporaryDirectory
from xml.sax.saxutils import escape

import cv2
import numpy as np
from reportlab.lib import colors
from reportlab.lib.pagesizes import A4
from reportlab.lib.styles import getSampleStyleSheet
from reportlab.lib.units import cm
from reportlab.platypus import Image, Paragraph, SimpleDocTemplate, Spacer, Table, TableStyle

from core.preprocessing import ensure_uint8


def encode_png(image: np.ndarray) -> bytes:
    """Encode a grayscale, boolean, RGB, or RGBA array as PNG bytes."""
    array = np.asarray(image)
    if array.dtype == bool:
        array = array.astype(np.uint8) * 255
    else:
        array = ensure_uint8(array)
    if array.ndim == 3 and array.shape[2] == 4:
        array = cv2.cvtColor(array, cv2.COLOR_RGBA2BGRA)
    elif array.ndim == 3:
        array = cv2.cvtColor(array, cv2.COLOR_RGB2BGR)
    encoded, buffer = cv2.imencode(".png", array)
    if not encoded:
        raise RuntimeError("failed to encode report image")
    return bytes(buffer)


def _report_image(path: Path, array: np.ndarray, max_width: float, max_height: float) -> Image:
    height, width = np.asarray(array).shape[:2]
    scale = min(max_width / max(width, 1), max_height / max(height, 1))
    return Image(str(path), width=width * scale, height=height * scale)


def _fmt(value: object, decimals: int = 3) -> str:
    if value is None:
        return "N/A"
    if isinstance(value, bool):
        return "Yes" if value else "No"
    if isinstance(value, (int, np.integer)):
        return str(int(value))
    if isinstance(value, (float, np.floating)):
        v = float(value)
        if abs(v) >= 1000:
            return f"{v:,.0f}"
        return f"{v:.{decimals}f}"
    return str(value)


def build_pdf_report(
    filename: str,
    result: dict,
    algorithm_name: str | None = None,
) -> bytes:
    """Generate a professional, comparison-focused PDF report."""
    original = result.get("original", result.get("grayscale"))
    enhanced = result.get("enhanced_image", result.get("ridge_restored", result.get("ridge_enhanced")))
    if original is None or enhanced is None:
        raise ValueError("report result must contain original and enhanced images")

    algo = algorithm_name or str(result.get("algorithm_name", "Fingerprint enhancement"))
    metrics = result.get("metrics", {})
    output = BytesIO()
    styles = getSampleStyleSheet()

    document = SimpleDocTemplate(
        output,
        pagesize=A4,
        rightMargin=1.4 * cm,
        leftMargin=1.4 * cm,
        topMargin=1.4 * cm,
        bottomMargin=1.4 * cm,
    )
    story = [
        Paragraph("Fingerprint Enhancement Report", styles["Title"]),
        Paragraph(f"Algorithm: <b>{escape(algo)}</b>", styles["Normal"]),
        Paragraph(f"Input file: {escape(filename)}", styles["Normal"]),
        Paragraph(f"Generated: {datetime.now().astimezone().strftime('%Y-%m-%d %H:%M:%S %Z')}", styles["Normal"]),
        Spacer(1, 0.4 * cm),
    ]

    # Section 1: KPI summary
    cii = float(metrics.get("cii", 1.0))
    ssim = float(metrics.get("ssim", 0.0))
    sharpness_pct = float(metrics.get("sharpness_improvement_pct", 0.0))
    coherence = float(metrics.get("mean_orientation_coherence", 0.0))
    minutiae = int(metrics.get("minutiae_total", 0))
    rvc_orig = float(metrics.get("original_ridge_valley_clarity", 0.0))
    rvc_proc = float(metrics.get("processed_ridge_valley_clarity", 0.0))
    rvc_pct = (rvc_proc - rvc_orig) / max(rvc_orig, 1e-6) * 100.0
    good_green = colors.Color(0.88, 0.97, 0.88)
    warn_yellow = colors.Color(1.0, 0.97, 0.80)

    kpi_data = [
        ["Metric", "Value", "Interpretation"],
        ["Contrast Improvement (CII)", f"{cii:.2f}x", ">= 1.0 = improved" if cii >= 1.0 else "< 1.0 = degraded"],
        ["Sharpness Improvement", f"{sharpness_pct:+.1f}%", "Positive = sharper ridges"],
        ["Ridge-Valley Clarity Change", f"{rvc_pct:+.1f}%", "Positive = cleaner ridge/valley separation"],
        ["SSIM (Structure Preserved)", f"{ssim:.3f}", ">= 0.80 = well preserved" if ssim >= 0.80 else "< 0.80 = structural change"],
        ["Mean Orientation Coherence", f"{coherence:.3f}", "Closer to 1.0 = more parallel ridge flow"],
        ["Minutiae Detected", str(minutiae), "Ridge endings + bifurcations on skeleton"],
    ]
    kpi_colors = [
        (good_green if cii >= 1.0 else warn_yellow),
        (good_green if sharpness_pct >= 0 else warn_yellow),
        (good_green if rvc_pct >= 0 else warn_yellow),
        (good_green if ssim >= 0.80 else warn_yellow),
        good_green,
        good_green,
    ]
    kpi_table = Table(kpi_data, colWidths=[6.5 * cm, 3.2 * cm, 7.8 * cm], repeatRows=1)
    row_style = [
        ("BACKGROUND", (0, 0), (-1, 0), colors.HexColor("#2B4590")),
        ("TEXTCOLOR", (0, 0), (-1, 0), colors.white),
        ("FONTNAME", (0, 0), (-1, 0), "Helvetica-Bold"),
        ("FONTNAME", (0, 1), (0, -1), "Helvetica-Bold"),
        ("GRID", (0, 0), (-1, -1), 0.5, colors.grey),
        ("ALIGN", (1, 1), (1, -1), "CENTER"),
        ("VALIGN", (0, 0), (-1, -1), "MIDDLE"),
        ("TOPPADDING", (0, 0), (-1, -1), 5),
        ("BOTTOMPADDING", (0, 0), (-1, -1), 5),
    ]
    for idx, c in enumerate(kpi_colors):
        row_style.append(("BACKGROUND", (0, idx + 1), (-1, idx + 1), c))
    kpi_table.setStyle(TableStyle(row_style))
    story.append(Paragraph("Enhancement Quality Summary", styles["Heading2"]))
    story.append(kpi_table)
    story.append(Spacer(1, 0.5 * cm))

    # Section 2: Image comparison
    with TemporaryDirectory() as tmp_dir:
        tmp = Path(tmp_dir)

        def save_img(key: str, arr: np.ndarray) -> Path:
            p = tmp / f"{key}.png"
            p.write_bytes(encode_png(arr))
            return p

        orig_arr = np.asarray(original)
        enh_arr = np.asarray(enhanced)
        orig_path = save_img("original", orig_arr)
        enh_path = save_img("enhanced", enh_arr)

        img_cells = [
            (_report_image(orig_path, orig_arr, 8.0 * cm, 5.5 * cm),
             Paragraph("Original fingerprint", styles["BodyText"])),
            (_report_image(enh_path, enh_arr, 8.0 * cm, 5.5 * cm),
             Paragraph(f"Enhanced by {escape(algo)}", styles["BodyText"])),
        ]
        orientation = result.get("orientation_visualisation")
        binary = result.get("ridge_binary")
        if orientation is not None:
            p = save_img("orientation", np.asarray(orientation))
            img_cells.append((
                _report_image(p, np.asarray(orientation), 8.0 * cm, 5.5 * cm),
                Paragraph("Ridge orientation field", styles["BodyText"]),
            ))
        if binary is not None:
            p = save_img("binary", np.asarray(binary))
            img_cells.append((
                _report_image(p, np.asarray(binary), 8.0 * cm, 5.5 * cm),
                Paragraph("Ridge binary skeleton", styles["BodyText"]),
            ))
        while len(img_cells) < 4:
            img_cells.append((Paragraph("", styles["BodyText"]), Paragraph("", styles["BodyText"])))

        image_table = Table(
            [
                [img_cells[0][0], img_cells[1][0]],
                [img_cells[0][1], img_cells[1][1]],
                [img_cells[2][0], img_cells[3][0]],
                [img_cells[2][1], img_cells[3][1]],
            ],
            colWidths=[8.8 * cm, 8.8 * cm],
        )
        image_table.setStyle(TableStyle([
            ("ALIGN", (0, 0), (-1, -1), "CENTER"),
            ("VALIGN", (0, 0), (-1, -1), "MIDDLE"),
            ("BOTTOMPADDING", (0, 0), (-1, -1), 4),
        ]))
        story.append(Paragraph("Visual Comparison", styles["Heading2"]))
        story.append(image_table)
        story.append(Spacer(1, 0.5 * cm))

        # Section 3: Comparison metrics table (Original | Enhanced | Change)
        comp_data = [["Metric", "Original", "Enhanced", "Change", "Status"]]
        paired_metrics = [
            ("Contrast (RMS)", "original_contrast", "processed_contrast", 4),
            ("Sharpness (Laplacian)", "original_sharpness", "processed_sharpness", 1),
            ("Edge Clarity (Sobel)", "original_edge_clarity", "processed_edge_clarity", 2),
            ("Ridge-Valley Clarity", "original_ridge_valley_clarity", "processed_ridge_valley_clarity", 1),
        ]
        for label, ok, pk, dec in paired_metrics:
            ov = metrics.get(ok)
            pv = metrics.get(pk)
            if ov is None or pv is None:
                continue
            ov, pv = float(ov), float(pv)
            delta = pv - ov
            status = "+" if pv >= ov else "-"
            comp_data.append([label, _fmt(ov, dec), _fmt(pv, dec), f"{delta:+.{dec}f}", status])

        single_metrics = [
            ("SSIM (Structure Similarity)", "ssim", 3),
            ("PSNR (dB)", "psnr", 2),
            ("MSE", "mse", 2),
            ("Foreground Coverage (%)", "foreground_coverage_percent", 1),
            ("Valid Orientation Blocks", "valid_orientation_blocks", 0),
            ("Mean Orientation Coherence", "mean_orientation_coherence", 3),
            ("Total Minutiae", "minutiae_total", 0),
            ("Ridge Endings", "ridge_endings", 0),
            ("Bifurcations", "bifurcations", 0),
            ("Processing Time (ms)", "processing_time_ms", 1),
        ]
        for label, key, dec in single_metrics:
            val = metrics.get(key)
            if val is not None:
                comp_data.append([label, "N/A", _fmt(val, dec), "N/A", ""])

        comp_table = Table(
            comp_data,
            colWidths=[6.5 * cm, 2.8 * cm, 2.8 * cm, 2.5 * cm, 1.0 * cm],
            repeatRows=1,
        )
        comp_table.setStyle(TableStyle([
            ("BACKGROUND", (0, 0), (-1, 0), colors.HexColor("#2B4590")),
            ("TEXTCOLOR", (0, 0), (-1, 0), colors.white),
            ("FONTNAME", (0, 0), (-1, 0), "Helvetica-Bold"),
            ("FONTNAME", (0, 1), (0, -1), "Helvetica-Bold"),
            ("GRID", (0, 0), (-1, -1), 0.5, colors.grey),
            ("ALIGN", (1, 1), (-1, -1), "RIGHT"),
            ("VALIGN", (0, 0), (-1, -1), "MIDDLE"),
            ("ROWBACKGROUNDS", (0, 1), (-1, -1), [colors.white, colors.Color(0.95, 0.95, 0.98)]),
            ("TOPPADDING", (0, 0), (-1, -1), 4),
            ("BOTTOMPADDING", (0, 0), (-1, -1), 4),
        ]))
        story.append(Paragraph("Detailed Metrics: Original vs Enhanced", styles["Heading2"]))
        story.append(comp_table)
        document.build(story)
    return output.getvalue()
