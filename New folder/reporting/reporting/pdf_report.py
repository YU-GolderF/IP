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
from reportlab.graphics.charts.barcharts import VerticalBarChart
from reportlab.graphics.charts.legends import Legend
from reportlab.graphics.shapes import Drawing, String
from reportlab.platypus import (
    Image,
    KeepTogether,
    Paragraph,
    SimpleDocTemplate,
    Spacer,
    Table,
    TableStyle,
)

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


def _grouped_bar_chart(
    title: str,
    categories: list[str],
    series: list[tuple[str, list[float], colors.Color]],
    *,
    value_min: float = 0.0,
    value_max: float | None = None,
) -> Drawing:
    """Build a compact, vector grouped bar chart for the PDF report."""
    drawing = Drawing(500, 235)
    drawing.add(String(250, 218, title, textAnchor="middle", fontName="Helvetica-Bold", fontSize=12))

    chart = VerticalBarChart()
    chart.x = 55
    chart.y = 42
    chart.width = 410
    chart.height = 145
    chart.data = [values for _, values, _ in series]
    chart.categoryAxis.categoryNames = categories
    chart.categoryAxis.labels.fontName = "Helvetica"
    chart.categoryAxis.labels.fontSize = 7
    chart.categoryAxis.labels.boxAnchor = "n"
    chart.categoryAxis.labels.dy = -5
    chart.valueAxis.valueMin = value_min
    all_values = [value for _, values, _ in series for value in values]
    computed_max = max(all_values, default=1.0)
    chart.valueAxis.valueMax = value_max or max(10.0, computed_max * 1.15)
    chart.valueAxis.labels.fontSize = 7
    chart.valueAxis.gridStrokeColor = colors.HexColor("#D7DCE5")
    chart.valueAxis.gridStrokeWidth = 0.5
    chart.barSpacing = 1.5
    chart.groupSpacing = 8
    for index, (_, _, colour) in enumerate(series):
        chart.bars[index].fillColor = colour
        chart.bars[index].strokeColor = colour
    drawing.add(chart)

    legend = Legend()
    legend.x = 315
    legend.y = 210
    legend.fontName = "Helvetica"
    legend.fontSize = 7
    legend.dx = 7
    legend.dy = 7
    legend.deltax = 85
    legend.colorNamePairs = [(colour, label) for label, _, colour in series]
    drawing.add(legend)
    return drawing


def _percentage_change(metrics: dict, original_key: str, processed_key: str) -> float:
    original = float(metrics.get(original_key, 0.0))
    processed = float(metrics.get(processed_key, 0.0))
    return (processed - original) / max(abs(original), 1e-9) * 100.0


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
    is_rhlt_report = all(
        key in result
        for key in (
            "traditional_rhlt_baseline",
            "improved_rhlt",
            "traditional_rhlt_metrics",
            "improved_rhlt_metrics",
        )
    )
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

    if is_rhlt_report:
        traditional_metrics = result["traditional_rhlt_metrics"]
        improved_metrics = result["improved_rhlt_metrics"]
        selected_output = str(result.get("selected_output", "unknown"))
        selected_label = {
            "improved_rhlt": "Proposed Improved RHLT",
            "traditional_rhlt_baseline": "Traditional RHLT Baseline",
            "original_quality_fallback": "Original Quality Fallback",
        }.get(selected_output, selected_output)
        selection_reason = str(result.get("selection_reason", "No selection reason was recorded."))

        story.append(Paragraph("RHLT Candidate Decision", styles["Heading2"]))
        story.append(
            Paragraph(
                f"Final selected output: <b>{escape(selected_label)}</b><br/>"
                f"{escape(selection_reason)}",
                styles["BodyText"],
            )
        )
        story.append(Spacer(1, 0.25 * cm))

        candidate_rows = [[
            "Metric",
            "Original",
            "Traditional",
            "Proposed",
            "Proposed delta",
        ]]
        candidate_specs = [
            ("Contrast", "original_contrast", "processed_contrast", 4),
            ("Ridge-Valley Clarity", "original_ridge_valley_clarity", "processed_ridge_valley_clarity", 1),
            ("Sharpness (Laplacian)", "original_sharpness", "processed_sharpness", 1),
            ("Edge Clarity (Sobel)", "original_edge_clarity", "processed_edge_clarity", 2),
        ]
        for label, original_key, processed_key, decimals in candidate_specs:
            original_value = float(improved_metrics.get(original_key, 0.0))
            traditional_value = float(traditional_metrics.get(processed_key, 0.0))
            improved_value = float(improved_metrics.get(processed_key, 0.0))
            candidate_rows.append([
                label,
                _fmt(original_value, decimals),
                _fmt(traditional_value, decimals),
                _fmt(improved_value, decimals),
                f"{improved_value - traditional_value:+.{decimals}f}",
            ])
        traditional_ssim = float(traditional_metrics.get("ssim", 0.0))
        improved_ssim = float(improved_metrics.get("ssim", 0.0))
        candidate_rows.append([
            "SSIM (Structural Similarity)",
            "1.000",
            f"{traditional_ssim:.3f}",
            f"{improved_ssim:.3f}",
            f"{improved_ssim - traditional_ssim:+.3f}",
        ])
        candidate_table = Table(
            candidate_rows,
            colWidths=[4.6 * cm, 2.5 * cm, 3.2 * cm, 3.8 * cm, 3.2 * cm],
            repeatRows=1,
        )
        candidate_table.setStyle(TableStyle([
            ("BACKGROUND", (0, 0), (-1, 0), colors.HexColor("#2B4590")),
            ("TEXTCOLOR", (0, 0), (-1, 0), colors.white),
            ("FONTNAME", (0, 0), (-1, 0), "Helvetica-Bold"),
            ("FONTNAME", (0, 1), (0, -1), "Helvetica-Bold"),
            ("GRID", (0, 0), (-1, -1), 0.5, colors.grey),
            ("ALIGN", (1, 1), (-1, -1), "RIGHT"),
            ("VALIGN", (0, 0), (-1, -1), "MIDDLE"),
            ("ROWBACKGROUNDS", (0, 1), (-1, -1), [colors.white, colors.HexColor("#F3F5FA")]),
            ("TOPPADDING", (0, 0), (-1, -1), 4),
            ("BOTTOMPADDING", (0, 0), (-1, -1), 4),
        ]))
        story.append(candidate_table)
        story.append(Spacer(1, 0.45 * cm))

        traditional_changes = [
            (float(traditional_metrics.get("cii", 1.0)) - 1.0) * 100.0,
            _percentage_change(traditional_metrics, "original_ridge_valley_clarity", "processed_ridge_valley_clarity"),
            float(traditional_metrics.get("sharpness_improvement_pct", 0.0)),
            float(traditional_metrics.get("edge_improvement_pct", 0.0)),
        ]
        improved_changes = [
            (float(improved_metrics.get("cii", 1.0)) - 1.0) * 100.0,
            _percentage_change(improved_metrics, "original_ridge_valley_clarity", "processed_ridge_valley_clarity"),
            float(improved_metrics.get("sharpness_improvement_pct", 0.0)),
            float(improved_metrics.get("edge_improvement_pct", 0.0)),
        ]
        chart_blue = colors.HexColor("#315EAD")
        chart_green = colors.HexColor("#20A464")
        story.append(
            _grouped_bar_chart(
                "Enhancement Change vs Original (%)",
                ["Contrast", "RVC", "Sharpness", "Edge"],
                [
                    ("Traditional RHLT", traditional_changes, chart_blue),
                    ("Proposed Improved", improved_changes, chart_green),
                ],
            )
        )
        story.append(Spacer(1, 0.25 * cm))

        traditional_score = float(result.get("traditional_quality_score", 0.0)) * 100.0
        improved_score = float(result.get("improved_quality_score", 0.0)) * 100.0
        story.append(
            _grouped_bar_chart(
                "Structural Preservation and Balanced Quality (%)",
                ["SSIM", "Quality score"],
                [
                    ("Traditional RHLT", [traditional_ssim * 100.0, traditional_score], chart_blue),
                    ("Proposed Improved", [improved_ssim * 100.0, improved_score], chart_green),
                ],
                value_max=105.0,
            )
        )
        story.append(Spacer(1, 0.35 * cm))

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

        if is_rhlt_report:
            traditional_arr = np.asarray(result["traditional_rhlt_baseline"])
            improved_arr = np.asarray(result["improved_rhlt"])
            traditional_path = save_img("traditional_rhlt", traditional_arr)
            improved_path = save_img("proposed_improved_rhlt", improved_arr)
            selected_output = str(result.get("selected_output", ""))
            traditional_caption = (
                "Traditional RHLT Baseline - Final Selected"
                if selected_output == "traditional_rhlt_baseline"
                else "Traditional RHLT Baseline"
            )
            improved_caption = (
                "Proposed Improved RHLT - Final Selected"
                if selected_output == "improved_rhlt"
                else "Proposed Improved RHLT"
            )
            image_table = Table(
                [
                    [
                        _report_image(orig_path, orig_arr, 5.4 * cm, 6.3 * cm),
                        _report_image(traditional_path, traditional_arr, 5.4 * cm, 6.3 * cm),
                        _report_image(improved_path, improved_arr, 5.4 * cm, 6.3 * cm),
                    ],
                    [
                        Paragraph("Original fingerprint", styles["BodyText"]),
                        Paragraph(traditional_caption, styles["BodyText"]),
                        Paragraph(improved_caption, styles["BodyText"]),
                    ],
                ],
                colWidths=[5.85 * cm, 5.85 * cm, 5.85 * cm],
            )
        else:
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

        if is_rhlt_report:
            diagnostics = []
            for key, label in (
                ("orientation_visualisation", "Ridge orientation field"),
                ("ridge_binary", "Ridge binary skeleton"),
                ("minutiae_overlay", "Minutiae overlay"),
            ):
                array = result.get(key)
                if array is None:
                    continue
                array = np.asarray(array)
                path = save_img(key, array)
                diagnostics.append((
                    _report_image(path, array, 5.4 * cm, 5.2 * cm),
                    Paragraph(label, styles["BodyText"]),
                ))
            if diagnostics:
                while len(diagnostics) < 3:
                    diagnostics.append((Paragraph("", styles["BodyText"]), Paragraph("", styles["BodyText"])))
                diagnostics_table = Table(
                    [
                        [cell[0] for cell in diagnostics[:3]],
                        [cell[1] for cell in diagnostics[:3]],
                    ],
                    colWidths=[5.85 * cm, 5.85 * cm, 5.85 * cm],
                )
                diagnostics_table.setStyle(TableStyle([
                    ("ALIGN", (0, 0), (-1, -1), "CENTER"),
                    ("VALIGN", (0, 0), (-1, -1), "MIDDLE"),
                    ("BOTTOMPADDING", (0, 0), (-1, -1), 4),
                ]))
                story.append(Paragraph("Diagnostic Outputs", styles["Heading2"]))
                story.append(diagnostics_table)
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
            colWidths=[6.2 * cm, 2.8 * cm, 2.8 * cm, 2.5 * cm, 1.3 * cm],
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
        story.append(KeepTogether([
            Paragraph("Detailed Metrics: Original vs Enhanced", styles["Heading2"]),
            comp_table,
        ]))
        document.build(story)
    return output.getvalue()

def build_comparison_report(
    filename: str,
    original: np.ndarray,
    results: dict[str, dict],
) -> bytes:
    """Generate a multi-algorithm comparison PDF report."""
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
        Paragraph("Multi-Algorithm Comparison Report", styles["Title"]),
        Paragraph(f"Input file: {escape(filename)}", styles["Normal"]),
        Paragraph(f"Generated: {datetime.now().astimezone().strftime('%Y-%m-%d %H:%M:%S %Z')}", styles["Normal"]),
        Spacer(1, 0.4 * cm),
    ]

    with TemporaryDirectory() as tmp_dir:
        tmp = Path(tmp_dir)

        def save_img(key: str, arr: np.ndarray) -> Path:
            p = tmp / f"{key}.png"
            p.write_bytes(encode_png(arr))
            return p

        # Section 1: Visual Comparison
        story.append(Paragraph("Visual Comparison", styles["Heading2"]))
        
        orig_arr = np.asarray(original)
        orig_path = save_img("original", orig_arr)
        
        img_cells = [
            (_report_image(orig_path, orig_arr, 5.0 * cm, 5.0 * cm), Paragraph("Original", styles["BodyText"]))
        ]
        
        for algo_name, result in results.items():
            enh_arr = np.asarray(result["enhanced_image"])
            enh_path = save_img(f"enhanced_{algo_name.replace(' ', '_')}", enh_arr)
            img_cells.append((
                _report_image(enh_path, enh_arr, 5.0 * cm, 5.0 * cm),
                Paragraph(escape(algo_name), styles["BodyText"])
            ))
            
        # Group into rows of 3
        table_data = []
        for i in range(0, len(img_cells), 3):
            row_images = [img_cells[j][0] if j < len(img_cells) else Paragraph("", styles["BodyText"]) for j in range(i, i+3)]
            row_labels = [img_cells[j][1] if j < len(img_cells) else Paragraph("", styles["BodyText"]) for j in range(i, i+3)]
            table_data.append(row_images)
            table_data.append(row_labels)
            
        image_table = Table(table_data, colWidths=[5.5 * cm, 5.5 * cm, 5.5 * cm])
        image_table.setStyle(TableStyle([
            ("ALIGN", (0, 0), (-1, -1), "CENTER"),
            ("VALIGN", (0, 0), (-1, -1), "MIDDLE"),
            ("BOTTOMPADDING", (0, 0), (-1, -1), 10),
        ]))
        story.append(image_table)
        story.append(Spacer(1, 0.5 * cm))

        # Section 2: Leaderboard Metrics
        story.append(Paragraph("Quantitative Benchmark", styles["Heading2"]))
        
        comp_data = [["Algorithm", "CII", "Sharpness Δ%", "Edge Δ%", "SSIM", "RVC (orig)", "RVC (enh)", "Time (ms)"]]
        
        for algo_name, result in results.items():
            m = result["metrics"]
            comp_data.append([
                algo_name,
                f"{m.get('cii', 1.0):.2f}x",
                f"{m.get('sharpness_improvement_pct', 0.0):+.1f}%",
                f"{m.get('edge_improvement_pct', 0.0):+.1f}%",
                f"{m.get('ssim', 0.0):.3f}",
                f"{m.get('original_ridge_valley_clarity', 0.0):.0f}",
                f"{m.get('processed_ridge_valley_clarity', 0.0):.0f}",
                f"{result.get('processing_time_ms', 0.0):.1f}"
            ])
            
        comp_table = Table(comp_data, colWidths=[4.0 * cm, 1.8 * cm, 2.5 * cm, 2.0 * cm, 1.5 * cm, 2.0 * cm, 2.0 * cm, 2.0 * cm])
        comp_table.setStyle(TableStyle([
            ("BACKGROUND", (0, 0), (-1, 0), colors.HexColor("#2B4590")),
            ("TEXTCOLOR", (0, 0), (-1, 0), colors.white),
            ("FONTNAME", (0, 0), (-1, 0), "Helvetica-Bold"),
            ("FONTSIZE", (0, 0), (-1, -1), 8),
            ("GRID", (0, 0), (-1, -1), 0.5, colors.grey),
            ("ALIGN", (1, 1), (-1, -1), "RIGHT"),
            ("VALIGN", (0, 0), (-1, -1), "MIDDLE"),
            ("ROWBACKGROUNDS", (0, 1), (-1, -1), [colors.white, colors.Color(0.95, 0.95, 0.98)]),
            ("TOPPADDING", (0, 0), (-1, -1), 4),
            ("BOTTOMPADDING", (0, 0), (-1, -1), 4),
        ]))
        story.append(comp_table)
        story.append(Spacer(1, 0.5 * cm))

        # Section 3: Best Algorithm Conclusion
        def _score(m: dict) -> float:
            return (
                float(m.get("cii", 1.0))
                + float(m.get("sharpness_improvement_pct", 0.0)) / 100.0
                + float(m.get("edge_improvement_pct", 0.0)) / 100.0
            )

        if results:
            best_overall = max(results.items(), key=lambda x: _score(x[1]["metrics"]))
            fastest = min(results.items(), key=lambda x: x[1].get("processing_time_ms", 9999))
            sharpest = max(
                results.items(),
                key=lambda x: float(x[1]["metrics"].get("sharpness_improvement_pct", 0.0)),
            )

            awards_data = [
                ["Award", "Algorithm", "Detail"],
                ["\U0001f947 Best Overall", best_overall[0], f"Score: {_score(best_overall[1]['metrics']):.3f}"],
                ["\u26a1 Fastest", fastest[0], f"{fastest[1].get('processing_time_ms', 0):.1f} ms"],
                ["\U0001f50d Sharpest", sharpest[0],
                 f"{sharpest[1]['metrics'].get('sharpness_improvement_pct', 0.0):+.1f}%"],
            ]
            awards_table = Table(awards_data, colWidths=[4.0 * cm, 6.0 * cm, 6.0 * cm])
            awards_table.setStyle(TableStyle([
                ("BACKGROUND", (0, 0), (-1, 0), colors.HexColor("#2B4590")),
                ("TEXTCOLOR", (0, 0), (-1, 0), colors.white),
                ("FONTNAME", (0, 0), (-1, 0), "Helvetica-Bold"),
                ("FONTNAME", (0, 1), (0, -1), "Helvetica-Bold"),
                ("GRID", (0, 0), (-1, -1), 0.5, colors.grey),
                ("VALIGN", (0, 0), (-1, -1), "MIDDLE"),
                ("ROWBACKGROUNDS", (0, 1), (-1, -1), [colors.Color(0.88, 0.97, 0.88), colors.white]),
                ("TOPPADDING", (0, 0), (-1, -1), 5),
                ("BOTTOMPADDING", (0, 0), (-1, -1), 5),
            ]))
            story.append(Paragraph("Best Algorithm Awards", styles["Heading2"]))
            story.append(awards_table)

        document.build(story)
    return output.getvalue()
