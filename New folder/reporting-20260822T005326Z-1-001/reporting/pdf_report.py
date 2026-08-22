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


def _metric_text(value: object) -> str:
    if value is None:
        return "Not calculated"
    if isinstance(value, (float, np.floating)):
        return f"{float(value):.5f}"
    return str(value)


def build_pdf_report(
    filename: str,
    result: dict,
    algorithm_name: str | None = None,
) -> bytes:
    """Generate a professional report from the common algorithm result mapping."""
    original = result.get("original", result.get("grayscale"))
    enhanced = result.get("enhanced_image", result.get("ridge_restored", result.get("ridge_enhanced")))
    if original is None or enhanced is None:
        raise ValueError("report result must contain original and enhanced images")

    name = algorithm_name or str(result.get("algorithm_name", "Fingerprint enhancement"))
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
        Paragraph("Fingerprint Enhancement System Report", styles["Title"]),
        Paragraph(f"Algorithm: {escape(name)}", styles["Normal"]),
        Paragraph(f"Input: {escape(filename)}", styles["Normal"]),
        Paragraph(f"Generated: {datetime.now().astimezone().strftime('%Y-%m-%d %H:%M:%S %Z')}", styles["Normal"]),
        Spacer(1, 0.35 * cm),
    ]

    candidates: list[tuple[str, str, np.ndarray]] = [
        ("original", "Original fingerprint", np.asarray(original)),
        ("enhanced", "Enhanced fingerprint", np.asarray(enhanced)),
    ]
    orientation = result.get("orientation_visualisation")
    if orientation is not None:
        candidates.append(("orientation", "Estimated ridge orientation", np.asarray(orientation)))
    binary = result.get("ridge_binary")
    if binary is not None:
        candidates.append(("binary", "Isolated ridge result", np.asarray(binary)))

    with TemporaryDirectory() as temporary_directory:
        temporary = Path(temporary_directory)
        cells: list[tuple[Image, Paragraph]] = []
        for key, label, array in candidates[:4]:
            path = temporary / f"{key}.png"
            path.write_bytes(encode_png(array))
            cells.append(
                (
                    _report_image(path, array, 7.8 * cm, 5.2 * cm),
                    Paragraph(label, styles["BodyText"]),
                )
            )
        while len(cells) < 4:
            cells.append((Paragraph("", styles["BodyText"]), Paragraph("", styles["BodyText"])))
        image_table = Table(
            [
                [cells[0][0], cells[1][0]],
                [cells[0][1], cells[1][1]],
                [cells[2][0], cells[3][0]],
                [cells[2][1], cells[3][1]],
            ],
            colWidths=[8.5 * cm, 8.5 * cm],
        )
        image_table.setStyle(
            TableStyle(
                [
                    ("ALIGN", (0, 0), (-1, -1), "CENTER"),
                    ("VALIGN", (0, 0), (-1, -1), "MIDDLE"),
                    ("BOTTOMPADDING", (0, 0), (-1, -1), 6),
                ]
            )
        )
        story.extend([image_table, Spacer(1, 0.35 * cm)])

        metrics = result.get("metrics", {})
        metric_rows = [["Metric", "Value"]]
        for metric, value in metrics.items():
            metric_rows.append([str(metric).replace("_", " ").title(), _metric_text(value)])
        metric_table = Table(metric_rows, colWidths=[10.5 * cm, 5.5 * cm], repeatRows=1)
        metric_table.setStyle(
            TableStyle(
                [
                    ("BACKGROUND", (0, 0), (-1, 0), colors.lightgrey),
                    ("GRID", (0, 0), (-1, -1), 0.5, colors.grey),
                    ("FONTNAME", (0, 0), (-1, 0), "Helvetica-Bold"),
                    ("ALIGN", (1, 1), (1, -1), "RIGHT"),
                    ("VALIGN", (0, 0), (-1, -1), "MIDDLE"),
                ]
            )
        )
        story.extend([Paragraph("Image-quality metrics", styles["Heading2"]), metric_table])
        document.build(story)
    return output.getvalue()

