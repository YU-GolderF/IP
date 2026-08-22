"""Backward-compatible imports for the shared report implementation."""

from reporting.pdf_report import build_pdf_report, encode_png

__all__ = ["build_pdf_report", "encode_png"]
