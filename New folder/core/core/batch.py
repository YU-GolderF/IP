"""Reusable fault-tolerant batch processing for every team algorithm."""

from __future__ import annotations

from collections.abc import Callable, Iterable
from time import perf_counter
from typing import Any

from .image_io import LoadedImage


def process_batch(
    images: Iterable[LoadedImage],
    algorithm_name: str,
    processor: Callable[[LoadedImage], dict[str, Any]],
) -> tuple[list[dict[str, Any]], list[dict[str, str]]]:
    """Process each image independently so one failure cannot abort a full batch."""
    records: list[dict[str, Any]] = []
    errors: list[dict[str, str]] = []
    for item in images:
        started = perf_counter()
        try:
            result = processor(item)
            elapsed_ms = float(result.get("processing_time_ms", (perf_counter() - started) * 1000.0))
            records.append(
                {
                    "filename": item.filename,
                    "algorithm": algorithm_name,
                    "processing_time_ms": elapsed_ms,
                    "metrics": result.get("metrics", {}),
                    "result": result,
                }
            )
        except Exception as exc:  # Batch boundary: preserve other valid results.
            errors.append({"filename": item.filename, "error": str(exc)})
    return records, errors

