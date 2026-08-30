"""Run the reproducible clean-reference RHLT experiment from the command line."""

from __future__ import annotations

import argparse
import re
import sys
from pathlib import Path

import numpy as np
import pandas as pd


PROJECT_ROOT = Path(__file__).resolve().parents[1]
DATASET_ROOT = PROJECT_ROOT.parent / "image"
for package in ("algorithms", "core", "reporting"):
    sys.path.insert(0, str(PROJECT_ROOT / package))

from algorithms.rhlt.config import RHLTConfig
from algorithms.rhlt.pipeline import run_rhlt
from core.degradation import DEGRADATION_PRESETS, degrade_fingerprint
from core.image_io import load_image


DATASET_PATTERN = re.compile(
    r"(?:left[1-5]|right[1-5]|special[12]|spectial3)\.bmp", re.IGNORECASE
)


def reference_score(metrics: dict) -> float:
    ssim = float(metrics.get("foreground_ssim_reference", 0.0))
    psnr = float(metrics.get("foreground_psnr_reference", 0.0))
    mse = float(metrics.get("foreground_mse_reference", 65025.0))
    return float(
        0.50 * np.clip(ssim, 0.0, 1.0)
        + 0.30 * np.clip(psnr / 40.0, 0.0, 1.0)
        + 0.20 * (1.0 - np.clip(mse / 65025.0, 0.0, 1.0))
    )


def run(seed: int) -> pd.DataFrame:
    paths = sorted(
        (path for path in DATASET_ROOT.iterdir() if DATASET_PATTERN.fullmatch(path.name)),
        key=lambda path: path.name.lower(),
    )
    rows: list[dict] = []
    for path in paths:
        clean = load_image(path).image
        for level in DEGRADATION_PRESETS:
            degraded = degrade_fingerprint(clean, level, seed=seed)
            result = run_rhlt(degraded, RHLTConfig(), reference_image=clean)
            for candidate, metrics, score in (
                (
                    "Traditional RHLT",
                    result["traditional_rhlt_metrics"],
                    reference_score(result["traditional_rhlt_metrics"]),
                ),
                (
                    "Proposed Improved RHLT",
                    result["improved_rhlt_metrics"],
                    reference_score(result["improved_rhlt_metrics"]),
                ),
            ):
                rows.append(
                    {
                        "filename": path.name,
                        "level": level,
                        "candidate": candidate,
                        "foreground_ssim_reference": metrics["foreground_ssim_reference"],
                        "foreground_psnr_reference": metrics["foreground_psnr_reference"],
                        "foreground_mse_reference": metrics["foreground_mse_reference"],
                        "reference_score": score,
                        "selected_output": result["selected_output"],
                        "selection_reason": result["selection_reason"],
                        "mean_fusion_weight": result["mean_fusion_weight"],
                        "maximum_fusion_weight": result["maximum_fusion_weight"],
                    }
                )
    return pd.DataFrame(rows)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--seed", type=int, default=7)
    parser.add_argument("--csv", type=Path)
    args = parser.parse_args()
    frame = run(args.seed)
    if frame.empty:
        raise SystemExit("No original BMP dataset images were found.")
    summary = frame.groupby(["level", "candidate"], sort=False).agg(
        {
            "foreground_ssim_reference": ["mean", "std"],
            "foreground_psnr_reference": ["mean", "std"],
            "foreground_mse_reference": ["mean", "std"],
            "reference_score": ["mean", "std"],
        }
    )
    print(summary.to_string())
    pivot = frame.pivot(index=["filename", "level"], columns="candidate", values="reference_score")
    wins = int((pivot["Proposed Improved RHLT"] > pivot["Traditional RHLT"]).sum())
    print(f"\nImproved wins: {wins}/{len(pivot)}")
    print("\nSelected outputs:")
    print(frame.drop_duplicates(["filename", "level"])["selected_output"].value_counts().to_string())
    if args.csv:
        args.csv.parent.mkdir(parents=True, exist_ok=True)
        frame.to_csv(args.csv, index=False)
        print(f"\nWrote {args.csv.resolve()}")


if __name__ == "__main__":
    main()
