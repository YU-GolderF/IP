"""Regenerate the two report examples used for visual quality assurance."""

from __future__ import annotations

import sys
from pathlib import Path

import cv2


ROOT = Path(__file__).resolve().parents[1]
for package in ("algorithms", "core", "reporting"):
    sys.path.insert(0, str(ROOT / package))

from algorithms import ALGORITHM_STATUS, get_algorithm_runner  # noqa: E402
from algorithms.rhlt import RHLTConfig, run_rhlt  # noqa: E402
from core import CalibrationConfig, PreprocessingConfig, calibrate_image  # noqa: E402
from reporting.pdf_report import build_comparison_report, build_pdf_report  # noqa: E402


def main() -> None:
    image_path = ROOT.parent / "image" / "1.png"
    image = cv2.imread(str(image_path), cv2.IMREAD_GRAYSCALE)
    if image is None:
        raise FileNotFoundError(image_path)

    calibrated = calibrate_image(image, CalibrationConfig())["image"]
    preprocessing = PreprocessingConfig()
    rhlt_result = run_rhlt(
        calibrated,
        RHLTConfig(),
        preprocessing_config=preprocessing,
    )

    comparison_results: dict[str, dict] = {}
    for item in ALGORITHM_STATUS:
        if not item["available"]:
            continue
        name = item["name"]
        if name == "RHLT":
            result = rhlt_result
        else:
            runner = get_algorithm_runner(name)
            kwargs = {"preprocessing_config": preprocessing}
            if name == "Unsharp Masking":
                kwargs["final_only"] = True
            result = runner(calibrated, **kwargs)
        comparison_results[name] = result

    output_dir = ROOT / "output" / "pdf"
    output_dir.mkdir(parents=True, exist_ok=True)
    (output_dir / "rhlt_fingerprint_report.pdf").write_bytes(
        build_pdf_report(image_path.name, rhlt_result, "RHLT")
    )
    (output_dir / "algorithm_comparison_report.pdf").write_bytes(
        build_comparison_report(image_path.name, image, comparison_results)
    )
    print(f"Generated reports in {output_dir}")


if __name__ == "__main__":
    main()
