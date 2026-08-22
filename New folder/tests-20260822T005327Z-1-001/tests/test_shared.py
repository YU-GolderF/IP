from __future__ import annotations

import cv2
import numpy as np

from core.batch import process_batch
from core.calibration import CalibrationConfig, calibrate_image, resize_preserve_aspect_ratio
from core.image_io import LoadedImage, load_images_from_folder
from core.metrics import calculate_image_metrics
from core.preprocessing import PreprocessingConfig, preprocess_fingerprint, preprocess_with_stages


def test_shared_preprocessing_accepts_colour_and_grayscale():
    gray = np.tile(np.arange(64, dtype=np.uint8), (48, 1))
    colour = np.dstack([gray, np.flipud(gray), gray])
    config = PreprocessingConfig(use_median_filter=True, median_kernel_size=3)

    gray_result = preprocess_fingerprint(gray, config)
    colour_stages = preprocess_with_stages(colour, config)

    assert gray_result.shape == gray.shape
    assert gray_result.dtype == np.uint8
    assert colour_stages["enhanced"].shape == gray.shape
    assert colour_stages["enhanced"].dtype == np.uint8


def test_calibration_preserves_aspect_ratio_and_tracks_dimensions():
    image = np.zeros((100, 200), dtype=np.uint8)
    resized = resize_preserve_aspect_ratio(image, max_width=100, max_height=100)
    result = calibrate_image(image, CalibrationConfig(max_width=100, max_height=100))

    assert resized.shape == (50, 100)
    assert result["original_dimensions"] == (200, 100)
    assert result["processed_dimensions"] == (100, 50)


def test_metrics_only_compute_ssim_with_reference():
    image = np.tile(np.arange(64, dtype=np.uint8), (64, 1))
    without_reference = calculate_image_metrics(image, image)
    with_reference = calculate_image_metrics(image, image, reference=image)

    assert without_reference["ssim"] is None
    assert with_reference["ssim"] > 0.999
    assert without_reference["original_sharpness"] >= 0


def test_folder_loading_ignores_unsupported_and_reports_corrupt_files(tmp_path):
    valid = np.full((16, 16), 127, dtype=np.uint8)
    encoded, buffer = cv2.imencode(".png", valid)
    assert encoded
    (tmp_path / "valid.png").write_bytes(bytes(buffer))
    (tmp_path / "notes.txt").write_text("not an image", encoding="utf-8")
    (tmp_path / "broken.jpg").write_bytes(b"not a valid jpeg")

    loaded, errors = load_images_from_folder(tmp_path)

    assert [item.filename for item in loaded] == ["valid.png"]
    assert len(errors) == 1
    assert errors[0]["filename"] == "broken.jpg"


def test_batch_processing_continues_after_one_failure():
    items = [
        LoadedImage("good.png", np.zeros((8, 8), dtype=np.uint8)),
        LoadedImage("bad.png", np.zeros((8, 8), dtype=np.uint8)),
    ]

    def processor(item):
        if item.filename == "bad.png":
            raise ValueError("intentional test failure")
        return {"metrics": {"score": 1.0}, "processing_time_ms": 1.0}

    records, errors = process_batch(items, "Test", processor)

    assert len(records) == 1
    assert records[0]["filename"] == "good.png"
    assert len(errors) == 1
    assert errors[0]["filename"] == "bad.png"

