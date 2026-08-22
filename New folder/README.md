# Fingerprint Enhancement System

This university Image Processing project provides one shared Streamlit application
for comparing four team fingerprint-enhancement algorithms. The current working
algorithm is RHLT Ridge Flow Restoration. The other three algorithm packages are
owned by other team members and are intentionally not implemented here.

The system focuses on restoring visible ridge flow, improving low-quality fingerprint
clarity, and isolating ridge structures for later feature analysis. It uses classical
image processing only. It does not contain video processing, deep learning, or a
pretrained neural network.

## Final project structure

```text
project/
├── app.py                         # one shared Streamlit website
├── core/                          # shared by every team algorithm
│   ├── __init__.py
│   ├── batch.py                   # fault-tolerant bulk processing
│   ├── calibration.py             # resize, rotation and optional rectification
│   ├── image_io.py                # single, multiple and folder image loading
│   ├── metrics.py                 # common comparison metrics and optional SSIM
│   └── preprocessing.py           # grayscale, denoise, normalise and CLAHE
├── reporting/
│   ├── __init__.py
│   └── pdf_report.py              # shared PDF report generator
├── algorithms/
│   ├── rhlt/                      # Member 1 implementation
│   │   ├── __init__.py
│   │   ├── config.py
│   │   ├── core.py                # retained spiral-phase RHLT diagnostic
│   │   ├── metrics.py             # RHLT-specific diagnostic metrics
│   │   ├── orientation.py         # Sobel ridge orientation and smoothing
│   │   ├── pipeline.py            # run_rhlt() and compatibility pipeline
│   │   ├── postprocess.py         # binary cleanup, skeleton and minutiae
│   │   ├── preprocess.py          # compatibility imports to shared core
│   │   ├── reporting.py           # compatibility import to shared reporting
│   │   ├── ridge_filter.py        # cached orientation-adaptive Gabor filtering
│   │   └── segmentation.py        # block-based foreground segmentation
│   ├── algorithm_2/               # Member 2; not modified
│   ├── algorithm_3/               # Member 3; not modified
│   └── algorithm_4/               # Member 4; not modified
├── tests/
│   ├── conftest.py
│   ├── test_core.py
│   ├── test_rhlt_flow.py
│   └── test_shared.py
├── requirements.txt
├── run_windows.bat
└── .gitignore
```

There is only one Streamlit website: `app.py`. Individual algorithm folders should
contain processing code, not separate websites.

## Shared preprocessing

`core/preprocessing.py` gives every algorithm the same conservative starting image:

1. Convert RGB/RGBA or grayscale input to uint8 grayscale.
2. Apply mild Gaussian noise removal.
3. Optionally apply a small median filter for impulse noise.
4. Perform percentile intensity normalisation.
5. Apply moderate CLAHE local contrast enhancement.

The values are configurable through `PreprocessingConfig`. Median filtering is off by
default to avoid unnecessary smoothing, and CLAHE uses a moderate clip limit so ridge
noise is not over-amplified.

## Calibration

`core/calibration.py` can resize images consistently while preserving aspect ratio,
apply a user-supplied rotation correction, and optionally rectify four known corner
points. It records both original and processed dimensions. It does not calculate
pixels-to-millimetres because this project has no physical reference scale.

## RHLT ridge-flow pipeline

```text
Fingerprint Input
        ↓
Shared Preprocessing
        ↓
Foreground Segmentation
        ↓
Sobel Gradient Calculation
        ↓
Local Ridge Orientation Estimation
        ↓
Orientation Field Smoothing
        ↓
Orientation-Adaptive Gabor Filtering
        ↓
Ridge Flow Restored Image
        ↓
Optional Ridge Isolation
        ↓
Dashboard / Metrics
```

The original spiral-phase Radial Hilbert Transform implementation is retained as a
separate diagnostic edge response and ablation experiment. The ridge-flow restoration
output uses local orientation-adaptive Gabor filtering as required by the assignment.

### Foreground segmentation

The preprocessed image is divided into local blocks. Blocks with sufficient local
standard deviation are considered fingerprint foreground. Smooth background and blank
images receive an empty mask rather than a permanent all-foreground debug mask, so
background regions are not aggressively enhanced.

### Sobel gradients and ridge orientation

Sobel calculates horizontal and vertical intensity changes, called `Gx` and `Gy`.
These gradients normally point across a ridge. For every foreground block, the system
accumulates:

```text
2 × Gx × Gy
Gx² - Gy²
```

The two structure-tensor terms estimate the dominant gradient direction. Rotating that
direction by 90 degrees gives the direction running along the fingerprint ridge. Blocks
with almost no gradient energy are rejected safely.

### Why doubled-angle smoothing is necessary

Ridge orientation repeats every 180 degrees. Directions of 1 degree and 179 degrees
are almost the same ridge direction, although an ordinary numeric average would give
90 degrees, which is incorrect. The implementation therefore converts every angle to:

```python
cos(2 * theta)
sin(2 * theta)
```

It smooths those components and reconstructs the angle with:

```python
theta = 0.5 * atan2(smoothed_sin, smoothed_cos)
```

### Orientation-adaptive Gabor filtering

Gabor filters are useful because fingerprint ridges have both a local direction and a
repeating ridge spacing. A bank of cached Gabor kernels represents several directions
between 0 and 180 degrees. For each valid block, the nearest kernel direction is chosen
from the smoothed orientation field. The system does not generate a new fingerprint;
it strengthens weak ridge evidence only where the image and neighbouring flow support
that direction. Uniform or unreliable regions are returned safely without invented
ridge patterns.

### Optional ridge isolation

The grayscale restored result is preserved as the main output. A separate binary result
uses Otsu thresholding and mild connected-component cleanup. Morphology is deliberately
limited so adjacent ridges are less likely to merge.

## Public algorithm interface

RHLT exposes both names below:

```python
from algorithms.rhlt import run, run_rhlt

result = run(image)
```

Important output keys include:

```text
preprocessed
foreground_mask
orientation_field
orientation_visualisation
ridge_restored
ridge_binary
enhanced_image
metrics
processing_time_ms
warnings
```

Other members can later expose the same `run(image)` concept from their own packages.
Their implementations do not need to import or modify RHLT.

## Dashboard and bulk input

The central Streamlit application supports:

- one or multiple uploaded fingerprint images;
- all supported images from a local Windows folder path;
- `.png`, `.jpg`, `.jpeg`, `.bmp`, `.tif`, and `.tiff`;
- safe skipping of invalid or corrupted files;
- original, preprocessing, mask, orientation, restored and binary previews;
- batch summary metrics, CSV export, ZIP image export and PDF reporting;
- a retained legacy RHLT PSF view and ablation study.

The folder field reads a path on the computer running Streamlit. Uploaded images remain
the appropriate input method if the website is later hosted on another machine.

## Metrics and interpretation

The shared dashboard reports input and enhanced contrast, standard deviation, variance,
Laplacian sharpness, Sobel edge clarity, dimensions and processing time. RHLT also
reports foreground coverage and the number of valid orientation blocks.

These are image-quality and diagnostic measurements. They do not measure fingerprint
matching accuracy. SSIM is calculated only when an actual reference image is supplied;
the normal dashboard does not treat the original degraded image as ground truth.

Crossing-number minutiae remain a visual diagnostic and must not be interpreted as
ground-truth recognition accuracy.

## Windows installation and startup

Open PowerShell in the project root:

```powershell
py -m venv .venv
& ".\.venv\Scripts\Activate.ps1"
python -m pip install -r requirements.txt
python -m streamlit run app.py
```

After dependencies are installed, `run_windows.bat` can be double-clicked. It always
uses the Python interpreter inside the project `.venv`.

## Running tests

With the project virtual environment active:

```powershell
python -m pytest -q
```

Tests cover blank, uniform, colour, grayscale and small images; valid dimensions;
finite orientations; segmentation masks; Gabor output; the `run_rhlt()` result contract;
shared preprocessing; calibration; metrics; folder loading; and fault-tolerant batches.
They do not depend on `algorithm_2`, `algorithm_3`, or `algorithm_4`.

## Dependencies

- NumPy: array and numerical processing
- OpenCV: filtering, gradients, calibration and image decoding
- pandas: dashboard tables and CSV data
- Pillow: Streamlit-compatible image dependency
- Streamlit: shared GUI/dashboard
- ReportLab: lightweight PDF reports
- pytest: automated tests

SciPy and scikit-image are not required. SSIM is implemented with NumPy and OpenCV.

## Useful experiment parameters

| Parameter | Purpose | Suggested starting point |
|---|---|---:|
| `block_size` | Local segmentation/orientation region | `16` |
| `orientation_smoothing_sigma` | Smooth neighbouring ridge directions | `1.0` |
| `orientation_bins` | Number of cached Gabor directions | `12` |
| `gabor_kernel_size` | Spatial support of each Gabor filter | `21` |
| `gabor_sigma` | Gabor envelope width | `4.0` |
| `gabor_lambda` | Expected ridge wavelength in pixels | `10.0` |
| `gabor_gamma` | Gabor aspect ratio | `0.5` |
| `gabor_strength` | Conservative restoration amount | `0.75` |
| `segmentation_min_std` | Reject smooth background blocks | `6.0` |

The most useful parameters to experiment with are `block_size`, `gabor_lambda`,
`gabor_strength`, and `orientation_smoothing_sigma`. Use the same dataset, calibration,
preprocessing and metrics for every team algorithm comparison.

## Assumptions and limitations

- Input arrays supplied directly to the API use RGB/RGBA or grayscale ordering.
- Automatic rotation estimation is not guessed; rotation is user-controlled.
- Local ridge frequency estimation is optional and is not enabled because an unstable
  estimate could harm the simpler, explainable baseline. `gabor_lambda` is configurable.
- No physical calibration is claimed without a real scale reference.
- No ridge is reconstructed where the image contains too little reliable evidence.
- No video processing is implemented anywhere in this project.

## Reference for the retained spiral-phase diagnostic

Wu, B., Zhang, S., Gao, W., Bi, Y., & Hu, X. (2024). A method for fingerprint
edge enhancement based on radial Hilbert transform. *Electronics, 13*(19), 3886.
https://doi.org/10.3390/electronics13193886
