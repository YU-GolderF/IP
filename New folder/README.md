# Fingerprint Enhancement System

This university Image Processing project provides one shared Streamlit application
for comparing four team fingerprint-enhancement algorithms. The current working
algorithm is RHLT Ridge Flow Restoration. The other three team algorithm packages
are also available through the same comparison interface.

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
Spiral-phase RHLT Convolution  (primary)
        ↓
RHLT Edge-guided Baseline Image
        ↓
Sobel Ridge Orientation Estimation
        ↓
Doubled-angle Orientation Smoothing
        ↓
Local Ridge-wavelength Estimation
        ↓
Weak-ridge / Local-quality Mapping
        ↓
Orientation-and-frequency-adaptive Gabor Support  (secondary, bounded)
        ↓
Bounded RHLT + Gabor Fusion → Improved RHLT
        ↓
Quality-preservation Guard (RHLT candidates only)
        ↓
Binarisation → Skeleton → Minutiae
        ↓
Dashboard / Metrics
```

The spiral-phase Radial Hilbert Transform is the **primary enhancement foundation**.
Gabor filtering is a **supporting component only**: it provides bounded
orientation-guided detail to the RHLT-based baseline and cannot silently replace it.

The pipeline produces three named intermediate images:

| Key | Description |
|-----|-------------|
| `traditional_rhlt_baseline` | RHLT edge-guided sharpening; directly depends on the RHLT edge response |
| `gabor_support` | Orientation-adaptive Gabor image; used only as bounded support |
| `improved_rhlt` | RHLT baseline fused with Gabor support using a confidence-weighted blend |
| `selected_output` | Which candidate was chosen: `improved_rhlt`, `traditional_rhlt_baseline`, or `original_quality_fallback` |

### Traditional RHLT baseline

The preprocessed image is convolved with the spiral-phase RHLT point spread function
(Wu et al. 2024). The resulting isotropic edge magnitude is used to compute a
fingerprint-like baseline via RHLT edge-guided sharpening:

```text
traditional_rhlt_baseline = preprocessed + gain × 64 × sign(local_contrast) × rhlt_edge_normalised
```

This image is visually interpretable as a fingerprint and is directly produced from the
RHLT edge response. The raw RHLT edge magnitude alone is retained as a separate diagnostic.

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

### Orientation-and-frequency-guided Gabor support

Gabor filters model the spatial frequency and direction of fingerprint ridges. The
pipeline rotates each reliable analysis block into a common ridge direction and uses
the autocorrelation peak of its one-dimensional ridge-normal projection to estimate
ridge wavelength. Invalid estimates are rejected explicitly and use `gabor_lambda` as
a documented fallback. A cached two-dimensional bank covers both orientation and
wavelength bins; no random frequency is generated.

The Gabor image is used **only as bounded directional support**. It is not the final
output. The maximum Gabor contribution per pixel is controlled by `hybrid_gabor_max_weight`
(default 0.40), and regions with unreliable orientation (coherence below
`minimum_orientation_coherence`) receive zero contribution.

### Defect-aware bounded RHLT + Gabor fusion

The proposed improved RHLT image is produced by confidence-weighted blending:

```text
quality(p) = clip(local_std(p) / weak_ridge_target_contrast, 0, 1)
weakness(p) = (1 - quality(p))^clear_region_protection
evidence(p) = rhlt_evidence_floor
              + (1 - rhlt_evidence_floor) × rhlt_edge(p)^rhlt_edge_gamma
weight(p) = hybrid_gabor_max_weight × weakness(p) × coherence(p)^rhlt_support_gamma × evidence(p)
improved(p) = rhlt_baseline(p) × (1 − weight(p)) + gabor_support(p) × weight(p)
```

The RHLT evidence floor prevents weak but coherent ridge defects from receiving an
approximately zero weight merely because their edge magnitude is weak. RHLT evidence
still modulates the support, orientation-rejected regions receive zero weight, and the
weight remains in `[0, hybrid_gabor_max_weight]`. Calibrated input background pixels
outside the foreground mask remain unchanged.

### Quality-preservation guard

The selector evaluates both candidates primarily inside the fingerprint foreground.
Its transparent composite score uses ridge-valley clarity, foreground contrast, edge
clarity, structural preservation and ridge continuity, with a penalty for excessive
new minutiae. Proposed Improved RHLT is the primary output when it is safe and its
score is not materially below Traditional by more than the bounded
`selector_regression_tolerance`.

1. `improved_rhlt` — selected when structurally safe and within the configured
   regression tolerance.
2. `traditional_rhlt_baseline` — safety fallback when Improved is unsafe or
   materially worse.
3. `original_quality_fallback` — the original grayscale is preserved when both
   RHLT-derived candidates seriously degrade the fingerprint.

Pure Gabor output and pure detail-preserving sharpening are **never** eligible.
The quality selection is recorded in the `selected_output` result key.

### Controlled clean-reference experiment

The Research Experiment tab uses only `left*.bmp`, `right*.bmp`, `special1.bmp`,
`special2.bmp` and `spectial3.bmp` by default. Mild, Medium and Severe degradation
presets apply deterministic Gaussian blur, contrast reduction and Gaussian noise in
memory. The original files are never overwritten. The undegraded image is the clean
ground truth, so the experiment reports `ssim_reference`, `psnr_reference` and
`mse_reference` separately from structural preservation against the degraded input.

The command-line equivalent is:

```powershell
python scripts/evaluate_controlled.py --seed 7
```

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

After dependencies are installed, `run_windows.bat` can be double-clicked. It uses
`.venv\Scripts\python.exe`, or the repository's existing
`venv\.venv\Scripts\python.exe` layout when present.

## Running tests

With the project virtual environment active:

```powershell
python -m pytest -q
```

Tests cover blank, uniform, colour, grayscale and small images; valid dimensions;
finite orientations; synthetic wavelength recovery; weak-ridge monotonicity; bounded
fusion weights; adaptive Gabor selection; deterministic degradation; clean-reference
metrics; selector margins; RHLT dependency; the `run_rhlt()` result contract; shared
preprocessing; calibration; folder loading; and fault-tolerant batches.
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
| `frequency_block_size` | Window used for projection/autocorrelation | `24` |
| `minimum_ridge_wavelength` | Smallest accepted ridge period | `3.0` |
| `maximum_ridge_wavelength` | Largest accepted ridge period | `14.0` |
| `weak_ridge_target_contrast` | Local standard deviation treated as clear | `45.0` |
| `hybrid_gabor_max_weight` | Hard upper bound on Gabor contribution | `0.40` |
| `selector_regression_tolerance` | Maximum tolerated no-reference score regression before safety fallback | `0.010` |

The most useful parameters to experiment with are the accepted wavelength range,
`weak_ridge_target_contrast`, `hybrid_gabor_max_weight` and the selector margin. Use
the same degradation seed, dataset, preprocessing and reference metrics for comparisons.

## Assumptions and limitations

- Input arrays supplied directly to the API use RGB/RGBA or grayscale ordering.
- Automatic rotation estimation is not guessed; rotation is user-controlled.
- Local ridge frequency estimates require reliable orientation and sufficient periodic
  evidence. Rejected blocks use the explicit `gabor_lambda` fallback.
- The quality score is a transparent engineering proxy, not fingerprint-matching accuracy.
- Severe degradation may destroy ridge evidence that no classical filter can reconstruct
  without risking invented ridges.
- No physical calibration is claimed without a real scale reference.
- No ridge is reconstructed where the image contains too little reliable evidence.
- No video processing is implemented anywhere in this project.

## Reference for the retained spiral-phase diagnostic

Wu, B., Zhang, S., Gao, W., Bi, Y., & Hu, X. (2024). A method for fingerprint
edge enhancement based on radial Hilbert transform. *Electronics, 13*(19), 3886.
https://doi.org/10.3390/electronics13193886
