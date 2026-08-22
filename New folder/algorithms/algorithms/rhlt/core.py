from __future__ import annotations

import numpy as np


def _radial_aperture(r_norm: np.ndarray, apodisation: float) -> np.ndarray:
    """Circular aperture with optional cosine taper to suppress PSF sidelobes."""
    window = np.zeros_like(r_norm, dtype=np.float64)
    if apodisation <= 1e-12:
        window[r_norm <= 1.0] = 1.0
        return window

    alpha = float(np.clip(apodisation, 1e-6, 0.95))
    flat_end = 1.0 - alpha
    inside = r_norm <= flat_end
    taper = (r_norm > flat_end) & (r_norm <= 1.0)
    window[inside] = 1.0
    z = (r_norm[taper] - flat_end) / alpha
    window[taper] = 0.5 * (1.0 + np.cos(np.pi * z))
    return window


def build_vortex_transfer(
    size: int,
    topological_charge: int = 1,
    aperture_ratio: float = 0.90,
    apodisation: float = 0.0,
) -> np.ndarray:
    """
    Build a discrete spiral-phase transfer function H(rho, phi) = A(rho)e^(i*l*phi).

    l=1 is the isotropic setting used in Wu et al. (2024). Optional radial
    apodisation is an experimental extension for reducing sidelobes.
    """
    if size % 2 == 0:
        raise ValueError("size must be odd")
    c = size // 2
    y, x = np.mgrid[-c:c + 1, -c:c + 1]
    rho = np.hypot(x, y)
    theta = np.arctan2(y, x)
    radius = max(aperture_ratio * c, 1e-6)
    r_norm = rho / radius
    aperture = _radial_aperture(r_norm, apodisation)
    transfer = aperture * np.exp(1j * topological_charge * theta)
    transfer[c, c] = 0.0
    return transfer


def build_rhlt_psf(
    size: int = 65,
    topological_charge: int = 1,
    aperture_ratio: float = 0.90,
    apodisation: float = 0.0,
) -> np.ndarray:
    """Create a complex spatial-domain RHLT point spread function (PSF)."""
    transfer = build_vortex_transfer(size, topological_charge, aperture_ratio, apodisation)
    psf = np.fft.fftshift(np.fft.ifft2(np.fft.ifftshift(transfer)))
    norm = np.sum(np.abs(psf))
    if norm > 0:
        psf = psf / norm
    return psf


def robust_rescale(array: np.ndarray, low: float = 1.0, high: float = 99.5, mask: np.ndarray | None = None) -> np.ndarray:
    """Convert a floating response to uint8 using robust percentiles."""
    values = np.asarray(array, dtype=np.float64)
    sample = values[mask] if mask is not None and np.any(mask) else values.ravel()
    lo, hi = np.percentile(sample, [low, high])
    if hi <= lo + 1e-12:
        return np.zeros(values.shape, dtype=np.uint8)
    out = (values - lo) / (hi - lo)
    return np.clip(out * 255.0, 0, 255).astype(np.uint8)


def _fftconvolve_same(image: np.ndarray, kernel: np.ndarray) -> np.ndarray:
    """Linear 2-D convolution using NumPy FFT only; returns the 'same' crop."""
    image = np.asarray(image)
    kernel = np.asarray(kernel)
    full_shape = (image.shape[0] + kernel.shape[0] - 1, image.shape[1] + kernel.shape[1] - 1)
    spectrum = np.fft.fft2(image, s=full_shape) * np.fft.fft2(kernel, s=full_shape)
    full = np.fft.ifft2(spectrum)
    sy = (kernel.shape[0] - 1) // 2
    sx = (kernel.shape[1] - 1) // 2
    return full[sy:sy + image.shape[0], sx:sx + image.shape[1]]


def apply_rhlt(gray: np.ndarray, psf: np.ndarray, mask: np.ndarray | None = None) -> tuple[np.ndarray, np.ndarray]:
    """
    Apply the complex RHLT PSF by convolution.

    Returns the complex response and its robustly scaled edge magnitude.
    Reflect padding reduces artificial image-border edges.
    """
    image = gray.astype(np.float64) / 255.0
    pad = psf.shape[0] // 2
    padded = np.pad(image, pad_width=pad, mode="reflect")
    response_padded = _fftconvolve_same(padded, psf)
    response = response_padded[pad:pad + gray.shape[0], pad:pad + gray.shape[1]]
    magnitude = np.abs(response)
    edge = robust_rescale(magnitude, 1.0, 99.5, mask)
    if mask is not None:
        edge = edge.copy()
        edge[~mask] = 0
    return response, edge


def psf_visualisation(psf: np.ndarray) -> np.ndarray:
    """Visualise PSF magnitude as an 8-bit image."""
    return robust_rescale(np.abs(psf), 0.0, 99.8)
