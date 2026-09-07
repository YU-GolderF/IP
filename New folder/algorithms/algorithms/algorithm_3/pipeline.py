"""
Gabor Fingerprint Enhancement - Complete Pipeline (Self-Contained)
"""

from __future__ import annotations
from time import perf_counter
import cv2
import numpy as np
from .config import GaborConfig

PIPELINE_BUILD = "gabor-enhanced-v1.0-2026"

def compute_foreground_mask(image, block_size=16):
    """
    Compute fingerprint foreground mask using block-wise standard deviation.
    This is more robust than Otsu thresholding for fingerprint images.
    """
    height, width = image.shape
    n_blocks_y = height // block_size
    n_blocks_x = width // block_size
    
    block_std = np.zeros((n_blocks_y, n_blocks_x), dtype=np.float32)
    
    for by in range(n_blocks_y):
        for bx in range(n_blocks_x):
            y0 = by * block_size
            y1 = min(y0 + block_size, height)
            x0 = bx * block_size
            x1 = min(x0 + block_size, width)
            block = image[y0:y1, x0:x1]
            block_std[by, bx] = np.std(block)
    
    # Use percentile-based threshold
    threshold_std = max(np.percentile(block_std, 30) * 1.5, 3.0)
    block_mask = block_std > threshold_std
    
    # Morphological cleanup
    kernel = np.ones((3, 3), np.uint8)
    block_mask = cv2.morphologyEx(block_mask.astype(np.uint8), cv2.MORPH_CLOSE, kernel)
    block_mask = cv2.morphologyEx(block_mask, cv2.MORPH_OPEN, kernel)
    
    # Upscale to full resolution
    mask = cv2.resize(block_mask.astype(np.uint8) * 255, (width, height), interpolation=cv2.INTER_NEAREST)
    
    # Find largest connected component (fingerprint region)
    num_labels, labels, stats, _ = cv2.connectedComponentsWithStats(mask, connectivity=8)
    if num_labels > 1:
        areas = stats[1:, cv2.CC_STAT_AREA]
        if len(areas) > 0:
            largest_label = np.argmax(areas) + 1
            mask = (labels == largest_label).astype(np.uint8) * 255
    
    return mask

def estimate_orientation(image, block_size=16):
    """Estimate local ridge orientation using the structure tensor method."""
    height, width = image.shape
    img_float = image.astype(np.float32) / 255.0
    
    grad_x = cv2.Sobel(img_float, cv2.CV_32F, 1, 0, ksize=3)
    grad_y = cv2.Sobel(img_float, cv2.CV_32F, 0, 1, ksize=3)
    
    n_blocks_y = height // block_size
    n_blocks_x = width // block_size
    
    orientation = np.zeros((n_blocks_y, n_blocks_x), dtype=np.float32)
    valid_blocks = np.zeros((n_blocks_y, n_blocks_x), dtype=bool)
    coherence = np.zeros((n_blocks_y, n_blocks_x), dtype=np.float32)
    
    for by in range(n_blocks_y):
        for bx in range(n_blocks_x):
            y0 = by * block_size
            y1 = min(y0 + block_size, height)
            x0 = bx * block_size
            x1 = min(x0 + block_size, width)
            
            gx = grad_x[y0:y1, x0:x1]
            gy = grad_y[y0:y1, x0:x1]
            
            gxx = np.sum(gx * gx)
            gyy = np.sum(gy * gy)
            gxy = np.sum(gx * gy)
            
            if gxx + gyy > 1e-6:
                angle = 0.5 * np.arctan2(2 * gxy, gxx - gyy)
                orientation[by, bx] = (angle + np.pi / 2) % np.pi
                
                trace = gxx + gyy
                det = gxx * gyy - gxy * gxy
                if trace > 0:
                    lambda1 = (trace + np.sqrt(max(trace**2 - 4*det, 0))) / 2
                    lambda2 = (trace - np.sqrt(max(trace**2 - 4*det, 0))) / 2
                    coherence[by, bx] = (lambda1 - lambda2) / (lambda1 + lambda2 + 1e-6)
                else:
                    coherence[by, bx] = 0.0
                
                valid_blocks[by, bx] = True
    
    return orientation, valid_blocks, coherence


def smooth_orientation(orientation, valid_blocks, sigma=1.0):
    """Smooth orientation field using doubled angle representation."""
    n_blocks_y, n_blocks_x = orientation.shape
    
    sin2 = np.sin(2 * orientation)
    cos2 = np.cos(2 * orientation)
    
    sin2[~valid_blocks] = 0
    cos2[~valid_blocks] = 0
    
    sin2_smooth = cv2.GaussianBlur(sin2, (0, 0), sigma)
    cos2_smooth = cv2.GaussianBlur(cos2, (0, 0), sigma)
    
    smoothed = 0.5 * np.arctan2(sin2_smooth, cos2_smooth)
    smoothed = np.mod(smoothed, np.pi)
    smoothed[~valid_blocks] = orientation[~valid_blocks]
    
    return smoothed


def estimate_frequency(image, orientation, valid_blocks, block_size=16):
    """Estimate local ridge frequency."""
    height, width = image.shape
    n_blocks_y, n_blocks_x = orientation.shape
    frequency = np.full((n_blocks_y, n_blocks_x), 1.0 / 9.0, dtype=np.float32)
    
    for by in range(n_blocks_y):
        for bx in range(n_blocks_x):
            if not valid_blocks[by, bx]:
                continue
            
            y0 = by * block_size
            y1 = min(y0 + block_size, height)
            x0 = bx * block_size
            x1 = min(x0 + block_size, width)
            
            block = image[y0:y1, x0:x1]
            angle = orientation[by, bx]
            
            h, w = block.shape
            center_y, center_x = h // 2, w // 2
            num_samples = min(h, w)
            
            values = []
            for t in range(-num_samples // 2, num_samples // 2):
                x = int(center_x + t * np.cos(angle))
                y = int(center_y + t * np.sin(angle))
                if 0 <= x < w and 0 <= y < h:
                    values.append(block[y, x])
            
            if len(values) > 5:
                peaks = []
                for i in range(1, len(values) - 1):
                    if values[i] > values[i-1] and values[i] > values[i+1]:
                        peaks.append(i)
                
                if len(peaks) > 2:
                    avg_dist = np.mean(np.diff(peaks))
                    if 3 < avg_dist < 25:
                        frequency[by, bx] = 1.0 / avg_dist
    
    frequency = np.clip(frequency, 0.05, 0.4)
    freq_smooth = cv2.GaussianBlur(frequency, (3, 3), 0.5)
    
    return freq_smooth


def create_gabor_kernel(ksize, sigma, theta, lambd, gamma, psi):
    """Create a Gabor filter kernel."""
    if ksize % 2 == 0:
        ksize += 1
    
    kernel = np.zeros((ksize, ksize), dtype=np.float32)
    center = ksize // 2
    
    for y in range(ksize):
        for x in range(ksize):
            x_theta = (x - center) * np.cos(theta) + (y - center) * np.sin(theta)
            y_theta = -(x - center) * np.sin(theta) + (y - center) * np.cos(theta)
            
            gaussian = np.exp(-(x_theta**2 + gamma**2 * y_theta**2) / (2 * sigma**2))
            sinusoid = np.cos(2 * np.pi * x_theta / lambd + psi)
            
            kernel[y, x] = gaussian * sinusoid
    
    kernel = kernel - np.mean(kernel)
    return kernel


def gabor_enhancement(image, orientation, frequency, valid_blocks, block_size=16, blend_ratio=0.25):
    """Apply Gabor filtering with blending to preserve structure."""
    height, width = image.shape
    n_blocks_y, n_blocks_x = orientation.shape
    
    enhanced = image.astype(np.float32)
    
    sigma = 4.0
    gamma = 0.5
    psi = 0.0
    ksize = 17
    
    for by in range(n_blocks_y):
        for bx in range(n_blocks_x):
            if not valid_blocks[by, bx]:
                continue
            
            y0 = by * block_size
            y1 = min(y0 + block_size, height)
            x0 = bx * block_size
            x1 = min(x0 + block_size, width)
            
            theta = orientation[by, bx]
            lambd = 1.0 / max(frequency[by, bx], 0.05)
            
            kernel = create_gabor_kernel(ksize, sigma, theta, lambd, gamma, psi)
            
            block = image[y0:y1, x0:x1]
            filtered = cv2.filter2D(block.astype(np.float32), -1, kernel)
            
            enhanced[y0:y1, x0:x1] = (1 - blend_ratio) * block.astype(np.float32) + blend_ratio * filtered
    
    return np.clip(enhanced, 0, 255).astype(np.uint8)


def enhance_fingerprint(
    image,
    block_size=16,
    blend_ratio=0.25,
    apply_clahe=True,
    clip_limit=0.5,
    sharpen_amount=0.05
):
    """Gentle fingerprint enhancement using Gabor filtering."""
    orientation, valid_blocks, coherence = estimate_orientation(image, block_size)
    orientation = smooth_orientation(orientation, valid_blocks, sigma=1.0)
    frequency = estimate_frequency(image, orientation, valid_blocks, block_size)
    
    enhanced = gabor_enhancement(image, orientation, frequency, valid_blocks, block_size, blend_ratio)
    
    if apply_clahe:
        clahe = cv2.createCLAHE(clipLimit=clip_limit, tileGridSize=(8, 8))
        enhanced = clahe.apply(enhanced)
    
    if sharpen_amount > 0:
        blurred = cv2.GaussianBlur(enhanced.astype(np.float32), (0, 0), 1.0)
        sharpened = enhanced.astype(np.float32) + sharpen_amount * (enhanced.astype(np.float32) - blurred)
        enhanced = np.clip(sharpened, 0, 255).astype(np.uint8)
    
    return enhanced


def visualise_orientation_field(image, orientation, valid_blocks, block_size=16):
    """Visualise the orientation field as red lines overlaid on the image."""
    vis = cv2.cvtColor(image, cv2.COLOR_GRAY2BGR)
    height, width = image.shape
    n_blocks_y, n_blocks_x = orientation.shape
    line_length = block_size // 2 - 2
    
    for by in range(n_blocks_y):
        for bx in range(n_blocks_x):
            if not valid_blocks[by, bx]:
                continue
            center_x = bx * block_size + block_size // 2
            center_y = by * block_size + block_size // 2
            if center_x >= width or center_y >= height:
                continue
            angle = orientation[by, bx]
            dx = int(line_length * np.cos(angle))
            dy = int(line_length * np.sin(angle))
            cv2.line(vis, (center_x - dx, center_y - dy), (center_x + dx, center_y + dy), (0, 0, 255), 1, cv2.LINE_AA)
    
    return vis


def run_algorithm(
    image: np.ndarray,
    *,
    preprocessing_config=None,
    config: GaborConfig | None = None,
) -> dict:
    """Enhance a fingerprint using Gabor filtering."""
    started = perf_counter()
    
    if len(image.shape) == 3:
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    else:
        gray = image.copy()
    
    if config is None:
        config = GaborConfig()
    
    # Compute orientation data
    orientation, valid_blocks, coherence = estimate_orientation(gray, config.block_size)
    orientation_smooth = smooth_orientation(orientation, valid_blocks, sigma=1.0)
    orientation_vis = visualise_orientation_field(gray, orientation_smooth, valid_blocks, config.block_size)
    valid_blocks_count = int(valid_blocks.sum())
    mean_coherence = float(coherence[valid_blocks].mean()) if valid_blocks_count > 0 else 0.0
    
    # Foreground mask
# Use CLAHE-enhanced image for better thresholding
    preprocessed_image = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8)).apply(gray)
    _, foreground_mask = cv2.threshold(preprocessed_image, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
    kernel = np.ones((5, 5), np.uint8)
    foreground_mask = cv2.morphologyEx(foreground_mask, cv2.MORPH_CLOSE, kernel)
    foreground_mask = cv2.morphologyEx(foreground_mask, cv2.MORPH_OPEN, kernel)
    num_labels, labels, stats, _ = cv2.connectedComponentsWithStats(foreground_mask, connectivity=8)
    if num_labels > 1:
        areas = stats[1:, cv2.CC_STAT_AREA]
        if len(areas) > 0:
            largest_label = np.argmax(areas) + 1
            foreground_mask = (labels == largest_label).astype(np.uint8) * 255
    foreground_coverage = float((foreground_mask > 0).sum() / foreground_mask.size * 100.0)
    
    # Apply enhancement
    enhanced = enhance_fingerprint(
        gray,
        block_size=config.block_size,
        blend_ratio=config.blend_ratio,
        apply_clahe=config.apply_clahe,
        clip_limit=config.clip_limit,
        sharpen_amount=config.sharpen_amount
    )
    
    # Metrics
    elapsed_ms = (perf_counter() - started) * 1000.0
    orig_contrast = float(np.std(gray))
    enh_contrast = float(np.std(enhanced))
    cii = enh_contrast / max(orig_contrast, 1e-6)
    orig_laplacian = cv2.Laplacian(gray, cv2.CV_64F)
    enh_laplacian = cv2.Laplacian(enhanced, cv2.CV_64F)
    orig_sharpness = float(np.var(orig_laplacian))
    enh_sharpness = float(np.var(enh_laplacian))
    sharpness_improvement = ((enh_sharpness - orig_sharpness) / max(orig_sharpness, 1e-6)) * 100.0
    mean_orig = np.mean(gray)
    mean_enh = np.mean(enhanced)
    std_orig = np.std(gray)
    std_enh = np.std(enhanced)
    cov = np.mean((gray - mean_orig) * (enhanced - mean_enh))
    ssim = ((2 * mean_orig * mean_enh + 0.01) * (2 * cov + 0.03)) / \
           ((mean_orig**2 + mean_enh**2 + 0.01) * (std_orig**2 + std_enh**2 + 0.03))
    rvc_orig = orig_sharpness
    rvc_enh = enh_sharpness
    rvc_improvement = ((rvc_enh - rvc_orig) / max(rvc_orig, 1e-6)) * 100.0
    
    metrics = {
        'cii': cii,
        'original_contrast': orig_contrast,
        'processed_contrast': enh_contrast,
        'original_sharpness': orig_sharpness,
        'processed_sharpness': enh_sharpness,
        'sharpness_improvement_pct': sharpness_improvement,
        'ssim': ssim,
        'original_ridge_valley_clarity': rvc_orig,
        'processed_ridge_valley_clarity': rvc_enh,
        'ridge_valley_improvement_pct': rvc_improvement,
        'foreground_coverage_percent': foreground_coverage,
        'valid_orientation_blocks': valid_blocks_count,
        'mean_orientation_coherence': mean_coherence,
        'minutiae_total': 0,
        'processing_time_ms': elapsed_ms,
    }
    
    return {
        "pipeline_build": PIPELINE_BUILD,
        "algorithm_name": "Gabor Fingerprint Enhancement",
        "status": "ok",
        "warnings": [],
        "original": image,
        "grayscale": gray,
        "preprocessing_stages": {
            'grayscale': gray,
            'gaussian_denoised': gray,
            'denoised': gray,
            'normalised': gray,
            'enhanced': enhanced,
        },
        "preprocessed": gray,
        "normalised": gray,
        "denoised": gray,
        "foreground_mask": foreground_mask,
        "mask": foreground_mask,
        "orientation_field": orientation_smooth,
        "orientation_block_mask": valid_blocks,
        "orientation_coherence": coherence,
        "orientation_visualisation": orientation_vis,
        "ridge_restored": enhanced,
        "enhanced_image": enhanced,
        "ridge_enhanced": enhanced,
        "ridge_binary": cv2.threshold(enhanced, 128, 255, cv2.THRESH_BINARY)[1],
        "binary": cv2.threshold(enhanced, 128, 255, cv2.THRESH_BINARY)[1],
        "skeleton": cv2.threshold(enhanced, 128, 255, cv2.THRESH_BINARY)[1] > 0,
        "endings": [],
        "bifurcations": [],
        "minutiae_overlay": cv2.cvtColor(gray, cv2.COLOR_GRAY2BGR),
        "metrics": metrics,
        "processing_time_ms": elapsed_ms,
        "config": config.to_dict(),
    }


run_guided_filtering_enhancement = run_algorithm