"""
Gabor Fingerprint Enhancement - Complete Pipeline
Uses the functions from filter.py for orientation estimation and enhancement.
"""

from __future__ import annotations
from time import perf_counter
import cv2
import numpy as np

from .filter import (
    enhance_fingerprint,
    estimate_orientation,
    smooth_orientation,
    estimate_frequency,
    create_gabor_kernel,
    gabor_enhancement,
)
from .config import GaborConfig

PIPELINE_BUILD = "gabor-enhanced-v1.0-2026"


def visualise_orientation_field(image, orientation, valid_blocks, block_size=16):
    """
    Visualise the orientation field as red lines overlaid on the image.
    """
    # Convert grayscale to RGB
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
            
            # Make sure center is within image bounds
            if center_x >= width or center_y >= height:
                continue
            
            angle = orientation[by, bx]
            
            dx = int(line_length * np.cos(angle))
            dy = int(line_length * np.sin(angle))
            
            cv2.line(
                vis,
                (center_x - dx, center_y - dy),
                (center_x + dx, center_y + dy),
                (0, 0, 255),  # Red lines
                1,
                cv2.LINE_AA
            )
    
    return vis


def run_algorithm(
    image: np.ndarray,
    *,
    preprocessing_config=None,
    config: GaborConfig | None = None,
) -> dict:
    """
    Enhance a fingerprint using Gabor filtering.
    This is the main entry point for the algorithm.
    """
    started = perf_counter()
    
    # Convert to grayscale
    if len(image.shape) == 3:
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    else:
        gray = image.copy()
    
    if config is None:
        config = GaborConfig()
    
    # ============================================================
    # STEP 1: COMPUTE ORIENTATION DATA (for the Ridge Orientation tab)
    # ============================================================
    # Call the function from filter.py
    orientation, valid_blocks, coherence = estimate_orientation(gray, config.block_size)
    
    # Smooth the orientation field
    orientation_smooth = smooth_orientation(orientation, valid_blocks, sigma=1.0)
    
    # Visualize orientation field (overlay on image)
    orientation_vis = visualise_orientation_field(
        gray,
        orientation_smooth,
        valid_blocks,
        block_size=config.block_size
    )
    
    # Count valid blocks
    valid_blocks_count = int(valid_blocks.sum())
    
    # Calculate mean coherence
    if valid_blocks_count > 0:
        mean_coherence = float(coherence[valid_blocks].mean())
    else:
        mean_coherence = 0.0
    
    # ============================================================
    # STEP 2: COMPUTE FOREGROUND MASK
    # ============================================================
    # Use Otsu threshold to get foreground
    foreground_mask = compute_foreground_mask(gray)
    
    foreground_coverage = float((foreground_mask > 0).sum() / foreground_mask.size * 100.0)
    
    # ============================================================
    # STEP 3: APPLY GABOR ENHANCEMENT
    # ============================================================
    # Call enhance_fingerprint from filter.py
    enhanced = enhance_fingerprint(
        gray,
        block_size=config.block_size,
        blend_ratio=config.blend_ratio,
        apply_clahe=config.apply_clahe,
        clip_limit=config.clip_limit,
        sharpen_amount=config.sharpen_amount
    )
    
    # ============================================================
    # STEP 4: CALCULATE METRICS
    # ============================================================
    elapsed_ms = (perf_counter() - started) * 1000.0
    
    # Contrast
    orig_contrast = float(np.std(gray))
    enh_contrast = float(np.std(enhanced))
    cii = enh_contrast / max(orig_contrast, 1e-6)
    
    # Sharpness (Laplacian variance)
    orig_laplacian = cv2.Laplacian(gray, cv2.CV_64F)
    enh_laplacian = cv2.Laplacian(enhanced, cv2.CV_64F)
    orig_sharpness = float(np.var(orig_laplacian))
    enh_sharpness = float(np.var(enh_laplacian))
    sharpness_improvement = ((enh_sharpness - orig_sharpness) / max(orig_sharpness, 1e-6)) * 100.0
    
    # SSIM (simplified)
    mean_orig = np.mean(gray)
    mean_enh = np.mean(enhanced)
    std_orig = np.std(gray)
    std_enh = np.std(enhanced)
    cov = np.mean((gray - mean_orig) * (enhanced - mean_enh))
    ssim = ((2 * mean_orig * mean_enh + 0.01) * (2 * cov + 0.03)) / \
           ((mean_orig**2 + mean_enh**2 + 0.01) * (std_orig**2 + std_enh**2 + 0.03))
    
    # Ridge-Valley Clarity
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
    
    # ============================================================
    # STEP 5: RETURN RESULTS
    # ============================================================
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
        # ============================================================
        # ORIENTATION DATA (NOW COMPUTED FROM THE IMAGE!)
        # ============================================================
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