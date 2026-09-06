"""
Classic Gabor Filtering for Fingerprint Enhancement
- Orientation field estimation using structure tensor
- Ridge frequency estimation
- Adaptive Gabor filtering
- Balanced post-processing
"""

import numpy as np
import cv2


def estimate_orientation(image, block_size=16):
    """
    Estimate local ridge orientation using the structure tensor method.
    """
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
                
                # Coherence (how strong the orientation is)
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
    """
    Smooth orientation field using doubled angle representation.
    """
    n_blocks_y, n_blocks_x = orientation.shape
    
    # Convert to doubled angle
    sin2 = np.sin(2 * orientation)
    cos2 = np.cos(2 * orientation)
    
    # Only use valid blocks
    sin2[~valid_blocks] = 0
    cos2[~valid_blocks] = 0
    
    # Smooth
    sin2_smooth = cv2.GaussianBlur(sin2, (0, 0), sigma)
    cos2_smooth = cv2.GaussianBlur(cos2, (0, 0), sigma)
    
    # Recover angle
    smoothed = 0.5 * np.arctan2(sin2_smooth, cos2_smooth)
    smoothed = np.mod(smoothed, np.pi)
    
    # Only keep valid blocks
    smoothed[~valid_blocks] = orientation[~valid_blocks]
    
    return smoothed


def estimate_frequency(image, orientation, valid_blocks, block_size=16):
    """
    Estimate local ridge frequency.
    """
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
            
            # Project along ridge direction
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
                # Find peaks in the projected signal
                peaks = []
                for i in range(1, len(values) - 1):
                    if values[i] > values[i-1] and values[i] > values[i+1]:
                        peaks.append(i)
                
                if len(peaks) > 2:
                    avg_dist = np.mean(np.diff(peaks))
                    if 3 < avg_dist < 25:
                        frequency[by, bx] = 1.0 / avg_dist
    
    # Clamp to reasonable range
    frequency = np.clip(frequency, 0.05, 0.4)
    
    # Smooth frequency field
    freq_smooth = cv2.GaussianBlur(frequency, (3, 3), 0.5)
    
    return freq_smooth


def create_gabor_kernel(ksize, sigma, theta, lambd, gamma, psi):
    """
    Create a Gabor filter kernel.
    """
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


def gabor_enhancement(image, orientation, frequency, valid_blocks, block_size=16, blend_ratio=0.4):
    """
    Apply Gabor filtering with blending to preserve structure.
    """
    height, width = image.shape
    n_blocks_y, n_blocks_x = orientation.shape
    
    enhanced = image.astype(np.float32)
    
    # Gabor parameters
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
            
            # Blend with original to preserve structure
            enhanced[y0:y1, x0:x1] = (1 - blend_ratio) * block.astype(np.float32) + blend_ratio * filtered
    
    return np.clip(enhanced, 0, 255).astype(np.uint8)


def enhance_fingerprint(
    image,
    block_size=16,
    blend_ratio=0.2,
    apply_clahe=True,
    clip_limit=0.5,
    sharpen_amount=0.05
):
    """
    Gentle fingerprint enhancement using Gabor filtering.
    """
    # Step 1: Estimate orientation
    orientation, valid_blocks, coherence = estimate_orientation(image, block_size)
    
    # Step 2: Smooth orientation
    orientation = smooth_orientation(orientation, valid_blocks, sigma=1.0)
    
    # Step 3: Estimate frequency
    frequency = estimate_frequency(image, orientation, valid_blocks, block_size)
    
    # Step 4: Gabor enhancement with GENTLE blending
    enhanced = gabor_enhancement(image, orientation, frequency, valid_blocks, block_size, blend_ratio)
    
    # Step 5: Gentle CLAHE
    if apply_clahe:
        clahe = cv2.createCLAHE(clipLimit=clip_limit, tileGridSize=(8, 8))
        enhanced = clahe.apply(enhanced)
    
    # Step 6: Very gentle sharpening (almost none)
    if sharpen_amount > 0:
        blurred = cv2.GaussianBlur(enhanced.astype(np.float32), (0, 0), 1.0)
        sharpened = enhanced.astype(np.float32) + sharpen_amount * (enhanced.astype(np.float32) - blurred)
        enhanced = np.clip(sharpened, 0, 255).astype(np.uint8)
    
    return enhanced