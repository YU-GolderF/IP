from __future__ import annotations

import cv2
import numpy as np

from .core import robust_rescale


def linear_grayscale_stretch(image: np.ndarray, mask: np.ndarray | None, low: float, high: float) -> np.ndarray:
    """Linear percentile stretch used after the RHLT edge response."""
    return robust_rescale(image.astype(np.float64), low, high, mask)


def rhlt_edge_guided_sharpen(
    denoised: np.ndarray,
    edge_map: np.ndarray,
    mask: np.ndarray,
    gain: float = 0.75,
) -> np.ndarray:
    """
    Optional project extension: use the RHLT edge magnitude to increase local
    ridge/valley separation while preserving the original grey-level structure.

    This is deliberately separated from the baseline RHLT output so experiments can
    report baseline vs. improved RHLT transparently.
    """
    f = denoised.astype(np.float32)
    local = cv2.GaussianBlur(f, (0, 0), sigmaX=2.0, sigmaY=2.0, borderType=cv2.BORDER_REFLECT)
    polarity = np.sign(f - local)
    e = edge_map.astype(np.float32) / 255.0
    enhanced = f + gain * 64.0 * polarity * e
    enhanced = np.clip(enhanced, 0, 255)
    result = enhanced.astype(np.uint8)
    result[~mask] = denoised[~mask]
    return result


def binarise_dark_ridges(gray: np.ndarray, mask: np.ndarray) -> np.ndarray:
    """Otsu binarisation assuming fingerprint ridges are darker than valleys."""
    vals = gray[mask]
    if vals.size < 32:
        vals = gray.ravel()
    # Otsu needs an image; a masked histogram image works reliably here.
    hist = cv2.calcHist([vals.astype(np.uint8).reshape(-1, 1)], [0], None, [256], [0, 256]).ravel()
    total = hist.sum()
    sum_total = np.dot(np.arange(256), hist)
    sum_b = 0.0
    w_b = 0.0
    best_var = -1.0
    threshold = 127
    for t in range(256):
        w_b += hist[t]
        if w_b == 0:
            continue
        w_f = total - w_b
        if w_f == 0:
            break
        sum_b += t * hist[t]
        m_b = sum_b / w_b
        m_f = (sum_total - sum_b) / w_f
        between = w_b * w_f * (m_b - m_f) ** 2
        if between > best_var:
            best_var = between
            threshold = t
    binary = (gray < threshold) & mask
    return binary


def clean_binary(binary: np.ndarray, min_component_area: int = 20) -> np.ndarray:
    """Remove isolated binary speckles before skeletonisation."""
    u8 = binary.astype(np.uint8)
    n, labels, stats, _ = cv2.connectedComponentsWithStats(u8, connectivity=8)
    cleaned = np.zeros_like(u8)
    for label in range(1, n):
        if stats[label, cv2.CC_STAT_AREA] >= max(1, min_component_area):
            cleaned[labels == label] = 255
    # A light close repairs one-pixel gaps without aggressively changing ridge shape.
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (3, 3))
    cleaned = cv2.morphologyEx(cleaned, cv2.MORPH_CLOSE, kernel)
    return cleaned > 0


def make_skeleton(binary: np.ndarray) -> np.ndarray:
    """Return a one-pixel-wide skeleton using vectorised Zhang-Suen thinning."""
    img = binary.astype(np.uint8).copy()
    if img.ndim != 2:
        raise ValueError("binary image must be 2-D")

    def neighbours(a: np.ndarray):
        p2 = np.roll(a, -1, axis=0)
        p3 = np.roll(p2, -1, axis=1)
        p4 = np.roll(a, -1, axis=1)
        p5 = np.roll(np.roll(a, 1, axis=0), -1, axis=1)
        p6 = np.roll(a, 1, axis=0)
        p7 = np.roll(p6, 1, axis=1)
        p8 = np.roll(a, 1, axis=1)
        p9 = np.roll(p2, 1, axis=1)
        return p2, p3, p4, p5, p6, p7, p8, p9

    changed = True
    while changed:
        changed = False
        for step in (0, 1):
            p = neighbours(img)
            b = sum(p)
            seq = p + (p[0],)
            atrans = sum(((seq[i] == 0) & (seq[i + 1] == 1)).astype(np.uint8) for i in range(8))
            if step == 0:
                c1 = (p[0] * p[2] * p[4] == 0)
                c2 = (p[2] * p[4] * p[6] == 0)
            else:
                c1 = (p[0] * p[2] * p[6] == 0)
                c2 = (p[0] * p[4] * p[6] == 0)
            marker = (img == 1) & (b >= 2) & (b <= 6) & (atrans == 1) & c1 & c2
            marker[[0, -1], :] = False
            marker[:, [0, -1]] = False
            if np.any(marker):
                img[marker] = 0
                changed = True
    return img.astype(bool)


def crossing_number_minutiae(
    skeleton: np.ndarray,
    mask: np.ndarray,
    border: int = 12,
    min_distance: int = 8,
) -> tuple[list[tuple[int, int]], list[tuple[int, int]]]:
    """
    Detect ridge endings and bifurcations with the crossing-number method.

    This is intended as a consistent downstream comparison metric. Counts should be
    interpreted together with visual inspection because enhancement artefacts can create
    false minutiae.
    """
    sk = skeleton.astype(np.uint8)
    valid = mask.astype(np.uint8) * 255
    if border > 0:
        k = 2 * border + 1
        valid = cv2.erode(valid, np.ones((k, k), np.uint8), iterations=1)
    valid = valid > 0

    endings: list[tuple[int, int]] = []
    bifurcations: list[tuple[int, int]] = []
    h, w = sk.shape
    for y in range(1, h - 1):
        for x in range(1, w - 1):
            if sk[y, x] == 0 or not valid[y, x]:
                continue
            p = [
                sk[y - 1, x], sk[y - 1, x + 1], sk[y, x + 1], sk[y + 1, x + 1],
                sk[y + 1, x], sk[y + 1, x - 1], sk[y, x - 1], sk[y - 1, x - 1],
            ]
            transitions = sum(abs(int(p[i]) - int(p[(i + 1) % 8])) for i in range(8))
            cn = transitions / 2.0
            if cn == 1:
                endings.append((x, y))
            elif cn == 3:
                bifurcations.append((x, y))

    def suppress(points: list[tuple[int, int]]) -> list[tuple[int, int]]:
        kept: list[tuple[int, int]] = []
        d2 = min_distance * min_distance
        for pt in points:
            if all((pt[0] - q[0]) ** 2 + (pt[1] - q[1]) ** 2 >= d2 for q in kept):
                kept.append(pt)
        return kept

    return suppress(endings), suppress(bifurcations)


def minutiae_overlay(gray: np.ndarray, endings: list[tuple[int, int]], bifurcations: list[tuple[int, int]]) -> np.ndarray:
    """Create a visual overlay: circles for endings and squares for bifurcations."""
    rgb = cv2.cvtColor(gray, cv2.COLOR_GRAY2RGB)
    for x, y in endings:
        cv2.circle(rgb, (x, y), 4, (255, 255, 255), 1, lineType=cv2.LINE_AA)
    for x, y in bifurcations:
        cv2.rectangle(rgb, (x - 4, y - 4), (x + 4, y + 4), (0, 0, 0), 1, lineType=cv2.LINE_AA)
    return rgb
