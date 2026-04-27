"""Line and word segmentation using connected-components + adaptive clustering.

    split_lines(image) -> list[line_crop_bgr]
    split_words(line_bgr) -> list[word_crop_bgr]
    split_lines_words(image) -> list[list[word_crop_bgr]]
    line_polygons(image) -> list[4x2 int polygons]  # for UI overlay
    word_polygons(image) -> list[list[4x2 int polygons]]  # per-line word boxes

The line splitter clusters connected components by centroid Y (adaptive to line
height) and deskews each line locally, so it handles slanted baselines and
variable line spacing better than a horizontal projection profile.
"""
from __future__ import annotations

import math
from typing import List, Tuple

import cv2
import numpy as np
from PIL import Image

# --- tuning knobs ---
WORD_DILATE_KERNEL = (15, 5)        # horizontal dilation to merge chars within a word
LINE_DILATE_KERNEL = (25, 3)        # stronger horizontal dilation to group chars into line-level blobs
MIN_COMPONENT_AREA = 30             # drop noise specks smaller than this
LINE_CLUSTER_RATIO = 0.7            # join components within this * median_height of the line's running centroid
MIN_LINE_HEIGHT = 10
LINE_PAD = 6
WORD_PAD = 4
MIN_WORD_WIDTH = 8
MIN_WORD_HEIGHT = 8
DESKEW_THRESHOLD_DEG = 2.0          # only rotate lines tilted by more than this


# ============================================================
# Core primitives
# ============================================================

def _to_gray(image) -> np.ndarray:
    if isinstance(image, Image.Image):
        arr = np.array(image.convert("RGB"))
    else:
        arr = image
    if arr.ndim == 3:
        return cv2.cvtColor(arr, cv2.COLOR_RGB2GRAY)
    return arr


def _binarize(gray: np.ndarray) -> np.ndarray:
    _, bw = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)
    return bw


def _cluster_components_to_lines(
    stats: np.ndarray, centroids: np.ndarray
) -> List[List[int]]:
    """Group component indices into lines by centroid Y.

    Components are sorted top-to-bottom, then greedily clustered: a component
    joins the current line if its centroid Y is within LINE_CLUSTER_RATIO *
    median_height of the running line centroid Y. Otherwise a new line begins.
    """
    # Skip background (label 0).
    valid_idx = [
        i for i in range(1, stats.shape[0])
        if stats[i, cv2.CC_STAT_AREA] >= MIN_COMPONENT_AREA
        and stats[i, cv2.CC_STAT_HEIGHT] >= MIN_WORD_HEIGHT
    ]
    if not valid_idx:
        return []

    heights = np.array([stats[i, cv2.CC_STAT_HEIGHT] for i in valid_idx])
    median_h = max(float(np.median(heights)), float(MIN_LINE_HEIGHT))
    gap_thresh = LINE_CLUSTER_RATIO * median_h

    # Sort by centroid Y ascending.
    valid_idx.sort(key=lambda i: centroids[i][1])

    lines: List[List[int]] = []
    current: List[int] = []
    running_cy = None
    for i in valid_idx:
        cy = centroids[i][1]
        if running_cy is None or abs(cy - running_cy) <= gap_thresh:
            current.append(i)
            # Update running centroid as a moving average weighted by count.
            running_cy = (
                cy if running_cy is None
                else (running_cy * (len(current) - 1) + cy) / len(current)
            )
        else:
            lines.append(current)
            current = [i]
            running_cy = cy
    if current:
        lines.append(current)
    return lines


def _line_bbox(stats: np.ndarray, indices: List[int], shape) -> Tuple[int, int, int, int]:
    H, W = shape[:2]
    x0 = min(stats[i, cv2.CC_STAT_LEFT] for i in indices)
    y0 = min(stats[i, cv2.CC_STAT_TOP] for i in indices)
    x1 = max(stats[i, cv2.CC_STAT_LEFT] + stats[i, cv2.CC_STAT_WIDTH] for i in indices)
    y1 = max(stats[i, cv2.CC_STAT_TOP] + stats[i, cv2.CC_STAT_HEIGHT] for i in indices)
    x0 = max(0, int(x0) - LINE_PAD)
    y0 = max(0, int(y0) - LINE_PAD)
    x1 = min(W, int(x1) + LINE_PAD)
    y1 = min(H, int(y1) + LINE_PAD)
    return x0, y0, x1, y1


def _line_tilt_deg(stats: np.ndarray, centroids: np.ndarray, indices: List[int]) -> float:
    """Estimate line tilt in degrees from component centroids (zero if too few)."""
    if len(indices) < 3:
        return 0.0
    xs = np.array([centroids[i][0] for i in indices])
    ys = np.array([centroids[i][1] for i in indices])
    if xs.max() - xs.min() < 5:
        return 0.0
    slope, _ = np.polyfit(xs, ys, 1)
    return math.degrees(math.atan(slope))


def _rotate_crop(crop: np.ndarray, angle_deg: float) -> np.ndarray:
    if abs(angle_deg) < DESKEW_THRESHOLD_DEG:
        return crop
    h, w = crop.shape[:2]
    M = cv2.getRotationMatrix2D((w / 2.0, h / 2.0), angle_deg, 1.0)
    return cv2.warpAffine(
        crop, M, (w, h),
        flags=cv2.INTER_LINEAR,
        borderMode=cv2.BORDER_REPLICATE,
    )


def _rect_to_polygon(x0: int, y0: int, x1: int, y1: int) -> np.ndarray:
    return np.array([[x0, y0], [x1, y0], [x1, y1], [x0, y1]], dtype=np.int32)


# ============================================================
# Public API
# ============================================================

def _line_clusters(gray: np.ndarray):
    """Shared line-clustering pass. Returns (stats, centroids, clusters)."""
    bw = _binarize(gray)
    kernel = cv2.getStructuringElement(cv2.MORPH_RECT, LINE_DILATE_KERNEL)
    dilated = cv2.dilate(bw, kernel, iterations=1)
    num, _, stats, centroids = cv2.connectedComponentsWithStats(dilated, connectivity=8)
    clusters = _cluster_components_to_lines(stats, centroids)
    return stats, centroids, clusters


def split_lines(image) -> List[np.ndarray]:
    """Return deskewed grayscale line crops, top-to-bottom."""
    gray = _to_gray(image)
    stats, centroids, clusters = _line_clusters(gray)
    if not clusters:
        return [gray]

    line_records = []
    for indices in clusters:
        x0, y0, x1, y1 = _line_bbox(stats, indices, gray.shape)
        if y1 - y0 < MIN_LINE_HEIGHT:
            continue
        tilt = _line_tilt_deg(stats, centroids, indices)
        crop = gray[y0:y1, x0:x1]
        crop = _rotate_crop(crop, tilt)
        line_records.append((y0, crop))

    if not line_records:
        return [gray]
    line_records.sort(key=lambda t: t[0])
    return [c for _, c in line_records]


def line_polygons(image) -> List[np.ndarray]:
    """Return 4x2 int polygons per detected line in original image coords."""
    gray = _to_gray(image)
    stats, _, clusters = _line_clusters(gray)
    polys: List[Tuple[int, np.ndarray]] = []
    for indices in clusters:
        x0, y0, x1, y1 = _line_bbox(stats, indices, gray.shape)
        if y1 - y0 < MIN_LINE_HEIGHT:
            continue
        polys.append((y0, _rect_to_polygon(x0, y0, x1, y1)))
    polys.sort(key=lambda t: t[0])
    return [p for _, p in polys]


def split_words(line_gray: np.ndarray) -> List[np.ndarray]:
    """Dilate horizontally + contour-find to split a line into word crops."""
    bw = _binarize(line_gray)
    kernel = cv2.getStructuringElement(cv2.MORPH_RECT, WORD_DILATE_KERNEL)
    dilated = cv2.dilate(bw, kernel, iterations=1)
    contours, _ = cv2.findContours(dilated, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    boxes = []
    for c in contours:
        x, y, w, h = cv2.boundingRect(c)
        if w >= MIN_WORD_WIDTH and h >= MIN_WORD_HEIGHT:
            boxes.append((x, y, w, h))
    boxes.sort(key=lambda b: b[0])

    H, W = line_gray.shape
    crops: List[np.ndarray] = []
    for x, y, w, h in boxes:
        x0 = max(0, x - WORD_PAD)
        y0 = max(0, y - WORD_PAD)
        x1 = min(W, x + w + WORD_PAD)
        y1 = min(H, y + h + WORD_PAD)
        crops.append(line_gray[y0:y1, x0:x1])
    if not crops:
        crops = [line_gray]
    return crops


def word_polygons(image) -> List[List[np.ndarray]]:
    """Return per-line word polygons in original image coordinates."""
    gray = _to_gray(image)
    stats, centroids, clusters = _line_clusters(gray)
    all_polys: List[List[np.ndarray]] = []
    for indices in clusters:
        x0, y0, x1, y1 = _line_bbox(stats, indices, gray.shape)
        if y1 - y0 < MIN_LINE_HEIGHT:
            continue
        tilt = _line_tilt_deg(stats, centroids, indices)
        crop = gray[y0:y1, x0:x1]
        crop_rot = _rotate_crop(crop, tilt)
        # Find word boxes inside the (possibly rotated) line crop.
        bw = _binarize(crop_rot)
        kernel = cv2.getStructuringElement(cv2.MORPH_RECT, WORD_DILATE_KERNEL)
        dilated = cv2.dilate(bw, kernel, iterations=1)
        contours, _ = cv2.findContours(dilated, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        line_polys: List[np.ndarray] = []
        for c in contours:
            wx, wy, ww, wh = cv2.boundingRect(c)
            if ww < MIN_WORD_WIDTH or wh < MIN_WORD_HEIGHT:
                continue
            # Map word box from (possibly rotated) line-crop coords back to original image.
            # For simplicity we approximate by unrotating the box center around the crop
            # center, then translating by (x0, y0).
            cx = wx + ww / 2.0
            cy = wy + wh / 2.0
            ch, cw = crop_rot.shape[:2]
            if abs(tilt) >= DESKEW_THRESHOLD_DEG:
                theta = -math.radians(tilt)
                cx_c = cx - cw / 2.0
                cy_c = cy - ch / 2.0
                cx2 = cx_c * math.cos(theta) - cy_c * math.sin(theta) + cw / 2.0
                cy2 = cx_c * math.sin(theta) + cy_c * math.cos(theta) + ch / 2.0
                cx, cy = cx2, cy2
            gx0 = int(x0 + cx - ww / 2.0 - WORD_PAD)
            gy0 = int(y0 + cy - wh / 2.0 - WORD_PAD)
            gx1 = int(x0 + cx + ww / 2.0 + WORD_PAD)
            gy1 = int(y0 + cy + wh / 2.0 + WORD_PAD)
            line_polys.append(_rect_to_polygon(gx0, gy0, gx1, gy1))
        line_polys.sort(key=lambda p: p[:, 0].min())
        all_polys.append(line_polys)
    return all_polys


def split_lines_words(image) -> List[List[np.ndarray]]:
    """Full page -> list of lines, each a list of word crops (grayscale)."""
    return [split_words(line) for line in split_lines(image)]
