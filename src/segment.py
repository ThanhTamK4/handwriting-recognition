"""Line and word segmentation for word-level recognizers (e.g. mltu CRNN).

    split_lines(image) -> list[line_crop_bgr]
    split_words(line_bgr) -> list[word_crop_bgr]
    split_lines_words(image) -> list[list[word_crop_bgr]]
"""
from __future__ import annotations

from typing import List

import cv2
import numpy as np
from PIL import Image

LINE_THRESHOLD_RATIO = 0.05
MIN_LINE_HEIGHT = 10
LINE_PAD = 6

WORD_DILATE_KERNEL = (15, 5)
MIN_WORD_WIDTH = 8
MIN_WORD_HEIGHT = 8
WORD_PAD = 4


def _to_gray(image) -> np.ndarray:
    if isinstance(image, Image.Image):
        arr = np.array(image.convert("RGB"))
    else:
        arr = image
    if arr.ndim == 3:
        return cv2.cvtColor(arr, cv2.COLOR_RGB2GRAY)
    return arr


def split_lines(image) -> List[np.ndarray]:
    """Horizontal projection profile to split a page/paragraph into line crops."""
    gray = _to_gray(image)
    _, bw = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)
    proj = bw.sum(axis=1)
    if proj.max() == 0:
        return [gray]
    threshold = proj.max() * LINE_THRESHOLD_RATIO

    lines: List[np.ndarray] = []
    in_line = False
    start = 0
    h = gray.shape[0]
    for y, val in enumerate(proj):
        if val > threshold and not in_line:
            in_line = True
            start = y
        elif val <= threshold and in_line:
            in_line = False
            if y - start > MIN_LINE_HEIGHT:
                lines.append(gray[max(0, start - LINE_PAD) : min(h, y + LINE_PAD), :])
    if in_line and h - start > MIN_LINE_HEIGHT:
        lines.append(gray[max(0, start - LINE_PAD) :, :])
    if not lines:
        lines = [gray]
    return lines


def split_words(line_gray: np.ndarray) -> List[np.ndarray]:
    """Dilate horizontally + contour-find to split a line into word crops."""
    _, bw = cv2.threshold(line_gray, 0, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)
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


def split_lines_words(image) -> List[List[np.ndarray]]:
    """Full page -> list of lines, each a list of word crops (grayscale)."""
    return [split_words(line) for line in split_lines(image)]
