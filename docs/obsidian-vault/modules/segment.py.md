---
type: module
tags: [type/module, layer/segmentation, lang/python]
aliases: [Segmentation, src/segment.py]
path: src/segment.py
---

# segment.py

Line and word segmentation using [[Connected Components Segmentation|connected components + centroid-Y clustering]] with per-line deskew.

Replaces the earlier horizontal-projection-profile algorithm, which failed on slanted baselines and variable line spacing (flagged by the user's professor).

## Public API

- `split_lines(image)` → `list[np.ndarray]` — deskewed grayscale line crops, top-to-bottom
- `split_words(line_gray)` → `list[np.ndarray]` — word crops inside a line
- `split_lines_words(image)` → `list[list[np.ndarray]]` — full-page decomposition
- `line_polygons(image)` → `list[4x2 int polygons]` — for UI overlay (red)
- `word_polygons(image)` → `list[list[4x2 int polygons]]` — for UI overlay (green)

## Algorithm (see [[Connected Components Segmentation]])

1. Otsu binarize → horizontal dilation to merge characters
2. `cv2.connectedComponentsWithStats`
3. Cluster components by centroid Y with threshold `0.7 * median_height`
4. Per-line tilt estimation via `np.polyfit` on centroids → rotate crop if >2°

## Used by

- [[recognizer.py]] — `predict_lines` splits before TrOCR inference
- [[mltu_recognizer.py]] — `predict_lines` feeds per-word CRNN batches
- [[streamlit_app.py]] — draws polygon overlays in multi-line mode

## Related

- [[Connected Components Segmentation]]
- [[Segmentation Overlay]]
