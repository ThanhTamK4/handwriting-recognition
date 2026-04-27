---
type: concept
tags: [type/concept, cv/segmentation]
aliases: [Connected Components Segmentation, CC segmentation]
---

# Connected Components Segmentation

The algorithm in [[segment.py]]'s `split_lines()`, replacing the earlier horizontal-projection-profile approach.

## Pipeline

1. **Binarize** — Otsu threshold (inverted)
2. **Dilate horizontally** — `(25, 3)` kernel merges chars into line-level blobs
3. **`cv2.connectedComponentsWithStats`** — labels each blob, returns centroid + bbox
4. **Filter noise** — drop components with `area < 30` or `height < 8`
5. **Sort components by centroid-Y**
6. **Greedy cluster** — component joins current line if `|Δcy| <= 0.7 * median_height`; otherwise start new line
7. **Per-line tilt** — `np.polyfit(centroids_x, centroids_y, 1)` → rotation in degrees
8. **Rotate crop** if tilt > 2° via `cv2.warpAffine` with `BORDER_REPLICATE`

## Why it's better than projection profile

- Adaptive — threshold scales with the image's median component height instead of a fixed 5% row-sum
- Handles slanted baselines (up to ~15°)
- Robust to variable inter-line spacing and mixed font sizes
- Works on cursive where projection profile creates false splits

## Used by

- [[segment.py]] — `split_lines`, `line_polygons`, `word_polygons`
- [[recognizer.py]] → `predict_lines` (TrOCR)
- [[mltu_recognizer.py]] → `predict_lines` (CRNN)

## Related

- [[Segmentation Overlay]]
