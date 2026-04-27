---
type: module
tags: [type/module, layer/preprocessing, lang/python]
aliases: [Preprocessing, src/preprocess.py]
path: src/preprocess.py
---

# preprocess.py

Image cleanup pipeline for webcam / phone-photographed handwriting. Each stage is optional, gated by `PreprocessOptions`.

## Pipeline order

1. **Perspective correction** — Canny + contour search → 4-point warp
2. **Grayscale + denoise** — `cv2.bilateralFilter`
3. **Enhance contrast** — CLAHE
4. **Deskew** — rotation via `cv2.minAreaRect` on foreground pixels
5. **Binarize** — adaptive threshold to B/W

## Public API

- `PreprocessOptions` — dataclass of toggles (perspective, deskew, denoise, enhance_contrast, binarize)
- `preprocess(pil_image, opts) -> PIL.Image`

## Used by

- [[streamlit_app.py]] — sidebar exposes each flag as a checkbox

## Related

- [[Inference Pipeline]]
