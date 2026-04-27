---
type: module
tags: [type/module, layer/ui, lang/python]
aliases: [Streamlit app, app/streamlit_app.py]
path: app/streamlit_app.py
---

# streamlit_app.py

The single-page Streamlit UI. Entry point users run with `streamlit run app/streamlit_app.py`.

## Responsibilities

- Sidebar: select between [[TrOCR Recognizer]] and [[mltu CRNN Recognizer]], toggle multi-line, toggle [[preprocess.py|preprocessing]] options.
- Two input tabs: file upload + `st.camera_input` webcam.
- Renders [[PredictionResult]]s with [[Confidence Badge|confidence badges]] and per-line breakdown.
- Draws [[Segmentation Overlay|segmentation overlays]] on detected lines (red) and words (green).

## Calls into

- [[recognizer.py]] → `Recognizer` (TrOCR)
- [[mltu_recognizer.py]] → `MltuRecognizer` (CRNN/ONNX)
- [[segment.py]] → `line_polygons`, `word_polygons`
- [[preprocess.py]] → `preprocess`, `PreprocessOptions`

## Exposes

- `run_pipeline()` — single call used by both upload and webcam tabs
- `draw_line_overlay()` — renders polygons on the PIL image
- `get_recognizer(choice)` — caches the chosen backend via `st.cache_resource`

## Related

- [[Streamlit UI]] (concept note)
- [[Inference Pipeline]]
