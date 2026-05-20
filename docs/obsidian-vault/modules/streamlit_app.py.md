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
- Three input tabs:
  - **Upload** — `st.file_uploader` for PNG / JPG
  - **Draw** — self-contained HTML5 canvas component at `app/components/handwriting_canvas/` (custom Streamlit component declared via `components.v1.declare_component`); sends strokes as a `data:image/png;base64,...` URL that flows through `src.loaders.load_image_from_url`
  - **URL / paste** — `streamlit_paste_button.paste_image_button` for clipboard, plus a text input that accepts http(s) URLs and `data:image/...` URIs (routed through `src.loaders.load_image_from_url`)
- Renders [[PredictionResult]]s with [[Confidence Badge|confidence badges]] and per-line breakdown.
- Draws [[Segmentation Overlay|segmentation overlays]] on detected lines (red) and words (green).

## Calls into

- [[recognizer.py]] → `Recognizer` (TrOCR)
- [[mltu_recognizer.py]] → `MltuRecognizer` (CRNN/ONNX)
- [[segment.py]] → `line_polygons`, `word_polygons`
- [[preprocess.py]] → `preprocess`, `PreprocessOptions`
- [[loaders.py]] → `load_image_from_url`

## Exposes

- `run_pipeline()` — single call used by the upload pane
- `draw_line_overlay()` — renders polygons on the PIL image
- `get_recognizer(choice)` — caches the chosen backend via `st.cache_resource`

## Related

- [[Streamlit UI]] (concept note)
- [[Inference Pipeline]]
