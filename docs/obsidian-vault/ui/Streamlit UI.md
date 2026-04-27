---
type: concept
tags: [type/concept, ui/component]
aliases: [Streamlit UI]
---

# Streamlit UI

The interactive frontend in [[streamlit_app.py]]. Only entry point the end-user touches.

## Panels

- **Input** — file uploader (PNG/JPG) + webcam snapshot tab
- **Preprocessing toggles** — perspective, deskew, denoise, CLAHE, binarize (all from [[preprocess.py]])
- **Mode switch** — single word vs multi-line
- **Backend switch** — [[TrOCR Recognizer]] or [[mltu CRNN Recognizer]]
- **Result panel** — rendered by `render_result()`:
  - text in `st.code` (copy button)
  - [[Confidence Badge]] (green/yellow/red)
  - per-line breakdown in multi-line mode
  - [[Segmentation Overlay]] below the image

## Related

- [[Inference Pipeline]]
- [[PredictionResult]]
