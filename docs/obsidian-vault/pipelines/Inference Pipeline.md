---
type: pipeline
tags: [type/pipeline, stage/inference]
aliases: [Inference Pipeline, Recognition Pipeline]
---

# Inference Pipeline

End-to-end flow from user input to prediction in [[streamlit_app.py]].

```
[Upload PNG/JPG]   [Webcam snapshot]
         \           /
          ▼         ▼
   PIL.Image.open(bytes)
          ↓
   [[preprocess.py]] (optional: perspective, deskew, denoise, CLAHE, binarize)
          ↓
   Multi-line?
    /        \
  no         yes
   ↓           ↓
 rec.predict  rec.predict_lines
               ↓
           [[segment.py]] — split into lines (+ words for mltu)
               ↓
   Backend selected:
     • [[TrOCR Recognizer]]  (line-level, torch)
     • [[mltu CRNN Recognizer]] (word-level, ONNX)
               ↓
          [[PredictionResult]]
               ↓
  render_result() in [[streamlit_app.py]]
   • text in st.code (copy button)
   • [[Confidence Badge]]
   • per-line breakdown
   • [[Segmentation Overlay]]
```

## Related

- [[Streamlit UI]], [[Training Pipeline]]
