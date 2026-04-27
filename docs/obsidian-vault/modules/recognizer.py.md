---
type: module
tags: [type/module, backend/trocr, lang/python]
aliases: [TrOCR backend, src/recognizer.py]
path: src/recognizer.py
---

# recognizer.py

Wraps Hugging Face's `microsoft/trocr-base-handwritten` model ([[TrOCR]]) and exposes a uniform [[PredictionResult]] interface.

## Public API

- `Recognizer.predict(image)` — single-crop prediction
- `Recognizer.predict_lines(image)` — splits via [[segment.py]], predicts each line, concatenates
- `line_boxes(image)` — backcompat shim returning `(y0, y1)` ranges
- `line_polygons(image)` — delegates to [[segment.py]]'s polygon helper

## Dependencies

- `transformers` — `TrOCRProcessor`, `VisionEncoderDecoderModel`
- `torch`
- [[segment.py]] — for line splitting
- [[PredictionResult]] (defined here)

## Confidence

Confidence = mean probability of chosen tokens over the generation steps (`output.scores` + `output.sequences`).

## Related

- [[TrOCR]] — model details
- [[mltu_recognizer.py]] — alternative backend
- [[Inference Pipeline]]
