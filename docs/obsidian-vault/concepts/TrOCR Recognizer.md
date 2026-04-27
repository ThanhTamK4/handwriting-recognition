---
type: concept
tags: [type/concept, ml/backend]
aliases: [TrOCR Recognizer, TrOCR backend]
---

# TrOCR Recognizer

The torch/transformers-based recognition backend wrapping [[TrOCR]] (`microsoft/trocr-base-handwritten`). Lives in [[recognizer.py]] as `Recognizer` class.

## Methods

- `predict(image)` — single crop → [[PredictionResult]]
- `predict_lines(image)` — splits via [[segment.py]] then recognizes each line

## Trade-offs

See [[TrOCR vs mltu]]. Strong on multi-line/contextual, weak on case-sensitive word-level IAM benchmarks.

## Related

- [[Inference Pipeline]]
