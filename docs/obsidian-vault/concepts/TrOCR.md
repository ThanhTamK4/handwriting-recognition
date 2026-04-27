---
type: concept
tags: [type/concept, ml/model, ml/transformer]
aliases: [TrOCR, trocr-base-handwritten]
---

# TrOCR

**Transformer-based OCR** by Microsoft (`microsoft/trocr-base-handwritten`). A pretrained encoder-decoder that combines a ViT image encoder with a RoBERTa text decoder.

## Key facts

- ~334 M parameters, ~1.4 GB weights
- Line-level recognition (expects whole sentences/phrases, not isolated words)
- Internal language model via beam-search decoder — **do not add external NLP post-correction**
- Zero training required for the project

## Strengths

- Handles diverse handwriting styles out of the box
- Contextual — leverages neighbouring characters in a line

## Weaknesses observed

- Case-sensitivity mismatches (`One → one`, `This → this`) — hurt exact-match eval
- Degrades on isolated word crops (domain mismatch)

See [[TrOCR vs mltu]] for numbers.

## Used by

- [[recognizer.py]] → [[Inference Pipeline]]

## Related

- [[CRNN]] — lightweight alternative
- [[PredictionResult]]
