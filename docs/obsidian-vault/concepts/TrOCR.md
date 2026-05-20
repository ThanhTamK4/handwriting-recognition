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
- Available as **two sibling checkpoints** that share the architecture:
  - `microsoft/trocr-base-handwritten` — cursive/handwritten
  - `microsoft/trocr-base-printed` — printed text / scanned documents
  Both load through the same `Recognizer(model_name=...)` constructor in [[recognizer.py]].

## Strengths

- Handles diverse handwriting styles out of the box
- Contextual — leverages neighbouring characters in a line

## Weaknesses observed

- Case-sensitivity mismatches (`One → one`, `This → this`) — hurt exact-match eval
- Degrades on isolated word crops (domain mismatch)

See [[TrOCR vs mltu]] for numbers.

## Used by

- [[recognizer.py]] → [[Inference Pipeline]]
- Streamlit exposes **two TrOCR options** in the model dropdown:
  - *TrOCR (base handwritten)* — default, for cursive
  - *TrOCR (printed)* — for scanned documents

  The English-dictionary correction checkbox auto-disables for both TrOCR
  variants since their internal LM decoder handles language priors.

## Related

- [[CRNN]] — lightweight alternative
- [[PredictionResult]]
