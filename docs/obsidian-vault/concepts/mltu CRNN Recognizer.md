---
type: concept
tags: [type/concept, ml/backend]
aliases: [mltu CRNN Recognizer, mltu backend, CRNN Recognizer]
---

# mltu CRNN Recognizer

ONNX-based word-level recognition backend, implemented in [[mltu_recognizer.py]]. Wraps the [[CRNN]] trained by [[train_mltu.py]] / [[train_mltu_colab.ipynb]].

## Pipeline

1. `split_lines_words()` from [[segment.py]] → word crops
2. Resize to configs.yaml `height × width`
3. [[ONNX Runtime]] forward pass
4. `_ctc_greedy_decode` → argmax + collapse repeats + drop blank
5. Confidence = mean per-step max prob (after [[Double Softmax Bug]] fix)

## Output

Per-line [[PredictionResult]] with joined words + confidence.

## Related

- [[TrOCR vs mltu]], [[Inference Pipeline]], [[Training Pipeline]]
