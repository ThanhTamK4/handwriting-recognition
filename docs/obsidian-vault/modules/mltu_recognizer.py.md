---
type: module
tags: [type/module, backend/mltu, lang/python]
aliases: [mltu ONNX backend, src/mltu_recognizer.py]
path: src/mltu_recognizer.py
---

# mltu_recognizer.py

ONNX-runtime inference wrapper for the [[CRNN]] trained via [[train_mltu.py]] or [[train_mltu_colab.ipynb]].

Depends on `onnxruntime`, `numpy`, `opencv`, `pyyaml`, `PIL` — **no TensorFlow** in the main venv (avoids protobuf conflicts).

## Public API

- `MltuRecognizer.predict(image)` — single word crop → [[PredictionResult]]
- `MltuRecognizer.predict_lines(image)` — full page via [[segment.py]] `split_lines_words` → one result per line, joined with spaces and newlines

## Loads

- `models/mltu/model.onnx` — weights
- `models/mltu/configs.yaml` — vocab, HEIGHT, WIDTH

## Key helpers

- `_prepare(image)` — resize to (WIDTH, HEIGHT), float32, no normalization (model has `Rescaling` layer inside)
- `_ctc_greedy_decode(logits)` — argmax, collapse repeats, drop blanks, mean-of-max probabilities for confidence
  - Contains fix for the [[Double Softmax Bug]]

## Related

- [[CRNN]], [[CTC Loss]], [[ONNX Runtime]]
- [[segment.py]], [[PredictionResult]]
- [[train_mltu.py]], [[train_mltu_colab.ipynb]]
