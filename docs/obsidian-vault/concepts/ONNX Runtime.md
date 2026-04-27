---
type: concept
tags: [type/concept, ml/runtime]
aliases: [ONNX Runtime, onnxruntime]
---

# ONNX Runtime

Cross-framework model runtime. In this project it enables running the TF-trained [[CRNN]] **without pulling TensorFlow into the main venv**, sidestepping the protobuf version conflict with `streamlit` / `transformers`.

## Usage in project

- Training (isolated `.venv-train`) → `Model2onnx` callback exports `model.h5` → `model.onnx`
- Inference (main venv) → `onnxruntime.InferenceSession(...)` in [[mltu_recognizer.py]]

## Providers used

- `CPUExecutionProvider` (default, sufficient at ~20 ms/word)

## File artifacts

- `models/mltu/model.onnx` (~10 MB)
- `models/mltu/configs.yaml` — vocab + height/width

## Related

- [[CRNN]], [[mltu_recognizer.py]]
- [[Training Pipeline]]
