---
type: module
tags: [type/module, layer/training, lang/python]
aliases: [Local training script, training/train_mltu.py]
path: training/train_mltu.py
---

# train_mltu.py

Local [[CRNN]] training script. Runs in the **isolated `.venv-train`** with TensorFlow 2.10 and `mltu==1.2.5` (avoids protobuf conflicts with the main venv's `streamlit` / `transformers`).

## Flow

1. Load samples from `data/IAM_Words/words.txt` (filters `err`, missing, empty PNGs)
2. Build vocab + `max_word_length`
3. `DataProvider` with `UnicodeImageReader` (works around OpenCV's non-ASCII path issue on Windows)
4. Wraps into `KerasSequenceProvider` (Keras Sequence subclass)
5. Trains residual-CNN + BiLSTM + [[CTC Loss]]
6. Exports to ONNX via `Model2onnx` callback

## Outputs

- `models/mltu/model.h5`
- `models/mltu/model.onnx`
- `models/mltu/configs.yaml` (written manually — mltu's BaseModelConfigs pickle breaks)

## Known limitations

- CPU-only on most dev machines → ~17 h for 50 epochs
- Use [[train_mltu_colab.ipynb]] for GPU acceleration

## Related

- [[Training Pipeline]]
- [[train_mltu_colab.ipynb]]
- [[Data Augmentation]], [[CNN Dropout]]
