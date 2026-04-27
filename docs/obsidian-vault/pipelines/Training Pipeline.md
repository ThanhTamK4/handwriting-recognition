---
type: pipeline
tags: [type/pipeline, stage/training]
aliases: [Training Pipeline]
---

# Training Pipeline

How the [[CRNN]] goes from raw data to a deployable ONNX model.

```
[[IAM Words Dataset]]   (data/IAM_Words/words.txt + words/*.png)
          ↓
  load_samples()  —  filter err rows, missing/empty PNGs
          ↓
  DataProvider  +  ImageReader (CVImage)
          ↓
  Transformers: ImageResizer → LabelIndexer → LabelPadding
          ↓
  split 90/10 → train_provider / val_provider
          ↓
  train_provider.augmentors = [[Data Augmentation]]
          ↓
  KerasSequenceProvider wrapper  ([[Keras3 Compatibility]])
          ↓
  [[CRNN]] compiled with [[CTC Loss]], CER/WER metrics
          ↓
  model.fit(epochs=50, callbacks=[
      EarlyStopping(val_CER patience=10),
      ModelCheckpoint best val_CER,
      ReduceLROnPlateau,
      CSVLogger → training_log.csv,
      TensorBoard,
      Model2onnx  ←  auto-exports on best checkpoint
  ])
          ↓
  Outputs:
    model.h5         — best TF checkpoint
    model.onnx       — deployable ([[ONNX Runtime]])
    configs.yaml     — vocab + dims
    training_log.csv — plotted in notebook phase 10
```

## Execution modes

- **Local CPU** — [[train_mltu.py]] in `.venv-train` (~17 h for 50 epochs)
- **Colab GPU** — [[train_mltu_colab.ipynb]] on T4, data staged to local SSD ([[Drive I O Bottleneck]]), ~30 min

## Related

- [[Inference Pipeline]] — consumes model.onnx + configs.yaml
