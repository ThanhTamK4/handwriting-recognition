---
type: concept
tags: [type/concept, ml/model, ml/cnn, ml/rnn]
aliases: [CRNN, Convolutional Recurrent Neural Network]
---

# CRNN

**Convolutional Recurrent Neural Network** — the architecture used by the mltu backend:

```
Input (32 × 256 × 3)
  ↓ Rescaling(1/255)
  ↓ ResBlock × 2 (16 filters) → Dropout(0.1)
  ↓ ResBlock(32, stride=2) + ResBlock(32) → Dropout(0.15)
  ↓ ResBlock(64, stride=2) + ResBlock(64) → Dropout(0.2)
  ↓ Permute(2,1,3)   # width becomes time axis
  ↓ Reshape → (64 timesteps, features)
  ↓ BiLSTM(128) → Dropout(0.25)
  ↓ Dense(vocab+1, softmax)   # +1 for CTC blank
```

## Compiled with

- Loss: [[CTC Loss]]
- Metrics: CER, WER (mltu's built-in Keras metrics)
- Optimizer: Adam, lr=1e-3

## Size / speed

- ~850 k parameters (~3.2 MB h5, ~10 MB ONNX)
- ~20 ms/word on CPU via [[ONNX Runtime]]

## Produced by

- [[train_mltu.py]] (local, ~17 h CPU)
- [[train_mltu_colab.ipynb]] (Colab, ~30 min GPU)

## Related

- [[TrOCR]] — heavyweight alternative
- [[CNN Dropout]], [[Data Augmentation]]
- [[Keras3 Compatibility]]
