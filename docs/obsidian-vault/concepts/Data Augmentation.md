---
type: concept
tags: [type/concept, ml/training]
aliases: [Data Augmentation, Augmentation]
---

# Data Augmentation

Training-time image transformations that expand effective dataset size and reduce overfitting.

Applied **to training batches only** (not validation) via `train_provider.augmentors = [...]` in [[train_mltu_colab.ipynb]] / [[train_mltu.py]].

## Augmentors used

From `mltu.augmentors`:

- `RandomBrightness(random_chance=0.3, delta=100)`
- `RandomRotate(random_chance=0.3, angle=5)` — ±5°
- `RandomErodeDilate(random_chance=0.3, kernel_size=(1,1))` — simulates pen-thickness variation
- `RandomGaussianBlur(random_chance=0.2, sigma=1.0)` — camera focus imperfection
- `RandomElasticTransform(random_chance=0.2, ...)` — small non-rigid warp

## Target impact

- Target: val CER ~5–7 % (from ~9.9 % at 10 epochs no aug)
- Combined with [[CNN Dropout]] for regularization

## Related

- [[CNN Dropout]], [[CRNN]], [[Training Pipeline]]
