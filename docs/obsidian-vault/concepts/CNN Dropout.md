---
type: concept
tags: [type/concept, ml/regularization]
aliases: [CNN Dropout, Escalating Dropout]
---

# CNN Dropout

Escalating dropout rates between residual-block pairs in the [[CRNN]]'s convolutional stage.

## Schedule

| Position | Rate |
|---|---|
| After first ResBlock pair (16 filters) | 0.10 |
| After second pair (32 filters, stride=2) | 0.15 |
| After third pair (64 filters, stride=2) | 0.20 |
| After BiLSTM | 0.25 |

## Rationale

Rates escalate as feature maps become more abstract and more prone to overfit. Lightweight — no extra parameters, only inference-time no-op.

## Complements

- [[Data Augmentation]] on the input side
- Early-stopping + ReduceLROnPlateau on the training side

## Related

- [[CRNN]], [[train_mltu_colab.ipynb]]
