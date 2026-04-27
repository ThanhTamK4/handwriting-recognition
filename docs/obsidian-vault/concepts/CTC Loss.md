---
type: concept
tags: [type/concept, ml/loss]
aliases: [CTC, CTC Loss, Connectionist Temporal Classification]
---

# CTC Loss

**Connectionist Temporal Classification** — loss function that lets a network emit variable-length predictions without needing per-timestep alignment labels.

## Core rule

`input_length >= 2 * label_length - 1`

If the image is too narrow relative to the label, CTC throws "Not enough time steps" — see [[CTC Not Enough Time]].

## Decoding

Greedy CTC decode (used in [[mltu_recognizer.py]] `_ctc_greedy_decode`):

1. `argmax` over classes per timestep
2. Collapse consecutive duplicates
3. Drop the blank class (reserved at index `len(vocab)`)

## Used by

- [[CRNN]] — training + inference
- [[train_mltu.py]] / [[train_mltu_colab.ipynb]] — via `mltu.tensorflow.losses.CTCloss`

## Related

- [[Double Softmax Bug]] — the model's final Dense layer already has softmax; decoding must not softmax again
