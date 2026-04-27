---
type: bug
tags: [type/bug, fix/confidence]
aliases: [Double Softmax Bug]
status: fixed
---

# Double Softmax Bug

## Symptom

Every [[mltu_recognizer.py]] prediction reported `conf = 0.03` regardless of correctness. TrOCR showed varied confidences (0.50–1.00), mltu did not.

## Cause

The [[CRNN]]'s output `Dense(output_dim + 1, activation='softmax')` layer already normalizes logits. Inside `_ctc_greedy_decode`, code called `_softmax(logits)` **again**, squashing probabilities toward uniform (`~1/num_classes ≈ 0.03`).

## Fix (src/mltu_recognizer.py)

Auto-detect whether rows already sum to ~1:

```python
row_sums = logits.sum(axis=-1)
probs = logits if np.allclose(row_sums, 1.0, atol=0.1) else _softmax(logits, axis=-1)
```

## Verification

After fix, `python -m src.eval_iam --n 5 --model mltu` → confidences of 0.86–1.00.

## Related

- [[mltu_recognizer.py]], [[CTC Loss]], [[CRNN]]
