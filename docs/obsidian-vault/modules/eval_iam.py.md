---
type: module
tags: [type/module, layer/eval, lang/python]
aliases: [Evaluation script, src/eval_iam.py]
path: src/eval_iam.py
---

# eval_iam.py

Sample-and-compare evaluation harness over the [[IAM Words Dataset]].

## CLI

```bash
python -m src.eval_iam --n 20 --model mltu
python -m src.eval_iam --n 20 --model trocr
```

- `--model {trocr,mltu}` — picks backend (lazy import to avoid TF in TrOCR path)
- `--n N` — sample size from `words.txt`
- `--seed SEED` — reproducibility

## Output

Per-sample line: `OK truth='...' pred='...' conf=N.NN` plus a final `K/N exact-match` tally.

## Results history

- TrOCR: 7/20 (case-sensitivity + line-level model on word crops)
- mltu CRNN (Colab-trained, 50 epochs): **19/20**

See [[TrOCR vs mltu]] for the full analysis.

## Related

- [[IAM Words Dataset]]
- [[recognizer.py]], [[mltu_recognizer.py]]
