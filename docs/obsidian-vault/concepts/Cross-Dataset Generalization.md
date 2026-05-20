---
type: concept
tags: [type/concept, ml/evaluation, ml/generalization]
aliases: [Cross-Dataset Generalization, Out-of-Domain Generalization, Archive Benchmark]
---

# Cross-Dataset Generalization

Does the project's recognizer stack work on **handwritten data outside the IAM Words training distribution**? Empirical answer: **no, not without fine-tuning** — for either backend.

## The benchmark

The user's `~/OneDrive/Máy tính/archive` folder is the Kaggle **Handwriting Recognition dataset** (Landlord) — 330k train + 41k test + 41k validation images of handwritten names, indexed by `FILENAME,IDENTITY` CSVs. Same task as IAM (word-level offline handwriting recognition) but a different *domain*: handwritten **personal names**, frequently in ALL-CAPS block style, by a different population of writers.

Harness: `src/eval_archive.py` (mirrors `src/eval_iam.py` but reads the Kaggle CSV layout). Shared scoring helpers in `src/_eval_utils.py` (CER + case-aware exact match).

## Measured numbers

`python -m src.eval_archive --n 100 --model both --ignore-case` (seed 0, test split):

| Backend | Exact-match | Mean CER |
|---|---|---|
| [[mltu CRNN Recognizer]] (trained on [[IAM Words Dataset]]) | **0 / 100** | 0.864 |
| [[TrOCR]] (`microsoft/trocr-base-handwritten`) | **8 / 100** | 1.748 |

For comparison, the same backends on **IAM Words** (their training-style distribution):

| Backend | Exact-match (IAM) | Mean CER (IAM) |
|---|---|---|
| mltu | 91 / 100 | 0.036 |
| mltu + [[NLP Correction]] | 92 / 100 | 0.035 |

The gap is **near-total collapse on archive** for mltu, and a different-flavor failure for TrOCR.

## Why each backend fails

**mltu CRNN (0 / 100):** trained only on IAM, which is mostly **lowercase cursive English words**. Archive names are typically **all-caps block letters** — a different shape distribution entirely. The CRNN learned visual features for `the`, `suddenly`, `machine`-style stroke patterns; it has no template for `NATHAN`, `LACOSTE`, `BALTHAZAR`. Sample outputs are pure hallucination: `'NATHAN' → 'Prinr'`, `'LACOSTE' → 'icosre'`. Confidence stays in the 0.6–0.8 range — the model is *confidently wrong*.

**TrOCR (8 / 100, CER > 1.0):** different failure mode. TrOCR's RoBERTa decoder is **sentence-aware** — it expects natural-language context. Given a single isolated name crop, it confabulates context: `'NATHAN' → 'Prenom : N A.T.H.A.N. 0200'`, `'DELFO' → 'Lavomacket to the United States'`, `'LACOSTE' → 'Lacoste 1887'`. CER above 1.0 reflects that TrOCR's predictions are *longer* than the truth on average — the LM is filling in fake surroundings.

Both backends are bad, in opposite ways: mltu under-reads (visual mismatch), TrOCR over-reads (context hallucination).

## What this means for the project

1. **The current shipped models are IAM-specific tools.** Calling the app "a handwriting recognizer" without qualification overstates what it does. The honest claim is *"a handwriting recognizer for IAM-style cursive English text"* + *"a printed-text recognizer via [[TrOCR]]-printed"*.

2. **The path to fixing this is fine-tuning, not architecture.** Same CRNN trained on the archive's 330k name images would likely match IAM-style accuracy on archive. The training notebook ([[train_mltu_colab.ipynb]]) already supports this — point the `DRIVE_DATA_DIR` at the archive and adjust the data loader for the CSV format. The Colab GPU training run is ~30 min.

3. **TrOCR-handwritten is not a "general handwriting" oracle.** Its LM bias makes it worse than mltu on isolated short tokens in the wrong domain. Use TrOCR for paragraphs; use a fine-tuned CRNN for isolated word/name crops.

## Reproducing

```powershell
python -m src.eval_archive --n 100 --model mltu --ignore-case    # 0/100, CER 0.864
python -m src.eval_archive --n 100 --model trocr --ignore-case   # 8/100, CER 1.748
python -m src.eval_archive --n 100 --model both  --ignore-case   # side-by-side
```

The `--archive-dir` flag overrides the default path if the archive moves.

## Related

- [[eval_iam.py]] — sibling benchmark on the in-distribution dataset
- [[mltu CRNN Recognizer]], [[TrOCR]], [[CRNN]]
- [[IAM Words Dataset]]
- [[NLP Correction]] (no effect here — most archive errors are not OOV typos, they're complete misreads)
