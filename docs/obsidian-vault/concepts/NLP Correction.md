---
type: concept
tags: [type/concept, ml/postprocess, nlp]
aliases: [NLP Correction, Dictionary Correction, Word Corrector, WordCorrector]
---

# NLP Correction

Optional post-processing pass that snaps out-of-vocabulary [[CRNN]] output to the nearest English word. Lives in `src/postprocess.py` as `WordCorrector`.

## Why

The [[mltu CRNN Recognizer]] has no language prior — so visually ambiguous char pairs (`rn` ↔ `m`, `cl` ↔ `d`, `0` ↔ `o`) produce non-words like `rnove`, `dover`, `h0me`. A dictionary lookup fixes these without retraining.

[[TrOCR]] already has a RoBERTa decoder and does not benefit, so correction is gated to mltu only.

## How

1. **Dictionary**: [Kaggle English Dictionary](https://www.kaggle.com/datasets/kaleidawave/english-dictionary) at `data/english_dictionary/words.txt` (370k words). Falls back to symspellpy's bundled 82k list if missing.
2. **Index**: [SymSpell](https://github.com/wolfgarbe/SymSpell) — precomputes a delete-variant index giving O(1) lookup up to edit distance 2.
3. **Confidence gate**: uses the per-character CTC confidences already produced in `_ctc_greedy_decode` (see [[mltu_recognizer.py]]). If the characters that *would change* have mean confidence ≥ `conf_threshold` (default 0.85), the correction is rejected — we don't overwrite chars the model was sure about.
4. **Case + punctuation preserved**: `Helo,` → `Hello,`, `ONE` → `ONE` (already a word), `rnove` → `move`.

## API

```python
from src.postprocess import WordCorrector
corrector = WordCorrector()                             # auto-loads dict
word, changed = corrector.correct_word("rnove")         # ("move", True)
text, flags   = corrector.correct_text("rnove dover")   # ("move over", [True, True])
```

## Wiring

- [[mltu_recognizer.py]] `predict` / `predict_lines` accept an optional `corrector=` kwarg. When present, output `PredictionResult` carries both `text` (corrected) and `raw_text` (original) plus `corrected: bool`.
- [[streamlit_app.py]] exposes a sidebar checkbox "English-dictionary correction"; renders a side-by-side raw/corrected diff when they differ.
- [[eval_iam.py]] `--correct` flag prints **both** raw and corrected exact-match counts so the delta is visible in the same run.

## Measured result

`python -m src.eval_iam --n 100 --model mltu --correct` (seed 0):

| Run | Exact-match |
|---|---|
| raw | 91/100 |
| corrected | **92/100** (+1, no regressions) |

The single recovered case: `Stinned → Stunned` (1-edit OOV typo, both characters around the swap had low CTC confidence).

## Failure-class breakdown (the 8 remaining errors)

Out of the 9 raw errors, only 1 was correctable. The other 8 are unfixable by any flat-dictionary approach:

| Class | Example | Why unfixable |
|---|---|---|
| Pure punctuation | `,` ↔ `.` | stripped to empty core, nothing to look up |
| Real-word substitution | `grin` ↔ `gain` | both valid English words |
| Plural / morphology | `lettings` ↔ `letting` | base form is valid, corrector is conservative |
| Proper-noun abbreviation | `Co.` ↔ `Ko.` | not in the English dictionary |
| Edit-distance > 2 | `mechim` ↔ `machine` | 3 substitutions; even if reachable, equal-edit closer neighbors win |
| Frequency-tied 1-edit ambiguity | `aas` ↔ `was` | `aas`, `wad`, `way` all 1-edit |
| Contractions / possessives | `Delaney's` | apostrophe-gated by design (was the source of the only regression we saw) |

These need a context-aware language model (n-gram, KenLM, or transformer LM) over **sequences**, not a per-word dictionary.

## Tuning history (what didn't work, for posterity)

- **Edit distance 3 + threshold 0.99 + Kaggle 479k**: tried to recover `mechim → machine`. It found a 1-edit nonsense word in the 479k list (`mechir`) and "fixed" `Delaney's → Delaney`. Net `-1`.
- **SymSpell official 500k frequency dictionary** (the "obvious upgrade"): the larger dict's web-corpus crud (`aas`, `mecham`, `alld` are all in it) caused fewer corrections to fire, and frequency-based ranking picked `Sinned` over `Stunned` for input `Stinned` (more frequent). Net `+0` — strictly worse than the layered approach.
- **Final config (winning)**: bundled symspellpy 82k frequency dict (for ranking quality) + Kaggle `words.txt` 479k layered on top at `count=1` (for membership coverage without polluting frequencies) + apostrophe-gating + `max_edit_distance=3` + `conf_threshold=0.99`.

## Related

- [[CTC Loss]], [[Double Softmax Bug]] (per-char confidences depend on the softmax fix), [[PredictionResult]]

## Related

- [[CTC Loss]], [[Double Softmax Bug]] (per-char confidences depend on the softmax fix), [[PredictionResult]]
