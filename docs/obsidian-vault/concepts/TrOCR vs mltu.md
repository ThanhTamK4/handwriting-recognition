---
type: concept
tags: [type/concept, ml/comparison]
aliases: [TrOCR vs mltu]
---

# TrOCR vs mltu

Head-to-head on 20 IAM word samples.

|  | [[TrOCR]] | [[mltu CRNN Recognizer]] |
|---|---|---|
| Exact-match | 7/20 | **19/20** |
| Confidence range | 0.50–1.00 | 0.80–1.00 (post-[[Double Softmax Bug]] fix) |
| Size | ~1.4 GB | ~10 MB ONNX |
| Latency | ~800 ms/line | ~20 ms/word |
| Training cost | None (pretrained) | ~30 min GPU / ~17 h CPU |
| Strength | Multi-line, contextual | Word-level, specialized on IAM |

## Why TrOCR loses on this eval

1. **Case sensitivity** — predicts `one/this/suddenly` vs truths `One/This/Suddenly`. Three "misses" from casing alone.
2. **Domain mismatch** — TrOCR expects whole sentences; single-word crops lack context for its decoder LM.

TrOCR still wins on real multi-line paragraphs or unseen writers. mltu wins on IAM-style isolated word crops.

## Three-backend split (current state)

After the live demo round, the UI ships with three options:

| Backend | When to pick it |
|---|---|
| **TrOCR (base handwritten)** | Cursive multi-line pages, unseen writers, real handwriting |
| **TrOCR (printed)** | Printed/scanned documents — same architecture, `microsoft/trocr-base-printed` checkpoint |
| **mltu CRNN (IAM words)** | Isolated IAM-style cursive words — fastest, highest exact-match on this eval |

The English-dictionary correction post-processor is gated to the mltu backend only, since both TrOCR variants already use a transformer LM decoder.

## Related

- [[eval_iam.py]], [[Inference Pipeline]], [[NLP Correction]]
