---
type: concept
tags: [type/concept, code/interface]
aliases: [PredictionResult]
---

# PredictionResult

Shared dataclass returned by both [[recognizer.py]] and [[mltu_recognizer.py]]. Keeps the UI backend-agnostic.

```python
@dataclass
class PredictionResult:
    text: str                              # final string (possibly multi-line)
    confidence: float                       # 0..1, mean per-token/char probability
    line_results: List[PredictionResult]   # per-line breakdown (populated by predict_lines)
```

Defined in [[recognizer.py]]; re-imported by [[mltu_recognizer.py]].

## Consumers

- [[streamlit_app.py]] — `render_result()` renders `text` + [[Confidence Badge]] + expander with `line_results`
- [[eval_iam.py]] — compares `result.text` against ground truth
