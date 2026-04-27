---
type: concept
tags: [type/concept, ui/component]
aliases: [Confidence Badge]
---

# Confidence Badge

Visual indicator shown next to predictions in [[streamlit_app.py]].

| Range | Badge | Color |
|---|---|---|
| ≥ 0.90 | 🟢 | Green |
| ≥ 0.70 | 🟡 | Yellow |
| < 0.70 | 🔴 | Red |

Implemented in `confidence_badge(conf: float) -> str`. Displayed both for the top-level [[PredictionResult]] and each `line_results[i]` in the expander.
