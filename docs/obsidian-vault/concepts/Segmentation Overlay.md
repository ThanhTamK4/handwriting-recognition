---
type: concept
tags: [type/concept, ui/component]
aliases: [Segmentation Overlay]
---

# Segmentation Overlay

The red+green visualization drawn in multi-line mode by [[streamlit_app.py]]'s `draw_line_overlay()`.

- **Red polygons** — per-line boxes from `line_polygons()`
- **Green polygons** — per-word boxes from `word_polygons()`

Exposes the exact regions that the [[mltu CRNN Recognizer]] will receive as input batches, making segmentation failures easy to diagnose visually.

## Relies on

- [[Connected Components Segmentation]]
- [[segment.py]] helpers

## Related

- [[Streamlit UI]]
