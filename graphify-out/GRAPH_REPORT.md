# Graph Report - handwriting  (2026-05-20)

## Corpus Check
- 14 files · ~44,131,995 words
- Verdict: corpus is large enough that graph structure adds value.

## Summary
- 189 nodes · 370 edges · 16 communities detected
- Extraction: 69% EXTRACTED · 31% INFERRED · 0% AMBIGUOUS · INFERRED: 115 edges (avg confidence: 0.62)
- Token cost: 0 input · 0 output

## Community Hubs (Navigation)
- [[_COMMUNITY_Community 0|Community 0]]
- [[_COMMUNITY_Community 1|Community 1]]
- [[_COMMUNITY_Community 2|Community 2]]
- [[_COMMUNITY_Community 3|Community 3]]
- [[_COMMUNITY_Community 4|Community 4]]
- [[_COMMUNITY_Community 5|Community 5]]
- [[_COMMUNITY_Community 6|Community 6]]
- [[_COMMUNITY_Community 7|Community 7]]
- [[_COMMUNITY_Community 8|Community 8]]
- [[_COMMUNITY_Community 9|Community 9]]
- [[_COMMUNITY_Community 12|Community 12]]
- [[_COMMUNITY_Community 13|Community 13]]
- [[_COMMUNITY_Community 14|Community 14]]
- [[_COMMUNITY_Community 15|Community 15]]
- [[_COMMUNITY_Community 16|Community 16]]
- [[_COMMUNITY_Community 17|Community 17]]

## God Nodes (most connected - your core abstractions)
1. `MltuRecognizer` - 23 edges
2. `WordCorrector` - 23 edges
3. `ImageLoadError` - 22 edges
4. `load_image_from_url()` - 21 edges
5. `Recognizer` - 19 edges
6. `PredictionResult` - 17 edges
7. `word_polygons()` - 10 edges
8. `predict()` - 9 edges
9. `run_pipeline()` - 8 edges
10. `main()` - 8 edges

## Surprising Connections (you probably didn't know these)
- `_load_trocr()` --calls--> `Recognizer`  [INFERRED]
  app\streamlit_app.py → src\recognizer.py
- `_load_trocr_printed()` --calls--> `Recognizer`  [INFERRED]
  app\streamlit_app.py → src\recognizer.py
- `_load_mltu()` --calls--> `MltuRecognizer`  [INFERRED]
  app\streamlit_app.py → src\mltu_recognizer.py
- `_load_corrector()` --calls--> `WordCorrector`  [INFERRED]
  app\streamlit_app.py → src\postprocess.py
- `draw_line_overlay()` --calls--> `line_polygons()`  [INFERRED]
  app\streamlit_app.py → src\segment.py

## Communities

### Community 0 - "Community 0"
Cohesion: 0.09
Nodes (39): _decode_data_uri(), _flatten_to_rgb(), ImageLoadError, load_image_from_url(), _pil_from_bytes(), Pure image-loading helpers shared by the Streamlit UI tabs.  These are deliberat, Drop alpha by compositing onto an opaque background.      PNGs with transparent, Raised when a URL / data URI can't be turned into an image.      Subclasses ``Va (+31 more)

### Community 1 - "Community 1"
Cohesion: 0.16
Nodes (22): Benchmark recognizers on the Kaggle "Handwriting Recognition" dataset (handwritt, Return (csv_path, image_dir) for the requested split.      The Kaggle dataset us, Run one backend over all samples, print per-row + summary., Run both backends side-by-side, print per-row + summary., Sanity-check the recognizer on N random IAM_Words samples.  Usage:     python -m, MltuRecognizer, ONNX-based recognizer for the mltu CRNN trained on IAM_Words.  Depends only on o, Resize + color-convert to the exact tensor the ONNX model expects. (+14 more)

### Community 2 - "Community 2"
Cohesion: 0.17
Nodes (22): _binarize(), _cluster_components_to_lines(), _line_bbox(), _line_clusters(), line_polygons(), _line_tilt_deg(), Line and word segmentation using connected-components + adaptive clustering., Estimate line tilt in degrees from component centroids (zero if too few). (+14 more)

### Community 3 - "Community 3"
Cohesion: 0.22
Nodes (15): load_samples(), main(), _open(), _resolve_split(), _run_both(), _run_single(), load_samples(), main() (+7 more)

### Community 4 - "Community 4"
Cohesion: 0.15
Nodes (11): Sequence, build_model(), KerasSequenceProvider, load_samples(), main(), Train an mltu CRNN+CTC word recognizer on the local IAM_Words dataset.  Ported f, Drop-in replacement for mltu.preprocessors.ImageReader that handles     non-ASCI, Wraps an mltu DataProvider so Keras 2.10 recognizes it as a Sequence. (+3 more)

### Community 5 - "Community 5"
Cohesion: 0.15
Nodes (11): _confs_for_core(), English-dictionary word-correction post-processor for CRNN output.  Wraps SymSpe, Return (leading-punct, alnum-core, trailing-punct)., Mirror original_core's case pattern onto corrected_lower., Slice `char_confs` to the positions covered by `core`.      If lengths don't lin, Only accept the correction if the characters being changed were     low-confiden, Return (possibly-corrected word, did_change)., Split on whitespace, correct each token, rejoin.          `per_word_confs[i]` is (+3 more)

### Community 6 - "Community 6"
Cohesion: 0.35
Nodes (10): confidence_badge(), draw_line_overlay(), get_recognizer(), _highlight_diff(), _load_corrector(), _load_mltu(), _load_trocr(), _load_trocr_printed() (+2 more)

### Community 7 - "Community 7"
Cohesion: 0.25
Nodes (4): _looks_like_freq_dict(), Load words + frequencies into self._symspell.          Resolution order:, Layer a newline-separated word list (Kaggle format) on top of any         alread, Sniff the first non-empty lines: if at least half are `<word> <int>`     (SymSpe

### Community 8 - "Community 8"
Cohesion: 0.32
Nodes (6): _is_blank(), line_polygons(), _mean_token_probability(), predict(), TrOCR-based handwriting recognizer.  Loads `microsoft/trocr-base-handwritten` on, Return list of 4x2 int polygons for line overlay drawing.

### Community 9 - "Community 9"
Cohesion: 0.53
Nodes (5): _deskew(), _order_points(), _perspective_correct(), preprocess(), Image preprocessing for real-world handwriting photos / scans.  Pipeline (config

### Community 12 - "Community 12"
Cohesion: 1.0
Nodes (1): Raised when a URL / data URI can't be turned into an image.      Subclasses ``Va

### Community 13 - "Community 13"
Cohesion: 1.0
Nodes (1): Fetch an image from an ``http(s)://`` URL or a ``data:image/...`` URI.      Alwa

### Community 14 - "Community 14"
Cohesion: 1.0
Nodes (1): Decode ``data:image/png;base64,...`` (and similar) into a PIL image.      Plain

### Community 15 - "Community 15"
Cohesion: 1.0
Nodes (1): Drop alpha by compositing onto an opaque background.      PNGs with transparent

### Community 16 - "Community 16"
Cohesion: 1.0
Nodes (1): True iff the canvas has no strokes (every pixel transparent).

### Community 17 - "Community 17"
Cohesion: 1.0
Nodes (1): Flatten the RGBA canvas array onto ``bg_color`` and return a PIL.Image.      Ret

## Knowledge Gaps
- **42 isolated node(s):** `Pure image-loading helpers shared by the Streamlit UI tabs.  These are deliberat`, `Raised when a URL / data URI can't be turned into an image.      Subclasses ``Va`, `Fetch an image from an ``http(s)://`` URL or a ``data:image/...`` URI.      Alwa`, `Decode ``data:image/png;base64,...`` (and similar) into a PIL image.      Plain`, `Drop alpha by compositing onto an opaque background.      PNGs with transparent` (+37 more)
  These have ≤1 connection - possible missing edges or undocumented components.
- **Thin community `Community 12`** (1 nodes): `Raised when a URL / data URI can't be turned into an image.      Subclasses ``Va`
  Too small to be a meaningful cluster - may be noise or needs more connections extracted.
- **Thin community `Community 13`** (1 nodes): `Fetch an image from an ``http(s)://`` URL or a ``data:image/...`` URI.      Alwa`
  Too small to be a meaningful cluster - may be noise or needs more connections extracted.
- **Thin community `Community 14`** (1 nodes): `Decode ``data:image/png;base64,...`` (and similar) into a PIL image.      Plain`
  Too small to be a meaningful cluster - may be noise or needs more connections extracted.
- **Thin community `Community 15`** (1 nodes): `Drop alpha by compositing onto an opaque background.      PNGs with transparent`
  Too small to be a meaningful cluster - may be noise or needs more connections extracted.
- **Thin community `Community 16`** (1 nodes): `True iff the canvas has no strokes (every pixel transparent).`
  Too small to be a meaningful cluster - may be noise or needs more connections extracted.
- **Thin community `Community 17`** (1 nodes): `Flatten the RGBA canvas array onto ``bg_color`` and return a PIL.Image.      Ret`
  Too small to be a meaningful cluster - may be noise or needs more connections extracted.

## Suggested Questions
_Questions this graph is uniquely positioned to answer:_

- **Why does `ImageLoadError` connect `Community 0` to `Community 1`, `Community 4`?**
  _High betweenness centrality (0.464) - this node is a cross-community bridge._
- **Why does `WordCorrector` connect `Community 1` to `Community 3`, `Community 5`, `Community 6`, `Community 7`?**
  _High betweenness centrality (0.224) - this node is a cross-community bridge._
- **Why does `MltuRecognizer` connect `Community 1` to `Community 3`, `Community 6`?**
  _High betweenness centrality (0.146) - this node is a cross-community bridge._
- **Are the 17 inferred relationships involving `MltuRecognizer` (e.g. with `Streamlit UI: upload image -> TrOCR / mltu CRNN prediction.  Features:     - Ima` and `Overlay red line polygons + green word polygons on the original image.`) actually correct?**
  _`MltuRecognizer` has 17 INFERRED edges - model-reasoned connections that need verification._
- **Are the 14 inferred relationships involving `WordCorrector` (e.g. with `Streamlit UI: upload image -> TrOCR / mltu CRNN prediction.  Features:     - Ima` and `Overlay red line polygons + green word polygons on the original image.`) actually correct?**
  _`WordCorrector` has 14 INFERRED edges - model-reasoned connections that need verification._
- **Are the 16 inferred relationships involving `ImageLoadError` (e.g. with `Streamlit UI: upload image -> TrOCR / mltu CRNN prediction.  Features:     - Ima` and `Overlay red line polygons + green word polygons on the original image.`) actually correct?**
  _`ImageLoadError` has 16 INFERRED edges - model-reasoned connections that need verification._
- **Are the 16 inferred relationships involving `load_image_from_url()` (e.g. with `.iter_content()` and `test_data_uri_base64_png_round_trips()`) actually correct?**
  _`load_image_from_url()` has 16 INFERRED edges - model-reasoned connections that need verification._