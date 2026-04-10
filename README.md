# Handwriting Recognition

A Python handwriting recognition app with two interchangeable model backends
and a Streamlit web UI featuring image upload and webcam scanning.

Trained and evaluated on the [IAM Handwriting Dataset](https://fki.tic.heia-fr.ch/databases/iam-handwriting-database)
(word-level subset from Kaggle).

## Features

- **Two recognizer backends** — switch in the sidebar at runtime:
  - **TrOCR** (`microsoft/trocr-base-handwritten`) — pretrained, line-level, zero training required
  - **mltu CRNN** — CNN + BiLSTM + CTC trained locally on IAM_Words, exported to ONNX (~10 MB, ~20 ms/word)
- **Streamlit web app** with two input tabs:
  - Upload image (PNG / JPG)
  - Webcam snapshot (`st.camera_input`)
- **Multi-line mode** — splits paragraphs into lines (and words for CRNN) automatically
- **Image preprocessing** sidebar — deskew, denoise, contrast enhancement, perspective correction, binarize
- **Confidence scoring** — per-prediction and per-line, shown as color-coded badges
- **Copy-to-clipboard** button on results

## Quick start

### 1. Install

```bash
python -m venv .venv
.venv\Scripts\activate          # Windows
# source .venv/bin/activate     # macOS / Linux
pip install -r requirements.txt
```

### 2. Run the app (TrOCR — no training needed)

```bash
streamlit run app/streamlit_app.py
```

First launch downloads ~1.4 GB of TrOCR weights (one-time). Select
**"TrOCR (base handwritten)"** in the sidebar and upload an image or use the
webcam tab.

### 3. (Optional) Train the mltu CRNN backend

Requires a separate venv with TensorFlow 2.10 (avoids protobuf conflicts):

```bash
python -m venv .venv-train
.venv-train\Scripts\activate
pip install -r training/requirements-train.txt
python training/train_mltu.py
deactivate
```

Training reads from `data/IAM_Words/` (must be extracted beforehand). After
training, `models/mltu/model.onnx` + `configs.yaml` are produced. Restart
Streamlit and select **"mltu CRNN (IAM words)"** in the sidebar.

See [`training/README.md`](training/README.md) for details and troubleshooting.

### 4. Evaluate on IAM words

```bash
python -m src.eval_iam --n 20                   # TrOCR (default)
python -m src.eval_iam --n 20 --model mltu      # mltu CRNN
```

Picks N random `ok` words, prints `truth`, `pred`, and `confidence` per sample,
then reports exact-match count.

## Project layout

```
handwriting/
├── app/
│   └── streamlit_app.py            # Streamlit UI (upload + webcam + model selector)
├── src/
│   ├── recognizer.py               # TrOCR backend (predict, predict_lines, PredictionResult)
│   ├── mltu_recognizer.py          # mltu CRNN ONNX backend (same interface)
│   ├── segment.py                  # OpenCV line/word segmentation
│   ├── preprocess.py               # deskew, denoise, CLAHE, perspective, binarize
│   └── eval_iam.py                 # IAM word-level evaluation script
├── training/
│   ├── train_mltu.py               # CRNN training script (TF 2.10 + mltu)
│   ├── requirements-train.txt      # isolated training dependencies
│   ├── README.md                   # training setup guide
│   └── ARCHITECTURE.md             # mltu CRNN training route diagram
├── docs/
│   └── MODELS.md                   # logical maps for both backends
├── data/
│   └── IAM_Words/                  # dataset (gitignored, extract locally)
│       ├── words.txt
│       └── words/                  # word PNG images
├── models/
│   └── mltu/                       # trained ONNX model + configs (gitignored)
├── requirements.txt                # main app dependencies
└── .gitignore
```

## Model comparison

| | TrOCR (base) | mltu CRNN |
|---|---|---|
| Parameters | ~334M | ~1.5M |
| Disk size | ~1.4 GB (HF cache) | ~10 MB (.onnx) |
| CPU inference | 1–3 s | ~20 ms/word |
| Training required | No | Yes (~hours CPU) |
| Input unit | Full line | Single word |
| Multi-line | Native (line-level model) | Via segmentation pipeline |
| Best for | Paragraphs, webcam photos | Single words, fast scanning |

See [`docs/MODELS.md`](docs/MODELS.md) for detailed logical maps and pipeline
diagrams of both backends.

## Preprocessing options

Available in the Streamlit sidebar under "Preprocessing":

| Option | What it does | When to use |
|---|---|---|
| Perspective correction | Detects page quad, warps to flat rectangle | Tilted full-page photos |
| Deskew | Rotates text to horizontal baseline | Slightly rotated scans |
| Denoise | Bilateral filter (preserves edges) | Grainy phone/webcam shots |
| Enhance contrast | CLAHE (adaptive histogram equalization) | Faint pencil, uneven lighting |
| Binarize | Adaptive threshold to pure black/white | Heavily stained paper (last resort) |

## Tech stack

- **Inference (main venv):** PyTorch, HuggingFace Transformers, ONNX Runtime, OpenCV, Streamlit
- **Training (isolated venv):** TensorFlow 2.10, mltu 1.2.5
- **No TensorFlow in the main app** — the CRNN is exported to ONNX; inference
  uses `onnxruntime` only, avoiding all protobuf version conflicts.

## Notes

- TrOCR was trained on IAM *lines*, not words. Word-level exact-match scores
  (~35%) are misleadingly low; real-world line/paragraph accuracy is much higher.
- mltu CRNN was trained on IAM *words*. It scores ~12% CER after 10 epochs and
  reaches ~7% CER with 40–50 epochs.
- CUDA GPU is used automatically when available for both backends.
- HuggingFace model cache lives at `~/.cache/huggingface/` by default; set
  `HF_HOME` env var to relocate.
