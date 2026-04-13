# Training the mltu CRNN recognizer

This folder contains an isolated training pipeline for the second recognizer
backend: a **CRNN + CTC** model (mltu Tutorial 03) trained on the local
`data/IAM_Words/` dataset and auto-exported to **ONNX** so the main app can run
inference without TensorFlow.

## Why a separate venv?

mltu pins `tensorflow==2.10`, which in turn requires `protobuf<3.20`. The main
app uses modern `torch`/`transformers` which want `protobuf>=3.20`. These are
mutually incompatible, so training gets its own venv. Inference uses
`onnxruntime` only, so no TF ever enters the main venv.

## One-time setup

From the project root:

```bash
python -m venv .venv-train
.venv-train\Scripts\activate          # Windows
# source .venv-train/bin/activate     # macOS / Linux
pip install -r training/requirements-train.txt
```

## Train (local)

Make sure `data/IAM_Words/words.txt` and the extracted `data/IAM_Words/words/`
tree exist (same layout the TrOCR recognizer uses).

```bash
python training/train_mltu.py
```

Expected runtime:
- **CPU:** multiple hours (~20 min/epoch)
- **GPU (CUDA):** ~20–40 min total

Checkpoints, TensorBoard logs, and the final `model.onnx` + `configs.yaml` land
in `models/mltu/`. Early stopping on validation CER typically ends training
before epoch 50.

## Train (Google Colab — recommended)

Much faster: ~30 min for 50 epochs on Colab's free GPU.

1. Upload `data/IAM_Words/` to your Google Drive (one-time).
2. Open `training/train_mltu_colab.ipynb` in Colab (or via VS Code Colab extension).
3. Edit the `DATA_DIR` and `OUT_DIR` paths in the Config cell to match your Drive.
4. Run all cells. Training outputs `model.onnx` + `configs.yaml` to Drive.
5. Copy the two files from Drive to your local `models/mltu/`.
6. Restart Streamlit and select "mltu CRNN" in the sidebar.

### Data augmentation

The training pipeline (both local and Colab) applies random augmentations to
training images only (not validation):

- **RandomBrightness** — lightens/darkens by up to ±100 (30% chance)
- **RandomRotate** — tilts ±5° (30% chance)
- **RandomErodeDilate** — thins/thickens strokes (30% chance)
- **RandomGaussianBlur** — slight blur (20% chance)
- **RandomElasticTransform** — local warping for handwriting variation (20% chance)

This reduces overfitting and typically closes the train/val CER gap by 2–3%.

## Verify

Deactivate the training venv and return to your main venv:

```bash
deactivate
.venv\Scripts\activate          # main venv
python -c "from src.mltu_recognizer import MltuRecognizer; from PIL import Image; r=MltuRecognizer(); print(r.predict(Image.open('data/IAM_Words/words/a01/a01-000u/a01-000u-00-00.png')))"
```

You should see a `PredictionResult` with the correct word and a high
confidence score.

## Use in the app

Restart Streamlit and pick **"mltu CRNN (IAM words)"** from the sidebar model
selector.

```bash
streamlit run app/streamlit_app.py
```

## Troubleshooting

- **`protobuf` conflict during install**: you're not in `.venv-train`. Activate
  it first.
- **`onnx` export fails at the end of training**: install `tf2onnx` and `onnx`
  (already in `requirements-train.txt`), rerun the script — the best checkpoint
  is kept so training doesn't need to restart.
- **Out of memory**: lower `configs.batch_size` in `train_mltu.py` (e.g. 32 or 16).
