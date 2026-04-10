"""Streamlit UI: upload image OR webcam snapshot -> TrOCR prediction.

Features:
    - Upload + webcam tabs
    - Multi-line mode with line preview overlay
    - Optional preprocessing (deskew / denoise / contrast / binarize / perspective)
    - Confidence score
    - Copy-to-clipboard via st.code
    - Blank-image / error handling
"""
from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import streamlit as st
from PIL import Image, ImageDraw

# Make `src` importable when run via `streamlit run app/streamlit_app.py`
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from src.preprocess import PreprocessOptions, preprocess  # noqa: E402
from src.recognizer import PredictionResult, Recognizer, line_boxes  # noqa: E402

TROCR_LABEL = "TrOCR (base handwritten)"
MLTU_LABEL = "mltu CRNN (IAM words)"


@st.cache_resource(show_spinner="Loading TrOCR model (first run downloads ~1.4 GB)...")
def _load_trocr() -> Recognizer:
    return Recognizer()


@st.cache_resource(show_spinner="Loading mltu CRNN ONNX model...")
def _load_mltu():
    from src.mltu_recognizer import MltuRecognizer  # imported lazily

    return MltuRecognizer()


def get_recognizer(choice: str):
    if choice == MLTU_LABEL:
        return _load_mltu()
    return _load_trocr()


def draw_line_overlay(image: Image.Image) -> Image.Image:
    boxes = line_boxes(image)
    if not boxes:
        return image
    overlay = image.convert("RGB").copy()
    draw = ImageDraw.Draw(overlay)
    w = overlay.width
    for y0, y1 in boxes:
        draw.rectangle([(0, y0), (w - 1, y1)], outline=(255, 0, 0), width=3)
    return overlay


def confidence_badge(conf: float) -> str:
    if conf >= 0.9:
        return f"🟢 **{conf:.0%}**"
    if conf >= 0.7:
        return f"🟡 **{conf:.0%}**"
    return f"🔴 **{conf:.0%}**"


def render_result(result: PredictionResult, key_prefix: str) -> None:
    if not result.text:
        st.warning("No text detected. Try preprocessing or a clearer image.")
        return
    st.subheader("Prediction")
    st.markdown(f"Confidence: {confidence_badge(result.confidence)}")
    st.code(result.text, language=None)  # built-in copy button
    if result.line_results:
        with st.expander("Per-line breakdown"):
            for i, lr in enumerate(result.line_results, 1):
                st.markdown(
                    f"**Line {i}** &middot; {confidence_badge(lr.confidence)} &middot; `{lr.text}`",
                    unsafe_allow_html=True,
                )


def run_pipeline(
    image: Image.Image,
    model_choice: str,
    multiline: bool,
    apply_preproc: bool,
    opts: PreprocessOptions,
    key_prefix: str,
) -> None:
    try:
        processed = preprocess(image, opts) if apply_preproc else image
    except Exception as e:  # pragma: no cover
        st.error(f"Preprocessing failed: {e}")
        return

    if apply_preproc:
        with st.expander("Preprocessed image"):
            st.image(processed, use_container_width=True)

    if multiline:
        st.image(
            draw_line_overlay(processed),
            caption="Detected lines",
            use_container_width=True,
        )

    try:
        rec = get_recognizer(model_choice)
    except FileNotFoundError as e:
        st.error(
            f"{e}\n\nTrain the model first — see `training/README.md`, or switch "
            f"to **{TROCR_LABEL}** in the sidebar."
        )
        return
    except Exception as e:  # pragma: no cover
        st.error(f"Could not load model: {e}")
        return

    try:
        with st.spinner("Recognizing..."):
            result = (
                rec.predict_lines(processed) if multiline else rec.predict(processed)
            )
    except Exception as e:  # pragma: no cover
        st.error(f"Recognition failed: {e}")
        return

    render_result(result, key_prefix)


# ---------- page ----------

st.set_page_config(page_title="Handwriting Recognition", layout="centered")
st.title("Handwriting Recognition")

with st.sidebar:
    st.header("Options")
    model_choice = st.selectbox(
        "Model",
        options=[TROCR_LABEL, MLTU_LABEL],
        index=0,
        help="TrOCR is line-level and strong on paragraphs. mltu CRNN is word-level, fast, and trained locally on IAM.",
    )
    multiline = st.checkbox(
        "Multi-line mode", value=False, help="Split image into horizontal lines."
    )
    st.divider()
    st.subheader("Preprocessing")
    apply_preproc = st.checkbox("Apply preprocessing", value=False)
    opts = PreprocessOptions(
        perspective=st.checkbox("Perspective correction", value=False, disabled=not apply_preproc),
        deskew=st.checkbox("Deskew", value=True, disabled=not apply_preproc),
        denoise=st.checkbox("Denoise", value=True, disabled=not apply_preproc),
        enhance_contrast=st.checkbox("Enhance contrast", value=True, disabled=not apply_preproc),
        binarize=st.checkbox("Binarize", value=False, disabled=not apply_preproc),
    )

upload_tab, webcam_tab = st.tabs(["📁 Upload image", "📷 Webcam"])

with upload_tab:
    uploaded = st.file_uploader("Choose an image", type=["png", "jpg", "jpeg"])
    if uploaded is not None:
        try:
            image = Image.open(uploaded)
        except Exception as e:
            st.error(f"Could not open image: {e}")
        else:
            st.image(image, caption="Input", use_container_width=True)
            run_pipeline(image, model_choice, multiline, apply_preproc, opts, key_prefix="upload")

with webcam_tab:
    snapshot = st.camera_input("Take a photo of handwriting")
    if snapshot is not None:
        try:
            image = Image.open(snapshot)
        except Exception as e:
            st.error(f"Could not read snapshot: {e}")
        else:
            run_pipeline(image, model_choice, multiline, apply_preproc, opts, key_prefix="webcam")
