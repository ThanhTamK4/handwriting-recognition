"""Streamlit UI: upload / draw / paste an image -> TrOCR / mltu CRNN prediction.

Three input tabs share the same recognition pipeline:
    - File upload (PNG / JPG)
    - Drawable canvas (live handwriting via mouse / stylus)
    - URL or clipboard paste (data: URIs and http(s) image URLs)

Plus:
    - Multi-line mode with line preview overlay
    - Optional preprocessing (deskew / denoise / contrast / binarize / perspective)
    - Confidence score
    - Copy-to-clipboard via st.code
    - Blank-image / error handling
"""
from __future__ import annotations

import sys
from pathlib import Path

import streamlit as st
from PIL import Image, ImageDraw

# Make `src` importable when run via `streamlit run app/streamlit_app.py`
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import streamlit.components.v1 as components  # noqa: E402

from src.loaders import (  # noqa: E402
    ImageLoadError,
    load_image_from_url,
)
from src.preprocess import PreprocessOptions, preprocess  # noqa: E402
from src.recognizer import PredictionResult, Recognizer  # noqa: E402
from src.segment import line_polygons, word_polygons  # noqa: E402

# Self-contained drawable canvas. We ship our own minimal component instead
# of `streamlit-drawable-canvas` because that package (last released 2023)
# silently fails to render inside `st.tabs` on Streamlit >= 1.40 — the
# iframe is created but lazy-mount collapses it to zero height.
_CANVAS_DIR = Path(__file__).resolve().parent / "components" / "handwriting_canvas"
_handwriting_canvas = components.declare_component(
    "handwriting_canvas", path=str(_CANVAS_DIR)
)

TROCR_LABEL = "TrOCR (base handwritten)"
TROCR_PRINTED_LABEL = "TrOCR (printed)"
MLTU_LABEL = "mltu CRNN (IAM words)"


@st.cache_resource(show_spinner="Loading TrOCR model (first run downloads ~1.4 GB)...")
def _load_trocr() -> Recognizer:
    return Recognizer()


@st.cache_resource(show_spinner="Loading TrOCR (printed) model (first run downloads ~1.4 GB)...")
def _load_trocr_printed() -> Recognizer:
    return Recognizer(model_name="microsoft/trocr-base-printed")


@st.cache_resource(show_spinner="Loading mltu CRNN ONNX model...")
def _load_mltu():
    from src.mltu_recognizer import MltuRecognizer  # imported lazily

    return MltuRecognizer()


@st.cache_resource(show_spinner="Loading English dictionary...")
def _load_corrector():
    from src.postprocess import WordCorrector

    return WordCorrector()


def get_recognizer(choice: str):
    if choice == MLTU_LABEL:
        return _load_mltu()
    if choice == TROCR_PRINTED_LABEL:
        return _load_trocr_printed()
    return _load_trocr()


def draw_line_overlay(image: Image.Image) -> Image.Image:
    """Overlay red line polygons + green word polygons on the original image."""
    l_polys = line_polygons(image)
    w_polys = word_polygons(image)
    if not l_polys and not w_polys:
        return image
    overlay = image.convert("RGB").copy()
    draw = ImageDraw.Draw(overlay)
    for line in w_polys:
        for poly in line:
            pts = [tuple(map(int, p)) for p in poly]
            draw.polygon(pts, outline=(0, 180, 0))
    for poly in l_polys:
        pts = [tuple(map(int, p)) for p in poly]
        draw.polygon(pts, outline=(255, 0, 0))
        # Thicken line outline by re-drawing with slight offsets.
        for dx, dy in ((1, 0), (-1, 0), (0, 1), (0, -1)):
            shifted = [(x + dx, y + dy) for x, y in pts]
            draw.polygon(shifted, outline=(255, 0, 0))
    return overlay


def confidence_badge(conf: float) -> str:
    if conf >= 0.9:
        return f"🟢 **{conf:.0%}**"
    if conf >= 0.7:
        return f"🟡 **{conf:.0%}**"
    return f"🔴 **{conf:.0%}**"


def _highlight_diff(raw: str, corrected: str) -> str:
    """Return markdown with per-word differences highlighted in corrected text."""
    raw_tokens = raw.split()
    corr_tokens = corrected.split()
    out = []
    for i, tok in enumerate(corr_tokens):
        if i >= len(raw_tokens) or raw_tokens[i] != tok:
            out.append(f"**:green[{tok}]**")
        else:
            out.append(tok)
    return " ".join(out)


def render_result(result: PredictionResult, key_prefix: str) -> None:
    if not result.text:
        st.warning("No text detected. Try preprocessing or a clearer image.")
        return
    st.subheader("Prediction")
    st.markdown(f"Confidence: {confidence_badge(result.confidence)}")

    if result.corrected and result.raw_text and result.raw_text != result.text:
        c_raw, c_corr = st.columns(2)
        with c_raw:
            st.caption("Raw (CRNN output)")
            st.code(result.raw_text, language=None)
        with c_corr:
            st.caption("Corrected (dictionary)")
            st.code(result.text, language=None)
        st.markdown(
            "Changed words: " + _highlight_diff(result.raw_text, result.text)
        )
    else:
        st.code(result.text, language=None)  # built-in copy button

    if result.line_results:
        with st.expander("Per-line breakdown"):
            for i, lr in enumerate(result.line_results, 1):
                line_display = lr.text
                if lr.corrected and lr.raw_text and lr.raw_text != lr.text:
                    line_display = f"{lr.raw_text}  →  {lr.text}"
                st.markdown(
                    f"**Line {i}** &middot; {confidence_badge(lr.confidence)} &middot; `{line_display}`",
                    unsafe_allow_html=True,
                )


def run_pipeline(
    image: Image.Image,
    model_choice: str,
    multiline: bool,
    apply_preproc: bool,
    opts: PreprocessOptions,
    key_prefix: str,
    apply_correction: bool = False,
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

    corrector = None
    if apply_correction and model_choice == MLTU_LABEL:
        try:
            corrector = _load_corrector()
        except Exception as e:  # pragma: no cover
            st.warning(f"Could not load dictionary corrector: {e}")

    try:
        with st.spinner("Recognizing..."):
            if corrector is not None:
                result = (
                    rec.predict_lines(processed, corrector=corrector)
                    if multiline
                    else rec.predict(processed, corrector=corrector)
                )
            else:
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
        options=[TROCR_LABEL, TROCR_PRINTED_LABEL, MLTU_LABEL],
        index=0,
        help=(
            "TrOCR (handwritten): line-level, strong on cursive paragraphs.  "
            "TrOCR (printed): same architecture trained on printed text — use "
            "for scanned documents.  "
            "mltu CRNN: word-level, fast, trained locally on IAM cursive."
        ),
    )
    multiline = st.checkbox(
        "Multi-line mode", value=False, help="Split image into horizontal lines."
    )
    is_mltu = model_choice == MLTU_LABEL
    apply_correction = st.checkbox(
        "English-dictionary correction",
        value=False,
        disabled=not is_mltu,
        help=(
            "Snap out-of-vocabulary CRNN output to the nearest English word, "
            "weighted by CTC per-character confidence. mltu backend only."
        ),
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

upload_tab, draw_tab, url_tab = st.tabs(
    ["📁 Upload", "✏️ Draw", "🔗 URL / paste"]
)

with upload_tab:
    uploaded = st.file_uploader("Choose an image", type=["png", "jpg", "jpeg"])
    if uploaded is not None:
        try:
            image = Image.open(uploaded)
        except Exception as e:
            st.error(f"Could not open image: {e}")
        else:
            st.image(image, caption="Input", use_container_width=True)
            run_pipeline(
                image,
                model_choice,
                multiline,
                apply_preproc,
                opts,
                key_prefix="upload",
                apply_correction=apply_correction,
            )

with draw_tab:
    st.caption(
        "Draw with your mouse, stylus, or finger. Click **Use this drawing** "
        "in the canvas toolbar to send it to the recognizer."
    )
    # The component returns the most recent PNG data URL the user submitted,
    # or None if nothing has been clicked yet. We re-run on every rerun
    # (matching the upload tab) so the result panel survives unrelated
    # widget changes; the recognizer is the only expensive step and a hash
    # of the URL would still be re-decoded each time anyway.
    canvas_data_url = _handwriting_canvas(key="canvas", default=None)
    if (
        canvas_data_url
        and isinstance(canvas_data_url, str)
        and canvas_data_url.startswith("data:image/")
    ):
        try:
            image = load_image_from_url(canvas_data_url)
        except ImageLoadError as e:
            st.error(str(e))
        else:
            run_pipeline(
                image,
                model_choice,
                multiline,
                apply_preproc,
                opts,
                key_prefix="draw",
                apply_correction=apply_correction,
            )

with url_tab:
    # Two parallel inputs in this tab: paste from system clipboard (via the
    # paste-button component) and paste/type a URL. The clipboard component
    # is optional — we fall back gracefully if the plugin isn't installed.
    paste_result = None
    try:
        from streamlit_paste_button import paste_image_button
    except ImportError:
        st.caption(
            "Tip: install `streamlit-paste-button` to enable clipboard paste."
        )
    else:
        paste_result = paste_image_button(
            label="📋 Paste image from clipboard",
            key="paste_btn",
            errors="ignore",
        )

    pasted_image: Image.Image | None = None
    if paste_result is not None and getattr(paste_result, "image_data", None) is not None:
        pasted_image = paste_result.image_data
        if pasted_image.mode != "RGB":
            pasted_image = pasted_image.convert("RGB")

    url = st.text_input(
        "…or paste an image URL / data URI",
        key="url_input",
        placeholder="https://example.com/note.png  or  data:image/png;base64,…",
    )

    url_image: Image.Image | None = None
    if url:
        try:
            with st.spinner("Fetching image…"):
                url_image = load_image_from_url(url)
        except ImageLoadError as e:
            st.error(str(e))

    # Recognize whichever was provided most recently. If both arrived in
    # the same rerun, prefer the clipboard paste — it's the more "fresh"
    # gesture and the URL field is sticky between reruns.
    image = pasted_image or url_image
    if image is not None:
        st.image(image, caption="Input", use_container_width=True)
        run_pipeline(
            image,
            model_choice,
            multiline,
            apply_preproc,
            opts,
            key_prefix="url",
            apply_correction=apply_correction,
        )
