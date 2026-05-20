"""Pure image-loading helpers shared by the Streamlit UI tabs.

Framework-agnostic (no `streamlit` import) so the recognizer pipeline and
unit tests can call them directly without mounting a session.

Exposed:
    * ``load_image_from_url(url)`` -- HTTP(S) URLs and ``data:image/...``
      URIs; returns a PIL Image (RGB). Also serves the drawable-canvas
      flow, which sends its strokes as a ``data:image/png;base64,...``
      payload through this same helper.
"""
from __future__ import annotations

import base64
import binascii
import io
from typing import Tuple
from urllib.parse import urlparse

import requests
from PIL import Image, UnidentifiedImageError

DEFAULT_TIMEOUT_S = 10
# Cap how much we'll pull from a remote URL. 25 MB easily covers a 4K
# JPEG and still bounds memory + time on a hostile redirect target.
MAX_DOWNLOAD_BYTES = 25 * 1024 * 1024
_ALLOWED_URL_SCHEMES = {"http", "https"}


class ImageLoadError(ValueError):
    """Raised when a URL / data URI can't be turned into an image.

    Subclasses ``ValueError`` so callers that catch broad input-validation
    errors keep working; the specific type makes UI error messages cleaner.
    """


def load_image_from_url(url: str, *, timeout: float = DEFAULT_TIMEOUT_S) -> Image.Image:
    """Fetch an image from an ``http(s)://`` URL or a ``data:image/...`` URI.

    Always returns an RGB PIL image (alpha is flattened onto white). Raises
    :class:`ImageLoadError` on any non-recoverable failure -- the Streamlit
    tab catches that one type and surfaces its message via ``st.error``.
    """
    if not url or not isinstance(url, str):
        raise ImageLoadError("Empty URL.")
    url = url.strip()
    if url.startswith("data:"):
        return _decode_data_uri(url)

    parsed = urlparse(url)
    if parsed.scheme not in _ALLOWED_URL_SCHEMES:
        raise ImageLoadError(
            f"Unsupported URL scheme {parsed.scheme!r}. "
            "Use http(s):// or a data:image/... URI."
        )
    try:
        resp = requests.get(url, timeout=timeout, stream=True)
    except requests.RequestException as e:
        raise ImageLoadError(f"Could not fetch {url}: {e}") from e
    if resp.status_code != 200:
        raise ImageLoadError(f"HTTP {resp.status_code} from {url}.")

    # Stream the body with a hard byte cap so a hostile server can't
    # exhaust memory by advertising a small Content-Length and then
    # sending a much larger response.
    buf = io.BytesIO()
    total = 0
    for chunk in resp.iter_content(chunk_size=64 * 1024):
        if not chunk:
            continue
        total += len(chunk)
        if total > MAX_DOWNLOAD_BYTES:
            raise ImageLoadError(
                f"Image at {url} exceeds {MAX_DOWNLOAD_BYTES // (1024 * 1024)} MB cap."
            )
        buf.write(chunk)
    return _pil_from_bytes(buf.getvalue(), origin=url)


def _decode_data_uri(uri: str) -> Image.Image:
    """Decode ``data:image/png;base64,...`` (and similar) into a PIL image.

    Plain (non-base64) data URIs are rejected -- in practice every browser
    clipboard or canvas export uses base64, and parsing the percent-encoded
    variant is more code than the rare benefit warrants.
    """
    try:
        header, payload = uri.split(",", 1)
    except ValueError as e:
        raise ImageLoadError("Malformed data URI (missing comma).") from e
    if ";base64" not in header:
        raise ImageLoadError("Only base64-encoded data URIs are supported.")
    # `validate=True` makes `b64decode` reject stray non-base64 characters
    # instead of silently discarding them (which would otherwise hand
    # callers an `Empty image payload` error and obscure the real cause).
    try:
        raw = base64.b64decode(payload, validate=True)
    except (binascii.Error, ValueError) as e:
        raise ImageLoadError(f"Invalid base64 payload: {e}") from e
    return _pil_from_bytes(raw, origin="data URI")


def _pil_from_bytes(data: bytes, *, origin: str) -> Image.Image:
    if not data:
        raise ImageLoadError(f"Empty image payload from {origin}.")
    try:
        img = Image.open(io.BytesIO(data))
        img.load()
    except (UnidentifiedImageError, OSError) as e:
        raise ImageLoadError(f"Bytes from {origin} are not a valid image: {e}") from e
    return _flatten_to_rgb(img)


def _flatten_to_rgb(
    img: Image.Image, *, bg_color: Tuple[int, int, int] = (255, 255, 255)
) -> Image.Image:
    """Drop alpha by compositing onto an opaque background.

    PNGs with transparent backgrounds are surprisingly common for clipboard
    paste (e.g., dragging a screenshot from a markdown preview). Without
    flattening, OpenCV's `cvtColor(..., RGB2GRAY)` interprets the alpha
    channel as luminance and the recognizer reads pure white.
    """
    if img.mode == "RGB":
        return img
    if img.mode in ("RGBA", "LA"):
        bg = Image.new("RGB", img.size, bg_color)
        bg.paste(img, mask=img.split()[-1])
        return bg
    return img.convert("RGB")
