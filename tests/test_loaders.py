"""Unit tests for `src.loaders` (URL + data-URI image fetcher).

We exercise the public surface only; the internal helpers (`_pil_from_bytes`,
`_flatten_to_rgb`, ...) get covered transitively. Network calls are stubbed
via monkeypatched `requests.get` so the tests stay hermetic.

The drawable-canvas tab routes through ``load_image_from_url`` too, since the
HTML component sends its strokes as a ``data:image/png;base64,...`` payload.
"""
from __future__ import annotations

import base64
import io
from pathlib import Path
from typing import Optional

import pytest
from PIL import Image

# Make the repo root importable when pytest is invoked from anywhere.
REPO_ROOT = Path(__file__).resolve().parents[1]
import sys

sys.path.insert(0, str(REPO_ROOT))

from src.loaders import (  # noqa: E402  (sys.path mutation above)
    ImageLoadError,
    load_image_from_url,
)


# ---------- fixtures ----------


def _png_bytes(color=(10, 200, 50), size=(8, 6)) -> bytes:
    """Encode a tiny solid-color PNG into bytes."""
    img = Image.new("RGB", size, color)
    buf = io.BytesIO()
    img.save(buf, format="PNG")
    return buf.getvalue()


def _png_data_uri(color=(10, 200, 50), size=(8, 6)) -> str:
    b64 = base64.b64encode(_png_bytes(color, size)).decode("ascii")
    return f"data:image/png;base64,{b64}"


class _FakeResponse:
    """Minimal `requests.Response` stand-in supporting iter_content."""

    def __init__(self, body: bytes, status_code: int = 200):
        self._body = body
        self.status_code = status_code

    def iter_content(self, chunk_size: int = 1024):
        for i in range(0, len(self._body), chunk_size):
            yield self._body[i : i + chunk_size]


def _install_fake_get(monkeypatch, response: Optional[_FakeResponse], *, raise_exc=None):
    """Patch `requests.get` inside `src.loaders` to return `response`.

    If `raise_exc` is provided, the patched `get` raises it instead.
    """
    import src.loaders as loaders_mod

    def fake_get(url, timeout=10, stream=False):  # noqa: ARG001 (kw match real signature)
        if raise_exc is not None:
            raise raise_exc
        return response

    monkeypatch.setattr(loaders_mod.requests, "get", fake_get)


# ---------- load_image_from_url: data URIs ----------


def test_data_uri_base64_png_round_trips():
    img = load_image_from_url(_png_data_uri(color=(10, 200, 50), size=(4, 4)))
    assert img.size == (4, 4)
    assert img.mode == "RGB"
    # The exact pixel should survive PNG round-trip.
    assert img.getpixel((0, 0)) == (10, 200, 50)


def test_data_uri_strips_whitespace():
    """Users paste with stray newlines surprisingly often — be lenient."""
    uri = _png_data_uri()
    img = load_image_from_url("   " + uri + "\n")
    assert img.size == (8, 6)


def test_data_uri_rejects_non_base64():
    with pytest.raises(ImageLoadError, match="base64"):
        load_image_from_url("data:image/png,not-base64-encoded")


def test_data_uri_rejects_malformed():
    with pytest.raises(ImageLoadError, match="comma"):
        load_image_from_url("data:image/png;base64-no-comma-here")


def test_data_uri_rejects_invalid_base64_payload():
    with pytest.raises(ImageLoadError, match="base64"):
        # `!!!` is not valid base64 (decodes to garbage, then PIL fails on
        # the resulting non-image bytes — both paths raise ImageLoadError).
        load_image_from_url("data:image/png;base64,!!!")


def test_data_uri_rejects_valid_base64_that_isnt_image():
    payload = base64.b64encode(b"not an image, just text").decode("ascii")
    with pytest.raises(ImageLoadError, match="not a valid image"):
        load_image_from_url(f"data:image/png;base64,{payload}")


# ---------- load_image_from_url: HTTP ----------


def test_http_url_success(monkeypatch):
    _install_fake_get(monkeypatch, _FakeResponse(_png_bytes(size=(10, 10))))
    img = load_image_from_url("https://example.com/note.png")
    assert img.size == (10, 10)
    assert img.mode == "RGB"


def test_http_url_non_200(monkeypatch):
    _install_fake_get(monkeypatch, _FakeResponse(b"", status_code=404))
    with pytest.raises(ImageLoadError, match="HTTP 404"):
        load_image_from_url("https://example.com/missing.png")


def test_http_url_request_exception(monkeypatch):
    import requests

    _install_fake_get(monkeypatch, None, raise_exc=requests.ConnectionError("nope"))
    with pytest.raises(ImageLoadError, match="Could not fetch"):
        load_image_from_url("https://example.com/dead")


def test_http_url_size_cap(monkeypatch):
    """Body exceeding MAX_DOWNLOAD_BYTES raises before PIL ever sees it."""
    import src.loaders as loaders_mod

    monkeypatch.setattr(loaders_mod, "MAX_DOWNLOAD_BYTES", 100)
    _install_fake_get(monkeypatch, _FakeResponse(b"x" * 500))
    with pytest.raises(ImageLoadError, match="exceeds"):
        load_image_from_url("https://example.com/big")


# ---------- load_image_from_url: validation ----------


def test_empty_url_rejected():
    with pytest.raises(ImageLoadError, match="Empty"):
        load_image_from_url("")


def test_none_url_rejected():
    with pytest.raises(ImageLoadError, match="Empty"):
        load_image_from_url(None)  # type: ignore[arg-type]


def test_unsupported_scheme_rejected():
    with pytest.raises(ImageLoadError, match="scheme"):
        load_image_from_url("ftp://example.com/note.png")


def test_file_scheme_rejected():
    """File URLs in particular must be rejected — the URL tab is for remote
    content, allowing `file://` would let a deployed app read its own disk."""
    with pytest.raises(ImageLoadError, match="scheme"):
        load_image_from_url("file:///etc/passwd")


# ---------- canvas-via-data-URI integration ----------


def test_transparent_png_flattens_onto_white():
    """The drawable-canvas component exports PNGs with an alpha channel.
    Even when the user fills the background opaque, screenshot pastes and
    clipboard-snipped images often have transparency. Flattening must
    composite black-on-transparent into black-on-white so the recognizer
    doesn't see an all-white image after alpha drop."""
    img = Image.new("RGBA", (4, 4), (0, 0, 0, 0))  # fully transparent
    img.putpixel((2, 2), (0, 0, 0, 255))  # one opaque black "stroke" pixel
    buf = io.BytesIO()
    img.save(buf, format="PNG")
    uri = "data:image/png;base64," + base64.b64encode(buf.getvalue()).decode()

    loaded = load_image_from_url(uri)
    assert loaded.mode == "RGB"
    assert loaded.getpixel((0, 0)) == (255, 255, 255)  # transparent -> white
    assert loaded.getpixel((2, 2)) == (0, 0, 0)  # stroke preserved
