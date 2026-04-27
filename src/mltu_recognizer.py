"""ONNX-based recognizer for the mltu CRNN trained on IAM_Words.

Depends only on onnxruntime, numpy, opencv, pyyaml, PIL — no TensorFlow.
Exposes the same `PredictionResult` interface as `src.recognizer.Recognizer`
so the Streamlit UI can treat both backends identically.
"""
from __future__ import annotations

from pathlib import Path
from typing import List, Optional

import cv2
import numpy as np
import onnxruntime as ort
import yaml
from PIL import Image

from .recognizer import PredictionResult
from .segment import split_lines_words
from .postprocess import WordCorrector

DEFAULT_MODEL_DIR = Path(__file__).resolve().parents[1] / "models" / "mltu"


class MltuRecognizer:
    def __init__(self, model_dir: Path | str | None = None):
        model_dir = Path(model_dir) if model_dir else DEFAULT_MODEL_DIR
        onnx_path = model_dir / "model.onnx"
        cfg_path = model_dir / "configs.yaml"
        if not onnx_path.exists() or not cfg_path.exists():
            raise FileNotFoundError(
                f"mltu model not found in {model_dir}. Train it first — see training/README.md."
            )

        self.session = ort.InferenceSession(str(onnx_path), providers=["CPUExecutionProvider"])
        self.input_name = self.session.get_inputs()[0].name

        with open(cfg_path, "r", encoding="utf-8") as fh:
            cfg = yaml.safe_load(fh)
        self.vocab: str = cfg["vocab"]
        self.height: int = int(cfg.get("height", 32))
        self.width: int = int(cfg.get("width", 128))
        self.blank_index = len(self.vocab)  # CTC blank is the last class

    # ---------- public API ----------

    def predict(
        self,
        image: Image.Image | np.ndarray,
        corrector: Optional[WordCorrector] = None,
    ) -> PredictionResult:
        prepared = self._prepare(image)
        if prepared is None:
            return PredictionResult(text="", confidence=0.0)
        logits = self.session.run(None, {self.input_name: prepared[None, ...]})[0][0]
        text, conf, char_confs = self._ctc_greedy_decode(logits)
        if corrector is not None and text:
            corrected_text, flags = corrector.correct_text(text, [char_confs])
            if any(flags):
                return PredictionResult(
                    text=corrected_text,
                    confidence=conf,
                    raw_text=text,
                    corrected=True,
                )
            return PredictionResult(text=text, confidence=conf, raw_text=text)
        return PredictionResult(text=text, confidence=conf)

    def predict_lines(
        self,
        image: Image.Image | np.ndarray,
        corrector: Optional[WordCorrector] = None,
    ) -> PredictionResult:
        lines = split_lines_words(image)
        line_results: List[PredictionResult] = []
        for word_crops in lines:
            batch = [self._prepare(c) for c in word_crops]
            batch = [b for b in batch if b is not None]
            if not batch:
                continue
            arr = np.stack(batch, axis=0)
            logits_batch = self.session.run(None, {self.input_name: arr})[0]
            word_texts: List[str] = []
            word_confs: List[float] = []
            word_char_confs: List[List[float]] = []
            for logits in logits_batch:
                t, c, cc = self._ctc_greedy_decode(logits)
                if t:
                    word_texts.append(t)
                    word_confs.append(c)
                    word_char_confs.append(cc)
            if not word_texts:
                continue

            raw_line = " ".join(word_texts)
            line_conf = float(np.mean(word_confs))
            if corrector is not None:
                corrected_line, flags = corrector.correct_text(raw_line, word_char_confs)
                if any(flags):
                    line_results.append(
                        PredictionResult(
                            text=corrected_line,
                            confidence=line_conf,
                            raw_text=raw_line,
                            corrected=True,
                        )
                    )
                    continue
                line_results.append(
                    PredictionResult(text=raw_line, confidence=line_conf, raw_text=raw_line)
                )
            else:
                line_results.append(
                    PredictionResult(text=raw_line, confidence=line_conf)
                )

        text = "\n".join(r.text for r in line_results if r.text)
        confidence = (
            float(np.mean([r.confidence for r in line_results])) if line_results else 0.0
        )
        if corrector is not None and line_results:
            raw_joined = "\n".join(
                (r.raw_text if r.raw_text is not None else r.text)
                for r in line_results
                if (r.raw_text or r.text)
            )
            return PredictionResult(
                text=text,
                confidence=confidence,
                line_results=line_results,
                raw_text=raw_joined,
                corrected=any(r.corrected for r in line_results),
            )
        return PredictionResult(text=text, confidence=confidence, line_results=line_results)

    # ---------- helpers ----------

    def _prepare(self, image) -> Optional[np.ndarray]:
        """Resize + color-convert to the exact tensor the ONNX model expects."""
        if isinstance(image, Image.Image):
            arr = np.array(image.convert("RGB"))
        elif isinstance(image, np.ndarray):
            arr = image
            if arr.ndim == 2:
                arr = cv2.cvtColor(arr, cv2.COLOR_GRAY2RGB)
            elif arr.shape[-1] == 4:
                arr = cv2.cvtColor(arr, cv2.COLOR_RGBA2RGB)
        else:
            return None
        if arr.size == 0:
            return None
        arr = cv2.resize(arr, (self.width, self.height))
        return arr.astype(np.float32)  # model has a /255 Lambda layer inside

    def _ctc_greedy_decode(
        self, logits: np.ndarray
    ) -> tuple[str, float, List[float]]:
        """logits shape: (timesteps, num_classes).

        Returns (text, mean_confidence, per_character_confidences).
        """
        # Model output already has softmax activation, so logits are probabilities.
        # Only apply softmax if values don't already sum to ~1.
        row_sums = logits.sum(axis=-1)
        if np.allclose(row_sums, 1.0, atol=0.1):
            probs = logits
        else:
            probs = _softmax(logits, axis=-1)
        best = probs.argmax(axis=-1)
        best_probs = probs.max(axis=-1)

        chars: List[str] = []
        confidences: List[float] = []
        prev = -1
        for idx, p in zip(best, best_probs):
            if idx != prev and idx != self.blank_index:
                if 0 <= idx < len(self.vocab):
                    chars.append(self.vocab[idx])
                    confidences.append(float(p))
            prev = idx
        text = "".join(chars)
        conf = float(np.mean(confidences)) if confidences else 0.0
        return text, conf, confidences


def _softmax(x: np.ndarray, axis: int = -1) -> np.ndarray:
    x = x - x.max(axis=axis, keepdims=True)
    e = np.exp(x)
    return e / e.sum(axis=axis, keepdims=True)
