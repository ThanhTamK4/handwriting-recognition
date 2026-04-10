"""TrOCR-based handwriting recognizer.

Loads `microsoft/trocr-base-handwritten` once and exposes:
    Recognizer().predict(pil_image) -> PredictionResult
    Recognizer().predict_lines(pil_image) -> PredictionResult
    line_boxes(pil_image) -> list of (y0, y1) tuples for preview overlays
"""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import List, Tuple

import cv2
import numpy as np
import torch
from PIL import Image
from transformers import TrOCRProcessor, VisionEncoderDecoderModel

MODEL_NAME = "microsoft/trocr-base-handwritten"

# Anything below this fraction of the brightest row is considered "background".
LINE_THRESHOLD_RATIO = 0.05
MIN_LINE_HEIGHT = 10
LINE_PAD = 6


@dataclass
class PredictionResult:
    text: str
    confidence: float  # 0..1, mean per-token probability
    line_results: List["PredictionResult"] = field(default_factory=list)

    def __str__(self) -> str:  # convenience for st.text_area
        return self.text


class Recognizer:
    def __init__(self, model_name: str = MODEL_NAME):
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        self.processor = TrOCRProcessor.from_pretrained(model_name)
        self.model = VisionEncoderDecoderModel.from_pretrained(model_name).to(self.device)
        self.model.eval()

    @torch.inference_mode()
    def predict(self, image: Image.Image) -> PredictionResult:
        if image.mode != "RGB":
            image = image.convert("RGB")
        if _is_blank(image):
            return PredictionResult(text="", confidence=0.0)

        pixel_values = self.processor(images=image, return_tensors="pt").pixel_values.to(self.device)
        output = self.model.generate(
            pixel_values,
            max_new_tokens=64,
            return_dict_in_generate=True,
            output_scores=True,
        )
        text = self.processor.batch_decode(output.sequences, skip_special_tokens=True)[0].strip()
        confidence = _mean_token_probability(output.scores, output.sequences)
        return PredictionResult(text=text, confidence=confidence)

    def predict_lines(self, image: Image.Image) -> PredictionResult:
        crops = _line_crops(image)
        if not crops:
            return self.predict(image)
        line_results = [self.predict(c) for c in crops]
        text = "\n".join(r.text for r in line_results if r.text)
        non_empty = [r.confidence for r in line_results if r.text]
        confidence = float(np.mean(non_empty)) if non_empty else 0.0
        return PredictionResult(text=text, confidence=confidence, line_results=line_results)


# ---------- helpers ----------

def line_boxes(image: Image.Image) -> List[Tuple[int, int]]:
    """Return list of (y0, y1) horizontal-line spans for preview overlay."""
    rgb = np.array(image.convert("RGB"))
    return _line_spans(rgb)


def _line_crops(image: Image.Image) -> List[Image.Image]:
    rgb = np.array(image.convert("RGB"))
    spans = _line_spans(rgb)
    return [Image.fromarray(rgb[y0:y1, :]) for y0, y1 in spans]


def _line_spans(rgb: np.ndarray) -> List[Tuple[int, int]]:
    gray = cv2.cvtColor(rgb, cv2.COLOR_RGB2GRAY)
    _, bw = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)
    proj = bw.sum(axis=1)
    if proj.max() == 0:
        return []
    threshold = proj.max() * LINE_THRESHOLD_RATIO

    spans: List[Tuple[int, int]] = []
    in_line = False
    start = 0
    h = rgb.shape[0]
    for y, val in enumerate(proj):
        if val > threshold and not in_line:
            in_line = True
            start = y
        elif val <= threshold and in_line:
            in_line = False
            if y - start > MIN_LINE_HEIGHT:
                spans.append((max(0, start - LINE_PAD), min(h, y + LINE_PAD)))
    if in_line and h - start > MIN_LINE_HEIGHT:
        spans.append((max(0, start - LINE_PAD), h))
    return spans


def _is_blank(image: Image.Image) -> bool:
    arr = np.array(image.convert("L"))
    if arr.size == 0:
        return True
    # Image is blank if foreground (dark) pixels < 0.1% of total.
    _, bw = cv2.threshold(arr, 0, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)
    foreground_ratio = (bw > 0).mean()
    return foreground_ratio < 0.001


def _mean_token_probability(scores, sequences) -> float:
    """Compute mean probability of the chosen tokens across generation steps."""
    if not scores:
        return 0.0
    # `sequences` includes the BOS token at index 0; generated tokens start at index 1.
    gen_tokens = sequences[0, 1 : 1 + len(scores)]
    probs = []
    for step, logits in enumerate(scores):
        step_probs = torch.softmax(logits[0], dim=-1)
        token_id = gen_tokens[step].item()
        probs.append(step_probs[token_id].item())
    return float(np.mean(probs)) if probs else 0.0
