"""TrOCR-based handwriting recognizer.

Loads `microsoft/trocr-base-handwritten` once and exposes:
    Recognizer().predict(pil_image) -> PredictionResult
    Recognizer().predict_lines(pil_image) -> PredictionResult
    line_polygons(pil_image) -> list of 4x2 int polygons for preview overlays
"""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import List, Optional, Tuple

import cv2
import numpy as np
import torch
from PIL import Image
from transformers import TrOCRProcessor, VisionEncoderDecoderModel

from .segment import line_polygons as _segment_line_polygons
from .segment import split_lines as _segment_split_lines

MODEL_NAME = "microsoft/trocr-base-handwritten"


@dataclass
class PredictionResult:
    text: str
    confidence: float  # 0..1, mean per-token probability
    line_results: List["PredictionResult"] = field(default_factory=list)
    raw_text: Optional[str] = None  # uncorrected text if post-processing ran
    corrected: bool = False  # True iff `raw_text` differs from `text`

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
        crops = [Image.fromarray(c) for c in _segment_split_lines(image)]
        if not crops:
            return self.predict(image)
        line_results = [self.predict(c) for c in crops]
        text = "\n".join(r.text for r in line_results if r.text)
        non_empty = [r.confidence for r in line_results if r.text]
        confidence = float(np.mean(non_empty)) if non_empty else 0.0
        return PredictionResult(text=text, confidence=confidence, line_results=line_results)


# ---------- helpers ----------

def line_polygons(image: Image.Image) -> List[np.ndarray]:
    """Return list of 4x2 int polygons for line overlay drawing."""
    return _segment_line_polygons(image)


# Backwards-compat shim: older callers imported `line_boxes` expecting (y0, y1).
def line_boxes(image: Image.Image) -> List[Tuple[int, int]]:
    polys = _segment_line_polygons(image)
    return [(int(p[:, 1].min()), int(p[:, 1].max())) for p in polys]


def _is_blank(image: Image.Image) -> bool:
    arr = np.array(image.convert("L"))
    if arr.size == 0:
        return True
    _, bw = cv2.threshold(arr, 0, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)
    foreground_ratio = (bw > 0).mean()
    return foreground_ratio < 0.001


def _mean_token_probability(scores, sequences) -> float:
    if not scores:
        return 0.0
    gen_tokens = sequences[0, 1 : 1 + len(scores)]
    probs = []
    for step, logits in enumerate(scores):
        step_probs = torch.softmax(logits[0], dim=-1)
        token_id = gen_tokens[step].item()
        probs.append(step_probs[token_id].item())
    return float(np.mean(probs)) if probs else 0.0
