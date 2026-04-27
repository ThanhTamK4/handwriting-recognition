"""Sanity-check the recognizer on N random IAM_Words samples.

Usage:
    python -m src.eval_iam --n 20                  # TrOCR (default)
    python -m src.eval_iam --n 20 --model mltu     # mltu CRNN ONNX
"""
from __future__ import annotations

import argparse
import random

from pathlib import Path
from typing import List, Tuple

from PIL import Image

DATA_DIR = Path(__file__).resolve().parents[1] / "data" / "IAM_Words"
WORDS_TXT = DATA_DIR / "words.txt"
WORDS_DIR = DATA_DIR / "words"


def load_samples() -> List[Tuple[Path, str]]:
    samples: List[Tuple[Path, str]] = []
    with open(WORDS_TXT, "r", encoding="utf-8") as fh:
        for line in fh:
            if line.startswith("#") or not line.strip():
                continue
            parts = line.strip().split(" ")
            if len(parts) < 9 or parts[1] != "ok":
                continue
            word_id = parts[0]
            transcription = " ".join(parts[8:])
            a, b, *_ = word_id.split("-")
            img_path = WORDS_DIR / a / f"{a}-{b}" / f"{word_id}.png"
            if img_path.exists() and img_path.stat().st_size > 0:
                samples.append((img_path, transcription))
    return samples


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--n", type=int, default=20)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--model", choices=["trocr", "mltu"], default="trocr")
    ap.add_argument(
        "--correct",
        action="store_true",
        help="Run English-dictionary correction on mltu output and print raw vs corrected.",
    )
    args = ap.parse_args()

    samples = load_samples()
    random.Random(args.seed).shuffle(samples)
    samples = samples[: args.n]

    corrector = None
    if args.correct:
        if args.model != "mltu":
            print("--correct is only supported for --model mltu; ignoring.")
        else:
            from .postprocess import WordCorrector

            corrector = WordCorrector()

    if args.model == "mltu":
        from .mltu_recognizer import MltuRecognizer

        rec = MltuRecognizer()
    else:
        from .recognizer import Recognizer

        rec = Recognizer()

    raw_correct = 0
    corrected_correct = 0
    for path, truth in samples:
        truth = truth.strip()
        if corrector is not None:
            result = rec.predict(Image.open(path), corrector=corrector)
        else:
            result = rec.predict(Image.open(path))
        pred = result.text
        raw_pred = result.raw_text if result.raw_text is not None else pred
        raw_ok = raw_pred == truth
        corr_ok = pred == truth
        raw_correct += int(raw_ok)
        corrected_correct += int(corr_ok)
        if corrector is not None:
            tag = "++" if (corr_ok and not raw_ok) else (
                "--" if (raw_ok and not corr_ok) else ("OK" if corr_ok else "  ")
            )
            print(
                f"{tag} truth={truth!r:25} raw={raw_pred!r:25} "
                f"corr={pred!r:25} conf={result.confidence:.2f}"
            )
        else:
            mark = "OK " if corr_ok else "   "
            print(f"{mark} truth={truth!r:25} pred={pred!r:25} conf={result.confidence:.2f}")

    total = len(samples)
    if corrector is not None:
        print(f"\nraw:       {raw_correct}/{total} exact-match")
        print(f"corrected: {corrected_correct}/{total} exact-match "
              f"(delta {corrected_correct - raw_correct:+d})")
    else:
        print(f"\n{corrected_correct}/{total} exact-match")


if __name__ == "__main__":
    main()
