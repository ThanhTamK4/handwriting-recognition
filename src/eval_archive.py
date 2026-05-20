"""Benchmark recognizers on the Kaggle "Handwriting Recognition" dataset
(handwritten names, indexed by `CSV/written_name_test.csv` with rows
`FILENAME,IDENTITY`).

Answers the "does it generalize?" question: mltu is trained only on IAM Words,
so this measures cross-dataset transfer.

Usage::

    # archive lives outside the repo by default
    python -m src.eval_archive --n 100 --model mltu
    python -m src.eval_archive --n 100 --model trocr
    python -m src.eval_archive --n 100 --model both --ignore-case
    python -m src.eval_archive --n 100 --model both \
        --archive-dir "C:/Users/qwert/OneDrive/Máy tính/archive"
"""
from __future__ import annotations

import argparse
import csv
import os
import random
from pathlib import Path
from typing import List, Tuple

from PIL import Image

from ._eval_utils import cer, exact_match

DEFAULT_ARCHIVE_DIR = Path(os.path.expanduser("~/OneDrive/Máy tính/archive"))


def _resolve_split(archive_dir: Path, split: str) -> Tuple[Path, Path]:
    """Return (csv_path, image_dir) for the requested split.

    The Kaggle dataset uses these layouts:
        archive/CSV/written_name_{test,train,validation}.csv
        archive/{test,train,validation}_v2/{test,train,validation}/<FILE>.jpg
    """
    csv_path = archive_dir / "CSV" / f"written_name_{split}.csv"
    image_dir = archive_dir / f"{split}_v2" / split
    return csv_path, image_dir


def load_samples(archive_dir: Path, split: str) -> List[Tuple[Path, str]]:
    csv_path, image_dir = _resolve_split(archive_dir, split)
    if not csv_path.exists():
        raise FileNotFoundError(f"CSV not found: {csv_path}")
    if not image_dir.exists():
        raise FileNotFoundError(f"Image dir not found: {image_dir}")

    samples: List[Tuple[Path, str]] = []
    with open(csv_path, "r", encoding="utf-8", newline="") as fh:
        reader = csv.DictReader(fh)
        for row in reader:
            fname = (row.get("FILENAME") or "").strip()
            label = (row.get("IDENTITY") or "").strip()
            if not fname or not label:
                continue
            # Skip the dataset's "UNREADABLE" sentinels.
            if label.upper() == "UNREADABLE":
                continue
            path = image_dir / fname
            if path.exists() and path.stat().st_size > 0:
                samples.append((path, label))
    return samples


def _open(path: Path) -> Image.Image:
    img = Image.open(path)
    img.load()
    return img


def _run_single(rec, samples, label, ignore_case: bool) -> None:
    """Run one backend over all samples, print per-row + summary."""
    correct = 0
    cers: List[float] = []
    for path, truth in samples:
        result = rec.predict(_open(path))
        pred = result.text
        ok = exact_match(pred, truth, ignore_case=ignore_case)
        correct += int(ok)
        cers.append(cer(pred, truth, ignore_case=ignore_case))
        mark = "OK " if ok else "   "
        print(f"{mark} truth={truth!r:20} pred={pred!r:20} conf={result.confidence:.2f}")
    total = len(samples)
    mean_cer = sum(cers) / len(cers) if cers else 0.0
    print(f"\n{label}: {correct}/{total} exact-match  |  CER {mean_cer:.3f}")


def _run_both(rec_mltu, rec_trocr, samples, ignore_case: bool) -> None:
    """Run both backends side-by-side, print per-row + summary."""
    mltu_correct = 0
    trocr_correct = 0
    mltu_cers: List[float] = []
    trocr_cers: List[float] = []
    print(
        f"{'truth':<20} | {'mltu pred':<20} | {'trocr pred':<20} | mltu | trocr"
    )
    print("-" * 84)
    for path, truth in samples:
        img = _open(path)
        m_res = rec_mltu.predict(img)
        t_res = rec_trocr.predict(img)
        m_ok = exact_match(m_res.text, truth, ignore_case=ignore_case)
        t_ok = exact_match(t_res.text, truth, ignore_case=ignore_case)
        mltu_correct += int(m_ok)
        trocr_correct += int(t_ok)
        mltu_cers.append(cer(m_res.text, truth, ignore_case=ignore_case))
        trocr_cers.append(cer(t_res.text, truth, ignore_case=ignore_case))
        print(
            f"{truth!r:<20} | {m_res.text!r:<20} | {t_res.text!r:<20} | "
            f"{'OK' if m_ok else '  '}   | {'OK' if t_ok else '  '}"
        )
    total = len(samples)
    print(
        f"\nmltu : {mltu_correct}/{total} exact-match  |  CER "
        f"{sum(mltu_cers)/len(mltu_cers):.3f}"
    )
    print(
        f"trocr: {trocr_correct}/{total} exact-match  |  CER "
        f"{sum(trocr_cers)/len(trocr_cers):.3f}"
    )


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--n", type=int, default=100)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--model", choices=["trocr", "mltu", "both"], default="both")
    ap.add_argument("--split", choices=["test", "validation", "train"], default="test")
    ap.add_argument(
        "--archive-dir",
        type=Path,
        default=DEFAULT_ARCHIVE_DIR,
        help="Path to the archive/ folder containing CSV/ and *_v2/.",
    )
    ap.add_argument(
        "--ignore-case",
        action="store_true",
        help="Case-insensitive match. Recommended since names are usually ALL-CAPS.",
    )
    args = ap.parse_args()

    samples = load_samples(args.archive_dir, args.split)
    random.Random(args.seed).shuffle(samples)
    samples = samples[: args.n]
    if not samples:
        raise SystemExit(
            f"No samples loaded from {args.archive_dir}; check --archive-dir."
        )
    print(
        f"Loaded {len(samples)} samples from {args.archive_dir} "
        f"(split={args.split}, ignore_case={args.ignore_case})\n"
    )

    if args.model in ("mltu", "both"):
        from .mltu_recognizer import MltuRecognizer

        rec_mltu = MltuRecognizer()
    if args.model in ("trocr", "both"):
        from .recognizer import Recognizer

        rec_trocr = Recognizer()

    if args.model == "mltu":
        _run_single(rec_mltu, samples, "mltu", args.ignore_case)
    elif args.model == "trocr":
        _run_single(rec_trocr, samples, "trocr", args.ignore_case)
    else:
        _run_both(rec_mltu, rec_trocr, samples, args.ignore_case)


if __name__ == "__main__":
    main()
