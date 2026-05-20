"""Shared scoring helpers for `eval_iam.py` and `eval_archive.py`.

Kept intentionally narrow — just CER + match — because the two scripts have
materially different per-sample print formats (correction diff vs. side-by-side
backend comparison), so a single `_run_eval()` would be a leaky abstraction.
"""
from __future__ import annotations

import difflib
from typing import Iterable


def normalize(text: str, *, ignore_case: bool) -> str:
    """Strip whitespace; optionally lower-case for case-insensitive compare."""
    s = (text or "").strip()
    return s.lower() if ignore_case else s


def exact_match(pred: str, truth: str, *, ignore_case: bool = False) -> bool:
    return normalize(pred, ignore_case=ignore_case) == normalize(truth, ignore_case=ignore_case)


def cer(pred: str, truth: str, *, ignore_case: bool = False) -> float:
    """Character error rate = edit_distance(pred, truth) / max(len(truth), 1).

    Uses `python-Levenshtein` if available (transitive dep via `mltu`), else
    falls back to `difflib.SequenceMatcher` which computes a similar metric
    via `ratio()`. Both give the same answer in the limit of small edits.
    """
    p = normalize(pred, ignore_case=ignore_case)
    t = normalize(truth, ignore_case=ignore_case)
    denom = max(len(t), 1)
    try:
        import Levenshtein  # type: ignore

        return Levenshtein.distance(p, t) / denom
    except ImportError:
        # SequenceMatcher.ratio() = 2 * matches / (len(a)+len(b)); convert.
        if not p and not t:
            return 0.0
        ratio = difflib.SequenceMatcher(None, p, t).ratio()
        matches = ratio * (len(p) + len(t)) / 2
        edits = max(len(p), len(t)) - matches
        return edits / denom


def mean_cer(preds: Iterable[str], truths: Iterable[str], *, ignore_case: bool = False) -> float:
    cers = [cer(p, t, ignore_case=ignore_case) for p, t in zip(preds, truths)]
    return sum(cers) / len(cers) if cers else 0.0
