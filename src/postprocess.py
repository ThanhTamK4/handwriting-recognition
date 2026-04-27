"""English-dictionary word-correction post-processor for CRNN output.

Wraps SymSpell (https://github.com/wolfgarbe/SymSpell) with:
    * case-pattern + punctuation preservation
    * CTC-confidence-weighted acceptance so high-confidence characters
      are never overwritten by a "nearest word" heuristic.

Typical use::

    from src.postprocess import WordCorrector
    corrector = WordCorrector()                 # auto-detects dictionary
    new_text, flags = corrector.correct_text("rnove dover h0me")
    #  -> ("move over home", [True, True, True])
"""
from __future__ import annotations

import string
import warnings
from pathlib import Path
from typing import Iterable, List, Optional, Sequence, Tuple

try:
    from symspellpy import SymSpell, Verbosity
except ImportError as e:  # pragma: no cover
    raise ImportError(
        "symspellpy is required for src.postprocess. "
        "Install with `pip install symspellpy>=6.7`."
    ) from e

_DICT_DIR = Path(__file__).resolve().parents[1] / "data" / "english_dictionary"
# Preferred: SymSpell's official 500k frequency dictionary (word + count per line).
# Fallback: a flat word list (one word per line, e.g. Kaggle 479k).
_FREQ_DICT_PATH = _DICT_DIR / "frequency.txt"
_FLAT_DICT_PATH = _DICT_DIR / "words.txt"
DEFAULT_DICT_PATH = _FREQ_DICT_PATH if _FREQ_DICT_PATH.exists() else _FLAT_DICT_PATH


class WordCorrector:
    """Dictionary-backed spell corrector tuned for single-word CRNN output."""

    def __init__(
        self,
        dict_path: Path | str | None = None,
        max_edit_distance: int = 3,
        conf_threshold: float = 0.99,
    ) -> None:
        self.max_edit_distance = max_edit_distance
        self.conf_threshold = conf_threshold
        self._symspell = SymSpell(
            max_dictionary_edit_distance=max_edit_distance,
            prefix_length=7,
        )
        self._known: set[str] = set()
        self._load(Path(dict_path) if dict_path else DEFAULT_DICT_PATH)

    # ---------- public API ----------

    def correct_word(
        self, word: str, char_confs: Optional[Sequence[float]] = None
    ) -> Tuple[str, bool]:
        """Return (possibly-corrected word, did_change)."""
        if not word:
            return word, False

        # Contractions / possessives ("don't", "Delaney's") aren't reliably in
        # flat dictionaries; SymSpell tends to "fix" them by deleting the 's.
        # Trust the CRNN's output for any apostrophe-bearing token.
        if "'" in word or "’" in word:
            return word, False

        lead, core, trail = _strip_affixes(word)
        if not core:
            return word, False

        core_lower = core.lower()
        if core_lower in self._known:
            return word, False  # already a valid dictionary word

        suggestion = self._best_suggestion(core_lower)
        if suggestion is None or suggestion == core_lower:
            return word, False

        # Confidence gate: if the characters being changed were high-confidence,
        # keep the original. `char_confs` aligns with the stripped core.
        if char_confs is not None:
            core_confs = _confs_for_core(word, core, char_confs)
            if not _should_accept(core_lower, suggestion, core_confs, self.conf_threshold):
                return word, False

        restored = _restore_case(suggestion, core)
        return f"{lead}{restored}{trail}", True

    def correct_text(
        self,
        text: str,
        per_word_confs: Optional[Sequence[Sequence[float]]] = None,
    ) -> Tuple[str, List[bool]]:
        """Split on whitespace, correct each token, rejoin.

        `per_word_confs[i]` is the list of per-character CTC confidences for
        token i (including any leading/trailing punctuation — affixes are
        skipped internally).
        """
        tokens = text.split()
        out: List[str] = []
        flags: List[bool] = []
        for i, tok in enumerate(tokens):
            confs = per_word_confs[i] if per_word_confs and i < len(per_word_confs) else None
            new, changed = self.correct_word(tok, confs)
            out.append(new)
            flags.append(changed)
        return " ".join(out), flags

    # ---------- internals ----------

    def _load(self, dict_path: Path) -> None:
        """Load words + frequencies into self._symspell.

        Resolution order:
          1. If `dict_path` is a frequency dict (`word<sep>count` per line),
             load it directly. This is the preferred path — SymSpell's
             official 500k freq dict gives real frequencies for tie-breaking.
          2. Otherwise, load the bundled 82k frequency dict for ranking, then
             layer a flat word list on top for extra coverage at count=1.
          3. If the path doesn't exist, fall back to bundled-only.
        """
        if dict_path.exists() and _looks_like_freq_dict(dict_path):
            # utf-8-sig strips a leading BOM if present (the upstream 500k
            # SymSpell file ships with one, which would otherwise corrupt the
            # first word — `the` → `﻿the`).
            self._symspell.load_dictionary(
                str(dict_path), term_index=0, count_index=1, encoding="utf-8-sig"
            )
            self._known = set(self._symspell.words.keys())
            return

        # Always load the bundled frequency dictionary first — it gives SymSpell
        # real word-frequency counts to rank tie-edit-distance suggestions
        # (`was` beats `ass` for input `aas`, `the` beats `tho`, etc.).
        self._load_bundled()
        # Then layer the flat word list on top for additional coverage.
        # Each unseen word gets a low frequency (1) so it loses ranking ties to
        # bundled high-frequency words but is still reachable as a fallback.
        if dict_path.exists():
            extra = self._load_flat(dict_path)
            if extra == 0:
                warnings.warn(f"Dictionary at {dict_path} was empty; bundled list only.")
        else:
            warnings.warn(
                f"Dictionary not found at {dict_path}; using bundled list only. "
                f"For better coverage, drop SymSpell's official 500k frequency "
                f"dict at {_FREQ_DICT_PATH} (preferred) or a flat word list at "
                f"{_FLAT_DICT_PATH}."
            )

    def _load_flat(self, path: Path) -> int:
        """Layer a newline-separated word list (Kaggle format) on top of any
        already-loaded entries. New words get count=1 so bundled-frequency
        words win ranking ties; pre-existing words keep their bundled count.
        Returns the number of *new* words added.
        """
        added = 0
        with open(path, "r", encoding="utf-8") as fh:
            for raw in fh:
                word = raw.strip().lower()
                if not word or not word.isalpha():
                    continue
                if word in self._known:
                    continue  # already in bundled dict with proper frequency
                self._symspell.create_dictionary_entry(word, 1)
                self._known.add(word)
                added += 1
        return added

    def _load_bundled(self) -> None:
        import importlib.resources as resources

        with resources.as_file(
            resources.files("symspellpy") / "frequency_dictionary_en_82_765.txt"
        ) as bundled:
            self._symspell.load_dictionary(str(bundled), term_index=0, count_index=1)
        self._known = set(self._symspell.words.keys())

    def _best_suggestion(self, word: str) -> Optional[str]:
        suggestions = self._symspell.lookup(
            word, Verbosity.TOP, max_edit_distance=self.max_edit_distance
        )
        if not suggestions:
            return None
        return suggestions[0].term


# ---------- helpers ----------

_PUNCT = set(string.punctuation)


def _looks_like_freq_dict(path: Path, sample_lines: int = 20) -> bool:
    """Sniff the first non-empty lines: if at least half are `<word> <int>`
    (SymSpell freq-dict format), treat the file as a frequency dictionary.
    Otherwise it's a flat word list (one word per line).
    """
    hits = 0
    looked = 0
    try:
        with open(path, "r", encoding="utf-8", errors="ignore") as fh:
            for raw in fh:
                line = raw.strip()
                if not line:
                    continue
                looked += 1
                parts = line.split()
                if len(parts) >= 2 and parts[1].isdigit():
                    hits += 1
                if looked >= sample_lines:
                    break
    except OSError:
        return False
    return looked > 0 and hits >= max(1, looked // 2)


def _strip_affixes(word: str) -> Tuple[str, str, str]:
    """Return (leading-punct, alnum-core, trailing-punct)."""
    start = 0
    end = len(word)
    while start < end and word[start] in _PUNCT:
        start += 1
    while end > start and word[end - 1] in _PUNCT:
        end -= 1
    return word[:start], word[start:end], word[end:]


def _restore_case(corrected_lower: str, original_core: str) -> str:
    """Mirror original_core's case pattern onto corrected_lower."""
    if original_core.isupper():
        return corrected_lower.upper()
    if original_core[:1].isupper() and original_core[1:].islower():
        return corrected_lower.capitalize()
    # Character-wise mirror for mixed case; fall back to original casing for
    # positions beyond the corrected word's length.
    out = []
    for i, ch in enumerate(corrected_lower):
        if i < len(original_core) and original_core[i].isupper():
            out.append(ch.upper())
        else:
            out.append(ch)
    return "".join(out)


def _confs_for_core(
    whole_word: str, core: str, char_confs: Sequence[float]
) -> Sequence[float]:
    """Slice `char_confs` to the positions covered by `core`.

    If lengths don't line up (rare: stripped punct not in confs), fall back to
    the full list so the caller still has something to threshold against.
    """
    if len(char_confs) != len(whole_word):
        return char_confs
    start = whole_word.find(core)
    if start < 0:
        return char_confs
    return char_confs[start : start + len(core)]


def _should_accept(
    original: str,
    suggestion: str,
    char_confs: Sequence[float],
    threshold: float,
) -> bool:
    """Only accept the correction if the characters being changed were
    low-confidence. Equal-length diff uses positional comparison; length
    mismatches fall back to mean confidence of the whole word."""
    if not char_confs:
        return True
    if len(original) == len(suggestion) and len(char_confs) == len(original):
        diff_idx = [i for i, (a, b) in enumerate(zip(original, suggestion)) if a != b]
        if not diff_idx:
            return True
        mean_diff_conf = sum(char_confs[i] for i in diff_idx) / len(diff_idx)
        return mean_diff_conf < threshold
    # Length mismatch (insertion/deletion): gate on overall word confidence.
    return (sum(char_confs) / len(char_confs)) < threshold


def iter_known_words(corrector: WordCorrector) -> Iterable[str]:  # pragma: no cover
    """Debug helper."""
    return iter(corrector._known)
