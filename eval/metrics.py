"""Scoring metrics for the evaluation sets (TESTS.md §3.2–3.3).

Pure functions, standard library only (no new dependencies, D10):
- normalise(): the fixed Armenian normalisation of TESTS.md §3.2
- cer(), wer(): corpus-level error rates on normalised text
- chrf(): corpus-level chrF, character n-grams 1–6, beta 2, whitespace removed
  (the default definition used by sacrebleu; our own implementation, not
  cross-checked against sacrebleu in Claude's workspace)
"""

from __future__ import annotations

import re
import unicodedata
from collections import Counter
from dataclasses import dataclass

# TESTS.md §3.2 step 3: marks written inside a word, deleted.
_INNER_MARKS = "՞՛՜"  # ՞ question, ՛ emphasis, ՜ exclamation
# Step 4: other punctuation, replaced by a space.
_OTHER_PUNCT = "։՝«»֊" + ".,?!:;\"'()-…"  # ։ ՝ « » ֊ …
_INNER_RE = re.compile(f"[{re.escape(_INNER_MARKS)}]")
_PUNCT_RE = re.compile(f"[{re.escape(_OTHER_PUNCT)}]")
_SPACE_RE = re.compile(r"\s+")


def normalise(text: str) -> str:
    """Apply TESTS.md §3.2 steps 1–5, in order."""
    text = unicodedata.normalize("NFKC", text)  # 1 (also և → եւ)
    text = text.lower()  # 2
    text = _INNER_RE.sub("", text)  # 3
    text = _PUNCT_RE.sub(" ", text)  # 4
    return _SPACE_RE.sub(" ", text).strip()  # 5


def edit_distance(ref: list[str] | str, hyp: list[str] | str) -> int:
    """Levenshtein distance (substitutions, insertions, deletions; each costs 1)."""
    prev = list(range(len(hyp) + 1))
    for i, r in enumerate(ref, start=1):
        cur = [i]
        for j, h in enumerate(hyp, start=1):
            cur.append(min(prev[j] + 1, cur[j - 1] + 1, prev[j - 1] + (r != h)))
        prev = cur
    return prev[-1]


@dataclass(frozen=True)
class ErrorRate:
    edits: int
    ref_len: int

    @property
    def rate(self) -> float:
        if self.ref_len == 0:
            raise ValueError("reference is empty after normalisation; cannot compute a rate")
        return self.edits / self.ref_len


def cer(refs: list[str], hyps: list[str]) -> ErrorRate:
    """Character error rate over the corpus; spaces count as characters."""
    _check_pairs(refs, hyps)
    edits = ref_len = 0
    for r, h in zip(refs, hyps):
        r, h = normalise(r), normalise(h)
        edits += edit_distance(r, h)
        ref_len += len(r)
    return ErrorRate(edits, ref_len)


def wer(refs: list[str], hyps: list[str]) -> ErrorRate:
    """Word error rate over the corpus."""
    _check_pairs(refs, hyps)
    edits = ref_len = 0
    for r, h in zip(refs, hyps):
        rw, hw = normalise(r).split(), normalise(h).split()
        edits += edit_distance(rw, hw)
        ref_len += len(rw)
    return ErrorRate(edits, ref_len)


def _char_ngrams(text: str, n: int) -> Counter[str]:
    s = _SPACE_RE.sub("", text)
    return Counter(s[i : i + n] for i in range(len(s) - n + 1))


def chrf(refs: list[str], hyps: list[str], max_n: int = 6, beta: float = 2.0) -> float:
    """Corpus chrF on a 0–100 scale.

    n-gram statistics are summed over the corpus per order; precision and
    recall are averaged over the orders that have n-grams on both sides; then
    the F-beta score. Text is not lowercased and keeps punctuation; only
    whitespace is removed.
    """
    _check_pairs(refs, hyps)
    match = [0] * (max_n + 1)
    hyp_tot = [0] * (max_n + 1)
    ref_tot = [0] * (max_n + 1)
    for r, h in zip(refs, hyps):
        for n in range(1, max_n + 1):
            rc, hc = _char_ngrams(r, n), _char_ngrams(h, n)
            match[n] += sum((rc & hc).values())
            hyp_tot[n] += sum(hc.values())
            ref_tot[n] += sum(rc.values())
    precs, recs = [], []
    for n in range(1, max_n + 1):
        if hyp_tot[n] > 0 and ref_tot[n] > 0:
            precs.append(match[n] / hyp_tot[n])
            recs.append(match[n] / ref_tot[n])
    if not precs:
        return 0.0
    p, r = sum(precs) / len(precs), sum(recs) / len(recs)
    if p + r == 0:
        return 0.0
    b2 = beta * beta
    return 100.0 * (1 + b2) * p * r / (b2 * p + r)


def _check_pairs(refs: list[str], hyps: list[str]) -> None:
    if len(refs) != len(hyps):
        raise ValueError(f"{len(refs)} references but {len(hyps)} hypotheses")
    if not refs:
        raise ValueError("nothing to score: no reference/hypothesis pairs")
