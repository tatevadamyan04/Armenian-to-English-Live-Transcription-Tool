"""eval/cross_speakers.py: the simulated two-speaker call."""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "eval"))

from cross_speakers import GAP, interleave  # noqa: E402

from hyen.segmenter import Segment  # noqa: E402


def seg(n, start, length):
    return Segment(f"system-{n}", "system", np.zeros(1, dtype=np.float32), start, start + length, start + length)


def test_interleave_alternates_turns_and_retimes():
    a = [(seg(i, 100 + i * 10, 2.0), "A", None) for i in range(3)]
    c = [(seg(i, 5 + i * 10, 1.0), "C", None) for i in range(5)]
    out = interleave(a, c, turn=2)
    assert [letter for _, letter, _ in out] == ["A", "A", "C", "C", "A", "C", "C", "C"]
    assert [s.seg_id for s, _, _ in out] == [f"system-{n}" for n in range(1, 9)]
    starts = [s.t_speech_start for s, _, _ in out]
    assert starts[0] == 0.0 and starts[1] == 2.0 + GAP and starts == sorted(starts)
    assert all(abs((s.t_speech_end - s.t_speech_start) - (2.0 if l == "A" else 1.0)) < 1e-9 for s, l, _ in out)


def test_interleave_keeps_samples_object_for_cached_embeddings():
    first = seg(1, 0, 2.0)
    out = interleave([(first, "A", None)], [])
    assert out[0][0].samples is first.samples
