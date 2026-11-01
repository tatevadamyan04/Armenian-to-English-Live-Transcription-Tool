"""Two-speaker check for M4 from two recordings (S05; T10, T11; D7).

E3's call side is mostly one person (A) and E4's call side is one other person, both over
Zoom through BlackHole. This tool takes E3's single-letter segments (letter from the
reference) and every E4 segment (letter given on the command line), then:

1. separation — cosines within each person and between them, no clustering;
2. a simulated call — the two people's segments interleaved in turns of TURN segments,
   on a fresh session clock, replayed through M4 at each threshold.

    python eval/cross_speakers.py data/e3/system.wav data/e3/e3_ref.tsv data/e4/system.wav
    python eval/cross_speakers.py data/e3/system.wav data/e3/e3_ref.tsv data/e4/system.wav --center

Caveat: the two people come from different sessions, so session differences (level, line,
room) can make them easier to separate than two people on one call. Treat the result as a
best case. Prints only letters, labels and numbers.
"""

from __future__ import annotations

import argparse
import sys
from dataclasses import replace
from pathlib import Path

import numpy as np

from score_speakers import DEFAULT_THRESHOLDS, center, is_single, match, metrics, read_reference, score_run, separation

TURN = 2  # segments per turn in the simulated call
GAP = 0.5  # seconds of silence between segments on the simulated clock


def segments(wav: Path, embedder, min_embed: float):
    from segment_offline import VAD_MODEL, chunks_of, read_wav

    from hyen.segmenter import Segmenter
    from hyen.vad import SileroVad

    segmenter = Segmenter("system", SileroVad(VAD_MODEL), log=lambda m: None)
    out = []
    for chunk in chunks_of(read_wav(wav), "system"):
        seg = segmenter.feed(chunk, now=chunk.t_start)
        if seg is not None:
            long_enough = len(seg.samples) / 16_000 >= min_embed
            out.append((seg, embedder.embed(seg.samples) if long_enough else None))
    return out


def interleave(first: list, second: list, turn: int = TURN) -> list:
    """Alternate turns of `turn` segments; re-time them on one session clock."""
    order, i, j = [], 0, 0
    while i < len(first) or j < len(second):
        order += first[i : i + turn]
        i += turn
        order += second[j : j + turn]
        j += turn
    clock, out = 0.0, []
    for n, (seg, letter, vector) in enumerate(order, start=1):
        length = seg.t_speech_end - seg.t_speech_start
        moved = replace(seg, seg_id=f"system-{n}", t_speech_start=clock, t_speech_end=clock + length)
        out.append((moved, letter, vector))
        clock += length + GAP
    return out


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("e3_wav", type=Path)
    ap.add_argument("e3_ref", type=Path)
    ap.add_argument("e4_wav", type=Path)
    ap.add_argument("--e4-letter", default="C", help="letter for E4's call-side speaker (default C)")
    ap.add_argument("--thresholds", type=float, nargs="+", default=DEFAULT_THRESHOLDS)
    ap.add_argument("--min-new", type=float, nargs="+", default=None,
                    help="seconds a segment needs to create a speaker (default: M4's MIN_NEW_SECONDS)")
    ap.add_argument("--center", action="store_true", help="offline experiment: subtract the mean embedding")
    args = ap.parse_args(argv)

    from hyen.modelfiles import SPEAKER_MODEL, require
    from hyen.speakers import MIN_EMBED_SECONDS, MIN_NEW_SECONDS, SherpaEmbedder

    embedder = SherpaEmbedder(require(SPEAKER_MODEL, "speaker embedding model"))
    rows = read_reference(args.e3_ref)
    ref_end = max(r.end for r in rows)
    e3 = []
    for seg, vector in segments(args.e3_wav, embedder, MIN_EMBED_SECONDS):
        if seg.t_speech_start > ref_end:
            break
        letter = match(seg.t_speech_start, seg.t_speech_end, rows)
        if is_single(letter):
            e3.append((seg, letter, vector))
    e4 = [(seg, args.e4_letter, vector) for seg, vector in segments(args.e4_wav, embedder, MIN_EMBED_SECONDS)]
    if not e3 or not e4:
        raise SystemExit(f"error: no segments (E3 single-speaker: {len(e3)}, E4: {len(e4)})")

    # E3's own minority speaker(s) are left out, so the call has exactly two people.
    main_e3 = max({l for _, l, _ in e3}, key=lambda l: sum(1 for _, x, _ in e3 if x == l))
    e3 = [item for item in e3 if item[1] == main_e3]
    if main_e3 == args.e4_letter:
        raise SystemExit(f"error: E3's speaker is also {main_e3!r}; pass another --e4-letter")
    combined = interleave(e3, e4)
    if args.center:
        combined = center(combined)
        print("embeddings centred on the mean (--center, offline experiment)")
    seconds = lambda items: sum(s.t_speech_end - s.t_speech_start for s, _, _ in items)  # noqa: E731
    print(f"E3 speaker {main_e3}: {len(e3)} segments, {seconds(e3):.0f} s · "
          f"E4 speaker {args.e4_letter}: {len(e4)} segments, {seconds(e4):.0f} s · "
          f"simulated call {seconds(combined) / 60:.1f} min of speech, turns of {TURN} segments")
    print()

    sep = separation(combined, MIN_NEW_SECONDS)
    print(f"separation (segments >= {MIN_NEW_SECONDS} s, no clustering): n={sep['n']} {sep['counts']}")
    for key, (lo, med, hi, count) in sep["pairs"].items():
        print(f"  cosine {key}: min {lo:.2f}  median {med:.2f}  max {hi:.2f}  (pairs {count})")
    print(f"  nearest other segment has the same letter: {sep['nn']:.2f}")
    print(f"  closer to own letter's mean than to the other's: {sep['centroid']:.2f}")
    print()
    print("min_new  thresh  labels  consist.  purity   T10   T11    per speaker: letter→label (consistency)")
    for min_new in args.min_new or [MIN_NEW_SECONDS]:
        for threshold in args.thresholds:
            m = metrics(score_run(combined, threshold, min_new))
            t11 = "–" if not m["t11"] else ("pass" if all(m["t11"].values()) else "FAIL")
            per = "  ".join(f"{k}→{m['top'][k].replace('Speaker ', 'S')} ({m['consistency'][k]:.2f})"
                            for k in sorted(m["top"]))
            print(f"{min_new:7.1f}  {threshold:6.2f}  {m['labels']:6d}  {m['consistency_w']:8.2f}  "
                  f"{m['purity_w']:6.2f}  {'pass' if m['t10'] else 'FAIL':>4}  {t11:>4}    {per}")
        print()
    return 0


if __name__ == "__main__":
    sys.exit(main())
