"""Speaker metrics on E3 and the threshold sweep for M4 (TESTS.md §3.3, T10, T11; D7).

Runs the program's own segmenter (M2) over the call-side recording, embeds every segment
once with M4's embedder, then replays M4's labeler for each threshold and scores it
against the speaker letters in the reference. ASR and MT are not run, so a sweep is fast.

    python eval/score_speakers.py data/e3/system.wav data/e3/e3_ref.tsv
    python eval/score_speakers.py data/e3/system.wav data/e3/e3_ref.tsv --thresholds 0.4 0.5 0.6

Matching: each program segment gets the reference row it overlaps most in time, if that
overlap covers at least half of the segment; otherwise it is counted as unmatched.
Rows with one letter (A, B, C) are scored; rows with two people ("A+B") are reported
separately; rows marked "?" (speaker unknown) or with no letter are not scored.

The "separation" block checks the embeddings alone, before any clustering: if A–B
cosines are as high as A–A and B–B, no threshold can split them.

Prints only labels, letters, times and numbers — never the transcript.
"""

from __future__ import annotations

import argparse
import csv
import sys
import time
from collections import Counter
from dataclasses import dataclass
from pathlib import Path

import numpy as np

REF_COLUMNS = ("start", "end", "speaker")  # required header names in the reference TSV
MIN_OVERLAP = 0.5  # share of a segment that must overlap its reference row
LATE_GAP = 180.0  # T11: "heard again >= 3 min later"
DEFAULT_THRESHOLDS = [round(x, 2) for x in np.arange(0.20, 0.81, 0.05)]


@dataclass(frozen=True)
class RefRow:
    start: float
    end: float
    speaker: str  # "A", "B", "A+B", …


@dataclass(frozen=True)
class Scored:
    seg_id: str
    start: float
    end: float
    speaker: str | None  # reference letter(s); None if unmatched
    label: str
    reason: str = ""
    similarity: float | None = None


def is_single(speaker: str | None) -> bool:
    """One known person: a single letter A–Z. "A+B" is mixed; "?" (or similar) is unknown."""
    return speaker is not None and len(speaker) == 1 and speaker.isalpha()


def kind(speaker: str | None) -> str:
    if speaker is None:
        return "unmatched"
    if is_single(speaker):
        return "single"
    return "mixed" if "+" in speaker else "unknown"


def parse_time(text: str) -> float:
    """Seconds, given as '73.5', '1:13.5' or '0:01:13.5'."""
    parts = text.strip().split(":")
    try:
        value = 0.0
        for part in parts:
            value = value * 60 + float(part)
    except ValueError:
        raise ValueError(f"cannot read time {text!r}") from None
    return value


def read_reference(path: Path) -> list[RefRow]:
    if not path.is_file():
        raise SystemExit(f"error: reference not found: {path}")
    with path.open(encoding="utf-8", newline="") as f:
        reader = csv.DictReader(f, delimiter="\t", quoting=csv.QUOTE_NONE)
        missing = [c for c in REF_COLUMNS if c not in (reader.fieldnames or [])]
        if missing:
            raise SystemExit(f"error: {path} has no column(s) {missing}; header is {reader.fieldnames}")
        rows = []
        for line, row in enumerate(reader, start=2):
            speaker = (row["speaker"] or "").strip().upper()
            if not speaker:
                continue
            try:
                rows.append(RefRow(parse_time(row["start"]), parse_time(row["end"]), speaker))
            except (ValueError, AttributeError) as error:
                raise SystemExit(f"error: {path} line {line}: {error}") from None
    return rows


def match(start: float, end: float, rows: list[RefRow]) -> str | None:
    best, best_overlap = None, 0.0
    for row in rows:
        overlap = min(end, row.end) - max(start, row.start)
        if overlap > best_overlap:
            best, best_overlap = row, overlap
    if best is None or best_overlap < MIN_OVERLAP * (end - start):
        return None
    return best.speaker


def metrics(items: list[Scored]) -> dict:
    """TESTS.md §3.3 on single-speaker segments: consistency, purity, labels created; T10, T11."""
    single = [s for s in items if is_single(s.speaker)]
    by_speaker: dict[str, Counter] = {}
    by_label: dict[str, Counter] = {}
    for s in single:
        by_speaker.setdefault(s.speaker, Counter())[s.label] += 1
        by_label.setdefault(s.label, Counter())[s.speaker] += 1
    consistency = {k: c.most_common(1)[0][1] / sum(c.values()) for k, c in by_speaker.items()}
    purity = {k: c.most_common(1)[0][1] / sum(c.values()) for k, c in by_label.items()}
    top = {k: c.most_common(1)[0][0] for k, c in by_speaker.items()}
    t10 = len(set(top.values())) == len(top) and len(top) > 0
    t11 = {}
    for speaker in by_speaker:
        own = sorted((s for s in single if s.speaker == speaker), key=lambda s: s.start)
        late = [s for s in own if s.start >= own[0].start + LATE_GAP]
        if late:
            late_top = Counter(s.label for s in late).most_common(1)[0][0]
            t11[speaker] = late_top == top[speaker]
    weighted = lambda d, counts: (  # noqa: E731
        sum(d[k] * sum(counts[k].values()) for k in d) / max(1, sum(sum(c.values()) for c in counts.values()))
    )
    return {
        "segments": len(single),
        "speakers": len(by_speaker),
        "labels": len({s.label for s in items}),
        "consistency": consistency,
        "purity": purity,
        "consistency_w": weighted(consistency, by_speaker),
        "purity_w": weighted(purity, by_label),
        "top": top,
        "t10": t10,
        "t11": t11,
    }


def score_run(embedded, threshold: float, min_new: float | None = None) -> list[Scored]:
    """Replay M4 over precomputed embeddings; return one Scored per segment."""
    from hyen.speakers import MIN_NEW_SECONDS, SpeakerLabeler

    if min_new is None:
        min_new = MIN_NEW_SECONDS

    table = {id(seg.samples): vector for seg, _, vector in embedded if vector is not None}

    class Cached:
        def embed(self, samples):
            return table[id(samples)]

    labeler = SpeakerLabeler(Cached(), threshold=threshold, min_new_seconds=min_new)
    scored = []
    for seg, speaker, _ in embedded:
        a = labeler.assign(seg)
        scored.append(Scored(seg.seg_id, seg.t_speech_start, seg.t_speech_end, speaker, a.label, a.reason, a.similarity))
    return scored


def center(embedded):
    """Subtract the mean of all unit embeddings (offline experiment; see --center)."""
    units = [v / np.linalg.norm(v) for _, _, v in embedded if v is not None]
    mean = np.mean(units, axis=0)
    return [(seg, sp, None if v is None else v / np.linalg.norm(v) - mean) for seg, sp, v in embedded]


def separation(embedded, min_seconds: float) -> dict:
    """Do the embeddings tell the speakers apart, before any clustering?

    Uses single-speaker segments of at least `min_seconds` (the ones that may create or
    update a speaker). Returns cosine statistics per letter pair, the share of segments
    whose nearest other segment has the same letter (1-NN), and the share closer to the
    mean of their own letter than to any other letter's mean (leave-one-out).
    """
    items = [(sp, v / np.linalg.norm(v)) for seg, sp, v in embedded
             if v is not None and is_single(sp) and seg.t_speech_end - seg.t_speech_start >= min_seconds]
    letters = sorted({sp for sp, _ in items})
    pairs: dict[str, list[float]] = {}
    for i, (si, vi) in enumerate(items):
        for sj, vj in items[i + 1:]:
            key = "–".join(sorted((si, sj)))
            pairs.setdefault(key, []).append(float(vi @ vj))
    nn_hits = 0
    centroid_hits = 0
    sums = {k: sum(v for sp, v in items if sp == k) for k in letters}
    counts = Counter(sp for sp, _ in items)
    for i, (si, vi) in enumerate(items):
        others = [(float(vi @ vj), sj) for j, (sj, vj) in enumerate(items) if j != i]
        if others:
            nn_hits += max(others)[1] == si
        best = None
        for k in letters:
            total = sums[k] - (vi if k == si else 0)
            n = counts[k] - (1 if k == si else 0)
            if n == 0:
                continue
            sim = float(vi @ (total / np.linalg.norm(total)))
            if best is None or sim > best[0]:
                best = (sim, k)
        centroid_hits += best is not None and best[1] == si
    return {
        "n": len(items),
        "counts": dict(counts),
        "pairs": {k: (min(v), float(np.median(v)), max(v), len(v)) for k, v in sorted(pairs.items())},
        "nn": nn_hits / max(1, len(items)),
        "centroid": centroid_hits / max(1, len(items)),
    }


def write_template(wav: Path, out: Path) -> int:
    """One row per segment the program finds (M2), speaker column empty, for lettering by ear."""
    from segment_offline import VAD_MODEL, chunks_of, read_wav

    from hyen.segmenter import Segmenter
    from hyen.vad import SileroVad

    if out.exists():
        raise SystemExit(f"error: {out} exists; delete it or choose another name so no lettering is lost")
    segmenter = Segmenter("system", SileroVad(VAD_MODEL), log=lambda m: None)
    lines = ["start\tend\tspeaker\tclock"]
    for chunk in chunks_of(read_wav(wav), "system"):
        seg = segmenter.feed(chunk, now=chunk.t_start)
        if seg is not None:
            whole = int(seg.t_speech_start)
            lines.append(f"{seg.t_speech_start:.2f}\t{seg.t_speech_end:.2f}\t\t{whole // 60}:{whole % 60:02d}")
    out.write_text("\n".join(lines) + "\n", encoding="utf-8")
    print(f"{len(lines) - 1} segments → {out}. Fill the speaker column: A, B, A+B, or ? (unknown).")
    return 0


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("wav", type=Path)
    ap.add_argument("ref", type=Path)
    ap.add_argument("--thresholds", type=float, nargs="+", default=DEFAULT_THRESHOLDS)
    ap.add_argument("--details", type=float, help="print every segment's letter and label at this threshold")
    ap.add_argument("--template", action="store_true",
                    help="write REF as a lettering template (one row per program segment, speaker empty) and stop")
    ap.add_argument("--center", action="store_true",
                    help="experiment, offline only: subtract the mean embedding of the whole recording first "
                         "(uses future audio, so the live program can't do exactly this)")
    args = ap.parse_args(argv)

    from segment_offline import VAD_MODEL, chunks_of, read_wav

    from hyen.modelfiles import SPEAKER_MODEL, require
    from hyen.segmenter import Segmenter
    from hyen.speakers import MIN_EMBED_SECONDS, MIN_NEW_SECONDS, SAMPLE_RATE, SherpaEmbedder
    from hyen.vad import SileroVad

    if args.template:
        return write_template(args.wav, args.ref)
    rows = read_reference(args.ref)
    if not rows:
        raise SystemExit(f"error: no rows with a speaker letter in {args.ref}")
    ref_end = max(r.end for r in rows)
    samples = read_wav(args.wav)
    segmenter = Segmenter("system", SileroVad(VAD_MODEL), log=lambda m: None)
    embedder = SherpaEmbedder(require(SPEAKER_MODEL, "speaker embedding model"))

    embedded, times = [], []
    for chunk in chunks_of(samples, "system"):
        seg = segmenter.feed(chunk, now=chunk.t_start)
        if seg is None or seg.t_speech_start > ref_end:
            continue
        vector = None
        if len(seg.samples) / SAMPLE_RATE >= MIN_EMBED_SECONDS:
            t = time.perf_counter()
            vector = embedder.embed(seg.samples)
            times.append(time.perf_counter() - t)
        embedded.append((seg, match(seg.t_speech_start, seg.t_speech_end, rows), vector))

    if args.center:
        embedded = center(embedded)
        print("embeddings centred on the recording's mean (--center, offline experiment)")
    kinds = Counter(kind(sp) for _, sp, _ in embedded)
    print(f"reference: {len(rows)} rows with letters, up to {ref_end:.0f} s · "
          f"segments up to there: {len(embedded)} ({kinds['single']} single, {kinds['mixed']} mixed, "
          f"{kinds['unknown']} unknown, {kinds['unmatched']} unmatched)")
    print(f"embedding time: median {np.median(times) * 1000:.0f} ms, max {max(times) * 1000:.0f} ms "
          f"over {len(times)} segments")
    print()
    sep = separation(embedded, MIN_NEW_SECONDS)
    print(f"separation (single-speaker segments >= {MIN_NEW_SECONDS} s, no clustering): n={sep['n']} {sep['counts']}")
    for key, (lo, med, hi, count) in sep["pairs"].items():
        print(f"  cosine {key}: min {lo:.2f}  median {med:.2f}  max {hi:.2f}  (pairs {count})")
    print(f"  nearest other segment has the same letter: {sep['nn']:.2f}")
    print(f"  closer to own letter's mean than to the other's: {sep['centroid']:.2f}")
    print()
    print("thresh  labels  speakers  consist.  purity   T10   T11    per speaker: letter→label (consistency)")
    for threshold in args.thresholds:
        m = metrics(score_run(embedded, threshold))
        t11 = "–" if not m["t11"] else ("pass" if all(m["t11"].values()) else "FAIL")
        per = "  ".join(f"{k}→{m['top'][k].replace('Speaker ', 'S')} ({m['consistency'][k]:.2f})"
                        for k in sorted(m["top"]))
        print(f"{threshold:6.2f}  {m['labels']:6d}  {m['speakers']:8d}  {m['consistency_w']:8.2f}  "
              f"{m['purity_w']:6.2f}  {'pass' if m['t10'] else 'FAIL':>4}  {t11:>4}    {per}")

    if args.details is not None:
        print(f"\nsegments at threshold {args.details} (letter, label, M4's reason, similarity to the chosen speaker):")
        run = score_run(embedded, args.details)
        for s in run:
            sim = "" if s.similarity is None else f"{s.similarity:.2f}"
            print(f"  {s.seg_id:12s} {s.start:7.2f}–{s.end:7.2f} {s.end - s.start:5.2f} s  "
                  f"{s.speaker or '-':4s} {s.label:10s} {s.reason:8s} {sim}")
        mixed = [s for s in run if kind(s.speaker) == "mixed"]
        if mixed:
            print(f"mixed segments ({len(mixed)}): " + ", ".join(f"{s.speaker}→{s.label}" for s in mixed))
    return 0


if __name__ == "__main__":
    sys.exit(main())
