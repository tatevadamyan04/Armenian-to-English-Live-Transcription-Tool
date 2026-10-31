"""Score E3 (the recorded call) against Tatev's transcript (TESTS.md §3.1, §3.3).

The reference eval/refs/e3.tsv is a list of turns on the call side (system.wav):

    start	end	speaker	hy	en
    0:04.5	0:21	A	Բարև ձեզ, ես …	Hello, I …
    0:22	0:30	B	…

- start, end: the turn's time in system.wav, as M:SS or M:SS.s
- speaker: a letter per person (A, B, C) — no names in the repository
- hy: what was said; may be empty for turns noted only for the speaker reference
- en: Tatev's translation; filled for ~15 turns, empty otherwise

Segments and turns don't line up one to one, so ASR is scored over a time window: the
reference text of every turn whose midpoint lies in the window, in time order, against
the recognised text of every segment whose midpoint lies in it. Put the window's edges in
pauses. Default window: the span of the turns that have Armenian text.

    python eval/score_e3.py eval/refs/e3.tsv data/e3/hyp_system.tsv
    python eval/score_e3.py eval/refs/e3.tsv data/e3/hyp_system.tsv --export-mt eval/refs/e3_mt.tsv

--export-mt writes the turns that have an English translation in the id/hy/en format, so
eval/run_offline.py --mt-input ref and eval/score.py can rate MT on conversational speech.
"""

from __future__ import annotations

import argparse
import csv
import sys
from dataclasses import dataclass
from pathlib import Path

from metrics import cer, wer

REF_COLUMNS = ["start", "end", "speaker", "hy", "en"]
HYP_COLUMNS = ["seg_id", "source", "start", "end", "cut", "hy", "en", "asr_s", "mt_s"]


class InputError(Exception):
    pass


@dataclass(frozen=True)
class Span:
    start: float
    end: float
    speaker: str
    hy: str
    en: str

    @property
    def mid(self) -> float:
        return (self.start + self.end) / 2


def parse_time(text: str) -> float:
    """'1:05.5' → 65.5 s; '65.5' (plain seconds) is also accepted."""
    text = text.strip()
    try:
        if ":" in text:
            minutes, seconds = text.split(":")
            if not minutes.isdecimal():
                raise ValueError
            value = int(minutes) * 60 + float(seconds)
        else:
            value = float(text)
    except ValueError:
        raise InputError(f"bad time {text!r}; write M:SS, M:SS.s or seconds") from None
    if value < 0:
        raise InputError(f"negative time {text!r}")
    return value


def read_rows(path: Path, columns: list[str]) -> list[dict[str, str]]:
    if not path.is_file():
        raise InputError(f"file not found: {path}")
    with path.open(encoding="utf-8", newline="") as f:
        reader = csv.DictReader(f, delimiter="\t", quoting=csv.QUOTE_NONE)
        if reader.fieldnames != columns:
            raise InputError(f"{path}: header must be {columns}, found {reader.fieldnames}")
        rows = list(reader)
    for n, row in enumerate(rows, start=2):
        if None in row or any(v is None for v in row.values()):
            raise InputError(f"{path}:{n}: expected {len(columns)} tab-separated fields")
    return rows


def read_ref(path: Path) -> list[Span]:
    spans = []
    for n, row in enumerate(read_rows(path, REF_COLUMNS), start=2):
        span = Span(parse_time(row["start"]), parse_time(row["end"]), row["speaker"].strip(),
                    row["hy"].strip(), row["en"].strip())
        if span.end <= span.start:
            raise InputError(f"{path}:{n}: end {row['end']} is not after start {row['start']}")
        if not span.speaker:
            raise InputError(f"{path}:{n}: speaker is empty")
        spans.append(span)
    if not spans:
        raise InputError(f"{path}: no rows")
    return sorted(spans, key=lambda s: s.start)


def read_hyp(path: Path, source: str = "system") -> list[Span]:
    rows = read_rows(path, HYP_COLUMNS)
    other = {row["source"] for row in rows} - {source}
    if other:
        raise InputError(f"{path}: rows from source(s) {sorted(other)}; E3's reference covers {source!r} only")
    return [Span(float(r["start"]), float(r["end"]), "", r["hy"].strip(), r["en"].strip()) for r in rows]


def in_window(spans: list[Span], start: float, end: float) -> list[Span]:
    return [s for s in spans if start <= s.mid < end]


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("ref", type=Path)
    ap.add_argument("hyp", type=Path)
    ap.add_argument("--from", dest="t_from", help="window start (M:SS); default: first turn with text")
    ap.add_argument("--to", dest="t_to", help="window end (M:SS); default: last turn with text")
    ap.add_argument("--export-mt", type=Path, help="write the turns with English as an id/hy/en TSV")
    args = ap.parse_args(argv)

    try:
        ref = read_ref(args.ref)
        hyp = read_hyp(args.hyp)
        texted = [s for s in ref if s.hy]
        if not texted:
            raise InputError(f"{args.ref}: no turn has Armenian text")
        t_from = parse_time(args.t_from) if args.t_from else texted[0].start
        t_to = parse_time(args.t_to) if args.t_to else texted[-1].end
        if t_to <= t_from:
            raise InputError("window end is not after its start")
        ref_w = in_window(ref, t_from, t_to)
        missing = [s for s in ref_w if not s.hy]
        if missing:
            raise InputError(
                f"{len(missing)} turn(s) inside the window have no Armenian text "
                f"(first at {missing[0].start:.1f} s); transcribe them or narrow the window"
            )
        hyp_w = in_window(hyp, t_from, t_to)
        if args.export_mt:
            if args.export_mt.exists():
                raise InputError(f"{args.export_mt} exists; not overwriting")
            with_en = [s for s in ref if s.hy and s.en]
            if not with_en:
                raise InputError(f"{args.ref}: no turn has both Armenian and English")
            with args.export_mt.open("w", encoding="utf-8", newline="") as f:
                w = csv.writer(f, delimiter="\t", quoting=csv.QUOTE_NONE, lineterminator="\n")
                w.writerow(["id", "hy", "en"])
                for n, s in enumerate(with_en, start=1):
                    w.writerow([f"e3_{n:02d}", s.hy, s.en])
            print(f"wrote {len(with_en)} turns with English to {args.export_mt}")
    except InputError as e:
        print(f"error: {e}", file=sys.stderr)
        return 1

    ref_text = " ".join(s.hy for s in ref_w)
    hyp_text = " ".join(s.hy for s in hyp_w)
    if not hyp_text.strip():
        print("error: no recognised text in the window", file=sys.stderr)
        return 1
    c, w_ = cer([ref_text], [hyp_text]), wer([ref_text], [hyp_text])
    speakers = sorted({s.speaker for s in ref})
    print(f"window {t_from:.1f}–{t_to:.1f} s ({(t_to - t_from) / 60:.1f} min): "
          f"{len(ref_w)} reference turns, {len(hyp_w)} segments")
    print(f"speakers in the reference: {', '.join(speakers)}")
    print(f"CER: {100 * c.rate:.2f}%  ({c.edits} edits / {c.ref_len} chars)")
    print(f"WER: {100 * w_.rate:.2f}%  ({w_.edits} edits / {w_.ref_len} words)")
    return 0


if __name__ == "__main__":
    sys.exit(main())
