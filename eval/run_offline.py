"""Run ASR and translation over a folder of recordings and write a hypothesis TSV.

Development tooling (M8), not part of the program. The ids come from the reference TSV;
each id's audio is <wav-dir>/<id>.wav (16 kHz, mono, 16-bit PCM, as written by
eval/record.py). The output has the reference's format (id, hy, en), so eval/score.py
can compare the two.

    # ASR + MT, as the program does it:
    python eval/run_offline.py eval/refs/e1.tsv data/e1 data/e1/hyp_beam1.tsv
    # MT only, on the reference Armenian, to judge translation without ASR errors:
    python eval/run_offline.py eval/refs/e1.tsv data/e1 data/e1/mt_beam4.tsv --mt-input ref --beam 4

Per-item times go to stderr; a summary of ASR and MT times is printed at the end.
"""

from __future__ import annotations

import argparse
import csv
import statistics
import sys
import time
import wave
from pathlib import Path

import numpy as np

from hyen.asr import SAMPLE_RATE, Recognizer
from hyen.translate import BEAM_SIZE, NO_REPEAT_NGRAM_SIZE, REPETITION_PENALTY, Translator


def read_ids_and_refs(ref_path: Path) -> list[tuple[str, str]]:
    with ref_path.open(encoding="utf-8", newline="") as f:
        reader = csv.DictReader(f, delimiter="\t", quoting=csv.QUOTE_NONE)
        if reader.fieldnames != ["id", "hy", "en"]:
            raise SystemExit(f"error: {ref_path}: header must be id, hy, en; found {reader.fieldnames}")
        return [(row["id"], row["hy"]) for row in reader]


def read_wav(path: Path) -> np.ndarray:
    """16 kHz mono 16-bit PCM → float32 in [-1, 1]. Anything else is an error, not converted."""
    if not path.is_file():
        raise SystemExit(f"error: recording not found: {path}")
    with wave.open(str(path), "rb") as w:
        rate, channels, width = w.getframerate(), w.getnchannels(), w.getsampwidth()
        if (rate, channels, width) != (SAMPLE_RATE, 1, 2):
            raise SystemExit(
                f"error: {path}: {rate} Hz, {channels} channel(s), {8 * width}-bit; "
                f"expected {SAMPLE_RATE} Hz, mono, 16-bit"
            )
        data = w.readframes(w.getnframes())
    return np.frombuffer(data, dtype="<i2").astype(np.float32) / 32768.0


def summary(name: str, times: list[float]) -> str:
    # "inclusive" keeps p95 within the observed range; the default method extrapolates
    # beyond the maximum for small samples (seen: p95 1.74 s > max 1.60 s with n=5).
    p95 = statistics.quantiles(times, n=20, method="inclusive")[-1] if len(times) >= 2 else times[0]
    return (f"{name}: median {statistics.median(times):.2f} s  p95 {p95:.2f} s  "
            f"max {max(times):.2f} s  (n={len(times)})")


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("ref", type=Path, help="reference TSV (id, hy, en); gives the ids")
    ap.add_argument("wav_dir", type=Path, help="folder with <id>.wav")
    ap.add_argument("out", type=Path, help="hypothesis TSV to write (must not exist)")
    ap.add_argument("--mt-input", choices=["asr", "ref"], default="asr",
                    help="translate the ASR output (default) or the reference Armenian")
    ap.add_argument("--beam", type=int, default=BEAM_SIZE)
    ap.add_argument("--rep-penalty", type=float, default=REPETITION_PENALTY)
    ap.add_argument("--no-repeat-ngram", type=int, default=NO_REPEAT_NGRAM_SIZE)
    args = ap.parse_args(argv)

    if args.out.exists():
        raise SystemExit(f"error: {args.out} exists; choose a new name so earlier results stay")
    items = read_ids_and_refs(args.ref)
    print(f"MT settings: beam {args.beam}, repetition penalty {args.rep_penalty}, "
          f"no-repeat n-gram {args.no_repeat_ngram}; MT input: {args.mt_input}", file=sys.stderr)

    recognizer = Recognizer() if args.mt_input == "asr" else None
    translator = Translator(beam_size=args.beam, repetition_penalty=args.rep_penalty,
                            no_repeat_ngram_size=args.no_repeat_ngram)

    rows, asr_times, mt_times = [], [], []
    for item_id, ref_hy in items:
        if recognizer is not None:
            samples = read_wav(args.wav_dir / f"{item_id}.wav")
            t0 = time.perf_counter()
            hy = recognizer.transcribe(samples)
            asr_times.append(time.perf_counter() - t0)
        else:
            hy = ref_hy
        if hy:
            t0 = time.perf_counter()
            en = translator.translate(hy)
            mt_times.append(time.perf_counter() - t0)
        else:
            en = ""
            print(f"{item_id}: ASR returned no text; EN left empty", file=sys.stderr)
        print(f"{item_id}  {hy}  →  {en}", file=sys.stderr)
        rows.append((item_id, hy.replace("\t", " "), en.replace("\t", " ")))

    with args.out.open("w", encoding="utf-8", newline="") as f:
        w = csv.writer(f, delimiter="\t", quoting=csv.QUOTE_NONE, lineterminator="\n")
        w.writerow(["id", "hy", "en"])
        w.writerows(rows)

    if asr_times:
        print(summary("ASR", asr_times))
    if mt_times:
        print(summary("MT ", mt_times))
    print(f"wrote {len(rows)} rows to {args.out}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
