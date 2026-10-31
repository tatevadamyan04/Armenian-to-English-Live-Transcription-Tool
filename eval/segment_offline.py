"""Run the program's own M2 → M3 → M5 chain over one recorded source file (development tooling, M8).

The recording is cut into 512-sample AudioChunks with sample-counted times (as M1 does, D15)
and fed to the same Segmenter and SileroVad as a live session, so the segments match what
the program would produce from that audio. A phrase still open when the file ends is not
emitted, as when a live session is stopped.

    python eval/segment_offline.py data/e3/system.wav system data/e3/hyp_system.tsv
    python eval/segment_offline.py data/e3/system.wav system data/e3/hyp_system_asr.tsv --no-mt

Output TSV columns: seg_id, source, start, end, cut, hy, en, asr_s, mt_s
(start = speech start, end = speech end, both seconds since the start of the file).
"""

from __future__ import annotations

import argparse
import csv
import sys
import time
import wave
from pathlib import Path

import numpy as np

from hyen.asr import Recognizer
from hyen.audio import BLOCK, SAMPLE_RATE, SOURCES, AudioChunk
from hyen.segmenter import Segmenter
from hyen.translate import Translator
from hyen.vad import SileroVad

REPO = Path(__file__).resolve().parents[1]
VAD_MODEL = REPO / "models" / "vad-silero" / "silero_vad.onnx"
COLUMNS = ["seg_id", "source", "start", "end", "cut", "hy", "en", "asr_s", "mt_s"]


def read_wav(path: Path) -> np.ndarray:
    """16 kHz mono 16-bit PCM → float32. Anything else is an error, not converted."""
    if not path.is_file():
        raise SystemExit(f"error: recording not found: {path}")
    with wave.open(str(path), "rb") as w:
        fmt = (w.getframerate(), w.getnchannels(), w.getsampwidth())
        if fmt != (SAMPLE_RATE, 1, 2):
            raise SystemExit(f"error: {path}: {fmt[0]} Hz, {fmt[1]} ch, {8 * fmt[2]}-bit; expected 16 kHz mono 16-bit")
        data = w.readframes(w.getnframes())
    return np.frombuffer(data, dtype="<i2").astype(np.float32) / 32768.0


def chunks_of(samples: np.ndarray, source: str):
    """AudioChunks with t_start counted from samples; a last partial block is left out."""
    for n in range(len(samples) // BLOCK):
        yield AudioChunk(source, samples[n * BLOCK : (n + 1) * BLOCK].copy(), n * BLOCK / SAMPLE_RATE)


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("wav", type=Path)
    ap.add_argument("source", choices=SOURCES)
    ap.add_argument("out", type=Path, help="TSV to write (must not exist)")
    ap.add_argument("--no-mt", action="store_true", help="ASR only; the en column stays empty")
    args = ap.parse_args(argv)
    if args.out.exists():
        raise SystemExit(f"error: {args.out} exists; choose a new name so earlier results stay")

    samples = read_wav(args.wav)
    segmenter = Segmenter(args.source, SileroVad(VAD_MODEL))
    recognizer = Recognizer()
    translator = None if args.no_mt else Translator()

    rows = []
    for chunk in chunks_of(samples, args.source):
        segment = segmenter.feed(chunk, now=chunk.t_start + BLOCK / SAMPLE_RATE)
        if segment is None:
            continue
        t0 = time.perf_counter()
        hy = recognizer.transcribe(segment.samples)
        asr_s = time.perf_counter() - t0
        en, mt_s = "", 0.0
        if hy and translator is not None:
            t0 = time.perf_counter()
            en = translator.translate(hy)
            mt_s = time.perf_counter() - t0
        if not hy:
            print(f"{segment.seg_id}: ASR returned no text (dropped in a live session)", file=sys.stderr)
        print(f"{segment.seg_id} {segment.t_speech_start:7.2f}–{segment.t_speech_end:7.2f}  {hy}", file=sys.stderr)
        rows.append([segment.seg_id, args.source, f"{segment.t_speech_start:.2f}", f"{segment.t_speech_end:.2f}",
                     int(segment.cut), hy.replace("\t", " "), en.replace("\t", " "), f"{asr_s:.3f}", f"{mt_s:.3f}"])

    with args.out.open("w", encoding="utf-8", newline="") as f:
        w = csv.writer(f, delimiter="\t", quoting=csv.QUOTE_NONE, lineterminator="\n")
        w.writerow(COLUMNS)
        w.writerows(rows)
    print(f"{len(rows)} segments ({sum(int(r[4]) for r in rows)} cut at the cap, "
          f"{segmenter.dropped} too short and dropped) → {args.out}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
