"""M8 recorder (session 02): records what M1 delivers into WAV files.

    python eval/record.py session --mic DEVICE --system DEVICE --out DIR [--seconds N]
        Both sources into DIR/mic.wav and DIR/system.wav, for T04 and E3. Stops after
        N seconds of microphone audio, or on Ctrl+C. Exits non-zero if audio was lost,
        if a source was pure digital silence, or if the two files differ in length by
        more than 0.1 s (T04).

    python eval/record.py prompts --mic DEVICE --refs FILE --out DIR
        One file per sentence of a reference file, for E1. Shows the Armenian text;
        Enter starts and stops; then Enter keeps the take or r records it again.
        Sentences that already have a file are skipped, so an interrupted session
        can be continued with the same command.

Files are 16 kHz mono 16-bit PCM: exactly what the program hears. This is development
tooling for tests and evaluation sets (PLAN.md §5, M8), not a product feature.
"""

from __future__ import annotations

import argparse
import re
import sys
import time
import wave
from dataclasses import dataclass
from pathlib import Path

import numpy as np

from hyen import audio

MAX_LENGTH_DIFFERENCE = 0.1  # seconds; T04's limit
MIN_TAKE_SECONDS = 0.5  # shorter takes are treated as a mis-press and recorded again
REFS_HEADER = ["id", "hy", "en"]
ID_PATTERN = re.compile(r"[A-Za-z0-9_-]+")
PERMISSION_HINT = "Terminal probably has no microphone permission (System Settings → Privacy & Security → Microphone)."
SILENCE_HINTS = {
    "mic": f"mic.wav is pure digital silence: {PERMISSION_HINT}",
    "system": "system.wav is pure digital silence: nothing reached BlackHole. Check that the Multi-Output Device "
    "is the sound output and that something was playing, and Terminal's microphone permission.",
}


class RecordingError(Exception):
    """A recording can't be made or used; the message says why and what to do."""


@dataclass(frozen=True)
class RefRow:
    id: str
    hy: str
    en: str


def read_refs(path: Path) -> list[RefRow]:
    """Read a reference file: UTF-8, a header row id<TAB>hy<TAB>en, then one sentence per line."""
    lines = path.read_text(encoding="utf-8-sig").splitlines()
    if not lines or [field.strip() for field in lines[0].split("\t")] != REFS_HEADER:
        raise RecordingError(f"{path}, line 1: expected the header id<TAB>hy<TAB>en")
    rows: list[RefRow] = []
    seen: set[str] = set()
    for number, line in enumerate(lines[1:], start=2):
        if not line.strip():
            continue
        fields = line.split("\t")
        if len(fields) != 3:
            raise RecordingError(
                f"{path}, line {number}: {len(fields)} tab-separated field(s), expected 3 (id, hy, en). "
                "Check that the separators are tab characters, not spaces."
            )
        row = RefRow(*(field.strip() for field in fields))
        if not ID_PATTERN.fullmatch(row.id):
            raise RecordingError(f"{path}, line {number}: the id {row.id!r} may only contain letters, digits, _ and -")
        if row.id in seen:
            raise RecordingError(f"{path}, line {number}: the id {row.id!r} appears twice")
        if not row.hy or not row.en:
            raise RecordingError(f"{path}, line {number}: both the Armenian and the English text are required")
        seen.add(row.id)
        rows.append(row)
    if not rows:
        raise RecordingError(f"{path}: no sentences after the header")
    return rows


def to_pcm16(samples: np.ndarray) -> tuple[bytes, int]:
    """float32 in [-1, 1] → 16-bit little-endian PCM, plus how many samples were clipped."""
    clipped = int(np.count_nonzero(np.abs(samples) > 1.0))
    pcm = np.round(np.clip(samples, -1.0, 1.0) * 32767.0).astype("<i2")
    return pcm.tobytes(), clipped


class WavWriter:
    """Streams 16 kHz mono float32 audio into a 16-bit WAV file and keeps simple statistics."""

    def __init__(self, path: Path) -> None:
        self.path = path
        self.samples = 0
        self.peak = 0.0
        self.clipped = 0
        self.nonzero = False
        self._wav = wave.open(str(path), "wb")
        self._wav.setnchannels(1)
        self._wav.setsampwidth(2)
        self._wav.setframerate(audio.SAMPLE_RATE)

    def write(self, samples: np.ndarray) -> None:
        data, clipped = to_pcm16(samples)
        self._wav.writeframes(data)
        self.samples += len(samples)
        self.peak = max(self.peak, audio.peak(samples))
        self.clipped += clipped
        self.nonzero = self.nonzero or bool(np.any(samples))

    @property
    def seconds(self) -> float:
        return self.samples / audio.SAMPLE_RATE

    def close(self) -> None:
        self._wav.close()


def write_wav(path: Path, samples: np.ndarray) -> None:
    writer = WavWriter(path)
    try:
        writer.write(samples)
    finally:
        writer.close()


def resolve(specs: dict[str, str]) -> dict[str, audio.InputDevice]:
    """Resolve --mic/--system exactly like python -m hyen does: exact name or index, no defaults."""
    devices = audio.list_input_devices()
    resolved = {name: audio.resolve_input_device(spec, devices) for name, spec in specs.items()}
    if "mic" in resolved and "system" in resolved:
        audio.check_distinct(resolved["mic"], resolved["system"])
    return resolved


# session mode (T04, E3)


def record_session(args: argparse.Namespace) -> int:
    devices = resolve({"mic": args.mic, "system": args.system})
    out = Path(args.out)
    paths = {name: out / f"{name}.wav" for name in audio.SOURCES}
    existing = [str(path) for path in paths.values() if path.exists()]
    if existing:
        raise RecordingError(f"{', '.join(existing)} already exist; move them away or choose another --out.")
    out.mkdir(parents=True, exist_ok=True)
    limit = None if args.seconds is None else round(args.seconds * audio.SAMPLE_RATE)
    print(
        f"Recording  mic: {audio.describe(devices['mic'])}  ·  system: {audio.describe(devices['system'])}",
        file=sys.stderr,
    )
    length = "until Ctrl+C" if limit is None else f"{args.seconds:g} s of microphone audio; Ctrl+C stops earlier"
    print(f"Into {out}/, {length}.", file=sys.stderr)
    with audio.Capture(devices) as capture:
        capture.start(time.monotonic())
        writers = {name: WavWriter(paths[name]) for name in audio.SOURCES}
        try:
            try:
                while limit is None or writers["mic"].samples < limit:
                    chunk = capture.get("mic")
                    wanted = len(chunk.samples) if limit is None else limit - writers["mic"].samples
                    writers["mic"].write(chunk.samples[:wanted])
                    for other in capture.drain("system"):
                        writers["system"].write(other.samples)
            except KeyboardInterrupt:
                print("Stopped.", file=sys.stderr)
            capture.stop()
            for other in capture.drain("system"):
                writers["system"].write(other.samples)
            stats = {name: capture.stats(name) for name in audio.SOURCES}
        finally:
            for writer in writers.values():
                writer.close()
    return report_session(writers, stats)


def report_session(writers: dict[str, WavWriter], stats: dict[str, audio.SourceStats]) -> int:
    print("\nSummary", file=sys.stderr)
    for name in audio.SOURCES:
        writer, stat = writers[name], stats[name]
        gap = "n/a" if stat.clock_gap is None else f"{stat.clock_gap:+.3f} s"
        print(
            f"  {name:<7}{writer.seconds:8.2f} s   peak {audio.dbfs(writer.peak):>11}   "
            f"clipped {writer.clipped}   overflows {stat.overflows}   clock gap {gap}",
            file=sys.stderr,
        )
    difference = abs(writers["mic"].seconds - writers["system"].seconds)
    print(f"  length difference {difference:.3f} s (T04 allows {MAX_LENGTH_DIFFERENCE} s)", file=sys.stderr)
    problems = []
    for name in audio.SOURCES:
        if stats[name].overflows:
            problems.append(f"{name}: audio was lost {stats[name].overflows} time(s); close other apps and record again.")
        if not writers[name].nonzero:
            problems.append(SILENCE_HINTS[name])
        if writers[name].clipped:
            print(f"warning: {name}: {writers[name].clipped} samples clipped at full scale; lower that level.", file=sys.stderr)
    if difference > MAX_LENGTH_DIFFERENCE:
        problems.append(f"the two files differ in length by {difference:.3f} s, more than T04's {MAX_LENGTH_DIFFERENCE} s.")
    for problem in problems:
        print(f"error: {problem}", file=sys.stderr)
    return 1 if problems else 0


# prompts mode (E1)


def record_prompts(args: argparse.Namespace) -> int:
    rows = read_refs(Path(args.refs))
    devices = resolve({"mic": args.mic})
    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    done = [row.id for row in rows if (out / f"{row.id}.wav").exists()]
    todo = [row for row in rows if row.id not in done]
    if done:
        print(f"Already recorded, skipped: {', '.join(done)} (delete a file to record it again).")
    if not todo:
        print("Nothing left to record.")
        return 0
    print(f"Microphone: {audio.describe(devices['mic'])}. {len(todo)} sentence(s) into {out}/.")
    print("Ctrl+C stops at any point; finished files are kept and the same command continues later.")
    with audio.Capture(devices) as capture:
        capture.start(time.monotonic())
        try:
            for number, row in enumerate(todo, start=1):
                record_one(capture, row, out, f"[{number}/{len(todo)}]")
        except KeyboardInterrupt:
            print("\nStopped. Finished files are kept.")
            return 0
    print(f"\nDone: {len(todo)} file(s) in {out}/.")
    return 0


def record_one(capture: audio.Capture, row: RefRow, out: Path, counter: str) -> None:
    while True:
        print(f"\n{counter} {row.id}\n  {row.hy}")
        input("  Enter = start ")
        capture.drain("mic")  # audio from before the start is discarded
        lost_before = capture.stats("mic").overflows
        input("  ● recording… Enter = stop ")
        chunks = capture.drain("mic")
        take = np.concatenate([chunk.samples for chunk in chunks]) if chunks else np.zeros(0, np.float32)
        seconds = len(take) / audio.SAMPLE_RATE
        if seconds < MIN_TAKE_SECONDS:
            print(f"  Only {seconds:.1f} s; record it again.")
            continue
        if not np.any(take):
            raise RecordingError(f"the take is pure digital silence: {PERMISSION_HINT}")
        print(f"  {seconds:.1f} s, peak {audio.dbfs(audio.peak(take))}")
        if capture.stats("mic").overflows > lost_before:
            print("  Audio was lost during this take; record it again.")
            continue
        if input("  Enter = keep, r = record again: ").strip().lower() == "r":
            continue
        write_wav(out / f"{row.id}.wav", take)
        return


# command line


def positive_seconds(text: str) -> float:
    value = float(text)
    if not value > 0:
        raise argparse.ArgumentTypeError("must be more than 0")
    return value


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="python eval/record.py",
        description="Record what M1 delivers into 16 kHz mono WAV files (M8).",
    )
    modes = parser.add_subparsers(dest="mode", required=True)
    device_help = "exact name or index from python -m hyen --list-devices"

    session = modes.add_parser("session", help="both sources into mic.wav and system.wav (T04, E3)")
    session.add_argument("--mic", required=True, metavar="DEVICE", help=f"microphone: {device_help}")
    session.add_argument("--system", required=True, metavar="DEVICE", help=f"system audio: {device_help}")
    session.add_argument("--out", required=True, metavar="DIR", help="folder for mic.wav and system.wav, e.g. data/t04")
    session.add_argument("--seconds", type=positive_seconds, metavar="N", help="stop after N s of microphone audio")
    session.set_defaults(run=record_session)

    prompts = modes.add_parser("prompts", help="one file per sentence of a reference file (E1)")
    prompts.add_argument("--mic", required=True, metavar="DEVICE", help=f"microphone: {device_help}")
    prompts.add_argument("--refs", required=True, metavar="FILE", help="id<TAB>hy<TAB>en with a header row")
    prompts.add_argument("--out", required=True, metavar="DIR", help="folder for <id>.wav, e.g. data/e1")
    prompts.set_defaults(run=record_prompts)
    return parser


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    try:
        return args.run(args)
    except audio.DeviceError as error:
        print(f"error: {error}", file=sys.stderr)
        return 2
    except RecordingError as error:
        print(f"error: {error}", file=sys.stderr)
        return 1


if __name__ == "__main__":
    sys.exit(main())
