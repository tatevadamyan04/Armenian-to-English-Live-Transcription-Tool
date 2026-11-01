"""Join pieces of recordings into one WAV for a replay test through BlackHole (S07, P01 replacement).

    python eval/make_replay.py OUT.wav FILE:START:END [FILE:START:END ...]

START and END are in seconds within FILE. Pieces are joined in the given order with
1 s of silence between them. Standard library only. All inputs must have the same
sample rate, channel count and sample width; anything else is an error (no conversion).
Prints each piece's place in the output, so labels can be checked against the log.
"""

from __future__ import annotations

import sys
import wave
from pathlib import Path

GAP_SECONDS = 1.0


def parse(spec: str) -> tuple[Path, float, float]:
    try:
        name, start, end = spec.rsplit(":", 2)
        start_s, end_s = float(start), float(end)
    except ValueError:
        raise SystemExit(f"error: {spec!r} is not FILE:START:END (seconds)")
    if not 0 <= start_s < end_s:
        raise SystemExit(f"error: {spec!r}: need 0 <= START < END")
    return Path(name), start_s, end_s


def main(argv: list[str]) -> int:
    if len(argv) < 2:
        print(__doc__, file=sys.stderr)
        return 2
    out_path = Path(argv[0])
    pieces = [parse(spec) for spec in argv[1:]]
    params = None
    frames: list[bytes] = []
    position = 0.0
    for path, start, end in pieces:
        if not path.is_file():
            raise SystemExit(f"error: {path} not found")
        with wave.open(str(path), "rb") as source:
            shape = (source.getnchannels(), source.getsampwidth(), source.getframerate())
            if params is None:
                params = shape
            elif shape != params:
                raise SystemExit(
                    f"error: {path} has channels/width/rate {shape}, the first file has {params}; "
                    "all inputs must match"
                )
            rate = source.getframerate()
            length = source.getnframes() / rate
            if end > length:
                raise SystemExit(f"error: {path} is {length:.1f} s long; piece ends at {end:.1f} s")
            source.setpos(int(start * rate))
            frames.append(source.readframes(int((end - start) * rate)))
        print(f"{position:7.1f}–{position + end - start:7.1f} s  {path}  [{start:.1f}–{end:.1f} s]")
        position += end - start
        channels, width, rate = params
        frames.append(b"\x00" * int(GAP_SECONDS * rate) * channels * width)
        position += GAP_SECONDS
    channels, width, rate = params
    with wave.open(str(out_path), "wb") as out:
        out.setnchannels(channels)
        out.setsampwidth(width)
        out.setframerate(rate)
        out.writeframes(b"".join(frames))
    print(f"wrote {out_path}: {position:.1f} s, {channels} ch, {rate} Hz")
    return 0


if __name__ == "__main__":
    sys.exit(main(sys.argv[1:]))
