"""python -m hyen — the command line (PLAN.md §5, M7).

Session 02 provides two things:
  --list-devices          every input device with its index (T01)
  --mic DEV --system DEV  a capture check: opens both inputs and prints one peak-level
                          line per 2 s of audio on stderr until Ctrl+C (T02)
Session 03 replaces the capture check with the transcription pipeline.
"""

from __future__ import annotations

import argparse
import sys
import time

from hyen import audio

LEVEL_WINDOW_SECONDS = 2


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="python -m hyen",
        description="Live Armenian-to-English call transcription (session 02: capture check).",
    )
    parser.add_argument("--list-devices", action="store_true", help="print every input device with its index and exit")
    parser.add_argument("--mic", metavar="DEVICE", help='microphone: exact name or index, e.g. "MacBook Pro Microphone"')
    parser.add_argument("--system", metavar="DEVICE", help='system audio: exact name or index, e.g. "BlackHole 2ch"')
    return parser


def main(argv: list[str] | None = None) -> int:
    parser = build_parser()
    args = parser.parse_args(argv)
    devices = audio.list_input_devices()
    if args.list_devices:
        if not devices:
            print("error: PortAudio reports no input devices.", file=sys.stderr)
            return 1
        print("Input devices (index, name, input channels, native sample rate):")
        print(audio.format_devices(devices))
        return 0
    missing = [flag for flag, value in (("--mic", args.mic), ("--system", args.system)) if value is None]
    if missing:
        verb = "is" if len(missing) == 1 else "are"
        parser.error(f"{' and '.join(missing)} {verb} required; python -m hyen --list-devices shows the names")
    try:
        mic = audio.resolve_input_device(args.mic, devices)
        system = audio.resolve_input_device(args.system, devices)
        audio.check_distinct(mic, system)
    except audio.DeviceError as error:
        print(f"error: {error}", file=sys.stderr)
        return 2
    return capture_check(mic, system)


def capture_check(mic: audio.InputDevice, system: audio.InputDevice) -> int:
    """Print the peak level of both inputs for every 2 s of microphone audio, until Ctrl+C."""
    print(
        f"Capturing  mic: {audio.describe(mic)}  ·  system: {audio.describe(system)}  → 16 kHz mono",
        file=sys.stderr,
    )
    print(f"Capture check: one level line per {LEVEL_WINDOW_SECONDS} s of audio. Ctrl+C stops.", file=sys.stderr)
    window = LEVEL_WINDOW_SECONDS * audio.SAMPLE_RATE
    with audio.Capture({"mic": mic, "system": system}) as capture:
        capture.start(time.monotonic())
        try:
            counted, mic_peak, system_peak = 0, 0.0, 0.0
            while True:
                chunk = capture.get("mic")
                mic_peak = max(mic_peak, audio.peak(chunk.samples))
                for other in capture.drain("system"):
                    system_peak = max(system_peak, audio.peak(other.samples))
                counted += len(chunk.samples)
                if counted >= window:
                    end = chunk.t_start + len(chunk.samples) / audio.SAMPLE_RATE
                    print(
                        f"[{clock(end)}] mic {audio.dbfs(mic_peak):>11} · system {audio.dbfs(system_peak):>11}",
                        file=sys.stderr,
                    )
                    counted, mic_peak, system_peak = counted - window, 0.0, 0.0
        except KeyboardInterrupt:
            print("Stopped.", file=sys.stderr)
    return 0


def clock(seconds: float) -> str:
    """Session time as HH:MM:SS."""
    whole = int(seconds)
    return f"{whole // 3600:02d}:{whole // 60 % 60:02d}:{whole % 60:02d}"


if __name__ == "__main__":
    sys.exit(main())
