"""python -m hyen — the command line (PLAN.md §5, M7).

  --list-devices          every input device with its index (T01)
  --mic DEV --system DEV  the live session: transcribe and translate both sources until Ctrl+C
"""

from __future__ import annotations

import os

os.environ["HF_HUB_OFFLINE"] = "1"  # local only at runtime (D12); set before any model library is imported

import argparse
import sys

from hyen import audio


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="python -m hyen",
        description="Live Armenian-to-English call transcription.",
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

    from hyen.modelfiles import ModelMissingError
    from hyen.pipeline import run_live

    try:
        return run_live(mic, system)
    except ModelMissingError as error:  # F03: name the model and the download command, exit non-zero
        print(f"error: {error}", file=sys.stderr)
        return 1


if __name__ == "__main__":
    sys.exit(main())
