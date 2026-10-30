"""Clock check (S02 diagnostic for D14 and D15): does a device deliver as many samples as time passes?

For each device and rate, this opens one input stream for N seconds and compares the
samples delivered with the time elapsed on the computer's clock — the same "clock gap"
the recorder prints — every 10 s. It also lists the block sizes PortAudio delivered and
any status flags. Nothing is recorded; the content of the audio doesn't matter.

    python eval/clock_check.py --device "MacBook Pro Microphone" --device "BlackHole 2ch" --rates 48000 16000 --seconds 30

At a device's own rate (48000 Hz here) audio goes straight through. At 16000 Hz, Core Audio
converts inside PortAudio (D14). A gap that keeps growing means samples are missing.
"""

from __future__ import annotations

import argparse
import sys
import time
from collections import Counter
from dataclasses import dataclass

import sounddevice as sd

from hyen import audio

REPORT_EVERY = 10  # seconds


@dataclass(frozen=True)
class Result:
    device: audio.InputDevice
    rate: int
    gap_per_minute: float  # seconds of audio missing per minute (negative) or extra (positive)


def check(device: audio.InputDevice, rate: int, seconds: int) -> Result:
    frames_seen = 0
    first: float | None = None
    last = 0.0
    blocks: Counter[int] = Counter()
    flags: Counter[str] = Counter()
    failures: list[BaseException] = []

    def callback(_indata, frames, _time, status) -> None:
        nonlocal frames_seen, first, last
        try:
            now = time.monotonic()
            if first is None:
                first = now - frames / rate
            frames_seen += frames
            last = now
            blocks[frames] += 1
            if status.input_overflow:
                flags["input overflow"] += 1
            if status.input_underflow:
                flags["input underflow"] += 1
        except BaseException as error:
            failures.append(error)  # raised in the main thread; sounddevice alone would only print it
            raise sd.CallbackAbort from error

    def gap(stream: sd.InputStream) -> float:
        if failures:
            raise RuntimeError(f"the callback for {audio.describe(device)} failed") from failures[0]
        if not stream.active:
            raise RuntimeError(f"the stream on {audio.describe(device)} stopped by itself")
        if first is None:
            raise RuntimeError(f"no audio arrived from {audio.describe(device)}; check Terminal's microphone permission")
        return first + frames_seen / rate - last

    settings = sd.CoreAudioSettings(
        change_device_parameters=False, fail_if_conversion_required=False, conversion_quality="max"
    )
    print(f"\n{audio.describe(device)} opened at {rate} Hz", flush=True)
    with sd.InputStream(
        device=device.index,
        channels=device.channels,
        samplerate=rate,
        dtype="float32",
        callback=callback,
        extra_settings=settings,
    ) as stream:
        start = time.monotonic()
        for mark in range(REPORT_EVERY, seconds + 1, REPORT_EVERY):
            time.sleep(max(0.0, start + mark - time.monotonic()))
            final_gap = gap(stream)
            print(f"  {mark:4d} s   samples {frames_seen:>9}   clock gap {final_gap:+.4f} s", flush=True)
    per_minute = final_gap / (last - first) * 60
    common = ", ".join(f"{size} frames ×{count}" for size, count in blocks.most_common(3))
    print(f"  blocks: {common}   flags: {dict(flags) or 'none'}")
    return Result(device, rate, per_minute)


def multiple_of_ten(text: str) -> int:
    value = int(text)
    if value < REPORT_EVERY or value % REPORT_EVERY:
        raise argparse.ArgumentTypeError(f"must be a multiple of {REPORT_EVERY}")
    return value


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        prog="python eval/clock_check.py",
        description="Compare the samples an input device delivers with the time that passes (D14, D15).",
    )
    parser.add_argument("--device", action="append", required=True, metavar="DEVICE", help="exact name or index; repeat for several")
    parser.add_argument("--rates", type=int, nargs="+", required=True, metavar="HZ", help="sample rates to open the device at")
    parser.add_argument("--seconds", type=multiple_of_ten, required=True, metavar="N", help="seconds per run, a multiple of 10")
    args = parser.parse_args(argv)
    devices = audio.list_input_devices()
    try:
        chosen = [audio.resolve_input_device(spec, devices) for spec in args.device]
    except audio.DeviceError as error:
        print(f"error: {error}", file=sys.stderr)
        return 2
    results = [check(device, rate, args.seconds) for device in chosen for rate in args.rates]
    print("\nSummary — clock gap per minute (0 means no samples missing)")
    for result in results:
        print(f"  {result.device.name:<28}{result.rate:>6} Hz   {result.gap_per_minute:+.3f} s per minute")
    return 0


if __name__ == "__main__":
    sys.exit(main())
