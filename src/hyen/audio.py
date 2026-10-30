"""M1 · audio capture (PLAN.md §5).

Opens one input stream per source — the MacBook microphone ("mic") and BlackHole 2ch
("system") — and turns what PortAudio delivers into AudioChunk messages: float32 mono
at 16 kHz with a start time on the session clock.

Decisions: D2 (BlackHole is read as an ordinary input device), D8 (one capture callback
per device, one queue per source), D14 (Core Audio converts to 16 kHz inside PortAudio),
D15 (chunk start times are counted from samples).
"""

from __future__ import annotations

import math
import queue
import sys
import time
from collections.abc import Callable
from dataclasses import dataclass
from typing import Self

import numpy as np
import sounddevice as sd

SAMPLE_RATE = 16_000
SOURCES = ("mic", "system")


@dataclass(frozen=True)
class AudioChunk:
    """A block of audio from one source, as M2 receives it."""

    source: str  # "mic" or "system"
    samples: np.ndarray  # float32, mono, 16 kHz
    t_start: float  # seconds since the session started (D15)


@dataclass(frozen=True)
class InputDevice:
    index: int  # PortAudio's device index
    name: str
    channels: int  # input channels the device offers; all are averaged to mono
    native_rate: float  # the device's own sample rate in Hz


@dataclass(frozen=True)
class SourceStats:
    samples: int  # samples delivered so far
    overflows: int  # how often audio was lost before it reached us
    clock_gap: float | None  # sample-count time minus computer-clock time, in seconds (D15)


class DeviceError(Exception):
    """A device given on the command line can't be used; the message says why and what to do."""


class CaptureError(Exception):
    """A capture stream failed: its callback raised, or the stream stopped by itself."""


# Devices


def list_input_devices() -> list[InputDevice]:
    """Every device PortAudio can record from, in PortAudio's order (T01)."""
    devices = []
    for index, info in enumerate(sd.query_devices()):
        if info["max_input_channels"] > 0:
            devices.append(
                InputDevice(
                    index=index,
                    name=info["name"],
                    channels=int(info["max_input_channels"]),
                    native_rate=float(info["default_samplerate"]),
                )
            )
    return devices


def describe(device: InputDevice) -> str:
    return f"[{device.index}] {device.name}  ({device.channels} ch, {device.native_rate:g} Hz)"


def format_devices(devices: list[InputDevice]) -> str:
    return "\n".join(f"  {describe(device)}" for device in devices)


def resolve_input_device(spec: str, devices: list[InputDevice]) -> InputDevice:
    """Find a device by its exact name, or by its index; anything else is an error (T03).

    sounddevice on its own would also accept part of a name. This function deliberately
    doesn't, so a typo can never select a different device.
    """
    named = [device for device in devices if device.name == spec]
    if len(named) == 1:
        return named[0]
    if len(named) > 1:
        indices = ", ".join(str(device.index) for device in named)
        raise DeviceError(f"{len(named)} input devices are named {spec!r} (indices {indices}); give the index instead.")
    if spec.isdecimal():
        numbered = [device for device in devices if device.index == int(spec)]
        if numbered:
            return numbered[0]
    raise DeviceError(
        f"no input device {spec!r}.\nUse the exact name in quotes, or the index. Input devices:\n"
        + format_devices(devices)
    )


def check_distinct(mic: InputDevice, system: InputDevice) -> None:
    """One device for both sources would print every phrase twice; R1 needs two inputs."""
    if mic.index == system.index:
        raise DeviceError(
            f"--mic and --system both point to {describe(mic)}; "
            "they must be two different devices (the microphone and BlackHole 2ch)."
        )


# Capture


def to_mono(block: np.ndarray) -> np.ndarray:
    """Average a (frames, channels) block into a new float32 mono array."""
    if block.ndim != 2:
        raise ValueError(f"expected a (frames, channels) block, got shape {block.shape}")
    return block.mean(axis=1, dtype=np.float32)


class _Source:
    """One input stream for one source, and the queue its chunks go to."""

    def __init__(self, name: str, device: InputDevice, clock: Callable[[], float]) -> None:
        self.name = name
        self.device = device
        self.clock = clock
        self.queue: queue.Queue[AudioChunk | CaptureError] = queue.Queue()
        self.stream: sd.InputStream | None = None
        self.t0 = 0.0  # the session clock's zero, read from clock()
        self.anchor: float | None = None  # session time of this source's first sample (D15)
        self.samples_seen = 0
        self.last_callback = 0.0  # session time of the latest callback
        self.overflows = 0
        self.reported_overflows = 0
        self.stopping = False

    def open(self, t0: float) -> None:
        self.t0 = t0
        settings = sd.CoreAudioSettings(
            change_device_parameters=False,  # never change the device's own settings (D14)
            fail_if_conversion_required=False,  # Core Audio converts to 16 kHz (D14)
            conversion_quality="max",
        )
        try:
            self.stream = sd.InputStream(
                device=self.device.index,
                channels=self.device.channels,
                samplerate=SAMPLE_RATE,
                dtype="float32",
                callback=self.callback,
                finished_callback=self.finished,
                extra_settings=settings,
            )
        except sd.PortAudioError as error:
            raise CaptureError(
                f"{self.name}: cannot open {describe(self.device)} at {SAMPLE_RATE} Hz: {error}"
            ) from error

    def callback(self, indata: np.ndarray, frames: int, _time: object, status: sd.CallbackFlags) -> None:
        """Runs on PortAudio's audio thread for every block. Times come from sample counts (D15)."""
        try:
            now = self.clock() - self.t0
            if status.input_overflow:
                self.overflows += 1
            if self.anchor is None:
                self.anchor = now - frames / SAMPLE_RATE
            t_start = self.anchor + self.samples_seen / SAMPLE_RATE
            samples = to_mono(indata)
            self.samples_seen += frames
            self.last_callback = now
            self.queue.put(AudioChunk(self.name, samples, t_start))
        except BaseException as error:
            # sounddevice would only print this and let the program carry on, so the error
            # goes to the consumer, which raises it, and the stream is aborted.
            failure = CaptureError(f"{self.name}: the capture callback failed: {error!r}")
            failure.__cause__ = error
            self.queue.put(failure)
            raise sd.CallbackAbort from error

    def finished(self) -> None:
        """Called by sounddevice whenever the stream becomes inactive."""
        if not self.stopping:
            self.queue.put(
                CaptureError(
                    f"{self.name}: the input stream on {describe(self.device)} stopped by itself. "
                    "Check that the device is still connected and the audio routing unchanged, then start again."
                )
            )

    def take(self, item: AudioChunk | CaptureError) -> AudioChunk:
        self.report_overflows()
        if isinstance(item, CaptureError):
            raise item
        return item

    def report_overflows(self) -> None:
        count = self.overflows
        if count > self.reported_overflows:
            self.reported_overflows = count
            print(f"warning: {self.name}: input overflow, audio was lost ({count} so far)", file=sys.stderr)

    def stats(self) -> SourceStats:
        gap = None
        if self.anchor is not None:
            gap = self.anchor + self.samples_seen / SAMPLE_RATE - self.last_callback
        return SourceStats(self.samples_seen, self.overflows, gap)


class Capture:
    """One input stream per source (D8), each feeding its own queue.

    Use it as a context manager so the streams are always closed:

        with Capture({"mic": mic, "system": system}) as capture:
            capture.start(time.monotonic())
            chunk = capture.get("mic")
    """

    def __init__(self, devices: dict[str, InputDevice], clock: Callable[[], float] = time.monotonic) -> None:
        unknown = sorted(set(devices) - set(SOURCES))
        if unknown:
            raise ValueError(f"unknown source names {unknown}; expected {SOURCES}")
        self._sources = {name: _Source(name, device, clock) for name, device in devices.items()}

    def start(self, t0: float) -> None:
        """Open every stream, then start them back to back. t0 is the session clock's zero."""
        for source in self._sources.values():
            source.open(t0)
        for source in self._sources.values():
            source.stream.start()

    def get(self, name: str) -> AudioChunk:
        """The next chunk of a source, waiting for it. Raises CaptureError if the stream failed."""
        source = self._sources[name]
        return source.take(source.queue.get())

    def drain(self, name: str) -> list[AudioChunk]:
        """Every chunk of a source that has arrived so far, without waiting."""
        source = self._sources[name]
        chunks = []
        while True:
            try:
                item = source.queue.get_nowait()
            except queue.Empty:
                source.report_overflows()
                return chunks
            chunks.append(source.take(item))

    def stats(self, name: str) -> SourceStats:
        return self._sources[name].stats()

    def stop(self) -> None:
        """Stop and close every stream; errors while closing are raised, not ignored."""
        for source in self._sources.values():
            source.stopping = True
        for source in self._sources.values():
            if source.stream is not None:
                if source.stream.active:
                    source.stream.stop(ignore_errors=False)
                source.stream.close(ignore_errors=False)
                source.stream = None

    def __enter__(self) -> Self:
        return self

    def __exit__(self, *exc_info: object) -> None:
        self.stop()


# Levels, for diagnostics on stderr


def peak(samples: np.ndarray) -> float:
    return float(np.max(np.abs(samples))) if samples.size else 0.0


def dbfs(level: float) -> str:
    """A peak level as text: "silent" for digital zeros, otherwise dB relative to full scale."""
    return "silent" if level <= 0.0 else f"{20 * math.log10(level):.1f} dBFS"
