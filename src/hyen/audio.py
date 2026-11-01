"""M1 · audio capture (PLAN.md §5).

Opens one input stream per source — the MacBook microphone ("mic") and BlackHole 2ch
("system") — and turns what PortAudio delivers into AudioChunk messages: float32 mono
at 16 kHz, 512 samples each, with a start time on the session clock.

Each source has its own reader thread (D17). PortAudio buffers the incoming audio in C,
so Python never runs on Core Audio's audio thread; the reader takes 512 samples at a time.
If the reader ever falls further behind than PortAudio's buffer holds, PortAudio reports
an overflow, and M1 prints it.

Decisions: D2 (BlackHole is read as an ordinary input device), D8 (one capture reader per
device, one queue per source), D14 (Core Audio converts to 16 kHz inside PortAudio),
D15 (chunk start times are counted from samples), D17 (reader threads, 512-sample blocks).
"""

from __future__ import annotations

import math
import queue
import sys
import threading
import time
from collections.abc import Callable
from dataclasses import dataclass
from typing import Self

import numpy as np
import sounddevice as sd

SAMPLE_RATE = 16_000
BLOCK = 512  # samples per AudioChunk: 32 ms at 16 kHz (D17)
LATENCY = 0.5  # seconds; PortAudio sizes its input buffer from this (D17)
SOURCES = ("mic", "system")
STALL_SECONDS = 2.0  # a running stream that delivers nothing for this long is an error (F02, D27)
ZERO_SECONDS = 5.0  # the mic delivering only exact zeros for this long is an error (F01, D26)


@dataclass(frozen=True)
class AudioChunk:
    """A block of audio from one source, as M2 receives it."""

    source: str  # "mic" or "system"
    samples: np.ndarray  # float32, mono, 16 kHz, BLOCK samples
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
    overflows: int  # how often PortAudio reported lost audio
    start: float | None  # session time of the source's first sample (D15)
    clock_gap: float | None  # newest sample's sample-count time minus the computer clock, in seconds (D15)


class DeviceError(Exception):
    """A device given on the command line can't be used; the message says why and what to do."""


class CaptureError(Exception):
    """A capture stream failed: reading raised an error, or the stream stopped by itself."""


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
    """One input stream for one source, its reader thread, and the queue its chunks go to."""

    def __init__(
        self,
        name: str,
        device: InputDevice,
        clock: Callable[[], float],
        stall_seconds: float = STALL_SECONDS,
        zero_seconds: float | None = None,
        stall_clock: Callable[[], float] = time.monotonic,
    ) -> None:
        self.name = name
        self.stall_clock = stall_clock  # its own clock, so the session clock is read once per block, as before
        self.device = device
        self.clock = clock
        self.stall_seconds = stall_seconds
        self.zero_seconds = zero_seconds  # None: no zero check (BlackHole is legitimately silent)
        self.zero_run = 0  # exact-zero samples in a row
        self.last_block_at = 0.0  # stall_clock() when the latest block was read
        self.queue: queue.Queue[AudioChunk | CaptureError] = queue.Queue()
        self.stream: sd.InputStream | None = None
        self.thread: threading.Thread | None = None
        self.t0 = 0.0  # the session clock's zero, read from clock()
        self.anchor: float | None = None  # session time of this source's first sample (D15)
        self.samples_seen = 0
        self.newest = 0  # samples captured so far, read or still in PortAudio's buffer, at the latest read
        self.newest_at = 0.0  # session time of the latest read
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
            self.stream = sd.InputStream(  # no callback: PortAudio buffers, we read (D17)
                device=self.device.index,
                channels=self.device.channels,
                samplerate=SAMPLE_RATE,
                dtype="float32",
                latency=LATENCY,
                finished_callback=self.finished,
                extra_settings=settings,
            )
        except sd.PortAudioError as error:
            raise CaptureError(
                f"{self.name}: cannot open {describe(self.device)} at {SAMPLE_RATE} Hz: {error}"
            ) from error

    def start(self) -> None:
        self.stream.start()
        self.thread = threading.Thread(target=self.read_loop, name=f"capture-{self.name}", daemon=True)
        self.thread.start()

    def read_loop(self) -> None:
        """The reader thread: takes BLOCK samples at a time from PortAudio's buffer (D17).

        It sleeps until a full block is there instead of waiting inside PortAudio, so it
        never gets stuck in C and stopping is always clean.

        If PortAudio's buffer stays short of a block for stall_seconds, the stream has stopped
        delivering without PortAudio saying so; that is an error (F02, D27). A reader that is
        merely late finds a full buffer when it wakes, so lateness never trips this check.
        """
        try:
            self.last_block_at = self.stall_clock()
            while not self.stopping:
                available = self.stream.read_available
                if available < BLOCK:
                    self.check_stall()
                    time.sleep((BLOCK - available) / SAMPLE_RATE)
                    continue
                now = self.clock() - self.t0
                block, overflowed = self.stream.read(BLOCK)
                self.last_block_at = self.stall_clock()
                self.add_block(block, overflowed, now, available)
        except CaptureError as failure:
            self.queue.put(failure)
        except BaseException as error:  # noqa: BLE001 — every error must reach the consumer, which raises it
            failure = CaptureError(f"{self.name}: reading {describe(self.device)} failed: {error!r}")
            failure.__cause__ = error
            self.queue.put(failure)

    def check_stall(self) -> None:
        waited = self.stall_clock() - self.last_block_at
        if waited > self.stall_seconds:
            raise CaptureError(
                f"{self.name}: no audio from {describe(self.device)} for {waited:.1f} s while its stream is running. "
                "The device was probably disconnected or the audio routing changed "
                "(for BlackHole: the Multi-Output Device). Check the devices, then start again."
            )

    def check_zeros(self, block: np.ndarray) -> None:
        """Exact digital zeros for zero_seconds: a microphone never does this when it works (F01, D26)."""
        if self.zero_seconds is None:
            return
        self.zero_run = self.zero_run + len(block) if not np.any(block) else 0
        if self.zero_run >= self.zero_seconds * SAMPLE_RATE:
            raise CaptureError(
                f"{self.name}: {describe(self.device)} has delivered only digital silence (exact zeros) "
                f"for {self.zero_run / SAMPLE_RATE:.1f} s. Most likely Terminal has no microphone permission: "
                "System Settings → Privacy & Security → Microphone → turn on Terminal, then quit and reopen Terminal. "
                "Also check that the input volume (System Settings → Sound → Input) is not at zero."
            )

    def add_block(self, block: np.ndarray, overflowed: bool, now: float, available: int) -> None:
        """Turn one block into an AudioChunk. Times come from sample counts (D15).

        `now` is the session time at which PortAudio's buffer held `available` samples,
        this block first; the newest of them had just arrived.
        """
        if available < len(block):
            raise ValueError(f"{available} samples available, but a block of {len(block)} was read")
        if overflowed:
            self.overflows += 1
        self.check_zeros(block)
        if self.anchor is None:
            # Starting a Core Audio stream can return after audio is already flowing, so the
            # buffer may hold more than this block: all of it was captured before `now`.
            self.anchor = now - available / SAMPLE_RATE
        t_start = self.anchor + self.samples_seen / SAMPLE_RATE
        samples = to_mono(block)
        self.newest = self.samples_seen + available
        self.newest_at = now
        self.samples_seen += len(block)
        self.queue.put(AudioChunk(self.name, samples, t_start))

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
            gap = self.anchor + self.newest / SAMPLE_RATE - self.newest_at
        return SourceStats(self.samples_seen, self.overflows, self.anchor, gap)


class Capture:
    """One input stream and one reader thread per source (D8, D17), each feeding its own queue.

    Use it as a context manager so the streams are always closed:

        with Capture({"mic": mic, "system": system}) as capture:
            capture.start(time.monotonic())
            chunk = capture.get("mic")
    """

    def __init__(
        self,
        devices: dict[str, InputDevice],
        clock: Callable[[], float] = time.monotonic,
        stall_seconds: float = STALL_SECONDS,
        zero_seconds: float = ZERO_SECONDS,
    ) -> None:
        unknown = sorted(set(devices) - set(SOURCES))
        if unknown:
            raise ValueError(f"unknown source names {unknown}; expected {SOURCES}")
        # Only the mic gets the zero check: BlackHole delivers zeros whenever nothing plays.
        self._sources = {
            name: _Source(name, device, clock, stall_seconds, zero_seconds if name == "mic" else None)
            for name, device in devices.items()
        }

    def start(self, t0: float) -> None:
        """Open every stream, then start them back to back. t0 is the session clock's zero."""
        for source in self._sources.values():
            source.open(t0)
        for source in self._sources.values():
            source.start()

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
        """Let the readers finish their current block, then stop and close the streams.

        Errors while stopping are raised, not ignored.
        """
        for source in self._sources.values():
            source.stopping = True
        for source in self._sources.values():
            if source.thread is not None:
                source.thread.join()
                source.thread = None
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
