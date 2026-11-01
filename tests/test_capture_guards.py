"""M1 guards against silent failure (TESTS.md F01, F02; D26, D27), with a fake PortAudio stream.

F01: a microphone that delivers only exact digital zeros (e.g. no microphone permission)
stops the program with a message naming the likely cause; BlackHole is not checked,
because it delivers zeros whenever nothing plays.
F02: a running stream that stops delivering audio stops the program; a reader that is
merely late (the buffer is full when it wakes) does not.
"""

from __future__ import annotations

import time

import numpy as np

from hyen.audio import BLOCK, Capture, CaptureError, InputDevice, _Source

MIC = InputDevice(3, "MacBook Pro Microphone", 1, 48000.0)
BLACKHOLE = InputDevice(2, "BlackHole 2ch", 2, 48000.0)


def block(value: float, channels: int = 1) -> np.ndarray:
    return np.full((BLOCK, channels), value, dtype=np.float32)


def queued(source: _Source) -> list:
    items = []
    while not source.queue.empty():
        items.append(source.queue.get_nowait())
    return items


class FakeStream:
    """Always has `available` samples; read() returns blocks of `value`; stops the source after `reads`."""

    def __init__(self, source: _Source, value: float, available: int, reads: int | None = None) -> None:
        self.source, self.value, self.available, self.reads = source, value, available, reads
        self.count = 0

    @property
    def read_available(self) -> int:
        return self.available

    def read(self, frames: int):
        self.count += 1
        if self.reads is not None and self.count >= self.reads:
            self.source.stopping = True
        return block(self.value, self.source.device.channels)[:frames], False


def test_mic_of_exact_zeros_stops_with_the_permission_hint():  # F01
    source = _Source("mic", MIC, time.monotonic, zero_seconds=0.1)  # 1600 samples → the 4th block
    source.stream = FakeStream(source, 0.0, available=BLOCK, reads=100)
    source.read_loop()
    items = queued(source)
    assert len(items) == 4 and all(not isinstance(item, CaptureError) for item in items[:3])
    error = items[3]
    assert isinstance(error, CaptureError)
    assert "digital silence" in str(error) and "Microphone" in str(error) and "MacBook Pro Microphone" in str(error)


def test_a_quiet_but_real_mic_is_fine():
    source = _Source("mic", MIC, time.monotonic, zero_seconds=0.1)
    for index in range(50):  # 1.6 s; a real noise floor is never exactly zero
        source.add_block(block(1e-6), False, index * 0.032, BLOCK)
    assert not any(isinstance(item, CaptureError) for item in queued(source))


def test_zero_run_restarts_after_any_nonzero_block():
    source = _Source("mic", MIC, time.monotonic, zero_seconds=0.1)
    for index in range(30):  # 3 zero blocks, 1 noise block, repeated: never 4 zero blocks in a row
        source.add_block(block(1e-6 if index % 4 == 3 else 0.0), False, index * 0.032, BLOCK)
    assert not any(isinstance(item, CaptureError) for item in queued(source))


def test_capture_checks_zeros_on_the_mic_only():
    capture = Capture({"mic": MIC, "system": BLACKHOLE})
    assert capture._sources["mic"].zero_seconds == 5.0
    assert capture._sources["system"].zero_seconds is None


def test_blackhole_zeros_are_fine():
    source = Capture({"mic": MIC, "system": BLACKHOLE})._sources["system"]
    for index in range(400):  # 12.8 s of zeros: nothing plays
        source.add_block(block(0.0, 2), False, index * 0.032, BLOCK)
    assert not any(isinstance(item, CaptureError) for item in queued(source))


def test_stream_that_stops_delivering_is_an_error():  # F02
    source = _Source("system", BLACKHOLE, time.monotonic, stall_seconds=0.1)
    source.stream = FakeStream(source, 0.0, available=0)
    started = time.monotonic()
    source.read_loop()  # returns once the stall is detected
    assert time.monotonic() - started < 1.0
    (error,) = queued(source)
    assert isinstance(error, CaptureError) and "no audio from" in str(error) and "BlackHole 2ch" in str(error)


def test_a_late_reader_is_not_a_stall():
    """The clock jumps 5 s between calls, but a full block is always waiting: no stall."""
    ticks = iter(range(0, 10_000, 5))
    source = _Source("system", BLACKHOLE, time.monotonic, stall_seconds=2.0, stall_clock=lambda: float(next(ticks)))
    source.stream = FakeStream(source, 0.1, available=BLOCK * 4, reads=20)
    source.read_loop()
    items = queued(source)
    assert len(items) == 20 and not any(isinstance(item, CaptureError) for item in items)


def test_default_limits():  # D26, D27
    from hyen import audio

    assert audio.ZERO_SECONDS == 5.0 and audio.STALL_SECONDS == 2.0
