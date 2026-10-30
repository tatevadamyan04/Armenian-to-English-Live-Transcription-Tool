"""Unit tests for M1's pure logic, derived from T01–T03, D15 and the fail-loud rule (CLAUDE.md).

No audio device is opened: the callback is called directly with arrays.
"""

from types import SimpleNamespace

import numpy as np
import pytest
import sounddevice as sd

from hyen import audio

DEVICES = [
    audio.InputDevice(0, "MacBook Pro Microphone", 1, 48000.0),
    audio.InputDevice(1, "BlackHole 2ch", 2, 48000.0),
    audio.InputDevice(4, "ZoomAudioDevice", 2, 48000.0),
]
OK = SimpleNamespace(input_overflow=False)
OVERFLOW = SimpleNamespace(input_overflow=True)


# T01 · the device list shows index and name


def test_device_list_shows_index_and_name():
    listing = audio.format_devices(DEVICES)
    assert "[0] MacBook Pro Microphone" in listing
    assert "[1] BlackHole 2ch" in listing


# T02 · an exact name or an index selects the device


@pytest.mark.parametrize(("spec", "index"), [("MacBook Pro Microphone", 0), ("BlackHole 2ch", 1), ("4", 4)])
def test_exact_name_or_index_selects_the_device(spec, index):
    assert audio.resolve_input_device(spec, DEVICES).index == index


# T03 · anything else is an error that names it and lists the inputs; nothing is guessed


@pytest.mark.parametrize("spec", ["No Such Mic", "BlackHole", "blackhole 2ch", "9", ""])
def test_unknown_device_is_an_error(spec):
    with pytest.raises(audio.DeviceError) as error:
        audio.resolve_input_device(spec, DEVICES)
    message = str(error.value)
    assert repr(spec) in message
    assert "MacBook Pro Microphone" in message
    assert "BlackHole 2ch" in message


def test_duplicate_names_must_be_given_by_index():
    devices = [*DEVICES, audio.InputDevice(7, "BlackHole 2ch", 2, 48000.0)]
    with pytest.raises(audio.DeviceError, match="index"):
        audio.resolve_input_device("BlackHole 2ch", devices)


def test_mic_and_system_must_be_different_devices():
    with pytest.raises(audio.DeviceError, match="different devices"):
        audio.check_distinct(DEVICES[0], DEVICES[0])


# D15 · chunk times are counted from samples, not taken from when callbacks happen


def make_source(callback_times):
    ticks = iter(callback_times)
    source = audio._Source("mic", DEVICES[0], clock=lambda: next(ticks))
    source.t0 = 100.0
    return source


def next_chunk(source):
    return source.take(source.queue.get_nowait())


def test_chunk_times_are_counted_from_samples():
    source = make_source([100.5, 100.537, 100.561])  # callbacks arrive irregularly
    for _ in range(3):
        source.callback(np.zeros((512, 1), np.float32), 512, None, OK)
    times = [next_chunk(source).t_start for _ in range(3)]
    assert times == pytest.approx([0.468, 0.500, 0.532])


def test_chunks_are_float32_mono_copies():
    source = make_source([100.1])
    block = np.array([[0.5, -0.5], [1.0, 0.0]], np.float32)
    source.callback(block, 2, None, OK)
    block[:] = 9.0  # PortAudio reuses its buffer after the callback
    chunk = next_chunk(source)
    assert chunk.samples.dtype == np.float32
    assert chunk.samples.tolist() == [0.0, 0.5]


def test_clock_gap_compares_sample_time_with_the_computer_clock():
    source = make_source([100.032, 101.040])  # the second callback is 8 ms later than the samples say
    source.callback(np.zeros((512, 1), np.float32), 512, None, OK)
    source.callback(np.zeros((16000, 1), np.float32), 16000, None, OK)
    assert source.stats().clock_gap == pytest.approx(-0.008)


# Fail loudly (CLAUDE.md) · lost audio, callback errors and dead streams all surface


def test_overflow_is_reported_on_stderr(capsys):
    source = make_source([100.1])
    source.callback(np.zeros((512, 1), np.float32), 512, None, OVERFLOW)
    next_chunk(source)
    assert "input overflow" in capsys.readouterr().err
    assert source.stats().overflows == 1


def test_callback_error_reaches_the_consumer():
    source = make_source([100.1])
    with pytest.raises(sd.CallbackAbort):
        source.callback(np.zeros(512, np.float32), 512, None, OK)  # not a (frames, channels) block
    with pytest.raises(audio.CaptureError, match="callback failed"):
        next_chunk(source)


def test_stream_that_stops_by_itself_is_an_error():
    source = make_source([])
    source.finished()
    with pytest.raises(audio.CaptureError, match="stopped by itself"):
        next_chunk(source)


def test_stopping_on_purpose_is_not_an_error():
    source = make_source([])
    source.stopping = True
    source.finished()
    assert source.queue.empty()
