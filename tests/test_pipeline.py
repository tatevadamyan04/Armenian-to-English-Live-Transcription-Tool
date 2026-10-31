"""M6 output and M7 pipeline with fake capture and fake models (PLAN.md §8).

From the requirements: the three-line layout with the speech-start time (R9, T12),
"Me" for every mic segment (R6, T9) and the placeholder label for the call side (D21),
empty ASR text dropped, and an exception in any thread stopping the program (F04).
"""

from __future__ import annotations

import io
import threading
import time

import numpy as np
import pytest

from hyen.audio import AudioChunk, CaptureError
from hyen.output import Result, clock, format_result
from hyen.pipeline import Pipeline
from hyen.segmenter import CHUNK, CHUNK_SECONDS, Segmenter


def test_layout():  # T12
    result = Result("system-1", 72.9, "Speaker 1", "Բարև, ինչպե՞ս ես։", "Hi, how are you?")
    assert format_result(result) == "[00:01:12] Speaker 1\n  HY: Բարև, ինչպե՞ս ես։\n  EN: Hi, how are you?\n"


def test_clock():
    assert clock(0) == "00:00:00"
    assert clock(3725.9) == "01:02:05"


def loud(samples: np.ndarray) -> float:
    return 0.9 if np.max(np.abs(samples)) >= 0.1 else 0.05


class FakeCapture:
    """Delivers one phrase per listed source (1 s of speech, then silence), then blocks."""

    def __init__(self, speaking: set[str], fail: str | None = None) -> None:
        self.speaking = speaking
        self.fail = fail
        self.index = {"mic": 0, "system": 0}
        self.never = threading.Event()

    def get(self, name: str) -> AudioChunk:
        if name == self.fail:
            raise CaptureError(f"{name}: the input stream stopped by itself")
        index = self.index[name]
        if index >= 80:
            self.never.wait()  # nothing more from this source
        self.index[name] = index + 1
        value = 0.2 if name in self.speaking and 5 <= index < 36 else 0.0
        return AudioChunk(name, np.full(CHUNK, value, dtype=np.float32), index * CHUNK_SECONDS)


class FakeRecognizer:
    def __init__(self, text: str = "Բարև", error: Exception | None = None) -> None:
        self.text, self.error = text, error

    def transcribe(self, samples: np.ndarray) -> str:
        if self.error is not None:
            raise self.error
        return self.text


class FakeTranslator:
    def translate(self, hy: str) -> str:
        return f"EN({hy})"


def make_pipeline(capture, recognizer, logs, out):
    segmenters = {name: Segmenter(name, loud, log=logs.append) for name in ("mic", "system")}
    return Pipeline(capture, segmenters, recognizer, FakeTranslator(), time.monotonic, time.monotonic(), out, logs.append)


def run_briefly(pipeline, seconds: float = 0.5):
    """Run in a thread; return the error it raised, or None if it was still running after `seconds`."""
    raised: list[BaseException] = []
    thread = threading.Thread(target=lambda: raised.append(_catch(pipeline.run)), daemon=True)
    thread.start()
    thread.join(seconds)
    return raised[0] if raised else None


def _catch(function):
    try:
        function()
    except BaseException as error:  # noqa: BLE001
        return error


def test_both_sources_are_printed_with_their_labels():  # T06 format, T09, D21
    out, logs = io.StringIO(), []
    run_briefly(make_pipeline(FakeCapture({"mic", "system"}), FakeRecognizer(), logs, out))
    text = out.getvalue()
    assert "[00:00:00] Me\n  HY: Բարև\n  EN: EN(Բարև)\n" in text
    assert "[00:00:00] Speaker 1\n  HY: Բարև\n  EN: EN(Բարև)\n" in text
    assert sum(line.startswith("timing ") for line in logs) == 2


def test_empty_asr_text_is_dropped_with_a_note():
    out, logs = io.StringIO(), []
    run_briefly(make_pipeline(FakeCapture({"mic"}), FakeRecognizer(text=""), logs, out))
    assert out.getvalue() == ""
    assert any("no text" in line for line in logs)


def test_error_in_the_worker_stops_the_pipeline():  # F04
    out, logs = io.StringIO(), []
    error = run_briefly(make_pipeline(FakeCapture({"mic"}), FakeRecognizer(error=RuntimeError("asr broke")), logs, out))
    assert isinstance(error, RuntimeError) and "asr broke" in str(error)
    assert any("worker" in note for note in error.__notes__)


def test_capture_error_stops_the_pipeline():  # F04 (and F02's "visible error")
    out, logs = io.StringIO(), []
    error = run_briefly(make_pipeline(FakeCapture(set(), fail="system"), FakeRecognizer(), logs, out))
    assert isinstance(error, CaptureError)
    assert any("segmenter-system" in note for note in error.__notes__)


def test_the_program_exits_non_zero_on_a_thread_error():  # F04, end to end
    import subprocess
    import sys
    from pathlib import Path

    code = (
        "import io, time; from tests.test_pipeline import *;"
        "p = make_pipeline(FakeCapture({'mic'}), FakeRecognizer(error=RuntimeError('asr broke')), [], io.StringIO());"
        "p.run()"
    )
    finished = subprocess.run([sys.executable, "-c", code], capture_output=True, text=True, timeout=30,
        cwd=Path(__file__).resolve().parents[1],
    )
    assert finished.returncode != 0
    assert "Traceback" in finished.stderr and "asr broke" in finished.stderr
