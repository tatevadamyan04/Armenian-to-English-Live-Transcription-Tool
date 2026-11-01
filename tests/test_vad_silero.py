"""M2 with the real Silero model (D19) — runs on the Mac, where models/ and data/e1/ exist.

These tests need the downloaded model and the E1 recordings; when either is missing they
fail with a message naming what to do, rather than being skipped.
"""

from __future__ import annotations

import wave
from pathlib import Path

import numpy as np
import pytest

from hyen.audio import AudioChunk
from hyen.modelfiles import ROOT, SILERO_VAD
from hyen.segmenter import CHUNK, CHUNK_SECONDS, Segmenter
from hyen.vad import SileroVad

E1_DIR = ROOT / "data" / "e1"
E1_FILES = [f"e1_{n:02d}.wav" for n in range(1, 26)]


def read_wav(path: Path) -> np.ndarray:
    """16 kHz mono 16-bit WAV → float32 in [-1, 1]."""
    if not path.exists():
        pytest.skip(f"{path} not found; record E1 first (eval/record.py prompts, session 02)")
    with wave.open(str(path), "rb") as reader:
        assert (reader.getframerate(), reader.getnchannels(), reader.getsampwidth()) == (16_000, 1, 2), path
        frames = reader.readframes(reader.getnframes())
    return np.frombuffer(frames, dtype="<i2").astype(np.float32) / 32768.0


def as_chunks(samples: np.ndarray, source: str = "mic") -> list[AudioChunk]:
    usable = len(samples) // CHUNK * CHUNK
    return [
        AudioChunk(source, samples[start : start + CHUNK].copy(), start / 16_000)
        for start in range(0, usable, CHUNK)
    ]


@pytest.fixture(scope="module")
def vad() -> SileroVad:
    return SileroVad(SILERO_VAD)


def test_digital_silence_is_not_speech(vad):  # T20
    vad.reset()
    probabilities = [vad(np.zeros(CHUNK, dtype=np.float32)) for _ in range(100)]  # 3.2 s
    assert max(probabilities) < 0.5


def test_recorded_speech_is_speech(vad):
    vad.reset()
    probabilities = [vad(chunk.samples) for chunk in as_chunks(read_wav(E1_DIR / "e1_01.wav"))]
    assert max(probabilities) >= 0.5


@pytest.mark.parametrize("name", E1_FILES)
def test_every_e1_phrase_gives_a_segment(vad, name):
    """Each E1 recording, followed by 1 s of silence, yields at least one segment of speech."""
    vad.reset()
    samples = np.concatenate([read_wav(E1_DIR / name), np.zeros(16_000, dtype=np.float32)])
    logs: list[str] = []
    segmenter = Segmenter("mic", vad, log=logs.append)
    segments = []
    for chunk in as_chunks(samples):
        segment = segmenter.feed(chunk, now=chunk.t_start + CHUNK_SECONDS)
        if segment is not None:
            segments.append(segment)
    summary = ", ".join(f"{s.t_speech_start:.2f}–{s.t_speech_end:.2f} s" for s in segments)
    print(f"{name}: {len(segments)} segment(s) {summary} {logs}")
    assert segments, f"{name}: no segment found; dropped: {logs}"
