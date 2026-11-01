"""M2 segmenter, pure logic with a fake VAD (PLAN.md §8).

The cases come from the requirements: nothing printed on silence (T20), one segment per
phrase with the right times (T12, T14), long speech split at the cap with nothing lost
(T16), short noises dropped (T21), and loud failure on broken input (CLAUDE.md).
"""

from __future__ import annotations

import numpy as np
import pytest

from hyen.audio import AudioChunk
from hyen.segmenter import CHUNK, CHUNK_SECONDS, Segmenter, SegmenterConfig

SPEECH, SILENCE = 0.2, 0.0  # sample values; the fake VAD only looks at the level


def fake_vad(samples: np.ndarray) -> float:
    """Speech probability from the level: 0.9 for a loud chunk, 0.05 for a quiet one."""
    return 0.9 if np.max(np.abs(samples)) >= 0.1 else 0.05


def chunks(pattern: str, source: str = "mic", t0: float = 1.0) -> list[AudioChunk]:
    """One chunk per character: 's' speech, '.' silence."""
    out = []
    for index, mark in enumerate(pattern):
        value = SPEECH if mark == "s" else SILENCE
        out.append(AudioChunk(source, np.full(CHUNK, value, dtype=np.float32), t0 + index * CHUNK_SECONDS))
    return out


def run(pattern: str, config: SegmenterConfig = SegmenterConfig(), logs: list[str] | None = None):
    segmenter = Segmenter("mic", fake_vad, config, log=(logs.append if logs is not None else lambda _: None))
    results = []
    for chunk in chunks(pattern):
        segment = segmenter.feed(chunk, now=chunk.t_start + CHUNK_SECONDS)
        if segment is not None:
            results.append(segment)
    return segmenter, results


def test_silence_gives_no_segments():  # T20
    _, segments = run("." * 2000)  # 64 s
    assert segments == []


def test_one_phrase_one_segment_with_times():
    config = SegmenterConfig()
    pattern = "." * 10 + "s" * 30 + "." * 25
    _, segments = run(pattern, config)
    assert len(segments) == 1
    segment = segments[0]
    assert segment.seg_id == "mic-1" and segment.source == "mic" and not segment.cut
    assert segment.t_speech_start == pytest.approx(1.0 + 10 * CHUNK_SECONDS)
    assert segment.t_speech_end == pytest.approx(1.0 + 40 * CHUNK_SECONDS)
    # emitted once end_silence quiet chunks have been seen (the "now" passed in)
    assert segment.t_end_detected == pytest.approx(1.0 + (40 + config.end_silence) * CHUNK_SECONDS)
    # audio = pre-roll + speech + tail
    assert len(segment.samples) == (config.pre_roll + 30 + config.tail) * CHUNK
    assert segment.samples.dtype == np.float32


def test_phrase_not_ended_before_enough_silence():
    config = SegmenterConfig()
    pause = "." * (config.end_silence - 1)
    _, segments = run("s" * 20 + pause + "s" * 20 + "." * 30, config)
    assert len(segments) == 1  # a pause shorter than end_silence stays inside the phrase


def test_two_phrases_two_segments_in_order():
    _, segments = run("s" * 20 + "." * 30 + "s" * 20 + "." * 30)
    assert [segment.seg_id for segment in segments] == ["mic-1", "mic-2"]
    assert segments[0].t_speech_end < segments[1].t_speech_start


def test_dip_between_thresholds_keeps_phrase_going():
    """A chunk at 0.4 is below `start` but above `continue_`, so it doesn't count as silence."""
    probabilities = iter([0.9] * 10 + [0.4] * 25 + [0.9] * 10 + [0.05] * 30)
    segmenter = Segmenter("mic", lambda _: next(probabilities), log=lambda _: None)
    results = [segmenter.feed(chunk, now=0.0) for chunk in chunks("." * 75)]
    assert len([segment for segment in results if segment is not None]) == 1


def test_long_speech_is_cut_at_the_cap_and_nothing_is_lost():  # T16
    config = SegmenterConfig()
    speech = 30 * 1000 // 32  # ~30 s with no pause
    _, segments = run("s" * speech + "." * 40, config)
    assert len(segments) == 4  # 8 s + 8 s + 8 s + ~6 s (cap 250 chunks, D23)
    assert [segment.cut for segment in segments] == [True, True, True, False]
    assert all(len(segment.samples) <= config.max_chunks * CHUNK for segment in segments)
    for before, after in zip(segments, segments[1:]):
        assert after.t_speech_start == pytest.approx(before.t_speech_end)  # no gap between pieces
    speech_samples = sum(len(segment.samples) for segment in segments) - config.tail * CHUNK
    assert speech_samples == speech * CHUNK


def run_probabilities(probabilities: list[float], config: SegmenterConfig = SegmenterConfig()):
    """Feed one chunk per probability; the VAD returns them in order."""
    values = iter(probabilities)
    segmenter = Segmenter("mic", lambda _: next(values), config, log=lambda _: None)
    results = []
    for chunk in chunks("." * len(probabilities)):
        segment = segmenter.feed(chunk, now=chunk.t_start + CHUNK_SECONDS)
        if segment is not None:
            results.append(segment)
    return results


def speech_with_dips(length: int, dips: dict[int, float]) -> list[float]:
    return [dips.get(index, 0.9) for index in range(length)] + [0.05] * 40


def test_cap_cuts_after_the_quietest_chunk_in_the_last_two_seconds():  # T16 fix, D25
    config = SegmenterConfig()
    segments = run_probabilities(speech_with_dips(300, {220: 0.2}), config)
    assert len(segments) == 2
    first, second = segments
    assert first.cut and len(first.samples) == 221 * CHUNK  # chunks 0..220: the cut is right after the dip
    assert first.t_speech_end == pytest.approx(1.0 + 221 * CHUNK_SECONDS)
    assert second.t_speech_start == pytest.approx(first.t_speech_end)  # no gap
    assert len(first.samples) + len(second.samples) == (300 + config.tail) * CHUNK  # nothing lost


def test_a_dip_before_the_window_is_ignored():
    segments = run_probabilities(speech_with_dips(300, {100: 0.05}))
    assert len(segments[0].samples) == 250 * CHUNK  # no lower chunk in the window: cut at the cap


def test_the_later_of_two_equally_quiet_chunks_is_chosen():
    segments = run_probabilities(speech_with_dips(300, {200: 0.1, 230: 0.1}))
    assert len(segments[0].samples) == 231 * CHUNK


def test_a_lower_but_earlier_dip_wins_over_a_later_one():
    segments = run_probabilities(speech_with_dips(300, {195: 0.05, 240: 0.3}))
    assert len(segments[0].samples) == 196 * CHUNK


def test_cut_window_must_fit_in_the_cap():
    with pytest.raises(ValueError, match="cut_window"):
        SegmenterConfig(max_chunks=100, cut_window=101)


def test_short_noise_is_dropped_and_logged():  # T21
    logs: list[str] = []
    segmenter, segments = run("." * 10 + "ss" + "." * 40, logs=logs)
    assert segments == []
    assert segmenter.dropped == 1
    assert len(logs) == 1 and "dropped" in logs[0]


def test_pre_roll_comes_from_the_silence_after_an_earlier_phrase():
    config = SegmenterConfig()
    _, segments = run("s" * 20 + "." * config.end_silence + "s" * 20 + "." * 30, config)
    assert len(segments) == 2
    assert len(segments[1].samples) == (config.pre_roll + 20 + config.tail) * CHUNK


def test_chunk_from_the_wrong_source_is_an_error():
    segmenter = Segmenter("mic", fake_vad, log=lambda _: None)
    with pytest.raises(ValueError, match="system"):
        segmenter.feed(chunks("s", source="system")[0], now=0.0)


def test_gap_between_chunks_is_an_error():
    segmenter = Segmenter("mic", fake_vad, log=lambda _: None)
    first, _, third = chunks("...")
    segmenter.feed(first, now=0.0)
    with pytest.raises(ValueError, match="no gaps"):
        segmenter.feed(third, now=0.0)


def test_wrong_chunk_size_is_an_error():
    segmenter = Segmenter("mic", fake_vad, log=lambda _: None)
    with pytest.raises(ValueError, match="512"):
        segmenter.feed(AudioChunk("mic", np.zeros(100, dtype=np.float32), 0.0), now=0.0)
