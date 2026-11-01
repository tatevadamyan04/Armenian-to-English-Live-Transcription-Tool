"""M2 · phrase segmenter (PLAN.md §5, D4).

Turns the endless chunk stream of one source into Segments, one per spoken phrase:

- A phrase starts at the first chunk whose speech probability reaches START (0.5). The
  few chunks before it are kept too (PRE_ROLL), because the VAD reacts a little late.
- Inside a phrase, a chunk counts as speech while its probability stays at or above
  CONTINUE (0.35). Using a lower threshold to continue than to start keeps short dips
  inside a word from ending the phrase.
- The phrase ends after END_SILENCE chunks in a row below CONTINUE (19 × 32 ms = 608 ms).
  The segment then ends TAIL chunks after the last speech chunk.
- A phrase that reaches MAX_CHUNKS (250 × 32 ms = 8 s; 12 s until D23) is cut, so long
  monologues still print (T16). The cut falls after the chunk with the lowest speech
  probability among the last CUT_WINDOW chunks (62 × 32 ms ≈ 2 s; the latest one on a tie),
  so it lands in a pause between words rather than inside one (D25). The chunks after the
  cut start the next phrase, so nothing is lost and there is no gap.
- A segment with less than MIN_SPEECH chunks of speech (8 × 32 ms = 256 ms) is dropped
  and noted on stderr: clicks and short noises shouldn't reach ASR (T20, T21).

The segmenter is pure logic: the VAD and the current time are passed in, so it is
unit-tested with a fake VAD.
"""

from __future__ import annotations

import sys
from collections import deque
from collections.abc import Callable
from dataclasses import dataclass, field
from typing import TYPE_CHECKING

import numpy as np

if TYPE_CHECKING:
    from hyen.audio import AudioChunk

SAMPLE_RATE = 16_000
CHUNK = 512  # samples per AudioChunk (D17)
CHUNK_SECONDS = CHUNK / SAMPLE_RATE  # 0.032 s


@dataclass(frozen=True)
class SegmenterConfig:
    start: float = 0.5  # probability that starts a phrase
    continue_: float = 0.35  # probability that keeps a phrase going
    end_silence: int = 19  # chunks below `continue_` that end a phrase (608 ms)
    pre_roll: int = 3  # chunks kept from before the start (96 ms)
    tail: int = 3  # chunks kept after the last speech chunk (96 ms)
    max_chunks: int = 250  # a phrase is cut at this length (8.0 s; D23 keeps MT time bounded)
    cut_window: int = 62  # at the cap, cut at the quietest of the last this-many chunks (1.98 s; D25)
    min_speech: int = 8  # fewer speech chunks than this → dropped (256 ms)

    def __post_init__(self) -> None:
        if not 1 <= self.cut_window <= self.max_chunks:
            raise ValueError(f"cut_window must be 1..max_chunks ({self.max_chunks}), got {self.cut_window}")


@dataclass(frozen=True)
class Segment:
    """One phrase from one source, as the inference worker receives it."""

    seg_id: str  # "<source>-<n>", unique within the session
    source: str  # "mic" or "system"
    samples: np.ndarray  # float32, mono, 16 kHz
    t_speech_start: float  # session time of the first speech chunk (printed as the line's time)
    t_speech_end: float  # session time where speech ended (or where a long phrase was cut)
    t_end_detected: float  # session time at which the segmenter emitted the segment
    cut: bool = False  # True when the phrase hit the length cap rather than ending in silence


@dataclass
class _Phrase:
    chunks: list[np.ndarray] = field(default_factory=list)
    probabilities: list[float] = field(default_factory=list)  # one per chunk; pre-roll chunks too
    t_audio_start: float = 0.0  # session time of the first kept sample (pre-roll included)
    t_speech_start: float = 0.0
    speech_chunks: int = 0
    last_speech_index: int = -1  # index in `chunks` of the latest speech chunk
    silence_run: int = 0


class Segmenter:
    """Phrase detection for one source. Feed it every chunk of that source, in order."""

    def __init__(
        self,
        source: str,
        vad: Callable[[np.ndarray], float],
        config: SegmenterConfig = SegmenterConfig(),
        log: Callable[[str], None] | None = None,
    ) -> None:
        self.source = source
        self.vad = vad
        self.config = config
        self.log = log if log is not None else (lambda message: print(message, file=sys.stderr))
        self._recent: deque[tuple[np.ndarray, float, float]] = deque(maxlen=config.pre_roll)
        self._phrase: _Phrase | None = None
        self._next_t: float | None = None  # expected t_start of the next chunk
        self._count = 0
        self.dropped = 0  # segments dropped for being too short

    def feed(self, chunk: AudioChunk, now: float) -> Segment | None:
        """Process one chunk; return a Segment if a phrase just ended. `now` is the session time."""
        self._check(chunk)
        probability = float(self.vad(chunk.samples))
        phrase = self._phrase
        if phrase is None:
            if probability >= self.config.start:
                self._start(chunk, probability)
            else:
                self._recent.append((chunk.samples, chunk.t_start, probability))
            return None
        self._append(phrase, chunk.samples, probability, chunk.t_start)
        if phrase.silence_run >= self.config.end_silence:
            return self._finish(now)
        if len(phrase.chunks) >= self.config.max_chunks:
            return self._cut(now)
        return None

    def _check(self, chunk: AudioChunk) -> None:
        if chunk.source != self.source:
            raise ValueError(f"segmenter for {self.source!r} was given a chunk from {chunk.source!r}")
        if chunk.samples.shape != (CHUNK,):
            raise ValueError(f"{self.source}: expected {CHUNK} samples per chunk, got {chunk.samples.shape}")
        if self._next_t is not None and abs(chunk.t_start - self._next_t) > 1e-6:
            raise ValueError(
                f"{self.source}: chunk starts at {chunk.t_start:.6f} s, expected {self._next_t:.6f} s; "
                "chunks must arrive in order with no gaps (D15)"
            )
        self._next_t = chunk.t_start + CHUNK_SECONDS

    def _append(self, phrase: _Phrase, samples: np.ndarray, probability: float, t_start: float) -> None:
        phrase.chunks.append(samples)
        phrase.probabilities.append(probability)
        if probability >= self.config.continue_:
            if phrase.speech_chunks == 0:  # the first speech after a cut
                phrase.t_speech_start = t_start
            phrase.speech_chunks += 1
            phrase.last_speech_index = len(phrase.chunks) - 1
            phrase.silence_run = 0
        else:
            phrase.silence_run += 1

    def _start(self, chunk: AudioChunk, probability: float) -> None:
        recent = list(self._recent)
        self._recent.clear()
        t_audio_start = recent[0][1] if recent else chunk.t_start
        self._phrase = _Phrase(
            chunks=[samples for samples, _, _ in recent] + [chunk.samples],
            probabilities=[p for _, _, p in recent] + [probability],
            t_audio_start=t_audio_start,
            t_speech_start=chunk.t_start,
            speech_chunks=1,
            last_speech_index=len(recent),
        )

    def _finish(self, now: float) -> Segment | None:
        """The phrase ended in silence."""
        phrase = self._phrase
        keep = min(len(phrase.chunks), phrase.last_speech_index + 1 + self.config.tail)
        t_speech_end = phrase.t_audio_start + (phrase.last_speech_index + 1) * CHUNK_SECONDS
        self._phrase = None
        # The silence at the end of this phrase is the pre-roll of the next one.
        first = len(phrase.chunks) - self.config.pre_roll
        for index in range(max(first, 0), len(phrase.chunks)):
            self._recent.append(
                (phrase.chunks[index], phrase.t_audio_start + index * CHUNK_SECONDS, phrase.probabilities[index])
            )
        return self._emit(phrase, keep, phrase.speech_chunks, t_speech_end, now, cut=False)

    def _cut(self, now: float) -> Segment | None:
        """The phrase reached the cap: cut after the quietest chunk of the last cut_window (D25)."""
        phrase = self._phrase
        length = len(phrase.chunks)
        window = phrase.probabilities[length - self.config.cut_window :]
        lowest = min(window)
        # the latest chunk with the lowest probability, so the segment stays as long as possible
        offset = max(i for i, p in enumerate(window) if p == lowest)
        keep = length - self.config.cut_window + offset + 1
        t_speech_end = phrase.t_audio_start + keep * CHUNK_SECONDS
        speech = sum(1 for p in phrase.probabilities[:keep] if p >= self.config.continue_)
        # The chunks after the cut start the next phrase; its start time is set by its first speech chunk.
        rest = _Phrase(t_audio_start=t_speech_end)
        for index in range(keep, length):
            self._append(rest, phrase.chunks[index], phrase.probabilities[index], t_speech_end + (index - keep) * CHUNK_SECONDS)
        self._phrase = rest
        return self._emit(phrase, keep, speech, t_speech_end, now, cut=True)

    def _emit(
        self, phrase: _Phrase, keep: int, speech_chunks: int, t_speech_end: float, now: float, cut: bool
    ) -> Segment | None:
        if speech_chunks < self.config.min_speech:
            self.dropped += 1
            if speech_chunks > 0:
                self.log(
                    f"{self.source}: dropped {speech_chunks * CHUNK_SECONDS * 1000:.0f} ms of speech "
                    f"at {phrase.t_speech_start:.2f} s (shorter than {self.config.min_speech * CHUNK_SECONDS * 1000:.0f} ms)"
                )
            return None
        self._count += 1
        return Segment(
            seg_id=f"{self.source}-{self._count}",
            source=self.source,
            samples=np.concatenate(phrase.chunks[:keep]),
            t_speech_start=phrase.t_speech_start,
            t_speech_end=t_speech_end,
            t_end_detected=now,
            cut=cut,
        )
