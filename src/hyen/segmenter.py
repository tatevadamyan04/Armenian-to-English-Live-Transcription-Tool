"""M2 · phrase segmenter (PLAN.md §5, D4).

Turns the endless chunk stream of one source into Segments, one per spoken phrase:

- A phrase starts at the first chunk whose speech probability reaches START (0.5). The
  few chunks before it are kept too (PRE_ROLL), because the VAD reacts a little late.
- Inside a phrase, a chunk counts as speech while its probability stays at or above
  CONTINUE (0.35). Using a lower threshold to continue than to start keeps short dips
  inside a word from ending the phrase.
- The phrase ends after END_SILENCE chunks in a row below CONTINUE (19 × 32 ms = 608 ms).
  The segment then ends TAIL chunks after the last speech chunk.
- A phrase that reaches MAX_CHUNKS (250 × 32 ms = 8 s; 12 s until D23) is cut there, so long monologues
  still print (T16); the speech that follows becomes the next segment without a gap.
- A segment with less than MIN_SPEECH chunks of speech (8 × 32 ms = 256 ms) is dropped
  and noted on stderr: clicks and short noises shouldn't reach ASR (T20, T21).

All values are starting points to tune in S03/S04. The segmenter is pure logic: the
VAD and the current time are passed in, so it is unit-tested with a fake VAD.
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
    min_speech: int = 8  # fewer speech chunks than this → dropped (256 ms)


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
        self._recent: deque[tuple[np.ndarray, float]] = deque(maxlen=config.pre_roll)
        self._phrase: _Phrase | None = None
        self._next_t: float | None = None  # expected t_start of the next chunk
        self._count = 0
        self.dropped = 0  # segments dropped for being too short

    def feed(self, chunk: AudioChunk, now: float) -> Segment | None:
        """Process one chunk; return a Segment if a phrase just ended. `now` is the session time."""
        self._check(chunk)
        probability = self.vad(chunk.samples)
        phrase = self._phrase
        if phrase is None:
            if probability >= self.config.start:
                self._start(chunk)
            else:
                self._recent.append((chunk.samples, chunk.t_start))
            return None
        phrase.chunks.append(chunk.samples)
        if probability >= self.config.continue_:
            if phrase.speech_chunks == 0:  # the first speech after a cut
                phrase.t_speech_start = chunk.t_start
            phrase.speech_chunks += 1
            phrase.last_speech_index = len(phrase.chunks) - 1
            phrase.silence_run = 0
        else:
            phrase.silence_run += 1
        if phrase.silence_run >= self.config.end_silence:
            return self._finish(now, cut=False)
        if len(phrase.chunks) >= self.config.max_chunks:
            return self._finish(now, cut=True)
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

    def _start(self, chunk: AudioChunk) -> None:
        recent = list(self._recent)
        self._recent.clear()
        t_audio_start = recent[0][1] if recent else chunk.t_start
        self._phrase = _Phrase(
            chunks=[samples for samples, _ in recent] + [chunk.samples],
            t_audio_start=t_audio_start,
            t_speech_start=chunk.t_start,
            speech_chunks=1,
            last_speech_index=len(recent),
        )

    def _finish(self, now: float, cut: bool) -> Segment | None:
        phrase = self._phrase
        if cut:
            keep = len(phrase.chunks)
            t_speech_end = phrase.t_audio_start + keep * CHUNK_SECONDS
            # The cut falls inside continuing audio: the next segment starts right after it,
            # with no speech counted yet; its start time is set by its first speech chunk.
            self._phrase = _Phrase(t_audio_start=t_speech_end, speech_chunks=0, last_speech_index=-1)
        else:
            keep = min(len(phrase.chunks), phrase.last_speech_index + 1 + self.config.tail)
            t_speech_end = phrase.t_audio_start + (phrase.last_speech_index + 1) * CHUNK_SECONDS
            self._phrase = None
            # The silence at the end of this phrase is the pre-roll of the next one.
            first = len(phrase.chunks) - self.config.pre_roll
            for index in range(max(first, 0), len(phrase.chunks)):
                self._recent.append((phrase.chunks[index], phrase.t_audio_start + index * CHUNK_SECONDS))
        if phrase.speech_chunks < self.config.min_speech:
            self.dropped += 1
            if phrase.speech_chunks > 0:
                self.log(
                    f"{self.source}: dropped {phrase.speech_chunks * CHUNK_SECONDS * 1000:.0f} ms of speech "
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
