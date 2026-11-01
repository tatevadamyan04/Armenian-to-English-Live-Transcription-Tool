"""M4 · speaker labels (PLAN.md §5, R6–R8, D7 as revised in S05).

"Me" for every microphone segment (R6); never embedded.

Call-side segments: a speaker embedding (a fixed-length "voice fingerprint") is computed
for each segment and compared by cosine similarity with every known speaker's centroid,
the normalised mean of that speaker's embeddings so far.

- best similarity >= threshold          → that speaker's label
- otherwise, segment >= MIN_NEW_SECONDS → a new "Speaker N" (R7)
- otherwise (short, no match)           → the nearest existing speaker's label
- no speaker exists yet                 → "Speaker 1"; its centroid is created only by a
                                          segment >= MIN_NEW_SECONDS
- shorter than MIN_EMBED_SECONDS        → not embedded; gets the previous call-side label
                                          (or "Speaker 1"), with a note on stderr

Only segments >= MIN_NEW_SECONDS create or update a centroid, so short noisy pieces
don't found a speaker or pull one away. Labels are never renamed or merged, so a label lasts the whole session (R8).

THRESHOLD and MIN_NEW_SECONDS were chosen in S05 with eval/cross_speakers.py on E3 + E4
(DECISIONS D7). Known weakness: a person gets their own label only after speaking
MIN_NEW_SECONDS at a stretch; until then their segments carry the nearest label.
"""

from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass
from pathlib import Path
from typing import Protocol

import numpy as np

SAMPLE_RATE = 16_000
THRESHOLD = 0.77  # D7 (S05): two speakers kept apart at 0.74–0.80 on E3 + E4
MIN_NEW_SECONDS = 7.0  # D7 (S05): shortest segment that may create a speaker or update a centroid
MIN_EMBED_SECONDS = 0.5  # shorter segments are not embedded (too little voice to compare)


class Embedder(Protocol):
    def embed(self, samples: np.ndarray) -> np.ndarray: ...  # 1-D float vector


class SegmentLike(Protocol):
    seg_id: str
    source: str
    samples: np.ndarray


@dataclass(frozen=True)
class Assignment:
    label: str
    similarity: float | None  # cosine similarity to the chosen centroid; None if not compared
    reason: str  # "mic", "match", "new", "nearest", "first", "short"


def _unit(vector: np.ndarray) -> np.ndarray:
    vector = np.asarray(vector, dtype=np.float64).ravel()
    norm = np.linalg.norm(vector)
    if not np.isfinite(norm) or norm == 0.0:
        raise ValueError(f"speaker embedding has norm {norm}; cannot compare it (model or input fault)")
    return vector / norm


class SpeakerLabeler:
    def __init__(
        self,
        embedder: Embedder,
        threshold: float = THRESHOLD,
        min_new_seconds: float = MIN_NEW_SECONDS,
        min_embed_seconds: float = MIN_EMBED_SECONDS,
        log: Callable[[str], None] | None = None,
    ) -> None:
        if not -1.0 <= threshold <= 1.0:
            raise ValueError(f"threshold must be a cosine similarity in [-1, 1], got {threshold}")
        self.embedder = embedder
        self.threshold = threshold
        self.min_new_seconds = min_new_seconds
        self.min_embed_seconds = min_embed_seconds
        self.log = log if log is not None else (lambda message: None)
        self._sums: list[np.ndarray] = []  # per speaker: sum of unit embeddings
        self._last: str | None = None  # last call-side label

    @property
    def count(self) -> int:
        return len(self._sums)

    def label(self, segment: SegmentLike) -> str:
        return self.assign(segment).label

    def assign(self, segment: SegmentLike) -> Assignment:
        if segment.source == "mic":
            return Assignment("Me", None, "mic")
        if segment.source != "system":
            raise ValueError(f"unknown source {segment.source!r} in segment {segment.seg_id}")
        seconds = len(segment.samples) / SAMPLE_RATE
        if seconds < self.min_embed_seconds:
            if self._last is None:
                # Nobody has spoken yet: call it Speaker 1 without a centroid. The first
                # embedded segment then creates Speaker 1 with one (see "first" below).
                result = Assignment(self._name(0), None, "first")
            else:
                result = Assignment(self._last, None, "short")
            self.log(
                f"{segment.seg_id}: {seconds:.2f} s is too short to embed; labelled {result.label} ({result.reason})"
            )
            self._last = result.label
            return result

        vector = _unit(self.embedder.embed(segment.samples))
        long_enough = seconds >= self.min_new_seconds
        if not self._sums:
            if long_enough:
                result = Assignment(self._create(vector), None, "first")
            else:
                # Too short to found a speaker: "Speaker 1" for now, no centroid. The first
                # segment >= min_new_seconds creates Speaker 1 (S05: system-1 on E3, 0.74 s,
                # had seeded the centroid against the agreed rule).
                result = Assignment(self._name(0), None, "first")
        else:
            similarities = [float(vector @ _unit(total)) for total in self._sums]
            best = int(np.argmax(similarities))
            if similarities[best] >= self.threshold:
                if long_enough:
                    self._sums[best] += vector
                result = Assignment(self._name(best), similarities[best], "match")
            elif long_enough:
                result = Assignment(self._create(vector), similarities[best], "new")
            else:
                result = Assignment(self._name(best), similarities[best], "nearest")
        self._last = result.label
        return result

    def _create(self, vector: np.ndarray) -> str:
        self._sums.append(vector.copy())
        return self._name(len(self._sums) - 1)

    @staticmethod
    def _name(index: int) -> str:
        return f"Speaker {index + 1}"


class SherpaEmbedder:
    """Speaker embeddings from an ONNX model run by sherpa-onnx (D7 revision, S05).

    sherpa-onnx computes the model's filterbank features itself. API used, from
    sherpa-onnx's speaker-identification example (to verify on the Mac by the smoke test):
    SpeakerEmbeddingExtractorConfig, SpeakerEmbeddingExtractor, create_stream,
    accept_waveform, input_finished, is_ready, compute.
    """

    def __init__(self, model: Path, num_threads: int = 2) -> None:
        import sherpa_onnx

        if not model.is_file():
            raise FileNotFoundError(
                f"speaker model not found at {model}; run: python scripts/download_models.py"
            )
        config = sherpa_onnx.SpeakerEmbeddingExtractorConfig(
            model=str(model), num_threads=num_threads, debug=False, provider="cpu"
        )
        if not config.validate():
            raise RuntimeError(f"sherpa-onnx rejected the speaker model config for {model}")
        self._extractor = sherpa_onnx.SpeakerEmbeddingExtractor(config)
        self.dim = self._extractor.dim

    def embed(self, samples: np.ndarray) -> np.ndarray:
        stream = self._extractor.create_stream()
        stream.accept_waveform(sample_rate=SAMPLE_RATE, waveform=np.ascontiguousarray(samples, dtype=np.float32))
        stream.input_finished()
        if not self._extractor.is_ready(stream):
            raise ValueError(f"segment of {len(samples) / SAMPLE_RATE:.2f} s is too short for the speaker model")
        return np.asarray(self._extractor.compute(stream), dtype=np.float32)
