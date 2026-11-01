"""M4 unit tests (R6, R7, R8; D7 rules agreed in S05). No model: a fake embedder maps
each test "voice" to a fixed direction, plus a little noise where it matters."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pytest

from hyen.speakers import MIN_NEW_SECONDS, SAMPLE_RATE, THRESHOLD, SpeakerLabeler

DIM = 8
MIN_NEW = 1.5  # the rules are tested with short segments; the tuned value is tested at the end


@dataclass
class Seg:
    seg_id: str
    source: str
    samples: np.ndarray


def voice(seconds: float, who: int) -> np.ndarray:
    """Audio whose first sample encodes the voice id (read by FakeEmbedder)."""
    samples = np.zeros(int(seconds * SAMPLE_RATE), dtype=np.float32)
    samples[0] = who
    return samples


class FakeEmbedder:
    def __init__(self, directions: dict[int, np.ndarray]) -> None:
        self.directions = directions
        self.calls = 0

    def embed(self, samples: np.ndarray) -> np.ndarray:
        self.calls += 1
        return self.directions[int(samples[0])]


def basis(i: int) -> np.ndarray:
    v = np.zeros(DIM)
    v[i] = 1.0
    return v


def make(threshold: float = 0.5, **directions: np.ndarray) -> tuple[SpeakerLabeler, FakeEmbedder]:
    dirs = {1: basis(0), 2: basis(1), 3: basis(2)}
    emb = FakeEmbedder(dirs)
    return SpeakerLabeler(emb, threshold=threshold, min_new_seconds=MIN_NEW), emb


n = 0


def sys(seconds: float, who: int) -> Seg:
    global n
    n += 1
    return Seg(f"system-{n}", "system", voice(seconds, who))


# R6 -------------------------------------------------------------------------------------


def test_mic_is_me_and_never_embedded():
    labeler, emb = make()
    for _ in range(5):
        assert labeler.label(Seg("mic-1", "mic", voice(3.0, 1))) == "Me"
    assert emb.calls == 0
    assert labeler.count == 0


def test_mic_does_not_affect_call_side_numbering():
    labeler, _ = make()
    labeler.label(Seg("mic-1", "mic", voice(3.0, 1)))
    assert labeler.label(sys(3.0, 1)) == "Speaker 1"


# R7 -------------------------------------------------------------------------------------


def test_two_voices_get_two_labels():
    labeler, _ = make()
    assert labeler.label(sys(3.0, 1)) == "Speaker 1"
    assert labeler.label(sys(3.0, 2)) == "Speaker 2"
    assert labeler.label(sys(3.0, 3)) == "Speaker 3"
    assert labeler.count == 3


def test_same_voice_keeps_label():
    labeler, _ = make()
    labels = [labeler.label(sys(2.0, 1)) for _ in range(4)]
    assert labels == ["Speaker 1"] * 4
    assert labeler.count == 1


# R8 -------------------------------------------------------------------------------------


def test_label_stable_after_long_absence():
    labeler, _ = make()
    labeler.label(sys(3.0, 1))
    for _ in range(30):  # "minutes" of the other speaker
        labeler.label(sys(3.0, 2))
    assert labeler.label(sys(3.0, 1)) == "Speaker 1"


def test_label_stable_with_noisy_embeddings():
    rng = np.random.default_rng(0)
    base = {1: basis(0), 2: basis(1)}

    class Noisy:
        def embed(self, samples):
            v = base[int(samples[0])] + rng.normal(0, 0.15, DIM)
            return v

    labeler = SpeakerLabeler(Noisy(), threshold=0.5, min_new_seconds=MIN_NEW)
    labels = [(who, labeler.label(sys(2.5, who))) for who in [1, 2] * 25]
    assert {lab for who, lab in labels if who == 1} == {"Speaker 1"}
    assert {lab for who, lab in labels if who == 2} == {"Speaker 2"}


# Threshold ------------------------------------------------------------------------------


def test_threshold_decides_match_or_new():
    a = basis(0)
    b = np.cos(np.pi / 3) * basis(0) + np.sin(np.pi / 3) * basis(1)  # cosine 0.5 to a
    for threshold, expected in [(0.4, "Speaker 1"), (0.6, "Speaker 2")]:
        labeler = SpeakerLabeler(FakeEmbedder({1: a, 2: b}), threshold=threshold, min_new_seconds=MIN_NEW)
        labeler.label(sys(3.0, 1))
        assert labeler.label(sys(3.0, 2)) == expected


def test_assignment_reports_similarity_and_reason():
    labeler, _ = make()
    first = labeler.assign(sys(3.0, 1))
    assert (first.reason, first.similarity) == ("first", None)
    again = labeler.assign(sys(3.0, 1))
    assert again.reason == "match" and again.similarity == pytest.approx(1.0)
    other = labeler.assign(sys(3.0, 2))
    assert other.reason == "new" and other.similarity == pytest.approx(0.0)


@pytest.mark.parametrize("bad", [1.5, -2.0])
def test_threshold_out_of_range_is_an_error(bad):
    with pytest.raises(ValueError, match="threshold"):
        SpeakerLabeler(FakeEmbedder({}), threshold=bad)


# Short segments -------------------------------------------------------------------------


def test_short_unmatched_segment_takes_nearest_not_new():
    labeler, _ = make()
    labeler.label(sys(3.0, 1))
    result = labeler.assign(sys(1.0, 2))  # below MIN_NEW_SECONDS, matches nobody
    assert (result.label, result.reason) == ("Speaker 1", "nearest")
    assert labeler.count == 1


def test_short_segment_does_not_update_centroid():
    a, b = basis(0), basis(1)
    mix = (a + b) / np.sqrt(2)  # cosine 0.707 to a
    labeler = SpeakerLabeler(FakeEmbedder({1: a, 2: mix, 3: b}), threshold=0.6, min_new_seconds=MIN_NEW)
    labeler.label(sys(3.0, 1))
    for _ in range(20):
        assert labeler.label(sys(1.0, 2)) == "Speaker 1"  # short matches, must not move centroid
    # Centroid still exactly a: b (cosine 0) must create Speaker 2.
    assert labeler.label(sys(3.0, 3)) == "Speaker 2"


def test_long_segment_updates_centroid():
    a, b = basis(0), basis(1)
    mix = (a + b) / np.sqrt(2)
    labeler = SpeakerLabeler(FakeEmbedder({1: a, 2: mix, 3: b}), threshold=0.6, min_new_seconds=MIN_NEW)
    labeler.label(sys(3.0, 1))
    for _ in range(20):
        labeler.label(sys(3.0, 2))  # long matches pull the centroid towards mix
    # Centroid now ≈ mix direction (cosine 0.70 to b) → b matches Speaker 1.
    assert labeler.label(sys(3.0, 3)) == "Speaker 1"


def test_very_short_segment_not_embedded_takes_previous_label():
    logs: list[str] = []
    labeler, emb = make()
    labeler.log = logs.append
    labeler.label(sys(3.0, 1))
    labeler.label(sys(3.0, 2))
    calls = emb.calls
    result = labeler.assign(sys(0.3, 1))
    assert (result.label, result.reason) == ("Speaker 2", "short")
    assert emb.calls == calls
    assert "too short" in logs[-1]


def test_very_short_first_segment_is_speaker_1_and_next_voice_too():
    labeler, _ = make()
    assert labeler.assign(sys(0.2, 2)).reason == "first"
    assert labeler.label(sys(3.0, 1)) == "Speaker 1"  # first embedded creates Speaker 1
    assert labeler.label(sys(3.0, 2)) == "Speaker 2"


def test_short_first_segment_is_speaker_1_without_a_centroid():
    labeler, _ = make()
    assert labeler.label(sys(1.0, 1)) == "Speaker 1"
    assert labeler.count == 0


def test_short_first_segment_does_not_seed_the_centroid():  # E3 system-1 (0.74 s) in S05
    labeler, _ = make()
    labeler.label(sys(1.0, 2))  # a short piece of voice 2 comes first
    assert labeler.label(sys(3.0, 1)) == "Speaker 1"  # first long segment founds Speaker 1
    assert labeler.label(sys(3.0, 2)) == "Speaker 2"  # voice 2 is not matched to a seed of itself
    assert labeler.label(sys(3.0, 1)) == "Speaker 1"


# Fail loudly ----------------------------------------------------------------------------


def test_zero_embedding_is_an_error():
    labeler = SpeakerLabeler(FakeEmbedder({1: np.zeros(DIM)}))
    with pytest.raises(ValueError, match="norm"):
        labeler.label(sys(3.0, 1))


def test_nan_embedding_is_an_error():
    labeler = SpeakerLabeler(FakeEmbedder({1: np.full(DIM, np.nan)}))
    with pytest.raises(ValueError, match="norm"):
        labeler.label(sys(3.0, 1))


def test_unknown_source_is_an_error():
    labeler, _ = make()
    with pytest.raises(ValueError, match="unknown source"):
        labeler.label(Seg("x-1", "zoom", voice(3.0, 1)))


# D7 values chosen in S05 ------------------------------------------------------------------


def test_d7_values():
    assert (THRESHOLD, MIN_NEW_SECONDS) == (0.77, 7.0)


def test_with_d7_values_a_new_person_needs_a_long_segment():
    labeler = SpeakerLabeler(FakeEmbedder({1: basis(0), 2: basis(1)}))
    assert labeler.label(sys(8.0, 1)) == "Speaker 1"
    assert labeler.assign(sys(5.0, 2)).reason == "nearest"  # 5 s: not enough to create Speaker 2
    assert labeler.label(sys(7.5, 2)) == "Speaker 2"
    assert labeler.label(sys(2.0, 2)) == "Speaker 2"  # once known, short segments match
