"""eval/score_speakers.py: the speaker metrics of TESTS.md §3.3 and the T10/T11 checks."""

from __future__ import annotations

import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "eval"))

from score_speakers import RefRow, Scored, match, metrics, parse_time, read_reference  # noqa: E402


def S(speaker, label, start=0.0):
    return Scored("system-x", start, start + 2.0, speaker, label)


def test_parse_time():
    assert parse_time("73.5") == 73.5
    assert parse_time("1:13.5") == 73.5
    assert parse_time("0:01:13") == 73.0
    with pytest.raises(ValueError):
        parse_time("1m13")


def test_match_by_largest_overlap():
    rows = [RefRow(0, 5, "A"), RefRow(5, 9, "B")]
    assert match(3.5, 7.5, rows) == "B"  # 1.5 s with A, 2.5 s with B
    assert match(0.5, 2.0, rows) == "A"


def test_match_needs_half_the_segment():
    rows = [RefRow(0, 1, "A")]
    assert match(0.5, 3.0, rows) is None  # 0.5 s of 2.5 s


def test_perfect_labelling():
    m = metrics([S("A", "Speaker 1"), S("A", "Speaker 1"), S("B", "Speaker 2")])
    assert m["consistency"] == {"A": 1.0, "B": 1.0}
    assert m["purity"] == {"Speaker 1": 1.0, "Speaker 2": 1.0}
    assert (m["labels"], m["speakers"], m["t10"]) == (2, 2, True)


def test_consistency_and_purity_definitions():
    items = [S("A", "Speaker 1")] * 3 + [S("A", "Speaker 2")] + [S("B", "Speaker 2")] * 2
    m = metrics(items)
    assert m["consistency"] == {"A": 0.75, "B": 1.0}
    assert m["purity"]["Speaker 1"] == 1.0
    assert m["purity"]["Speaker 2"] == pytest.approx(2 / 3)
    assert m["consistency_w"] == pytest.approx((0.75 * 4 + 1.0 * 2) / 6)


def test_t10_fails_when_two_speakers_share_their_top_label():
    m = metrics([S("A", "Speaker 1"), S("B", "Speaker 1")])
    assert m["t10"] is False


def test_mixed_and_unmatched_are_not_scored_but_their_labels_count():
    m = metrics([S("A", "Speaker 1"), S("A+B", "Speaker 3"), S(None, "Speaker 2")])
    assert m["segments"] == 1 and m["speakers"] == 1
    assert m["labels"] == 3  # labels created counts every label the program printed


def test_t11_same_label_three_minutes_later():
    ok = metrics([S("A", "Speaker 1", 10), S("A", "Speaker 1", 200), S("A", "Speaker 1", 250)])
    assert ok["t11"] == {"A": True}
    drift = metrics([S("A", "Speaker 1", 10), S("A", "Speaker 1", 20), S("A", "Speaker 1", 30),
                     S("A", "Speaker 3", 200), S("A", "Speaker 3", 250)])
    assert drift["t11"] == {"A": False}
    short = metrics([S("A", "Speaker 1", 10), S("A", "Speaker 1", 100)])
    assert short["t11"] == {}  # never heard 3 min later: T11 can't be judged


def test_reference_without_speaker_column_is_an_error(tmp_path):
    ref = tmp_path / "ref.tsv"
    ref.write_text("seg_id\tstart\tend\thy\nsystem-1\t0.5\t2.0\tx\n", encoding="utf-8")
    with pytest.raises(SystemExit, match="speaker"):
        read_reference(ref)


def test_reference_rows_without_letter_are_skipped(tmp_path):
    ref = tmp_path / "ref.tsv"
    ref.write_text("start\tend\tspeaker\n0.5\t2.0\ta\n2.0\t3.0\t\n3.0\t4.0\tA+B\n", encoding="utf-8")
    assert read_reference(ref) == [RefRow(0.5, 2.0, "A"), RefRow(3.0, 4.0, "A+B")]


def test_score_run_replays_m4_over_cached_embeddings():
    import numpy as np

    from hyen.segmenter import Segment
    from score_speakers import score_run

    def seg(n, start, who):
        return Segment(f"system-{n}", "system", np.zeros(48_000, dtype=np.float32), start, start + 3.0, start + 3.6)

    a, b = np.array([1.0, 0.0]), np.array([0.0, 1.0])
    embedded = [(seg(1, 0, "A"), "A", a), (seg(2, 5, "B"), "B", b), (seg(3, 10, "A"), "A", a)]
    labels = [s.label for s in score_run(embedded, 0.5, min_new=1.5)]
    assert labels == ["Speaker 1", "Speaker 2", "Speaker 1"]
    assert [s.label for s in score_run(embedded, 0.5, min_new=1.5)] == labels  # each run starts fresh


def test_unknown_speaker_is_not_scored_as_a_person():
    from score_speakers import kind

    assert (kind("A"), kind("A+B"), kind("?"), kind(None)) == ("single", "mixed", "unknown", "unmatched")
    m = metrics([S("A", "Speaker 1"), S("B", "Speaker 2"), S("?", "Speaker 1")])
    assert m["speakers"] == 2 and m["t10"] is True


def test_separation_on_clean_and_on_inseparable_embeddings():
    import numpy as np

    from hyen.segmenter import Segment
    from score_speakers import separation

    rng = np.random.default_rng(1)

    def seg(n):
        return Segment(f"system-{n}", "system", np.zeros(32_000, dtype=np.float32), n * 3.0, n * 3.0 + 2.0, 0.0)

    a, b = np.eye(4)[0], np.eye(4)[1]
    clean = [(seg(i), sp, base + rng.normal(0, 0.05, 4)) for i, (sp, base) in enumerate([("A", a), ("B", b)] * 5)]
    sep = separation(clean, 1.5)
    assert sep["nn"] == 1.0 and sep["centroid"] == 1.0
    assert sep["pairs"]["A–B"][2] < sep["pairs"]["A–A"][0]  # max across < min within
    same = [(seg(i), sp, a + rng.normal(0, 0.05, 4)) for i, sp in enumerate(["A", "B"] * 10)]
    assert separation(same, 1.5)["centroid"] < 0.8  # no structure → near chance


def test_separation_skips_short_mixed_and_unknown():
    import numpy as np

    from hyen.segmenter import Segment
    from score_speakers import separation

    def seg(n, length):
        return Segment(f"system-{n}", "system", np.zeros(1, dtype=np.float32), 0.0, length, 0.0)

    v = np.array([1.0, 0.0])
    embedded = [(seg(1, 3.0), "A", v), (seg(2, 1.0), "A", v), (seg(3, 3.0), "A+B", v), (seg(4, 3.0), "?", v),
                (seg(5, 3.0), "B", None)]
    assert separation(embedded, 1.5)["n"] == 1


def test_center_removes_the_shared_component():
    import numpy as np

    from hyen.segmenter import Segment
    from score_speakers import center, separation

    def seg(n):
        return Segment(f"system-{n}", "system", np.zeros(1, dtype=np.float32), n * 3.0, n * 3.0 + 2.0, 0.0)

    shared = np.array([5.0, 0.0, 0.0])  # a big common part makes every cosine high
    a, b = shared + [0, 1, 0], shared + [0, 0, 1]
    embedded = [(seg(i), sp, v) for i, (sp, v) in enumerate([("A", a), ("B", b)] * 4)] + [(seg(9), "A", None)]
    assert separation(embedded, 1.5)["pairs"]["A–B"][1] > 0.9
    centred = center(embedded)
    assert centred[-1][2] is None
    assert separation(centred, 1.5)["pairs"]["A–B"][1] < 0


def test_filled_template_reads_back(tmp_path):  # the --template layout, after lettering
    ref = tmp_path / "e4_ref.tsv"
    ref.write_text("start\tend\tspeaker\tclock\n3.10\t9.50\tA\t0:03\n10.02\t12.00\t\t0:10\n12.4\t20.0\tb\t0:12\n",
                   encoding="utf-8")
    assert read_reference(ref) == [RefRow(3.10, 9.50, "A"), RefRow(12.4, 20.0, "B")]


def test_score_run_min_new_controls_who_may_create_a_speaker():
    import numpy as np

    from hyen.segmenter import Segment
    from score_speakers import score_run

    def seg(n, length):
        return Segment(f"system-{n}", "system", np.zeros(int(length * 16_000), dtype=np.float32), n * 10.0,
                       n * 10.0 + length, 0.0)

    a, b = np.array([1.0, 0.0]), np.array([0.0, 1.0])
    embedded = [(seg(1, 4.0), "A", a), (seg(2, 2.0), "B", b)]
    assert [s.label for s in score_run(embedded, 0.5, min_new=1.5)] == ["Speaker 1", "Speaker 2"]
    assert [s.label for s in score_run(embedded, 0.5, min_new=3.0)] == ["Speaker 1", "Speaker 1"]  # 2 s too short
