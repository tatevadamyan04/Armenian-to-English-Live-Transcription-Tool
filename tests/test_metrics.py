"""Unit tests for eval/metrics.py and eval/score.py (TESTS.md §3.2–3.3).

Expected values are worked out by hand from the definitions, not from the code.
"""

from __future__ import annotations

import sys
from pathlib import Path

import pytest

EVAL_DIR = Path(__file__).resolve().parents[1] / "eval"
sys.path.insert(0, str(EVAL_DIR))  # eval/ holds scripts, not an installed package (PLAN.md §6)

from metrics import cer, chrf, edit_distance, normalise, wer  # noqa: E402
import score  # noqa: E402


# --- normalisation, TESTS.md §3.2 -------------------------------------------

@pytest.mark.parametrize(
    "text, expected",
    [
        ("Սա ի՞նչ արժե։", "սա ինչ արժե"),  # step 3: inner mark deleted, not split (e1_05)
        ("Դուք խոսո՞ւմ եք անգլերեն։", "դուք խոսում եք անգլերեն"),  # e1_14
        ("Ամեն ինչ լավ է, շնորհակալ եմ։", "ամեն ինչ լավ է շնորհակալ եմ"),
        ("Կասե՞ք՝ ժամը քանիսն է։", "կասեք ժամը քանիսն է"),  # ՝ → space
        ("ԲԱՐԵՎ", "բարեվ"),  # step 2
        ("«Բարև»  —  ձեզ", "բարեւ — ձեզ"),  # step 1 և→եւ; « » removed; em dash is not in the list
        ("Վա՜յ, լա՛վ", "վայ լավ"),  # ՜ and ՛ deleted
        ("a-b...c", "a b c"),
        ("  ։։  ", ""),
    ],
)
def test_normalise(text: str, expected: str) -> None:
    assert normalise(text) == expected


def test_ligature_and_spelled_out_match() -> None:
    assert normalise("մինչև") == normalise("մինչեւ")


# --- edit distance, CER, WER --------------------------------------------------

def test_edit_distance_basic() -> None:
    assert edit_distance("kitten", "sitting") == 3
    assert edit_distance("", "abc") == 3
    assert edit_distance(["a", "b"], ["a", "b"]) == 0


def test_cer_counts_after_normalisation() -> None:
    # "ի՞նչ" vs "ինչ": identical after step 3
    assert cer(["ի՞նչ"], ["ինչ"]).edits == 0
    # ref "սա ինչ" (6 chars), hyp "սա ինց": 1 substitution
    r = cer(["Սա ի՞նչ։"], ["սա ինց"])
    assert (r.edits, r.ref_len) == (1, 6)


def test_wer_corpus_level_sums_edits() -> None:
    # item 1: 4 ref words, 1 substitution; item 2: 2 ref words, 1 deletion → 2/6
    r = wer(["ամեն ինչ լավ է", "շնորհակալ եմ"], ["ամեն ինչ լավ էր", "շնորհակալ"])
    assert (r.edits, r.ref_len) == (2, 6)
    assert r.rate == pytest.approx(1 / 3)


def test_rate_on_empty_reference_raises() -> None:
    with pytest.raises(ValueError):
        cer(["։"], ["բան"]).rate


def test_mismatched_lengths_raise() -> None:
    with pytest.raises(ValueError):
        wer(["ա"], [])


# --- chrF ------------------------------------------------------------------------

def test_chrf_identical_is_100() -> None:
    assert chrf(["Hello. How are you?"], ["Hello. How are you?"]) == pytest.approx(100.0)


def test_chrf_ignores_whitespace_only() -> None:
    assert chrf(["How are you?"], ["Howareyou?"]) == pytest.approx(100.0)


def test_chrf_disjoint_is_0() -> None:
    assert chrf(["abc"], ["xyz"]) == 0.0


def test_chrf_hand_computed() -> None:
    # ref "ab", hyp "abc", max_n 2:
    # n=1: matches 2, hyp 3, ref 2 → P 2/3, R 1
    # n=2: matches 1 ("ab"), hyp 2, ref 1 → P 1/2, R 1
    # avg P = 7/12, avg R = 1; F2 = 5 * P * R / (4P + R) = (35/12) / (40/12) = 0.875
    assert chrf(["ab"], ["abc"], max_n=2) == pytest.approx(87.5)


def test_chrf_penalises_added_content_less_than_missing() -> None:
    # beta = 2 weighs recall more: adding words costs less than dropping them
    ref = "My name is Anna."
    added = chrf([ref], ["My name is Anna. I am 20 years old."])
    dropped = chrf([ref], ["My name."])
    assert added > dropped


# --- score.py ---------------------------------------------------------------------

def _tsv(path: Path, rows: list[tuple[str, str, str]]) -> Path:
    path.write_text("id\thy\ten\n" + "".join("\t".join(r) + "\n" for r in rows), encoding="utf-8")
    return path


def test_score_reports_and_exits_0(tmp_path: Path, capsys) -> None:
    ref = _tsv(tmp_path / "ref.tsv", [("a", "Սա ի՞նչ արժե։", "How much is this?")])
    hyp = _tsv(tmp_path / "hyp.tsv", [("a", "Սա ինչ արժե", "How much is this?")])
    assert score.main([str(ref), str(hyp)]) == 0
    out = capsys.readouterr().out
    assert "CER: 0.00%" in out and "identical after normalisation: 1" in out


def test_score_missing_id_fails_loudly(tmp_path: Path, capsys) -> None:
    ref = _tsv(tmp_path / "ref.tsv", [("a", "ա", "a"), ("b", "բ", "b")])
    hyp = _tsv(tmp_path / "hyp.tsv", [("a", "ա", "a")])
    assert score.main([str(ref), str(hyp)]) == 1
    assert "missing in hypothesis ['b']" in capsys.readouterr().err


def test_score_bad_header_fails(tmp_path: Path, capsys) -> None:
    ref = tmp_path / "ref.tsv"
    ref.write_text("id\ttext\na\tա\n", encoding="utf-8")
    hyp = _tsv(tmp_path / "hyp.tsv", [("a", "ա", "a")])
    assert score.main([str(ref), str(hyp)]) == 1
    assert "header must be" in capsys.readouterr().err


def test_rating_sheet_round_trip(tmp_path: Path, capsys) -> None:
    ref = _tsv(tmp_path / "ref.tsv", [("a", "ա", "x"), ("b", "բ", "y")])
    hyp = _tsv(tmp_path / "hyp.tsv", [("a", "ա", "x"), ("b", "բ", "z")])
    sheet = tmp_path / "rate.tsv"
    assert score.main([str(ref), str(hyp), "--rating-sheet", str(sheet)]) == 0
    assert score.main([str(ref), str(hyp), "--rating-sheet", str(sheet)]) == 1  # no overwrite

    lines = sheet.read_text(encoding="utf-8").splitlines()
    filled = [lines[0], lines[1] + "3", lines[2] + "1"]
    sheet.write_text("\n".join(filled) + "\n", encoding="utf-8")
    capsys.readouterr()
    assert score.main([str(ref), str(hyp), "--ratings", str(sheet)]) == 0
    assert "rating: mean 2.00  (1: 1, 2: 0, 3: 1)" in capsys.readouterr().out


def test_rating_outside_1_to_3_fails(tmp_path: Path, capsys) -> None:
    ref = _tsv(tmp_path / "ref.tsv", [("a", "ա", "x")])
    hyp = _tsv(tmp_path / "hyp.tsv", [("a", "ա", "x")])
    sheet = tmp_path / "rate.tsv"
    sheet.write_text("id\thy_ref\thy_hyp\ten_ref\ten_hyp\trating\na\tա\tա\tx\tx\t4\n", encoding="utf-8")
    assert score.main([str(ref), str(hyp), "--ratings", str(sheet)]) == 1
    assert "must be 1, 2 or 3" in capsys.readouterr().err
