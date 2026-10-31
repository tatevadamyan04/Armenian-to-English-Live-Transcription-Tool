"""Unit tests for eval/score_e3.py: time parsing, the window rule, loud failure on bad input."""

from __future__ import annotations

import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "eval"))  # eval/ holds scripts (PLAN.md §6)

import score_e3  # noqa: E402
from score_e3 import InputError, Span, in_window, parse_time  # noqa: E402

HYP_HEADER = "seg_id\tsource\tstart\tend\tcut\thy\ten\tasr_s\tmt_s\n"


@pytest.mark.parametrize("text, seconds", [("0:04.5", 4.5), ("1:05", 65.0), ("10:00", 600.0), ("65.5", 65.5)])
def test_parse_time(text: str, seconds: float) -> None:
    assert parse_time(text) == pytest.approx(seconds)


@pytest.mark.parametrize("text", ["", "1:xx", "a:05", "-3"])
def test_bad_time_is_an_error(text: str) -> None:
    with pytest.raises(InputError):
        parse_time(text)


def test_window_uses_midpoints() -> None:
    spans = [Span(0, 4, "A", "", ""), Span(9, 13, "A", "", ""), Span(12, 20, "B", "", "")]
    # midpoints 2, 11, 16; window [10, 16) keeps only the second
    assert in_window(spans, 10, 16) == [spans[1]]


def _write(path: Path, text: str) -> Path:
    path.write_text(text, encoding="utf-8")
    return path


def test_scores_the_window(tmp_path: Path, capsys) -> None:
    ref = _write(tmp_path / "ref.tsv",
                 "start\tend\tspeaker\thy\ten\n0:00\t0:04\tA\tԲարև ձեզ։\tHello.\n0:05\t0:09\tB\tԻ՞նչ կա։\t\n")
    hyp = _write(tmp_path / "hyp.tsv", HYP_HEADER
                 + "system-1\tsystem\t0.10\t3.80\t0\tբարև ձեզ\t\t0.01\t0\n"
                 + "system-2\tsystem\t5.20\t8.90\t0\tինչ կա\t\t0.01\t0\n")
    assert score_e3.main([str(ref), str(hyp)]) == 0
    out = capsys.readouterr().out
    assert "CER: 0.00%" in out and "2 reference turns, 2 segments" in out


def test_untranscribed_turn_inside_window_fails(tmp_path: Path, capsys) -> None:
    ref = _write(tmp_path / "ref.tsv", "start\tend\tspeaker\thy\ten\n"
                 "0:00\t0:04\tA\tԲարև։\t\n0:05\t0:09\tB\t\t\n0:10\t0:14\tA\tԼավ։\t\n")
    hyp = _write(tmp_path / "hyp.tsv", HYP_HEADER + "system-1\tsystem\t0.1\t3.8\t0\tբարև\t\t0\t0\n")
    assert score_e3.main([str(ref), str(hyp)]) == 1
    assert "no Armenian text" in capsys.readouterr().err


def test_end_before_start_fails(tmp_path: Path, capsys) -> None:
    ref = _write(tmp_path / "ref.tsv", "start\tend\tspeaker\thy\ten\n0:09\t0:04\tA\tԲարև։\t\n")
    hyp = _write(tmp_path / "hyp.tsv", HYP_HEADER + "system-1\tsystem\t0.1\t3.8\t0\tբարև\t\t0\t0\n")
    assert score_e3.main([str(ref), str(hyp)]) == 1
    assert "not after start" in capsys.readouterr().err


def test_export_mt(tmp_path: Path) -> None:
    ref = _write(tmp_path / "ref.tsv",
                 "start\tend\tspeaker\thy\ten\n0:00\t0:04\tA\tԲարև։\tHi.\n0:05\t0:09\tB\tԼավ։\t\n")
    hyp = _write(tmp_path / "hyp.tsv", HYP_HEADER + "system-1\tsystem\t0.1\t3.8\t0\tբարև\t\t0\t0\n"
                 + "system-2\tsystem\t5.1\t8.8\t0\tլավ\t\t0\t0\n")
    out = tmp_path / "mt.tsv"
    assert score_e3.main([str(ref), str(hyp), "--export-mt", str(out)]) == 0
    assert out.read_text(encoding="utf-8") == "id\thy\ten\ne3_01\tԲարև։\tHi.\n"
