"""D28: the ASR's spelling of "hm" («Հովհ») is removed before the D24 guard (pure text, no model)."""

from __future__ import annotations

import pytest

from hyen.asr import MIN_LETTERS, armenian_letters, remove_fillers


@pytest.mark.parametrize(
    ("asr", "expected"),
    [
        ("Հովհ.", ""),  # E4 replay, 0:34
        ("Հովհ է.", "է."),  # 0:07 → one letter left, dropped by D24
        ("ովհ է։", "է։"),  # 0:29, first letter lost
        ("Հովհ գյուղ։", "գյուղ։"),  # 1:12 → the real word is kept and translated
        ("հովհ, լավ եմ", "լավ եմ"),
        ("ՀՈՎՀ։", ""),
    ],
)
def test_filler_is_removed(asr, expected):
    assert remove_fillers(asr) == expected


@pytest.mark.parametrize(
    "text",
    ["Հովհաննեսը եկավ։", "Բարեւ Տաթեւ։", "Լավ է, դու ինչպե՞ս ես։", "Տովհ"],
)
def test_real_words_are_untouched(text):
    assert remove_fillers(text) == text


def test_a_lone_letter_left_is_dropped_by_d24():
    assert armenian_letters(remove_fillers("Հովհ է.")) < MIN_LETTERS


class FakeModel:
    def __init__(self, text: str) -> None:
        self.text = text

    def recognize(self, samples, sample_rate):
        return self.text


def transcribe_with(text: str, capsys) -> tuple[str, str]:
    import numpy as np

    from hyen.asr import Recognizer

    recognizer = Recognizer.__new__(Recognizer)  # skip loading the real model
    recognizer._model = FakeModel(text)
    result = recognizer.transcribe(np.zeros(16_000, dtype=np.float32))
    return result, capsys.readouterr().err


def test_transcribe_drops_a_segment_that_was_only_hm(capsys):
    result, err = transcribe_with("Հովհ է.", capsys)
    assert result == ""
    assert "D28" in err and "D24" in err


def test_transcribe_keeps_the_rest_and_notes_the_removal(capsys):
    result, err = transcribe_with("Հովհ գյուղ։", capsys)
    assert result == "գյուղ։" and "D28" in err


def test_transcribe_is_silent_when_there_is_no_filler(capsys):
    result, err = transcribe_with("Բարեւ Տաթեւ։", capsys)
    assert result == "Բարեւ Տաթեւ։" and err == ""
