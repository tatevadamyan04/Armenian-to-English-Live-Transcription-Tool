"""D24 guards: very short ASR text counts as no text; MT output length is bounded by input length."""

from __future__ import annotations

import pytest

from hyen.asr import MIN_LETTERS, armenian_letters
from hyen.translate import MAX_TOKENS, OUTPUT_MARGIN, OUTPUT_RATIO, max_output_tokens


@pytest.mark.parametrize(
    "text, letters",
    [("Հ։", 1), ("հա", 2), ("Ու՞", 1 + 1), ("և", 1), ("H: 12", 0), ("Բարև ձեզ։", 7), ("", 0)],
)
def test_armenian_letters(text: str, letters: int) -> None:
    assert armenian_letters(text) == letters


def test_the_e3_fragment_is_below_the_minimum_and_short_replies_are_not() -> None:
    assert armenian_letters("Հ։") < MIN_LETTERS  # E3 system-27
    for reply in ("հա", "ու", "այո", "չէ"):
        assert armenian_letters(reply) >= MIN_LETTERS


def test_output_limit_grows_with_input_and_is_capped() -> None:
    assert max_output_tokens(1) == int(OUTPUT_RATIO) + OUTPUT_MARGIN
    assert max_output_tokens(10) == int(OUTPUT_RATIO * 10) + OUTPUT_MARGIN
    assert max_output_tokens(10_000) == MAX_TOKENS
    assert max_output_tokens(4) < 256  # a one-letter input can no longer run to 256 tokens


def test_output_limit_rejects_empty_input() -> None:
    with pytest.raises(ValueError):
        max_output_tokens(0)
