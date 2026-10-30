"""M3 and M5 with the real models — runs on the Mac, where models/ and data/e1/ exist.

Checks from TESTS.md: Armenian script on every HY line (T07), a non-empty English line
for every Armenian text (T08), nothing recognised in silence (T20). Prints every result
and the time each call took, as the first latency numbers for PLAN.md §5; CER/WER and
chrF are computed in S04.

    python -m pytest -q -s tests/test_asr_mt.py
"""

from __future__ import annotations

import csv
import re
import time
import wave
from pathlib import Path

import numpy as np
import pytest

from hyen.asr import Recognizer
from hyen.modelfiles import ROOT
from hyen.translate import Translator

E1_DIR = ROOT / "data" / "e1"
E1_REFS = ROOT / "eval" / "refs" / "e1.tsv"
ARMENIAN_LETTER = re.compile("[Ա-Ֆա-և]")
LATIN_LETTER = re.compile("[A-Za-z]")


def read_wav(path: Path) -> np.ndarray:
    if not path.exists():
        pytest.fail(f"{path} not found; record E1 first (eval/record.py prompts, session 02)")
    with wave.open(str(path), "rb") as reader:
        assert (reader.getframerate(), reader.getnchannels(), reader.getsampwidth()) == (16_000, 1, 2), path
        frames = reader.readframes(reader.getnframes())
    return np.frombuffer(frames, dtype="<i2").astype(np.float32) / 32768.0


def load_refs() -> list[dict[str, str]]:
    with E1_REFS.open(encoding="utf-8", newline="") as handle:
        return list(csv.DictReader(handle, delimiter="\t"))


REFS = load_refs()


@pytest.fixture(scope="module")
def recognizer() -> Recognizer:
    started = time.perf_counter()
    model = Recognizer()
    print(f"\nASR loaded in {time.perf_counter() - started:.1f} s")
    return model


@pytest.fixture(scope="module")
def translator() -> Translator:
    started = time.perf_counter()
    model = Translator()
    print(f"\nMT loaded in {time.perf_counter() - started:.1f} s")
    return model


@pytest.mark.xfail(
    strict=True,
    reason="known limitation (K10): on 3 s of digital zeros the ASR model alone returned 'Ինչ էին։' "
    "(2026-10-31). In the program the VAD keeps silence away from ASR, so T20 rests on M2.",
)
def test_asr_alone_gives_no_text_on_silence(recognizer):
    assert recognizer.transcribe(np.zeros(3 * 16_000, dtype=np.float32)) == ""


@pytest.mark.parametrize("ref", REFS, ids=[ref["id"] for ref in REFS])
def test_e1_asr_gives_armenian_text(recognizer, ref):  # T07
    samples = read_wav(E1_DIR / f"{ref['id']}.wav")
    started = time.perf_counter()
    hy = recognizer.transcribe(samples)
    seconds = time.perf_counter() - started
    print(f"\n{ref['id']}  ASR {seconds:.2f} s for {len(samples) / 16_000:.1f} s\n  hyp: {hy}\n  ref: {ref['hy']}")
    assert ARMENIAN_LETTER.search(hy), f"{ref['id']}: no Armenian letters in {hy!r}"


@pytest.mark.parametrize("ref", REFS, ids=[ref["id"] for ref in REFS])
def test_e1_translation_gives_english_text(translator, ref):  # T08
    started = time.perf_counter()
    en = translator.translate(ref["hy"])
    seconds = time.perf_counter() - started
    print(f"\n{ref['id']}  MT {seconds:.2f} s\n  hyp: {en}\n  ref: {ref['en']}")
    assert LATIN_LETTER.search(en), f"{ref['id']}: no English letters in {en!r}"
