"""M5 · Armenian → English translation (PLAN.md §5, D6).

MADLAD-400-3B-MT as a CTranslate2 int8 model on the CPU, loaded from
models/mt-madlad400-3b-ct2-int8 (D20). The input is tokenised with the model's own
SentencePiece file and prefixed with "<2en>", MADLAD's tag for the target language
(usage as on the CTranslate2 model cards; sources in DECISIONS.md D6).

Greedy decoding (beam size 1) to start with, for latency; the S04 measurements decide
whether a larger beam fits the budget.
"""

from __future__ import annotations

from pathlib import Path

import ctranslate2
import sentencepiece

from hyen.modelfiles import MT_DIR, require

TARGET_TAG = "<2en>"
THREADS = 8  # CPU threads for translation; starting value to measure in S04 (M4 Pro: 12+ cores)
BEAM_SIZE = 1
MAX_TOKENS = 256  # an upper bound on the output length, so a runaway output can't stall the worker
MODEL_FILES = ("model.bin", "config.json", "shared_vocabulary.json", "spiece.model")


class Translator:
    """English text for a piece of Armenian text."""

    def __init__(self, model_dir: Path = MT_DIR) -> None:
        for name in MODEL_FILES:
            require(model_dir / name, f"translation model file {name}")
        self._tokenizer = sentencepiece.SentencePieceProcessor(model_file=str(model_dir / "spiece.model"))
        self._model = ctranslate2.Translator(
            str(model_dir), device="cpu", compute_type="int8", inter_threads=1, intra_threads=THREADS
        )

    def translate(self, hy: str) -> str:
        if not hy.strip():
            raise ValueError("nothing to translate: the Armenian text is empty")
        tokens = self._tokenizer.encode(f"{TARGET_TAG} {hy}", out_type=str)
        results = self._model.translate_batch(
            [tokens], beam_size=BEAM_SIZE, max_decoding_length=MAX_TOKENS
        )
        return self._tokenizer.decode(results[0].hypotheses[0]).strip()
