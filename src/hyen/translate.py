"""M5 · Armenian → English translation (PLAN.md §5, D6).

MADLAD-400-3B-MT as a CTranslate2 int8 model on the CPU, loaded from
models/mt-madlad400-3b-ct2-int8 (D20). The input is tokenised with the model's own
SentencePiece file and prefixed with "<2en>", MADLAD's tag for the target language
(usage as on the CTranslate2 model cards; sources in DECISIONS.md D6).

Greedy decoding (beam size 1) to start with, for latency; the S04 measurements decide
whether a larger beam or repetition limits are used (D22). The decoding options are
CTranslate2's TranslationOptions fields (beam_size, repetition_penalty, no_repeat_ngram_size,
max_decoding_length); their defaults here equal CTranslate2's own, except beam size.
"""

from __future__ import annotations

from pathlib import Path

import ctranslate2
import sentencepiece

from hyen.modelfiles import MT_DIR, require

TARGET_TAG = "<2en>"
THREADS = 8  # CPU threads for translation; starting value to measure in S04 (M4 Pro: 12+ cores)
BEAM_SIZE = 1
REPETITION_PENALTY = 1.0  # CTranslate2's default: no penalty; > 1 penalises tokens already generated
NO_REPEAT_NGRAM_SIZE = 0  # CTranslate2's default: off; N > 0 forbids repeating any N-gram
MAX_TOKENS = 256  # an upper bound on the output length, so a runaway output can't stall the worker
MODEL_FILES = ("model.bin", "config.json", "shared_vocabulary.json", "spiece.model")


class Translator:
    """English text for a piece of Armenian text."""

    def __init__(
        self,
        model_dir: Path = MT_DIR,
        beam_size: int = BEAM_SIZE,
        repetition_penalty: float = REPETITION_PENALTY,
        no_repeat_ngram_size: int = NO_REPEAT_NGRAM_SIZE,
    ) -> None:
        if beam_size < 1 or repetition_penalty <= 0 or no_repeat_ngram_size < 0:
            raise ValueError(
                f"invalid decoding settings: beam {beam_size}, repetition penalty {repetition_penalty}, "
                f"no-repeat n-gram {no_repeat_ngram_size}"
            )
        self._decoding = dict(
            beam_size=beam_size,
            repetition_penalty=repetition_penalty,
            no_repeat_ngram_size=no_repeat_ngram_size,
            max_decoding_length=MAX_TOKENS,
        )
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
        results = self._model.translate_batch([tokens], **self._decoding)
        return self._tokenizer.decode(results[0].hypotheses[0]).strip()
