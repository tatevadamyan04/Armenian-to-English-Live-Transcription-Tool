"""M5 · Armenian → English translation (PLAN.md §5, D6).

MADLAD-400-3B-MT as a CTranslate2 int8 model on the CPU, loaded from
models/mt-madlad400-3b-ct2-int8 (D20). The input is tokenised with the model's own
SentencePiece file and prefixed with "<2en>", MADLAD's tag for the target language
(usage as on the CTranslate2 model cards; sources in DECISIONS.md D6).

Decoding (D23): beam 4, repetition penalty 1.2, no repeated 3-token sequence, chosen on E1
and long.tsv in S04. The options are CTranslate2's TranslationOptions fields (beam_size,
repetition_penalty, no_repeat_ngram_size, max_decoding_length). To keep translation time
bounded, M2 cuts segments at 8 s.
"""

from __future__ import annotations

import sys
from pathlib import Path

import ctranslate2
import sentencepiece

from hyen.modelfiles import MT_DIR, require

TARGET_TAG = "<2en>"
THREADS = 8  # CPU threads for translation; fixed, not tuned (S04 tuned decoding only, D23)
BEAM_SIZE = 4
REPETITION_PENALTY = 1.2  # > 1 penalises tokens already generated (CTranslate2's default is 1, off)
NO_REPEAT_NGRAM_SIZE = 3  # forbids repeating any 3-token sequence (CTranslate2's default is 0, off)
MAX_TOKENS = 256  # absolute upper bound on the output length
# D24: the output may be at most OUTPUT_RATIO × input tokens + OUTPUT_MARGIN tokens, so a
# runaway (E3: «Հ։» → "H: H: H: …", 256 tokens, 8.16 s at beam 1) can't break the 5-s limit.
# Set from the largest output/input ratio measured on E1, long.tsv and E3 (S04).
OUTPUT_RATIO = 2.0
OUTPUT_MARGIN = 16
MODEL_FILES = ("model.bin", "config.json", "shared_vocabulary.json", "spiece.model")


def max_output_tokens(input_tokens: int) -> int:
    """The decoding length limit for an input of this many tokens (D24)."""
    if input_tokens < 1:
        raise ValueError(f"input has {input_tokens} tokens")
    return min(MAX_TOKENS, int(OUTPUT_RATIO * input_tokens) + OUTPUT_MARGIN)


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
        limit = max_output_tokens(len(tokens))
        results = self._model.translate_batch([tokens], max_decoding_length=limit, **self._decoding)
        output = results[0].hypotheses[0]
        if len(output) >= limit:
            print(f"translate: output reached the length limit ({limit} tokens for {len(tokens)} input tokens) "
                  "and may be cut short (D24)", file=sys.stderr)
        return self._tokenizer.decode(output).strip()
