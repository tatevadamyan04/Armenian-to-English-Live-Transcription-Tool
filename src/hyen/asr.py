"""M3 · Armenian speech recognition (PLAN.md §5, D5).

NVIDIA stt_hy_fastconformer_hybrid_large_pc, CTC head, from the OpenVoiceOS ONNX export,
run by onnx-asr on ONNX Runtime (CPU). Loaded from models/asr-hy-fastconformer (D20).

onnx-asr's load_model creates a missing model folder and downloads into it (onnx-asr
usage docs). To keep runtime local-only (C2, D12) the three model files are checked here
first, and huggingface_hub, which that download would need, isn't installed.
"""

from __future__ import annotations

import re
import sys
from pathlib import Path

import numpy as np
import onnx_asr

from hyen.modelfiles import ASR_DIR, require

SAMPLE_RATE = 16_000
MODEL_TYPE = "nemo-conformer-ctc"  # as declared in the export's config.json
MODEL_FILES = ("model.onnx", "vocab.txt", "config.json")
# D24: text with fewer Armenian letters than this is treated as nothing recognised. In E3 a
# fragment came out as «Հ։», which MT turned into an invented reply. Two-letter replies
# («հա», «ու») stay.
MIN_LETTERS = 2
_ARMENIAN_LETTER = re.compile("[\u0531-\u0556\u0561-\u0587]")  # capitals; small letters incl. և


def armenian_letters(text: str) -> int:
    """How many Armenian letters the text holds (punctuation, digits and Latin not counted)."""
    return len(_ARMENIAN_LETTER.findall(text))


# D28: in the E4 replay the ASR wrote a "hm" as «Հովհ» (also «ովհ», first letter lost), and MT
# turned it into invented text ("John 15:23-24", "2013. ^ a b c …"): «Հովհ.» abbreviates
# Հովհաննես in Bible citations. A lone «Հ» is the same sound cut shorter («Հ գյուղ» → "2013.
# (in Japanese) 村田, 久保."). Both are removed only as whole words, with the punctuation
# after them, so «Հովհաննես» and «Հայաստան» stay.
_FILLER = re.compile(r"(?<!\w)(?:հ?ովհ|հ)(?!\w)[.։,՝:;!?…]*", re.IGNORECASE)


def remove_fillers(text: str) -> str:
    """The text without the ASR's spellings of "hm" — «Հովհ», «ովհ», a lone «Հ» (D28) — whitespace collapsed."""
    return " ".join(_FILLER.sub(" ", text).split())


class Recognizer:
    """Armenian text for a segment of 16 kHz mono float32 audio."""

    def __init__(self, model_dir: Path = ASR_DIR) -> None:
        for name in MODEL_FILES:
            require(model_dir / name, f"ASR model file {name}")
        self._model = onnx_asr.load_model(MODEL_TYPE, str(model_dir), providers=["CPUExecutionProvider"])

    def transcribe(self, samples: np.ndarray) -> str:
        """The recognised text, stripped; an empty string means nothing was recognised.

        The fillers «Հովհ» and a lone «Հ» ("hm") are removed first (D28). Text with fewer than MIN_LETTERS
        Armenian letters then counts as nothing recognised (D24), so the caller drops it like
        empty text. Both are noted on stderr.
        """
        if samples.ndim != 1 or samples.dtype != np.float32:
            raise ValueError(f"expected 1-D float32 samples, got {samples.dtype} {samples.shape}")
        text = self._model.recognize(samples, sample_rate=SAMPLE_RATE)
        if not isinstance(text, str):
            raise TypeError(f"onnx-asr returned {type(text).__name__}, expected str for one waveform")
        text = text.strip()
        cleaned = remove_fillers(text)
        if cleaned != " ".join(text.split()):
            print(f"asr: removed filler (hm): {text!r} → {cleaned!r} (D28)", file=sys.stderr)
        text = cleaned
        if text and armenian_letters(text) < MIN_LETTERS:
            print(f"asr: {text!r} has fewer than {MIN_LETTERS} Armenian letters; treated as no text (D24)",
                  file=sys.stderr)
            return ""
        return text
