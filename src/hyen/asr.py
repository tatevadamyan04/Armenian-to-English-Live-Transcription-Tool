"""M3 · Armenian speech recognition (PLAN.md §5, D5).

NVIDIA stt_hy_fastconformer_hybrid_large_pc, CTC head, from the OpenVoiceOS ONNX export,
run by onnx-asr on ONNX Runtime (CPU). Loaded from models/asr-hy-fastconformer (D20).

onnx-asr's load_model creates a missing model folder and downloads into it (onnx-asr
usage docs). To keep runtime local-only (C2, D12) the three model files are checked here
first, and huggingface_hub, which that download would need, isn't installed.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import onnx_asr

from hyen.modelfiles import ASR_DIR, require

SAMPLE_RATE = 16_000
MODEL_TYPE = "nemo-conformer-ctc"  # as declared in the export's config.json
MODEL_FILES = ("model.onnx", "vocab.txt", "config.json")


class Recognizer:
    """Armenian text for a segment of 16 kHz mono float32 audio."""

    def __init__(self, model_dir: Path = ASR_DIR) -> None:
        for name in MODEL_FILES:
            require(model_dir / name, f"ASR model file {name}")
        self._model = onnx_asr.load_model(MODEL_TYPE, str(model_dir), providers=["CPUExecutionProvider"])

    def transcribe(self, samples: np.ndarray) -> str:
        """The recognised text, stripped; an empty string means nothing was recognised."""
        if samples.ndim != 1 or samples.dtype != np.float32:
            raise ValueError(f"expected 1-D float32 samples, got {samples.dtype} {samples.shape}")
        text = self._model.recognize(samples, sample_rate=SAMPLE_RATE)
        if not isinstance(text, str):
            raise TypeError(f"onnx-asr returned {type(text).__name__}, expected str for one waveform")
        return text.strip()
