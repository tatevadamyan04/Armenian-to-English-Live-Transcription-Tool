"""M2 · Silero VAD run directly with onnxruntime (D4, D19).

One SileroVad per source: the model is recurrent, so each source keeps its own state.
At 16 kHz the model takes exactly 512 new samples per call — one AudioChunk (D17) —
plus the last 64 samples of the previous call as context, and returns the probability
that the chunk contains speech.

The calling convention (input = context + window, state of shape (2, 1, 128), sr as an
int64 scalar; outputs = probability and new state) follows Silero's own ONNX wrapper;
see DECISIONS.md D19 for the sources.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import onnxruntime as ort

from hyen.modelfiles import require

SAMPLE_RATE = 16_000
WINDOW = 512  # samples per call at 16 kHz
CONTEXT = 64  # samples carried over from the previous call at 16 kHz
STATE_SHAPE = (2, 1, 128)
INPUT_NAMES = {"input", "state", "sr"}


class SileroVad:
    """Speech probability for consecutive 512-sample chunks of one source."""

    def __init__(self, model_path: Path) -> None:
        require(model_path, "Silero VAD model")
        options = ort.SessionOptions()
        options.intra_op_num_threads = 1  # a tiny model; one thread per source leaves the cores to ASR and MT
        options.inter_op_num_threads = 1
        self._session = ort.InferenceSession(str(model_path), options, providers=["CPUExecutionProvider"])
        names = {node.name for node in self._session.get_inputs()}
        if names != INPUT_NAMES:
            raise RuntimeError(
                f"{model_path} has inputs {sorted(names)}, expected {sorted(INPUT_NAMES)}; "
                "this is not the Silero VAD v5/v6 model the code was written for (D19)."
            )
        self._sr = np.array(SAMPLE_RATE, dtype=np.int64)
        self.reset()

    def reset(self) -> None:
        """Forget everything heard so far, as at the start of a stream."""
        self._state = np.zeros(STATE_SHAPE, dtype=np.float32)
        self._context = np.zeros(CONTEXT, dtype=np.float32)

    def __call__(self, samples: np.ndarray) -> float:
        """Probability (0–1) that these 512 samples contain speech."""
        if samples.shape != (WINDOW,) or samples.dtype != np.float32:
            raise ValueError(f"expected float32 samples of shape ({WINDOW},), got {samples.dtype} {samples.shape}")
        x = np.concatenate([self._context, samples])[np.newaxis, :]
        outputs = self._session.run(None, {"input": x, "state": self._state, "sr": self._sr})
        if len(outputs) != 2:
            raise RuntimeError(f"Silero VAD returned {len(outputs)} outputs, expected 2 (probability, state)")
        probability, self._state = outputs
        self._context = x[0, -CONTEXT:]
        return float(probability.reshape(-1)[0])
