"""S05 smoke test for M4's embedder (SherpaEmbedder) on a real recording.

Checks that the sherpa-onnx API used in src/hyen/speakers.py exists and works, that it
loads in the same process as onnxruntime (ASR/VAD), the embedding size, the time per
embedding, and same-vs-different similarities on two stretches of one recording.

    python eval/check_speaker_model.py MODEL.onnx data/e3/system.wav A_START A_END B_START B_END

A_* and B_* are seconds: two stretches you know are spoken by different people (one
person each). Each stretch is cut into 2-s pieces.
"""

from __future__ import annotations

import sys
import time
import wave
from pathlib import Path

import numpy as np


def read_wav(path: Path) -> np.ndarray:
    with wave.open(str(path), "rb") as w:
        if (w.getframerate(), w.getnchannels(), w.getsampwidth()) != (16_000, 1, 2):
            sys.exit(f"error: {path} must be 16 kHz mono 16-bit; it is "
                     f"{w.getframerate()} Hz, {w.getnchannels()} ch, {8 * w.getsampwidth()}-bit")
        return np.frombuffer(w.readframes(w.getnframes()), dtype=np.int16).astype(np.float32) / 32768


def pieces(audio: np.ndarray, start: float, end: float, seconds: float = 2.0) -> list[np.ndarray]:
    step = int(seconds * 16_000)
    a, b = int(start * 16_000), int(end * 16_000)
    return [audio[i : i + step] for i in range(a, b - step + 1, step)]


def main() -> None:
    if len(sys.argv) != 7:
        sys.exit(__doc__)
    import onnxruntime  # loaded first on purpose: the live program has it loaded already

    from hyen.speakers import SherpaEmbedder

    model, wav = Path(sys.argv[1]), Path(sys.argv[2])
    a0, a1, b0, b1 = map(float, sys.argv[3:])
    print(f"onnxruntime {onnxruntime.__version__}")
    t = time.perf_counter()
    embedder = SherpaEmbedder(model)
    print(f"model loaded in {time.perf_counter() - t:.2f} s · embedding size {embedder.dim}")

    audio = read_wav(wav)
    groups = {"A": pieces(audio, a0, a1), "B": pieces(audio, b0, b1)}
    vectors: dict[str, list[np.ndarray]] = {}
    times = []
    for name, group in groups.items():
        vectors[name] = []
        for piece in group:
            t = time.perf_counter()
            v = embedder.embed(piece)
            times.append(time.perf_counter() - t)
            vectors[name].append(v / np.linalg.norm(v))
    print(f"embeddings: {len(times)} × 2 s · median {np.median(times) * 1000:.0f} ms · max {max(times) * 1000:.0f} ms")
    t = time.perf_counter()
    embedder.embed(audio[: 8 * 16_000])
    print(f"one 8-s segment: {(time.perf_counter() - t) * 1000:.0f} ms")

    def sims(x: list[np.ndarray], y: list[np.ndarray], same: bool) -> list[float]:
        return [float(u @ v) for i, u in enumerate(x) for j, v in enumerate(y) if not (same and j <= i)]

    for label, values in [("A–A", sims(vectors["A"], vectors["A"], True)),
                          ("B–B", sims(vectors["B"], vectors["B"], True)),
                          ("A–B", sims(vectors["A"], vectors["B"], False))]:
        if values:
            print(f"cosine {label}: n={len(values)} min {min(values):.2f} median {np.median(values):.2f} max {max(values):.2f}")


if __name__ == "__main__":
    main()
