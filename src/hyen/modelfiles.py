"""Where the downloaded model files live (D20), and the check that they are there (F03).

The files are fetched only by scripts/download_models.py into models/ at the repository
root. Nothing here downloads anything: a missing file is an error that names the file
and the command that fetches it.
"""

from __future__ import annotations

from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]  # src/hyen/modelfiles.py → repository root
MODELS_DIR = ROOT / "models"

SILERO_VAD = MODELS_DIR / "vad-silero" / "silero_vad.onnx"  # D4, D19
ASR_DIR = MODELS_DIR / "asr-hy-fastconformer"  # D5
MT_DIR = MODELS_DIR / "mt-madlad400-3b-ct2-int8"  # D6


class ModelMissingError(Exception):
    """A model file isn't on disk; the message names it and the download command."""


def require(path: Path, what: str) -> Path:
    """Return path if it exists; otherwise raise ModelMissingError (F03)."""
    if not path.exists():
        raise ModelMissingError(
            f"{what} not found at {path}.\n"
            "Download the models first (this is the only step that uses the network):\n"
            "  python scripts/download_models.py"
        )
    return path
