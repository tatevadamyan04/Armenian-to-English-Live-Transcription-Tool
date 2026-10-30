"""Download the models into models/ — the only part of the project that uses the network (D12, D20).

Standard library only (D20): the Hugging Face files come from a fixed commit, so their
content can't change under us; the Silero VAD file comes from a release tag of its GitHub
repository (D19). Every model file is checked against a pinned SHA-256; small text files
(vocabularies, configs, licences) are fetched from the same pinned commit or tag. Each
model folder gets a SOURCE.txt with the source, version, licence and every file's SHA-256.

    python scripts/download_models.py           download what is missing or wrong
    python scripts/download_models.py --verify  check the files on disk only (no network)

Re-running is safe: a large file whose checksum already matches is not downloaded again.
There are no retries (CLAUDE.md): if a download fails, the error is printed and the
script exits non-zero; run it again to continue with the files still missing.
"""

from __future__ import annotations

import argparse
import hashlib
import shutil
import sys
import time
import urllib.request
from dataclasses import dataclass
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
MODELS_DIR = ROOT / "models"
CHUNK = 1 << 20  # 1 MiB per read
FREE_MARGIN = 1 << 30  # keep at least 1 GiB free after the downloads


@dataclass(frozen=True)
class File:
    name: str  # file name in the model folder
    sha256: str | None  # pinned for model files; None for small text files
    approx_bytes: int  # for the disk-space check and progress only
    remote: str | None = None  # path at the source, when it differs from `name`


@dataclass(frozen=True)
class Model:
    folder: str  # folder under models/
    source: str  # web page of the source repository
    version: str  # full commit hash or release tag
    url_base: str  # each file is downloaded from url_base/<remote or name>
    licence: str
    files: tuple[File, ...]


def hf_model(folder: str, repo: str, commit: str, licence: str, files: tuple[File, ...]) -> Model:
    """A model from a Hugging Face repository, pinned to a full commit hash."""
    return Model(
        folder=folder,
        source=f"https://huggingface.co/{repo}",
        version=commit,
        url_base=f"https://huggingface.co/{repo}/resolve/{commit}",
        licence=licence,
        files=files,
    )


SILERO_TAG = "v6.2.1"

MODELS = (
    Model(  # D4, D19 · phrase segmentation
        folder="vad-silero",
        source="https://github.com/snakers4/silero-vad",
        version=SILERO_TAG,
        url_base=f"https://raw.githubusercontent.com/snakers4/silero-vad/{SILERO_TAG}",
        licence="MIT (Silero Team; LICENSE file saved alongside)",
        files=(
            # SHA-256 from an independent mirror that states it matches the upstream v6.2.1 file
            # byte for byte (https://huggingface.co/bitsydarel/silero-vad-onnx); a mismatch stops the script.
            File("silero_vad.onnx", "1a153a22f4509e292a94e67d6f9b85e8deb25b4988682b7e174c65279d8788e3", 2_300_000,
                 remote="src/silero_vad/data/silero_vad.onnx"),
            File("LICENSE", None, 1_100),
        ),
    ),
    hf_model(  # D5 · ASR
        folder="asr-hy-fastconformer",
        repo="OpenVoiceOS/stt_hy_fastconformer_hybrid_large_pc_onnx",
        commit="f7db1fad9af6bd8a7a9e08fca02068c403b35468",
        licence="CC-BY-4.0 (NVIDIA stt_hy_fastconformer_hybrid_large_pc; ONNX export by OpenVoiceOS)",
        files=(
            File("model.onnx", "09ed4278c8536193f5324ae5b1e21be3ef1abaa92779e8c7b4577644071b131a", 458_000_000),
            File("vocab.txt", None, 13_000),
            File("config.json", None, 100),
            File("README.md", None, 3_000),
        ),
    ),
    hf_model(  # D6 · translation
        folder="mt-madlad400-3b-ct2-int8",
        repo="Nextcloud-AI/madlad400-3b-mt-ct2-int8",
        commit="aa32bbdeba7880eff2096ec044cb155a340a9400",
        licence="Apache-2.0 (google/madlad400-3b-mt; CTranslate2 int8 conversion by Nextcloud)",
        files=(
            File("model.bin", "77b9fd9ab97c1259d07089b5f854393dad81bc5fb5647d3f9a5d101c94f40daa", 2_950_000_000),
            File("spiece.model", "ef11ac9a22c7503492f56d48dce53be20e339b63605983e9f27d2cd0e0f3922c", 4_430_000),
            File("config.json", None, 300),
            File("shared_vocabulary.json", None, 5_480_000),
            File("README.md", None, 10_000),
        ),
    ),
)


class DownloadError(Exception):
    """A file couldn't be downloaded or didn't match its checksum; the message says which."""


def url_of(model: Model, file: File) -> str:
    return f"{model.url_base}/{file.remote or file.name}"


def sha256_of(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        while block := handle.read(CHUNK):
            digest.update(block)
    return digest.hexdigest()


def is_valid(path: Path, file: File) -> bool:
    """A large file is valid when its checksum matches; a small file when it exists and isn't empty."""
    if not path.is_file() or path.stat().st_size == 0:
        return False
    return file.sha256 is None or sha256_of(path) == file.sha256


def download(url: str, target: Path, file: File) -> str:
    """Stream url into target via a .part file; return the SHA-256 of what was received."""
    part = target.with_name(target.name + ".part")
    digest = hashlib.sha256()
    received = 0
    last_report = time.monotonic()
    request = urllib.request.Request(url, headers={"User-Agent": "hyen-download-models"})
    try:
        with urllib.request.urlopen(request) as response, part.open("wb") as out:
            total = int(response.headers.get("Content-Length") or file.approx_bytes)
            while block := response.read(CHUNK):
                out.write(block)
                digest.update(block)
                received += len(block)
                if time.monotonic() - last_report >= 5:
                    last_report = time.monotonic()
                    print(f"    {received / 1e6:8.0f} / {total / 1e6:.0f} MB", file=sys.stderr)
    except OSError as error:  # URLError and HTTPError are OSErrors
        part.unlink(missing_ok=True)
        raise DownloadError(f"downloading {url} failed: {error}") from error
    actual = digest.hexdigest()
    if file.sha256 is not None and actual != file.sha256:
        part.unlink()
        raise DownloadError(
            f"{url}: SHA-256 is {actual}, expected {file.sha256}. The file was deleted; "
            "run the script again, and if it fails the same way, the published file has changed."
        )
    part.replace(target)
    return actual


def check_disk_space(missing_bytes: int) -> None:
    free = shutil.disk_usage(MODELS_DIR).free
    if free < missing_bytes + FREE_MARGIN:
        raise DownloadError(
            f"not enough disk space: {free / 1e9:.1f} GB free, about "
            f"{(missing_bytes + FREE_MARGIN) / 1e9:.1f} GB needed. Free some space and run again."
        )


def write_source(model: Model, folder: Path, hashes: dict[str, str]) -> None:
    lines = [
        f"source: {model.source}",
        f"version: {model.version}",
        f"licence: {model.licence}",
        "files (SHA-256):",
        *(f"  {name}  {hashes[name]}" for name in sorted(hashes)),
    ]
    (folder / "SOURCE.txt").write_text("\n".join(lines) + "\n", encoding="utf-8")


def run_download() -> None:
    MODELS_DIR.mkdir(exist_ok=True)
    todo = [
        (model, file)
        for model in MODELS
        for file in model.files
        if not is_valid(MODELS_DIR / model.folder / file.name, file)
    ]
    check_disk_space(sum(file.approx_bytes for _, file in todo))
    for model in MODELS:
        folder = MODELS_DIR / model.folder
        folder.mkdir(exist_ok=True)
        hashes = {}
        print(f"{model.folder}  ({model.source} @ {model.version[:7]})", file=sys.stderr)
        for file in model.files:
            target = folder / file.name
            if (model, file) in todo:
                print(f"  downloading {file.name} (~{file.approx_bytes / 1e6:,.0f} MB)", file=sys.stderr)
                hashes[file.name] = download(url_of(model, file), target, file)
            else:
                print(f"  {file.name}: already present and valid", file=sys.stderr)
                hashes[file.name] = sha256_of(target)
        write_source(model, folder, hashes)
    print(f"Done. Models are in {MODELS_DIR}", file=sys.stderr)


def run_verify() -> int:
    problems = 0
    for model in MODELS:
        for file in model.files:
            path = MODELS_DIR / model.folder / file.name
            if is_valid(path, file):
                print(f"ok       {model.folder}/{file.name}", file=sys.stderr)
            else:
                problems += 1
                print(f"MISSING  {model.folder}/{file.name} (missing, empty or wrong checksum)", file=sys.stderr)
    if problems:
        print(f"{problems} file(s) not valid. Run: python scripts/download_models.py", file=sys.stderr)
        return 1
    print("All model files present and valid.", file=sys.stderr)
    return 0


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Download the models into models/ (D20).")
    parser.add_argument("--verify", action="store_true", help="check the files on disk only; no network")
    args = parser.parse_args(argv)
    if args.verify:
        return run_verify()
    try:
        run_download()
    except DownloadError as error:
        print(f"error: {error}", file=sys.stderr)
        return 1
    return 0


if __name__ == "__main__":
    sys.exit(main())