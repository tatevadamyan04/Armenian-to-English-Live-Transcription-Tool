# DECISIONS

Decision log for the live Armenian → English transcription project. Each entry gives the decision, the options considered, Claude's recommendation, Tatev's choice, the reasons, the evidence and what would make us revisit it. Claude researched the options; Tatev decided.

Status: **decided** · **decided — confirm by measurement** (the default stands unless the S04/S05 numbers say otherwise) · **superseded**

| ID | Topic | Choice | Status |
|---|---|---|---|
| D1 | Python environment | installed Python 3.14.5 + venv | decided |
| D2 | System-audio capture | BlackHole 2ch + Multi-Output Device | decided |
| D3 | Listening and microphone | AirPods for listening, MacBook mic as input | decided |
| D4 | Phrase segmentation | Silero VAD | decided — licence check in S03 |
| D5 | Armenian ASR | NVIDIA hy FastConformer (ONNX, onnx-asr) | decided — confirm by measurement |
| D6 | Translation | MADLAD-400-3B-MT (CTranslate2 int8) | decided — confirm by measurement |
| D7 | Speaker labels | ECAPA embeddings + online clustering | decided |
| D8 | Pipeline structure | one process, threads + queues, one inference worker | decided |
| D9 | Evaluation data and metrics | E1 read + E2 FLEURS + E3 call | decided |
| D10 | Licence policy | MIT, Apache-2.0, BSD, CC-BY-4.0 only | decided |
| D11 | Workflow | Claude sends files; Tatev commits and runs commands | decided |
| D12 | Proof of "local only" | runs with networking off | decided |
| D13 | Language scope | Eastern Armenian; code-switching is a known limitation | decided |

---

## D1 · Python environment

**Decided 2026-10-27 · Claude recommended A · Tatev chose C**

**Decision:** Use the installed Python 3.14.5 with a plain `venv`. Pin exact versions in `requirements.txt` once everything installs.

**Options**
- A — Python 3.12 managed by uv: widest wheel coverage for ML libraries; one extra tool.
- B — Python 3.13 managed by uv: same idea, slightly more risk of lagging libraries.
- **C — installed Python 3.14.5 (chosen).**

**Why (Tatev):** all the needed dependencies are already installed.

**Supporting evidence:** the core packages ship Python 3.14 builds for Apple Silicon — onnxruntime 1.31.0, ctranslate2 4.8.3, mlx 0.32.4 — and PyTorch recommends Python 3.10–3.14 on macOS.

**Accepted risk:** smaller dependencies (e.g. speechbrain, torchaudio) are unverified on 3.14.

**Revisit if:** a chosen dependency has no 3.14 build or fails to install → stop and re-decide D1. No building from source as a workaround.

**Sources:** [PyTorch Get Started](https://pytorch.org/get-started/locally/) · [onnxruntime files](https://simple-repository.app.cern.ch/project/onnxruntime) · [ctranslate2 files](https://simple-repository.app.cern.ch/project/ctranslate2) · [mlx files](https://simple-repository.app.cern.ch/project/mlx)

---

## D2 · System-audio capture on macOS 26

**Decided 2026-10-27 · Claude recommended A · Tatev chose A**

**Decision:** BlackHole 2ch inside a Multi-Output Device. The program reads "BlackHole 2ch" as an ordinary input device with python-sounddevice.

**Options**
- **A — BlackHole 2ch + Multi-Output Device (chosen).**
- B — Core Audio process tap via catap (MIT, 0.6.0, Aug 2026, beta, macOS 14.2+): no driver and can capture Zoom alone; but beta with one maintainer, needs the system-audio recording permission (otherwise silent buffers), and doesn't recover after sleep or device loss.
- C — audiotee (MIT Swift command-line tool) piped into Python: a non-Python helper to build, default output device only, interface may change.

**Why:** It is what the brief suggests and the best-documented route, and it works with any output device.

**Known issues**
- Set up in Audio MIDI Setup, with drift correction on every device except the clock device. BlackHole's FAQ says the built-in output should be enabled and listed first in a Multi-Output Device, and that AirPods shouldn't be the clock device (their mic runs at a lower sample rate). The exact configuration is settled in S02 and recorded here.
- Volume keys reportedly don't work with a Multi-Output Device.
- BlackHole states "macOS 10.10 and newer" and doesn't mention Tahoe. A vendor-written article claims it is glitchy on Tahoe (unverified).
- BlackHole is GPL-3.0; it is installed separately and isn't part of our code (D10).

**Revisit if:** crackles, dropouts or silence in T04/T05 → switch to B. Named now so it is a decision, not a silent runtime fallback.

**Sources:** [BlackHole README and FAQ](https://github.com/ExistentialAudio/BlackHole) · [vendor article on Tahoe](https://medium.com/@andrew_best/how-to-record-system-audio-on-mac-without-blackhole-macos-tahoe-fix-6cc488b728d1) · [catap](https://pypi.org/project/catap/) · [audiotee](https://github.com/makeusabrew/audiotee) · [Core Audio taps overview](https://www.recall.ai/blog/core-audio-taps)

---

## D3 · Listening and microphone setup

**Decided 2026-10-27 · Claude recommended A · Tatev chose A**

**Decision:** AirPods 4 for listening only, through the Multi-Output Device. "MacBook Pro Microphone" is the input for both Zoom and the program.

**Options**
- **A — AirPods for listening, MacBook mic as input (chosen).**
- B — laptop speakers + MacBook mic: the call leaks into the mic, so the other side's words also appear as "Me" (echo handling is out of scope).
- C — AirPods for listening and as the mic: macOS switches AirPods to compressed call audio while their mic is in use.

**Why:** Nothing leaks into the mic, and the AirPods keep full audio quality.

**Consequence:** Zoom's microphone must be set to the MacBook mic (README setup step; P02).

**Sources:** [AirPods quality drop when their mic is used](https://nicolasbouliane.com/blog/airpod-problems) · [BlackHole FAQ](https://github.com/ExistentialAudio/BlackHole)

---

## D4 · Phrase segmentation

**Decided 2026-10-27 · Claude recommended A · Tatev chose A · licence check pending**

**Decision:** Silero VAD (silero-vad 6.2.3) on 16 kHz audio. A phrase ends after about 600 ms of silence; segments are capped at about 12 s so long monologues still print. Both values are tuned in S03.

**Options**
- **A — Silero VAD (chosen).**
- B — WebRTC VAD: tiny and fast, but older and more easily triggered by noise; licence not verified.
- C — fixed 3 s windows: cuts words, feeds silence to ASR, and leaves no "end of phrase" for C1.

**Why:** Accurate and cheap; its documentation says it was trained on more than 6,000 languages.

**Open:** the PyPI text and the repository metadata say MIT, but a README badge says CC BY-NC 4.0. Read the LICENSE file in S03 before adopting it (D10).

**Sources:** [silero-vad on PyPI](https://pypi.org/project/silero-vad) · [repository metadata](https://science.ecosyste.ms/projects/32425)

---

## D5 · Armenian speech recognition

**Decided 2026-10-27 · Claude recommended A, with B measured against it · Tatev chose A · confirm by measurement**

**Decision:** NVIDIA `stt_hy_fastconformer_hybrid_large_pc` (CC-BY-4.0), run from the community ONNX export with onnx-asr (MIT) on ONNX Runtime — no PyTorch or NeMo.

**Options**
- **A — NVIDIA hy FastConformer (chosen).**
- B — BuzzASR/armenian (MIT, Whisper-large-v3 fine-tune, ~1.6B parameters): best self-reported numbers (11.89% WER on FLEURS), but tested on the same datasets it was trained on and not independently benchmarked; ~3 GB; speed on the Apple GPU unknown; Whisper-style invented text possible.
- C — Chillarmo/whisper-large-v3-turbo-armenian (Apache-2.0, 0.8B): 33.00% combined WER on ArmBench, behind A; same Whisper-style risk.

**Excluded:** SeamlessM4T v2 — CC-BY-NC-4.0 (D10). On accuracy (ArmBench combined WER): Whisper large-v3 65.12%, Omnilingual ASR 37.76% at best, MMS-1b-all 47.50%.

**Why:** Best open model on ArmBench-ASR (19.86% combined WER; 7.70% Common Voice 26; 15.00% FLEURS). Small (459 MB checkpoint) and fast on CPU. CTC decoding, expected (not yet measured) to invent whole sentences less often than Whisper-style decoders — tested by T20 and T21.

**Known weaknesses:** 60.78% WER on ArmBench "Movies" (dialogue). The ONNX export contains only the CTC decoder, and ArmBench doesn't say which of the model's two decoders it scored (checked by M01). The export is community-made, not NVIDIA's.

**Confirm by measurement (S04):** compare with B on E1–E3. Rule: lowest CER on E3, then E1, within the latency budget (TESTS.md §3.5). Confirmed by Tatev on 2026-10-30: B is still measured against A in S04.

**Sources:** [ArmBench-ASR](https://metric-ai-armbench-asr.static.hf.space/index.html) · [ArmBench release (Runtime Wire)](https://runtimewire.com/article/metric-armbench-asr-armenian-speech-benchmark) · [NVIDIA model files and licence](https://huggingface.co/nvidia/stt_hy_fastconformer_hybrid_large_pc/tree/refs%2Fpr%2F3) · [ONNX export](https://huggingface.co/OpenVoiceOS/stt_hy_fastconformer_hybrid_large_pc_onnx) · [onnx-asr](https://pypi.org/project/onnx-asr/) · [NeMo README, Apple silicon section (fork copy)](https://github.com/michalivne/NeMo) · [BuzzASR/armenian](https://huggingface.co/BuzzASR/armenian) · [BuzzASR paper](https://arxiv.org/abs/2609.09554) · [Chillarmo turbo licence](https://mixpeek.com/model/Chillarmo/whisper-large-v3-turbo-armenian) · [SeamlessM4T v2 licence](https://www.endorlabs.com/ai-model/facebook-seamless-m4t-v2-large) · [Careless Whisper (Whisper hallucinations)](https://arxiv.org/abs/2402.08021v2)

---

## D6 · Armenian → English translation

**Decided 2026-10-27 · Claude recommended A, with B measured against it · Tatev chose A · confirm by measurement**

**Decision:** MADLAD-400-3B-MT (Apache-2.0, Armenian listed) as a CTranslate2 int8 model on CPU; each input is prefixed with `<2en>`.

**Options**
- **A — MADLAD-400-3B-MT (chosen).**
- B — Gemma 4 12B instruct (Apache-2.0, June 2026), 4-bit via mlx-lm on the GPU: possibly more fluent (unverified); Armenian quality unknown; may paraphrase or add content; roughly 7 GB (estimate).
- C — OPUS-MT hy-en: tiny and fast; it exists, but its licence and quality are unverified.

**Excluded:** NLLB-200 — CC-BY-NC (D10). TranslateGemma — Armenian appears only in its English→Armenian training data, it isn't evaluated on Armenian, and no licence statement was found. arm-gemma-e4b — no instruction-tuned version yet.

**Why:** A dedicated translation model, expected (not yet measured) to add less content than an LLM; on CPU it leaves the GPU free.

**Unknowns:** no published hy→en score found; the model card says it was "not assessed for production"; CPU latency to measure.

**Confirm by measurement (S04):** compare with B on E1 and E3 — Tatev's 1–3 rating first, chrF as tie-breaker — within the latency budget. Confirmed by Tatev on 2026-10-30: B is still measured against A in S04.

**Sources:** [MADLAD-400-3B-MT CT2 card](https://huggingface.co/Nextcloud-AI/madlad400-3b-mt-ct2-int8/raw/aa32bbdeba7880eff2096ec044cb155a340a9400/README.md) · [CTranslate2 usage](https://huggingface.co/santhosh/madlad400-3b-ct2) · [Gemma 4 under Apache 2.0](https://opensource.googleblog.com/2026/03/gemma-4-expanding-the-gemmaverse-with-apache-20.html) · [Gemma 4 sizes](https://en.wikipedia.org/wiki/Gemma_(language_model)) · [gemma-4-12B-it-4bit (MLX)](https://huggingface.co/mlx-community/gemma-4-12B-it-4bit/discussions/2) · [TranslateGemma report](https://arxiv.org/pdf/2601.09012) · [arm-gemma-e4b paper](https://arxiv.org/abs/2609.03350) · [OPUS-MT hy-en listing](https://ai.azure.com/catalog/models/helsinki-nlp-opus-mt-hy-en) · [NLLB-200 card](https://huggingface.co/facebook/nllb-200-distilled-600M/blob/main/README.md)

---

## D7 · Speaker labels on the call side

**Decided 2026-10-27 · Claude recommended A · Tatev chose A**

**Decision:** For every system-audio segment, compute a speaker embedding — a fixed-length vector that describes the voice ("voice fingerprint") — with SpeechBrain ECAPA-TDNN (`speechbrain/spkrec-ecapa-voxceleb`, Apache-2.0). Compare it by cosine similarity with each known speaker's centroid, the mean of that speaker's embeddings so far. If the best similarity reaches the threshold, the segment gets that label and the centroid is updated; otherwise a new "Speaker N" is created. Mic segments are always "Me" and are never embedded.

**Options**
- **A — embeddings + online clustering (chosen).**
- B — pyannote community-1 (CC-BY-4.0, gated) over a rolling window, mapped to session speakers: handles a speaker change inside a segment, but is built for whole files (streaming not mentioned), heavier and slower.
- C — diart streaming diarization: last release Feb 2025; recommends pyannote.audio < 3.1; Python versions listed only up to 3.10.

**Why:** Simple and transparent; one label per printed segment; labels last the whole session by construction.

**Known weaknesses:** a segment with two people gets one label; the threshold must be tuned on real Zoom audio; the model was trained on VoxCeleb (celebrity interviews, mostly English), so its behaviour on Armenian voices over Zoom is unknown (Claude's caveat, not from a source).

**To settle in S05:** the threshold; the minimum segment length before a new speaker may be created; whether short segments update centroids.

**Revisit if:** E3 shows many mixed-speaker segments → consider B.

**Sources:** [SpeechBrain ECAPA](https://huggingface.co/speechbrain/spkrec-ecapa-voxceleb/blob/5a40f2c96080d36cc8d73d0560ec5eabf96d6209/README.md) · [pyannote community-1](https://huggingface.co/pyannote/speaker-diarization-community-1) · [community-1 announcement](https://pyannote.ai/blog/community-1) · [diart](https://pypi.org/project/diart)

---

## D8 · Pipeline structure

**Decided 2026-10-27 · Claude recommended A · Tatev chose A**

**Decision:** One process. One capture callback per device → one segmenter thread per source → one queue → one inference worker that runs ASR → speaker → translation and prints.

**Options**
- **A — threads + queues, one inference worker (chosen).**
- B — one inference worker per source: parallel, but two copies of the models in 24 GB shared with Zoom, and GPU contention makes timing unpredictable.
- C — a process per stage: more moving parts, no clear gain at this size.

**Why:** The models load once and don't compete; it is the simplest structure to reason about and debug.

**Consequences:** when both sides talk at once, one segment waits, and that wait counts in the measured latency (T15). An exception in any thread stops the whole program (F04).

---

## D9 · Evaluation data and metrics

**Decided 2026-10-27 · Claude recommended A · Tatev chose A**

**Decision:** three evaluation sets and fixed metrics; details in TESTS.md §3.
- E1 read set — ~25 Armenian sentences Tatev records through M1, with her own Armenian and English references.
- E2 FLEURS subset — ~30 `hy_am` clips (CC-BY-4.0); checks our scoring against ArmBench's FLEURS numbers.
- E3 call set — a 5–10 min Zoom call with 2–3 Armenian speakers, recorded with their consent; ~3 min transcribed with speaker turns.
- Metrics: CER/WER after a fixed Armenian normalisation; Tatev's 1–3 rating + chrF; latency max/p95/median; speaker consistency, purity and labels created.

**Options**
- **A — E1 + E2 + E3 (chosen).**
- B — E1 + E3 only: saves ~30 min, but nothing checks our scoring.
- C — live tests only: no numbers.

**Why:** E3 is the only realistic test of dialogue and speakers; E2 validates the scoring; E1 is cheap and controlled.

**Rule:** recordings live in `data/` (git-ignored); only reference texts are committed (DL4).

**Sources:** [FLEURS](https://www.huggingface.co/datasets/google/fleurs/tree/main) · [ArmBench-ASR](https://metric-ai-armbench-asr.static.hf.space/index.html)

---

## D10 · Licence policy

**Decided 2026-10-26, confirmed 2026-10-27 · Tatev**

**Decision:** Only permissive licences: MIT, Apache-2.0, BSD, CC-BY-4.0, all credited in the README. Not allowed: non-commercial licences (e.g. CC-BY-NC), custom licences with use restrictions, unknown licences.

**Scope:** models and libraries used by the project. BlackHole (GPL-3.0) is a driver Tatev installs, not part of our code.

**Rule:** check the licence before adding anything, and record it in the README credits.

---

## D11 · Workflow between Claude and the Mac

**Decided 2026-10-27 · Tatev chose C**

**Decision:** Claude writes files in its cloud workspace and sends them; Tatev saves them into the repository, commits, runs commands on the Mac and pastes the output back.

**Options**
- A — link the chat to the Mac so Claude runs approved commands there directly.
- B — Claude commits to the GitHub repository from the cloud session.
- **C — files are sent; Tatev commits and runs commands (chosen).**

**Consequence:** Claude can't run or test anything on the Mac. Every command comes with what it does and the output to expect. If the Mac is linked later, update this entry.

---

## D12 · Proof of "local only"

**Decided 2026-10-26 · Tatev**

**Decision:** Once `scripts/download_models.py` has fetched the models, the program must run with networking off. The program sets `HF_HUB_OFFLINE=1`, so nothing is downloaded silently at runtime; a missing model is an error (F03).

**Why:** a direct, demonstrable test of C2 (T17).

---

## D13 · Language scope

**Decided 2026-10-26 · Tatev**

**Decision:** Eastern Armenian only. Russian or English words mixed into Armenian speech are a known limitation, not handled.

**Why:** confirmed by Tatev; the brief puts other input languages out of scope.
