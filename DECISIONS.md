# DECISIONS

Decision log for the live Armenian → English transcription project. Each entry gives the decision, the options considered, Claude's recommendation, Tatev's choice, the reasons, the evidence and what would make us revisit it. Claude researched the options; Tatev decided.

Status: **decided** · **decided — confirm by measurement** (the default stands unless the S04/S05 numbers say otherwise) · **superseded**

| ID | Topic | Choice | Status |
|---|---|---|---|
| D1 | Python environment | installed Python 3.14.5 + venv | decided |
| D2 | System-audio capture | BlackHole 2ch + Multi-Output Device | decided |
| D3 | Listening and microphone | AirPods for listening, MacBook mic as input | decided |
| D4 | Phrase segmentation | Silero VAD | decided — licence checked in S03 (MIT) |
| D5 | Armenian ASR | NVIDIA hy FastConformer (ONNX, onnx-asr) | decided — confirm by measurement |
| D6 | Translation | MADLAD-400-3B-MT (CTranslate2 int8) | decided — confirm by measurement |
| D7 | Speaker labels | ECAPA embeddings + online clustering | decided |
| D8 | Pipeline structure | one process, threads + queues, one inference worker | decided |
| D9 | Evaluation data and metrics | E1 read + E2 FLEURS + E3 call | decided |
| D10 | Licence policy | permissive only: MIT, MIT-0, Apache-2.0, BSD, 0BSD, Zlib, CC0-1.0, CC-BY-4.0 | decided — revised 2026-10-30 |
| D11 | Workflow | Claude sends files; Tatev commits and runs commands | decided |
| D12 | Proof of "local only" | runs with networking off | decided |
| D13 | Language scope | Eastern Armenian; code-switching is a known limitation | decided |
| D14 | Resampling to 16 kHz | Core Audio converts inside PortAudio | decided |
| D15 | Chunk timestamps | counted from samples | decided |
| D16 | Running the package | `pip install -e .` with flit_core | decided |
| D17 | Capture mechanism | a reader thread per source; PortAudio buffers in C | decided |
| D18 | How T04 checks for lost audio | each source's clock gap within ± 0.1 s, and 0 overflows | decided |
| D19 | Running Silero VAD | the ONNX file directly with onnxruntime; no torch | decided |
| D20 | Downloading the models | standard-library script; pinned versions and SHA-256; no huggingface_hub | decided |
| D21 | Call-side label before M4 | fixed "Speaker 1" placeholder until S05 | decided — temporary |

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

**Settled in S02 (2026-10-30):** the Multi-Output Device contains BlackHole 2ch and the AirPods; its primary (clock) device is BlackHole 2ch at 48 kHz, with drift correction on the AirPods only. The MacBook speakers and the external monitor are not in the group (D3). macOS output is the Multi-Output Device and macOS input the MacBook Pro Microphone. Result: the call audio was heard clearly on the AirPods (T05 passed); the T04 recordings had no crackle or dropouts and no overflows on BlackHole. The switch to B was not needed. BlackHole 2ch runs at 48 kHz on macOS 26.5.2.

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

**Licence settled in S03 (2026-10-31):** the LICENSE file is MIT ("Copyright (c) 2020-present Silero Team"), as published in a mirror that states it is a verbatim copy of the v6.2.1 LICENSE; the citation metadata and the package text also say MIT. The CC BY-NC badge contradicts the LICENSE file and is treated as outdated. The download script saves the upstream LICENSE next to the model, so the claim is checked on the Mac. How Silero is run: D19.

**Sources:** [silero-vad on PyPI](https://pypi.org/project/silero-vad) · [repository metadata](https://science.ecosyste.ms/projects/32425) · [LICENSE copy (onnx-community mirror)](https://huggingface.co/onnx-community/silero-vad/blob/main/LICENSE) · [v6.2.1 ONNX mirror with SHA-256](https://huggingface.co/bitsydarel/silero-vad-onnx)

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

**S03 observations (2026-10-31, before the S04 measurement):**
- E1 offline (tests/test_asr_mt.py): 16 of 25 recordings recognised word for word; 2 more differ only in punctuation or «և»/«եւ». Errors: «կրկնում» for «կրկնեք», a missing «է», «Անն է» for «Աննա», «խոսում եմ եք» for «խոսո՞ւմ եք», «ունի» for «ունեմ».
- Speed: 0.01–0.02 s per 2–4 s phrase; 0.14–0.15 s for a 12-s segment in the live runs. Well inside the 1.0-s budget.
- On 3 s of digital zeros the model alone returned «Ինչ էին։»: the CTC decoder does invent text on silence, contrary to the expectation above (K10). In the program the VAD keeps silence away from ASR (T20 passed); recorded as a known limitation (xfail test).
- At a forced 12-s cut the word split by the cut is misrecognised (T16) — a segmentation issue for S06, not a model choice.

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

**S03 observations (2026-10-31, before the S04 measurement), settings: int8, 8 threads, beam 1:**
- Speed: 0.22–0.79 s for short phrases; 1.1–2.55 s for 12-s segments, which is over the 1.5-s planning figure but kept every line within 5 s (max 2.88 s).
- Added content: «Իմ անունն է Աննա։» → "My name is Anna. **I am 20 years old.**"; repeated sentences ("Today the weather is very hot. It is very hot today."; «Մեկ։» → "One. One. One.").
- Names replaced, systematically: «Հայաստանի Հանրապետության» → "Republic of **Georgia**", «Հայաստանում» → "in **Georgia**" and "in **Kazakhstan**", «հայկական» → "**Bulgarian**", «հայոց այբուբենը» → "the **Persian** alphabet", «Մեսրոպ Մաշտոց» → "**Mehmed** Mashtots". The ASR text had the names right.
- Consequence: the A-vs-B comparison in S04 is the most important one; named entities should be part of the rating. Settings to try on A before comparing: a larger beam and repetition limits (CTranslate2's translate_batch options, to verify in S04).

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

**Decided 2026-10-27 · Claude recommended A · Tatev chose A · revised 2026-10-30 by D17**

**Decision:** One process. One capture reader thread per device → one segmenter thread per source → one queue → one inference worker that runs ASR → speaker → translation and prints. (Until D17 this read "one capture callback per device".)

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

**Decided 2026-10-26, confirmed 2026-10-27 · Tatev · revised 2026-10-30: Claude recommended A · Tatev chose A**

**Decision:** Only permissive licences: MIT and MIT-0, Apache-2.0, BSD (including 0BSD), Zlib, CC0-1.0 and CC-BY-4.0, all credited in the README. Not allowed: non-commercial licences (e.g. CC-BY-NC), custom licences with use restrictions, GPL, LGPL and AGPL, and unknown licences. Any other licence is raised with Tatev before anything is installed.

**Scope:** models and libraries used by the project, including their transitive dependencies. BlackHole (GPL-3.0) is a driver Tatev installs, not part of our code.

**Rule:** check the licence before adding anything, and record it in the README credits.

**Revision 2026-10-30 (new evidence from the S02 licence check):** the original wording allowed "MIT, Apache-2.0, BSD, CC-BY-4.0" and nothing else. The core libraries use permissive licences that were not on that list: NumPy declares BSD-3-Clause AND 0BSD AND MIT AND Zlib AND CC0-1.0, and cffi 2.x, which sounddevice uses to call PortAudio, is MIT No Attribution (MIT-0). Read literally, the old wording ruled out NumPy, which every planned model runtime needs. sounddevice (MIT), the PortAudio library it bundles (MIT) and pycparser (BSD-3-Clause) already qualified.
- **A — revise the allowed list as above (chosen).**
- B — keep the old wording: NumPy and cffi don't qualify, and the project stops at S02 step 6.

**Sources:** [NumPy licence metadata](https://anaconda.org/anaconda/numpy/files/modal/info/6967990c7f81d6ec453c5b84) · [cffi 2.0.0 LICENSE](https://sources.debian.org/src/python-cffi/2.0.0-3/LICENSE) · [sounddevice on PyPI](https://pypi.python.org/pypi/sounddevice) · [PortAudio licence](https://portaudio.com/license.html) · [pycparser](https://packages.msys2.org/base/python-pycparser)

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

---

## D14 · Getting the audio to 16 kHz

**Decided 2026-10-30 · Claude recommended A · Tatev chose A**

**Decision:** M1 opens both input streams at 16 kHz and lets Core Audio convert from each device's own rate inside PortAudio. The settings are written out in the code — `sd.CoreAudioSettings(change_device_parameters=False, fail_if_conversion_required=False, conversion_quality="max")` — so PortAudio never changes a device's own settings and BlackHole stays at the Multi-Output Device's rate.

**Options**
- **A — Core Audio converts inside PortAudio (chosen):** no resampling code and no new dependency; the converter's quality and added latency are not visible to us (unverified).
- B — capture at the native 48 kHz, low-pass filter and keep every third sample in numpy: fully visible and unit-testable; fails loudly unless the device runs at exactly 48 kHz; our own DSP code to get right.
- C — `scipy.signal.resample_poly` on each chunk: one call, but each chunk is filtered on its own, which leaves seams at chunk edges; adds SciPy for one function.

**Why:** the simplest correct option, and the devices' own settings stay untouched. sounddevice's documentation warns that letting PortAudio change device parameters can disrupt a device other programs are using.

**Checked by:** T04 — listening, the overflow count, and the recorder's comparison of sample-count time with the computer's clock, which would expose a wrong rate — and E1's CER later.

**Revisit if:** T04 shows artefacts or a wrong rate → switch to B.

**Sources:** [sounddevice: Core Audio settings](https://python-sounddevice.readthedocs.io/api/platform-specific-settings.html)

---

## D15 · Timestamps of audio chunks

**Decided 2026-10-30 · Claude recommended A · Tatev chose A**

**Decision:** Each chunk's `t_start` is counted from samples: the source's anchor plus the samples delivered so far ÷ 16 000. The anchor is the session-clock time (`time.monotonic()`) at the source's first callback minus the length of that first chunk. Each source has its own anchor.

**Options**
- **A — count samples (chosen):** the chunks of a source line up exactly, which M2 needs, and late callbacks don't move them. The count drifts with the device's clock (measured by the recorder); lost samples (overflows) would shift later times and are reported on stderr.
- B — `time.monotonic()` in every callback: no drift, but late callbacks move the times (jitter), chunks don't line up, and M2 would need smoothing.
- C — PortAudio's ADC time per callback: closest to the hardware, but each stream stamps on its own time base, so two time bases must be converted to the session clock; unverified on Core Audio.

**Why:** the simplest input for M2, and it checks itself. The timing errors involved are far below the printed one-second resolution and the 5-second limit.

**Checked by:** the recorder's summary prints, per source, the gap between sample-count time and the computer's clock (T04, E3).

**Revisit if:** T04 or E3 shows a gap large enough to distort the latency figures (T14).

**Correction 2026-10-30 (implementation, the decision stands):** with D17's reader threads the anchor rule above was wrong. The first read can find much more than one block in PortAudio's buffer, most likely because starting a Core Audio stream returns only after audio is already flowing (unverified). Anchoring at "first read minus one block" then put every timestamp of that source about 0.2 s late, which would have made the measured latency look about 0.2 s better than it was. Evidence: the first 120-s run with D17 showed equal clock gaps of +0.222 s (mic) and +0.216 s (system) with 0 overflows, and the simulator reproduced the pattern when its start returned 0.2 s after audio began. The anchor is now the session time of the first read minus everything buffered at that moment (`read_available`), and the clock gap is measured at the newest buffered sample. After the correction, a 120-s run gave clock gaps of +0.002 s (mic) and +0.009 s (system), and both streams' first audio within 5 ms of each other. What remains unmeasured is Core Audio's own input latency, the delay from the microphone into PortAudio's buffer, estimated at tens of milliseconds (unverified).

**Sources:** [sounddevice: streams and callback time information](https://python-sounddevice.readthedocs.io/en/latest/api/streams.html)

---

## D16 · Running the package

**Decided 2026-10-30 · Claude recommended A · Tatev chose A**

**Decision:** `pip install -e .` with a minimal `pyproject.toml` whose build backend is flit_core. The package declares no dependencies; `requirements.txt` stays the single pinned list (D1).

**Options**
- **A — editable install with flit_core (chosen):** a one-time step; `python -m hyen` and the `eval/` scripts then work from any folder. flit_core is BSD-3-Clause, has no dependencies besides Python, finds a `src/` folder without configuration, and is used only at install time.
- B — `PYTHONPATH=src`: no extra file, but needed in every new Terminal window; forgetting it fails with `ModuleNotFoundError`.

**Why:** a src layout has to be installed before its code can run (Python packaging guide). setuptools was not used because it vendors several packages, one of which (autocommand) Gentoo lists as LGPL-3.

**Note:** not tested in Claude's workspace, which can't reach PyPI; the first `pip install -e .` on the Mac is the test.

**Sources:** [Packaging guide: src layout vs flat layout](https://packaging.python.org/en/latest/discussions/src-layout-vs-flat-layout/) · [Flit documentation](https://flit.pypa.io/) · [Flit release history (mirror)](https://daobook.github.io/flit/_sources/history.rst.txt) · [flit_core licence](https://aur.archlinux.org/packages/pypy3-flit-core) · [setuptools' vendored licences (Gentoo)](https://gpo.zugaina.org/AJAX/Ebuild/56078437/View)

---

## D17 · How M1 takes audio from PortAudio

**Decided 2026-10-30 · Claude recommended A, then B after Tatev asked for the comparison · Tatev chose B**

**Decision:** Each source has its own reader thread. The streams run without a Python callback, so PortAudio buffers the incoming audio in C and Python never runs on Core Audio's audio thread. The reader takes 512 samples (32 ms) at a time, stamps them (D15) and puts an `AudioChunk` on its queue; it sleeps until a full block is there instead of waiting inside PortAudio, so stopping is always clean. The streams' latency setting is 0.5 s, from which PortAudio sizes its buffer (exactly how on macOS is unverified). If a reader falls further behind than the buffer holds, PortAudio reports an overflow, which M1 prints and the recorder treats as a failed take.

**Evidence (S02)**
- T04 recordings with both streams: the mic lost 0.13%, 0.37% and 0.17% of its samples without an overflow flag (clock gap −0.080 s after 60 s, −0.662 s after 180 s, −0.209 s after 120 s); BlackHole lost none.
- `eval/clock_check.py`, one stream at a time: no loss on either device at 48 kHz or 16 kHz, so the conversion itself is exact (D14 stands). At 16 kHz PortAudio delivered the mic in 15-sample blocks — about 1,070 Python callbacks per second — and BlackHole in 1,600-sample blocks.
- Working explanation: the callbacks compete with other Python threads for the interpreter lock; when Core Audio's audio thread has to wait, macOS drops input without an overflow flag.

**Options**
- A — keep the callback with fixed 512-sample blocks: about 31 callbacks per second and two lines of code, but Python still runs on the audio thread, so drops under load stay possible and silent.
- **B — a reader thread per source, PortAudio buffering in C (chosen):** removes the cause; any loss beyond the buffer is flagged as an overflow; a bigger change to M1's internals; revises D8.
- C — capture at 48 kHz and convert in our code (D14-B): avoids the 15-sample blocks, but adds conversion code and keeps Python on the audio thread.

**Why:** B removes the cause instead of reducing how often it can strike, and makes any remaining loss loud. S03 adds model inference in other threads — the kind of load that triggers the drops. Claude first recommended A as the smaller change and switched to B when Tatev asked for the comparison.

**Consequences:** M1's outputs are unchanged except that every chunk is exactly 512 samples (the agreed contract said "whatever PortAudio delivers"). D8 now reads "one capture reader thread per device".

**Checked by:** repeating the 120-s recording and T04 — 0 overflows, the mic's clock gap near 0, length difference under 0.1 s. In Claude's workspace, a simulated device with a Python thread hogging the interpreter lock for 30 s lost nothing.

**Revisit if:** overflows appear under S03's load → raise the latency setting; or the mic's clock gap still grows → the working explanation is wrong; investigate again.

**First result (2026-10-30, 120 s with both streams):** 0 overflows on both; the clock gaps were equal on both sources (+0.222 s and +0.216 s), so the mic no longer lost audio relative to BlackHole. The common offset was a timestamp error, corrected in D15. The length difference (0.192 s) was not lost audio, since the clock gaps were equal, but timing at the start and stop of the two streams; after the D15 correction both streams' first audio was within 5 ms. How T04 checks for lost audio was then revised (D18).

**Sources:** [sounddevice: streams — blocking read, read_available, latency](https://python-sounddevice.readthedocs.io/en/latest/api/streams.html)

---

## D18 · How T04 checks for lost audio

**Decided 2026-10-31 · Claude recommended B · Tatev chose B**

**Decision:** T04's "equal length ± 0.1 s" is replaced by "each source's clock gap within ± 0.1 s, and 0 overflows". The rest of T04 stays: the right speech in each file, no crackle or dropouts. The recorder checks both conditions and exits non-zero when either fails; with `--seconds N` it cuts both files at exactly N seconds of session time, so they share one timeline (useful for E3).

**Evidence (S02)**
- After D17 and the D15 correction the capture is right: in 120 s both streams' first audio was within 5 ms, the clock gaps were +0.002 s (mic) and +0.009 s (system), and there were no overflows. The file lengths still differed by 0.133 s, because the recorder cut only the mic file at `--seconds`.
- File length can't measure lost audio reliably. If both files are cut at the same session time, they are equal by construction: in the simulator, a source silently losing 1% of its samples still gave 15.00 s against 15.00 s. If they aren't cut, they differ by up to about 0.1 s with nothing lost, because BlackHole delivers audio in blocks of 0.1 s (1,600 samples at 16 kHz in `eval/clock_check.py`).
- The clock gap compares each source's sample count with the computer's clock to within a few milliseconds. In the simulator it caught a 1% loss on either source (−0.15 s after 15 s).

**Options**
- **B — clock gap within ± 0.1 s and 0 overflows (chosen).**
- C — keep "equal length ± 0.1 s" and compare uncut files stopped at the same moment: keeps the wording, but BlackHole's 0.1-s blocks can make it fail with nothing lost, and smaller losses blur into that noise.
- Keeping the wording with both files cut at the same session time was ruled out: the lengths would then be equal by construction, so the check would test nothing.

**Why:** B measures directly what "equal length" stood in for, no lost audio and no drift, with the same tolerance and far better precision. Gaps below the threshold stay visible in the recorded numbers.

**Revisit if:** a dropout is heard in a recording whose clock gap passed; then the clock gap misses a kind of loss.

---

## D19 · How Silero VAD is run

**Decided 2026-10-31 · Claude recommended A · Tatev chose A**

**Decision:** M2 loads Silero VAD's ONNX file (`silero_vad.onnx`, release v6.2.1) with onnxruntime, which the project already needs for ASR, through a small wrapper of our own. At 16 kHz the model takes exactly 512 samples per call plus the last 64 samples of the previous call as context, a recurrent state of shape (2, 1, 128) and the sample rate; it returns a speech probability and the new state. Each source keeps its own state. M1's chunks are exactly 512 samples (D17), so no re-buffering is needed.

**Options**
- **A — the ONNX file with onnxruntime (chosen):** no new library; about 40 lines of our own code to unit-test.
- B — the `silero-vad` pip package: ready-made helpers, but it requires torch and torchaudio (several hundred MB, a second inference stack), and torch's dependency tree brings licences outside D10.

**Why:** the package's only extra value is helper code we can write in a few lines; the dependency cost is large.

**Evidence:** the package's required dependencies are packaging, torch and torchaudio (PyPI Stats; the PyPI page). Silero's own ONNX wrapper and two independent ports use 512 samples + 64 samples of context at 16 kHz with a (2, 1, 128) state. The file's path in the repository is `src/silero_vad/data/silero_vad.onnx` at tag v6.2.1; its SHA-256 comes from a mirror that states it matches the upstream file byte for byte (one source; the download script stops on a mismatch).

**Note:** D4 named the package version 6.2.3; the model file is pinned at v6.2.1 because that is the release with a published checksum.

**Sources:** [silero-vad 6.2.2 on PyPI (dependencies)](https://pypi.org/project/silero-vad/6.2.2/) · [PyPI Stats: required dependencies](https://pypistats.org/packages/silero-vad) · [Silero's ONNX wrapper (copy)](https://huggingface.co/spaces/nuernie/ai-server/blob/main/whisper_live/vad.py) · [LiveKit's Silero ONNX wrapper](https://lab.atomdrift.org/file/218e0e7becbd45d41e107d97dfe55945e0f249d5730531b3e4bbe3be86974cff) · [v6.2.1 ONNX mirror with SHA-256](https://huggingface.co/bitsydarel/silero-vad-onnx)

---

## D20 · How the models are downloaded

**Decided 2026-10-31 · Claude recommended A · Tatev chose A**

**Decision:** `scripts/download_models.py` uses only the Python standard library. Hugging Face files come from a full commit hash, the Silero file from a release tag; every model file is checked against a pinned SHA-256, and the script stops on a mismatch, on a network error (no retries) or when disk space is short. Files go to `models/<folder>/` (git-ignored) with a SOURCE.txt recording source, version, licence and every file's SHA-256; licence files and model cards are saved alongside. `--verify` checks the files without the network. The program loads models from these paths, so huggingface_hub is never installed; it still sets `HF_HUB_OFFLINE=1` (D12).

| Folder | Source @ version | Files | Size |
|---|---|---|---|
| `vad-silero` | snakers4/silero-vad @ v6.2.1 | silero_vad.onnx, LICENSE | ~2 MB |
| `asr-hy-fastconformer` | OpenVoiceOS/stt_hy_fastconformer_hybrid_large_pc_onnx @ f7db1fa | model.onnx (fp32), vocab.txt, config.json, README.md | ~458 MB |
| `mt-madlad400-3b-ct2-int8` | Nextcloud-AI/madlad400-3b-mt-ct2-int8 @ aa32bbd | model.bin, spiece.model, config.json, shared_vocabulary.json, README.md | ~2.96 GB |

**Options**
- **A — standard-library script (chosen).**
- B — huggingface_hub's `snapshot_download`: less code, but it brings packages such as tqdm (MPL-2.0 AND MIT) and filelock (Unlicense) that D10 doesn't allow without an exception.

**Why:** no new dependencies, every byte pinned and checked, and a missing model is a plain file error (F03).

**Known gaps:** an interrupted download restarts that file from zero; the Hugging Face redirect path was not tested in Claude's workspace (no access to huggingface.co), so the first run on the Mac is the test.

**Note:** the ASR repository also holds a `model.int8.onnx` (132 MB), although its card says there is no int8 variant. Not used (D5 is the fp32 model); a candidate if ASR latency becomes the bottleneck.

**Sources:** [ASR export: files and commit](https://huggingface.co/OpenVoiceOS/stt_hy_fastconformer_hybrid_large_pc_onnx/tree/main) · [ASR model.onnx SHA-256](https://huggingface.co/OpenVoiceOS/stt_hy_fastconformer_hybrid_large_pc_onnx/blob/main/model.onnx) · [MADLAD CT2 files and commits](https://huggingface.co/Nextcloud-AI/madlad400-3b-mt-ct2-int8/commits/main) · [model.bin SHA-256](https://huggingface.co/Nextcloud-AI/madlad400-3b-mt-ct2-int8/blob/main/model.bin) · [spiece.model SHA-256](https://huggingface.co/Nextcloud-AI/madlad400-3b-mt-ct2-int8/blob/main/spiece.model) · [onnx-asr dependencies](https://awesome.ecosyste.ms/projects/github.com%2Fistupakov%2Fonnx-asr)

**Licence check of the S03 libraries (2026-10-31, `pip install --dry-run --report` on the Mac, Python 3.14.5):** ctranslate2 4.8.4 MIT · flatbuffers 25.12.19 Apache-2.0 · numpy 2.5.4 BSD-3-Clause AND 0BSD AND MIT AND Zlib AND CC0-1.0 · onnx-asr 0.12.0 MIT · onnxruntime 1.31.1 MIT · packaging 26.3 Apache-2.0 OR BSD-2-Clause · protobuf 7.36.2 BSD-3-Clause · PyYAML 6.0.3 MIT · sentencepiece 0.2.2 Apache-2.0. All allowed by D10, all installed from wheels (cp314, abi3 or pure Python). ctranslate2 4.8.4 no longer requires setuptools (older versions did, which would have brought vendored LGPL code). A conda build labels onnxruntime "MIT AND BSL-1.0"; the PyPI wheel's metadata says MIT.

---

## D21 · Call-side label until M4 exists

**Decided 2026-10-31 · Claude recommended A · Tatev chose A · temporary until S05**

**Decision:** until M4 (S05), every system-audio segment is labelled "Speaker 1"; a startup line on stderr says that speaker labels are not implemented yet.

**Options**
- **A — a fixed "Speaker 1" placeholder (chosen):** the output format (T06, T12) can be tested now.
- B — "Speaker ?": honest, but T06 stays open until S05.

**Why:** the format is tested early; the stderr line keeps the placeholder from being mistaken for a working feature.

**Ends:** when M4 replaces it in S05.
