# DECISIONS

Decision log for the live Armenian → English transcription project. Each entry gives the decision, the options considered, Claude's recommendation, Tatev's choice, the reasons, the evidence and what would make us revisit it. Claude researched the options; Tatev decided.

Status: **decided** · **decided — confirm by measurement** (the default stands unless the S04/S05 numbers say otherwise) · **superseded**

| ID | Topic | Choice | Status |
|---|---|---|---|
| D1 | Python environment | installed Python 3.14.5 + venv | decided |
| D2 | System-audio capture | BlackHole 2ch + Multi-Output Device | decided |
| D3 | Listening and microphone | AirPods for listening, MacBook mic as input | decided |
| D4 | Phrase segmentation | Silero VAD | decided — licence checked in S03 (MIT) |
| D5 | Armenian ASR | NVIDIA hy FastConformer (ONNX, onnx-asr) | decided — B comparison cut 2026-11-01 (D22) |
| D6 | Translation | MADLAD-400-3B-MT (CTranslate2 int8) | decided — B comparison cut 2026-11-01 (D22) |
| D7 | Speaker labels | WeSpeaker ResNet34-LM embeddings (sherpa-onnx) + online clustering; threshold 0.77, new speaker from 7 s | decided — revised 2026-11-01 |
| D8 | Pipeline structure | one process, threads + queues, one inference worker | decided |
| D9 | Evaluation data and metrics | E1 read + E2 FLEURS + E3 call | decided — E3 reference revised 2026-11-01 |
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
| D21 | Call-side label before M4 | fixed "Speaker 1" placeholder until S05 | superseded by M4 (S05) |
| D22 | Model B comparisons | cut; A measured only, A's MT decoding tuned | decided |
| D23 | MT decoding and segment cap | beam 4, repetition penalty 1.2, no-repeat 3-gram; segments cut at 8 s | decided — confirmed by T14/T15 |
| D24 | Guards on short ASR text and MT output length | < 2 Armenian letters → no text; output ≤ 2 × input tokens + 16 | decided |
| D25 | Where a phrase is cut at the 8-s cap | after the quietest VAD chunk of the last ~2 s | decided — T16: helps only at pauses |
| D26 | Mic delivering only digital zeros | stop with an error after 5 s (mic only) | decided — confirmed by F01 |
| D27 | A source that stops delivering audio | stop with an error after 2 s without a block | decided — F02 passed; stall check unit-tested only |
| D28 | The ASR's «Հովհ» and lone «Հ» for "hm" | removed as whole words before the D24 guard | decided — extended 2026-11-01 |

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

**S04 measurement (2026-11-01), E1 (25 read phrases, MacBook mic), end to end:** CER 1.93% (12/622 chars), WER 5.41% (6/111 words), 20/25 identical after the TESTS.md §3.2 normalisation. Errors: «կրկնում» for «կրկնեք», a missing «է», «անն է» for «Աննա», «խոսում եմ եք» for «խոսում եք», «ունի» for «ունեմ». Read speech is the best case. B not measured (D22).

**S04 measurement (2026-11-01), E3 (Zoom call, system side, 0:13–2:01, 12 reference turns, 17 segments):** CER 30.39% (355/1168 chars), WER 50.46% (110/218 words). The speech is colloquial («խփելա», «կարացե», «պրոստո») and half the turns overlap (two people talking at once, 6 of 12 turns `A+B`). The reference was made by correcting the program's output (D9 revision), so these figures are somewhat too low. For comparison, ArmBench reports 60.78% WER for this model on its "Movies" dialogue set.

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

**S04 measurements (2026-11-01):** see D23 for decoding. On Tatev's reference Armenian (E1, MT only), chrF 80.6 at the S03 settings and 82.3 at the D23 settings. Names on long.tsv (5 sentences, 15–23 words; drafted by Claude, checked by Tatev): «Հայաստան» → "Georgia" in 2 of 2 sentences and «հայոց»/«հայերը» → "Persian"/"the Persians", **under every decoding setting tried** (beam 1, beam 2 and 4, with and without repetition limits). «Հայկական խոհանոց» → "Armenian cuisine" was right. Other names anglicised or mangled (Tolman, gulash, Gaghart, Lake Sevana).

**Names — decided 2026-11-01 · Claude recommended A · Tatev chose A:** a known limitation, documented in the README with these examples. Options: A — limitation (chosen; no time cost, S05 has priority); B — a glossary replacing a few Armenian name forms with English before MT (terminology injection; Armenian case endings make a hand list partial; whether MADLAD keeps a Latin word in Armenian input is unverified; ~1 h); C — reopen D22 and test Gemma 4 (2–3 h). Improvement for the README: B, or an LLM translator measured against MADLAD.

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

**Revision 2026-11-01 (S05) — the embedding model · Claude recommended A · Tatev chose A**

New evidence: SpeechBrain fails D10. Its required dependencies include tqdm (MPL-2.0 AND MIT, ruled out in D20), huggingface_hub (kept out on purpose, D20) and torch, whose tree brings licences outside D10 (D19). The method above stands; only the model and its runtime change.

- **A — sherpa-onnx (Apache-2.0) running WeSpeaker `voxceleb_resnet34_LM` as ONNX (chosen).** sherpa-onnx 2.0.1 has a cp314 macOS arm64 wheel and computes the model's filterbank features itself. The model is about 26 MB, 256-dimensional, with a VoxCeleb1-O EER of 0.72–0.80% on its card. WeSpeaker states that its VoxCeleb models follow the VoxCeleb licence, CC-BY-4.0.
- B — onnxruntime only, the same model, our own 80-band filterbank in numpy: no new dependency, but a small mismatch with the training features would degrade the embeddings without any error.
- C — like A, with NVIDIA TitaNet-small: its CC-BY-4.0 licence was seen only on a third-party page.

Runtime: `SherpaEmbedder` in `src/hyen/speakers.py`; model from the `speaker-recongition-models` release (sic), pinned by SHA-256 in `scripts/download_models.py` (D20). Measured on E3: 28 ms median and 70 ms max per segment, against the 0.3-s budget.

**Settled in S05 (2026-11-01) · Claude recommended the 7.0 s / 0.77 setting · Tatev chose it**

- **Rules:** mic segments are "Me" and are never embedded. A call-side segment matching a speaker's centroid at cosine ≥ **0.77** gets that label. A non-matching segment of **≥ 7.0 s** creates a new "Speaker N"; a shorter one gets the nearest label. Only segments ≥ 7.0 s create or update a centroid. Segments under 0.5 s are not embedded and take the previous call-side label, with a note on stderr. Until the first ≥ 7.0 s segment, the call side is "Speaker 1".
- **Bug found on the way:** the first version started Speaker 1's centroid from E3's first segment (0.74 s), against the agreed rule. Fixed, with a test.
- **Why not E3 alone:** in E3's lettered part (0:00–7:20), B speaks alone in only 4 segments (about 7 s), and 14 segments are A+B, so R7 can't be measured on it. E4 was recorded instead (TESTS.md §3.1): a Zoom call with one other person. `eval/cross_speakers.py` interleaves E3's speaker A (32 segments, 106 s) with E4's speaker C (52 segments, 110 s) as a simulated 3.6-min call.
- **The embeddings separate the two people:** a segment's nearest other segment is the same person 98% of the time, and 93% are closer to their own person's mean. Cosines: A–A median 0.77, C–C 0.77, A–C 0.68, with overlapping ranges (0.45–0.93 within, 0.46–0.83 across).
- **The creation rule was the problem:** with a 1.5-s minimum no threshold gave 2 labels (merged up to 0.70, fragmented from 0.71).

| New speaker from | Working thresholds (2 labels, T10 + T11 pass) | Consistency / purity there |
|---|---|---|
| 1.5 s | none | — |
| 5.0 s | 0.75–0.77 | 0.87 / 0.87 |
| 6.0 s | 0.75–0.77 | 0.86 / 0.86 |
| **7.0 s** | **0.74–0.80** | **0.79 / 0.79** |

Centring the embeddings on the recording's mean (offline experiment) widened the gap between people but not the working range; dropped.

**Why:** the widest working range. Live audio can shift the similarities, and the value was tuned on the data it is scored on.

**Known weaknesses:**
- A new person gets their own label only after speaking 7 s at a stretch (segments are cut at 8 s). Until then their segments carry the nearest label. Mitigation for the interview: each caller opens with a longer sentence (P02).
- One label per segment; mixed segments get one speaker's label.
- Numbers are a best case: the two people came from different sessions, and the tuning and scoring data are the same.
- Model trained on VoxCeleb (mostly English); Armenian voices over Zoom are covered only by E3/E4.

**Revisit if:** the rehearsal (P01) merges or splits the speakers → tune again on the P01 recording with `eval/score_speakers.py`.

**Licence check (2026-11-01, on the Mac, Python 3.14.5):** installing sherpa-onnx added exactly two packages: sherpa_onnx 2.0.1 ("Apache licensed") and sherpa-onnx-core 2.0.1 (Apache-2.0), which requires nothing else. Both are allowed by D10. `download_models.py --verify` confirmed the model's pinned SHA-256 (the value had come from one source).

**Sources:** [speechbrain dependencies (PyPI Stats)](https://pypistats.org/packages/speechbrain) · [sherpa-onnx on PyPI](https://pypi.org/project/sherpa-onnx/) · [WeSpeaker ResNet34-LM card](https://huggingface.co/Wespeaker/wespeaker-voxceleb-resnet34-LM/blob/80304c92fe55443239c3afa400b19551b75a3eab/README.md?code=true) · [licence statement (pyannote wrapper card)](https://huggingface.co/pyannote/wespeaker-voxceleb-resnet34-LM/blame/main/README.md) · [sherpa-onnx speaker model list](https://huggingface.co/spaces/k2-fsa/speaker-diarization/blob/1919bdd566f2cbd82222e824c63f9e13611ad7d7/model.py) · [model SHA-256 (omni-dev manifest)](https://docs.rs/omni-dev/0.29.0/src/omni_dev/voice/models.rs.html)

**Original sources:** [SpeechBrain ECAPA](https://huggingface.co/speechbrain/spkrec-ecapa-voxceleb/blob/5a40f2c96080d36cc8d73d0560ec5eabf96d6209/README.md) · [pyannote community-1](https://huggingface.co/pyannote/speaker-diarization-community-1) · [community-1 announcement](https://pyannote.ai/blog/community-1) · [diart](https://pypi.org/project/diart)

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

**Revision 2026-11-01 (S04) · Claude recommended A · Tatev chose A — how the E3 reference is made and where it lives:**
- **Made by correcting the program's output**, not transcribed from scratch: `eval/segment_offline.py` runs the program's own segmenter, ASR and MT over `data/e3/system.wav`; Tatev corrects the Armenian of every segment in the first 3 minutes, gives each segment a speaker letter (A, B, C; `A+B` for a segment with two people) for the whole call, and writes her own English for ~15 segments. Option B (transcribing from scratch, as planned) gives an unbiased CER but takes ~45 min more, which the schedule doesn't have (D22). **Consequence:** errors Tatev doesn't notice stay in the reference, so E3's CER is somewhat too low; the README says so.
- **Kept in `data/e3/e3_ref.tsv` (git-ignored), not in `eval/refs/`:** the participants agreed to a recording that "stays only on my computer and won't be published"; the reviewers read the repository. Only the numbers are reported. E1 and long.tsv (Tatev's own sentences) stay committed.
- **Scope cut the same day (Tatev):** no English for E3 (MT is rated on E1 and long.tsv only, so conversational MT is not rated); Armenian corrected for 0:13–2:01 only; speaker letters for 0:00–7:20. The recording is 11:00; the rest is not used for scoring.

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

**Ended 2026-11-01:** M4 replaced it in S05 (D7).

---

## D22 · Cutting the model B comparisons

**Decided 2026-11-01 · Claude recommended A · Tatev chose A · revises the "confirm by measurement" part of D5 and D6**

**Decision:** BuzzASR (D5-B) and Gemma 4 12B (D6-B) are not measured. S04 measures only the chosen models on E1, E2 and E3. About 30 minutes go into MADLAD's decoding settings (beam size, repetition limits) to reduce the added content and the replaced names seen in S03. The name substitutions are reported in the README as a known limitation, with the examples from D6.

**Options**
- **A — cut both B comparisons (chosen).**
- B — cut BuzzASR, keep Gemma 4 for MT: targets the most visible failure, but needs a licence check of mlx-lm and its dependencies, a ~7 GB download and a few hours, taken from S05 or S06.
- C — keep both: does not fit in the time left.

**Why (new evidence: the schedule):** S04 starts at 00:50 on the deadline day (23:59 Yerevan time), with S04–S07, the E3 recording and the rehearsal still to do — about 12 h of planned work. Speaker labels (R7, R8) are Must requirements and are still the D21 placeholder; model comparisons are not a Must. PLAN.md §9 already names the B comparisons as the first item to cut.

**Consequence:** C5 ("models chosen deliberately") rests on the research in D5/D6 and on A's measured numbers, not on a head-to-head comparison. The README says so. K1 and K3 stay open risks.

**Revisit if:** the deadline moves, or A's MT rating on E1 + E3 is so low that the live result is unusable.

---

## D23 · MT decoding settings and the segment cap

**Decided 2026-11-01 · Claude recommended A · Tatev chose A · confirm by T14/T15**

**Decision:** MADLAD decodes with beam 4, repetition penalty 1.2 and no repeated 3-token sequence (CTranslate2 TranslationOptions `beam_size`, `repetition_penalty`, `no_repeat_ngram_size`). M2 cuts a phrase at 250 chunks (8.0 s) instead of 375 (12.0 s), so translation time per segment stays near what was measured.

**Evidence (S04, MT only on Tatev's reference Armenian, `eval/run_offline.py --mt-input ref`):**

| Setting | E1 chrF | E1 repeats / added content | E1 MT median / max | long.tsv MT median / max |
|---|---|---|---|---|
| beam 1 (S03) | 80.6 | e1_08 repeated; e1_13 "I am 20 years old" | — | 1.22 / 1.60 s |
| beam 1 + limits | 79.9 | invented: "It's 10 degrees Celsius today", "I am 20 years old and from Russia" | 0.32 / 0.66 s | 1.20 / 1.22 s |
| beam 2 + limits | — | none on e1_08, 10, 13, 16, 18, 20 (other 19 not reviewed) | 0.48 / 1.02 s | 1.75 / 1.97 s |
| beam 4 | 77.8 | e1_08, 10, 13, 16 doubled | 0.57 / 0.83 s | — |
| **beam 4 + limits** | **82.3** | **none in 25** | 0.54 / 0.99 s | 1.91 / 2.04 s* |
| beam 4 + limits, no-repeat 4 | — | — | — | 1.83 / 1.92 s* |

*p95 from the first version of the summary, which extrapolated past the maximum for n = 5; medians and maxima are right. "Limits" = repetition penalty 1.2, no-repeat 3.

None of the settings fixed the replaced names (D6).

**Options**
- **A — beam 4 + limits, cap 8 s (chosen):** the only setting clean on all of E1, with the best chrF.
- B — beam 2 + limits, cap 8 s: ~0.15 s faster per long sentence; 19 E1 lines not reviewed.
- C — beam 4 + limits, cap 12 s: no change to M2, but beam 1 already took 2.55 s on a 12-s segment in S03; scaled by long.tsv's beam-4/beam-1 ratio (1.91/1.22) that is ~4 s of MT, + 0.6 s end-of-phrase wait — over 4.6 s before any queue wait (estimate).

**Why:** invented content is worse than repetition, because it reads as fact; beam 1 + limits invented it. An 8-s segment holds about the 15–23 words of long.tsv (MT max ≈ 2 s): ≈ 0.6 s wait + 0.1 s ASR + 2 s MT + ≤ 1 s queue ≈ 3.7 s (estimate; T14/T15 measure it).

**Consequences:** more cuts in long monologues, so the S06 fix (cut at the quietest chunk before the cap, T16) matters more. tests/test_segmenter.py T16 case now expects 4 pieces for 30 s.

**Revisit if:** T14/T15 show MT pushing any segment over 5 s → beam 2 (B), or a shorter cap.

**Confirmed 2026-11-01 (S06):** in all S06 live runs the slowest MT was 2.20 s and the highest latency 3.08 s (a 7.1-s mic segment); T14, T15 and T18 pass.

---

## D24 · Guards: very short ASR text, MT output length

**Decided 2026-11-01 · Claude recommended A · Tatev chose A**

**Decision:**
- M3 treats recognised text with fewer than 2 Armenian letters as no text, with a note on stderr; the worker drops it like empty text. Two-letter replies («հա», «ու») stay.
- M5 limits each translation to at most 2 × the input tokens + 16 (never more than 256). If an output reaches the limit, stderr says it may be cut short.

**Evidence (E3, S04):** segment system-27 (1.2 s) was recognised as «Հ։». At the S03 settings MT produced "H: H: H: …" up to the 256-token limit in 8.16 s, which alone breaks C1. With D23 it took 0.88 s but invented a reply: "H.: I am not sure what you mean by that." With D23 on the whole of E3 (92 segments, 16 cut at 8 s) the slowest MT was 2.08 s and the slowest ASR 0.12 s.

**Options**
- **A — both guards (chosen).**
- B — only the short-text guard: fixes what was seen; a runaway stays possible.
- C — neither; both listed as limitations.

**Why:** the short-text guard removes an observed invented line; the length limit bounds every translation's time whatever the model does, at no cost when outputs are normal.

**Checked 2026-11-01:** over 122 translations (E1 and long.tsv at D23 settings, E3 at D23), the largest output/input token ratios were 3.25 (13 output tokens for 4 input, «Հ։», now dropped by the first guard) and 2.17 (13 for 6, «Հովհ.»). The limit for those inputs is 24 and 28 tokens, so no measured translation would have been cut. **Cost:** a one-letter real reply such as «է» is lost (rare).

---

## D25 · Where a phrase is cut at the segment cap

**Decided 2026-11-01 · Claude recommended A · Tatev chose A · confirm by T16**

**Decision:** when a phrase reaches 250 chunks (8 s, D23), M2 cuts after the chunk with the lowest Silero speech probability among the last 62 chunks (≈ 2 s); on a tie, the latest one. The chunks after the cut start the next phrase, so nothing is lost and there is no gap. Every segment stays ≤ 8 s, so the latency budget is unchanged.

**Evidence:** T16 (S03, 12-s cap): every cut fell mid-word and the ASR misread the word at each cut. With the 8-s cap (D23) there are more cuts.

**Options**
- **A — lowest VAD probability in the last ~2 s (chosen):** M2 already computes the probability of every chunk; no new computation.
- B — lowest RMS energy in the same window: independent of the VAD, but Zoom's noise suppression flattens the energy, so "quietest" may still be inside a word.
- C — keep the hard cut and list it as a limitation.

**Why a 2-s window:** a cut earlier than ~6 s would make segments of ≥ 7 s rarer, and only those create or update a speaker (D7).

**Consequence:** a cut segment is 6–8 s long. Tested in `tests/test_segmenter.py` (dip inside and before the window, ties, nothing lost).

**Revisit if:** T16 still shows a misread word at most cuts → the probability dips don't line up with word gaps on this audio.

**T16 result (2026-11-01):** 30 s of fluent counting gave one cut, 2 chunks before the cap, and it still split «տասնվեց». In fluent speech the probability doesn't dip in the last 2 s, so D25 helps only where there is a pause. One cut is too little to revisit; kept, and listed as a limitation.

---

## D26 · The microphone delivering only digital zeros

**Decided 2026-11-01 · Claude recommended A · Tatev chose A · confirm by F01**

**Decision:** M1 counts exact-zero samples in a row from the `mic` source. After 5 s of them it raises a CaptureError, which stops the program (F04), naming the likely causes: Terminal has no microphone permission, or the input volume is at zero. BlackHole is not checked: it delivers zeros whenever nothing plays.

**Options**
- **A — stop with an error (chosen):** a working microphone never delivers exact zeros (its noise floor is never exactly 0), so the error can't fire during a quiet room.
- B — warn on stderr and keep running: the session would go on transcribing nothing.
- C — rely on the P02 pre-flight checklist only.

**Verified 2026-11-01 by F01:** without microphone permission macOS delivers exact zeros; the program stopped 5 s after "Listening." with this message.

---

## D27 · A source that stops delivering audio

**Decided 2026-11-01 · Claude recommended A · Tatev chose A · confirm by F02 · a timeout allowed by this entry (CLAUDE.md)**

**Decision:** if a source's PortAudio buffer stays short of one block (512 samples) for 2 s while its stream is running, M1 raises a CaptureError naming the device, which stops the program. A reader that is merely late finds a full buffer when it wakes, so CPU load can't trigger it. The existing check stays: if PortAudio stops a stream by itself, `finished_callback` raises a CaptureError.

**Options**
- **A — stop after 2 s without a block (chosen).**
- B — test F02 first and decide afterwards.
- C — no watchdog: rely on `finished_callback` only; a silent stall stays possible.

**Known gap:** if the AirPods disconnect but BlackHole keeps running, BlackHole delivers zeros, which look the same as a quiet call; neither D26 nor D27 can tell them apart. F02 shows what actually happens; if the program carries on silently, that goes into the README as a limitation, together with the P02 pre-flight.

**F02 result (2026-11-01):** with the AirPods put away mid-replay, macOS kept the Multi-Output Device with BlackHole alone: the call audio still reached the program and was transcribed normally, but Tatev heard nothing. Not a dead source, so nothing for D27 to catch; the stall check stays covered by unit tests only. The listener losing the call goes into the README and P02.

---

## D28 · «Հովհ»: the ASR's spelling of "hm"

**Decided 2026-11-01 · Claude recommended A · Tatev chose A**

**Decision:** M3 removes «Հովհ» (any case, also «ովհ» with the first letter lost) where it stands alone as a word, with the punctuation after it, and notes this on stderr. The D24 guard runs afterwards, so a segment that was only "hm", or "hm" plus one letter, is dropped. «Հովհաննես» and other words containing these letters are untouched.

**Evidence (2026-11-01, E4 replayed live, T15 run):** the call-side person's "hm" (Tatev, listening) came out as «Հովհ է.», «ովհ է։», «Հովհ.» and «Հովհ գյուղ», four times in about 75 s. MT turned them into "It is Joseph.", "Whoa! It's awful.", "John 15:23-24; Jn." and "2013. ^ a b c d e f g h i j k l m n o p q r". «Հովհ.» is the abbreviation of Հովհաննես used for the Gospel of John in Bible citations, which explains the output. Nobody says «Հովհ» as a word in conversation.

**Options**
- **A — remove «Հովհ» only (chosen):** the one filler spelling seen in our data; ~20 min.
- B — A plus a hand list of other filler spellings (հմ, ըհը, մմ …): none seen in our data, unverified.
- C — limitation only.

**Why not a longer minimum segment:** in the live mic run, «Այո» and «Ոչ» were 0.45–0.51 s long; a minimum that drops "hm" would drop them too.

**Known gap:** other fillers or noises may come out as other words and still be translated into invented text (README limitation).

**Extension 2026-11-01 · Claude recommended A · Tatev chose A — a lone «Հ»:** in two later live runs (E4 replay ~3 min; counting run), the ASR wrote a lone «Հ» 8 times: 7 times alone («Հ։», already dropped by D24) and once before a word, «Հ գյուղ», which MT turned into "2013. (in Japanese) 村田, 久保." A lone «Հ» is never a word on its own, so it is removed like «Հովհ». Options: A — lone «Հ» only (chosen); B — any lone letter except real one-letter words (է, ա, և): broader, with less evidence (other lone letters seen: «Ն», «Տ», once each), risk of dropping real words; C — leave it as a limitation. Lone «Ն», «Տ» and similar fragments stay a known limitation.
