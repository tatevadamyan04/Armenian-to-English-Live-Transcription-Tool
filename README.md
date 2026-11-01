# Armenian-to-English live transcription

A Python terminal program for macOS that listens to a live Zoom call. It captures the MacBook microphone and the call audio at the same time, transcribes the spoken Armenian, translates each phrase into English and prints it with the speaker: "Me" for the microphone, "Speaker 1", "Speaker 2", … for the call side. Each line appears a few seconds after the phrase ends (measured median 1.3–1.6 s, maximum 3.48 s; limit 5 s). All models run locally; after the one-time model download the program runs with networking off.

```
[00:01:12] Speaker 1
  HY: Բարև, ինչպե՞ս ես։
  EN: Hi, how are you?
```

The time is when the phrase started, counted from the start of the session. stdout carries only these lines; diagnostics and per-segment timings go to stderr.

Built for the Mathrix AI coding challenge by Tatev Adamyan, with Claude (see [How it was built](#how-it-was-built)). Planning documents: [PLAN.md](PLAN.md) · [DECISIONS.md](DECISIONS.md) (every choice, its alternatives and evidence, as D1–D28) · [TESTS.md](TESTS.md) (test cases and results).

## Contents

1. [Requirements](#requirements)
2. [Setup](#setup)
3. [Running](#running)
4. [Pre-flight checklist](#pre-flight-checklist)
5. [Architecture](#architecture)
6. [Models and alternatives](#models-and-alternatives)
7. [Measured results](#measured-results)
8. [Known limitations](#known-limitations)
9. [Possible improvements](#possible-improvements)
10. [Tests](#tests)
11. [How it was built](#how-it-was-built)
12. [Licences and credits](#licences-and-credits)

## Requirements

- A Mac with Apple silicon. Developed and tested on a MacBook Pro, Apple M4 Pro, 24 GB, macOS 26.5.2.
- Python 3.14.5 (D1).
- [BlackHole 2ch](https://github.com/ExistentialAudio/BlackHole), a virtual audio device that passes the call audio to the program (D2).
- Headphones that aren't used as a microphone: AirPods 4 here (D3). Speakers would let the call leak into the microphone, and the other side's words would be labelled "Me"; echo handling is out of scope.
- About 3.5 GB of disk space for the models, plus 1 GB free (the download script checks).

## Setup

### 1. Audio routing (D2, D3)

The call audio has to reach two places: your headphones, so you hear the call, and BlackHole, so the program hears it. A Multi-Output Device sends it to both.

1. Install BlackHole 2ch, following [its README](https://github.com/ExistentialAudio/BlackHole) (installer or Homebrew).
2. Open **Audio MIDI Setup** (Applications → Utilities), click **+** → **Create Multi-Output Device**, and tick **BlackHole 2ch** and your headphones (AirPods 4).
3. Set the **primary (clock) device** to BlackHole 2ch at **48 kHz**. Turn **drift correction** on for the AirPods only. Don't add the MacBook speakers.
4. **System Settings → Sound:** Output = **Multi-Output Device**, Input = **MacBook Pro Microphone**.
5. **Zoom → Settings → Audio:** Speaker = **Multi-Output Device**, Microphone = **MacBook Pro Microphone**. Using the AirPods as microphone would switch them to low-quality call audio (D3).

The volume keys reportedly don't work with a Multi-Output Device; set the headphone volume before the call.

### 2. Microphone permission

Terminal needs microphone access. macOS asks on the first run; if access was refused: **System Settings → Privacy & Security → Microphone → Terminal on**, then quit Terminal with **Cmd+Q** and reopen it. Switching the toggle without quitting Terminal left the old state in place in our test (F01).

Without the permission macOS delivers exact digital silence; the program detects this after 5 s and stops with an error that says so (D26).

### 3. Python environment and models

```bash
git clone <repository URL>
cd Armenian-to-English-Live-Transcription-Tool
python3.14 -m venv .venv            # Python 3.14.5
source .venv/bin/activate
pip install -r requirements.txt     # exact pinned versions (D1)
pip install -e .                    # makes `python -m hyen` work from any folder (D16)
python scripts/download_models.py   # ~3.5 GB; the only step that uses the network
python scripts/download_models.py --verify   # checks every file against its pinned SHA-256, offline
```

The download script uses only the Python standard library. Each file comes from a fixed commit or release tag and is checked against a pinned SHA-256; on any error it stops and names the file, with no retries. It is safe to re-run: valid files are kept. Models go to `models/` (git-ignored), each folder with a `SOURCE.txt` recording source, version, licence and checksums (D20).

## Running

List the input devices:

```bash
python -m hyen --list-devices
```

Start a session with the microphone and BlackHole, by exact name or by index:

```bash
python -m hyen --mic "MacBook Pro Microphone" --system "BlackHole 2ch"
```

The program loads all models (a few seconds), prints `Listening. Ctrl+C stops.` on stderr and then prints each phrase as it is processed. **Ctrl+C** ends the session; a phrase still in progress is not printed.

Both flags are required and must name two different devices. An unknown name is an error that lists the available devices; the program never picks a default device or matches part of a name (T03, T23).

**stderr** shows one timing line per segment, e.g. (format; numbers illustrative):

```
timing system-12: speech 3.41 s · end-detect 0.62 · queue 0.00 · asr 0.03 · spk 0.03 (match 0.82) · mt 0.71 · latency 1.39 s
```

`speech` is the phrase length, `end-detect` the silence wait that ends a phrase, `queue` the wait behind the other source, then the time of each model, the speaker rule that chose the label with its cosine similarity, and the latency (print time − end of speech). A line over 5 s is flagged `** OVER 5 s (C1) **`. stderr also notes every dropped segment and every filler removed, with the reason.

## Pre-flight checklist

Before a call (TESTS.md P02):

- [ ] Headphones charged and connected; macOS output = Multi-Output Device. Check again after the headphones connect: macOS may switch the output to them directly, and the program then hears nothing from the call (seen in the rehearsal)
- [ ] Zoom speaker = Multi-Output Device; Zoom microphone = MacBook Pro Microphone
- [ ] Terminal has microphone permission
- [ ] `python -m hyen --list-devices` shows both devices
- [ ] `python scripts/download_models.py --verify` reports all files valid
- [ ] A 30-s check with both sources printing lines
- [ ] Each person on the call side opens with one uninterrupted sentence of at least 7 s, so they get their own label early (D7)

## Architecture

```
MacBook mic ───► M1 capture ──► M2 segmenter ──┐
                                               ├──► queue ──► inference worker ──────────────────► M6 output ──► stdout
BlackHole 2ch ─► M1 capture ──► M2 segmenter ──┘              M3 ASR → M4 speaker → M5 translate
```

One process (D8). Each source has a capture reader thread and a segmenter thread; one inference worker runs the models in order, so they are loaded once and never compete for CPU. The main thread only waits for an error.

| Module | File | What it does |
|---|---|---|
| M1 capture | `audio.py` | Opens each device at 16 kHz (Core Audio converts, D14); a reader thread takes 512-sample blocks from PortAudio's buffer (D17), mixes to mono and stamps each block from the sample count (D15) |
| M2 segmenter | `segmenter.py`, `vad.py` | Silero VAD gives each 32-ms chunk a speech probability (D19). A phrase starts at 0.5, continues while ≥ 0.35, ends after 608 ms below it; a phrase reaching 8 s is cut at the quietest chunk of its last 2 s (D23, D25); under 256 ms of speech is dropped |
| M3 ASR | `asr.py` | Armenian text from the NVIDIA FastConformer (D5); removes the model's spellings of "hm" (D28); text with fewer than 2 Armenian letters counts as nothing (D24) |
| M4 speakers | `speakers.py` | "Me" for the microphone; for the call side, a voice embedding compared with each known speaker (D7) |
| M5 translate | `translate.py` | English from MADLAD-400-3B-MT (D6, D23); output length bounded (D24) |
| M6 output | `output.py` | The three-line block, flushed at once |
| M7 pipeline, CLI | `pipeline.py`, `__main__.py` | Threads and queue; loads every model before the session clock starts; command-line flags |
| Downloads | `scripts/download_models.py`, `modelfiles.py` | Fetches and verifies the models; the program only checks they exist (D20) |

**Latency** is measured from the end of speech, as found by the VAD, to the moment the English line is printed. It includes the 608-ms silence wait, queueing, all three models and printing.

**Failing loudly** (CLAUDE.md): there are no silent fallbacks. An exception in any thread stops the program with a traceback and a non-zero exit code (F04). It also stops when a model file is missing (F03), when the microphone delivers only digital silence for 5 s (D26), when a stream delivers nothing for 2 s (D27), and when a stream stops by itself. Runtime is local only: the program sets `HF_HUB_OFFLINE=1`, huggingface_hub isn't installed, and nothing in `src/` uses the network (D12, T17).

## Models and alternatives

Each choice was researched with 2–3 options, recommended by Claude and decided by Tatev; full reasoning, sources and measurements are in [DECISIONS.md](DECISIONS.md). Licences were checked first; only permissive licences were allowed (D10), which excluded several strong models. The planned head-to-head comparisons with the runner-up ASR and MT models were cut for time on the last day (D22), so those choices rest on published benchmarks and our measurements of the chosen model only.

### Phrase segmentation: Silero VAD v6.2.1 (MIT) · D4, D19

- **Why:** accurate, tiny, trained on many languages; run as an ONNX file with onnxruntime, which the ASR already needs, so no PyTorch.
- **Alternatives:** WebRTC VAD (older, more easily triggered by noise; licence unverified); fixed 3-s windows (cut words, no "end of phrase" for the latency limit).
- **Measured:** 60 s of room silence printed nothing (T20); 60 s of typing, clicks and a notification printed nothing (T21).

### Speech recognition: NVIDIA `stt_hy_fastconformer_hybrid_large_pc` (CC-BY-4.0) · D5

- **Why:** the best open model on the ArmBench-ASR benchmark (19.86% combined WER), small (459 MB) and fast on CPU. Run from the OpenVoiceOS ONNX export with onnx-asr, without NeMo or PyTorch.
- **Alternatives:** BuzzASR/armenian (MIT, Whisper-large-v3 fine-tune; best self-reported numbers, but not independently benchmarked, ~3 GB, Whisper-style invented text possible); Chillarmo whisper-large-v3-turbo-armenian (Apache-2.0; 33.00% on ArmBench). **Excluded:** SeamlessM4T v2 (non-commercial licence); Whisper large-v3 (65.12% on ArmBench).
- **Measured:** read speech (E1) CER 1.93%, WER 5.41%; a real Zoom call (E3) CER 30.39%, WER 50.46%. 0.01–0.15 s per segment.

### Translation: MADLAD-400-3B-MT (Apache-2.0), CTranslate2 int8 · D6, D23

- **Why:** a dedicated translation model that lists Armenian, under a permissive licence, fast on CPU with CTranslate2 int8 (the Nextcloud conversion).
- **Decoding (D23):** beam 4, repetition penalty 1.2, no repeated 3-token sequence. Beam 1 repeated sentences or invented content ("I am 20 years old and from Russia"); this setting was the only one clean on all 25 E1 phrases.
- **Alternatives:** Gemma 4 12B instruct, 4-bit via mlx-lm (possibly more fluent, ~7 GB, may paraphrase or add content; not tested, D22); OPUS-MT hy-en (licence and quality unverified). **Excluded:** NLLB-200 (non-commercial); TranslateGemma (not evaluated on Armenian input, no licence statement found).
- **Measured:** chrF 82.3 on E1 (Tatev's Armenian references, her English translations). MT time median 0.54 s on short phrases, ~2 s on 15–23-word sentences; highest in the live runs 2.20 s.

### Speaker labels: WeSpeaker ResNet34-LM embeddings (CC-BY-4.0) via sherpa-onnx · D7

- **How:** each call-side segment gets a 256-number voice embedding, compared by cosine similarity with each known speaker's average. ≥ 0.77 → that speaker. Below that, a segment of ≥ 7 s creates a new "Speaker N"; a shorter one gets the nearest label. Only segments of ≥ 7 s create or update a speaker, so short noisy pieces don't. Labels are never renamed, so a label lasts the whole session.
- **Why:** simple, transparent, one label per printed line; 28 ms per segment.
- **Alternatives:** pyannote community-1 (handles speaker changes within a segment, but built for whole files and heavier); diart (Python ≤ 3.10 listed, last release Feb 2025). SpeechBrain ECAPA was the first choice but its dependencies fail the licence policy (D7 revision).
- **Measured:** on a simulated two-person call (3.6 min, two people from two recordings): 2 labels, consistency 0.79, purity 0.79, both people keep their label after 3 min (T10, T11). Thresholds from 0.74 to 0.80 all work. A best case: the numbers are on the data the values were tuned on.

## Measured results

All on the MacBook Pro M4 Pro. Details and procedures in [TESTS.md](TESTS.md).

| What | Data | Result |
|---|---|---|
| ASR | E1: 25 read everyday phrases, MacBook mic | CER 1.93%, WER 5.41%; 20/25 exact |
| ASR | E3: Zoom call, colloquial, half the turns overlapping | CER 30.39%, WER 50.46%¹ |
| Translation | E1, from the reference Armenian | chrF 82.3 |
| Speakers | simulated 2-person call (E3 + E4), offline | 2 labels, consistency 0.79, purity 0.79 |
| Speakers, live | rehearsal: real Zoom call, 2 callers on separate devices, 2:50 | 1 label for both callers: no call-side turn reached 7 s (longest 5.9 s) |
| Speakers, live replay | 5:51 through BlackHole: Tatev's voice, then two E3 speakers, then Tatev's voice again | the E3 part (long turns) split into Speaker 1 and 2, matches 0.80–0.92; Tatev's voice (short turns only) was Speaker 1 at the start and Speaker 2 at the end, by nearest label |

¹ The E3 reference was made by correcting the program's output (D9), so errors nobody noticed stay in it; the true error rate is somewhat higher.

**Latency** (call audio replayed through BlackHole while Tatev spoke into the microphone):

| Run | Segments | Median | Max |
|---|---|---|---|
| ~75 s, both sources (T15) | 19 | 1.37 s | 2.49 s |
| ~3 min, both sources (F05) | 28 | 1.39 s | 2.46 s |
| 30 s of continuous counting (T16) | 15 | 1.58 s | 3.08 s |
| on battery, Wi-Fi off (T17, T18) | 17 | 1.50 s | 2.39 s |
| rehearsal: real Zoom call, 2:50 (P01) | 41 | 1.32 s | 1.99 s |
| rehearsal replay, 5:51 (P01) | 46 | 1.44 s | 3.48 s |

Every segment in every run was printed within 5 s. Models load in about 2–3 s.

## Known limitations

**Recognition and translation**

- **Conversational speech is hard:** about half of the words in a real Zoom call are wrong (E3), against 5% for read speech. ArmBench reports 60.78% WER for the same model on film dialogue.
- **Names are replaced in translation:** «Հայաստան» (Armenia) became "Georgia" and «հայոց» (Armenian) became "Persian" in every decoding setting tried; other names are anglicised or mangled (D6).
- **Invented text from fragments:** the translator can turn a single word or a fragment into unrelated text, e.g. «գյուղ։» ("village") → "2013. pp. 49. The Village.". The ASR's spellings of "hm" (which had become Bible references) are removed (D28), but other lone-letter fragments («Ն …», «Տ …») still reach the translator.
- **Long phrases are cut at 8 s:** the cut moves to the quietest moment of the last 2 s (D25), but in fluent speech there may be no pause, and the word at the cut is then split and misrecognised (T16).
- **Eastern Armenian only;** Russian or English words in Armenian speech aren't handled (D13).
- **Silence alone isn't safe for the ASR:** on pure digital silence the model invents a phrase; the VAD keeps silence away from it (T20).

**Speakers**

- **A person who never speaks 7 s at a stretch never gets their own label** (D7). Until then their lines carry the nearest existing label, which can change during the call. This was the main weakness in the rehearsal: in a real call the two callers spoke in short turns and shared "Speaker 1"; in the replay, a short-turn voice was Speaker 1 at the start and Speaker 2 at the end. Speakers with longer turns were told apart (similarity 0.80–0.92 to their own label). Hence the pre-flight request that everyone opens with a long sentence. A shorter minimum was not adopted: in tuning, 1.5 s merged or split speakers at every threshold (D7).
- One label per printed line: a segment with two people talking gets one of them.
- The tuning numbers are a best case (two recordings, tuned on the scored data).

**Audio setup**

- If the headphones disconnect, macOS keeps the Multi-Output Device with BlackHole only: transcription continues normally but you no longer hear the call. The program can't see this (F02).
- When both sides talk at once, one segment waits for the other (one inference worker, D8); this wait is included in the measured latency.
- No echo handling: headphones are required (D3). In the rehearsal, a replay that went to the MacBook speakers by mistake was picked up by the microphone and printed as "Me".

## Possible improvements

- **Translation:** a glossary for Armenian names and their case forms applied before MT, or an LLM translator (e.g. Gemma 4) measured against MADLAD on the same sets (D6).
- **Recognition:** compare BuzzASR on E1/E3 (D22); try the export's int8 ASR file if CPU time becomes tight (D20).
- **Fragments:** drop or mark low-confidence one-word segments instead of translating them (D28).
- **Speakers:** build a new speaker from several short segments that agree with each other, instead of requiring one 7-s segment, since real turns are short; a check for a speaker change inside a segment, or pyannote over a rolling window (D7 option B); tune on a real multi-person call.
- **Measurement:** an unbiased E3 reference transcribed from scratch; FLEURS (E2) to compare our scoring with ArmBench (M01, not done).

## Tests

```bash
python -m pytest -q
```

The tests cover the segmenter with a fake VAD, the capture guards, the speaker rules, output layout and pipeline error handling, plus the VAD, ASR and MT models on Tatev's E1 recordings. The models must be downloaded first. One test is an expected failure (`xfail`): the ASR inventing text on digital silence, which the VAD prevents in the program.

The E1 recordings (`data/e1/`) are not in the repository (DL4). Without them, the 51 tests that need them are skipped and pytest lists each with the reason: **188 passed, 51 skipped, 1 xfailed**. With the recordings: 239 passed, 1 xfailed. `python -m pytest -rs` prints the skip reasons.

The requirement tests, fail-loud tests and their results are in [TESTS.md](TESTS.md).

## How it was built

The work ran in seven sessions, each a separate chat with Claude (Anthropic) started from the previous session's handover in `handovers/`. Every chat is exported unedited in `chats/`; no other AI assistant was used.

- **Roles:** Tatev was the architect and decided; Claude researched options with sources, recommended one and wrote the code. Each entry in DECISIONS.md records Claude's recommendation and Tatev's choice; in D1 and D17 they differed.
- **Workflow (D11):** Claude couldn't run anything on the Mac. It wrote files; Tatev saved them, committed, ran every command and test and pasted the output back.
- **Method:** requirements and test cases were written before code ([PLAN.md](PLAN.md), [TESTS.md](TESTS.md)); model choices were measured on recordings of read speech and real calls; problems found in tests became recorded decisions (e.g. D15, D17, D24–D28).

## Licences and credits

Only permissive licences are used (D10). Models are downloaded by the user and not distributed with this repository.

**Models**

| Model | Licence | Credit |
|---|---|---|
| Silero VAD v6.2.1 | MIT | Silero Team, [snakers4/silero-vad](https://github.com/snakers4/silero-vad) |
| stt_hy_fastconformer_hybrid_large_pc | CC-BY-4.0 | NVIDIA, [model card](https://huggingface.co/nvidia/stt_hy_fastconformer_hybrid_large_pc); ONNX export by [OpenVoiceOS](https://huggingface.co/OpenVoiceOS/stt_hy_fastconformer_hybrid_large_pc_onnx) (CC-BY-4.0) |
| MADLAD-400-3B-MT | Apache-2.0 | Google, [google/madlad400-3b-mt](https://huggingface.co/google/madlad400-3b-mt); CTranslate2 int8 conversion by [Nextcloud](https://huggingface.co/Nextcloud-AI/madlad400-3b-mt-ct2-int8) (Apache-2.0) |
| WeSpeaker voxceleb_resnet34_LM | CC-BY-4.0 | [WeSpeaker](https://github.com/wenet-e2e/wespeaker), trained on VoxCeleb; ONNX export by [sherpa-onnx](https://github.com/k2-fsa/sherpa-onnx) |

**Libraries** (versions in `requirements.txt`)

| Library | Licence | Use |
|---|---|---|
| sounddevice, with bundled PortAudio | MIT | audio capture |
| cffi | MIT-0 | used by sounddevice |
| pycparser | BSD-3-Clause | used by cffi |
| numpy | BSD-3-Clause AND 0BSD AND MIT AND Zlib AND CC0-1.0 | arrays |
| onnxruntime | MIT | VAD and ASR inference |
| onnx-asr | MIT | ASR model wrapper |
| flatbuffers, packaging, protobuf, PyYAML | Apache-2.0; Apache-2.0 OR BSD-2-Clause; BSD-3-Clause; MIT | onnxruntime / onnx-asr dependencies |
| ctranslate2 | MIT | translation inference |
| sentencepiece | Apache-2.0 | translation tokeniser |
| sherpa-onnx, sherpa-onnx-core | Apache-2.0 | speaker embeddings |
| pytest, pluggy, iniconfig | MIT | tests only |
| Pygments | BSD-2-Clause | used by pytest |
| flit_core | BSD-3-Clause | build backend for `pip install -e .` |

[BlackHole](https://github.com/ExistentialAudio/BlackHole) (GPL-3.0) is an audio driver the user installs separately; it isn't part of this code.

Benchmark figures quoted above come from [ArmBench-ASR](https://metric-ai-armbench-asr.static.hf.space/index.html).
