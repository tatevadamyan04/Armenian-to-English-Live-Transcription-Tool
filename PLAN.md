# PLAN — live Armenian → English call transcription

Mathrix AI coding challenge (brief of 21 Oct 2026) · Tatev Adamyan · deadline 1 Nov 2026 · budget ≈ 20 h
See also: [DECISIONS.md](DECISIONS.md) · [TESTS.md](TESTS.md) · [CLAUDE.md](CLAUDE.md) · `handovers/`

## 1. Goal

A Python program for the macOS terminal that listens to a live Zoom call. It captures the MacBook microphone and the system audio at the same time, transcribes the spoken Armenian, translates each phrase into English and prints it with the speaker — "Me" for the microphone, "Speaker 1, 2, …" for the call side — no later than 5 seconds after the phrase ends. All models run locally on Tatev's MacBook.

## 2. Requirements

IDs are used in TESTS.md and DECISIONS.md.

| ID | Requirement (from the brief) |
|---|---|
| R1 | Capture the microphone and the system audio output at the same time |
| R2 | The user still hears the call (macOS: BlackHole + Multi-Output Device) |
| R3 | The user picks the input devices (printed device list or command-line flags) |
| R4 | Spoken Armenian becomes Armenian text |
| R5 | Every transcribed segment is translated into English |
| R6 | Everything from the microphone is labelled "Me" |
| R7 | People on the system-audio side are told apart: "Speaker 1", "Speaker 2", … |
| R8 | The same person keeps the same label for the whole session |
| R9 | Each segment is printed as it is processed, with time, speaker, Armenian original and English translation |
| C1 | At most 5 s from the end of a spoken phrase to its English line |
| C2 | Transcription, translation and speaker identification run locally; no cloud speech or translation APIs |
| C3 | Python |
| C4 | Runs on the interview laptop: MacBook Pro, Apple M4 Pro, 24 GB, macOS 26.5.2 |
| C5 | Models chosen deliberately; the README explains the choice, the alternatives and why |
| DL1 | Git repository with the source code (private, reviewers have access) |
| DL2 | README: setup incl. audio devices · models, alternatives, reasons · architecture · limitations and improvements |
| DL3 | `chats/`: every session exported in full, unedited, one .md file per session |
| DL4 | No model weights, audio recordings or API keys in the repository |
| I1 | Interview: live Armenian conversation with ≥ 2 people on the call side |
| I2 | Walkthrough of the code and the decisions |
| I3 | Original chat history available in the app |
| I4 | Program installed and audio setup working before the interview starts |

**Out of scope — do not build:** GUI · installer · echo handling · naming speakers or remembering them across sessions · saving transcripts · input languages other than Armenian. Spare time goes into making the core more reliable.

**Interpretations:** Eastern Armenian only; Russian or English words mixed into speech are a known limitation (D13). "Local only" is proven by running with networking off (D12). Permissive licences only (D10).

## 3. Audio setup on the Mac (D2, D3)

- Zoom: speaker = **Multi-Output Device**, microphone = **MacBook Pro Microphone**.
- Multi-Output Device = **BlackHole 2ch** + **AirPods 4**. Which device is the clock is settled in session 02.
- The program opens two inputs: the MacBook microphone (source `mic`) and BlackHole 2ch (source `system`).
- Tatev listens on the AirPods only, so the call doesn't leak into the microphone.

## 4. Architecture (D8)

```
MacBook mic ───► M1 capture ──► M2 segmenter ──┐
                                               ├──► queue ──► inference worker ──────────────────► M6 output ──► terminal (stdout)
BlackHole 2ch ─► M1 capture ──► M2 segmenter ──┘              M3 ASR → M4 speaker → M5 translate
```

One process: one capture reader thread per device (D17), one segmenter thread per source and one inference worker, so the models load once and never compete for CPU or GPU. Per-stage timings go to stderr; stdout carries only the transcript.

## 5. Modules

Messages between modules (dataclasses):

- `AudioChunk(source, samples, t_start)` — `source` is `mic` or `system`; `samples` are float32 mono at 16 kHz, 512 per chunk (D17); `t_start` is seconds since the session started.
- `Segment(seg_id, source, samples, t_speech_start, t_speech_end, t_end_detected)`
- `Result(seg_id, t_speech_start, label, hy, en, timings)`

| Module | Input | Output | How (decision) | Session |
|---|---|---|---|---|
| M1 capture — `audio.py` | device names or indices from the CLI | `AudioChunk` stream per source | python-sounddevice; a reader thread per device takes 512-sample blocks from PortAudio's buffer (D17); BlackHole stereo downmixed; converted to 16 kHz by Core Audio inside PortAudio (D14); `t_start` counted from samples (D15); `--list-devices`; an unknown device is an error (D2) | S02 |
| M2 segmenter — `segmenter.py` | `AudioChunk` stream of one source | `Segment` when a phrase ends | Silero VAD; a phrase ends after ~600 ms of silence; segments capped at ~12 s; both tuned in S03 (D4) | S03 |
| M3 ASR — `asr.py` | `Segment.samples` | Armenian text; empty text drops the segment with a note on stderr | NVIDIA hy FastConformer, ONNX export via onnx-asr (D5) | S03, S04 |
| M4 speakers — `speakers.py` | `Segment` | `"Me"` for mic; `"Speaker N"` for system | ECAPA embedding + online clustering by cosine similarity (D7) | S05 |
| M5 translate — `translate.py` | Armenian text | English text | MADLAD-400-3B-MT, CTranslate2 int8 on CPU (D6) | S03, S04 |
| M6 output — `output.py` | `Result` | three lines on stdout, flushed at once | layout below | S03 |
| M7 pipeline + CLI — `pipeline.py`, `__main__.py` | CLI flags | session running until Ctrl+C | loads every model before audio starts; an error in any thread stops the program (F04) | S03 |
| M8 eval tools — `eval/` | recordings + reference files | CER/WER, chrF, latency and speaker reports | development tooling, not a product feature; recorder in S02, scorers in S04 | S02–S05 |
| Model download — `scripts/download_models.py` | — | models in the local cache | the only step that uses the network; the program sets `HF_HUB_OFFLINE=1` (D12) | S03 |

Output layout (from the brief). The time is the segment's speech start, counted from the session start:

```
[00:01:12] Speaker 1
  HY: Բարև, ինչպե՞ս ես։
  EN: Hi, how are you?
```

**Latency** = time the EN line is printed − end of speech as found by the VAD. It includes the silence wait, queueing, ASR, speaker matching, translation and printing. Planning budget — estimates to check in S03, not measurements:

| Stage | Target |
|---|---|
| Silence wait before the phrase counts as ended | ~0.6 s |
| Queue wait while the other source is processed | ≤ 1.0 s |
| ASR, 12 s segment | ≤ 1.0 s |
| Speaker embedding | ≤ 0.3 s |
| Translation | ≤ 1.5 s |
| Margin | ≥ 0.6 s |
| **Total** | **≤ 5.0 s** |

## 6. Repository layout

```
README.md                  written in S07
PLAN.md  DECISIONS.md  TESTS.md  CLAUDE.md
handovers/                 one file per session hand-off
chats/                     every session exported unedited, one .md each
requirements.txt           exact versions (Python 3.14.5 venv, D1)
pyproject.toml             package metadata for `pip install -e .` (D16)
src/hyen/                  M1–M7
scripts/download_models.py
eval/                      M8 tools; eval/refs/ holds reference texts only
tests/                     pytest unit tests for pure logic
data/                      recordings — git-ignored, never committed
```

## 7. Risks

| # | Risk | Affects | Mitigation | Checked by |
|---|---|---|---|---|
| K1 | Armenian ASR is weak on conversational call audio (best open model: 60.78% WER on ArmBench "Movies") | R4, live result | Choose by numbers on the call set; compare with BuzzASR in S04; report limits honestly | T07, E3 |
| K2 | The ONNX export has only the CTC decoder; which decoder ArmBench scored is unknown | R4 | Compare our FLEURS WER with ArmBench's 15.00% | M01 |
| K3 | hy→en translation quality is unknown; ASR errors carry into the English | R5 | Tatev's rating + chrF; compare with Gemma 4 12B in S04 | T08, E1, E3 |
| K4 | Speaker labels drift on Zoom-processed voices; one segment can hold two speakers | R7, R8 | Tune the threshold on E3 replay; minimum length before creating a speaker; report consistency | T10, T11 |
| K5 | BlackHole on macOS 26.5.2 unverified (one vendor article claims glitches on Tahoe) | R1, R2 | Test first in S02; switch to the pre-named alternative (catap) if it fails | T04, T05 |
| K6 | Latency over 5 s when both sides talk, on long phrases, or on battery | C1 | One worker; per-stage timing from S03; segment cap; runs on battery | T14–T16, T18 |
| K7 | A dependency has no Python 3.14 build (D1) | setup | Install all dependencies early (S02–S04); if one fails, stop and revisit D1 | install log |
| K8 | Silero VAD licence unclear (a badge says CC BY-NC; text and metadata say MIT) | D10 | Read the LICENSE file before adopting it in S03 | D4 |
| K9 | Missing macOS permission or wrong routing gives silent input that looks like "nobody talks" | R1 | Pre-flight checklist; loud warning when the mic delivers only zeros | F01, P02 |
| K10 | Invented text on silence or noise | R4 | VAD gating; CTC model; silence and noise tests | T20, T21 |
| K11 | Russian/English words mixed into Armenian | R4, R5 | Known limitation in the README (D13) | README |
| K12 | 20 h budget with downloads and setup | all | Schedule with a cut list (§9); three-strikes rule (CLAUDE.md) | daily |
| K13 | Recordings or weights committed by accident | DL4 | `.gitignore` from S02; check `git ls-files` before each push | G03 |

## 8. Testing approach

- Test cases come from §2, not from the code: [TESTS.md](TESTS.md).
- **Unit tests** (pytest) for pure logic: text normalisation, clustering decisions, output layout, the segmenter on synthetic audio.
- **Replay tests:** recordings are played on the Mac so they arrive through BlackHole exactly like a call — repeatable without other people.
- **Measurement:** eval sets E1–E3 with CER/WER, Tatev's rating and chrF, latency and speaker metrics (TESTS.md §3). Models are compared with numbers.
- **Live tests:** Zoom with friends — recording E3 on Saturday morning, rehearsal on Sunday.

## 9. Sessions and schedule

One chat per session; each starts from the previous handover.

| Day | Session | Work | Done when | h |
|---|---|---|---|---|
| Mon 26 – Tue 27 | 01 Plan | brief review, decisions, the four docs, handover | docs committed | 2 |
| Fri 30, afternoon | 02 Audio capture | BlackHole + Multi-Output; M1 + recorder; record E1 | S02 tests pass; E1 recorded | 2.5 |
| Fri 30, evening | 03 Walking skeleton | M2, M3 (A), M5 (A), M6, M7; model download script; replay through BlackHole; timing log | S03 tests pass on E1 | 3 |
| Sat 31, morning | — | Zoom with 2–3 friends: record E3 (~10 min) | E3 recorded | 0.5 |
| Sat 31 | 04 Measure | transcribe ~3 min of E3; scoring scripts; E2; ASR and translation A vs B; numbers into D5/D6 | TESTS.md §3 numbers logged | 4 |
| Sat 31, evening | 05 Speakers | M4; tune on E3 replay | S05 tests measured | 3.5 |
| Sun 1 Nov | 06 Hardening, 07 README | full TESTS.md run; fix the weakest link; Wi-Fi-off and battery runs; README; rehearsal call | Must tests pass or are documented; README done | 4 |
| Sun 1 Nov | — | buffer; export chats; submit the repository link | G01–G05 | 0.5 |
| | | | **Total** | **20** |

"S0X tests" = the TESTS.md cases whose *From* column is that session.

Rescheduled on 30 Oct: session 02 started three days after the original plan (Tue 27), so sessions 02–07 moved to Fri 30 – Sun 1 Nov with the same work, exit criteria and hours. Tatev is available until the deadline. The B comparisons stay in S04 (confirmed 30 Oct); the cut list below is unchanged and applies only if we fall behind.

**If we fall behind, cut in this order:** (1) the B comparisons in S04 — keep A and report it as a limitation; (2) E2; (3) S06 shrinks to the Must tests.

## 10. How the evaluation areas are covered

| Area (brief) | Where it shows |
|---|---|
| Working method | this plan, TESTS.md, handovers, measurements |
| Use of AI tools | DECISIONS.md records Claude's recommendation and Tatev's choice; claims carry sources or "unverified" |
| Decisions | DECISIONS.md → README "Models and alternatives" |
| Live result | TESTS.md §1 and the rehearsal (P01) |
| Code | module boundaries in §5 |
| Communication | README (G01, G02) |
