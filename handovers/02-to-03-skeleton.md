# Handover: session 02 (audio capture) → session 03 (walking skeleton)

Attach to the first message of chat 03, together with CLAUDE.md, PLAN.md, DECISIONS.md and TESTS.md.

## Goal of session 03

The walking skeleton (PLAN.md §9): M2 segmenter (Silero VAD), M3 ASR (A: NVIDIA hy FastConformer, ONNX via onnx-asr), M5 translation (A: MADLAD-400-3B-MT, CTranslate2 int8), M6 output, M7 pipeline and CLI, `scripts/download_models.py`, replay through BlackHole, per-stage timing log on stderr. Done when the S03 tests pass on E1: T06, T07, T08, T09, T12, T14, T16, T20, F03, F04.

## State after session 02 (31 Oct 2026)

**Audio routing (D2, settled).** BlackHole 2ch is installed from the official installer and runs at 48 kHz on macOS 26.5.2. The Multi-Output Device holds BlackHole 2ch (primary/clock device, 48 kHz) and the AirPods (drift correction on); the MacBook speakers and the external monitor are not in it. macOS output is the Multi-Output Device; macOS input is the MacBook Pro Microphone. Zoom's audio settings are not confirmed yet (see "Before session 03").

**Python.** Python 3.14.5 venv; `pip install -e .` works (flit_core, D16). `requirements.txt` is pinned:

| Package | Version | Licence (package metadata) |
|---|---|---|
| numpy | 2.5.4 | BSD-3-Clause AND 0BSD AND MIT AND Zlib AND CC0-1.0 |
| sounddevice | 0.5.6 | MIT (bundles PortAudio, MIT) |
| cffi | 2.1.1 | MIT-0 |
| pycparser | 3.12 | BSD-3-Clause |
| pytest | 9.2.0 | MIT |
| iniconfig | 2.3.1 | MIT |
| packaging | 26.3 | Apache-2.0 OR BSD-2-Clause |
| pluggy | 1.7.0 | MIT |
| Pygments | 2.21.0 | BSD-2-Clause |

All are allowed by D10 (revised on 30 Oct). This table goes into the README credits (S07).

**Code.**
- `src/hyen/audio.py` (M1): `list_input_devices()`; `resolve_input_device()` takes an exact name or an index, never a default or a partial name; `Capture(devices)` with `start(t0)`, `get(name)` (waits for the next chunk), `drain(name)` (no waiting), `stats(name)` and `stop()`. One input stream and one reader thread per source (D17). Core Audio converts to 16 kHz (D14). Chunks are `AudioChunk(source, samples, t_start)`: float32, mono, exactly 512 samples (32 ms), `t_start` counted from samples on the session clock (D15, corrected). Read errors and streams that stop by themselves are raised in the consumer as `CaptureError`; overflows are printed on stderr.
- `src/hyen/__main__.py`: `--list-devices`; `--mic` and `--system` are both required. For S02 only, it runs a capture check (peak levels on stderr every 2 s until Ctrl+C); M7 replaces this in S03.
- `eval/record.py` (M8): `session` mode records both sources, aligned to the session clock and cut at exactly N seconds, and fails on an overflow, a clock gap over 0.1 s or a silent source (D18); `prompts` mode records E1, one file per reference row, and skips rows that already have a file.
- `eval/clock_check.py`: diagnostic; opens one device at a time and compares delivered samples with the computer's clock.
- `tests/test_audio.py`: 21 unit tests; pass on the Mac.

**Tests (TESTS.md, with values).** T01, T02, T03, T04, T05, T13 and T23 pass. T04 failed in five runs before passing, from three causes, each documented: silent sample loss (→ D17), the timestamp anchor (→ D15 correction) and the recorder's length comparison (→ D18).

**E1.** 25 everyday phrases recorded in `data/e1/` (`e1_01.wav` … `e1_25.wav`, 16 kHz mono 16-bit). References in `eval/refs/e1.tsv`, written by Tatev; one Cyrillic letter in e1_15 was corrected to the Armenian one. All 25 are short (2–6 words); three or four long sentences (15+ words) were suggested as e1_26 onwards and are still open.

## Decisions made in session 02

- **D14** — 16 kHz by Core Audio inside PortAudio (A).
- **D15** — chunk times counted from samples (A); corrected 31 Oct: the anchor includes the audio already buffered at the first read.
- **D16** — `pip install -e .` with flit_core (A).
- **D10 revised** — allowed: MIT, MIT-0, Apache-2.0, BSD, 0BSD, Zlib, CC0-1.0, CC-BY-4.0; transitive dependencies included; anything else is raised first.
- **D17** — a reader thread per source, PortAudio buffering in C (B; Claude first recommended A). D8 now reads "one capture reader thread per device".
- **D18** — T04 judges lost audio by each source's clock gap (± 0.1 s) and overflows (B).
- **D2 settled** — the configuration above; the switch to catap was not needed.
- **Schedule** — sessions 02–07 moved to 30 Oct – 1 Nov (PLAN.md §9).

## Measured on the Mac (useful for S03)

- Native rates: MacBook Pro Microphone and BlackHole 2ch 48 kHz; the AirPods' microphone 24 kHz (not used).
- In callback mode at 16 kHz the mic arrived in 15-sample blocks and BlackHole in 1,600-sample blocks (0.1 s); at 48 kHz, 512 and 4,096.
- From the session clock's zero to the first audio: 0.22–1.87 s (opening and starting both streams); the two streams' first audio within about 10 ms of each other.
- Clock gaps after 60–120 s: within ±0.013 s on both sources — no loss, no drift.
- Levels: room noise about −50 dBFS; speech peaks −22 to −27 dBFS at normal distance; video through BlackHole peaked at −0.6 to −2 dBFS, so turn the player or Zoom volume down for headroom (peaks around −6 dBFS).
- Device indices change between runs (the mic was [2], later [3]); always select by name.
- Not measured: Core Audio's input latency (estimated tens of milliseconds, D15) and PortAudio's buffer size in reading mode (D17).

## Open questions for session 03

- Licences before installing anything (D10, incl. transitive dependencies): silero-vad (read the LICENSE file, K8), onnx-asr, onnxruntime, ctranslate2, the MADLAD tokenizer library, huggingface_hub. Each must install on Python 3.14 from wheels; if one fails, stop and revisit D1 (K7).
- How system-side segments are labelled until M4 exists (S05) — agree in S03.
- Where the session clock starts: M7 loads every model first, then takes `t0` and starts capture.
- Silero VAD's window size at 16 kHz versus M1's 512-sample chunks (unverified).
- A stream that stops delivering without an error makes the consumer wait forever; detecting it needs a timeout, which needs a decision (F02, S06).
- E1 replay: `afplay data/e1/<id>.wav` plays to the Multi-Output Device and reaches the program through BlackHole.
- The exact submission time on Sunday.

## Schedule (proposal — agree at the start of S03)

S02 took about 5 h of active work instead of 2.5 h (the BlackHole install and the T04 investigation). About 7 h of the ≈ 20-h budget are used; the remaining plan needs about 15.5 h.

| When | Session | h |
|---|---|---|
| Sat 31, morning | 03 Walking skeleton — start the model downloads first | 3 |
| Sat 31, midday | E3: Zoom call with 2–3 friends (~10 min usable) | 0.5 |
| Sat 31, afternoon | 04 Measure | 4 |
| Sat 31, evening | 05 Speakers | 3.5 |
| Sun 1 | 06 Hardening, 07 README, rehearsal (P01) | 4 |
| Sun 1 | export chats, submit | 0.5 |

Proposed checkpoint: if S03 isn't passing on E1 by Saturday 15:00, take the first cut from PLAN.md §9 (the B comparisons in S04).

## M1 QA questions (TESTS.md §6)

- *Swallowed exceptions, retries, timeouts, silent defaults?* None. Reader-thread errors and streams that stop by themselves are raised in the consumer; devices must be named exactly; stopping and closing raise their errors. One known gap: a stalled stream (no audio, no error) waits forever — needs a timeout decision (F02).
- *Network at runtime?* None; M1 and the tools use only local audio devices.
- *Licences allowed and credited?* Allowed: yes (table above). Credited: not yet; the README comes in S07.
- *README claims backed?* No README yet. S02's entries in DECISIONS.md carry sources or measurements.
- *"Unverified" items?* BlackHole on macOS 26.5.2 and the flit_core install are now verified. Still unverified: Core Audio's input latency and why the first read finds a backlog (D15), PortAudio's buffer size (D17). Not yet tested: Zoom's microphone auto-adjust (E3, P02).
- *Features beyond the brief?* None. The recorder and the clock check are M8 tooling; the capture-check mode is a temporary S02 diagnostic.
- *Last failed test documented?* T04 — five failed runs before passing, all documented in TESTS.md and DECISIONS.md (D15, D17, D18).

## Before session 03 (Tatev)

- Zoom → Settings → Audio: Speaker = Multi-Output Device; Microphone = MacBook Pro Microphone (not "Same as System"); untick "Automatically adjust microphone volume". Needed for E3 and the interview (P02).
- Arrange E3 (2–3 Armenian-speaking friends, ~15 min on Zoom, everyone's consent to record) for Saturday, and the rehearsal (≥ 2 call-side speakers, ≥ 5 min, on battery) for Sunday.
- Optional: write the three or four long E1 sentences and bring them to chat 03.
- Delete the two draft sentence files from the repo root; `eval/refs/e1.tsv` is the reference.
- Check that `chats/01-brief-review-and-plan.md` is committed; export this chat unedited to `chats/02-audio-capture.md` and commit it.
- Before pushing: `git log --name-only --format= | grep -Ei '\.(wav|flac|mp3|m4a|onnx|bin|safetensors|nemo|pt)$|^(\.venv|data|models)/'` must print nothing (G03).
- Keep several GB of disk space free for the models (exact sizes checked in S03).

## Next step

Open chat 03 with this handover and the four docs. Agree the schedule and the M2–M7 interfaces, check the licences, then write and run `scripts/download_models.py` first, because the downloads are large.
