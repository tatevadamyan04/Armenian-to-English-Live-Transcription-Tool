# Handover: session 06 (hardening) → session 07 (README, rehearsal, submission)

Attach to the first message of chat 07, together with CLAUDE.md, PLAN.md, DECISIONS.md and TESTS.md, plus `requirements.txt`, `pyproject.toml`, `.gitignore` and the `src/hyen/` files (`__main__.py`, `audio.py`, `segmenter.py`, `vad.py`, `asr.py`, `speakers.py`, `translate.py`, `output.py`, `pipeline.py`, `modelfiles.py`) and `scripts/download_models.py`.

## Goal of session 07

1. README (G01, DL2): setup incl. BlackHole, Multi-Output Device, Zoom settings and macOS permissions · models, alternatives and reasons (DECISIONS.md) · architecture · measured numbers · known limitations and improvements · licence credits (D10).
2. Code review for T19 (Python, pip-installable) and T22 (nothing out of scope) over all of `src/hyen/` and `requirements.txt`.
3. G02 fresh install from a clean clone and a new venv.
4. Rehearsal P01 (≥ 2 Armenian speakers on the call side, ≥ 5 min, on battery), after the P02 pre-flight.
5. G03–G05, chat exports, submission before 23:59 (Yerevan).

## State after session 06 (1 Nov 2026, ~14:50)

**Decisions (all Claude recommended A, Tatev chose A)**
- **D25:** at the 8-s cap, cut after the chunk with the lowest Silero probability in the last 62 chunks (≈ 2 s), the latest on a tie.
- **D26:** the mic delivering only exact zeros for 5 s stops the program with a permission / input-volume hint (mic only).
- **D27:** a source whose buffer stays short of a block for 2 s stops the program (a timeout allowed by this entry). Timed on its own clock, so the session clock is read once per block as before.
- **D28:** the ASR's spellings of "hm" — «Հովհ», «ովհ» and a lone «Հ» — are removed as whole words before the D24 guard.
- D23 confirmed by T14/T15 (max MT 2.20 s, max latency 3.08 s); D26 confirmed by F01.

**Code changed in S06:** `src/hyen/segmenter.py` (D25), `src/hyen/audio.py` (D26, D27), `src/hyen/asr.py` (D28); tests `tests/test_segmenter.py` (5 new), `tests/test_capture_guards.py` (new, 9), `tests/test_asr_fillers.py` (new). On the Mac: **239 passed, 1 xfailed**.

**Test results (details in TESTS.md):** T14, T15, T17, T18, F01, F02, F05 pass; T16 passes on latency, but the quiet cut still split a word in fluent counting; T21: 0 lines from 60 s of typing, clicks and a notification. E4 (one call-side speaker) replaced E3 for the replay tests.

| Run (S06, E4 replay through BlackHole + Tatev speaking) | Segments | Latency median / max |
|---|---|---|
| ~75 s, plugged in | 19 | 1.37 / 2.49 s |
| ~3 min, plugged in | 28 | 1.39 / 2.46 s |
| counting run (T16), plugged in | 15 | 1.58 / 3.08 s |
| on battery, Wi-Fi off (T17, T18) | 17 | 1.50 / 2.39 s |

## For the README: limitations seen and measured

- ASR on conversational call audio: E3 CER 30.39%, WER 50.46% (reference made by correcting the output, so somewhat too low); read speech E1 CER 1.93%, WER 5.41% (D5).
- MT replaces names (Armenia → Georgia, Armenian → Persian…) under every setting tried (D6).
- MT invents text on some one-word inputs: «գյուղ։» → "2013. pp. 49. The Village." (S06); fillers and noise once turned into Bible references and footnotes before D28.
- Other lone-letter fragments («Ն …», «Տ …») still reach MT (D28 extension).
- A cut in fluent speech can still split a word (T16, D25).
- Speaker labels: new person only after a ≥ 7-s segment; one label per segment; numbers are a best case (D7). Live check: P01.
- Losing the AirPods: transcription continues, the listener hears nothing; the program can't see it (F02, D27).
- Eastern Armenian only; code-switching not handled (D13). Echo not handled; AirPods required (D3).
- `armenian_everyday_phrases.md` is committed (Tatev's own text).

## QA questions (TESTS.md §6) for S06's changes

- *Swallowed exceptions, silent defaults?* No. D26 and D27 raise CaptureError, which reaches the main thread (F04); D28 notes every removal on stderr; D25 changes only where a cut falls.
- *Network at runtime?* None added; T17 ran with Wi-Fi off.
- *Licences?* No new packages.
- *README claims backed?* The numbers above are in TESTS.md and DECISIONS.md.
- *Unverified items?* D26's zero assumption — verified by F01. D27's stall check — unit tests only, no live trigger found. Whether PortAudio raises or stalls on a real device loss — not observed (F02 kept BlackHole alive).
- *Features beyond the brief?* None; the guards serve "fail loudly" and R4/R5.
- *Last failed test?* `tests/test_audio.py`: 2 tests failed after the first D27 version (the stall check read the session clock and used up the fake clock's ticks) → fixed by giving the stall check its own clock; 239 passed after.

## Before session 07 (Tatev)

- Save DECISIONS.md, TESTS.md and this handover (to `handovers/`); commit: `git add DECISIONS.md TESTS.md handovers && git commit -m "S06: test results and handover"`.
- Book the rehearsal (P01) with ≥ 2 Armenian speakers; each opens with a sentence of ≥ 7 s (D7).
- Export this chat unedited to `chats/06-hardening.md` and commit it.
