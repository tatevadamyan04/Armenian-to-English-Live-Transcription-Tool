# Handover: session 07 (README, rehearsal, submission) → interview

## Goal of session 07

README (G01), code review (T19, T22), fresh install (G02), rehearsal (P01), deliverables (G03–G05), submission before 23:59 Yerevan on 1 Nov 2026.

## Done in session 07 (1 Nov 2026)

- **README.md written** (option A: self-contained for setup and running, ~6–10 lines per model linking to DECISIONS.md): setup incl. BlackHole, Multi-Output Device, Zoom and permissions; architecture; models and alternatives; measured results; limitations; improvements; tests; how it was built; licence credits (licences of the model conversions and of pytest/pluggy/iniconfig/Pygments checked on the Mac).
- **requirements.txt:** sherpa-onnx 2.0.1 and sherpa-onnx-core 2.0.1 were missing; a fresh clone installed without them and failed at `import sherpa_onnx` (the tests don't import it). Fixed and pushed.
- **D29:** tests that need the E1 recordings skip with the reason when `data/e1/` is missing (fresh clone: 188 passed, 51 skipped, 1 xfailed; with recordings: 239 passed, 1 xfailed).
- **Small fixes:** `translate.py` THREADS comment; PLAN.md cap 12 s → 8 s; TESTS.md table rows with an extra empty column (T15, T17, T18, T21).
- **New tool:** `eval/make_replay.py` joins pieces of recordings into one WAV for replay through BlackHole (standard library only).
- **Tests:** T19, T22, G01–G05 pass. P01 partial (below). P02 done; one item added.

## Rehearsal (P01)

- **Real Zoom call**, 2 callers on separate devices, run from the fresh clone, 2:50 (the callers then had to leave): 41 lines, latency median 1.32 s, p95 1.89 s, max 1.99 s; all mic lines "Me"; 11 filler segments dropped (D28/D24). No call-side turn reached 7 s (longest 5.92 s), so both callers shared "Speaker 1".
- **Replay through BlackHole**, 5:51: Tatev's voice (E4 mic track) 0–2:00, E3 0:13–2:01 (two speakers) 2:01–3:49, Tatev's voice (E4 3:00–5:00) 3:50–5:50. 46 lines, median 1.44 s, p95 3.39 s, max 3.48 s. The E3 part split into Speaker 1 and Speaker 2 with matches 0.80–0.92 (which line belongs to whom not checked). Tatev's voice, in short turns only (≤ 4.8 s), was Speaker 1 at the start and Speaker 2 at the end (nearest label, 0.54–0.71).
- **Decision (Claude recommended A, Tatev chose A):** keep D7's 7-s minimum; the finding goes into the README limitations; interview mitigation = everyone opens with a sentence of ≥ 7 s. Not tried: a lower minimum (S05: 1.5 s failed at every threshold; 5 s would not have helped turns of ≤ 4.8 s).
- **Found on the way:** the macOS output was not the Multi-Output Device, so the first replay never reached BlackHole; the MacBook microphone picked it up from the speakers and printed it as "Me" (D3 confirmed live). P02 now re-checks the output after the AirPods connect.

## Open / known

- R7/R8 not shown in a real call (P01 partial). Live speaker separation works only for people who speak ≥ 7 s at a stretch.
- M01 (E2 FLEURS) not done (cut list item 2).
- Everything else as in the README limitations.

## For the interview day

1. P02 checklist from the README, including the macOS output re-check after the AirPods connect.
2. Ask each reviewer on the call side to open with one uninterrupted sentence of ≥ 7 s.
3. Run from the repository (its own venv), with `2>&1 | tee data/interview/session.log` if a log is wanted.
4. P04: walk through each module and decision without notes.

## Before submitting (Tatev)

- Save README.md, TESTS.md, `eval/make_replay.py` and this handover (`handovers/07-final.md`); add the D7 note to DECISIONS.md; commit and push.
- Export this chat unedited to `chats/07-readme-rehearsal.md`; commit and push; check `git status` is clean and up to date.
- Submit the repository link.
