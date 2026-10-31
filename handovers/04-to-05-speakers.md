# Handover: session 04 (measure) → session 05 (speakers)

Attach to the first message of chat 05, together with CLAUDE.md, PLAN.md, DECISIONS.md and TESTS.md, plus `src/hyen/pipeline.py`, `src/hyen/output.py` and `requirements.txt`.

## Goal of session 05

PLAN.md §9: M4 `speakers.py` — "Me" for the mic, "Speaker N" for the call side, stable for the whole session (R6–R8, D7). Replace the D21 placeholder. Tune on E3 offline (`data/e3/system.wav` + speaker letters in `data/e3/e3_ref.tsv`), then measure T10 and T11.

## State after session 04 (1 Nov 2026, ~03:25)

**Decisions made in S04**
- **D22** — B comparisons cut (BuzzASR, Gemma 4); only A measured. Reason: S04 started at 00:50 on the deadline day.
- **D23** — MT decoding: beam 4, repetition penalty 1.2, no repeated 3-token sequence; M2 cuts segments at 8 s (was 12 s).
- **D24** — guards: ASR text with < 2 Armenian letters → no text (stderr note); MT output ≤ 2 × input tokens + 16 (stderr note if reached).
- **D6** — replaced names (Armenia → Georgia, Armenian → Persian) are a known limitation (option A); no decoding setting fixed them.
- **D9 revision** — E3 reference made by correcting the program's output; kept in `data/e3/` (git-ignored, the participants were told the recording stays on the laptop); no English for E3; Armenian 0:13–2:01; speaker letters 0:00–7:20.
- TESTS.md §3.2 normalisation confirmed by Tatev.

**Code added or changed in S04**
- `eval/metrics.py` (normalisation, CER, WER, chrF; stdlib only), `eval/score.py` (id/hy/en TSVs; rating sheet), `eval/run_offline.py` (ASR + MT over `<id>.wav`; `--mt-input ref`; decoding flags), `eval/segment_offline.py` (the program's M2 → M3 → M5 over one recording), `eval/score_e3.py` (CER/WER over a time window; `--export-mt`).
- `src/hyen/translate.py` (D23 defaults, D24 output limit), `src/hyen/segmenter.py` (cap 250 chunks = 8 s), `src/hyen/asr.py` (D24 short-text guard).
- `eval/refs/long.tsv` — 5 long sentences with names; drafted by Claude, checked by Tatev (Tatev made no changes).
- Tests: `test_metrics.py` (25), `test_score_e3.py` (13), `test_guards.py` (10); `test_segmenter.py` expects 4 pieces for 30 s. On the Mac: **164 passed, 1 xfailed**.

**Numbers (TESTS.md T07, T08; DECISIONS D5, D6, D23, D24)**
- E1 ASR: CER 1.93%, WER 5.41%, 20/25 identical.
- E3 ASR (call side 0:13–2:01, colloquial, 6/12 turns overlapping): CER 30.39%, WER 50.46% (somewhat too low: corrected-output reference).
- MT chrF on E1 reference Armenian: 80.6 (S03 settings) → 82.3 (D23). End to end at S03 settings: 78.4.
- E3 with D23 (92 segments, 16 cut at 8 s): slowest MT 2.08 s, slowest ASR 0.12 s → estimated worst latency ≈ 3.8 s.

## Not done in S04 (moved)

- E2 / M01 — cut (PLAN.md cut list item 2); README says the scoring wasn't checked against ArmBench. **Tatev to confirm** (proposed at 03:12, answer pending).
- MT rating 1–3 (TESTS.md §3.3) — S06 if time allows, on E1.
- T15, F05, latency report from the stderr timing log — S06 full run.

## Open for S05 — check first, before any code

- **D7's model needs PyTorch.** SpeechBrain ECAPA (`speechbrain/spkrec-ecapa-voxceleb`) runs on torch + torchaudio. Unverified: Python 3.14 wheels (D1) and the licences of their whole dependency trees (D10; D19 already found torch's tree brings licences outside D10). If it fails: propose a speaker-embedding model exported to ONNX that runs on the existing onnxruntime (licence and source to check), as a revision of D7.
- E3 for tuning: of 12 turns in 0:13–2:01, 6 are `A+B` (two people at once). Score speaker metrics on single-speaker rows; report mixed rows separately.
- How to run the speaker labels offline: extend `eval/segment_offline.py` (label column) and add a speaker scorer (consistency, purity, labels created; TESTS.md §3.3) matching hyp segments to `e3_ref.tsv` rows by time overlap.

## Schedule (from 03:25, deadline 23:59)

| Work | h |
|---|---|
| S05 speakers | 3–4 |
| S06 hardening (Must tests, live replay T14/T15/F05, Wi-Fi off, battery, cut at quietest chunk, F02) | 2.5–3 |
| S07 README | 1–1.5 |
| Rehearsal P01 | 0.5 |
| Export chats, G01–G05, submit | 0.5 |

Cut order if late: S06 → Must tests only; shorter rehearsal; shorter README.

## QA questions (TESTS.md §6) for S04's changes

- *Swallowed exceptions, silent defaults?* The D24 guards change behaviour but each notes it on stderr. Eval scripts stop on missing ids, bad headers, existing output files.
- *Network at runtime?* None added.
- *Licences?* No new dependencies (metrics are our own code).
- *README claims backed?* Numbers above are in TESTS.md and DECISIONS.md.
- *Unverified items?* CTranslate2 options verified in its source (translation.h). chrF implementation not cross-checked against sacrebleu.
- *Features beyond the brief?* None; the guards and eval tools serve R4, R5, C1.
- *Last failed test?* `score_e3.py` on `e3_ref.tsv` line 9 (missing tab), fixed in the data; Claude's miscounted expectation in `test_guards.py` (7 letters, not 8), fixed before sending.

## Before session 05 (Tatev)

- Save `DECISIONS.md` and this file; commit: `git add -A eval src tests DECISIONS.md TESTS.md PLAN.md handovers && git commit -m "S04 results and handover"` — then run `git ls-files | grep -E "\.(wav|onnx|bin)$|data/"`, which must print nothing (G03).
- Export this chat unedited to `chats/04-measure.md` and commit it.
