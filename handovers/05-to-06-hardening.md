# Handover: session 05 (speakers) → session 06 (hardening)

Attach to the first message of chat 06, together with CLAUDE.md, PLAN.md, DECISIONS.md and TESTS.md, plus `src/hyen/pipeline.py`, `src/hyen/segmenter.py`, `src/hyen/speakers.py` and `requirements.txt`.

## Goal of session 06

PLAN.md §9 and TESTS.md: the Must tests, live replay (T14, T15, F05), Wi-Fi off (T17), battery (T18), cutting at the quietest chunk before the cap (T16 fix), F01, F02; QA questions (§6). Then S07 README and the rehearsal P01.

## State after session 05 (1 Nov 2026, ~14:00)

**Decisions**
- **D7 revised:** SpeechBrain ruled out by D10 (tqdm, huggingface_hub, torch's tree). Embeddings now come from WeSpeaker `voxceleb_resnet34_LM` (CC-BY-4.0) run by sherpa-onnx 2.0.1 (Apache-2.0). Settings chosen: threshold **0.77**, new speaker only from a segment of **≥ 7.0 s**, centroids updated only by such segments, segments < 0.5 s not embedded.
- **D21** ended: the placeholder is gone.

**Code added or changed in S05**
- `src/hyen/speakers.py` (M4: `SpeakerLabeler`, `SherpaEmbedder`), wired into `pipeline.py` (ASR → label → MT; the timing line shows `spk <s> (<reason> <similarity>)`).
- `src/hyen/modelfiles.py` (`SPEAKER_MODEL`), `scripts/download_models.py` (folder `spk-wespeaker`, pinned SHA-256).
- `eval/score_speakers.py` (scorer, sweep, separation check, `--template`, `--details`, `--center`, `--min-new`), `eval/cross_speakers.py` (simulated two-speaker call from E3 + E4), `eval/check_speaker_model.py` (smoke test).
- Tests: `test_speakers.py`, `test_score_speakers.py`, `test_cross_speakers.py`; `test_pipeline.py` updated. On the Mac: **206 passed, 1 xfailed**.

**Numbers (TESTS.md T10, T11; DECISIONS D7)**
- E3 alone can't measure R7 (speaker B alone ~7 s in the lettered 7:20).
- E4: Zoom call with one other person (5 min). Simulated call E3-A + E4-C (3.6 min of speech): nearest-neighbour same person 0.98; at 0.77 / 7.0 s → 2 labels, consistency 0.79, purity 0.79, T10 and T11 pass; working range 0.74–0.80. A best case.
- Embedding time: 28 ms median, 70 ms max.

## Open

- Live check of speakers: P01 (two people on one call). If labels merge or split: record P01 with the recorder, letter it with `score_speakers.py --template`, re-sweep.
- README (S07): D7 weaknesses — 7-s rule, one label per segment, best-case numbers; licence credits for sherpa-onnx and WeSpeaker/VoxCeleb (CC-BY-4.0 attribution).

## QA questions (TESTS.md §6) for S05's changes

- *Swallowed exceptions, silent defaults?* A zero/NaN embedding, an unknown source, a bad threshold and an embedder error all raise (tests). Segments < 0.5 s get the previous label with a stderr note. The labeler is a required pipeline argument.
- *Network at runtime?* None; the model is a file loaded by path; missing → F03 error with the download command.
- *Licences?* sherpa_onnx 2.0.1 and sherpa-onnx-core 2.0.1, both Apache-2.0 and the only new packages; model CC-BY-4.0. In D7.
- *README claims backed?* Numbers above are in D7 and TESTS.md.
- *Unverified items?* sherpa-onnx API — verified by the E3/E4 runs; model SHA-256 — confirmed by `--verify`.
- *Features beyond the brief?* None.
- *Last failed test?* 5 pipeline tests on the Mac (old `pipeline.py` saved with the new tests) — fixed by saving the new file; `test_score_run_replays…` after the D7 values changed (test used defaults) — fixed before sending.

## Before session 06 (Tatev)

- Save the files; `python -m pytest -q`; commit: `git add -A eval src tests scripts DECISIONS.md TESTS.md PLAN.md handovers requirements.txt && git commit -m "S05: M4 speaker labels (D7 revision) and handover"`; then `git ls-files | grep -E "\.(wav|onnx|bin)$|data/"` must print nothing (G03).
- Export this chat unedited to `chats/05-speakers.md` and commit it.
