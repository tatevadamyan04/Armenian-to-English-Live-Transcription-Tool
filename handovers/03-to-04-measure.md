# Handover: session 03 (walking skeleton) → session 04 (measure)

Attach to the first message of chat 04, together with CLAUDE.md, PLAN.md, DECISIONS.md and TESTS.md.

## Goal of session 04

PLAN.md §9: transcribe ~3 min of E3; scoring scripts (CER/WER with the TESTS.md §3.2 normalisation, chrF, latency report); E2 (FLEURS) and the M01 sanity check; ASR and translation A vs B; the numbers and outcomes into D5 and D6. Also T15 (E3 replay through BlackHole while Tatev reads into the mic) and F05 (long replay, nothing dropped).

## State after session 03 (31 Oct 2026, ~02:30)

**The program runs end to end.** `python -m hyen --mic "MacBook Pro Microphone" --system "BlackHole 2ch"` loads all models (2.4 s), then captures both sources, segments, transcribes, translates and prints each phrase with "Me" or "Speaker 1" (D21 placeholder). One timing line per segment on stderr.

**Code (new in S03)**
- `scripts/download_models.py` (D20): standard library only; Silero VAD @ v6.2.1 (GitHub), ASR @ f7db1fa and MADLAD CT2 int8 @ aa32bbd (Hugging Face); every model file checked against a pinned SHA-256; `--verify` checks without the network. All 11 files downloaded and valid on the Mac (~3.4 GB in `models/`, git-ignored).
- `src/hyen/modelfiles.py`: paths into `models/`; `require()` raises `ModelMissingError` naming the file and the download command (F03).
- `src/hyen/vad.py` (M2, D19): Silero ONNX on onnxruntime, 512 samples + 64 context, state (2, 1, 128), one instance per source, 1 thread.
- `src/hyen/segmenter.py` (M2, D4): start ≥ 0.5, continue ≥ 0.35, end after 19 quiet chunks (608 ms), pre-roll and tail 3 chunks (96 ms), cut at 375 chunks (12 s), drop < 8 speech chunks (256 ms) with a note on stderr. `seg_id` = `<source>-<n>`. Raises on wrong source, wrong size or a gap between chunks.
- `src/hyen/asr.py` (M3, D5): onnx-asr `load_model("nemo-conformer-ctc", models/asr-hy-fastconformer, providers=["CPUExecutionProvider"])`; checks the three model files first (onnx-asr would otherwise create the folder and try to download).
- `src/hyen/translate.py` (M5, D6): CTranslate2 int8 on CPU, 8 threads, beam 1, max 256 tokens; SentencePiece `spiece.model`; input `"<2en> " + text`.
- `src/hyen/output.py` (M6): `Result`, `clock()`, the three-line layout, flushed at once.
- `src/hyen/pipeline.py` (M7): segmenter thread per source → one queue → one worker (ASR → label → MT → print). Thread errors are re-raised in the main thread with the thread's name (F04). Ctrl+C stops at once; a phrase in progress and queued segments are not printed.
- `src/hyen/__main__.py`: sets `HF_HUB_OFFLINE=1` first; S02's capture check removed; `ModelMissingError` → message + exit 1.
- Tests: `test_segmenter.py` (11, fake VAD), `test_vad_silero.py` (27, real model + E1), `test_asr_mt.py` (51, real models + E1; one strict xfail), `test_pipeline.py` (7, fakes; F04 incl. a subprocess exit code). On the Mac: **116 passed, 1 xfailed**.
- `requirements.txt`: + ctranslate2 4.8.4, flatbuffers 25.12.19, onnx-asr 0.12.0, onnxruntime 1.31.1, protobuf 7.36.2, PyYAML 6.0.3, sentencepiece 0.2.2 (licences in D20).

**Tests (values in TESTS.md):** T06, T07 (format), T08 (format), T09, T12, T14 (E1 replay), T16, T20, F03 and F04 pass. TESTS.md §3.2 step 3 was revised (inner Armenian marks deleted, not replaced by a space) — **confirm at the start of S04**.

## Decisions made in session 03

- **D4** licence settled: MIT (LICENSE file); the CC BY-NC badge is treated as outdated.
- **D19** — Silero ONNX run directly with onnxruntime, no torch (A).
- **D20** — standard-library download script, pinned versions and SHA-256, no huggingface_hub (A).
- **D21** — every call-side segment is "Speaker 1" until M4 (A, temporary).
- D5 and D6 got "S03 observations" (below).

## Measured on the Mac (S03)

- Model loading: ASR 0.5 s, MT 1.7 s; all models 2.4 s.
- ASR: 0.01–0.02 s per E1 phrase, 0.14–0.15 s per 12-s segment. E1 offline: 16/25 identical to the reference, 2 more only punctuation/«և».
- MT: 0.22–0.79 s per short phrase, 1.1–2.55 s per 12-s segment (over the 1.5-s planning figure).
- End-of-phrase detection: 0.6–0.86 s after the last speech; 0.13–0.23 s at a forced cut.
- Latency, run 1 (19 segments, E1 replay + mic): median 1.22 s, p95 1.68 s, max 2.88 s. Run 2 (continuous reading, 5 segments): 1.36–1.97 s. Queue wait ≤ 0.05 s.
- ASR alone on 3 s of digital zeros: «Ինչ էին։» (K10; kept away by the VAD).

## Problems found (for S04 and S06)

1. **MT replaces names** (D6): Armenia → Georgia (×2), Kazakhstan; «հայկական» → Bulgarian; «հայոց» → Persian; Mesrop → Mehmed. **Adds content**: "I am 20 years old."; repeats sentences ("One. One. One."). The S04 A-vs-B comparison should include named entities in the rating; try a larger beam and repetition limits on A first (CTranslate2 options, unverified).
2. **Cuts at 12 s split words** (T16): the word at each cut is misrecognised. S06: cut at the quietest chunk in the last ~1–2 s before the cap.
3. **MT time grows with segment length**: 2.55 s for one 12-s segment. Fine alone, but with a queue wait (T15, both sources talking) it could approach 5 s. Measure in T15.
4. **E1 is all short phrases** (2–4 s); the long sentences (e1_26+) are still not recorded. Run 2 shows long speech is where the errors are.

## Open questions for session 04

- Confirm the revised normalisation (TESTS.md §3.2).
- Model B for ASR (BuzzASR, ~3 GB) and MT (Gemma 4 12B 4-bit via mlx-lm, ~7 GB): licences of the runtimes and their transitive dependencies (D10), Python 3.14 wheels (D1), download size and time. Check before downloading; this is the first item on the cut list if time runs short (PLAN.md §9).
- How E3 is transcribed (Tatev, ~3 min with speaker turns) and how replay timing is matched to the reference.
- The exact submission time on Sunday (still open from S02).

## Schedule

S03 took about 2 h (00:46–02:30) on top of S02, so about 9 h of the ≈ 20-h budget are used. Remaining plan (from the S02 handover):

| When | Session | h |
|---|---|---|
| Sat 31, midday | E3: Zoom call with 2–3 friends (~10 min usable) | 0.5 |
| Sat 31, afternoon | 04 Measure | 4 |
| Sat 31, evening | 05 Speakers | 3.5 |
| Sun 1 | 06 Hardening, 07 README, rehearsal (P01) | 4 |
| Sun 1 | export chats, submit | 0.5 |

The S03 checkpoint (passing on E1 by Saturday 15:00) is met.

## QA questions for M2–M7 (TESTS.md §6)

- *Swallowed exceptions, retries, timeouts, silent defaults?* None added. Thread errors reach the main thread; short segments and empty ASR text are dropped **with a note on stderr**. Known gap still open: a stalled stream (no audio, no error) waits forever (F02, S06).
- *Network at runtime?* None: `HF_HUB_OFFLINE=1`, huggingface_hub not installed, model files checked before onnx-asr loads. The download script is the only network step. T17 (Wi-Fi off) is in S06.
- *Licences allowed and credited?* Allowed: all (D10; D20 table). Credited: README in S07.
- *README claims backed?* No README yet; S03 numbers are in TESTS.md and D5/D6.
- *"Unverified" items?* Verified in S03: Silero's input contract and checksum, onnx-asr's local loading, CTranslate2 int8 on Apple Silicon (it runs). Still unverified: why the first read finds a backlog and Core Audio's input latency (D15), PortAudio's buffer size (D17), CTranslate2's repetition options.
- *Features beyond the brief?* None. `test_vad_silero.py`/`test_asr_mt.py` are test tooling.
- *Last failed test documented?* `test_asr_mt.py` silence check: failed because the ASR model returned text on digital zeros; now a strict xfail and recorded in D5 and T20. Before it: the Silero tests errored because the first download-script version had no Silero; fixed by re-running the second version.

## Before session 04 (Tatev)

- Commit the S03 docs: `git add TESTS.md DECISIONS.md handovers/03-to-04-measure.md && git commit -m "S03 results and handover"`.
- Export this chat unedited to `chats/03-walking-skeleton.md` and commit it.
- E3: Zoom call with 2–3 Armenian-speaking friends, consent to record, Zoom audio settings from the S02 handover (Speaker = Multi-Output Device; Microphone = MacBook Pro Microphone; auto-adjust off). Record with `eval/record.py session`.
- Optional: record 3–4 long E1 sentences (15+ words).
- Before pushing: the G03 check from the S02 handover must print nothing.

## Next step

Open chat 04 with this handover and the four docs: confirm the normalisation, check B's licences and sizes, then write the scoring scripts while E3 is being transcribed.
