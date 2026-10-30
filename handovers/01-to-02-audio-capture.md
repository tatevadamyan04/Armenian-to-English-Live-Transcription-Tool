# Handover: session 01 (plan) → session 02 (audio capture)

Attach to the first message of chat 02, together with CLAUDE.md, PLAN.md, DECISIONS.md and TESTS.md.

## Goal of session 02

Capture the MacBook mic and the system audio at the same time on Tatev's Mac, with device selection, and record the first evaluation set (E1). Module M1 only (PLAN.md §5), plus the recorder from M8.

## State after session 01 (27 Oct 2026)

- Planning is done: PLAN.md, DECISIONS.md (D1–D13), TESTS.md and CLAUDE.md are in the repository.
- No code yet. Nothing is installed for the project on the Mac (no BlackHole, no venv).
- Decisions in force for this session:
  - D1 — Python 3.14.5 in a venv.
  - D2 — BlackHole 2ch + Multi-Output Device. The alternative (catap) is used only if T04/T05 fail.
  - D3 — AirPods for listening, MacBook mic as input.
  - D8 — threads + queues, one inference worker.
  - D10 — permissive licences only.
  - D11 — Claude sends files; Tatev saves, commits and runs commands, and pastes the output back.
- Confirmed on 30 Oct: D5 and D6 keep the A-vs-B comparison in S04; Tatev uses the built-in Terminal app; D1's reason is recorded.

## Plan for session 02 (~2.5 h)

Start by agreeing a compressed schedule: PLAN.md §9 assumed this session on Tue 27 Oct, and the deadline is Sun 1 Nov. Use the cut list in §9.

1. Install BlackHole 2ch from Existential Audio's official installer; restart if asked.
2. In Audio MIDI Setup, create the Multi-Output Device with BlackHole 2ch and the AirPods. Settle the clock device. BlackHole's FAQ says the built-in output should be enabled and listed first in a Multi-Output Device, and that AirPods shouldn't be the clock device. Try BlackHole 2ch as the clock with drift correction on the AirPods first, then record in D2 what works.
3. Zoom: speaker = Multi-Output Device, microphone = MacBook Pro Microphone. Allow Terminal to use the microphone when macOS asks.
4. Create the venv with Python 3.14.5. Add `.gitignore` (data/, models/, audio and weight files, .env). Check the licences of python-sounddevice and numpy (D10), install them and pin the versions in `requirements.txt`.
5. Write M1 (`src/hyen/audio.py`: device list, two input streams, mono 16 kHz chunks with timestamps) and the recorder (`eval/record.py`: writes `mic.wav` and `system.wav`).
6. Run T01–T05 and T13. If BlackHole crackles, drops out or stays silent: stop and switch to D2-B, as decided.
7. Record E1: the ~25 sentences through the mic into `data/e1/`; references in `eval/refs/e1.tsv`.
8. Commit without audio (check G03). Write `handovers/02-to-03-skeleton.md`.

## Open questions for session 02

- Which device is the Multi-Output clock, and is drift correction enough with the AirPods?
- The devices' native sample rates (probably 48 kHz) and how to resample to 16 kHz — choose a method with an allowed licence.
- Timestamps: PortAudio's stream time or `time.monotonic()` in the callback — choose one and document it.
- Does BlackHole behave on macOS 26.5.2 (T04/T05)?
- How the package runs so that `python -m hyen` works: `pip install -e .` with a minimal `pyproject.toml`, or `PYTHONPATH=src`. Pick the simpler one.

## Exit criteria

T01–T05 and T13 pass; E1 recorded with references; `requirements.txt` pinned; handover 02 → 03 written.

## Not in session 02

Phrase detection, ASR, translation, speaker labels (sessions 03–05).

## Before session 02 (Tatev)

- Write the ~25 E1 sentences and your English translations: everyday call phrases, short and long, no digits or abbreviations, no online translator.
- Ask 2–3 Armenian-speaking friends for a ~15 min Zoom call (recorded for E3, with their consent) and a ~20 min rehearsal. The original Wed 28 Oct slot has passed, so set new times with the compressed schedule.
- Export chat 01 to `chats/01-brief-review-and-plan.md`.
