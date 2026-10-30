# CLAUDE.md — project rules

Read this first in every chat. The plan is in PLAN.md, the decisions in DECISIONS.md, the tests in TESTS.md, and the latest state in the newest file in `handovers/`.

## The project

A Python terminal program for macOS that captures the MacBook mic and the system audio (Zoom, via BlackHole) at the same time, transcribes Armenian, translates it into English and labels speakers ("Me" / "Speaker N"), each line within 5 s, with local models only. Mathrix AI coding challenge · Tatev Adamyan · deadline 1 Nov 2026 · ≈ 20 h budget.

## How we work

- Tatev is the architect and decides; Claude researches, proposes and implements. For every real choice: 2–3 options with trade-offs and source links → Tatev picks → log it in DECISIONS.md with Claude's recommendation and Tatev's choice.
- Logged decisions stand. Reopen one only with new evidence, by proposing a revision — never by quietly doing something else.
- One module per chat (PLAN.md §9). Agree the module's inputs and outputs before writing code. Keep the context small and relevant.
- End every chat with a handover in `handovers/NN-to-MM-<topic>.md`: goal, current state, decisions made, open questions, next step.
- Workflow (D11): Claude can't run anything on the Mac. Claude writes files and sends them; Tatev saves them into the repo, commits and runs commands. With every command, say what it does and what output to expect, and ask for the exact output back.
- Every chat is exported unedited to `chats/` and read by Mathrix. No other AI assistant is used on this project.

## Scope

- Build only what the brief asks (PLAN.md §2). Out of scope: GUI, installer, echo handling, naming or remembering speakers, saving transcripts, non-Armenian input. If something seems needed but isn't in the brief, ask first.
- Spare time goes into the reliability of the core, not into features.

## Code

- **Fail loudly.** No silent fallbacks: no `except` that hides an error, no automatic substitution of another device or model, no default that masks missing configuration. No retries or timeouts unless DECISIONS.md says so. Error messages name the cause and the fix.
- An exception in any thread stops the whole program with a traceback and a non-zero exit code.
- **Local only at runtime:** the program sets `HF_HUB_OFFLINE=1`; models are fetched only by `scripts/download_models.py`; no network calls in `src/`.
- stdout carries only the transcript lines (layout in PLAN.md §5); diagnostics and timings go to stderr.
- Python 3.14.5 in a venv (D1); exact versions in `requirements.txt`. If a package won't install on 3.14, stop and raise it — no source builds or version hacks.
- Keep it simple: modules match PLAN.md §5; small functions; type hints; dataclasses for the messages between modules; no frameworks, plugin systems or config layers we don't need.
- Code and docs in English; Armenian only in data and test strings (UTF-8).

## Facts, models and licences

- Never invent APIs, parameters, model names or numbers. Give a source for claims about models and libraries; say "unverified" when something hasn't been checked.
- Check the licence of every model and library before proposing it (D10: MIT, Apache-2.0, BSD, CC-BY-4.0 only).
- Measure, don't guess: model choices are confirmed with the evaluation sets in TESTS.md §3.

## Tests

- Tests come from the requirements (TESTS.md), not from the code just written. New test ideas go into TESTS.md.
- After each module: run its TESTS.md cases and the QA questions in TESTS.md §6; record the results.

## Git

- Never commit model weights, audio recordings, or tokens and keys. `.gitignore` covers `data/`, `models/`, audio and weight files and `.env`. Check `git ls-files` before pushing.
- Commit after every working step; change one thing at a time.
- If the same problem fails three times: stop. Go back to the last working commit, cut the problem down to a minimal example, or rethink the approach.

## The Mac (D2, D3)

- MacBook Pro, Apple M4 Pro, 24 GB, macOS 26.5.2, Python 3.14.5, the built-in Terminal app.
- Program inputs: "MacBook Pro Microphone" (source `mic`) and "BlackHole 2ch" (source `system`); exact device names are confirmed by T01.
- Zoom: speaker = Multi-Output Device (BlackHole 2ch + AirPods 4), microphone = MacBook Pro Microphone. Tatev listens on the AirPods only.
