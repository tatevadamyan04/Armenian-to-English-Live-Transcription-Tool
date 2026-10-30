# TESTS

Test cases come from the brief's requirements (IDs in [PLAN.md §2](PLAN.md)) and from the project rules in [CLAUDE.md](CLAUDE.md) — not from the code. Run the full list before submitting and again before the interview. Write results into the Result column with the date.

**Must** = brief requirement; failing it blocks submission unless the README documents it · **Measure** = numbers for the README; the brief sets no threshold · **Rule** = from CLAUDE.md

*From* = the first session in which the test can run (PLAN.md §9).

## 1. Requirement tests

| ID | Req | Type | Procedure | Pass | From | Result |
|---|---|---|---|---|---|---|
| T01 | R3 | Must | Run `python -m hyen --list-devices` | Every input device printed with index and name, including "MacBook Pro Microphone" and "BlackHole 2ch"; exit code 0 | S02 | Pass 2026-10-30: 4 inputs listed, incl. MacBook Pro Microphone (1 ch, 48 kHz) and BlackHole 2ch (2 ch, 48 kHz); exit 0 |
| T02 | R3 | Must | Start with `--mic "MacBook Pro Microphone" --system "BlackHole 2ch"` | Startup line names both devices; the session runs | S02 | Pass 2026-10-30: startup line names [3] MacBook Pro Microphone and [2] BlackHole 2ch; level lines every 2 s (mic −51 to −25 dBFS) |
| T03 | R3, Rule | Must | Start with `--mic "No Such Mic"` | Error names the unknown device and lists the available ones; exit code ≠ 0; no default device used | S02 | Pass 2026-10-31: "no input device 'No Such Mic'" with the 4 input devices listed; exit 2; no stream opened |
| T04 | R1 | Must | Play Armenian speech through Zoom (or another app) while Tatev speaks; record 60 s of both sources with the recorder | Both files hold the right speech; each source's clock gap within ± 0.1 s and 0 overflows (recorder summary); no crackle or dropouts | S02 | Pass 2026-10-31 (official 60-s run): clock gaps −0.003 s (mic) and −0.013 s (system), 0 overflows, 0 clipped, both files 60.00 s, first audio 11 ms apart; right speech in each file and no crackle or gaps (Tatev, listening). History: content right in all runs: right speech in each file, no crackle or gaps heard, 0 overflows, 0 clipped. Lengths: (1) callback design: the mic lost 0.13–0.37% of its samples without an overflow flag (clock gap −0.080 s after 60 s, −0.209 s after 120 s, −0.662 s after 180 s; BlackHole exact) → D17 reader threads; (2) with D17, 120 s: 0 overflows, equal clock gaps (+0.222 s mic, +0.216 s system) from a timestamp error → D15 correction; length difference 0.192 s from the timing of the streams' start and stop, not lost audio; (3) after the D15 correction, 120 s: clock gaps +0.002 s and +0.009 s, both streams' first audio 5 ms apart, 0 overflows; length difference 0.133 s because the recorder cut only the mic file → criterion revised (D18) |
| T05 | R2 | Must | During T04, listen on the AirPods | Call audio heard clearly through the Multi-Output Device | S02 | Pass 2026-10-30: the video heard clearly on the AirPods during T04 (Multi-Output Device: BlackHole 2ch as clock at 48 kHz, drift correction on the AirPods) |
| T06 | R1 | Must | Run the program while the call side and Tatev both speak | "Me" and "Speaker N" segments appear in the same session | S03 | |
| T07 | R4 | Must + Measure | E1 replay and live reading | Armenian script on every HY line; CER/WER recorded (§3) | S03 | |
| T08 | R5 | Must + Measure | Any run | Every printed segment with HY text has a non-empty EN line; rating and chrF recorded (§3) | S03 | |
| T09 | R6 | Must | Live: Tatev speaks into the mic | Every mic segment is labelled exactly "Me", never "Speaker N" | S03 | |
| T10 | R7 | Must + Measure | E3 replay (2–3 call-side speakers) | Each real speaker's most frequent label differs from the others'; consistency, purity and labels created recorded (§3) | S05 | |
| T11 | R8 | Must + Measure | E3 replay and rehearsal (P01), ≥ 5 min | A speaker heard early and again ≥ 3 min later carries the same label; consistency recorded | S05 | |
| T12 | R9 | Must | Any run | Each segment printed as `[HH:MM:SS] label`, `  HY: …`, `  EN: …`; time = speech start since session start; segments appear one by one during the conversation | S03 | |
| T13 | R9, C4 | Must | Print a line with ՞ ։ ՝ and Armenian letters in Terminal | All characters display correctly | S02 | Pass 2026-10-30: "Բարև, ինչպե՞ս ես։ Լավ եմ՝ շնորհակալություն։ և" printed exactly in Terminal; stdout encoding utf-8 |
| T14 | C1 | Must + Measure | E1 replay, E3 replay and rehearsal, with the timing log on stderr | Latency ≤ 5.0 s for every segment; median, p95 and max recorded | S03 | |
| T15 | C1 | Must | E3 replay through BlackHole while Tatev reads E1 sentences into the mic | Latency ≤ 5.0 s for every segment with both sources active | S04 | |
| T16 | C1 | Must | 30 s of continuous speech with no pause | Speech is split at the segment cap; each piece is printed ≤ 5.0 s after it ends | S03 | |
| T17 | C2 | Must | Models downloaded; Wi-Fi off; run T06 | Runs normally; no download attempts | S06 | |
| T18 | C4 | Must | Repeat T14 unplugged, on battery, Low Power Mode off | Latency ≤ 5.0 s for every segment | S06 | |
| T19 | C3 | Must | Code review | The program is Python; dependencies install with pip | S06 | |
| T20 | R4 | Must | 60 s of room silence on both inputs | Nothing printed | S03 | |
| T21 | R4 | Measure | 60 s of typing, mouse clicks and a notification sound | Number of segments printed and their text, reported in the README | S06 | |
| T22 | out of scope | Must | Code review | No GUI, installer, echo handling, speaker names or persistence, transcript saving, or other input languages | S06 | |
| T23 | R3, Rule | Must | Start with `--system "BlackHole 2ch"` and no `--mic` | Usage error naming the missing flag; exit code 2; no stream opened | S02 | Pass 2026-10-31: "--mic is required" with the usage line; exit 2; no stream opened |

T04's pass criterion was revised on 2026-10-31 (D18): "equal length ± 0.1 s" became "each source's clock gap within ± 0.1 s and 0 overflows".

## 2. Fail-loud tests

| ID | Rule | Procedure | Pass | From | Result |
|---|---|---|---|---|---|
| F01 | no silent failure | Remove Terminal's microphone permission, then start | Within 10 s, stderr says the mic delivers only digital silence and names the likely cause (permission). Mic only: BlackHole legitimately delivers zeros while nothing plays | S06 | |
| F02 | no silent failure | Disconnect the AirPods (breaking the Multi-Output Device) mid-session | A visible error; the program doesn't carry on with a dead source | S06 | |
| F03 | local only, fail loudly | Remove one model from the cache; start (the program sets `HF_HUB_OFFLINE=1`) | Error names the model and the download command; exit code ≠ 0 | S03 | |
| F04 | no dead threads | Unit test: a stage raises an exception | The whole program stops with a traceback and exit code ≠ 0 | S03 | |
| F05 | no hidden backlog | Long E3 replay with both sources | No segment dropped; every segment's latency appears in the timing log | S04 | |

## 3. Measurement protocol

### 3.1 Evaluation sets (D9)

| Set | Content | Size | References | Files |
|---|---|---|---|---|
| E1 read | Tatev reads everyday call phrases through M1 (mic) | ~25 sentences, short and long, no digits or abbreviations | Armenian text and English translation, both written by Tatev — no online translator | audio `data/e1/` (ignored) · refs `eval/refs/e1.tsv` |
| E2 FLEURS | google/fleurs `hy_am` test split | ~30 clips | dataset transcriptions; English only if the parallel English sentences can be matched (check in S04) | `data/e2/` · `eval/refs/e2.tsv` |
| E3 call | Zoom call: 2–3 Armenian speakers on the call side + Tatev, recorded through M1 with everyone's consent | 5–10 min; ~3 min transcribed | Armenian transcript with speaker turns and times; English for ~15 segments | `data/e3/` · `eval/refs/e3.tsv` |

### 3.2 Text normalisation

Applied identically to hypothesis and reference before CER/WER:

1. Unicode NFKC — this also turns the ligature "և" into "եւ", so both spellings match.
2. Lowercase.
3. Replace punctuation with a space: Armenian ։ ՝ ՞ ՛ ՜ « » ֊ and ASCII . , ? ! : ; " ' ( ) - …
4. Collapse whitespace.

Numbers are not normalised, which is why E1 avoids digits.

### 3.3 Metrics

- **ASR:** CER (the main metric, since Armenian words are long) and WER (comparable with ArmBench), per set.
- **Translation:** Tatev rates each segment — 1 wrong or misleading, 2 understandable with errors, 3 correct — plus chrF against her references.
- **Latency:** per segment, print time − speech end on the session clock; median, p95 and max; per-stage times from the stderr log.
- **Speakers (E3):** consistency = for each real speaker, the share of their segments carrying their most frequent label; purity = for each label, the share of its segments from its most frequent speaker; labels created vs real speakers.

### 3.4 Sanity check

| ID | Procedure | Pass | Result |
|---|---|---|---|
| M01 | Score model A on E2 with our scripts | WER within ~5 points of ArmBench's FLEURS 15.00%; if not, find out why (normalisation, decoder, audio path) before comparing models. ArmBench doesn't publish its normalisation, so a small gap is expected | |

### 3.5 Model selection rule (S04; D5 and D6)

- **ASR:** lowest CER on E3, then on E1, within the latency budget (PLAN.md §5).
- **Translation:** highest mean rating on E1 + E3, chrF as tie-breaker, within the budget.
- Record the numbers and the outcome in DECISIONS.md.

## 4. Deliverables and git hygiene

| ID | Req | Check | Pass | Result |
|---|---|---|---|---|
| G01 | DL2, C5 | README sections | Setup incl. BlackHole, Multi-Output, Zoom settings, macOS permissions · models, alternatives, reasons (from DECISIONS.md) · architecture · known limitations and improvements · licence credits (D10) | |
| G02 | DL2 | Fresh install: follow the README from a fresh clone and a new venv | Program runs; T06 passes | |
| G03 | DL4 | `git ls-files` | No audio (.wav .flac .mp3 .m4a), no weights (.onnx .bin .safetensors .nemo .pt or model folders), no tokens (search "hf_") | |
| G04 | DL3 | `chats/` | One unedited .md per session, matching the sessions in handovers/ | |
| G05 | DL1 | Repository access | Reviewers can open the private repository | |

## 5. Interview readiness

| ID | Req | Check | Pass | Result |
|---|---|---|---|---|
| P01 | I1 | Rehearsal: Zoom with ≥ 2 Armenian speakers on the call side, ≥ 5 min, on battery | T06–T14 pass | |
| P02 | I4 | Pre-flight on the day: AirPods charged and connected · system output = Multi-Output Device · Zoom speaker = Multi-Output Device, Zoom mic = MacBook Pro Microphone · Terminal has microphone permission · `--list-devices` shows both devices · models present · 30 s check with both sources | All ticked | |
| P03 | I3 | Original chats still in the app; exports in `chats/` | Both | |
| P04 | I2 | Walkthrough without notes: each module (PLAN.md §5) and each decision (DECISIONS.md) | Tatev can explain all of them | |

## 6. Critical QA questions

Ask at the end of every module session and before submitting. Answers per module are in that session's handover (M1: `handovers/02-to-03-skeleton.md`).

- Does any code path swallow an exception, retry, time out or substitute a default without saying so?
- Can anything be downloaded or sent over the network at runtime?
- Is every model and library licence allowed under D10 and credited in the README?
- Is every claim in the README backed by a measurement here or by a source?
- Has every "unverified" item been verified or listed as a limitation?
- Is there any feature the brief didn't ask for?
- Which test failed last, and is that documented?
