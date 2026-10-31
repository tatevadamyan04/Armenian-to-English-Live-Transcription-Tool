"""M7 · the live pipeline (PLAN.md §4–5, D8).

Threads, each started by Pipeline.run():

    capture reader (M1, one per source) → segmenter thread (M2, one per source)
        → one queue → inference worker: ASR (M3) → label → translation (M5) → print (M6)

- Every model is loaded before the session clock starts (run_live), so loading time
  never counts as latency.
- Labels: "Me" for the microphone (R6). Until M4 exists (S05), every call-side segment
  is "Speaker 1" (D21); a startup line on stderr says so.
- A segment whose ASR text is empty is dropped with a note on stderr (PLAN.md §5, M3).
- One timing line per segment goes to stderr. Latency = print time − end of speech, as in
  PLAN.md §5; it includes the silence wait that ends a phrase.
- An exception in any thread stops the whole program: the main thread re-raises it, with
  the thread's name added, so Python prints the traceback and exits with code 1 (F04).
- Ctrl+C stops at once; a phrase still in progress and queued segments are not printed.
"""

from __future__ import annotations

import queue
import sys
import threading
import time
from collections.abc import Callable
from typing import Protocol, TextIO

import numpy as np

from hyen.output import Result, print_result
from hyen.segmenter import Segment, Segmenter

LATENCY_LIMIT = 5.0  # C1
SYSTEM_PLACEHOLDER = "Speaker 1"  # D21, until M4 (S05)


class ChunkSource(Protocol):
    def get(self, name: str): ...  # an AudioChunk; raises CaptureError if the stream failed


class Recognizer(Protocol):
    def transcribe(self, samples: np.ndarray) -> str: ...


class Translator(Protocol):
    def translate(self, hy: str) -> str: ...


def label_for(segment: Segment) -> str:
    return "Me" if segment.source == "mic" else SYSTEM_PLACEHOLDER


class Pipeline:
    def __init__(
        self,
        capture: ChunkSource,
        segmenters: dict[str, Segmenter],
        recognizer: Recognizer,
        translator: Translator,
        clock: Callable[[], float],
        t0: float,
        out: TextIO = sys.stdout,
        log: Callable[[str], None] | None = None,
    ) -> None:
        self.capture = capture
        self.segmenters = segmenters
        self.recognizer = recognizer
        self.translator = translator
        self.clock = clock
        self.t0 = t0
        self.out = out
        self.log = log if log is not None else (lambda message: print(message, file=sys.stderr, flush=True))
        self.segments: queue.Queue[Segment] = queue.Queue()
        self.errors: queue.Queue[tuple[str, BaseException]] = queue.Queue()

    def now(self) -> float:
        return self.clock() - self.t0

    def run(self) -> None:
        """Start the threads and wait. Returns only by raising: a thread's error, or KeyboardInterrupt."""
        for source in self.segmenters:
            self._start(f"segmenter-{source}", self._segment_loop, source)
        self._start("worker", self._worker_loop)
        name, error = self.errors.get()
        error.add_note(f"(raised in thread {name!r}; the program stops — F04)")
        raise error

    def _start(self, name: str, target: Callable[..., None], *args: object) -> None:
        def guarded() -> None:
            try:
                target(*args)
            except BaseException as error:  # noqa: BLE001 — handed to the main thread, which raises it
                self.errors.put((name, error))

        threading.Thread(target=guarded, name=name, daemon=True).start()

    def _segment_loop(self, source: str) -> None:
        segmenter = self.segmenters[source]
        while True:
            chunk = self.capture.get(source)
            segment = segmenter.feed(chunk, self.now())
            if segment is not None:
                self.segments.put(segment)

    def _worker_loop(self) -> None:
        while True:
            segment = self.segments.get()
            self._process(segment)

    def _process(self, segment: Segment) -> None:
        started = self.now()
        hy = self.recognizer.transcribe(segment.samples)
        recognised = self.now()
        if not hy:
            self.log(f"{segment.seg_id}: ASR returned no text; segment dropped")
            return
        en = self.translator.translate(hy)
        translated = self.now()
        result = Result(segment.seg_id, segment.t_speech_start, label_for(segment), hy, en)
        print_result(result, self.out)
        printed = self.now()
        latency = printed - segment.t_speech_end
        self.log(
            f"timing {segment.seg_id}: speech {segment.t_speech_end - segment.t_speech_start:.2f} s"
            f"{' (cut)' if segment.cut else ''} · end-detect {segment.t_end_detected - segment.t_speech_end:.2f}"
            f" · queue {started - segment.t_end_detected:.2f} · asr {recognised - started:.2f}"
            f" · mt {translated - recognised:.2f} · latency {latency:.2f} s"
            f"{'  ** OVER 5 s (C1) **' if latency > LATENCY_LIMIT else ''}"
        )


def run_live(mic, system) -> int:
    """Load every model, start the session clock and capture, run until Ctrl+C or an error."""
    from hyen import audio
    from hyen.asr import Recognizer as AsrModel
    from hyen.modelfiles import SILERO_VAD
    from hyen.translate import Translator as MtModel
    from hyen.vad import SileroVad

    def log(message: str) -> None:
        print(message, file=sys.stderr, flush=True)

    log(f"mic: {audio.describe(mic)}  ·  system: {audio.describe(system)}  → 16 kHz mono")
    started = time.perf_counter()
    vads = {"mic": SileroVad(SILERO_VAD), "system": SileroVad(SILERO_VAD)}
    recognizer = AsrModel()
    translator = MtModel()
    log(f"models loaded in {time.perf_counter() - started:.1f} s")
    log(f"note: speaker labels are not implemented yet; every call-side segment is '{SYSTEM_PLACEHOLDER}' (D21)")
    segmenters = {name: Segmenter(name, vads[name], log=log) for name in ("mic", "system")}
    with audio.Capture({"mic": mic, "system": system}) as capture:
        t0 = time.monotonic()
        capture.start(t0)
        log("Listening. Ctrl+C stops.")
        pipeline = Pipeline(capture, segmenters, recognizer, translator, time.monotonic, t0, log=log)
        try:
            pipeline.run()
        except KeyboardInterrupt:
            log("Stopped.")
    return 0
