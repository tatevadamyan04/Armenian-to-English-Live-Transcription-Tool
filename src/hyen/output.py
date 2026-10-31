"""M6 · transcript output (PLAN.md §5, R9, T12).

Each segment is printed on stdout as three lines and flushed at once:

    [00:01:12] Speaker 1
      HY: Բարև, ինչպե՞ս ես։
      EN: Hi, how are you?

The time is the segment's speech start, counted from the session start. stdout carries
only these lines; diagnostics and timings go to stderr.
"""

from __future__ import annotations

import sys
from dataclasses import dataclass, field
from typing import TextIO


@dataclass(frozen=True)
class Result:
    seg_id: str
    t_speech_start: float  # seconds since the session started
    label: str  # "Me" or "Speaker N"
    hy: str
    en: str
    timings: dict[str, float] = field(default_factory=dict)  # stage → seconds, for the stderr log


def clock(seconds: float) -> str:
    """Session time as HH:MM:SS."""
    whole = int(seconds)
    return f"{whole // 3600:02d}:{whole // 60 % 60:02d}:{whole % 60:02d}"


def format_result(result: Result) -> str:
    return f"[{clock(result.t_speech_start)}] {result.label}\n  HY: {result.hy}\n  EN: {result.en}\n"


def print_result(result: Result, out: TextIO = sys.stdout) -> None:
    out.write(format_result(result))
    out.flush()
