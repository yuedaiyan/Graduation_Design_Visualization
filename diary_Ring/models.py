from __future__ import annotations

from dataclasses import dataclass


@dataclass
class Entry:
    stem: str
    date: str
    content: str


@dataclass
class WindowMetric:
    idx: int
    start: int
    end: int
    size: int
    chars: int
    focus: float
    looseness: float
    novelty: float
    text: str


@dataclass
class RingSegment:
    idx: int
    chars: int
    size: int
    focus: float
    looseness: float
    novelty: float
    source_window_indices: list[int]


@dataclass
class EmotionProfile:
    primary: str
    arc: str
