from __future__ import annotations

import math
from pathlib import Path
from typing import Any

import numpy as np
from PIL import ImageDraw

from config import RING_COLOR, TAU
from models import RingSegment, WindowMetric
from utils import clamp01, rgba


def draw_arc(
    draw: ImageDraw.ImageDraw,
    center: float,
    radius: float,
    start: float,
    end: float,
    width: int,
    fill: tuple[int, int, int, int],
) -> None:
    if end <= start:
        return
    box = [center - radius, center - radius, center + radius, center + radius]
    if end - start >= 2 * math.pi - 0.01:
        draw.ellipse(box, outline=fill, width=width)
        return
    draw.arc(box, math.degrees(start), math.degrees(end), fill=fill, width=width)


def draw_fragmented_arc(
    draw: ImageDraw.ImageDraw,
    center: float,
    radius: float,
    start: float,
    end: float,
    width: int,
    fill: tuple[int, int, int, int],
    rng: np.random.Generator,
    continuity: float,
) -> None:
    span = end - start
    if span <= 0:
        return
    if continuity > 0.72:
        draw_arc(draw, center, radius, start, end, width, fill)
        return

    pieces = max(5, int(18 + span / (2 * math.pi) * 42))
    cursor = start
    for _ in range(pieces):
        chunk = span / pieces * rng.uniform(0.62, 1.24)
        gap = span / pieces * rng.uniform(0.12, 0.55 + (1.0 - continuity) * 0.9)
        seg_start = cursor + rng.uniform(0.0, gap * 0.35)
        seg_end = min(end, seg_start + chunk * (0.55 + 0.42 * continuity))
        if seg_end > seg_start:
            draw_arc(draw, center, radius, seg_start, seg_end, width, fill)
        cursor += chunk + gap
        if cursor >= end:
            break


def sector_spans(
    metrics: list[WindowMetric], rng: np.random.Generator, start_angle: float
) -> list[tuple[WindowMetric, float, float]]:
    if len(metrics) == 1:
        return [(metrics[0], start_angle, start_angle + 2 * math.pi - 0.002)]

    weights = np.array([m.chars for m in metrics], dtype=np.float64)
    weights = (
        np.sqrt(weights) + np.array([m.size for m in metrics], dtype=np.float64) * 0.45
    )
    weights = weights / weights.sum()

    gap = math.radians(1.4 if len(metrics) <= 4 else 0.9)
    usable = 2 * math.pi - gap * len(metrics)
    spans: list[tuple[WindowMetric, float, float]] = []
    angle = start_angle + rng.uniform(-0.08, 0.08)
    for metric, weight in zip(metrics, weights):
        span = max(math.radians(8), float(weight) * usable)
        spans.append((metric, angle, angle + span))
        angle += span + gap
    return spans


def point_on_circle(
    center: float, radius: float, theta: float, radial_offset: float = 0.0
) -> tuple[float, float]:
    r = radius + radial_offset
    return center + math.cos(theta) * r, center + math.sin(theta) * r


def draw_square(
    draw: ImageDraw.ImageDraw,
    x: float,
    y: float,
    size: float,
    fill: tuple[int, int, int, int],
) -> None:
    half = size / 2.0
    draw.rectangle([x - half, y - half, x + half, y + half], fill=fill)


class PillowRingTarget:
    def __init__(self, draw: ImageDraw.ImageDraw) -> None:
        self.draw = draw

    def arc(
        self,
        center: float,
        radius: float,
        start: float,
        end: float,
        width: int,
        fill: tuple[int, int, int, int],
    ) -> None:
        draw_arc(self.draw, center, radius, start, end, width, fill)

    def square(
        self,
        x: float,
        y: float,
        size: float,
        fill: tuple[int, int, int, int],
    ) -> None:
        draw_square(self.draw, x, y, size, fill)


def color_hex(fill: tuple[int, int, int, int]) -> str:
    return f"#{fill[0]:02x}{fill[1]:02x}{fill[2]:02x}"


def opacity(fill: tuple[int, int, int, int]) -> str:
    return f"{fill[3] / 255.0:.6g}"


class SvgRingTarget:
    def __init__(self, size_px: int, source_canvas: int) -> None:
        self.size_px = size_px
        self.scale = size_px / source_canvas
        self.elements: list[str] = []

    def _point(self, center: float, radius: float, theta: float) -> tuple[float, float]:
        x = (center + math.cos(theta) * radius) * self.scale
        y = (center + math.sin(theta) * radius) * self.scale
        return x, y

    def arc(
        self,
        center: float,
        radius: float,
        start: float,
        end: float,
        width: int,
        fill: tuple[int, int, int, int],
    ) -> None:
        if end <= start:
            return
        stroke = color_hex(fill)
        stroke_width = width * self.scale
        stroke_opacity = opacity(fill)
        cx = center * self.scale
        cy = center * self.scale
        r = radius * self.scale
        if end - start >= 2 * math.pi - 0.01:
            self.elements.append(
                f'<circle cx="{cx:.3f}" cy="{cy:.3f}" r="{r:.3f}" '
                f'fill="none" stroke="{stroke}" stroke-opacity="{stroke_opacity}" '
                f'stroke-width="{stroke_width:.3f}" />'
            )
            return

        x1, y1 = self._point(center, radius, start)
        x2, y2 = self._point(center, radius, end)
        large_arc = 1 if end - start > math.pi else 0
        self.elements.append(
            f'<path d="M {x1:.3f} {y1:.3f} '
            f'A {r:.3f} {r:.3f} 0 {large_arc} 1 {x2:.3f} {y2:.3f}" '
            f'fill="none" stroke="{stroke}" stroke-opacity="{stroke_opacity}" '
            f'stroke-width="{stroke_width:.3f}" />'
        )

    def square(
        self,
        x: float,
        y: float,
        size: float,
        fill: tuple[int, int, int, int],
    ) -> None:
        side = size * self.scale
        half = side / 2.0
        sx = x * self.scale - half
        sy = y * self.scale - half
        self.elements.append(
            f'<rect x="{sx:.3f}" y="{sy:.3f}" width="{side:.3f}" '
            f'height="{side:.3f}" fill="{color_hex(fill)}" '
            f'fill-opacity="{opacity(fill)}" />'
        )

    def save(self, path: Path) -> None:
        path.write_text(
            "\n".join(
                [
                    '<?xml version="1.0" encoding="UTF-8"?>',
                    (
                        f'<svg xmlns="http://www.w3.org/2000/svg" '
                        f'width="{self.size_px}" height="{self.size_px}" '
                        f'viewBox="0 0 {self.size_px} {self.size_px}">'
                    ),
                    *self.elements,
                    "</svg>",
                    "",
                ]
            ),
            encoding="utf-8",
        )


def smoothstep(x: float) -> float:
    x = float(clamp01(x))
    return x * x * (3.0 - 2.0 * x)


def merge_metrics(metrics: list[WindowMetric]) -> list[RingSegment]:
    return [
        RingSegment(
            idx=i,
            chars=m.chars,
            size=m.size,
            focus=m.focus,
            looseness=m.looseness,
            novelty=m.novelty,
            source_window_indices=[m.idx],
        )
        for i, m in enumerate(metrics)
    ]


def segment_spans(
    segments: list[RingSegment], start_angle: float
) -> list[tuple[RingSegment, float, float]]:
    if len(segments) == 1:
        return [(segments[0], start_angle, start_angle + TAU)]

    weights = np.array([max(1, s.chars) for s in segments], dtype=np.float64)
    weights = weights / weights.sum()

    spans: list[tuple[RingSegment, float, float]] = []
    angle = start_angle
    for segment, weight in zip(segments, weights):
        span = float(weight) * TAU
        spans.append((segment, angle, angle + span))
        angle += span

    overflow = spans[-1][2] - (spans[0][1] + TAU)
    if abs(overflow) > 1e-9:
        last, start, end = spans[-1]
        spans[-1] = (last, start, end - overflow)
    return spans


def interpolated_state(
    theta: float,
    spans: list[tuple[RingSegment, float, float]],
    smoothing_strength: float,
) -> tuple[float, float, float]:
    if len(spans) == 1:
        segment = spans[0][0]
        return segment.focus, segment.looseness, segment.novelty

    centers = np.array(
        [(start + end) * 0.5 for _, start, end in spans], dtype=np.float64
    )
    focus = np.array([segment.focus for segment, _, _ in spans], dtype=np.float64)
    looseness = np.array(
        [segment.looseness for segment, _, _ in spans], dtype=np.float64
    )
    novelty = np.array([segment.novelty for segment, _, _ in spans], dtype=np.float64)

    x = ((theta - centers[0]) % TAU) + centers[0]
    ext_centers = np.concatenate([centers, [centers[0] + TAU]])
    idx = int(np.searchsorted(ext_centers, x, side="right") - 1)
    idx = max(0, min(idx, len(spans) - 1))
    nxt = (idx + 1) % len(spans)
    left = ext_centers[idx]
    right = ext_centers[idx + 1]
    t = smoothstep((x - left) / max(1e-9, right - left))

    first_start = spans[0][1]
    x_by_span = ((theta - first_start) % TAU) + first_start
    hard = spans[-1][0]
    for segment, start, end in spans:
        if start <= x_by_span < end:
            hard = segment
            break
    soft = (
        float(focus[idx] * (1.0 - t) + focus[nxt] * t),
        float(looseness[idx] * (1.0 - t) + looseness[nxt] * t),
        float(novelty[idx] * (1.0 - t) + novelty[nxt] * t),
    )
    strength = float(clamp01(smoothing_strength))
    return (
        float(hard.focus * (1.0 - strength) + soft[0] * strength),
        float(hard.looseness * (1.0 - strength) + soft[1] * strength),
        float(hard.novelty * (1.0 - strength) + soft[2] * strength),
    )


def apply_visual_impact(
    focus: float, looseness: float, novelty: float, impact: float
) -> tuple[float, float, float]:
    impact = max(0.0, impact)
    return (
        float(clamp01(1.0 - (1.0 - focus) * impact)),
        float(clamp01(looseness * impact)),
        float(clamp01(novelty * impact)),
    )


def draw_ring_content(
    target: Any,
    metrics: list[WindowMetric],
    summary: dict[str, float],
    distinctiveness: float,
    seed: int,
    canvas: int,
    particle_multiplier: float,
    visual_impact: float,
    smoothing_strength: float,
) -> tuple[list[RingSegment], int]:
    rng = np.random.default_rng(seed)
    center = canvas / 2.0
    radius = canvas * (0.335 + 0.018 * distinctiveness)
    base_width = max(
        8, int(canvas * (0.0075 + 0.0025 * (1.0 - summary["global_looseness"])))
    )

    start_angle = -math.pi / 2.0
    segments = merge_metrics(metrics)
    spans = segment_spans(segments, start_angle)

    total_particles = 0
    mark = rgba(RING_COLOR, 1.0)

    for segment, start, end in spans:
        focus, looseness, _ = apply_visual_impact(
            segment.focus, segment.looseness, segment.novelty, visual_impact
        )
        if focus < 0.94 or looseness > 0.16:
            continue
        span = end - start
        pad = span * (0.18 if span < TAU - 1e-6 else 0.08)
        solid_start = start + pad
        solid_end = end - pad
        if solid_end <= solid_start:
            solid_start, solid_end = start, end
        target.arc(
            center,
            radius,
            solid_start,
            solid_end,
            max(3, int(base_width * 1.34)),
            mark,
        )

    line_steps = 960
    for i in range(line_steps):
        theta1 = start_angle + TAU * i / line_steps
        theta2 = start_angle + TAU * (i + 1) / line_steps
        mid = (theta1 + theta2) * 0.5
        focus, looseness, _ = interpolated_state(mid, spans, smoothing_strength)
        focus, looseness, _ = apply_visual_impact(focus, looseness, 0.0, visual_impact)
        coverage = 0.08 + 0.92 * focus**1.45
        if focus < 0.86 and rng.random() > coverage:
            continue

        cell = theta2 - theta1
        length_ratio = 0.16 + 0.84 * focus
        jitter = (1.0 - focus) * cell * rng.uniform(-0.22, 0.22)
        seg_mid = mid + jitter
        seg_half = cell * length_ratio * 0.5
        width = max(
            2, int(base_width * (0.34 + 1.08 * focus) * (1.0 - 0.28 * looseness))
        )
        radial = rng.normal(0.0, canvas * 0.0012 * looseness)
        target.arc(
            center,
            radius + radial,
            seg_mid - seg_half,
            seg_mid + seg_half,
            width,
            mark,
        )

    total_chars = sum(segment.chars for segment in segments)
    global_loose = (
        float(np.mean([segment.looseness for segment in segments])) if segments else 0.0
    )
    candidate_count = int(
        particle_multiplier
        * (360 + total_chars * 1.2 + 760 * global_loose + 95 * len(segments))
    )
    candidate_count = max(40, candidate_count)
    for _ in range(candidate_count):
        theta = start_angle + rng.uniform(0.0, TAU)
        focus, looseness, novelty = interpolated_state(theta, spans, smoothing_strength)
        focus, looseness, novelty = apply_visual_impact(
            focus, looseness, novelty, visual_impact
        )
        keep = 0.04 + 0.96 * looseness**1.18
        if focus > 0.88:
            keep *= 0.24
        if rng.random() > keep:
            continue

        spread = canvas * (0.0025 + 0.067 * looseness**1.45)
        stream_side = -1.0 if math.sin(theta * 1.7 + seed * 0.001) < 0 else 1.0
        radial = rng.normal(0.0, spread)
        if rng.random() < 0.36 * looseness:
            radial += stream_side * rng.gamma(1.35, spread * (0.75 + novelty * 0.55))
        theta += rng.normal(0.0, 0.006 + 0.035 * looseness)

        x, y = point_on_circle(center, radius, theta, radial)
        particle_size = rng.uniform(
            canvas * 0.0016,
            canvas * (0.0024 + 0.0012 * looseness),
        )
        target.square(x, y, particle_size, mark)
        total_particles += 1

    for segment, start, end in spans:
        focus, looseness, novelty = apply_visual_impact(
            segment.focus, segment.looseness, segment.novelty, visual_impact
        )
        if looseness < 0.82 or focus > 0.38:
            continue
        span = end - start
        scatter_count = int(
            particle_multiplier
            * (72 + segment.chars * 1.8 + 220 * min(1.0, span / TAU))
        )
        scatter_start = start + span * 0.12
        scatter_end = end - span * 0.12
        if scatter_end <= scatter_start:
            scatter_start, scatter_end = start, end
        for _ in range(max(36, scatter_count)):
            theta = rng.uniform(scatter_start, scatter_end)
            theta += rng.normal(0.0, 0.012 + 0.028 * looseness)
            spread = canvas * (0.030 + 0.060 * looseness)
            side = -1.0 if rng.random() < 0.5 else 1.0
            radial = rng.normal(0.0, spread * 0.36)
            if rng.random() < 0.62:
                radial += side * rng.gamma(1.35, spread * 0.42)
            x, y = point_on_circle(center, radius, theta, radial)
            particle_size = rng.uniform(canvas * 0.0015, canvas * 0.0038)
            target.square(x, y, particle_size, mark)
            total_particles += 1

    return segments, total_particles
