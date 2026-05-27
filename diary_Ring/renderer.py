from __future__ import annotations

from pathlib import Path
from typing import Any

import numpy as np
from PIL import Image, ImageDraw

from background import create_emotion_background
from config import SUPERSAMPLE
from models import EmotionProfile, WindowMetric
from ring import PillowRingTarget, SvgRingTarget, draw_ring_content


def render_ring(
    out_path: Path,
    metrics: list[WindowMetric],
    summary: dict[str, float],
    distinctiveness: float,
    emotion: EmotionProfile,
    seed: int,
    size_px: int,
    particle_multiplier: float,
    visual_impact: float,
    smoothing_strength: float,
    background_variation: float,
) -> dict[str, Any]:
    scale = SUPERSAMPLE
    canvas = size_px * scale
    image = create_emotion_background(
        canvas=canvas,
        primary=emotion.primary,
        arc=emotion.arc,
        seed=seed,
        variation_strength=background_variation,
        metrics=metrics,
    )
    draw = ImageDraw.Draw(image)
    segments, total_particles = draw_ring_content(
        target=PillowRingTarget(draw),
        metrics=metrics,
        summary=summary,
        distinctiveness=distinctiveness,
        seed=seed,
        canvas=canvas,
        particle_multiplier=particle_multiplier,
        visual_impact=visual_impact,
        smoothing_strength=smoothing_strength,
    )
    image = image.resize((size_px, size_px), Image.Resampling.LANCZOS).convert("RGB")
    image.save(out_path, quality=96)

    return {
        "file": out_path.name,
        "avg_focus": (
            float(np.mean([segment.focus for segment in segments])) if segments else 0.0
        ),
        "avg_looseness": (
            float(np.mean([segment.looseness for segment in segments]))
            if segments
            else 0.0
        ),
        "variation_count": len(segments),
        "particles": int(total_particles),
        "distinctiveness": float(distinctiveness),
        "emotion_primary": emotion.primary,
        "emotion_arc": emotion.arc,
        "segments": [
            {
                "idx": segment.idx,
                "chars": segment.chars,
                "focus": segment.focus,
                "looseness": segment.looseness,
                "source_window_indices": segment.source_window_indices,
            }
            for segment in segments
        ],
        **summary,
    }


def render_ring_svg(
    out_path: Path,
    metrics: list[WindowMetric],
    summary: dict[str, float],
    distinctiveness: float,
    seed: int,
    size_px: int,
    particle_multiplier: float,
    visual_impact: float,
    smoothing_strength: float,
) -> None:
    canvas = size_px * SUPERSAMPLE
    target = SvgRingTarget(size_px=size_px, source_canvas=canvas)
    draw_ring_content(
        target=target,
        metrics=metrics,
        summary=summary,
        distinctiveness=distinctiveness,
        seed=seed,
        canvas=canvas,
        particle_multiplier=particle_multiplier,
        visual_impact=visual_impact,
        smoothing_strength=smoothing_strength,
    )
    target.save(out_path)
