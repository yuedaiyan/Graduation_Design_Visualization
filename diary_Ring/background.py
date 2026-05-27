from __future__ import annotations

import math

import numpy as np
from PIL import Image, ImageDraw

from config import EMOTION_COLORS, TAU
from models import WindowMetric
from utils import blend, clamp01, normalize_range


def emotion_palette(primary: str) -> tuple[np.ndarray, np.ndarray]:
    base, accent = EMOTION_COLORS.get(primary, EMOTION_COLORS["平静"])
    return np.array(base, dtype=np.float32), np.array(accent, dtype=np.float32)


def semantic_tide_centers(
    metrics: list[WindowMetric] | None, rng: np.random.Generator
) -> list[tuple[float, float, float, WindowMetric | None]]:
    if not metrics:
        return [(rng.uniform(0.25, 0.75), rng.uniform(0.2, 0.8), 1.0, None)]

    total_chars = max(1, sum(metric.chars for metric in metrics))
    cursor = 0.0
    candidates: list[tuple[float, float, float, WindowMetric]] = []
    for metric in metrics:
        midpoint = (cursor + metric.chars * 0.5) / total_chars
        cursor += metric.chars
        angle = -math.pi / 2.0 + midpoint * TAU
        radius = 0.19 + 0.18 * metric.novelty + rng.normal(0.0, 0.025)
        px = float(clamp01(0.5 + math.cos(angle) * radius + rng.normal(0.0, 0.045)))
        py = float(clamp01(0.5 + math.sin(angle) * radius + rng.normal(0.0, 0.045)))
        weight = (
            0.42 * metric.looseness
            + 0.34 * metric.novelty
            + 0.24 * normalize_range(metric.chars, 1.0, total_chars * 0.24)
        )
        candidates.append((weight, px, py, metric))

    candidates.sort(key=lambda item: item[0], reverse=True)
    selected = candidates[: min(4, max(2, len(candidates)))]
    total_weight = max(1e-9, sum(item[0] for item in selected))
    return [
        (px, py, max(0.12, weight / total_weight), metric)
        for weight, px, py, metric in selected
    ]


def create_emotion_background(
    canvas: int,
    primary: str,
    arc: str,
    seed: int,
    variation_strength: float,
    metrics: list[WindowMetric] | None = None,
) -> Image.Image:
    base, accent = emotion_palette(primary)
    strength = max(0.0, variation_strength)
    rng = np.random.default_rng(seed ^ 0xA53A9B1D)

    y, x = np.mgrid[0:canvas, 0:canvas].astype(np.float32)
    xn = x / max(1, canvas - 1)
    yn = y / max(1, canvas - 1)

    soft_base = blend(base, np.array([255, 255, 255], dtype=np.float32), 0.42)
    soft_accent = blend(accent, np.array([255, 255, 255], dtype=np.float32), 0.32)
    diagonal = 0.5 * xn + 0.5 * (1.0 - yn)
    radial = np.sqrt((xn - 0.5) ** 2 + (yn - 0.5) ** 2)
    field = diagonal * 0.18 + (1.0 - radial) * 0.12

    if arc == "起伏":
        tide_centers = semantic_tide_centers(metrics, rng)
        wave = np.zeros_like(field)
        for cx, cy, weight, metric in tide_centers:
            direction = rng.uniform(0.0, TAU)
            stretch = rng.uniform(0.62, 1.55)
            shear = rng.uniform(-0.38, 0.38)
            dx = xn - cx
            dy = yn - cy
            rx = math.cos(direction) * dx + math.sin(direction) * dy
            ry = -math.sin(direction) * dx + math.cos(direction) * dy
            rx = rx * stretch + ry * shear
            ry = ry / max(0.35, stretch)
            dist = np.sqrt(rx * rx + ry * ry)
            angle = np.arctan2(ry, rx)
            focus = metric.focus if metric else rng.uniform(0.25, 0.75)
            looseness = metric.looseness if metric else rng.uniform(0.45, 0.85)
            novelty = metric.novelty if metric else rng.uniform(0.35, 0.75)
            freq = 7.0 + 8.0 * novelty + rng.uniform(-1.2, 2.4)
            swirl = 1.4 + 2.8 * looseness + rng.uniform(-0.6, 0.9)
            ripple_warp = (
                1.0
                + 0.22
                * np.sin((rx * rng.uniform(2.0, 5.4) + rng.uniform(-0.5, 0.5)) * TAU)
                + 0.16
                * np.cos((ry * rng.uniform(2.0, 4.8) + rng.uniform(-0.5, 0.5)) * TAU)
            )
            bend = (
                np.sin((rx * rng.uniform(1.0, 3.4) + rng.uniform(-0.4, 0.4)) * TAU)
                * 0.35
                + np.cos((ry * rng.uniform(1.0, 3.1) + rng.uniform(-0.4, 0.4)) * TAU)
                * 0.30
                + np.sin(
                    (
                        rx * rng.uniform(2.4, 6.0)
                        + ry * rng.uniform(-2.6, 3.4)
                        + rng.uniform(-0.5, 0.5)
                    )
                    * TAU
                )
                * rng.uniform(0.10, 0.26)
            )
            phase = (
                dist * freq * ripple_warp * TAU
                + angle * swirl
                + bend
                + rng.uniform(0.0, TAU)
            )
            local = np.sin(phase)
            if rng.random() < 0.55:
                local = 0.72 * local + 0.28 * np.sin(
                    phase * rng.uniform(1.45, 2.35) + rng.uniform(0.0, TAU)
                )
            broken = 0.72 + 0.28 * np.sin(
                (rx * rng.uniform(1.2, 3.3) + ry * rng.uniform(1.0, 3.0)) * TAU
                + rng.uniform(0.0, TAU)
            )
            envelope = np.exp(-(dist**2) / (0.018 + 0.055 * looseness))
            wave += local * broken * envelope * weight * (0.65 + 0.85 * (1.0 - focus))

        for _ in range(3):
            direction = rng.uniform(0.0, TAU)
            freq = rng.uniform(0.7, 1.9)
            phase = rng.uniform(0.0, TAU)
            warped_axis = (
                math.cos(direction) * xn
                + math.sin(direction) * yn
                + 0.08
                * np.sin(
                    (xn * rng.uniform(1.0, 2.5) + yn * rng.uniform(0.8, 2.2)) * TAU
                )
            )
            wave += 0.12 * np.sin(warped_axis * freq * TAU + phase)

        wave = (wave - float(wave.min())) / max(1e-9, float(wave.max() - wave.min()))
        field += wave * 0.22 * min(strength, 2.0)
    elif arc == "上升":
        field += (1.0 - yn) * 0.30 * min(strength, 2.0)
    elif arc == "下降":
        field += yn * 0.34 * min(strength, 2.0)
    else:
        field += rng.uniform(-0.015, 0.015, size=(canvas, canvas))

    field = np.clip(field, 0.0, 1.0)
    rgb = blend(soft_base, soft_accent, field[..., None] * min(1.0, 0.75 * strength))
    alpha = np.full((canvas, canvas, 1), 255.0, dtype=np.float32)
    image = Image.fromarray(
        np.clip(np.dstack([rgb, alpha]), 0, 255).astype(np.uint8), "RGBA"
    )
    draw = ImageDraw.Draw(image, "RGBA")

    particle_count = int(canvas * (0.30 + 0.95 * min(strength, 2.0)))
    particle_count = max(120, particle_count)
    particle_color = tuple(np.clip(accent * 0.86, 0, 255).astype(int).tolist())
    pale_color = tuple(np.clip(base * 1.08, 0, 255).astype(int).tolist())

    tide_centers = semantic_tide_centers(metrics, rng)
    tide_weights = np.array([center[2] for center in tide_centers], dtype=np.float64)
    tide_weights = tide_weights / tide_weights.sum()
    for _ in range(particle_count):
        if arc == "起伏":
            if rng.random() < 0.66 * min(1.0, strength):
                center_idx = int(rng.choice(len(tide_centers), p=tide_weights))
                tx, ty, _, metric = tide_centers[center_idx]
                spread = 0.07 + 0.08 * (metric.looseness if metric else 0.65)
                px = float(np.clip(rng.normal(tx, spread), 0.0, 1.0))
                py = float(
                    np.clip(rng.normal(ty, spread * rng.uniform(0.75, 1.35)), 0.0, 1.0)
                )
            else:
                px, py = float(rng.random()), float(rng.random())
            size = rng.uniform(0.9, 2.8 + 1.4 * strength)
            alpha_v = int(rng.uniform(12, 48 + 20 * strength))
        elif arc == "上升":
            py = float(rng.beta(1.25, 2.45))
            px = float(
                np.clip(rng.random() + rng.normal(0.0, 0.035 * strength), 0.0, 1.0)
            )
            size = rng.uniform(0.8, 2.4 + 1.2 * (1.0 - py) * strength)
            alpha_v = int(rng.uniform(10, 42 + 28 * (1.0 - py) * strength))
        elif arc == "下降":
            py = float(rng.beta(2.65, 1.18))
            px = float(
                np.clip(rng.random() + rng.normal(0.0, 0.045 * strength), 0.0, 1.0)
            )
            size = rng.uniform(0.8, 2.5 + 1.6 * py * strength)
            alpha_v = int(rng.uniform(10, 46 + 30 * py * strength))
        else:
            px, py = float(rng.random()), float(rng.random())
            size = rng.uniform(0.6, 1.8 + 0.45 * strength)
            alpha_v = int(rng.uniform(8, 24 + 8 * strength))

        jitter_x = rng.normal(0.0, 0.008 * strength)
        jitter_y = rng.normal(0.0, 0.008 * strength)
        cx = (px + jitter_x) * canvas
        cy = (py + jitter_y) * canvas
        color = particle_color if rng.random() < 0.56 else pale_color
        fill = (*color, max(0, min(90, alpha_v)))
        half = size * 0.5
        draw.rectangle([cx - half, cy - half, cx + half, cy + half], fill=fill)

        if arc == "下降" and rng.random() < 0.22 * min(1.0, strength):
            tail = rng.uniform(5.0, 18.0 + 12.0 * strength)
            draw.line(
                [cx, cy - tail, cx, cy], fill=(*color, max(6, alpha_v // 3)), width=1
            )
        elif arc == "上升" and rng.random() < 0.18 * min(1.0, strength):
            tail = rng.uniform(4.0, 14.0 + 10.0 * strength)
            draw.line(
                [cx, cy + tail, cx, cy], fill=(*color, max(5, alpha_v // 3)), width=1
            )

    return image
