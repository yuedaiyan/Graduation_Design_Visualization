from __future__ import annotations

import argparse
import json
import math
import re
from pathlib import Path

import numpy as np
from PIL import Image

from background import emotion_palette
from config import BACKGROUND_EMOTION_VARIATION, CANVAS_PX, SUPERSAMPLE, TAU
from io_data import load_emotion_profiles, load_entries
from metrics import build_window_metrics
from models import EmotionProfile, Entry, WindowMetric
from utils import blend, clamp01, normalize_range, stable_seed


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


def create_tide_wave_field(
    canvas: int,
    metrics: list[WindowMetric] | None,
    seed: int,
) -> np.ndarray:
    rng = np.random.default_rng(seed ^ 0xA53A9B1D)

    y, x = np.mgrid[0:canvas, 0:canvas].astype(np.float32)
    xn = x / max(1, canvas - 1)
    yn = y / max(1, canvas - 1)

    tide_centers = semantic_tide_centers(metrics, rng)
    wave = np.zeros((canvas, canvas), dtype=np.float32)
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

    return (wave - float(wave.min())) / max(1e-9, float(wave.max() - wave.min()))


def add_tide_wave_to_field(
    field: np.ndarray,
    metrics: list[WindowMetric] | None,
    seed: int,
    variation_strength: float,
) -> np.ndarray:
    wave = create_tide_wave_field(field.shape[0], metrics, seed)
    return field + wave * 0.22 * min(max(0.0, variation_strength), 2.0)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Generate tide-wave preview images")
    parser.add_argument(
        "--root", default=None, help="Project root. Defaults to parent of this script"
    )
    parser.add_argument(
        "--out-dir",
        default=None,
        help="Output base dir. Defaults to <root>/output_All/diary_Ring",
    )
    parser.add_argument(
        "--date",
        default=None,
        help="Optional date/stem filter. If omitted, renders a generic colored tide background.",
    )
    parser.add_argument(
        "--all",
        action="store_true",
        help="Render every diary entry instead of the generic preview.",
    )
    parser.add_argument("--seed", type=int, default=71, help="Base random seed")
    parser.add_argument(
        "--size", type=int, default=CANVAS_PX, help="Square output size in pixels"
    )
    parser.add_argument(
        "--background-variation",
        type=float,
        default=BACKGROUND_EMOTION_VARIATION,
        help="Same background variation strength used by main.py.",
    )
    parser.add_argument(
        "--primary",
        default="平静",
        help="Generic render emotion_primary. Ignored when --date matches diary entries.",
    )
    parser.add_argument(
        "--arc",
        default="起伏",
        help="Generic render emotion_arc. Ignored when --date matches diary entries.",
    )
    parser.add_argument(
        "--raw-wave",
        action="store_true",
        help="Output the extracted grayscale tide-wave field instead of the main colored background.",
    )
    return parser.parse_args()


def next_wave_try_dir(base: Path) -> Path:
    base.mkdir(parents=True, exist_ok=True)
    max_idx = 0
    for child in base.iterdir():
        if not child.is_dir():
            continue
        match = re.fullmatch(r"wave_try_(\d+)", child.name)
        if match:
            max_idx = max(max_idx, int(match.group(1)))
    target = base / f"wave_try_{max_idx + 1}"
    target.mkdir(parents=True, exist_ok=False)
    return target


def find_entries(entries: list[Entry], date: str | None, render_all: bool) -> list[Entry]:
    if render_all:
        return entries
    if not date:
        return []
    return [entry for entry in entries if entry.stem == date or entry.date == date]


def save_wave_preview(path: Path, wave: np.ndarray) -> None:
    image = Image.fromarray(np.clip(wave * 255.0, 0, 255).astype(np.uint8), "L")
    image.save(path)


def create_colored_wave_image(
    canvas: int,
    emotion: EmotionProfile,
    seed: int,
    variation_strength: float,
    metrics: list[WindowMetric] | None,
) -> Image.Image:
    base, accent = emotion_palette(emotion.primary)
    strength = max(0.0, variation_strength)

    y, x = np.mgrid[0:canvas, 0:canvas].astype(np.float32)
    xn = x / max(1, canvas - 1)
    yn = y / max(1, canvas - 1)

    soft_base = blend(base, np.array([255, 255, 255], dtype=np.float32), 0.42)
    soft_accent = blend(accent, np.array([255, 255, 255], dtype=np.float32), 0.32)
    diagonal = 0.5 * xn + 0.5 * (1.0 - yn)
    radial = np.sqrt((xn - 0.5) ** 2 + (yn - 0.5) ** 2)
    field = diagonal * 0.18 + (1.0 - radial) * 0.12

    if emotion.arc == "起伏":
        field = add_tide_wave_to_field(field, metrics, seed, strength)
    elif emotion.arc == "上升":
        field += (1.0 - yn) * 0.30 * min(strength, 2.0)
    elif emotion.arc == "下降":
        field += yn * 0.34 * min(strength, 2.0)

    field = np.clip(field, 0.0, 1.0)
    rgb = blend(soft_base, soft_accent, field[..., None] * min(1.0, 0.75 * strength))
    alpha = np.full((canvas, canvas, 1), 255.0, dtype=np.float32)
    return Image.fromarray(
        np.clip(np.dstack([rgb, alpha]), 0, 255).astype(np.uint8), "RGBA"
    )


def save_colored_background(
    path: Path,
    size_px: int,
    metrics: list[WindowMetric] | None,
    emotion: EmotionProfile,
    seed: int,
    background_variation: float,
) -> None:
    canvas = size_px * SUPERSAMPLE
    image = create_colored_wave_image(
        canvas=canvas,
        emotion=emotion,
        seed=seed,
        variation_strength=background_variation,
        metrics=metrics,
    )
    image = image.resize((size_px, size_px), Image.Resampling.LANCZOS).convert("RGB")
    image.save(path, quality=96)


def main() -> None:
    args = parse_args()
    script_dir = Path(__file__).resolve().parent
    root = Path(args.root).expanduser().resolve() if args.root else script_dir.parent
    out_base = (
        Path(args.out_dir).expanduser().resolve()
        if args.out_dir
        else root / "output_All" / "diary_Ring"
    )
    out_dir = next_wave_try_dir(out_base)

    summary: dict[str, object] = {
        "try_dir": str(out_dir),
        "size_px": args.size,
        "seed": args.seed,
        "date": args.date,
        "all": args.all,
        "mode": "raw_wave" if args.raw_wave else "colored_background",
        "background_variation": args.background_variation,
        "generic_emotion_primary": args.primary,
        "generic_emotion_arc": args.arc,
        "items": [],
    }
    items: list[dict[str, object]] = []

    all_entries = load_entries(root / "diary_entries.json")
    emotion_profiles = load_emotion_profiles(root, all_entries)
    entries = find_entries(all_entries, args.date, args.all)
    if args.all and args.date:
        raise SystemExit("Use either --all or --date, not both.")
    if args.date and not entries:
        raise SystemExit(f"No diary entries matched {args.date!r}")

    if entries:
        for idx, entry in enumerate(entries, start=1):
            metrics, _ = build_window_metrics(root, entry)
            seed = stable_seed(entry.stem, args.seed)
            out_path = out_dir / f"{entry.stem}.png"
            emotion = emotion_profiles.get(
                entry.stem, EmotionProfile(primary="平静", arc="稳定")
            )
            if args.raw_wave:
                wave = create_tide_wave_field(args.size, metrics, seed)
                save_wave_preview(out_path, wave)
                value_range = {"min": float(wave.min()), "max": float(wave.max())}
            else:
                save_colored_background(
                    path=out_path,
                    size_px=args.size,
                    metrics=metrics,
                    emotion=emotion,
                    seed=seed,
                    background_variation=args.background_variation,
                )
                value_range = {}
            items.append(
                {
                    "file": out_path.name,
                    "stem": entry.stem,
                    "date": entry.date,
                    "num_windows": len(metrics),
                    "emotion_primary": emotion.primary,
                    "emotion_arc": emotion.arc,
                    **value_range,
                }
            )
            print(
                f"  [{idx}/{len(entries)}] {out_path.name} windows={len(metrics)} "
                f"emotion={emotion.primary}/{emotion.arc}"
            )
    else:
        out_path = out_dir / "generic_tide_wave.png"
        emotion = EmotionProfile(primary=args.primary, arc=args.arc)
        if args.raw_wave:
            wave = create_tide_wave_field(args.size, metrics=None, seed=args.seed)
            save_wave_preview(out_path, wave)
            value_range = {"min": float(wave.min()), "max": float(wave.max())}
        else:
            save_colored_background(
                path=out_path,
                size_px=args.size,
                metrics=None,
                emotion=emotion,
                seed=args.seed,
                background_variation=args.background_variation,
            )
            value_range = {}
        items.append(
            {
                "file": out_path.name,
                "stem": None,
                "date": None,
                "num_windows": 0,
                "emotion_primary": emotion.primary,
                "emotion_arc": emotion.arc,
                **value_range,
            }
        )
        print(f"  {out_path.name} emotion={emotion.primary}/{emotion.arc}")

    summary["items"] = items
    with (out_dir / "summary.json").open("w", encoding="utf-8") as f:
        json.dump(summary, f, ensure_ascii=False, indent=2)

    print(f"输出目录: {out_dir}")


if __name__ == "__main__":
    main()
