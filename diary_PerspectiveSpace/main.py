#!/usr/bin/env python3
"""Render diary perspective-space images from diary-level and event-level vectors.

Usage:
  python3 main.py
  python3 main.py --date 2026-03-10
  python3 main.py --count 100 --svg-only

Outputs:
  ../output_All/diary_PerspectiveSpace/try_N/<date>.png
  ../output_All/diary_PerspectiveSpace/svg_try_N/<date>.svg
"""

from __future__ import annotations

import argparse
import colorsys
import hashlib
import re
from dataclasses import dataclass
from pathlib import Path

import numpy as np

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib import font_manager

BASE_CONTENT_SIZE = 1000
DEFAULT_CONTENT_SIZE = 1000
DEFAULT_RESOLUTION = 2160

CANVAS_W = DEFAULT_CONTENT_SIZE
CANVAS_H = DEFAULT_CONTENT_SIZE
OUTPUT_W = DEFAULT_RESOLUTION
OUTPUT_H = DEFAULT_RESOLUTION
RASTER_SCALE = OUTPUT_W / CANVAS_W
OUTER_MARGIN = 34
INNER_MARGIN = 63
HEADER_LABEL_FONT_PATH = Path.home() / "Library/Fonts/AkzidenzGrotesk-Regular.otf"
HEADER_LABEL_FONT_FAMILY = "Akzidenz-Grotesk BQ"
# 顶部标签字号，单位是逻辑画布坐标；数值越小，文字越小。
HEADER_LABEL_FONT_SIZE = 6.2
# 顶部标签垂直中心位置，单位是逻辑画布坐标；数值越小，文字越靠上。
HEADER_LABEL_Y = 14.5
# 左侧 Events 标签相对左侧内边框的水平偏移；数值越大，文字越向右。
HEADER_LEFT_LABEL_X_PADDING = -38.0
# 右侧日期标签相对右侧内边框的水平偏移；数值越大，文字越向左。
HEADER_RIGHT_LABEL_X_PADDING = -38.0
STYLE_CONTEXT_CACHE_VERSION = "v2"
STYLE_CONTEXT_BASIS_DIMS = 3
STYLE_CONTEXT_SCALE_PERCENTILE = 90


@dataclass(frozen=True)
class SvgLine:
    points: np.ndarray
    width: float
    opacity: float
    zorder: int = 4


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Generate PerspectiveSpace diary images"
    )
    parser.add_argument(
        "--root",
        default=None,
        help="Project root containing vector dirs. Defaults to parent of this script.",
    )
    parser.add_argument(
        "--out-dir",
        default=None,
        help="Output base directory. Defaults to <root>/output_All/<this program folder>.",
    )
    parser.add_argument("--date", default=None, help="Only render a specific date")
    parser.add_argument(
        "--count",
        type=int,
        default=None,
        help="Generate only the first N valid dates in batch mode.",
    )
    output_group = parser.add_mutually_exclusive_group()
    output_group.add_argument(
        "--png-only",
        action="store_true",
        help="Only render PNGs into try_N; skip SVG output.",
    )
    output_group.add_argument(
        "--svg-only",
        action="store_true",
        help="Only render SVGs into svg_try_N; skip PNG output.",
    )
    parser.add_argument("--dpi", type=int, default=220)
    parser.add_argument("--seed", type=int, default=7)
    parser.add_argument(
        "--content-size",
        type=int,
        default=DEFAULT_CONTENT_SIZE,
        help=f"Logical content coordinate size. Default: {DEFAULT_CONTENT_SIZE}.",
    )
    parser.add_argument(
        "--resolution",
        "--size",
        dest="resolution",
        type=int,
        default=DEFAULT_RESOLUTION,
        help=f"Final square PNG pixel size and SVG display size. Default: {DEFAULT_RESOLUTION}.",
    )
    args = parser.parse_args()
    if args.count is not None and args.count < 1:
        parser.error("--count must be a positive integer")
    if args.content_size < 1:
        parser.error("--content-size must be a positive integer")
    if args.resolution < 1:
        parser.error("--resolution must be a positive integer")
    return args


def set_render_sizes(content_size: int, resolution: int) -> None:
    global CANVAS_W, CANVAS_H, OUTPUT_W, OUTPUT_H, RASTER_SCALE
    CANVAS_W = int(content_size)
    CANVAS_H = int(content_size)
    OUTPUT_W = int(resolution)
    OUTPUT_H = int(resolution)
    RASTER_SCALE = OUTPUT_W / CANVAS_W


def raster_s(value: float) -> float:
    return value * RASTER_SCALE


def header_label_font() -> font_manager.FontProperties:
    if HEADER_LABEL_FONT_PATH.exists():
        font_manager.fontManager.addfont(str(HEADER_LABEL_FONT_PATH))
        return font_manager.FontProperties(
            fname=str(HEADER_LABEL_FONT_PATH), style="normal", weight="normal"
        )
    return font_manager.FontProperties(
        family=HEADER_LABEL_FONT_FAMILY, style="normal", weight="normal"
    )


def next_try_dirs(
    base: Path, export_png: bool = True, export_svg: bool = True
) -> tuple[Path | None, Path | None]:
    base.mkdir(parents=True, exist_ok=True)
    max_idx = 0
    for p in base.iterdir():
        if not p.is_dir():
            continue
        m = re.fullmatch(r"(?:svg_)?try_(\d+)", p.name)
        if m:
            max_idx = max(max_idx, int(m.group(1)))
    next_idx = max_idx + 1

    png_dir = base / f"try_{next_idx}" if export_png else None
    svg_dir = base / f"svg_try_{next_idx}" if export_svg else None

    if png_dir is not None:
        png_dir.mkdir(parents=True, exist_ok=False)
    if svg_dir is not None:
        svg_dir.mkdir(parents=True, exist_ok=False)

    return png_dir, svg_dir


def display_path(path: Path, start: Path) -> str:
    try:
        return str(path.relative_to(start))
    except ValueError:
        return str(path)


def discover_dates(root: Path, only_date: str | None) -> list[str]:
    diary_dir = root / "diary_vectors"
    sent_dir = root / "diary_sentence_vectors"
    if not diary_dir.exists() or not sent_dir.exists():
        raise FileNotFoundError("Missing diary_vectors or diary_sentence_vectors")

    if only_date:
        candidates = [only_date]
    else:
        candidates = sorted(p.stem for p in diary_dir.glob("*.npy"))

    valid: list[str] = []
    for d in candidates:
        if (diary_dir / f"{d}.npy").exists() and (
            sent_dir / d / "window_vectors.npy"
        ).exists():
            valid.append(d)

    if only_date and not valid:
        raise FileNotFoundError(f"Date not found or missing files: {only_date}")
    if not valid:
        raise RuntimeError("No valid dates discovered")
    return valid


def unit_rows(x: np.ndarray) -> np.ndarray:
    n = np.linalg.norm(x, axis=1, keepdims=True)
    n = np.clip(n, 1e-12, None)
    return x / n


def unit_vec(x: np.ndarray) -> np.ndarray:
    n = float(np.linalg.norm(x))
    if n < 1e-12:
        return x
    return x / n


def cache_dir(root: Path) -> Path:
    path = root / ".cache" / "derived" / "diary_PerspectiveSpace"
    path.mkdir(parents=True, exist_ok=True)
    return path


def style_context_signature(root: Path, dates: list[str]) -> str:
    parts = [
        f"version:{STYLE_CONTEXT_CACHE_VERSION}",
        f"basis_dims:{STYLE_CONTEXT_BASIS_DIMS}",
        f"scale_percentile:{STYLE_CONTEXT_SCALE_PERCENTILE}",
    ]
    for date in dates:
        path = root / "diary_vectors" / f"{date}.npy"
        stat = path.stat()
        parts.append(f"{date}:{stat.st_size}:{stat.st_mtime_ns}")
    return hashlib.sha256("\n".join(parts).encode("utf-8")).hexdigest()[:16]


def build_style_context(root: Path, dates: list[str]) -> dict[str, np.ndarray]:
    cache_path = (
        cache_dir(root) / f"style_context_{style_context_signature(root, dates)}.npz"
    )
    if cache_path.exists():
        data = np.load(cache_path)
        return {"mean": data["mean"], "basis": data["basis"], "scale": data["scale"]}

    vecs = np.stack(
        [np.load(root / "diary_vectors" / f"{d}.npy").reshape(-1) for d in dates]
    )
    mean = vecs.mean(axis=0)
    centered = vecs - mean
    _, _, vt = np.linalg.svd(centered, full_matrices=False)
    basis = vt[:STYLE_CONTEXT_BASIS_DIMS]
    proj = centered @ basis.T
    scale = np.percentile(np.abs(proj), STYLE_CONTEXT_SCALE_PERCENTILE, axis=0)
    scale = np.clip(scale, 1e-6, None)
    np.savez_compressed(cache_path, mean=mean, basis=basis, scale=scale)
    return {"mean": mean, "basis": basis, "scale": scale}


def vec_style(diary_vec: np.ndarray, ctx: dict[str, np.ndarray]) -> dict[str, float]:
    z = (diary_vec - ctx["mean"]) @ ctx["basis"].T
    z = np.clip(z / ctx["scale"], -2.0, 2.0)
    z0, z1, z2 = [float(v) for v in z]

    # Keep hue range around red/pink/purple to mimic the reference atmosphere.
    hue = (325.0 + 42.0 * np.tanh(z0) + 26.0 * np.tanh(z1)) % 360.0
    sat = float(
        np.clip(0.28 + 0.20 * abs(np.tanh(z1)) + 0.10 * abs(np.tanh(z2)), 0.2, 0.74)
    )
    val = float(np.clip(0.72 + 0.16 * np.tanh(z2), 0.55, 0.92))
    horizon = float(np.clip(0.39 + 0.09 * np.tanh(z2), 0.30, 0.50))
    bands = int(np.clip(round(4 + 4 * (0.5 + 0.5 * np.tanh(z1))), 3, 8))
    return {"hue": hue, "sat": sat, "val": val, "horizon": horizon, "bands": bands}


def hsv_to_rgb255(h: float, s: float, v: float) -> tuple[float, float, float]:
    r, g, b = colorsys.hsv_to_rgb(
        (h % 360.0) / 360.0, float(np.clip(s, 0, 1)), float(np.clip(v, 0, 1))
    )
    return (r, g, b)


def style_colors(
    style: dict[str, float],
) -> tuple[
    tuple[float, float, float], tuple[float, float, float], tuple[float, float, float]
]:
    base_rgb = hsv_to_rgb255(style["hue"], style["sat"], style["val"])
    accent_rgb = hsv_to_rgb255(
        style["hue"] + 14.0,
        min(style["sat"] + 0.14, 0.88),
        min(style["val"] + 0.06, 0.96),
    )
    line_rgb = hsv_to_rgb255(
        style["hue"] + 36.0,
        min(style["sat"] + 0.20, 0.92),
        max(style["val"] - 0.48, 0.16),
    )
    return base_rgb, accent_rgb, line_rgb


def event_vanishing_points(
    event_vecs: np.ndarray,
    diary_vec: np.ndarray,
    left: float,
    right: float,
    top: float,
    bottom: float,
    rng: np.random.Generator,
) -> tuple[np.ndarray, np.ndarray]:
    n = event_vecs.shape[0]
    if n == 0:
        x = np.array([(left + right) * 0.5], dtype=np.float64)
        y = np.array([(top + bottom) * 0.5], dtype=np.float64)
        w = np.array([1.0], dtype=np.float64)
        return np.column_stack([x, y]), w

    e = unit_rows(event_vecs.astype(np.float64))
    d = unit_vec(diary_vec.astype(np.float64))
    sims = np.clip(e @ d, -1.0, 1.0)
    w = (sims - sims.min()) / (max(sims.max() - sims.min(), 1e-9))
    w = 0.35 + 0.65 * w

    if n == 1:
        x_pos = np.array([(left + right) * 0.5], dtype=np.float64)
        y_pos = np.array([(top + bottom) * 0.5], dtype=np.float64)
    else:
        c = event_vecs - event_vecs.mean(axis=0, keepdims=True)
        _, _, vt = np.linalg.svd(c, full_matrices=False)
        axis_x = vt[0]
        axis_y = vt[1] if vt.shape[0] > 1 else np.roll(vt[0], 1)
        score_x = c @ axis_x
        score_y = c @ axis_y

        if float(np.max(score_x) - np.min(score_x)) < 1e-10:
            score_x = np.linspace(0.0, 1.0, n)
        else:
            score_x = (score_x - np.min(score_x)) / (np.max(score_x) - np.min(score_x))

        if float(np.max(score_y) - np.min(score_y)) < 1e-10:
            score_y = np.linspace(1.0, 0.0, n)
        else:
            score_y = (score_y - np.min(score_y)) / (np.max(score_y) - np.min(score_y))

        score_y = 1.0 - score_y

        # Keep a safe border margin while allowing full-plane distribution.
        x_pos = left + (0.08 + 0.84 * score_x) * (right - left)
        y_pos = top + (0.08 + 0.84 * score_y) * (bottom - top)
        x_pos += rng.normal(0.0, (right - left) * 0.012, size=n)
        y_pos += rng.normal(0.0, (bottom - top) * 0.012, size=n)

    # Use similarity to softly modulate vertical spread instead of locking to horizon.
    y_pos += (0.5 - w) * (bottom - top) * 0.06
    y_pos = np.clip(y_pos, top + 10.0, bottom - 10.0)

    points = np.column_stack([x_pos, y_pos])
    return points, w


def draw_background(
    ax: plt.Axes,
    style: dict[str, float],
    left: float,
    right: float,
    top: float,
    bottom: float,
    horizon_y: float,
) -> tuple[float, float, float]:
    base_rgb, _, line_rgb = style_colors(style)

    ax.set_facecolor(base_rgb)

    bands = int(style["bands"])
    top_bands = max(2, bands // 2)
    floor_bands = max(2, bands)

    top_h = max(horizon_y - top, 2.0)
    floor_h = max(bottom - horizon_y, 2.0)

    for i in range(top_bands):
        y0 = top + (top_h * i / top_bands)
        y1 = top + (top_h * (i + 1) / top_bands)
        t = i / max(top_bands - 1, 1)
        c = hsv_to_rgb255(
            style["hue"] - 6.0,
            style["sat"] * (0.9 - 0.2 * t),
            min(style["val"] + 0.08 - 0.10 * t, 0.98),
        )
        ax.add_patch(
            plt.Rectangle(
                (left, y0), right - left, y1 - y0, color=c, ec="none", zorder=0
            )
        )

    for i in range(floor_bands):
        y0 = horizon_y + (floor_h * i / floor_bands)
        y1 = horizon_y + (floor_h * (i + 1) / floor_bands)
        t = i / max(floor_bands - 1, 1)
        c = hsv_to_rgb255(
            style["hue"] + 8.0,
            min(style["sat"] + 0.10 + 0.06 * t, 0.88),
            max(style["val"] - 0.12 - 0.14 * t, 0.26),
        )
        ax.add_patch(
            plt.Rectangle(
                (left, y0), right - left, y1 - y0, color=c, ec="none", zorder=0
            )
        )

    # Horizon emphasis.
    ax.plot(
        [left, right],
        [horizon_y, horizon_y],
        color=line_rgb,
        lw=raster_s(2.0),
        alpha=0.65,
        zorder=3,
    )
    ax.plot(
        [left, right],
        [horizon_y + 6, horizon_y + 6],
        color=line_rgb,
        lw=raster_s(0.9),
        alpha=0.40,
        zorder=3,
    )
    return line_rgb


def collect_perspective_lines(
    points: np.ndarray,
    weights: np.ndarray,
    left: float,
    right: float,
    horizon_y: float,
    bottom: float,
    rng: np.random.Generator,
) -> list[SvgLine]:
    lines: list[SvgLine] = []
    floor_h = bottom - horizon_y

    # Horizontal perspective lines: dense near horizon.
    row_count = 22 + min(16, points.shape[0] * 2)
    for i in range(row_count):
        u = i / max(row_count - 1, 1)
        y = horizon_y + floor_h * (u**2.15)
        alpha = 0.16 + 0.45 * ((1.0 - u) ** 1.35)
        lw = 0.65 + 0.85 * ((1.0 - u) ** 1.10)
        jitter = rng.normal(0.0, 0.35)
        lines.append(
            SvgLine(
                points=np.array(
                    [[left, y + jitter], [right, y + jitter]], dtype=np.float64
                ),
                width=lw,
                opacity=alpha,
            )
        )

    # Event-driven vanishing lines from all borders, not just floor edge.
    width = right - left
    height = bottom - horizon_y
    for idx, ((vx, vy), w) in enumerate(zip(points, weights)):
        line_n = int(np.clip(round(6 + 20 * w), 6, 28))
        n_bottom = int(line_n * 0.45)
        n_top = int(line_n * 0.20)
        n_left = int(line_n * 0.18)
        n_right = line_n - n_bottom - n_top - n_left

        bottom_x = np.linspace(left, right, max(n_bottom, 1))
        top_x = np.linspace(left, right, max(n_top, 1))
        left_y = np.linspace(horizon_y, bottom, max(n_left, 1))
        right_y = np.linspace(horizon_y, bottom, max(n_right, 1))

        alpha = float(np.clip(0.10 + 0.27 * w, 0.08, 0.38))
        lw = float(np.clip(0.55 + 0.9 * w, 0.55, 1.55))

        bottom_x = bottom_x + rng.normal(0.0, width * 0.004, size=bottom_x.shape[0])
        top_x = top_x + rng.normal(0.0, width * 0.004, size=top_x.shape[0])
        left_y = left_y + rng.normal(0.0, height * 0.006, size=left_y.shape[0])
        right_y = right_y + rng.normal(0.0, height * 0.006, size=right_y.shape[0])

        for x0 in bottom_x:
            lines.append(
                SvgLine(
                    points=np.array([[x0, bottom], [vx, vy]], dtype=np.float64),
                    width=lw,
                    opacity=alpha,
                )
            )
        for x0 in top_x:
            lines.append(
                SvgLine(
                    points=np.array([[x0, horizon_y], [vx, vy]], dtype=np.float64),
                    width=lw * 0.95,
                    opacity=alpha * 0.78,
                )
            )
        for y0 in left_y:
            lines.append(
                SvgLine(
                    points=np.array([[left, y0], [vx, vy]], dtype=np.float64),
                    width=lw * 0.92,
                    opacity=alpha * 0.74,
                )
            )
        for y0 in right_y:
            lines.append(
                SvgLine(
                    points=np.array([[right, y0], [vx, vy]], dtype=np.float64),
                    width=lw * 0.92,
                    opacity=alpha * 0.74,
                )
            )

    # Add a stronger central group to reinforce the "room" feel.
    if points.shape[0] > 1:
        center = np.average(points, axis=0, weights=weights)
    else:
        center = points[0]
    center_x, center_y = float(center[0]), float(center[1])
    xs = np.linspace(left, right, 12)
    for x0 in xs:
        lines.append(
            SvgLine(
                points=np.array([[x0, bottom], [center_x, center_y]], dtype=np.float64),
                width=1.1,
                opacity=0.35,
                zorder=5,
            )
        )

    return lines


def draw_perspective_grid(
    ax: plt.Axes,
    points: np.ndarray,
    weights: np.ndarray,
    line_rgb: tuple[float, float, float],
    left: float,
    right: float,
    horizon_y: float,
    bottom: float,
    rng: np.random.Generator,
) -> None:
    for line in collect_perspective_lines(
        points, weights, left, right, horizon_y, bottom, rng
    ):
        ax.plot(
            line.points[:, 0],
            line.points[:, 1],
            color=line_rgb,
            lw=raster_s(line.width),
            alpha=line.opacity,
            zorder=line.zorder,
        )


def draw_noise(
    ax: plt.Axes, left: float, right: float, top: float, bottom: float, seed_val: int
) -> None:
    rng = np.random.default_rng(seed_val)
    h = 260
    w = 260
    noise = rng.normal(0.52, 0.18, size=(h, w))
    noise = np.clip(noise, 0.0, 1.0)
    ax.imshow(
        noise,
        cmap="gray",
        extent=[left, right, bottom, top],
        alpha=0.11,
        interpolation="bicubic",
        zorder=2,
    )


def fmt_num(value: float) -> str:
    text = f"{value:.3f}".rstrip("0").rstrip(".")
    return text or "0"


def rgb_to_hex(rgb: tuple[float, float, float]) -> str:
    vals = [int(np.clip(round(c * 255), 0, 255)) for c in rgb]
    return f"#{vals[0]:02X}{vals[1]:02X}{vals[2]:02X}"


def svg_points(points: np.ndarray) -> str:
    return " ".join(f"{fmt_num(float(x))},{fmt_num(float(y))}" for x, y in points)


def write_svg_foreground(
    svg_path: Path,
    date: str,
    diary_vec: np.ndarray,
    event_vecs: np.ndarray,
    ctx: dict[str, np.ndarray],
    seed: int,
) -> None:
    left = INNER_MARGIN
    right = CANVAS_W - INNER_MARGIN
    top = INNER_MARGIN
    bottom = CANVAS_H - INNER_MARGIN

    style = vec_style(diary_vec, ctx)
    horizon_y = top + (bottom - top) * style["horizon"]
    _, _, line_rgb = style_colors(style)
    line_color = rgb_to_hex(line_rgb)

    date_seed = seed + sum(ord(ch) for ch in date)
    rng = np.random.default_rng(date_seed)
    points, weights = event_vanishing_points(
        event_vecs, diary_vec, left, right, top, bottom, rng
    )
    lines = collect_perspective_lines(
        points, weights, left, right, horizon_y, bottom, rng
    )

    parts = [
        '<?xml version="1.0" encoding="UTF-8"?>',
        (
            f'<svg xmlns="http://www.w3.org/2000/svg" width="{OUTPUT_W}" '
            f'height="{OUTPUT_H}" viewBox="0 0 {CANVAS_W} {CANVAS_H}">'
        ),
        (
            f'<g fill="none" stroke="{line_color}" stroke-opacity="0.95">'
            f'<rect x="{OUTER_MARGIN}" y="{OUTER_MARGIN}" '
            f'width="{CANVAS_W - 2 * OUTER_MARGIN}" '
            f'height="{CANVAS_H - 2 * OUTER_MARGIN}" stroke-width="6" />'
            f'<rect x="{INNER_MARGIN}" y="{INNER_MARGIN}" '
            f'width="{CANVAS_W - 2 * INNER_MARGIN}" '
            f'height="{CANVAS_H - 2 * INNER_MARGIN}" stroke-width="1.4" />'
            "</g>"
        ),
        f'<g fill="none" stroke="{line_color}" stroke-linecap="round" stroke-linejoin="round">',
    ]

    for line in lines:
        parts.append(
            f'<polyline points="{svg_points(line.points)}" '
            f'stroke-width="{fmt_num(line.width)}" '
            f'opacity="{fmt_num(line.opacity)}" />'
        )

    parts.extend(["</g>", "</svg>"])
    svg_path.parent.mkdir(parents=True, exist_ok=True)
    svg_path.write_text("\n".join(parts) + "\n", encoding="utf-8")


def render_one(
    out_path: Path,
    date: str,
    diary_vec: np.ndarray,
    event_vecs: np.ndarray,
    ctx: dict[str, np.ndarray],
    dpi: int,
    seed: int,
) -> None:
    fig = plt.figure(figsize=(OUTPUT_W / dpi, OUTPUT_H / dpi), dpi=dpi)
    ax = fig.add_axes([0, 0, 1, 1])

    left = INNER_MARGIN
    right = CANVAS_W - INNER_MARGIN
    top = INNER_MARGIN
    bottom = CANVAS_H - INNER_MARGIN

    style = vec_style(diary_vec, ctx)
    horizon_y = top + (bottom - top) * style["horizon"]

    date_seed = seed + sum(ord(ch) for ch in date)
    rng = np.random.default_rng(date_seed)

    line_rgb = draw_background(ax, style, left, right, top, bottom, horizon_y)
    points, weights = event_vanishing_points(
        event_vecs, diary_vec, left, right, top, bottom, rng
    )
    draw_perspective_grid(
        ax, points, weights, line_rgb, left, right, horizon_y, bottom, rng
    )
    draw_noise(ax, left, right, top, bottom, date_seed)

    # Double border frame.
    frame_color = (*line_rgb, 0.95)
    ax.add_patch(
        plt.Rectangle(
            (OUTER_MARGIN, OUTER_MARGIN),
            CANVAS_W - 2 * OUTER_MARGIN,
            CANVAS_H - 2 * OUTER_MARGIN,
            fill=False,
            ec=frame_color,
            lw=raster_s(6.0),
            zorder=7,
        )
    )
    ax.add_patch(
        plt.Rectangle(
            (INNER_MARGIN, INNER_MARGIN),
            CANVAS_W - 2 * INNER_MARGIN,
            CANVAS_H - 2 * INNER_MARGIN,
            fill=False,
            ec=frame_color,
            lw=raster_s(1.4),
            zorder=7,
        )
    )

    # Header marks.
    header_font = header_label_font()
    ax.text(
        INNER_MARGIN + HEADER_LEFT_LABEL_X_PADDING,
        HEADER_LABEL_Y,
        f"Events {event_vecs.shape[0]}",
        color=frame_color,
        fontproperties=header_font,
        fontsize=raster_s(HEADER_LABEL_FONT_SIZE),
        ha="left",
        va="center",
    )
    ax.text(
        CANVAS_W - INNER_MARGIN - HEADER_RIGHT_LABEL_X_PADDING,
        HEADER_LABEL_Y,
        date,
        color=frame_color,
        fontproperties=header_font,
        fontsize=raster_s(HEADER_LABEL_FONT_SIZE),
        ha="right",
        va="center",
    )

    ax.set_xlim(0, CANVAS_W)
    ax.set_ylim(CANVAS_H, 0)
    ax.set_aspect("equal")
    ax.axis("off")

    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path, dpi=dpi, facecolor=fig.get_facecolor())
    plt.close(fig)


def main() -> None:
    args = parse_args()
    set_render_sizes(args.content_size, args.resolution)
    script_dir = Path(__file__).resolve().parent
    root = (
        script_dir.parent if args.root is None else script_dir / args.root
    ).resolve()
    if args.out_dir is None:
        out_base = root / "output_All" / script_dir.name
    else:
        out_base = Path(args.out_dir)
        if not out_base.is_absolute():
            out_base = root / out_base

    dates = discover_dates(root, args.date)
    if args.count is not None:
        dates = dates[: args.count]
    context_dates = discover_dates(root, None)
    ctx = build_style_context(root, context_dates)
    run_out, run_svg = next_try_dirs(
        out_base, export_png=not args.svg_only, export_svg=not args.png_only
    )

    out_parts = []
    if run_out is not None:
        out_parts.append(f"PNG: {display_path(run_out, root)}")
    if run_svg is not None:
        out_parts.append(f"SVG: {display_path(run_svg, root)}")
    print(f"Rendering {len(dates)} date(s). Output directories: {'; '.join(out_parts)}")

    for i, date in enumerate(dates, start=1):
        diary_vec = (
            np.load(root / "diary_vectors" / f"{date}.npy")
            .reshape(-1)
            .astype(np.float64)
        )
        event_vecs = np.load(
            root / "diary_sentence_vectors" / date / "window_vectors.npy"
        ).astype(np.float64)
        saved: list[str] = []
        if run_out is not None:
            out_file = run_out / f"{date}.png"
            render_one(out_file, date, diary_vec, event_vecs, ctx, args.dpi, args.seed)
            saved.append(out_file.name)
        if run_svg is not None:
            svg_file = run_svg / f"{date}.svg"
            write_svg_foreground(svg_file, date, diary_vec, event_vecs, ctx, args.seed)
            saved.append(svg_file.name)
        print(f"[{i:02d}/{len(dates):02d}] Saved: {', '.join(saved)}")


if __name__ == "__main__":
    main()
