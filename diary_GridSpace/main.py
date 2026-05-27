#!/usr/bin/env python3
"""Generate a PerspectiveSpace-style grid cutout from diary vectors.

Usage:
  # Batch mode: generate all dates
  python3 main.py
  # Single date
  python3 main.py --date 2026-03-10
  # First 100 dates, PNG only
  python3 main.py --count 100 --png-only
  # Stable 1000x1000 content exported at 2000x2000 pixels
  python3 main.py --date 2026-03-10 --content-size 1000 --resolution 2000

Outputs:
  ../output_All/diary_GridSpace/try_N/<date>.png
  ../output_All/diary_GridSpace/svg_try_N/<date>.svg
"""

from __future__ import annotations

import argparse
import json
import re
from dataclasses import dataclass
from pathlib import Path

import numpy as np

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.path import Path as MplPath
from sklearn.decomposition import PCA

BG_COLOR = "#EDE7E0"
LINE_COLOR = "#CD3700"
FRAME_COLOR = "#CD3700"
TEXT_COLOR = "#8C8C8C"

BASE_CONTENT_SIZE = 1000
DEFAULT_CONTENT_SIZE = 1000
DEFAULT_RESOLUTION = 2160
BASE_MARGIN = 82
BASE_GRID_STEP = 38


@dataclass(frozen=True)
class RenderSpec:
    content_w: int
    content_h: int
    output_w: int
    output_h: int

    @property
    def scale(self) -> float:
        return self.content_w / BASE_CONTENT_SIZE

    @property
    def margin(self) -> float:
        return BASE_MARGIN * self.scale

    @property
    def grid_step(self) -> float:
        return BASE_GRID_STEP * self.scale

    def s(self, value: float) -> float:
        return value * self.scale

    @property
    def raster_scale(self) -> float:
        return self.output_w / self.content_w

    def raster_s(self, value: float) -> float:
        return value * self.raster_scale


@dataclass(frozen=True)
class SvgPolyline:
    points: np.ndarray
    width: float
    opacity: float


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Generate PerspectiveSpace-style figure"
    )
    parser.add_argument(
        "--date", help="Diary date, e.g. 2026-03-10. Omit for batch mode."
    )
    parser.add_argument(
        "--count",
        type=int,
        default=None,
        help="Generate only the first N valid dates in batch mode.",
    )
    output_group = parser.add_mutually_exclusive_group()
    output_group.add_argument(
        "--svg-only",
        action="store_true",
        help="Only render SVGs into svg_try_N; skip PNG output.",
    )
    output_group.add_argument(
        "--with-svg",
        action="store_true",
        help="Render both PNGs into try_N and SVGs into svg_try_N. This is the default.",
    )
    output_group.add_argument(
        "--png-only",
        action="store_true",
        help="Only render PNGs into try_N; skip SVG output.",
    )
    parser.add_argument(
        "--content-size",
        type=int,
        default=DEFAULT_CONTENT_SIZE,
        help=(
            "Content coordinate size. Changing this changes the layout scale; "
            f"default is {DEFAULT_CONTENT_SIZE}."
        ),
    )
    parser.add_argument(
        "--resolution",
        "--size",
        dest="resolution",
        type=int,
        default=DEFAULT_RESOLUTION,
        help=(
            "Final square PNG pixel resolution and SVG width/height. --size is "
            "kept as a compatibility alias."
        ),
    )
    parser.add_argument(
        "--root",
        default=None,
        help="Project root. Defaults to parent of this script.",
    )
    parser.add_argument(
        "--out-dir",
        default=None,
        help="Output base dir. Defaults to <root>/output_All/<this program folder>.",
    )
    parser.add_argument("--seed", type=int, default=7, help="Random seed")
    parser.add_argument("--dpi", type=int, default=220, help="Matplotlib render DPI")
    args = parser.parse_args()
    if args.count is not None and args.count < 1:
        parser.error("--count must be a positive integer")
    if args.content_size < 1:
        parser.error("--content-size must be a positive integer")
    if args.resolution < 1:
        parser.error("--resolution must be a positive integer")
    return args


def discover_dates(root: Path, only_date: str | None = None) -> list[str]:
    diary_vec_dir = root / "diary_vectors"
    sentence_dir = root / "diary_sentence_vectors"
    if not diary_vec_dir.exists() or not sentence_dir.exists():
        raise FileNotFoundError(
            f"Missing vector dirs under {root}. Need both diary_vectors and diary_sentence_vectors."
        )

    if only_date:
        candidates = [only_date]
    else:
        candidates = sorted(p.stem for p in diary_vec_dir.glob("*.npy"))

    valid_dates: list[str] = []
    missing_sentence_dates: list[str] = []
    for date in candidates:
        sdir = sentence_dir / date
        required = ["diary_vector.npy", "sentence_vectors.npy", "meta.json"]
        if sdir.exists() and all((sdir / name).exists() for name in required):
            valid_dates.append(date)
        else:
            missing_sentence_dates.append(date)

    if only_date and not valid_dates:
        raise FileNotFoundError(
            f"Date {only_date} is missing required files in {sentence_dir / only_date}"
        )

    if missing_sentence_dates:
        print(
            "Skipped dates missing sentence-vector files:",
            ", ".join(missing_sentence_dates[:10])
            + (" ..." if len(missing_sentence_dates) > 10 else ""),
        )

    if not valid_dates:
        raise RuntimeError(f"No valid dates found under {root}")

    return valid_dates


def next_try_dirs(
    base_dir: Path, export_png: bool = True, export_svg: bool = True
) -> tuple[Path | None, Path | None]:
    base_dir.mkdir(parents=True, exist_ok=True)
    max_idx = 0
    for item in base_dir.iterdir():
        if not item.is_dir():
            continue
        m = re.fullmatch(r"(?:svg_)?try_(\d+)", item.name)
        if m:
            max_idx = max(max_idx, int(m.group(1)))
    next_idx = max_idx + 1

    png_dir = base_dir / f"try_{next_idx}" if export_png else None
    svg_dir = base_dir / f"svg_try_{next_idx}" if export_svg else None

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


def load_vectors(root: Path, date: str) -> tuple[np.ndarray, np.ndarray, dict]:
    sentence_dir = root / "diary_sentence_vectors" / date
    if not sentence_dir.exists():
        raise FileNotFoundError(f"Missing directory: {sentence_dir}")

    diary_vector = (
        np.load(sentence_dir / "diary_vector.npy").reshape(-1).astype(np.float64)
    )
    sentence_vectors = np.load(sentence_dir / "sentence_vectors.npy").astype(np.float64)

    with (sentence_dir / "meta.json").open("r", encoding="utf-8") as f:
        meta = json.load(f)

    return diary_vector, sentence_vectors, meta


def l2_normalize(x: np.ndarray, axis: int = -1) -> np.ndarray:
    n = np.linalg.norm(x, axis=axis, keepdims=True)
    n = np.clip(n, 1e-12, None)
    return x / n


def build_semantic_layout(
    diary_vec: np.ndarray, sentence_vecs: np.ndarray, rng: np.random.Generator
) -> tuple[np.ndarray, np.ndarray]:
    n_sent = sentence_vecs.shape[0]
    if n_sent == 1:
        # Single-sentence entry: synthesize a small cluster to keep visuals stable.
        anchors = np.array([[0.0, 0.0], [0.17, -0.08], [-0.14, 0.09]])
        weights = np.array([1.0, 0.75, 0.72])
        return anchors, weights

    X = np.vstack([diary_vec.reshape(1, -1), sentence_vecs])
    pca = PCA(n_components=2, random_state=42)
    Z = pca.fit_transform(X)
    diary_xy = Z[0]
    sent_xy = Z[1:] - diary_xy

    # Robust scaling and gentle jitter avoids accidental straight-line degeneracy.
    scale = np.percentile(np.linalg.norm(sent_xy, axis=1), 85)
    scale = max(scale, 1e-6)
    sent_xy = sent_xy / scale
    sent_xy += rng.normal(0.0, 0.03, size=sent_xy.shape)

    # Weight by diary-sentence cosine similarity.
    dv = l2_normalize(diary_vec)
    sv = l2_normalize(sentence_vecs, axis=1)
    sims = (sv @ dv).reshape(-1)
    sims = (sims - sims.min()) / (max(sims.max() - sims.min(), 1e-9))
    weights = 0.45 + 0.85 * sims

    return sent_xy, weights


def semantic_fields(
    points: np.ndarray,
    weights: np.ndarray,
    xx: np.ndarray,
    yy: np.ndarray,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    px = points[:, 0].reshape(-1, 1, 1)
    py = points[:, 1].reshape(-1, 1, 1)

    dx = xx[None, :, :] - px
    dy = yy[None, :, :] - py
    r2 = dx * dx + dy * dy

    sigma_hole = 0.22
    sigma_warp = 0.34

    # Scalar field controls cutout region.
    hole_field = (weights[:, None, None] * np.exp(-r2 / (2 * sigma_hole**2))).sum(
        axis=0
    )

    # Warp potential controls displacement around region.
    warp_field = (weights[:, None, None] * np.exp(-r2 / (2 * sigma_warp**2))).sum(
        axis=0
    )

    # Weighted centroid as gravity center for more "pool-like" cavity.
    center = (points * weights[:, None]).sum(axis=0) / np.clip(
        weights.sum(), 1e-9, None
    )
    return hole_field, warp_field, center


def pick_hole_polygon(
    hole_field: np.ndarray,
    xx: np.ndarray,
    yy: np.ndarray,
    center: np.ndarray,
) -> np.ndarray:
    lo = np.percentile(hole_field, 70)
    hi = np.percentile(hole_field, 95)
    level = 0.68 * lo + 0.32 * hi

    fig_tmp, ax_tmp = plt.subplots()
    cset = ax_tmp.contour(xx, yy, hole_field, levels=[level])

    best_poly = None
    best_score = -1.0

    if cset.allsegs and cset.allsegs[0]:
        for seg in cset.allsegs[0]:
            if seg.shape[0] < 12:
                continue
            area = polygon_area(seg)
            if area < 0.01:
                continue
            # Prefer contours closer to semantic center and with larger area.
            seg_center = seg.mean(axis=0)
            dist = float(np.linalg.norm(seg_center - center))
            score = area / (0.2 + dist)
            if score > best_score:
                best_score = score
                best_poly = seg

    plt.close(fig_tmp)

    if best_poly is None:
        # fallback ellipse-like shape around center
        t = np.linspace(0, 2 * np.pi, 200)
        rx, ry = 0.43, 0.63
        best_poly = np.column_stack(
            [
                center[0] + rx * np.cos(t) * (1 + 0.07 * np.cos(3 * t)),
                center[1] + ry * np.sin(t),
            ]
        )

    return smooth_closed_poly(best_poly, rounds=2)


def polygon_area(poly: np.ndarray) -> float:
    x = poly[:, 0]
    y = poly[:, 1]
    return 0.5 * abs(np.dot(x, np.roll(y, -1)) - np.dot(y, np.roll(x, -1)))


def smooth_closed_poly(poly: np.ndarray, rounds: int = 1) -> np.ndarray:
    out = poly.copy()
    for _ in range(rounds):
        out = 0.2 * np.roll(out, -1, axis=0) + 0.6 * out + 0.2 * np.roll(out, 1, axis=0)
    return out


def normalized_to_canvas(
    x: np.ndarray, y: np.ndarray, spec: RenderSpec
) -> tuple[np.ndarray, np.ndarray]:
    inner_w = spec.content_w - 2 * spec.margin
    inner_h = spec.content_h - 2 * spec.margin
    # normalize-space roughly [-1.2, 1.2] in x and [-1.5, 1.5] in y
    xpix = spec.margin + (x + 1.2) / 2.4 * inner_w
    ypix = spec.margin + (y + 1.5) / 3.0 * inner_h
    return xpix, ypix


def warp_points(
    x: np.ndarray,
    y: np.ndarray,
    hole_poly: np.ndarray,
    warp_field: np.ndarray,
    xx: np.ndarray,
    yy: np.ndarray,
    center: np.ndarray,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    # radial basis from semantic center
    dx = x - center[0]
    dy = y - center[1]
    rr = np.sqrt(dx * dx + dy * dy) + 1e-6

    # interpolate warp field to current points by nearest-neighbor on dense grid
    nx, ny = xx.shape[1], xx.shape[0]
    ix = np.clip(((x + 1.3) / 2.6 * (nx - 1)).astype(int), 0, nx - 1)
    iy = np.clip(((y + 1.6) / 3.2 * (ny - 1)).astype(int), 0, ny - 1)
    amp = warp_field[iy, ix]
    amp = (amp - np.percentile(warp_field, 10)) / max(
        np.percentile(warp_field, 95) - np.percentile(warp_field, 10), 1e-9
    )
    amp = np.clip(amp, 0.0, 1.0)

    # stronger near hole boundary, weaker far away
    boundary = MplPath(hole_poly)
    inside = boundary.contains_points(np.column_stack([x, y]))

    # signed distance approximation by distance to contour vertices
    p = np.column_stack([x, y])
    d = np.sqrt(((p[:, None, :] - hole_poly[None, :, :]) ** 2).sum(axis=2)).min(axis=1)
    edge_gain = np.exp(-((d / 0.18) ** 2))

    push = (0.045 + 0.09 * edge_gain) * amp
    push = np.where(inside, -0.05 * push, push)

    xw = x + dx / rr * push
    yw = y + dy / rr * push

    # mild swirl to mimic hand-pulled perspective tension
    swirl = 0.028 * amp * np.exp(-((rr / 1.35) ** 2))
    xw2 = xw - dy * swirl
    yw2 = yw + dx * swirl

    return xw2, yw2, inside


def collect_foreground_polylines(
    hole_poly: np.ndarray,
    warp_field: np.ndarray,
    xx: np.ndarray,
    yy: np.ndarray,
    center: np.ndarray,
    hole_field: np.ndarray,
    spec: RenderSpec,
) -> list[SvgPolyline]:
    polylines: list[SvgPolyline] = []

    x_vals = np.linspace(
        -1.2, 1.2, int((spec.content_w - 2 * spec.margin) / spec.grid_step) + 1
    )
    y_vals = np.linspace(
        -1.5, 1.5, int((spec.content_h - 2 * spec.margin) / spec.grid_step) + 1
    )

    hole_path = MplPath(hole_poly)

    def add_curve(x_line: np.ndarray, y_line: np.ndarray, lw: float = 1.05) -> None:
        xw, yw, _ = warp_points(x_line, y_line, hole_poly, warp_field, xx, yy, center)
        pts = np.column_stack([xw, yw])
        inside = hole_path.contains_points(pts)

        xp, yp = normalized_to_canvas(xw, yw, spec)
        start = 0
        n = len(xp)
        while start < n:
            while start < n and inside[start]:
                start += 1
            end = start
            while end < n and not inside[end]:
                end += 1
            if end - start >= 2:
                polylines.append(
                    SvgPolyline(
                        points=np.column_stack([xp[start:end], yp[start:end]]),
                        width=spec.s(lw),
                        opacity=0.97,
                    )
                )
            start = end + 1

    dense = 280

    for xv in x_vals:
        yline = np.linspace(-1.5, 1.5, dense)
        xline = np.full_like(yline, xv)
        add_curve(xline, yline, lw=0.95)

    for yv in y_vals:
        xline = np.linspace(-1.2, 1.2, dense)
        yline = np.full_like(xline, yv)
        add_curve(xline, yline, lw=0.95)

    levels = np.percentile(hole_field, [62, 70, 76, 82, 87])
    fig_tmp, ax_tmp = plt.subplots()
    contour_set = ax_tmp.contour(
        *normalized_to_canvas(xx, yy, spec),
        hole_field,
        levels=levels,
    )
    for level_segments in contour_set.allsegs:
        for segment in level_segments:
            if segment.shape[0] >= 2:
                polylines.append(
                    SvgPolyline(points=segment, width=spec.s(0.6), opacity=0.78)
                )
    plt.close(fig_tmp)

    hx, hy = normalized_to_canvas(hole_poly[:, 0], hole_poly[:, 1], spec)
    edge_points = np.column_stack([hx, hy])
    if not np.allclose(edge_points[0], edge_points[-1]):
        edge_points = np.vstack([edge_points, edge_points[0]])
    polylines.append(SvgPolyline(points=edge_points, width=spec.s(2.4), opacity=1.0))

    return polylines


def fmt_num(value: float) -> str:
    text = f"{value:.3f}".rstrip("0").rstrip(".")
    return text or "0"


def svg_points(points: np.ndarray) -> str:
    return " ".join(f"{fmt_num(float(x))},{fmt_num(float(y))}" for x, y in points)


def write_svg_foreground(
    svg_path: Path,
    hole_poly: np.ndarray,
    warp_field: np.ndarray,
    xx: np.ndarray,
    yy: np.ndarray,
    center: np.ndarray,
    hole_field: np.ndarray,
    spec: RenderSpec,
) -> None:
    polylines = collect_foreground_polylines(
        hole_poly=hole_poly,
        warp_field=warp_field,
        xx=xx,
        yy=yy,
        center=center,
        hole_field=hole_field,
        spec=spec,
    )

    parts = [
        '<?xml version="1.0" encoding="UTF-8"?>',
        (
            f'<svg xmlns="http://www.w3.org/2000/svg" width="{spec.output_w}" '
            f'height="{spec.output_h}" viewBox="0 0 {spec.content_w} {spec.content_h}">'
        ),
        (
            f'<g fill="none" stroke="{FRAME_COLOR}">'
            f'<rect x="{fmt_num(spec.s(36))}" y="{fmt_num(spec.s(36))}" '
            f'width="{fmt_num(spec.content_w - spec.s(72))}" '
            f'height="{fmt_num(spec.content_h - spec.s(72))}" '
            f'stroke-width="{fmt_num(spec.s(7))}" />'
            f'<rect x="{fmt_num(spec.s(58))}" y="{fmt_num(spec.s(58))}" '
            f'width="{fmt_num(spec.content_w - spec.s(116))}" '
            f'height="{fmt_num(spec.content_h - spec.s(116))}" '
            f'stroke-width="{fmt_num(spec.s(2))}" />'
            "</g>"
        ),
        f'<g fill="none" stroke="{LINE_COLOR}" stroke-linecap="round" '
        f'stroke-linejoin="round" transform="matrix(1 0 0 -1 0 {spec.content_h})">',
    ]

    for line in polylines:
        parts.append(
            f'<polyline points="{svg_points(line.points)}" '
            f'stroke-width="{fmt_num(line.width)}" '
            f'opacity="{fmt_num(line.opacity)}" />'
        )

    parts.extend(["</g>", "</svg>"])

    svg_path.parent.mkdir(parents=True, exist_ok=True)
    svg_path.write_text("\n".join(parts) + "\n", encoding="utf-8")


def plot_grid_with_hole(
    hole_poly: np.ndarray,
    warp_field: np.ndarray,
    xx: np.ndarray,
    yy: np.ndarray,
    center: np.ndarray,
    hole_field: np.ndarray,
    meta: dict,
    date: str,
    out_path: Path,
    dpi: int,
    spec: RenderSpec,
) -> None:
    fig = plt.figure(figsize=(spec.output_w / dpi, spec.output_h / dpi), dpi=dpi)
    ax = fig.add_axes([0, 0, 1, 1])
    fig.patch.set_facecolor(BG_COLOR)
    ax.set_facecolor(BG_COLOR)

    # Decorative frame
    ax.add_patch(
        plt.Rectangle(
            (spec.s(36), spec.s(36)),
            spec.content_w - spec.s(72),
            spec.content_h - spec.s(72),
            fill=False,
            lw=spec.raster_s(7),
            ec=FRAME_COLOR,
        )
    )
    ax.add_patch(
        plt.Rectangle(
            (spec.s(58), spec.s(58)),
            spec.content_w - spec.s(116),
            spec.content_h - spec.s(116),
            fill=False,
            lw=spec.raster_s(2.0),
            ec=FRAME_COLOR,
        )
    )

    # render hole as background-colored shape
    hx, hy = normalized_to_canvas(hole_poly[:, 0], hole_poly[:, 1], spec)
    hole_patch = plt.Polygon(
        np.column_stack([hx, hy]), closed=True, fc=BG_COLOR, ec="none", zorder=6
    )
    ax.add_patch(hole_patch)

    # Grid lines: create in normalized space then warp then draw outside hole.
    x_vals = np.linspace(
        -1.2, 1.2, int((spec.content_w - 2 * spec.margin) / spec.grid_step) + 1
    )
    y_vals = np.linspace(
        -1.5, 1.5, int((spec.content_h - 2 * spec.margin) / spec.grid_step) + 1
    )

    hole_path = MplPath(hole_poly)

    def draw_curve(x_line: np.ndarray, y_line: np.ndarray, lw: float = 1.05) -> None:
        xw, yw, _ = warp_points(x_line, y_line, hole_poly, warp_field, xx, yy, center)
        pts = np.column_stack([xw, yw])
        inside = hole_path.contains_points(pts)

        xp, yp = normalized_to_canvas(xw, yw, spec)
        start = 0
        n = len(xp)
        while start < n:
            while start < n and inside[start]:
                start += 1
            end = start
            while end < n and not inside[end]:
                end += 1
            if end - start >= 2:
                ax.plot(
                    xp[start:end],
                    yp[start:end],
                    color=LINE_COLOR,
                    lw=spec.raster_s(lw),
                    alpha=0.97,
                    zorder=3,
                )
            start = end + 1

    dense = 280

    for xv in x_vals:
        yline = np.linspace(-1.5, 1.5, dense)
        xline = np.full_like(yline, xv)
        draw_curve(xline, yline, lw=0.95)

    for yv in y_vals:
        xline = np.linspace(-1.2, 1.2, dense)
        yline = np.full_like(xline, yv)
        draw_curve(xline, yline, lw=0.95)

    # contour rings around the cutout to emphasize carved depth
    levels = np.percentile(hole_field, [62, 70, 76, 82, 87])
    cset = ax.contour(
        *normalized_to_canvas(xx, yy, spec),
        hole_field,
        levels=levels,
        colors=[LINE_COLOR],
        linewidths=spec.raster_s(0.6),
        alpha=0.78,
        zorder=5,
    )
    _ = cset

    # main hole edge
    ax.plot(hx, hy, color=LINE_COLOR, lw=spec.raster_s(2.4), zorder=8)

    # edition-like texts
    num_sent = meta.get("num_sentences", 0)
    ax.text(
        spec.s(70),
        spec.s(25),
        f"1/{max(num_sent,1)}",
        color=TEXT_COLOR,
        fontsize=spec.raster_s(10),
        style="italic",
    )
    ax.text(
        spec.content_w - spec.s(260),
        spec.s(26),
        date,
        color=TEXT_COLOR,
        fontsize=spec.raster_s(10),
        style="italic",
    )

    ax.set_xlim(0, spec.content_w)
    ax.set_ylim(0, spec.content_h)
    ax.set_aspect("equal")
    ax.axis("off")

    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path, dpi=dpi, facecolor=BG_COLOR)
    plt.close(fig)


def main() -> None:
    args = parse_args()
    script_dir = Path(__file__).resolve().parent
    root = Path(args.root).expanduser().resolve() if args.root else script_dir.parent
    out_base = (
        Path(args.out_dir).expanduser().resolve()
        if args.out_dir
        else root / "output_All" / script_dir.name
    )
    rng = np.random.default_rng(args.seed)
    dates = discover_dates(root, args.date)
    if args.count is not None:
        dates = dates[: args.count]
    spec = RenderSpec(
        content_w=args.content_size,
        content_h=args.content_size,
        output_w=args.resolution,
        output_h=args.resolution,
    )

    # Dense normalized field for mask + warp (shared for all dates).
    x = np.linspace(-1.3, 1.3, 360)
    y = np.linspace(-1.6, 1.6, 460)
    xx, yy = np.meshgrid(x, y)

    run_out_dir, run_svg_dir = next_try_dirs(
        out_base, export_png=not args.svg_only, export_svg=not args.png_only
    )
    out_parts = []
    if run_out_dir is not None:
        out_parts.append(f"PNG: {display_path(run_out_dir, root)}")
    if run_svg_dir is not None:
        out_parts.append(f"SVG: {display_path(run_svg_dir, root)}")
    print(f"Found {len(dates)} date(s). Output directories: {'; '.join(out_parts)}")

    for i, date in enumerate(dates, start=1):
        diary_vec, sentence_vecs, meta = load_vectors(root, date)
        points, weights = build_semantic_layout(diary_vec, sentence_vecs, rng)
        hole_field, warp_field, center = semantic_fields(points, weights, xx, yy)
        hole_poly = pick_hole_polygon(hole_field, xx, yy, center)

        saved: list[str] = []
        if run_out_dir is not None:
            out_file = run_out_dir / f"{date}.png"
            plot_grid_with_hole(
                hole_poly=hole_poly,
                warp_field=warp_field,
                xx=xx,
                yy=yy,
                center=center,
                hole_field=hole_field,
                meta=meta,
                date=date,
                out_path=out_file,
                dpi=args.dpi,
                spec=spec,
            )
            saved.append(out_file.name)

        if run_svg_dir is not None:
            svg_file = run_svg_dir / f"{date}.svg"
            write_svg_foreground(
                svg_path=svg_file,
                hole_poly=hole_poly,
                warp_field=warp_field,
                xx=xx,
                yy=yy,
                center=center,
                hole_field=hole_field,
                spec=spec,
            )
            saved.append(svg_file.name)

        print(f"[{i}/{len(dates)}] Saved: {', '.join(saved)}")


if __name__ == "__main__":
    main()
