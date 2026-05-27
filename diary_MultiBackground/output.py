"""Plot, CSV, and summary output helpers for MultiBackground."""

from __future__ import annotations

import csv
import re
import shutil
import textwrap
from collections import Counter
from pathlib import Path
from typing import Any

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.patches import Polygon

from diary_MultiBackground.config import *  # noqa: F403


def convex_hull(points: np.ndarray) -> np.ndarray:
    pts = sorted(set(map(tuple, points)))
    if len(pts) <= 2:
        return np.asarray(pts, dtype=np.float32)

    def cross(o, a, b):
        return (a[0] - o[0]) * (b[1] - o[1]) - (a[1] - o[1]) * (b[0] - o[0])

    lower = []
    for p in pts:
        while len(lower) >= 2 and cross(lower[-2], lower[-1], p) <= 0:
            lower.pop()
        lower.append(p)
    upper = []
    for p in reversed(pts):
        while len(upper) >= 2 and cross(upper[-2], upper[-1], p) <= 0:
            upper.pop()
        upper.append(p)
    return np.asarray(lower[:-1] + upper[:-1], dtype=np.float32)


def central_points(points: np.ndarray, keep_ratio: float) -> np.ndarray:
    if len(points) <= 3 or keep_ratio >= 1.0:
        return points
    if keep_ratio <= 0.0:
        keep_ratio = 1.0
    keep_count = max(3, min(len(points), int(np.ceil(len(points) * keep_ratio))))
    center = np.median(points, axis=0, keepdims=True)
    distances = np.linalg.norm(points - center, axis=1)
    keep_indexes = np.argpartition(distances, keep_count - 1)[:keep_count]
    return points[keep_indexes]


def hull_boundary(points: np.ndarray) -> np.ndarray:
    return convex_hull(central_points(points, HULL_KEEP_RATIO))


def hull_point_count(points: np.ndarray) -> int:
    if len(points) <= 3 or HULL_KEEP_RATIO >= 1.0:
        return len(points)
    if HULL_KEEP_RATIO <= 0.0:
        return len(points)
    return max(3, min(len(points), int(np.ceil(len(points) * HULL_KEEP_RATIO))))


def polygon_area(points: np.ndarray) -> float:
    if len(points) < 3:
        return 0.0
    x, y = points[:, 0], points[:, 1]
    return float(0.5 * abs(np.dot(x, np.roll(y, 1)) - np.dot(y, np.roll(x, 1))))


def bbox_area(points: np.ndarray) -> float:
    span = points.max(axis=0) - points.min(axis=0)
    return float(span[0] * span[1])


def scaled_area_points(points: np.ndarray, area_ratio: float) -> np.ndarray:
    if area_ratio <= 0.0 or area_ratio == 1.0:
        return points
    center = points.mean(axis=0, keepdims=True)
    linear_scale = float(np.sqrt(area_ratio))
    return center + (points - center) * linear_scale


def square_limits(points: np.ndarray, pad_ratio: float) -> tuple[tuple[float, float], tuple[float, float]]:
    mn = points.min(axis=0)
    mx = points.max(axis=0)
    center = (mn + mx) / 2
    span = max(float((mx - mn).max()), 1e-6)
    half = span * (0.5 + pad_ratio)
    return (
        (float(center[0] - half), float(center[0] + half)),
        (float(center[1] - half), float(center[1] + half)),
    )


def set_square_limits(ax, points: np.ndarray, pad_ratio: float = 0.06):
    xlim, ylim = square_limits(points, pad_ratio)
    ax.set_xlim(*xlim)
    ax.set_ylim(*ylim)
    return xlim, ylim


def points_inside_limits(
    points: np.ndarray,
    xlim: tuple[float, float],
    ylim: tuple[float, float],
    edge_inset_ratio: float = 0.0,
) -> np.ndarray:
    x_pad = (xlim[1] - xlim[0]) * edge_inset_ratio
    y_pad = (ylim[1] - ylim[0]) * edge_inset_ratio
    return (
        (points[:, 0] >= xlim[0] + x_pad)
        & (points[:, 0] <= xlim[1] - x_pad)
        & (points[:, 1] >= ylim[0] + y_pad)
        & (points[:, 1] <= ylim[1] - y_pad)
    )


def apply_figure_margins(fig):
    fig.subplots_adjust(
        left=FIGURE_MARGIN_RATIO,
        right=1 - FIGURE_MARGIN_RATIO,
        bottom=FIGURE_MARGIN_RATIO,
        top=FIGURE_TITLE_TOP,
    )


def set_plot_title(ax, title: str):
    wrapped_title = "\n".join(
        textwrap.wrap(title, width=66, break_long_words=False, break_on_hyphens=False)
    )
    ax.set_title(
        wrapped_title,
        color=TEXT_COLOR,
        fontfamily=TITLE_FONT_FAMILY,
        fontsize=raster_s(TITLE_FONT_SIZE_PT),
        linespacing=TITLE_LINE_HEIGHT_PT / TITLE_FONT_SIZE_PT,
        pad=raster_s(8),
    )


def raster_scale() -> float:
    return OUTPUT_SIZE / CANVAS_SIZE


def raster_s(value: float) -> float:
    return value * raster_scale()


def raster_area(value: float) -> float:
    scale = raster_scale()
    return value * scale * scale


def output_image_formats() -> list[str]:
    formats = OUTPUT_IMAGE_FORMATS
    if isinstance(formats, str):
        formats = [formats]
    cleaned = []
    for fmt in formats:
        fmt = str(fmt).lower().lstrip(".")
        if fmt not in {"svg", "png"}:
            raise ValueError(f"Unsupported output image format: {fmt}")
        if fmt not in cleaned:
            cleaned.append(fmt)
    return cleaned


def save_figure_all_formats(fig, stem: Path):
    for fmt in output_image_formats():
        if fmt == "svg":
            facecolor = "none"
            transparent = True
        else:
            facecolor = "white"
            transparent = False
        out_path = stem.with_suffix(f".{fmt}")
        fig.savefig(
            out_path,
            facecolor=facecolor,
            edgecolor="none",
            transparent=transparent,
            dpi=DPI,
            format=fmt,
        )
        if fmt == "png":
            copy_png_to_overview_folder(out_path)
        elif OUTPUT_SIZE != CANVAS_SIZE:
            set_svg_display_size(out_path, OUTPUT_SIZE, OUTPUT_SIZE)


def set_svg_display_size(svg_path: Path, width: int, height: int):
    text = svg_path.read_text(encoding="utf-8")
    text = re.sub(r'width="[^"]+"', f'width="{width}"', text, count=1)
    text = re.sub(r'height="[^"]+"', f'height="{height}"', text, count=1)
    svg_path.write_text(text, encoding="utf-8")


def copy_png_to_overview_folder(png_path: Path):
    run_dir = png_path.parent.parent
    if not run_dir.name.startswith("try_"):
        return
    overview_dir = run_dir / "png"
    if png_path.parent == overview_dir:
        return
    overview_dir.mkdir(parents=True, exist_ok=True)
    overview_name = f"{png_path.parent.name}__{png_path.name}"
    shutil.copy2(png_path, overview_dir / overview_name)


def rgb_to_hex(rgb: tuple[int, int, int]) -> str:
    return "#{:02x}{:02x}{:02x}".format(*rgb)


def mix_rgb(
    rgb: tuple[int, int, int],
    target: tuple[int, int, int],
    amount: float,
) -> tuple[int, int, int]:
    return tuple(
        round(channel + (target_channel - channel) * amount)
        for channel, target_channel in zip(rgb, target)
    )


def background_theme(bg_key: str) -> dict[str, str]:
    keys = list(BACKGROUND_SPECS.keys())
    try:
        index = keys.index(bg_key)
    except ValueError:
        index = sum(ord(char) for char in bg_key)
    rgb = THEME_RGB_COLORS[index % len(THEME_RGB_COLORS)]
    return {
        "background_point": rgb_to_hex(mix_rgb(rgb, (255, 255, 255), 0.18)),
        "diary_point": DIARY_COLOR,
        "hull": rgb_to_hex(mix_rgb(rgb, (0, 0, 0), 0.28)),
    }


def plot_world(
    out_stem: Path,
    bg_key: str,
    background_xy: np.ndarray,
    diary_xy: np.ndarray,
    diary_labels: list[str],
    title: str,
):
    theme = background_theme(bg_key)
    fig, ax = plt.subplots(figsize=(OUTPUT_SIZE / DPI, OUTPUT_SIZE / DPI), dpi=DPI)
    fig.patch.set_facecolor("none")
    ax.set_facecolor("none")
    ax.scatter(
        background_xy[:, 0],
        background_xy[:, 1],
        s=raster_area(BACKGROUND_POINT_SIZE),
        c=theme["background_point"],
        alpha=BACKGROUND_ALPHA,
        linewidths=0,
    )
    ax.scatter(
        diary_xy[:, 0],
        diary_xy[:, 1],
        s=raster_area(DIARY_POINT_SIZE),
        c=theme["diary_point"],
        alpha=DIARY_ALPHA,
        linewidths=0,
    )

    hull = hull_boundary(diary_xy)
    if len(hull) >= 3:
        ax.add_patch(
            Polygon(
                hull,
                closed=True,
                fill=False,
                edgecolor=theme["hull"],
                linewidth=raster_s(DIARY_HULL_LINEWIDTH),
                alpha=1.0,
            )
        )
    if SHOW_DIARY_LABEL_EVERY:
        for i in range(0, len(diary_xy), SHOW_DIARY_LABEL_EVERY):
            ax.text(
                diary_xy[i, 0],
                diary_xy[i, 1],
                diary_labels[i],
                fontsize=raster_s(4.8),
                color=theme["hull"],
            )

    set_square_limits(ax, np.vstack([background_xy, diary_xy]), pad_ratio=WORLD_PAD_RATIO)
    set_plot_title(ax, title)
    ax.set_xticks([])
    ax.set_yticks([])
    for spine in ax.spines.values():
        spine.set_visible(False)
    apply_figure_margins(fig)
    save_figure_all_formats(fig, out_stem)
    plt.close(fig)


def plot_world_diary_area_scaled(
    out_stem: Path,
    bg_key: str,
    background_xy: np.ndarray,
    diary_xy: np.ndarray,
    diary_labels: list[str],
    title: str,
):
    scaled_diary_xy = scaled_area_points(diary_xy, DIARY_AREA_SCALE_RATIO)
    scale_label = f"{DIARY_AREA_SCALE_RATIO * 100:.0f}%"
    plot_world(
        out_stem,
        bg_key,
        background_xy,
        scaled_diary_xy,
        diary_labels,
        f"{title} (diary visual area {scale_label})",
    )


def plot_zoom(
    out_stem: Path,
    bg_key: str,
    background_xy: np.ndarray,
    diary_xy: np.ndarray,
    title: str,
):
    theme = background_theme(bg_key)
    fig, ax = plt.subplots(figsize=(OUTPUT_SIZE / DPI, OUTPUT_SIZE / DPI), dpi=DPI)
    fig.patch.set_facecolor("none")
    ax.set_facecolor("none")
    xlim, ylim = square_limits(diary_xy, ZOOM_PAD_RATIO)
    background_mask = points_inside_limits(
        background_xy,
        xlim,
        ylim,
        edge_inset_ratio=ZOOM_EDGE_INSET_RATIO,
    )
    visible_background_xy = background_xy[background_mask]
    ax.scatter(
        visible_background_xy[:, 0],
        visible_background_xy[:, 1],
        s=raster_area(BACKGROUND_ZOOM_POINT_SIZE),
        c=theme["background_point"],
        alpha=1.0,
        linewidths=0,
    )
    ax.scatter(
        diary_xy[:, 0],
        diary_xy[:, 1],
        s=raster_area(DIARY_ZOOM_POINT_SIZE),
        c=theme["diary_point"],
        alpha=DIARY_ALPHA,
        linewidths=0,
    )
    hull = hull_boundary(diary_xy)
    if len(hull) >= 3:
        ax.add_patch(
            Polygon(
                hull,
                closed=True,
                fill=False,
                edgecolor=theme["hull"],
                linewidth=raster_s(DIARY_HULL_LINEWIDTH),
                alpha=1.0,
            )
        )
    ax.set_xlim(*xlim)
    ax.set_ylim(*ylim)
    set_plot_title(ax, title)
    ax.set_xticks([])
    ax.set_yticks([])
    for spine in ax.spines.values():
        spine.set_visible(False)
    apply_figure_margins(fig)
    save_figure_all_formats(fig, out_stem)
    plt.close(fig)


def save_points_csv(
    out: Path,
    source_name: str,
    background: Any,
    diary: Any,
    background_xy: np.ndarray,
    diary_xy: np.ndarray,
):
    with out.open("w", encoding="utf-8", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(
            [
                "source",
                "label",
                "x",
                "y",
                "stratum",
                "length_bucket",
                "chunk_length",
                "raw_length",
                "chunk_index",
                "chunk_count",
                "source_name",
                "source_kind",
                "file",
                "url",
            ]
        )
        metas = background.metas or [{} for _ in background.labels]
        for label, xy, meta in zip(background.labels, background_xy, metas):
            source_meta = meta.get("source_meta") if isinstance(meta.get("source_meta"), dict) else {}
            writer.writerow(
                [
                    source_name,
                    label,
                    f"{xy[0]:.8f}",
                    f"{xy[1]:.8f}",
                    meta.get("stratum", ""),
                    meta.get("length_bucket", ""),
                    meta.get("chunk_length", ""),
                    meta.get("raw_length", ""),
                    meta.get("chunk_index", ""),
                    meta.get("chunk_count", ""),
                    source_meta.get("source_name", ""),
                    source_meta.get("source_kind", ""),
                    source_meta.get("file", ""),
                    source_meta.get("url", ""),
                ]
            )
        for label, xy in zip(diary.labels, diary_xy):
            writer.writerow(
                [
                    "diary",
                    label,
                    f"{xy[0]:.8f}",
                    f"{xy[1]:.8f}",
                    "",
                    "",
                    "",
                    "",
                    "",
                    "",
                    "",
                    "",
                    "",
                    "",
                ]
            )


def summarize_metrics(
    title: str,
    bg_key: str,
    text_cache: Any,
    background_xy: np.ndarray,
    diary_xy: np.ndarray,
    combined_xy: np.ndarray,
    stratum_item_cap: int,
) -> str:
    diary_hull_area = polygon_area(hull_boundary(diary_xy))
    background_hull_area = polygon_area(hull_boundary(background_xy))
    combined_hull_area = polygon_area(hull_boundary(combined_xy))
    scaled_diary_xy = scaled_area_points(diary_xy, DIARY_AREA_SCALE_RATIO)
    scaled_combined_xy = np.vstack([background_xy, scaled_diary_xy]).astype(np.float32)
    scaled_diary_hull_area = polygon_area(hull_boundary(scaled_diary_xy))
    scaled_combined_hull_area = polygon_area(hull_boundary(scaled_combined_xy))
    diary_bbox_area = bbox_area(diary_xy)
    combined_bbox_area = bbox_area(combined_xy)
    background_to_diary_count = len(background_xy) / max(len(diary_xy), 1)

    nearest_center_dist = np.linalg.norm(
        background_xy - diary_xy.mean(axis=0, keepdims=True),
        axis=1,
    )
    p = np.percentile(nearest_center_dist, [1, 5, 10, 50])
    strata = Counter(str(meta.get("stratum", "unknown")) for meta in text_cache.metas)
    length_buckets = Counter(
        str(meta.get("length_bucket", "unknown")) for meta in text_cache.metas
    )
    chunk_lengths = [
        int(meta.get("chunk_length") or len(text))
        for meta, text in zip(text_cache.metas, text_cache.texts)
    ]
    chunk_p = np.percentile(chunk_lengths, [10, 50, 90]) if chunk_lengths else [0, 0, 0]

    def top_counts(counter: Counter[str], limit: int = 12) -> str:
        return ", ".join(f"{key}:{value}" for key, value in counter.most_common(limit))

    return "\n".join(
        [
            f"background_key: {bg_key}",
            f"background_title: {title}",
            f"text_cache: {text_cache.path}",
            f"text_cache_bytes: {text_cache.byte_size}",
            f"text_cache_items_available: {len(text_cache.texts)}",
            f"text_cache_schema: {TEXT_CACHE_SCHEMA_VERSION}",
            f"background_chunk_chars_min_target_max: {BACKGROUND_CHUNK_MIN_CHARS}/{BACKGROUND_CHUNK_TARGET_CHARS}/{BACKGROUND_CHUNK_MAX_CHARS}",
            f"chunk_length_p10_p50_p90: {chunk_p[0]:.0f} / {chunk_p[1]:.0f} / {chunk_p[2]:.0f}",
            f"stratified_sampling: {int(STRATIFIED_SAMPLING)}",
            f"stratum_max_share: {STRATUM_MAX_SHARE:.4f}",
            f"stratum_item_cap: {stratum_item_cap}",
            f"stratum_counts_top: {top_counts(strata)}",
            f"length_bucket_counts: {top_counts(length_buckets)}",
            f"vectorized_background_items: {len(background_xy)}",
            f"diary_items: {len(diary_xy)}",
            "diary_grain: one diary day per point; diary vectors are not chunked in this workflow.",
            f"background_to_diary_count: {background_to_diary_count:.4f}",
            f"reducer: {REDUCER}",
            f"fit_reducer_on: {FIT_REDUCER_ON}",
            f"hull_keep_ratio: {HULL_KEEP_RATIO:.4f}",
            f"hull_keep_counts diary/background/combined: {hull_point_count(diary_xy)} / {hull_point_count(background_xy)} / {hull_point_count(combined_xy)}",
            f"diary_area_scale_ratio: {DIARY_AREA_SCALE_RATIO:.4f}",
            f"diary_hull_area: {diary_hull_area:.6f}",
            f"background_hull_area: {background_hull_area:.6f}",
            f"combined_hull_area: {combined_hull_area:.6f}",
            f"diary_hull / combined_hull: {diary_hull_area / max(combined_hull_area, 1e-12):.8f}",
            f"diary_hull / background_hull: {diary_hull_area / max(background_hull_area, 1e-12):.8f}",
            f"scaled_diary_hull_area: {scaled_diary_hull_area:.6f}",
            f"scaled_combined_hull_area: {scaled_combined_hull_area:.6f}",
            f"scaled_diary_hull / scaled_combined_hull: {scaled_diary_hull_area / max(scaled_combined_hull_area, 1e-12):.8f}",
            f"scaled_diary_hull / background_hull: {scaled_diary_hull_area / max(background_hull_area, 1e-12):.8f}",
            f"diary_bbox / combined_bbox: {diary_bbox_area / max(combined_bbox_area, 1e-12):.8f}",
            "Distances from diary center to background points:",
            f"p01/p05/p10/p50: {p[0]:.6f} / {p[1]:.6f} / {p[2]:.6f} / {p[3]:.6f}",
            "Notes:",
            "- 02_diary_region_zoom intentionally magnifies the diary region; use 01_world_umap_or_pca for global scale.",
            "- Hull ratios use the central point share configured by hull_keep_ratio, so far outliers do not define the boundary.",
            "- If background_to_diary_count is near or below 1, the background is too sparse for area judgment.",
            "- Background points are chunked or packed toward diary-day scale, then sampled with per-stratum caps.",
        ]
    )
