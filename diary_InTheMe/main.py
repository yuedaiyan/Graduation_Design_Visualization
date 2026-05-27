#!/usr/bin/env python3
"""Render a whole-diary semantic map from diary-level vectors."""

from __future__ import annotations

import argparse
import json
import math
import re
from collections import Counter, defaultdict
from datetime import datetime
from pathlib import Path

import numpy as np

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.colors import ListedColormap
from sklearn.cluster import DBSCAN, KMeans
from sklearn.metrics import pairwise_distances
from sklearn.neighbors import NearestNeighbors
from sklearn.preprocessing import normalize


SEED = 23
CANVAS_W = 16
CANVAS_H = 12
DEFAULT_RESOLUTION = 2160
OUTPUT_W = DEFAULT_RESOLUTION
OUTPUT_H = int(DEFAULT_RESOLUTION * CANVAS_H / CANVAS_W)
RASTER_SCALE = OUTPUT_W / (CANVAS_W * 220)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Generate InTheMe diary semantic map")
    parser.add_argument("--root", default=None, help="Project root. Defaults to parent of this script.")
    parser.add_argument(
        "--out-dir",
        default=None,
        help="Output base dir. Defaults to <root>/output_All/<this program folder>.",
    )
    parser.add_argument("--dpi", type=int, default=220)
    parser.add_argument("--seed", type=int, default=SEED)
    parser.add_argument("--clusters", type=int, default=14)
    parser.add_argument("--neighbors", type=int, default=18)
    parser.add_argument("--png-only", action="store_true", help="Only write the PNG map; skip SVG output.")
    parser.add_argument(
        "--content-size",
        type=float,
        default=CANVAS_W,
        help=f"Logical content width in inches. Height keeps the current {CANVAS_W:g}:{CANVAS_H:g} ratio.",
    )
    parser.add_argument(
        "--resolution",
        "--size",
        dest="resolution",
        type=int,
        default=DEFAULT_RESOLUTION,
        help=f"Final PNG pixel width. Height keeps the content ratio. Default: {DEFAULT_RESOLUTION}.",
    )
    args = parser.parse_args()
    if args.content_size <= 0:
        parser.error("--content-size must be positive")
    if args.resolution < 1:
        parser.error("--resolution must be a positive integer")
    return args


def set_render_sizes(content_width: float, output_width: int, dpi: int) -> None:
    global CANVAS_W, CANVAS_H, OUTPUT_W, OUTPUT_H, RASTER_SCALE
    ratio = CANVAS_H / CANVAS_W
    CANVAS_W = float(content_width)
    CANVAS_H = float(content_width * ratio)
    OUTPUT_W = int(output_width)
    OUTPUT_H = int(round(output_width * ratio))
    RASTER_SCALE = OUTPUT_W / (CANVAS_W * dpi)


def raster_s(value: float) -> float:
    return value * RASTER_SCALE


def raster_area(value: float) -> float:
    return value * RASTER_SCALE * RASTER_SCALE


def next_try_dir(base_dir: Path) -> Path:
    base_dir.mkdir(parents=True, exist_ok=True)
    max_idx = 0
    for item in base_dir.iterdir():
        if not item.is_dir():
            continue
        m = re.fullmatch(r"try_(\d+)", item.name)
        if m:
            max_idx = max(max_idx, int(m.group(1)))
    target = base_dir / f"try_{max_idx + 1}"
    target.mkdir(parents=True, exist_ok=False)
    return target


def load_diary_entries(root: Path) -> dict[str, dict[str, object]]:
    path = root / "diary_entries.json"
    if not path.exists():
        return {}
    with path.open("r", encoding="utf-8") as f:
        rows = json.load(f)
    entries: dict[str, dict[str, object]] = {}
    duplicate_counts: defaultdict[str, int] = defaultdict(int)
    for row in rows:
        if not isinstance(row, dict) or not row.get("date"):
            continue
        date = str(row["date"])
        duplicate_counts[date] += 1
        key = date if duplicate_counts[date] == 1 else f"{date}_{duplicate_counts[date]}"
        content = str(row.get("content") or "")
        entries[key] = {
            "date": date,
            "chars": len(content),
            "location": row.get("location"),
            "time_of_day": row.get("time_of_day"),
        }
    return entries


def load_vectors(root: Path) -> tuple[list[str], np.ndarray, dict[str, dict[str, object]]]:
    vec_dir = root / "diary_vectors"
    if not vec_dir.exists():
        raise FileNotFoundError(f"Missing vector directory: {vec_dir}")

    files = sorted(vec_dir.glob("*.npy"))
    if not files:
        raise RuntimeError(f"No .npy vectors found under {vec_dir}")

    names: list[str] = []
    vectors: list[np.ndarray] = []
    for path in files:
        arr = np.load(path).reshape(-1).astype(np.float64)
        if not np.all(np.isfinite(arr)):
            continue
        names.append(path.stem)
        vectors.append(arr)

    if len(vectors) < 3:
        raise RuntimeError("Need at least three diary vectors to render a semantic map")

    entries = load_diary_entries(root)
    return names, np.vstack(vectors), entries


def semantic_layout(x: np.ndarray, seed: int, neighbors: int) -> np.ndarray:
    try:
        import umap

        reducer = umap.UMAP(
            n_components=2,
            n_neighbors=max(5, min(neighbors, len(x) - 1)),
            min_dist=0.08,
            metric="cosine",
            random_state=seed,
            spread=1.45,
        )
        xy = reducer.fit_transform(x)
    except Exception:
        # PCA fallback via SVD keeps the script useful even if umap is unavailable.
        centered = x - x.mean(axis=0, keepdims=True)
        _, _, vt = np.linalg.svd(centered, full_matrices=False)
        xy = centered @ vt[:2].T

    xy = np.asarray(xy, dtype=np.float64)
    xy -= xy.mean(axis=0, keepdims=True)
    scale = np.percentile(np.abs(xy), 98, axis=0)
    scale = np.clip(scale, 1e-9, None)
    xy /= scale
    return xy


def cluster_points(x_norm: np.ndarray, xy: np.ndarray, requested_clusters: int) -> tuple[np.ndarray, np.ndarray]:
    nn = NearestNeighbors(n_neighbors=min(10, len(x_norm)), metric="cosine").fit(x_norm)
    distances, _ = nn.kneighbors(x_norm)
    eps = float(np.percentile(distances[:, -1], 72))
    db = DBSCAN(eps=max(eps, 0.035), min_samples=5, metric="cosine")
    labels = db.fit_predict(x_norm)

    non_noise = labels[labels >= 0]
    n_clusters = len(set(int(v) for v in non_noise))
    if n_clusters < 6:
        n = max(4, min(requested_clusters, int(math.sqrt(len(xy))) + 4, len(xy)))
        labels = KMeans(n_clusters=n, random_state=17, n_init=20).fit_predict(xy)

    center_dist = pairwise_distances(x_norm, metric="cosine")
    kth = np.sort(center_dist, axis=1)[:, min(8, center_dist.shape[1] - 1)]
    isolation = (kth - kth.min()) / max(float(kth.max() - kth.min()), 1e-9)
    return labels.astype(int), isolation


def parse_year(name: str) -> int | None:
    try:
        return datetime.strptime(name[:10], "%Y-%m-%d").year
    except ValueError:
        return None


def cluster_summary(
    names: list[str],
    xy: np.ndarray,
    labels: np.ndarray,
    isolation: np.ndarray,
    entries: dict[str, dict[str, object]],
) -> dict[str, object]:
    summary: dict[str, object] = {
        "count": len(names),
        "clusters": [],
        "isolated_days": [],
    }
    for label in sorted(set(int(v) for v in labels)):
        idx = np.where(labels == label)[0]
        centroid = xy[idx].mean(axis=0)
        rep_idx = idx[np.argsort(np.linalg.norm(xy[idx] - centroid, axis=1))[:6]]
        years = Counter(parse_year(names[i]) for i in idx if parse_year(names[i]) is not None)
        chars = [int(entries.get(names[i], {}).get("chars", 0)) for i in idx]
        summary["clusters"].append(
            {
                "label": int(label),
                "size": int(len(idx)),
                "representative_days": [names[i] for i in rep_idx],
                "year_range": [min(years) if years else None, max(years) if years else None],
                "dominant_years": [year for year, _ in years.most_common(3)],
                "median_chars": int(np.median(chars)) if chars else 0,
            }
        )

    iso_idx = np.argsort(isolation)[-18:][::-1]
    summary["isolated_days"] = [
        {
            "day": names[i],
            "isolation": round(float(isolation[i]), 4),
            "chars": int(entries.get(names[i], {}).get("chars", 0)),
        }
        for i in iso_idx
    ]
    return summary


def draw_map(
    out_png: Path,
    out_svg: Path | None,
    names: list[str],
    xy: np.ndarray,
    labels: np.ndarray,
    isolation: np.ndarray,
    entries: dict[str, dict[str, object]],
    dpi: int,
) -> None:
    palette = [
        "#2B6CB0",
        "#D9480F",
        "#2F9E44",
        "#C2255C",
        "#5F3DC4",
        "#0B7285",
        "#E67700",
        "#6741D9",
        "#087F5B",
        "#A61E4D",
        "#1864AB",
        "#9C6644",
        "#364FC7",
        "#5C940D",
        "#E8590C",
        "#0C8599",
    ]
    uniq = sorted(set(int(v) for v in labels))
    color_map = {label: palette[i % len(palette)] for i, label in enumerate(uniq)}
    colors = [color_map[int(v)] for v in labels]

    years = np.array([parse_year(name) or 0 for name in names])
    chars = np.array([int(entries.get(name, {}).get("chars", 0)) for name in names], dtype=np.float64)
    sizes = 18 + 58 * np.sqrt(np.clip(chars, 0, np.percentile(chars, 95)) / max(np.percentile(chars, 95), 1))
    sizes = raster_area(sizes)

    fig = plt.figure(figsize=(OUTPUT_W / dpi, OUTPUT_H / dpi), dpi=dpi)
    ax = fig.add_axes([0.045, 0.055, 0.73, 0.81])
    side = fig.add_axes([0.805, 0.08, 0.17, 0.82])
    fig.patch.set_facecolor("#F7F4EF")
    ax.set_facecolor("#F7F4EF")
    side.set_facecolor("#F7F4EF")

    order = np.argsort(isolation)
    ax.scatter(
        xy[order, 0],
        xy[order, 1],
        s=sizes[order],
        c=[colors[i] for i in order],
        alpha=0.72,
        linewidths=raster_s(0.45),
        edgecolors="#1F2933",
    )

    by_year = defaultdict(list)
    for i, year in enumerate(years):
        if year:
            by_year[int(year)].append(i)
    for year, idxs in sorted(by_year.items()):
        idxs = sorted(idxs, key=lambda i: names[i])
        if len(idxs) > 2:
            ax.plot(xy[idxs, 0], xy[idxs, 1], color="#1F2933", alpha=0.10, lw=raster_s(0.85), zorder=0)

    iso_idx = np.argsort(isolation)[-22:]
    ax.scatter(
        xy[iso_idx, 0],
        xy[iso_idx, 1],
        s=sizes[iso_idx] + raster_area(58),
        facecolors="none",
        edgecolors="#111827",
        linewidths=raster_s(1.2),
        alpha=0.9,
    )
    for i in iso_idx[-12:]:
        ax.text(
            xy[i, 0] + 0.018,
            xy[i, 1] + 0.018,
            names[i],
            fontsize=raster_s(6.2),
            color="#111827",
            alpha=0.92,
        )

    for label in uniq:
        idx = np.where(labels == label)[0]
        if len(idx) < 8:
            continue
        centroid = xy[idx].mean(axis=0)
        nearest = idx[np.argmin(np.linalg.norm(xy[idx] - centroid, axis=1))]
        ax.text(
            centroid[0],
            centroid[1],
            names[nearest],
            ha="center",
            va="center",
            fontsize=raster_s(7.5),
            weight="bold",
            color="#F7F4EF",
            bbox=dict(boxstyle="round,pad=0.22", facecolor=color_map[label], edgecolor="none", alpha=0.86),
        )

    fig.text(
        0.045,
        0.925,
        "In The Me / Diary Semantic Space",
        fontsize=raster_s(18),
        weight="bold",
        color="#1F2933",
    )
    fig.text(
        0.045,
        0.895,
        "Each point is one diary day. Color = semantic cluster, ring = isolated day, faint lines = time order within a year.",
        fontsize=raster_s(8.5),
        color="#55616F",
    )
    ax.set_xticks([])
    ax.set_yticks([])
    for spine in ax.spines.values():
        spine.set_visible(False)
    ax.set_aspect("equal", adjustable="datalim")

    side.axis("off")
    side.text(0, 1.0, "Clusters", fontsize=raster_s(12), weight="bold", color="#1F2933", transform=side.transAxes)
    y = 0.955
    counts = Counter(int(v) for v in labels)
    for label, count in counts.most_common(12):
        idx = np.where(labels == label)[0]
        year_counts = Counter(int(years[i]) for i in idx if years[i])
        yr = ", ".join(str(v) for v, _ in year_counts.most_common(2))
        side.scatter([0.02], [y + 0.006], s=raster_area(42), color=color_map[label], transform=side.transAxes)
        side.text(0.08, y, f"{count:>3} days  {yr}", fontsize=raster_s(8.2), color="#26313D", transform=side.transAxes)
        y -= 0.045

    y -= 0.025
    side.text(0, y, "Most Isolated", fontsize=raster_s(12), weight="bold", color="#1F2933", transform=side.transAxes)
    y -= 0.042
    for i in np.argsort(isolation)[-12:][::-1]:
        side.text(
            0.0,
            y,
            f"{names[i]}  {isolation[i]:.2f}",
            fontsize=raster_s(7.6),
            color="#26313D",
            transform=side.transAxes,
        )
        y -= 0.034

    side.text(
        0,
        0.02,
        f"{len(names)} diary days | {min(n[:10] for n in names)} to {max(n[:10] for n in names)}",
        fontsize=raster_s(7.5),
        color="#667085",
        transform=side.transAxes,
    )

    fig.savefig(out_png, dpi=dpi, facecolor=fig.get_facecolor())
    if out_svg is not None:
        fig.savefig(out_svg, facecolor=fig.get_facecolor())
    plt.close(fig)


def main() -> None:
    args = parse_args()
    set_render_sizes(args.content_size, args.resolution, args.dpi)
    script_dir = Path(__file__).resolve().parent
    root = Path(args.root).resolve() if args.root else script_dir.parent.resolve()
    base = Path(args.out_dir).resolve() if args.out_dir else root / "output_All" / script_dir.name
    out_dir = next_try_dir(base)

    names, vectors, entries = load_vectors(root)
    x_norm = normalize(vectors)
    xy = semantic_layout(x_norm, seed=args.seed, neighbors=args.neighbors)
    labels, isolation = cluster_points(x_norm, xy, requested_clusters=args.clusters)

    out_png = out_dir / "diary_semantic_space.png"
    out_svg = None if args.png_only else out_dir / "diary_semantic_space.svg"
    draw_map(out_png, out_svg, names, xy, labels, isolation, entries, dpi=args.dpi)

    np.savez_compressed(
        out_dir / "semantic_layout_cache.npz",
        names=np.array(names),
        xy=xy,
        labels=labels,
        isolation=isolation,
    )
    summary = cluster_summary(names, xy, labels, isolation, entries)
    with (out_dir / "cluster_summary.json").open("w", encoding="utf-8") as f:
        json.dump(summary, f, ensure_ascii=False, indent=2)

    print(f"Wrote {out_png}")
    if out_svg is not None:
        print(f"Wrote {out_svg}")
    print(f"Wrote {out_dir / 'semantic_layout_cache.npz'}")
    print(f"Wrote {out_dir / 'cluster_summary.json'}")


if __name__ == "__main__":
    main()
