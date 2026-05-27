"""Build a map of relationships among background text contexts."""

from __future__ import annotations

import csv
import argparse
import json
import math
import re
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Any

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.patches import Ellipse
from sklearn.decomposition import PCA
from sklearn.neighbors import NearestNeighbors

from diary_BackgroundRelations.config import *  # noqa: F403


SCRIPT_DIR = Path(__file__).resolve().parent


@dataclass
class BackgroundVectors:
    key: str
    title: str
    path: Path
    vectors: np.ndarray
    metric_index: np.ndarray
    plot_index: np.ndarray

    @property
    def metric_vectors(self) -> np.ndarray:
        return self.vectors[self.metric_index]

    @property
    def plot_vectors(self) -> np.ndarray:
        return self.vectors[self.plot_index]


def resolve_path(path: str) -> Path:
    p = Path(path)
    if p.is_absolute():
        return p
    return (SCRIPT_DIR / p).resolve()


def slugify(value: str) -> str:
    value = re.sub(r"[^a-zA-Z0-9_.-]+", "_", value.strip())
    return value.strip("_") or "background_relations"


def next_try_dir(output_root: Path) -> Path:
    output_root.mkdir(parents=True, exist_ok=True)
    used = []
    for child in output_root.iterdir():
        m = re.fullmatch(r"try_(\d+)", child.name)
        if child.is_dir() and m:
            used.append(int(m.group(1)))
    out = output_root / f"try_{max(used, default=0) + 1}"
    out.mkdir(parents=True, exist_ok=False)
    return out


def l2_normalize(x: np.ndarray) -> np.ndarray:
    return x / (np.linalg.norm(x, axis=1, keepdims=True) + 1e-12)


def enabled_background_keys() -> list[str]:
    if RELATION_BACKGROUNDS == ["all"]:
        return list(BACKGROUND_SPECS.keys())
    unknown = [key for key in RELATION_BACKGROUNDS if key not in BACKGROUND_SPECS]
    if unknown:
        raise ValueError(f"Unknown backgrounds in BGREL_BACKGROUNDS: {unknown}")
    return list(RELATION_BACKGROUNDS)


def vector_cache_candidates(cache_dir: Path, bg_key: str) -> list[tuple[int, Path]]:
    bg_dir = cache_dir / bg_key
    candidates: list[tuple[int, Path]] = []
    pattern = re.compile(r"vectors_.*_n(\d+)_tok\d+_seed\d+\.npy")
    for path in bg_dir.glob("vectors_*.npy"):
        m = pattern.fullmatch(path.name)
        if m:
            candidates.append((int(m.group(1)), path))
    return sorted(candidates)


def load_background(bg_key: str, cache_dir: Path, rng: np.random.Generator) -> BackgroundVectors:
    candidates = vector_cache_candidates(cache_dir, bg_key)
    if not candidates:
        raise FileNotFoundError(f"No vector cache found for {bg_key} in {cache_dir / bg_key}")
    _, path = candidates[-1]
    vectors = np.load(path).astype(np.float32)
    if vectors.ndim != 2 or len(vectors) < MIN_BACKGROUND_ITEMS:
        raise ValueError(f"{bg_key}: unusable vector shape {vectors.shape}")
    vectors = l2_normalize(vectors)

    all_index = np.arange(len(vectors))
    metric_n = min(len(vectors), METRIC_SAMPLE_PER_BACKGROUND)
    plot_n = min(len(vectors), PLOT_SAMPLE_PER_BACKGROUND, metric_n)
    metric_index = np.sort(rng.choice(all_index, size=metric_n, replace=False))
    plot_index = np.sort(rng.choice(metric_index, size=plot_n, replace=False))
    return BackgroundVectors(
        key=bg_key,
        title=str(BACKGROUND_SPECS[bg_key]["title"]),
        path=path,
        vectors=vectors,
        metric_index=metric_index,
        plot_index=plot_index,
    )


def short_label(key: str) -> str:
    aliases = {
        "zh_wikipedia": "wiki",
        "thucnews": "news",
        "zhihu_kol": "zhihu",
        "douban_reviews": "douban",
        "weibo_senti": "weibo",
        "classical_poetry": "poetry",
        "historical_diary": "hist diary",
        "modern_essay": "essay",
        "psych_discourse": "psych",
        "legal_text": "legal",
        "religious_text": "religion",
        "interview_oral": "interview",
        "forum_emotional": "forum",
        "self_help": "self help",
        "travel_writing": "travel",
        "food_writing": "food",
        "academic_humanities": "humanities",
        "children_writing": "children",
        "ad_copy": "ad copy",
        "dream_record": "dream",
    }
    return aliases.get(key, key.replace("_", " "))


def colors_for(n: int) -> list[Any]:
    cmaps = [plt.get_cmap("tab20"), plt.get_cmap("tab20b"), plt.get_cmap("tab20c")]
    colors = []
    for i in range(n):
        colors.append(cmaps[(i // 20) % len(cmaps)]((i % 20) / 20))
    return colors


def confidence_ellipse(points: np.ndarray, ax, color: Any, scale: float = 1.0):
    if len(points) < 3:
        return
    cov = np.cov(points.T)
    if not np.all(np.isfinite(cov)):
        return
    vals, vecs = np.linalg.eigh(cov)
    vals = np.maximum(vals, 1e-12)
    order = vals.argsort()[::-1]
    vals = vals[order]
    vecs = vecs[:, order]
    angle = math.degrees(math.atan2(vecs[1, 0], vecs[0, 0]))
    scale = 2.15
    width, height = 2 * scale * np.sqrt(vals)
    center = points.mean(axis=0)
    ax.add_patch(
        Ellipse(
            xy=center,
            width=width,
            height=height,
            angle=angle,
            facecolor=color,
            edgecolor=color,
            linewidth=ELLIPSE_LINEWIDTH * scale,
            alpha=ELLIPSE_ALPHA,
            zorder=2,
        )
    )


def build_metric_arrays(backgrounds: list[BackgroundVectors]):
    metric_vectors = []
    metric_labels = []
    metric_keys = []
    for i, bg in enumerate(backgrounds):
        vectors = bg.metric_vectors
        metric_vectors.append(vectors)
        metric_labels.extend([i] * len(vectors))
        metric_keys.extend([bg.key] * len(vectors))
    return np.vstack(metric_vectors), np.asarray(metric_labels, dtype=np.int32), metric_keys


def reduce_backgrounds(backgrounds: list[BackgroundVectors]):
    metric_vectors, metric_labels, metric_keys = build_metric_arrays(backgrounds)
    reducer = PCA(n_components=2, random_state=RANDOM_SEED)
    metric_xy = reducer.fit_transform(metric_vectors).astype(np.float32)
    centroids = np.vstack([bg.metric_vectors.mean(axis=0) for bg in backgrounds])
    centroid_xy = reducer.transform(centroids).astype(np.float32)

    plot_xy_by_key: dict[str, np.ndarray] = {}
    for bg in backgrounds:
        plot_xy_by_key[bg.key] = reducer.transform(bg.plot_vectors).astype(np.float32)

    return metric_vectors, metric_labels, metric_keys, metric_xy, centroid_xy, plot_xy_by_key


def centroid_similarity(backgrounds: list[BackgroundVectors]) -> np.ndarray:
    centroids = np.vstack([bg.metric_vectors.mean(axis=0) for bg in backgrounds])
    centroids = l2_normalize(centroids.astype(np.float32))
    return np.clip(centroids @ centroids.T, -1.0, 1.0)


def neighbor_overlap(metric_vectors: np.ndarray, metric_labels: np.ndarray, n_bg: int) -> np.ndarray:
    k = max(1, min(NEIGHBOR_K + 1, len(metric_vectors)))
    nn = NearestNeighbors(n_neighbors=k, metric="cosine")
    nn.fit(metric_vectors)
    _, indices = nn.kneighbors(metric_vectors, return_distance=True)
    overlap = np.zeros((n_bg, n_bg), dtype=np.float64)
    totals = np.zeros(n_bg, dtype=np.float64)

    for row_idx, neighbors in enumerate(indices):
        src = int(metric_labels[row_idx])
        used = 0
        for neighbor_idx in neighbors:
            if neighbor_idx == row_idx:
                continue
            dst = int(metric_labels[neighbor_idx])
            overlap[src, dst] += 1.0
            used += 1
        totals[src] += used

    return overlap / np.maximum(totals[:, None], 1.0)


def save_matrix_csv(path: Path, keys: list[str], matrix: np.ndarray):
    with path.open("w", encoding="utf-8", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(["background"] + keys)
        for key, row in zip(keys, matrix):
            writer.writerow([key] + [f"{value:.8f}" for value in row])


def save_pair_csv(
    path: Path,
    backgrounds: list[BackgroundVectors],
    overlap: np.ndarray,
    similarity: np.ndarray,
):
    rows = []
    for i, a in enumerate(backgrounds):
        for j, b in enumerate(backgrounds):
            if i >= j:
                continue
            rows.append(
                {
                    "background_a": a.key,
                    "background_b": b.key,
                    "overlap_symmetric": (overlap[i, j] + overlap[j, i]) / 2,
                    "a_to_b_overlap": overlap[i, j],
                    "b_to_a_overlap": overlap[j, i],
                    "centroid_cosine_similarity": similarity[i, j],
                }
            )
    rows.sort(
        key=lambda row: (row["overlap_symmetric"], row["centroid_cosine_similarity"]),
        reverse=True,
    )
    with path.open("w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        for row in rows:
            writer.writerow(
                {
                    key: f"{value:.8f}" if isinstance(value, float) else value
                    for key, value in row.items()
                }
            )


def save_summary_csv(
    path: Path,
    backgrounds: list[BackgroundVectors],
    centroid_xy: np.ndarray,
    overlap: np.ndarray,
):
    with path.open("w", encoding="utf-8", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(
            [
                "background",
                "title",
                "vectors_loaded",
                "metric_sample",
                "plot_sample",
                "self_neighbor_share",
                "pca_centroid_x",
                "pca_centroid_y",
                "vector_cache",
            ]
        )
        for i, bg in enumerate(backgrounds):
            writer.writerow(
                [
                    bg.key,
                    bg.title,
                    len(bg.vectors),
                    len(bg.metric_index),
                    len(bg.plot_index),
                    f"{overlap[i, i]:.8f}",
                    f"{centroid_xy[i, 0]:.8f}",
                    f"{centroid_xy[i, 1]:.8f}",
                    str(bg.path),
                ]
            )


def _set_svg_display_size(svg_path: Path, width: int, height: int) -> None:
    text = svg_path.read_text(encoding="utf-8")
    text = re.sub(r'width="[^"]+"', f'width="{width}"', text, count=1)
    text = re.sub(r'height="[^"]+"', f'height="{height}"', text, count=1)
    svg_path.write_text(text, encoding="utf-8")


def save_figure_all_formats(fig, stem: Path, output_size: tuple[int, int]):
    formats = []
    for fmt in OUTPUT_IMAGE_FORMATS:
        fmt = fmt.lower().lstrip(".")
        if fmt not in {"svg", "png"}:
            raise ValueError(f"Unsupported output format: {fmt}")
        if fmt not in formats:
            formats.append(fmt)
    for fmt in formats:
        out_path = stem.with_suffix(f".{fmt}")
        fig.savefig(
            out_path,
            dpi=DPI,
            format=fmt,
            facecolor="none" if fmt == "svg" else "white",
            edgecolor="none",
            transparent=(fmt == "svg"),
        )
        if fmt == "svg":
            _set_svg_display_size(out_path, *output_size)


def set_render_sizes(content_size: int, resolution: int) -> None:
    global SCATTER_CANVAS_WIDTH, SCATTER_CANVAS_HEIGHT, SCATTER_OUTPUT_WIDTH, SCATTER_OUTPUT_HEIGHT
    global REPORT_CANVAS_WIDTH, REPORT_CANVAS_HEIGHT, REPORT_OUTPUT_WIDTH, REPORT_OUTPUT_HEIGHT
    content_size = int(content_size)
    resolution = int(resolution)
    report_ratio = REPORT_CANVAS_HEIGHT / REPORT_CANVAS_WIDTH
    SCATTER_CANVAS_WIDTH = content_size
    SCATTER_CANVAS_HEIGHT = content_size
    SCATTER_OUTPUT_WIDTH = resolution
    SCATTER_OUTPUT_HEIGHT = resolution
    REPORT_CANVAS_WIDTH = content_size
    REPORT_CANVAS_HEIGHT = int(round(content_size * report_ratio))
    REPORT_OUTPUT_WIDTH = resolution
    REPORT_OUTPUT_HEIGHT = int(round(resolution * report_ratio))


def raster_s(value: float, output_width: int, content_width: int) -> float:
    return value * (output_width / content_width)


def raster_area(value: float, output_width: int, content_width: int) -> float:
    scale = output_width / content_width
    return value * scale * scale


def collect_pair_rows(
    backgrounds: list[BackgroundVectors],
    overlap: np.ndarray,
    similarity: np.ndarray,
    labels: list[str],
) -> list[tuple[float, float, str, str]]:
    rows = []
    for i, a in enumerate(backgrounds):
        for j, b in enumerate(backgrounds):
            if i >= j:
                continue
            rows.append(
                (
                    (overlap[i, j] + overlap[j, i]) / 2,
                    similarity[i, j],
                    labels[i],
                    labels[j],
                )
            )
    rows.sort(reverse=True)
    return rows


def plot_scatter_map(
    out_stem: Path,
    backgrounds: list[BackgroundVectors],
    plot_xy_by_key: dict[str, np.ndarray],
    centroid_xy: np.ndarray,
):
    colors = colors_for(len(backgrounds))
    labels = [short_label(bg.key) for bg in backgrounds]

    fig = plt.figure(figsize=(SCATTER_OUTPUT_WIDTH / DPI, SCATTER_OUTPUT_HEIGHT / DPI), dpi=DPI)
    fig.patch.set_facecolor("none")
    ax_map = fig.add_subplot(111)
    scale = SCATTER_OUTPUT_WIDTH / SCATTER_CANVAS_WIDTH

    ax_map.set_facecolor("none")
    for i, bg in enumerate(backgrounds):
        xy = plot_xy_by_key[bg.key]
        ax_map.scatter(
            xy[:, 0],
            xy[:, 1],
            s=raster_area(POINT_SIZE, SCATTER_OUTPUT_WIDTH, SCATTER_CANVAS_WIDTH),
            c=[colors[i]],
            alpha=POINT_ALPHA,
            linewidths=0,
            rasterized=False,
            zorder=1,
        )
        confidence_ellipse(xy, ax_map, colors[i], scale=scale)
        ax_map.scatter(
            centroid_xy[i, 0],
            centroid_xy[i, 1],
            s=raster_area(CENTROID_SIZE, SCATTER_OUTPUT_WIDTH, SCATTER_CANVAS_WIDTH),
            c=[colors[i]],
            edgecolors="white",
            linewidths=raster_s(0.65, SCATTER_OUTPUT_WIDTH, SCATTER_CANVAS_WIDTH),
            zorder=4,
        )
        ax_map.text(
            centroid_xy[i, 0],
            centroid_xy[i, 1],
            f" {labels[i]}",
            fontsize=raster_s(6.2, SCATTER_OUTPUT_WIDTH, SCATTER_CANVAS_WIDTH),
            color=TEXT_COLOR,
            va="center",
            zorder=5,
        )

    all_xy = np.vstack(list(plot_xy_by_key.values()) + [centroid_xy])
    mn = all_xy.min(axis=0)
    mx = all_xy.max(axis=0)
    span = np.maximum(mx - mn, 1e-6)
    pad = span * 0.08
    ax_map.set_xlim(mn[0] - pad[0], mx[0] + pad[0])
    ax_map.set_ylim(mn[1] - pad[1], mx[1] + pad[1])
    ax_map.set_title("Background context relationship map", loc="left", fontsize=raster_s(12, SCATTER_OUTPUT_WIDTH, SCATTER_CANVAS_WIDTH), pad=raster_s(8, SCATTER_OUTPUT_WIDTH, SCATTER_CANVAS_WIDTH))
    ax_map.set_xlabel("PCA 1 from sampled background vectors", fontsize=raster_s(7, SCATTER_OUTPUT_WIDTH, SCATTER_CANVAS_WIDTH))
    ax_map.set_ylabel("PCA 2", fontsize=raster_s(7, SCATTER_OUTPUT_WIDTH, SCATTER_CANVAS_WIDTH))
    ax_map.tick_params(labelsize=raster_s(6, SCATTER_OUTPUT_WIDTH, SCATTER_CANVAS_WIDTH), colors="#6b6f75", length=raster_s(2, SCATTER_OUTPUT_WIDTH, SCATTER_CANVAS_WIDTH))
    for spine in ax_map.spines.values():
        spine.set_color(GRID_COLOR)
        spine.set_linewidth(raster_s(0.6, SCATTER_OUTPUT_WIDTH, SCATTER_CANVAS_WIDTH))

    fig.tight_layout(pad=raster_s(1.1, SCATTER_OUTPUT_WIDTH, SCATTER_CANVAS_WIDTH))
    save_figure_all_formats(fig, out_stem, (SCATTER_OUTPUT_WIDTH, SCATTER_OUTPUT_HEIGHT))
    plt.close(fig)


def plot_overlap_report(
    out_stem: Path,
    backgrounds: list[BackgroundVectors],
    overlap: np.ndarray,
    similarity: np.ndarray,
):
    labels = [short_label(bg.key) for bg in backgrounds]

    fig = plt.figure(figsize=(REPORT_OUTPUT_WIDTH / DPI, REPORT_OUTPUT_HEIGHT / DPI), dpi=DPI)
    fig.patch.set_facecolor("none")
    scale = REPORT_OUTPUT_WIDTH / REPORT_CANVAS_WIDTH
    gs = fig.add_gridspec(2, 1, height_ratios=[1.0, 0.48], hspace=0.52)
    ax_heat = fig.add_subplot(gs[0, 0])
    ax_pairs = fig.add_subplot(gs[1, 0])

    matrix = overlap.copy()
    np.fill_diagonal(matrix, np.nan)
    cmap = plt.get_cmap(MATRIX_CMAP).copy()
    cmap.set_bad("#f1f3f4")
    off_diag_max = float(np.nanmax(matrix)) if np.isfinite(matrix).any() else 0.0
    im = ax_heat.imshow(matrix, cmap=cmap, vmin=0, vmax=max(0.16, off_diag_max))
    ax_heat.set_title("Cross-background overlap", loc="left", fontsize=9 * scale, pad=7 * scale)
    ax_heat.set_xticks(np.arange(len(labels)))
    ax_heat.set_yticks(np.arange(len(labels)))
    ax_heat.set_xticklabels(labels, rotation=90, fontsize=5.6 * scale)
    ax_heat.set_yticklabels(labels, fontsize=5.6 * scale)
    ax_heat.tick_params(length=0)
    for spine in ax_heat.spines.values():
        spine.set_visible(False)
    cbar = fig.colorbar(im, ax=ax_heat, fraction=0.046, pad=0.02)
    cbar.ax.tick_params(labelsize=5.4 * scale, length=2 * scale)

    top_rows = collect_pair_rows(backgrounds, overlap, similarity, labels)[:9]
    ax_pairs.set_axis_off()
    ax_pairs.set_title("Most overlapping pairs", loc="left", fontsize=9 * scale, pad=6 * scale)
    for row, (score, sim, a, b) in enumerate(top_rows):
        y = 0.82 - row * 0.085
        ax_pairs.text(0.00, y, f"{a}  -  {b}", fontsize=6.4 * scale, color=TEXT_COLOR, va="center")
        ax_pairs.text(0.72, y, f"{score:.3f}", fontsize=6.4 * scale, color=TEXT_COLOR, va="center")
        ax_pairs.text(0.88, y, f"{sim:.3f}", fontsize=6.4 * scale, color="#6b6f75", va="center")
    ax_pairs.text(0.72, 0.94, "overlap", fontsize=5.5 * scale, color="#6b6f75", va="top")
    ax_pairs.text(0.88, 0.94, "cos", fontsize=5.5 * scale, color="#6b6f75", va="top")

    fig.suptitle(
        "Cross-background overlap and pair ranking",
        x=0.08,
        y=0.985,
        ha="left",
        fontsize=12 * scale,
        color=TEXT_COLOR,
    )
    save_figure_all_formats(fig, out_stem, (REPORT_OUTPUT_WIDTH, REPORT_OUTPUT_HEIGHT))
    plt.close(fig)


def params_snapshot() -> dict[str, Any]:
    params: dict[str, Any] = {}
    for k, v in globals().items():
        if k.isupper() and isinstance(v, (str, int, float, bool, list, dict)):
            if k == "BACKGROUND_SPECS":
                params[k] = {key: value.get("title", key) for key, value in v.items()}
            else:
                params[k] = v
    return params


def write_summary(
    out: Path,
    backgrounds: list[BackgroundVectors],
    overlap: np.ndarray,
    similarity: np.ndarray,
):
    rows = []
    for i, a in enumerate(backgrounds):
        for j, b in enumerate(backgrounds):
            if i >= j:
                continue
            rows.append(((overlap[i, j] + overlap[j, i]) / 2, similarity[i, j], a.key, b.key))
    rows.sort(reverse=True)
    separated = []
    for i, bg in enumerate(backgrounds):
        nearest = sorted(
            [
                ((overlap[i, j] + overlap[j, i]) / 2, similarity[i, j], backgrounds[j].key)
                for j in range(len(backgrounds))
                if i != j
            ],
            reverse=True,
        )
        separated.append((nearest[0][0] if nearest else 0.0, overlap[i, i], bg.key))
    separated.sort()

    lines = [
        "BackgroundRelations",
        f"backgrounds: {len(backgrounds)}",
        f"metric_sample_per_background: {METRIC_SAMPLE_PER_BACKGROUND}",
        f"plot_sample_per_background: {PLOT_SAMPLE_PER_BACKGROUND}",
        f"neighbor_k: {NEIGHBOR_K}",
        "",
        "Most overlapping pairs:",
    ]
    for score, sim, a, b in rows[:12]:
        lines.append(f"- {a} <-> {b}: overlap={score:.6f}, centroid_cosine={sim:.6f}")
    lines.extend(["", "Most separated backgrounds by nearest-pair overlap:"])
    for nearest_score, self_share, key in separated[:8]:
        lines.append(f"- {key}: nearest_overlap={nearest_score:.6f}, self_neighbor_share={self_share:.6f}")
    lines.extend(
        [
            "",
            "Interpretation:",
            "- overlap is the share of local nearest-neighbor slots crossing from one background into another.",
            "- high off-diagonal overlap means the two backgrounds occupy nearby or mixed regions.",
            "- high self_neighbor_share means a background stays locally clustered apart from other backgrounds.",
            "- centroid_cosine_similarity compares only the average vector direction, so it is less local than overlap.",
        ]
    )
    (out / "summary.txt").write_text("\n".join(lines) + "\n", encoding="utf-8")


def build_background_relation_outputs(
    out: Path,
    cache_dir: Path,
    bg_keys: list[str],
    *,
    verbose: bool = True,
) -> dict[str, Any]:
    out.mkdir(parents=True, exist_ok=True)
    rng = np.random.default_rng(RANDOM_SEED)
    if verbose:
        print("\n" + "═" * 68)
        print("  BackgroundRelations")
        print(f"  Source cache : {cache_dir}")
        print(f"  Backgrounds  : {len(bg_keys)}")
        print(f"  Output       : {out}")
        print("═" * 68 + "\n")

    backgrounds: list[BackgroundVectors] = []
    skipped: list[str] = []
    for bg_key in bg_keys:
        try:
            bg = load_background(bg_key, cache_dir, rng)
            backgrounds.append(bg)
            if verbose:
                print(
                    f"  loaded {bg.key:<20} "
                    f"vectors={len(bg.vectors):>5} metric={len(bg.metric_index):>4} "
                    f"plot={len(bg.plot_index):>4}"
                )
        except Exception as exc:
            skipped.append(f"{bg_key}: {type(exc).__name__}: {exc}")
            if verbose:
                print(f"  skipped {bg_key}: {type(exc).__name__}: {exc}")

    if len(backgrounds) < 2:
        raise ValueError("Need at least two backgrounds with vector caches to compare.")

    if verbose:
        print("\n  reducing and measuring relationships ...")
    metric_vectors, metric_labels, _, _, centroid_xy, plot_xy_by_key = reduce_backgrounds(backgrounds)
    similarity = centroid_similarity(backgrounds)
    overlap = neighbor_overlap(metric_vectors, metric_labels, len(backgrounds))

    keys = [bg.key for bg in backgrounds]
    save_matrix_csv(out / "nearest_neighbor_overlap_matrix.csv", keys, overlap)
    save_matrix_csv(out / "centroid_cosine_similarity_matrix.csv", keys, similarity)
    save_pair_csv(out / "relation_pairs.csv", backgrounds, overlap, similarity)
    save_summary_csv(out / "background_summary.csv", backgrounds, centroid_xy, overlap)
    write_summary(out, backgrounds, overlap, similarity)
    (out / "params.json").write_text(
        json.dumps(params_snapshot(), ensure_ascii=False, indent=2), encoding="utf-8"
    )
    if skipped:
        (out / "skipped.txt").write_text("\n".join(skipped) + "\n", encoding="utf-8")

    if verbose:
        print("  plotting ...")
    plot_scatter_map(out / "01_background_scatter_map", backgrounds, plot_xy_by_key, centroid_xy)
    plot_overlap_report(out / "02_background_overlap_report", backgrounds, overlap, similarity)

    if verbose:
        print("\n" + "═" * 68)
        print(f"  Done. Output: {out}")
        print("  Generated files:")
        for fp in sorted(out.iterdir()):
            if fp.is_file():
                print(f"    {fp}")
        print("═" * 68)

    return {
        "out": str(out),
        "backgrounds": [bg.key for bg in backgrounds],
        "skipped": skipped,
    }


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Build a map of relationships among background text contexts."
    )
    parser.add_argument(
        "--png-only",
        action="store_true",
        help="Only write PNG charts; skip SVG output.",
    )
    parser.add_argument(
        "--content-size",
        type=int,
        default=SCATTER_CANVAS_WIDTH,
        help=f"Logical square content size. Report keeps its current aspect ratio. Default: {SCATTER_CANVAS_WIDTH}.",
    )
    parser.add_argument(
        "--resolution",
        "--size",
        dest="resolution",
        type=int,
        default=SCATTER_OUTPUT_WIDTH,
        help=f"Final scatter PNG pixel size. Report height keeps its aspect ratio. Default: {SCATTER_OUTPUT_WIDTH}.",
    )
    args = parser.parse_args()
    if args.content_size < 1:
        parser.error("--content-size must be a positive integer")
    if args.resolution < 1:
        parser.error("--resolution must be a positive integer")
    return args


def main():
    args = parse_args()
    set_render_sizes(args.content_size, args.resolution)
    global OUTPUT_IMAGE_FORMATS
    if args.png_only:
        OUTPUT_IMAGE_FORMATS = ["png"]

    cache_dir = resolve_path(SOURCE_CACHE_DIR)
    out = next_try_dir(resolve_path(OUTPUT_ROOT))
    build_background_relation_outputs(out, cache_dir, enabled_background_keys(), verbose=True)


if __name__ == "__main__":
    try:
        main()
    except KeyboardInterrupt:
        sys.exit("\nInterrupted.")
