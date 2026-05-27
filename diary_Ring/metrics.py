from __future__ import annotations

import hashlib
import json
from pathlib import Path

import numpy as np

from config import DISTINCTIVENESS_CACHE_VERSION
from models import Entry, WindowMetric
from utils import clamp01, normalize_range, normalize_rows


def cache_dir(root: Path) -> Path:
    path = root / ".cache" / "derived" / "diary_Ring"
    path.mkdir(parents=True, exist_ok=True)
    return path


def vector_files_signature(root: Path, stems: list[str]) -> str:
    parts = [f"version:{DISTINCTIVENESS_CACHE_VERSION}"]
    for stem in stems:
        path = root / "diary_vectors" / f"{stem}.npy"
        if not path.exists():
            parts.append(f"{stem}:missing")
            continue
        stat = path.stat()
        parts.append(f"{stem}:{stat.st_size}:{stat.st_mtime_ns}")
    digest = hashlib.sha256("\n".join(parts).encode("utf-8")).hexdigest()
    return digest[:16]


def load_diary_distinctiveness(entries: list[Entry], root: Path) -> dict[str, float]:
    stems_for_cache = [entry.stem for entry in entries]
    cache_path = (
        cache_dir(root)
        / f"distinctiveness_{vector_files_signature(root, stems_for_cache)}.json"
    )
    if cache_path.exists():
        with cache_path.open("r", encoding="utf-8") as f:
            payload = json.load(f)
        return {str(k): float(v) for k, v in payload["distinctiveness"].items()}

    vectors: list[np.ndarray] = []
    stems: list[str] = []
    for entry in entries:
        path = root / "diary_vectors" / f"{entry.stem}.npy"
        if not path.exists():
            continue
        stems.append(entry.stem)
        vectors.append(np.load(path).astype(np.float32))

    if not vectors:
        return {}

    mat = normalize_rows(np.vstack(vectors))
    centroid = mat.mean(axis=0)
    centroid = centroid / max(np.linalg.norm(centroid), 1e-12)
    dist = 1.0 - (mat @ centroid)
    lo = float(np.percentile(dist, 5))
    hi = float(np.percentile(dist, 95))
    scaled = [normalize_range(float(x), lo, hi) for x in dist]
    distinctiveness = dict(zip(stems, scaled))
    with cache_path.open("w", encoding="utf-8") as f:
        json.dump(
            {
                "source": str(root / "diary_vectors"),
                "version": DISTINCTIVENESS_CACHE_VERSION,
                "stems": stems,
                "distinctiveness": distinctiveness,
            },
            f,
            ensure_ascii=False,
        )
    return distinctiveness


def offdiag_values(sim: np.ndarray) -> np.ndarray:
    if sim.shape[0] <= 1:
        return np.array([1.0], dtype=np.float32)
    mask = ~np.eye(sim.shape[0], dtype=bool)
    return sim[mask].astype(np.float32)


def window_pair_similarity(sim: np.ndarray, start: int, end: int) -> float:
    sub = sim[start : end + 1, start : end + 1]
    return float(offdiag_values(sub).mean())


def build_window_metrics(
    root: Path, entry: Entry
) -> tuple[list[WindowMetric], dict[str, float]]:
    base = root / "diary_sentence_vectors" / entry.stem
    meta_path = base / "meta.json"
    with meta_path.open("r", encoding="utf-8") as f:
        meta = json.load(f)

    sent_vecs = np.load(base / "sentence_vectors.npy").astype(np.float32)
    win_vecs = np.load(base / "window_vectors.npy").astype(np.float32)
    diary_vec = np.load(base / "diary_vector.npy").astype(np.float32)
    sim = np.load(base / "sentence_similarity.npy").astype(np.float32)

    global_sim = float(offdiag_values(sim).mean())
    global_focus = normalize_range(global_sim, 0.34, 0.76)
    fragmentation = normalize_range(float(meta["num_windows"]), 1.0, 7.0)
    global_looseness = float(
        clamp01(0.58 * (1.0 - global_focus) + 0.42 * fragmentation)
    )

    metrics: list[WindowMetric] = []
    previous_vec: np.ndarray | None = None
    for raw in meta["windows"]:
        start = int(raw["start_sentence_idx"])
        end = int(raw["end_sentence_idx"])
        size = int(raw["size"])
        text = str(raw["text"])
        sent_slice = sent_vecs[start : end + 1]
        win_vec = win_vecs[int(raw["idx"])]

        pair = window_pair_similarity(sim, start, end)
        pair_focus = normalize_range(pair, 0.33, 0.78)
        to_window = float(np.mean(sent_slice @ win_vec))
        window_focus = normalize_range(to_window, 0.72, 0.98)
        to_diary = float(np.mean(sent_slice @ diary_vec))
        diary_alignment = normalize_range(to_diary, 0.50, 0.88)

        if previous_vec is None:
            transition_drop = 0.0
        else:
            transition_drop = normalize_range(
                0.62 - float(previous_vec @ win_vec), 0.0, 0.34
            )
        previous_vec = win_vec

        focus = float(
            clamp01(
                0.48 * pair_focus
                + 0.32 * window_focus
                + 0.20 * diary_alignment
                - 0.12 * transition_drop
            )
        )
        if size == 1:
            focus = max(focus, 0.72)

        novelty = 1.0 - normalize_range(to_diary, 0.42, 0.86)
        looseness = float(
            clamp01(0.78 * (1.0 - focus) + 0.22 * global_looseness + 0.12 * novelty)
        )
        metrics.append(
            WindowMetric(
                idx=int(raw["idx"]),
                start=start,
                end=end,
                size=size,
                chars=max(1, len(text.replace("\n", ""))),
                focus=focus,
                looseness=looseness,
                novelty=float(clamp01(novelty)),
                text=text,
            )
        )

    summary = {
        "num_sentences": int(meta["num_sentences"]),
        "num_windows": int(meta["num_windows"]),
        "global_sentence_similarity": global_sim,
        "global_focus": global_focus,
        "global_looseness": global_looseness,
    }
    return metrics, summary
