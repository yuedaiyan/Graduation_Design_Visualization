"""
pipeline.py — 日记可视化主流程（复用根目录已有向量）
"""

import argparse
import hashlib
import json
import re
from collections import Counter
from pathlib import Path

import numpy as np

from .analysis import (
    analyze_function_words,
    classify_emotion,
    get_entry_content_word_data,
    tokenize,
)
from .config import (
    DEFAULT_CONTENT_SIZE,
    DEFAULT_RESOLUTION,
    EMOTION_SYSTEM,
    OUTPUT_BASE_DIR,
    ROOT_DIARY_JSON,
    ROOT_SENTENCE_VECTOR_DIR,
    ROOT_VECTOR_DIR,
)
from .rendering import (
    compose_and_save,
    compose_and_save_svg,
    map_content_words_to_metaballs,
    map_function_word_to_block,
    set_render_sizes,
)

DOC_FREQ_CACHE_VERSION = "v1"


def _program_dir() -> Path:
    return Path(__file__).resolve().parents[1]


def _resolve_from_program_dir(rel_path: str) -> Path:
    return (_program_dir() / rel_path).resolve()


def _default_output_root() -> Path:
    return _resolve_from_program_dir(OUTPUT_BASE_DIR) / _program_dir().name


def _cache_dir() -> Path:
    path = _program_dir() / ".cache" / "derived"
    path.mkdir(parents=True, exist_ok=True)
    return path


def _file_signature(path: Path) -> str:
    stat = path.stat()
    raw = f"{path.resolve()}:{stat.st_size}:{stat.st_mtime_ns}"
    return hashlib.sha256(raw.encode("utf-8")).hexdigest()[:16]


def _normalize_vector(vec: np.ndarray) -> np.ndarray:
    vec = vec.astype(np.float32).reshape(-1)
    norm = np.linalg.norm(vec)
    return vec / (norm + 1e-9)


def _build_emotion_centers(entry_vectors: np.ndarray, emotion_names: list[str]) -> tuple[np.ndarray, list[str]]:
    """
    使用轻量 k-means（cosine）从现有向量构建情绪中心。
    """
    n = len(entry_vectors)
    k = min(len(emotion_names), n)
    if k == 0:
        return np.zeros((0, 0), dtype=np.float32), []

    try:
        from sklearn.cluster import KMeans

        km = KMeans(n_clusters=k, random_state=42, n_init=10)
        km.fit(entry_vectors)
        centers = km.cluster_centers_.astype(np.float32)
        centers = centers / (np.linalg.norm(centers, axis=1, keepdims=True) + 1e-9)
    except Exception:
        init_idx = np.linspace(0, n - 1, k, dtype=int)
        centers = entry_vectors[init_idx].copy()
        for _ in range(20):
            sims = entry_vectors @ centers.T
            labels = np.argmax(sims, axis=1)
            new_centers = []
            for i in range(k):
                members = entry_vectors[labels == i]
                if len(members) == 0:
                    new_centers.append(centers[i])
                else:
                    c = np.mean(members, axis=0)
                    c = c / (np.linalg.norm(c) + 1e-9)
                    new_centers.append(c)
            new_centers = np.array(new_centers, dtype=np.float32)
            if np.allclose(centers, new_centers, atol=1e-5):
                centers = new_centers
                break
            centers = new_centers

    anchor = centers[0]
    scores = centers @ anchor
    order = np.argsort(scores)
    centers = centers[order]
    return centers, emotion_names[:k]


def _next_try_dirs(output_root: Path, export_png: bool = True, export_svg: bool = False) -> tuple[Path | None, Path | None]:
    output_root.mkdir(parents=True, exist_ok=True)
    max_idx = 0
    for child in output_root.iterdir():
        if not child.is_dir():
            continue
        m = re.fullmatch(r"(?:svg_)?try_(\d+)", child.name)
        if m:
            max_idx = max(max_idx, int(m.group(1)))
    next_idx = max_idx + 1
    try_dir = output_root / f"try_{next_idx}" if export_png else None
    svg_try_dir = output_root / f"svg_try_{next_idx}" if export_svg else None
    if try_dir is not None:
        try_dir.mkdir(parents=True, exist_ok=False)
    if svg_try_dir is not None:
        svg_try_dir.mkdir(parents=True, exist_ok=False)
    return try_dir, svg_try_dir


def _safe_stem(date: str, count: int) -> str:
    base = re.sub(r"[^0-9A-Za-z_-]", "_", date.strip()) or "unknown_date"
    return base if count == 0 else f"{base}_{count + 1}"


def _entry_date(entry: dict) -> str:
    return str(entry.get("date_iso") or entry.get("date") or "").strip()


def _load_entries_by_stem(all_entries: list[dict]) -> dict[str, dict]:
    entries_by_stem = {}
    seen = {}
    for entry in all_entries:
        date_key = _entry_date(entry)
        if not date_key:
            continue
        count = seen.get(date_key, 0)
        seen[date_key] = count + 1
        entries_by_stem[_safe_stem(date_key, count)] = entry
    return entries_by_stem


def _load_sentence_vectors(sentence_vector_dir: Path, stem: str) -> np.ndarray | None:
    sent_path = sentence_vector_dir / stem / "sentence_vectors.npy"
    if not sent_path.exists():
        return None
    return np.load(sent_path).astype(np.float32)


def _load_or_build_doc_freq(data_path: Path, all_entries: list[dict]) -> tuple[Counter, int]:
    cache_path = _cache_dir() / f"doc_freq_{DOC_FREQ_CACHE_VERSION}_{_file_signature(data_path)}.json"
    if cache_path.exists():
        with cache_path.open("r", encoding="utf-8") as f:
            payload = json.load(f)
        return Counter(payload["doc_freq"]), int(payload["total_docs"])

    doc_freq = Counter()
    for entry in all_entries:
        _, content_tokens = tokenize(entry["content"])
        for w in set(content_tokens):
            doc_freq[w] += 1

    payload = {
        "source": str(data_path),
        "version": DOC_FREQ_CACHE_VERSION,
        "total_docs": len(all_entries),
        "doc_freq": dict(doc_freq),
    }
    with cache_path.open("w", encoding="utf-8") as f:
        json.dump(payload, f, ensure_ascii=False)
    return doc_freq, len(all_entries)


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Generate color-block contour diary visuals")
    parser.add_argument("--date", default=None, help="Optional date/stem filter, e.g. 2026-03-10")
    parser.add_argument("--limit", type=int, default=None, help="Optional max number of entries to render")
    parser.add_argument(
        "--out-dir",
        default=None,
        help="Output base dir. Defaults to ../output_All/diary_ColorBlock_ContourLines",
    )
    output_group = parser.add_mutually_exclusive_group()
    output_group.add_argument("--png-only", action="store_true", help="Only render PNGs; skip sibling svg_try_N output")
    output_group.add_argument("--svg", action="store_true", help="Render sibling svg_try_N output alongside PNGs. This is the default")
    output_group.add_argument("--svg-only", action="store_true", help="Only render SVGs; skip sibling try_N PNG output")
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
    parser.add_argument(
        "--contour-density",
        type=float,
        default=1.0,
        help="Contour-line density multiplier. 1.0 keeps the default thresholds; 2.0 roughly doubles contour lines.",
    )
    args = parser.parse_args()
    if args.content_size < 1:
        parser.error("--content-size must be a positive integer")
    if args.resolution < 1:
        parser.error("--resolution must be a positive integer")
    if args.contour_density <= 0:
        parser.error("--contour-density must be greater than 0")
    return args


def run() -> None:
    args = _parse_args()
    set_render_sizes(args.content_size, args.resolution)
    data_path = _resolve_from_program_dir(ROOT_DIARY_JSON)
    vector_dir = _resolve_from_program_dir(ROOT_VECTOR_DIR)
    sentence_vector_dir = _resolve_from_program_dir(ROOT_SENTENCE_VECTOR_DIR)
    output_root = Path(args.out_dir).resolve() if args.out_dir else _default_output_root()

    with data_path.open("r", encoding="utf-8") as f:
        all_entries = [e for e in json.load(f) if e.get("content", "").strip()]

    entries_by_stem = _load_entries_by_stem(all_entries)

    vector_files = sorted([p for p in vector_dir.iterdir() if p.suffix == ".npy"])
    if not vector_files:
        print(f"[主程序] 未找到向量文件：{vector_dir}")
        return

    jobs = []
    missing_entry_count = 0
    missing_sentence_vector_count = 0
    for vf in vector_files:
        stem = vf.stem
        if args.date and args.date not in stem:
            continue
        entry = entries_by_stem.get(stem)
        if entry is None:
            missing_entry_count += 1
            entry = {"date": stem, "content": ""}
        vec = _normalize_vector(np.load(vf))
        sent_vectors = _load_sentence_vectors(sentence_vector_dir, stem)
        if sent_vectors is None:
            missing_sentence_vector_count += 1
        jobs.append(
            {
                "vec_name": vf.name,
                "vec_stem": stem,
                "entry": entry,
                "entry_vec": vec,
                "sentence_vectors": sent_vectors,
            }
        )
        if args.limit and len(jobs) >= args.limit:
            break

    if not jobs:
        print("[主程序] 没有可处理条目。")
        return

    print(f"[主程序] 匹配成功：{len(jobs)} / {len(vector_files)}")
    if missing_entry_count:
        print(f"[主程序] 警告：{missing_entry_count} 个向量没有对应日记，将只生成背景/可用图层。")
    if missing_sentence_vector_count:
        print(f"[主程序] 警告：{missing_sentence_vector_count} 个条目没有句向量，将使用篇目向量兜底。")

    doc_freq, total_docs = _load_or_build_doc_freq(data_path, all_entries)

    entry_vectors = np.array([j["entry_vec"] for j in jobs], dtype=np.float32)
    emotion_names = list(EMOTION_SYSTEM.keys())
    emotion_centers, used_emotion_names = _build_emotion_centers(entry_vectors, emotion_names)

    try_dir, svg_try_dir = _next_try_dirs(
        output_root,
        export_png=not args.svg_only,
        export_svg=not args.png_only,
    )
    if try_dir is not None:
        print(f"[主程序] 本次输出目录：{try_dir}")
    if svg_try_dir is not None:
        print(f"[主程序] 本次 SVG 输出目录：{svg_try_dir}")

    for i, job in enumerate(jobs, start=1):
        entry = job["entry"]
        vec_stem = job["vec_stem"]
        ev = job["entry_vec"]
        out_path = try_dir / f"{vec_stem}.png" if try_dir is not None else None
        svg_out_path = svg_try_dir / f"{vec_stem}.svg" if svg_try_dir is not None else None

        print(f"[{i:02d}/{len(jobs)}] 处理 {job['vec_name']} ...")
        emotion = classify_emotion(ev, emotion_centers, used_emotion_names)

        fw_tokens = analyze_function_words(entry["content"], job["sentence_vectors"], ev)
        _, clusters, weights = get_entry_content_word_data(entry["content"], doc_freq, total_docs)
        block_params = [
            map_function_word_to_block(tok, emotion, ev, token_index=idx, total_tokens=len(fw_tokens))
            for idx, tok in enumerate(fw_tokens)
        ]
        metaball_params = map_content_words_to_metaballs(clusters, weights, ev)

        if out_path is not None:
            compose_and_save(
                entry,
                emotion,
                block_params,
                metaball_params,
                str(out_path),
                contour_density=args.contour_density,
            )
        if svg_out_path is not None:
            compose_and_save_svg(
                emotion,
                block_params,
                metaball_params,
                str(svg_out_path),
                contour_density=args.contour_density,
            )
        export_paths = []
        if out_path is not None:
            export_paths.append(str(out_path))
        if svg_out_path is not None:
            export_paths.append(str(svg_out_path))
        export_msg = " / ".join(export_paths)
        print(f"  → 情绪: {emotion}，方块数: {len(block_params)}，导出: {export_msg}")

    if try_dir is not None:
        print(f"\n[主程序] 完成。输出目录：{try_dir}")
    else:
        print("\n[主程序] 完成。")
    if svg_try_dir is not None:
        print(f"[主程序] SVG 输出目录：{svg_try_dir}")
