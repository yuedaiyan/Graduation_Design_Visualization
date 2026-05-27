"""
Put diary vectors into multiple background text spaces.

Run from the repository root or from this folder:
    python3 MultiBackground/main.py

Outputs are written to:
    output_All/MultiBackground/try_N/

This script only downloads/samples/builds the backgrounds that you enable in
ENABLED_BACKGROUNDS. Each background has its own text, vector, and 2D cache.
"""

from __future__ import annotations

import gc
import glob
import hashlib
import csv
import json
import os
import re
import shutil
import sys
import time
import urllib.request
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Callable, Iterable

from diary_MultiBackground.config import *  # noqa: F403
from diary_MultiBackground.cleaning import clean_background_text

ROOT = Path(__file__).resolve().parents[1]
SCRIPT_DIR = Path(__file__).resolve().parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

if HF_ENDPOINT:
    os.environ.setdefault("HF_ENDPOINT", HF_ENDPOINT)
elif os.environ.get("HF_ENDPOINT") == "":
    os.environ.pop("HF_ENDPOINT", None)
os.environ.setdefault("TOKENIZERS_PARALLELISM", "false")
os.environ.setdefault("MPLCONFIGDIR", str(SCRIPT_DIR / ".plot_cache" / "mplconfig"))
os.environ.setdefault("XDG_CACHE_HOME", str(SCRIPT_DIR / ".plot_cache" / "xdg_cache"))
Path(os.environ["MPLCONFIGDIR"]).mkdir(parents=True, exist_ok=True)
Path(os.environ["XDG_CACHE_HOME"]).mkdir(parents=True, exist_ok=True)

import numpy as np
import torch
from diary_core.embedding import QwenEmbedder
from diary_MultiBackground.output import (
    hull_boundary,
    output_image_formats,
    polygon_area,
    plot_world,
    plot_world_diary_area_scaled,
    plot_zoom,
    save_points_csv,
    scaled_area_points,
    summarize_metrics,
)
from diary_MultiBackground.viewer import (
    load_diary_text_lookup,
    write_viewer_background_data,
    write_viewer_shell,
)
from diary_BackgroundRelations.main import build_background_relation_outputs
from sklearn.decomposition import PCA


def diary_area_scaled_output_stem() -> str:
    scale_percent = int(round(DIARY_AREA_SCALE_RATIO * 100))
    return f"03_world_diary_area{scale_percent}"


@dataclass
class PointData:
    labels: list[str]
    vectors: np.ndarray
    metas: list[dict[str, Any]] | None = None


@dataclass
class TextCache:
    labels: list[str]
    texts: list[str]
    metas: list[dict[str, Any]]
    path: Path
    byte_size: int


def resolve_path(path: str) -> Path:
    p = Path(path)
    if p.is_absolute():
        return p
    return (SCRIPT_DIR / p).resolve()


def slugify(value: str) -> str:
    value = re.sub(r"[^a-zA-Z0-9_.-]+", "_", value.strip())
    return value.strip("_") or "background"


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


def load_diary_vectors(input_dir: Path) -> PointData:
    files = sorted(glob.glob(str(input_dir / "*.npy")))
    if not files:
        raise FileNotFoundError(f"No diary .npy files found in {input_dir}")
    labels = [Path(fp).stem for fp in files]
    vectors = np.stack([np.load(fp).reshape(-1).astype(np.float32) for fp in files])
    return PointData(labels=labels, vectors=l2_normalize(vectors), metas=None)


def require_datasets():
    try:
        from datasets import load_dataset
    except ImportError as exc:
        raise SystemExit(
            "Missing dependency: datasets\n"
            "Install it with:\n"
            "    python3 -m pip install datasets\n"
        ) from exc
    return load_dataset


def rewrite_hf_url(url: str) -> str:
    if not HF_ENDPOINT:
        return url
    return url.replace("https://huggingface.co", HF_ENDPOINT.rstrip("/"))


def row_filter_text(row: dict[str, Any], spec: dict[str, Any]) -> str:
    default_fields = [
        "title",
        "name",
        "category",
        "label",
        "topic",
        "tag",
        "tags",
        "url",
        "text",
        "content",
        "review",
        "answer",
        "question",
        "lyric",
        "lyrics",
        "INSTRUCTION",
        "RESPONSE",
        "SOURCE",
        "METADATA",
    ]
    fields = list(dict.fromkeys(list(spec.get("filter_fields") or []) + default_fields))
    return normalize_text(" ".join(flatten_value(row.get(field)) for field in fields if field in row))


def row_matches_filters(row: dict[str, Any], spec: dict[str, Any]) -> bool:
    include = [str(x) for x in spec.get("include_keywords") or [] if str(x)]
    exclude = [str(x) for x in spec.get("exclude_keywords") or [] if str(x)]
    if not include and not exclude:
        return True

    haystack = row_filter_text(row, spec)
    if include and not any(keyword in haystack for keyword in include):
        return False
    if exclude and any(keyword in haystack for keyword in exclude):
        return False
    return True


def with_source_meta(row: dict[str, Any], spec: dict[str, Any]) -> dict[str, Any]:
    row = dict(row)
    if spec.get("source_name"):
        row.setdefault("source_name", spec["source_name"])
    if spec.get("source_kind"):
        row.setdefault("source_kind", spec["source_kind"])
    return row


def iter_local_file_rows(path: Path) -> Iterable[dict[str, Any]]:
    suffix = path.suffix.lower()
    if suffix == ".jsonl":
        with path.open(encoding="utf-8") as f:
            for i, line in enumerate(f):
                line = line.strip()
                if not line:
                    continue
                value = json.loads(line)
                if isinstance(value, dict):
                    row = value
                else:
                    row = {"text": flatten_value(value)}
                row.setdefault("title", f"{path.stem} #{i + 1}")
                row.setdefault("file", str(path))
                yield row
        return

    if suffix == ".json":
        value = json.loads(path.read_text(encoding="utf-8"))
        if isinstance(value, dict):
            records = value.get("data") or value.get("items") or value.get("records")
            if isinstance(records, list):
                values = records
            else:
                values = [value]
        elif isinstance(value, list):
            values = value
        else:
            values = [value]
        for i, item in enumerate(values):
            row = dict(item) if isinstance(item, dict) else {"text": flatten_value(item)}
            row.setdefault("title", f"{path.stem} #{i + 1}")
            row.setdefault("file", str(path))
            yield row
        return

    if suffix == ".csv":
        with path.open(encoding="utf-8-sig", newline="") as f:
            for i, row in enumerate(csv.DictReader(f)):
                row = dict(row)
                row.setdefault("title", f"{path.stem} #{i + 1}")
                row.setdefault("file", str(path))
                yield row
        return

    if suffix in {".txt", ".md"}:
        text = path.read_text(encoding="utf-8", errors="ignore")
        yield {"title": path.stem, "text": text, "file": str(path)}


def iter_local_text_rows(spec: dict[str, Any]) -> Iterable[dict[str, Any]]:
    paths = []
    for raw_path in spec.get("paths") or []:
        path = resolve_path(str(raw_path))
        if path.is_dir():
            paths.extend(
                p
                for p in path.rglob("*")
                if p.is_file() and p.suffix.lower() in {".txt", ".md", ".json", ".jsonl", ".csv"}
            )
        elif path.is_file():
            paths.append(path)

    for path in sorted(set(paths)):
        for row in iter_local_file_rows(path):
            yield with_source_meta(row, spec)


def download_cache_path(url: str) -> Path:
    digest = hashlib.sha1(url.encode("utf-8")).hexdigest()[:16]
    suffix = Path(url.split("?", 1)[0]).suffix or ".data"
    return SCRIPT_DIR / CACHE_DIR / "_downloads" / f"{digest}{suffix}"


def read_json_records(value: Any) -> list[Any]:
    if isinstance(value, dict):
        for key in ["data", "items", "records", "songs", "lyrics"]:
            records = value.get(key)
            if isinstance(records, list):
                return records
        return [value]
    if isinstance(value, list):
        return value
    return [value]


def iter_url_json_rows(spec: dict[str, Any]) -> Iterable[dict[str, Any]]:
    for url in spec.get("urls") or []:
        cache_path = download_cache_path(str(url))
        if not cache_path.exists():
            cache_path.parent.mkdir(parents=True, exist_ok=True)
            with urllib.request.urlopen(str(url), timeout=60) as response:
                cache_path.write_bytes(response.read())
        value = json.loads(cache_path.read_text(encoding="utf-8"))
        for i, item in enumerate(read_json_records(value)):
            row = dict(item) if isinstance(item, dict) else {"text": flatten_value(item)}
            row.setdefault("title", f"{cache_path.stem} #{i + 1}")
            row.setdefault("url", str(url))
            yield with_source_meta(row, spec)


def iter_single_source_rows(spec: dict[str, Any]) -> Iterable[dict[str, Any]]:
    kind = spec["source_kind"]
    split = spec.get("split") or "train"

    if kind == "local_texts":
        for row in iter_local_text_rows(spec):
            if row_matches_filters(row, spec):
                yield row
        return

    if kind == "url_json":
        for row in iter_url_json_rows(spec):
            if row_matches_filters(row, spec):
                yield row
        return

    load_dataset = require_datasets()

    if kind in {"hf_dataset", "hf_search_dataset"}:
        kwargs = {"split": split, "streaming": True}
        if spec.get("config"):
            ds = load_dataset(spec["dataset_name"], spec["config"], **kwargs)
        else:
            ds = load_dataset(spec["dataset_name"], **kwargs)
    elif kind == "hf_parquet":
        data_files = [rewrite_hf_url(url) for url in spec["data_files"]]
        ds = load_dataset("parquet", data_files=data_files, split=split, streaming=True)
    else:
        raise ValueError(f"Unknown source_kind: {kind}")

    buffer_size = int(spec.get("shuffle_buffer") or SHUFFLE_BUFFER_SIZE)
    if buffer_size > 0:
        ds = ds.shuffle(seed=RANDOM_SEED, buffer_size=buffer_size)

    scanned = 0
    max_scan_rows = int(spec.get("max_scan_rows") or 0)
    if MAX_SCAN_ROWS_OVERRIDE:
        max_scan_rows = min(max_scan_rows, MAX_SCAN_ROWS_OVERRIDE) if max_scan_rows else MAX_SCAN_ROWS_OVERRIDE
    for row in ds:
        scanned += 1
        if max_scan_rows and scanned > max_scan_rows:
            break
        row = with_source_meta(dict(row), spec)
        if row_matches_filters(row, spec):
            yield row


def iter_dataset_rows(spec: dict[str, Any]) -> Iterable[dict[str, Any]]:
    if spec["source_kind"] == "mixed_sources":
        for source in spec.get("sources") or []:
            source_spec = dict(spec)
            source_spec.pop("sources", None)
            source_spec.update(source)
            try:
                yielded = False
                for row in iter_dataset_rows(source_spec):
                    yielded = True
                    yield row
                if not yielded:
                    print(f"No rows from source {source_spec.get('source_name') or source_spec.get('dataset_name') or source_spec.get('paths')}")
            except Exception as exc:
                print(
                    "Skipping source "
                    f"{source_spec.get('source_name') or source_spec.get('dataset_name') or source_spec.get('paths')}: "
                    f"{type(exc).__name__}: {exc}"
                )
        return

    yield from iter_single_source_rows(spec)


def flatten_value(value: Any) -> str:
    if value is None:
        return ""
    if isinstance(value, str):
        return value
    if isinstance(value, (int, float, bool)):
        return str(value)
    if isinstance(value, list):
        return "\n".join(flatten_value(x) for x in value if flatten_value(x))
    if isinstance(value, dict):
        parts = []
        for key in ["title", "question", "answer", "content", "text", "paragraphs"]:
            if key in value:
                part = flatten_value(value[key])
                if part:
                    parts.append(part)
        if parts:
            return "\n".join(parts)
        return json.dumps(value, ensure_ascii=False)
    return str(value)


def effective_min_text_chars(spec: dict[str, Any] | None = None) -> int:
    if spec and spec.get("min_text_chars") is not None:
        return int(spec["min_text_chars"])
    return MIN_TEXT_CHARS


def effective_chunk_min_chars(spec: dict[str, Any] | None = None) -> int:
    # Keep very short-form corpora usable by packing several source rows together,
    # but keep individual long-text chunks near one diary-day scale.
    return min(effective_min_text_chars(spec), BACKGROUND_CHUNK_MIN_CHARS)


def first_field(row: dict[str, Any], fields: list[str]) -> str:
    for field in fields:
        if field in row:
            text = flatten_value(row.get(field)).strip()
            if text:
                return text
    return ""


def best_text_field(row: dict[str, Any], fields: list[str], min_chars: int) -> str:
    candidates = []
    for field in fields:
        if field not in row:
            continue
        text = flatten_value(row.get(field)).strip()
        if len(text) >= min_chars:
            return text
        if text:
            candidates.append(text)
    return max(candidates, key=len, default="")


def fallback_text(row: dict[str, Any], min_chars: int) -> str:
    preferred = [
        "text",
        "content",
        "review",
        "answer",
        "question",
        "title",
        "description",
        "paragraphs",
        "sentence",
        "lyric",
        "lyrics",
        "body",
        "abstract",
        "INSTRUCTION",
        "RESPONSE",
        "prompt",
        "response",
    ]
    text = best_text_field(row, preferred, min_chars)
    if text:
        return text
    candidates = []
    for value in row.values():
        text = flatten_value(value).strip()
        if len(text) >= min_chars:
            candidates.append(text)
    return max(candidates, key=len, default="")


def normalize_text(text: str) -> str:
    text = re.sub(r"\s+", " ", text or "").strip()
    return text


def titled_text(label: str, text: str) -> str:
    text = normalize_text(text)
    if TITLE_PREFIX and label and label not in text[:80]:
        return f"{label}。{text}"
    return text


def compact_meta_value(value: Any, max_len: int = 120) -> str:
    text = normalize_text(flatten_value(value))
    return text[:max_len]


def length_bucket(length: int) -> str:
    if length < BACKGROUND_CHUNK_MIN_CHARS:
        return "short"
    if length <= BACKGROUND_CHUNK_MAX_CHARS:
        return "diary_scale"
    if length <= 1200:
        return "medium_source"
    return "long_source"


def classify_zh_wikipedia(label: str, text: str) -> str:
    sample = f"{label} {text[:500]}"
    if re.search(r"(县|镇|乡|村|街道|行政区|自治州|市辖区|地级市|省份|岛|河|湖|山脉)", sample):
        return "wiki_place_admin"
    if re.search(r"(足球运动员|篮球运动员|运动员|奥运|联赛|球队|足球俱乐部)", sample):
        return "wiki_sports"
    if re.search(r"(出生|逝世|演员|歌手|作家|导演|政治人物|教授|人物|学者|诗人|画家)", sample):
        return "wiki_person"
    if "《" in sample and "》" in sample or re.search(
        r"(电影|电视剧|专辑|歌曲|小说|漫画|动画|书籍|节目)", sample
    ):
        return "wiki_media_culture"
    if re.search(r"(科学|技术|数学|物理|化学|生物|医学|计算机|工程|天文)", sample):
        return "wiki_science_technology"
    if re.search(r"(历史|战争|王朝|政治|政府|法律|革命|条约|总统|皇帝)", sample):
        return "wiki_history_politics"
    if re.search(r"(经济|社会|教育|宗教|语言|文化|哲学|艺术|民俗)", sample):
        return "wiki_society_culture"
    return "wiki_other"


# Thematically unified backgrounds: already filtered by topic, so generic regex
# sub-classification would produce misleading strata labels.
_SINGLE_STRATUM_BACKGROUNDS = frozenset({
    "thucnews", "zhihu_kol", "douban_reviews", "historical_diary",
    "modern_essay", "psych_discourse", "legal_text", "religious_text",
    "lyrics", "interview_oral", "forum_emotional", "self_help",
    "travel_writing", "food_writing", "academic_humanities",
    "children_writing", "ad_copy", "dream_record",
})


def classify_generic_background(bg_key: str, label: str, text: str, row: dict[str, Any]) -> str:
    if bg_key == "zh_wikipedia":
        return classify_zh_wikipedia(label, text)
    if bg_key == "weibo_senti":
        sentiment = compact_meta_value(row.get("label"), 20) or "unknown"
        return f"weibo_sentiment_{slugify(sentiment)}"
    if bg_key == "classical_poetry":
        dynasty = compact_meta_value(row.get("dynasty"), 20)
        return f"classical_{slugify(dynasty)}" if dynasty else "classical_poetry"
    if bg_key in _SINGLE_STRATUM_BACKGROUNDS:
        return bg_key

    for field in ["category", "label", "topic", "tag", "tags", "movie", "name"]:
        if field in row:
            value = compact_meta_value(row.get(field), 40)
            if value:
                return f"{bg_key}_{slugify(value)}"

    sample = f"{label} {text[:500]}"
    if re.search(r"(财经|经济|股票|基金|公司|税|市场|消费)", sample):
        return f"{bg_key}_economy"
    if re.search(r"(教育|学校|考试|大学|学生|课程)", sample):
        return f"{bg_key}_education"
    if re.search(r"(科技|技术|互联网|手机|AI|计算机|软件)", sample, flags=re.I):
        return f"{bg_key}_technology"
    if re.search(r"(电影|电视剧|音乐|综艺|小说|游戏|明星)", sample):
        return f"{bg_key}_culture_media"
    if re.search(r"(情感|生活|家庭|朋友|喜欢|难过|开心|焦虑)", sample):
        return f"{bg_key}_daily_feeling"
    return f"{bg_key}_other"


def _fmt_time(s: float) -> str:
    if s < 60:
        return f"{s:.1f}s"
    m = int(s // 60)
    sec = s - m * 60
    if m < 60:
        return f"{m}m{sec:.0f}s"
    h = m // 60
    m = m % 60
    return f"{h}h{m}m"


def source_meta(row: dict[str, Any]) -> dict[str, str]:
    meta = {}
    for field in [
        "category",
        "label",
        "topic",
        "tag",
        "tags",
        "title",
        "movie",
        "name",
        "dynasty",
        "author",
        "date",
        "time",
        "year",
        "created_at",
        "publish_time",
        "source_name",
        "source_kind",
        "file",
        "url",
        "SOURCE",
        "METADATA",
    ]:
        if field in row:
            value = compact_meta_value(row.get(field))
            if value:
                meta[field] = value
    return meta


def split_long_unit(unit: str) -> list[str]:
    return [
        unit[start : start + BACKGROUND_CHUNK_MAX_CHARS].strip()
        for start in range(0, len(unit), BACKGROUND_CHUNK_MAX_CHARS)
        if unit[start : start + BACKGROUND_CHUNK_MAX_CHARS].strip()
    ]


def split_text_chunks(text: str) -> list[str]:
    text = normalize_text(text)
    if not text:
        return []
    if len(text) <= BACKGROUND_CHUNK_MAX_CHARS:
        return [text]

    units = [
        part.strip()
        for part in re.split(r"(?<=[。！？；!?;])\s*", text)
        if part.strip()
    ]
    chunks: list[str] = []
    current = ""

    for unit in units:
        if len(unit) > BACKGROUND_CHUNK_MAX_CHARS:
            if current:
                chunks.append(current)
                current = ""
            chunks.extend(split_long_unit(unit))
            continue

        candidate = f"{current} {unit}".strip() if current else unit
        if len(candidate) <= BACKGROUND_CHUNK_MAX_CHARS:
            current = candidate
            if len(current) >= BACKGROUND_CHUNK_TARGET_CHARS:
                chunks.append(current)
                current = ""
            continue

        if current:
            chunks.append(current)
        current = unit

    if current:
        if chunks and len(current) < BACKGROUND_CHUNK_MIN_CHARS:
            candidate = f"{chunks[-1]} {current}".strip()
            if len(candidate) <= BACKGROUND_CHUNK_MAX_CHARS:
                chunks[-1] = candidate
            else:
                chunks.append(current)
        else:
            chunks.append(current)

    return chunks


def row_to_records(
    bg_key: str, row: dict[str, Any], spec: dict[str, Any]
) -> list[dict[str, Any]]:
    min_chars = effective_min_text_chars(spec)
    label = first_field(row, spec.get("title_fields") or [])
    text = best_text_field(row, spec.get("text_fields") or [], min_chars) or fallback_text(
        row, min_chars
    )
    label = re.sub(r"\s+", " ", label).strip()[:80]
    raw_text = normalize_text(text)
    raw_text = clean_background_text(bg_key, raw_text)
    if len(raw_text) < min_chars:
        return []

    raw_length = len(raw_text)
    stratum = classify_generic_background(bg_key, label, raw_text, row)
    chunks = split_text_chunks(titled_text(label, raw_text))
    records = []
    for idx, chunk in enumerate(chunks):
        if len(chunk) < effective_chunk_min_chars(spec):
            continue
        chunk_label = label or "untitled"
        if len(chunks) > 1:
            chunk_label = f"{chunk_label} #{idx + 1}"
        meta = {
            "schema_version": TEXT_CACHE_SCHEMA_VERSION,
            "source": bg_key,
            "source_label": label or "untitled",
            "stratum": stratum,
            "length_bucket": length_bucket(raw_length),
            "raw_length": raw_length,
            "chunk_index": idx,
            "chunk_count": len(chunks),
            "chunk_length": len(chunk),
            "source_meta": source_meta(row),
        }
        records.append({"label": chunk_label, "text": chunk, "meta": meta})
    return records


def text_cache_path(cache_dir: Path, bg_key: str, spec: dict[str, Any]) -> Path:
    min_chars = effective_min_text_chars(spec)
    return cache_dir / bg_key / (
        f"texts_items{TEXT_CACHE_TARGET_ITEMS}_bytes{TEXT_CACHE_MAX_BYTES}_"
        f"chunk{BACKGROUND_CHUNK_MIN_CHARS}-{BACKGROUND_CHUNK_MAX_CHARS}_"
        f"min{min_chars}_seed{RANDOM_SEED}.jsonl"
    )


def cache_version_suffix() -> str:
    return (
        f"{TEXT_CACHE_SCHEMA_VERSION}_"
        f"chunk{BACKGROUND_CHUNK_MIN_CHARS}-{BACKGROUND_CHUNK_MAX_CHARS}"
    )


def stratum_cap_for(bg_key: str) -> int:
    if not STRATIFIED_SAMPLING:
        return TEXT_CACHE_TARGET_ITEMS
    expected = max(1, int(STRATUM_EXPECTED_COUNTS.get(bg_key, 1)))
    max_share = max(STRATUM_MAX_SHARE, 1.05 / expected)
    return max(1, int(TEXT_CACHE_TARGET_ITEMS * max_share))


def cached_row_to_record(
    row: dict[str, Any], bg_key: str, spec: dict[str, Any]
) -> dict[str, Any]:
    label = str(row.get("label") or row.get("title") or "untitled").strip()
    text = normalize_text(str(row.get("text") or ""))
    meta = row.get("meta") if isinstance(row.get("meta"), dict) else {}
    meta = dict(meta)
    meta.setdefault("schema_version", row.get("schema_version") or "legacy")
    meta.setdefault("source", bg_key)
    meta.setdefault("source_label", label or "untitled")
    meta.setdefault("stratum", row.get("stratum") or "legacy_unstratified")
    meta.setdefault("length_bucket", length_bucket(len(text)))
    meta.setdefault("raw_length", len(text))
    meta.setdefault("chunk_index", int(row.get("chunk_index") or 0))
    meta.setdefault("chunk_count", int(row.get("chunk_count") or 1))
    meta["chunk_length"] = len(text)
    return {"label": label or "untitled", "text": text, "meta": meta}


def read_text_cache(path: Path, limit: int, spec: dict[str, Any]) -> TextCache:
    labels, texts, metas = [], [], []
    bytes_read = path.stat().st_size
    bg_key = path.parent.name
    with path.open(encoding="utf-8") as f:
        for line in f:
            row = json.loads(line)
            record = cached_row_to_record(row, bg_key, spec)
            if len(record["text"]) < effective_chunk_min_chars(spec):
                continue
            labels.append(record["label"])
            texts.append(record["text"])
            metas.append(record["meta"])
            if len(texts) >= limit:
                break
    if not texts:
        raise RuntimeError(f"Text cache is empty: {path}")
    return TextCache(labels=labels, texts=texts, metas=metas, path=path, byte_size=bytes_read)


def write_text_cache(path: Path, text_cache: TextCache) -> TextCache:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as f:
        for label, text, meta in zip(text_cache.labels, text_cache.texts, text_cache.metas):
            f.write(
                json.dumps({"label": label, "text": text, "meta": meta}, ensure_ascii=False)
                + "\n"
            )
    return TextCache(
        labels=text_cache.labels,
        texts=text_cache.texts,
        metas=text_cache.metas,
        path=path,
        byte_size=path.stat().st_size,
    )


def find_larger_text_cache(
    cache_dir: Path, bg_key: str, spec: dict[str, Any]
) -> Path | None:
    min_chars = effective_min_text_chars(spec)
    candidates: list[tuple[int, Path]] = []
    pattern = re.compile(
        rf"texts_items(\d+)_bytes\d+_chunk{BACKGROUND_CHUNK_MIN_CHARS}-{BACKGROUND_CHUNK_MAX_CHARS}_"
        rf"min{min_chars}_seed{RANDOM_SEED}\.jsonl"
    )
    for path in (cache_dir / bg_key).glob("texts_items*_*.jsonl"):
        m = pattern.fullmatch(path.name)
        if m and int(m.group(1)) >= TEXT_CACHE_TARGET_ITEMS:
            candidates.append((int(m.group(1)), path))

    return sorted(candidates)[0][1] if candidates else None


def collect_texts(bg_key: str, spec: dict[str, Any], cache_dir: Path) -> TextCache:
    min_chars = effective_min_text_chars(spec)
    path = text_cache_path(cache_dir, bg_key, spec)
    if path.exists():
        try:
            return read_text_cache(path, VECTORIZE_TEXT_LIMIT, spec)
        except RuntimeError:
            print(f"Discarding unusable text cache: {path}")
            path.unlink()

    larger_cache = find_larger_text_cache(cache_dir, bg_key, spec)
    if larger_cache is not None:
        print(f"Using local larger text cache: {larger_cache}")
        text_cache = read_text_cache(larger_cache, TEXT_CACHE_TARGET_ITEMS, spec)
        write_text_cache(path, text_cache)
        return read_text_cache(path, VECTORIZE_TEXT_LIMIT, spec)

    path.parent.mkdir(parents=True, exist_ok=True)
    labels, texts, metas = [], [], []
    bytes_written = 0
    seen_hashes: set[str] = set()
    stratum_counts: dict[str, int] = {}
    stratum_cap = stratum_cap_for(bg_key)
    short_buffers: dict[str, list[dict[str, Any]]] = {}
    overflow_records: dict[str, list[dict[str, Any]]] = {}

    def write_record(f, record: dict[str, Any], enforce_stratum_cap: bool = True) -> bool:
        nonlocal bytes_written
        text = record["text"]
        stratum = str(record["meta"]["stratum"])
        if (
            enforce_stratum_cap
            and STRATIFIED_SAMPLING
            and stratum_counts.get(stratum, 0) >= stratum_cap
        ):
            bucket = overflow_records.setdefault(stratum, [])
            overflow_limit = min(TEXT_CACHE_TARGET_ITEMS, 20000)
            if len(bucket) < overflow_limit:
                bucket.append(record)
            return False

        digest = hashlib.sha1(text.encode("utf-8")).hexdigest()
        if digest in seen_hashes:
            return False
        seen_hashes.add(digest)

        line = json.dumps(record, ensure_ascii=False) + "\n"
        encoded_len = len(line.encode("utf-8"))
        if labels and bytes_written + encoded_len > TEXT_CACHE_MAX_BYTES:
            return True

        f.write(line)
        labels.append(record["label"])
        texts.append(text)
        metas.append(record["meta"])
        stratum_counts[stratum] = stratum_counts.get(stratum, 0) + 1
        bytes_written += encoded_len
        return len(texts) >= TEXT_CACHE_TARGET_ITEMS

    def packed_record(buffer: list[dict[str, Any]], stratum: str) -> dict[str, Any]:
        labels_joined = [item["label"] for item in buffer]
        text = " ".join(item["text"] for item in buffer)
        raw_length = sum(int(item["meta"].get("raw_length") or len(item["text"])) for item in buffer)
        meta = {
            "schema_version": TEXT_CACHE_SCHEMA_VERSION,
            "source": bg_key,
            "source_label": labels_joined[0],
            "stratum": stratum,
            "length_bucket": "packed_short",
            "raw_length": raw_length,
            "chunk_index": 0,
            "chunk_count": 1,
            "chunk_length": len(text),
            "packed_source_count": len(buffer),
            "packed_source_labels": labels_joined[:8],
            "source_meta": {
                key: str(value)[:120]
                for item in buffer
                for key, value in (
                    item.get("meta", {}).get("source_meta")
                    if isinstance(item.get("meta", {}).get("source_meta"), dict)
                    else {}
                ).items()
                if value
            },
        }
        label = labels_joined[0] if len(buffer) == 1 else f"{labels_joined[0]} +{len(buffer) - 1}"
        return {"label": label, "text": text, "meta": meta}

    with path.open("w", encoding="utf-8") as f:
        for row in iter_dataset_rows(spec):
            stop = False
            for record in row_to_records(bg_key, row, spec):
                text = record["text"]
                stratum = str(record["meta"]["stratum"])
                if PACK_SHORT_TEXTS and len(text) < BACKGROUND_CHUNK_MIN_CHARS:
                    buffer = short_buffers.setdefault(stratum, [])
                    buffer.append(record)
                    packed_text = " ".join(item["text"] for item in buffer)
                    if len(packed_text) < BACKGROUND_CHUNK_MIN_CHARS:
                        continue
                    while len(packed_text) > BACKGROUND_CHUNK_MAX_CHARS and len(buffer) > 1:
                        buffer.pop()
                        packed_text = " ".join(item["text"] for item in buffer)
                    record = packed_record(buffer, stratum)
                    short_buffers[stratum] = []

                stop = write_record(f, record)
                if stop:
                    break
            if stop:
                break

        if len(texts) < TEXT_CACHE_TARGET_ITEMS and overflow_records:
            print(
                f"Relaxing stratum cap for {bg_key}: "
                f"{len(texts)}/{TEXT_CACHE_TARGET_ITEMS} items after first pass."
            )
            overflow_by_stratum = {key: list(value) for key, value in overflow_records.items()}

            stop = False
            while not stop and len(texts) < TEXT_CACHE_TARGET_ITEMS and overflow_by_stratum:
                for stratum in sorted(
                    list(overflow_by_stratum),
                    key=lambda key: (stratum_counts.get(key, 0), key),
                ):
                    bucket = overflow_by_stratum[stratum]
                    if not bucket:
                        overflow_by_stratum.pop(stratum, None)
                        continue
                    stop = write_record(f, bucket.pop(0), enforce_stratum_cap=False)
                    if not bucket:
                        overflow_by_stratum.pop(stratum, None)
                    if stop or len(texts) >= TEXT_CACHE_TARGET_ITEMS:
                        break

    if len(texts) < TEXT_CACHE_MIN_ITEMS:
        raise RuntimeError(
            f"{bg_key}: only collected {len(texts)} texts; "
            f"wanted at least {TEXT_CACHE_MIN_ITEMS}. Check dataset fields/split."
        )
    print(
        f"Collected {len(texts)} texts for {bg_key}; "
        f"text cache is {bytes_written / 1024 / 1024:.2f} MB"
    )
    return TextCache(
        labels=labels[:VECTORIZE_TEXT_LIMIT],
        texts=texts[:VECTORIZE_TEXT_LIMIT],
        metas=metas[:VECTORIZE_TEXT_LIMIT],
        path=path,
        byte_size=bytes_written,
    )


def vector_cache_path(cache_dir: Path, bg_key: str, n: int) -> Path:
    return cache_dir / bg_key / (
        f"vectors_{cache_version_suffix()}_n{n}_tok{EMBED_MAX_TOKENS}_"
        f"seed{RANDOM_SEED}.npy"
    )


def vector_part_dir(cache_dir: Path, bg_key: str, n: int) -> Path:
    return cache_dir / bg_key / (
        f"vector_parts_{cache_version_suffix()}_n{n}_tok{EMBED_MAX_TOKENS}_"
        f"seed{RANDOM_SEED}"
    )


def labels_cache_path(cache_dir: Path, bg_key: str, n: int) -> Path:
    return cache_dir / bg_key / f"labels_{cache_version_suffix()}_n{n}_seed{RANDOM_SEED}.json"


def find_larger_vector_cache(cache_dir: Path, bg_key: str, n: int) -> Path | None:
    candidates: list[tuple[int, Path]] = []
    pattern = re.compile(
        rf"vectors_{cache_version_suffix()}_n(\d+)_tok{EMBED_MAX_TOKENS}_seed{RANDOM_SEED}\.npy"
    )
    for path in (cache_dir / bg_key).glob("vectors_*.npy"):
        m = pattern.fullmatch(path.name)
        if m and int(m.group(1)) >= n:
            candidates.append((int(m.group(1)), path))
    return sorted(candidates)[0][1] if candidates else None


def find_larger_part_dirs(cache_dir: Path, bg_key: str, n: int) -> list[Path]:
    candidates: list[tuple[int, Path]] = []
    pattern = re.compile(
        rf"vector_parts_{cache_version_suffix()}_n(\d+)_tok{EMBED_MAX_TOKENS}_seed{RANDOM_SEED}"
    )
    for path in (cache_dir / bg_key).glob("vector_parts_*"):
        if not path.is_dir():
            continue
        m = pattern.fullmatch(path.name)
        if m:
            candidates.append((int(m.group(1)), path))
    return [path for _, path in sorted(candidates)]


def copy_compatible_part(
    start: int, end: int, part_path: Path, source_part_dirs: list[Path]
) -> bool:
    name = f"part_{start:06d}_{end:06d}.npy"
    for source_dir in source_part_dirs:
        source = source_dir / name
        if source.exists():
            part_path.parent.mkdir(parents=True, exist_ok=True)
            shutil.copy2(source, part_path)
            print(f"Reused larger-cache vector shard: {source}")
            return True
    return False


def embed_texts(
    texts: list[str],
    embedder: QwenEmbedder,
    part_dir: Path,
    source_part_dirs: list[Path],
    progress_name: str,
) -> np.ndarray:
    part_dir.mkdir(parents=True, exist_ok=True)
    parts = []

    def embed_part(part_texts: list[str], global_start: int) -> np.ndarray:
        return embedder.embed_texts(
            part_texts,
            max_tokens=EMBED_MAX_TOKENS,
            batch_size=EMBED_BATCH_SIZE,
            sort_by_length=EMBED_SORT_BY_LENGTH,
            progress_label=progress_name,
            progress_start=global_start,
            progress_total=len(texts),
        )

    for start in range(0, len(texts), EMBED_PART_SIZE):
        end = min(start + EMBED_PART_SIZE, len(texts))
        part_path = part_dir / f"part_{start:06d}_{end:06d}.npy"
        if part_path.exists():
            part = np.load(part_path).astype(np.float32)
            print(f"Loaded cached vector shard: {progress_name} {start}/{len(texts)}")
        elif copy_compatible_part(start, end, part_path, source_part_dirs):
            part = np.load(part_path).astype(np.float32)
        else:
            part = embed_part(texts[start:end], start)
            np.save(part_path, part)
            print(f"Saved vector shard: {part_path}")
        parts.append(part)

    return np.vstack(parts).astype(np.float32)


def load_or_create_background_vectors(
    bg_key: str,
    text_cache: TextCache,
    cache_dir: Path,
    get_embedder: Callable[[], QwenEmbedder],
) -> PointData:
    n = min(len(text_cache.texts), VECTORIZE_TEXT_LIMIT)
    texts = text_cache.texts[:n]
    labels = text_cache.labels[:n]
    vec_path = vector_cache_path(cache_dir, bg_key, n)
    part_dir = vector_part_dir(cache_dir, bg_key, n)
    label_path = labels_cache_path(cache_dir, bg_key, n)

    if vec_path.exists():
        vectors = np.load(vec_path).astype(np.float32)
    else:
        larger_vec_path = find_larger_vector_cache(cache_dir, bg_key, n)
        if larger_vec_path is not None:
            print(f"Using local larger vector cache: {larger_vec_path.name}")
            vectors = np.load(larger_vec_path).astype(np.float32)[:n]
        else:
            vectors = embed_texts(
                texts,
                get_embedder(),
                part_dir,
                find_larger_part_dirs(cache_dir, bg_key, n),
                bg_key,
            )
        np.save(vec_path, vectors)

    if not label_path.exists():
        label_path.write_text(
            json.dumps(labels, ensure_ascii=False, indent=2), encoding="utf-8"
        )
    return PointData(labels=labels, vectors=l2_normalize(vectors), metas=text_cache.metas[:n])


def reducer_cache_path(cache_dir: Path, bg_key: str, n: int, diary_n: int) -> Path:
    return cache_dir / bg_key / (
        f"xy_{cache_version_suffix()}_{REDUCER}_{FIT_REDUCER_ON}_n{n}_diary{diary_n}_"
        f"neighbors{UMAP_N_NEIGHBORS}_mindist{UMAP_MIN_DIST}_"
        f"whiten{int(PCA_WHITEN)}_seed{RANDOM_SEED}.npz"
    )


def find_reducer_caches(cache_dir: Path, bg_key: str, diary_n: int) -> list[tuple[int, Path]]:
    pattern = re.compile(
        rf"xy_{cache_version_suffix()}_{REDUCER}_{FIT_REDUCER_ON}_n(\d+)_diary{diary_n}_"
        rf"neighbors{UMAP_N_NEIGHBORS}_mindist{UMAP_MIN_DIST}_"
        rf"whiten{int(PCA_WHITEN)}_seed{RANDOM_SEED}\.npz"
    )
    candidates: list[tuple[int, Path]] = []
    for path in (cache_dir / bg_key).glob("xy_*.npz"):
        m = pattern.fullmatch(path.name)
        if m:
            candidates.append((int(m.group(1)), path))
    return sorted(candidates)


def find_text_cache_for_replot(
    cache_dir: Path, bg_key: str, spec: dict[str, Any], n: int
) -> TextCache | None:
    min_chars = effective_min_text_chars(spec)
    candidates: list[tuple[int, Path]] = []
    pattern = re.compile(
        rf"texts_items(\d+)_bytes\d+_chunk{BACKGROUND_CHUNK_MIN_CHARS}-{BACKGROUND_CHUNK_MAX_CHARS}_"
        rf"min{min_chars}_seed{RANDOM_SEED}\.jsonl"
    )
    for path in (cache_dir / bg_key).glob("texts_items*_*.jsonl"):
        m = pattern.fullmatch(path.name)
        if m and int(m.group(1)) >= n:
            candidates.append((int(m.group(1)), path))
    if not candidates:
        return None
    _, path = sorted(candidates)[-1]
    return read_text_cache(path, n, spec)


def reduce_points(
    bg_key: str,
    background_vectors: np.ndarray,
    diary_vectors: np.ndarray,
    cache_dir: Path,
) -> tuple[np.ndarray, np.ndarray]:
    if FIT_REDUCER_ON not in {"background", "combined"}:
        raise ValueError(f"Unknown FIT_REDUCER_ON: {FIT_REDUCER_ON}")

    cache_path = reducer_cache_path(
        cache_dir, bg_key, len(background_vectors), len(diary_vectors)
    )
    if cache_path.exists():
        data = np.load(cache_path)
        return data["background_xy"].astype(np.float32), data["diary_xy"].astype(
            np.float32
        )

    fit_vectors = background_vectors
    if FIT_REDUCER_ON == "combined":
        fit_vectors = np.vstack([background_vectors, diary_vectors]).astype(np.float32)

    if REDUCER == "umap":
        import umap

        reducer = umap.UMAP(
            n_components=2,
            n_neighbors=UMAP_N_NEIGHBORS,
            min_dist=UMAP_MIN_DIST,
            metric="cosine",
            random_state=RANDOM_SEED,
        )
        fit_xy = reducer.fit_transform(fit_vectors).astype(np.float32)
        if FIT_REDUCER_ON == "combined":
            background_xy = fit_xy[: len(background_vectors)]
            diary_xy = fit_xy[len(background_vectors) :]
        else:
            background_xy = fit_xy
            diary_xy = reducer.transform(diary_vectors).astype(np.float32)
    elif REDUCER == "pca":
        reducer = PCA(n_components=2, whiten=PCA_WHITEN, random_state=RANDOM_SEED)
        fit_xy = reducer.fit_transform(fit_vectors).astype(np.float32)
        if FIT_REDUCER_ON == "combined":
            background_xy = fit_xy[: len(background_vectors)]
            diary_xy = fit_xy[len(background_vectors) :]
        else:
            background_xy = fit_xy
            diary_xy = reducer.transform(diary_vectors).astype(np.float32)
    else:
        raise ValueError(f"Unknown REDUCER: {REDUCER}")

    np.savez_compressed(
        cache_path, background_xy=background_xy.astype(np.float32), diary_xy=diary_xy
    )
    return background_xy, diary_xy


def replot_cached_background(
    bg_key: str,
    spec: dict[str, Any],
    diary: PointData,
    diary_text_lookup: dict[str, dict[str, Any]],
    cache_dir: Path,
    out_root: Path,
    min_items: int,
) -> tuple[str, dict[str, Any] | None]:
    candidates = [
        (n, path)
        for n, path in find_reducer_caches(cache_dir, bg_key, len(diary.labels))
        if n >= min_items
    ]
    if not candidates:
        available = find_reducer_caches(cache_dir, bg_key, len(diary.labels))
        if available:
            largest = available[-1][0]
            return (
                f"{bg_key}: skipped; largest cached background is n={largest}, "
                f"below replot_min_items={min_items}.",
                None,
            )
        return f"{bg_key}: skipped; no compatible 2D cache found.", None

    n_items, xy_path = candidates[-1]
    bg_out = out_root / slugify(bg_key)
    bg_out.mkdir(parents=True, exist_ok=True)
    data = np.load(xy_path)
    background_xy = data["background_xy"].astype(np.float32)
    diary_xy = data["diary_xy"].astype(np.float32)
    combined_xy = np.vstack([background_xy, diary_xy]).astype(np.float32)
    title = spec["title"]
    text_cache = find_text_cache_for_replot(cache_dir, bg_key, spec, n_items)
    if text_cache is None:
        background_labels = [f"background_{i + 1:05d}" for i in range(n_items)]
        background_texts = ["" for _ in range(n_items)]
        background_metas = [{} for _ in range(n_items)]
    else:
        background_labels = text_cache.labels
        background_texts = text_cache.texts
        background_metas = text_cache.metas

    t = time.perf_counter()
    plot_world(
        bg_out / "01_world_umap_or_pca",
        bg_key,
        background_xy,
        diary_xy,
        diary.labels,
        f"Diary points inside {title}",
    )
    plot_zoom(
        bg_out / "02_diary_region_zoom",
        bg_key,
        background_xy,
        diary_xy,
        "Diary region zoomed in (local detail, not global scale)",
    )
    plot_world_diary_area_scaled(
        bg_out / diary_area_scaled_output_stem(),
        bg_key,
        background_xy,
        diary_xy,
        diary.labels,
        f"Diary points inside {title}",
    )
    write_viewer_background_data(
        out_root,
        bg_key,
        title,
        background_xy,
        diary_xy,
        background_labels,
        background_texts,
        background_metas,
        diary.labels,
        diary_text_lookup,
    )
    elapsed = time.perf_counter() - t
    diary_hull_area = polygon_area(hull_boundary(diary_xy))
    background_hull_area = polygon_area(hull_boundary(background_xy))
    combined_hull_area = polygon_area(hull_boundary(combined_xy))
    scaled_diary_xy = scaled_area_points(diary_xy, DIARY_AREA_SCALE_RATIO)
    scaled_combined_xy = np.vstack([background_xy, scaled_diary_xy]).astype(np.float32)
    scaled_diary_hull_area = polygon_area(hull_boundary(scaled_diary_xy))
    scaled_combined_hull_area = polygon_area(hull_boundary(scaled_combined_xy))
    summary = "\n".join(
        [
            f"background_key: {bg_key}",
            f"background_title: {title}",
            f"replot_from_cache: {xy_path}",
            f"vectorized_background_items: {n_items}",
            f"diary_items: {len(diary.labels)}",
            f"hull_keep_ratio: {HULL_KEEP_RATIO:.4f}",
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
            f"output_image_formats: {', '.join(output_image_formats())}",
            f"viewer_data: {'yes' if text_cache is not None else 'partial_without_background_text'}",
            "note: replot-cache mode only redraws cached 2D coordinates; it does not download, vectorize, or reduce.",
        ]
    )
    (bg_out / "summary.txt").write_text(summary + "\n", encoding="utf-8")
    return summary, {"bg_key": bg_key, "title": title, "total_s": elapsed, "n_items": n_items}


def params_snapshot() -> dict[str, Any]:
    params: dict[str, Any] = {}
    for k, v in globals().items():
        if k.isupper() and isinstance(v, (str, int, float, bool, list, dict)):
            if k == "BACKGROUND_SPECS":
                params[k] = v
            else:
                params[k] = v
    return params


def enabled_background_keys() -> list[str]:
    if ENABLED_BACKGROUNDS == ["all"]:
        return list(BACKGROUND_SPECS.keys())
    unknown = [key for key in ENABLED_BACKGROUNDS if key not in BACKGROUND_SPECS]
    if unknown:
        raise ValueError(f"Unknown backgrounds in ENABLED_BACKGROUNDS: {unknown}")
    return list(ENABLED_BACKGROUNDS)


def write_background_relation_chart(out_root: Path, cache_dir: Path, bg_keys: list[str]) -> str:
    relation_out = out_root / "background_relations"
    try:
        result = build_background_relation_outputs(
            relation_out,
            cache_dir,
            bg_keys,
            verbose=False,
        )
    except Exception as exc:
        relation_out.mkdir(parents=True, exist_ok=True)
        error_text = f"{type(exc).__name__}: {exc}"
        (relation_out / "ERROR.txt").write_text(error_text + "\n", encoding="utf-8")
        return f"Background relation chart failed: {error_text}"
    skipped = result.get("skipped") or []
    suffix = f", skipped {len(skipped)}" if skipped else ""
    return f"Background relation chart: {relation_out}{suffix}"


def build_one_background(
    bg_key: str,
    spec: dict[str, Any],
    diary: PointData,
    diary_text_lookup: dict[str, dict[str, Any]],
    cache_dir: Path,
    get_embedder: Callable[[], QwenEmbedder],
    out_root: Path,
    bg_index: int = 0,
    bg_total: int = 0,
) -> tuple[str, dict[str, Any]]:
    t_bg = time.perf_counter()
    print(f"\n{'─' * 62}")
    print(f"  [{bg_index}/{bg_total}] {bg_key}  —  {spec['title']}")
    print(f"{'─' * 62}")
    bg_out = out_root / slugify(bg_key)
    bg_out.mkdir(parents=True, exist_ok=True)

    t = time.perf_counter()
    print(f"  [1/4] collecting texts ...", end="", flush=True)
    text_cache = collect_texts(bg_key, spec, cache_dir)
    print(f" {_fmt_time(time.perf_counter() - t)}"
          f"  ({len(text_cache.texts)} items, {text_cache.byte_size / 1024 / 1024:.1f} MB)")

    t = time.perf_counter()
    print(f"  [2/4] vectorizing {len(text_cache.texts)} items ...", end="", flush=True)
    background = load_or_create_background_vectors(bg_key, text_cache, cache_dir, get_embedder)
    print(f" {_fmt_time(time.perf_counter() - t)}")

    if diary.vectors.shape[1] != background.vectors.shape[1]:
        raise ValueError(
            f"Vector dimension mismatch for {bg_key}: "
            f"diary={diary.vectors.shape}, background={background.vectors.shape}"
        )

    t = time.perf_counter()
    print(f"  [3/4] reducing ({REDUCER.upper()}) ...", end="", flush=True)
    background_xy, diary_xy = reduce_points(bg_key, background.vectors, diary.vectors, cache_dir)
    combined_xy = np.vstack([background_xy, diary_xy]).astype(np.float32)
    print(f" {_fmt_time(time.perf_counter() - t)}")

    t = time.perf_counter()
    print(f"  [4/4] plotting & saving ...", end="", flush=True)
    title = spec["title"]
    plot_world(
        bg_out / "01_world_umap_or_pca",
        bg_key,
        background_xy,
        diary_xy,
        diary.labels,
        f"Diary points inside {title}",
    )
    plot_zoom(
        bg_out / "02_diary_region_zoom",
        bg_key,
        background_xy,
        diary_xy,
        "Diary region zoomed in (local detail, not global scale)",
    )
    plot_world_diary_area_scaled(
        bg_out / diary_area_scaled_output_stem(),
        bg_key,
        background_xy,
        diary_xy,
        diary.labels,
        f"Diary points inside {title}",
    )
    save_points_csv(bg_out / "points.csv", bg_key, background, diary, background_xy, diary_xy)
    write_viewer_background_data(
        out_root,
        bg_key,
        title,
        background_xy,
        diary_xy,
        background.labels,
        text_cache.texts,
        text_cache.metas,
        diary.labels,
        diary_text_lookup,
    )
    summary = summarize_metrics(
        title, bg_key, text_cache, background_xy, diary_xy, combined_xy, stratum_cap_for(bg_key)
    )
    (bg_out / "summary.txt").write_text(summary + "\n", encoding="utf-8")
    print(f" {_fmt_time(time.perf_counter() - t)}")

    total_s = time.perf_counter() - t_bg
    print(f"  ✓ {bg_key} done in {_fmt_time(total_s)}")
    return summary, {
        "bg_key": bg_key,
        "title": title,
        "total_s": total_s,
        "n_items": len(text_cache.texts),
    }


def main():
    import argparse

    parser = argparse.ArgumentParser(
        description="Place diary vectors in multiple background text spaces."
    )
    parser.add_argument(
        "--test",
        action="store_true",
        help="Quick smoke-test: 100 items/background with per-step timing and a full-run estimate.",
    )
    parser.add_argument(
        "--replot-cache",
        action="store_true",
        help="Only redraw existing 2D caches; skip download, vectorization, and reduction.",
    )
    parser.add_argument(
        "--replot-min-items",
        type=int,
        default=101,
        help="Minimum cached background point count to replot in --replot-cache mode.",
    )
    parser.add_argument(
        "--png-only",
        action="store_true",
        help="Only write PNG plots; skip SVG output.",
    )
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
    if args.content_size < 1:
        parser.error("--content-size must be a positive integer")
    if args.resolution < 1:
        parser.error("--resolution must be a positive integer")

    global VECTORIZE_TEXT_LIMIT, TEXT_CACHE_TARGET_ITEMS, TEXT_CACHE_MIN_ITEMS, OUTPUT_IMAGE_FORMATS, CANVAS_SIZE, OUTPUT_SIZE
    CANVAS_SIZE = args.content_size
    OUTPUT_SIZE = args.resolution
    import diary_MultiBackground.output as output_module

    output_module.CANVAS_SIZE = args.content_size
    output_module.OUTPUT_SIZE = args.resolution
    if args.png_only:
        import diary_BackgroundRelations.main as background_relations_module

        OUTPUT_IMAGE_FORMATS = ["png"]
        output_module.OUTPUT_IMAGE_FORMATS = ["png"]
        background_relations_module.OUTPUT_IMAGE_FORMATS = ["png"]
    test_mode = args.test

    full_limit = VECTORIZE_TEXT_LIMIT
    if test_mode:
        VECTORIZE_TEXT_LIMIT = 100
        TEXT_CACHE_TARGET_ITEMS = 200
        TEXT_CACHE_MIN_ITEMS = 20

    np.random.seed(RANDOM_SEED)
    torch.manual_seed(RANDOM_SEED)

    diary_dir = resolve_path(DIARY_VECTORS_DIR)
    model_dir = resolve_path(LOCAL_MODEL_DIR)
    output_root = resolve_path(OUTPUT_ROOT)
    cache_dir = SCRIPT_DIR / CACHE_DIR
    embedder: QwenEmbedder | None = None

    def get_embedder() -> QwenEmbedder:
        nonlocal embedder
        if embedder is None:
            embedder = QwenEmbedder(model_dir)
        return embedder

    out = next_try_dir(output_root)
    bg_keys = enabled_background_keys()
    n_bg = len(bg_keys)

    if args.replot_cache:
        mode_label = "Replot cached 2D coordinates"
    elif test_mode:
        mode_label = "Test run  (100 items/background)"
    else:
        mode_label = f"Full run  ({VECTORIZE_TEXT_LIMIT:,} items/background)"
    print(f"\n{'═' * 62}")
    print(f"  MultiBackground — {mode_label}")
    print(f"  Backgrounds : {n_bg}")
    print(f"  Output      : {out}")
    if not test_mode and not args.replot_cache:
        print(f"  Resume      : re-running picks up from cached vector shards automatically.")
    print(f"{'═' * 62}\n")

    diary = load_diary_vectors(diary_dir)
    print(f"  Loaded {len(diary.labels)} diary vectors.\n")
    diary_text_lookup = load_diary_text_lookup(resolve_path(DIARY_TEXT_JSON), diary.labels)
    if diary_text_lookup:
        print(f"  Loaded diary text details: {len(diary_text_lookup)} items.\n")
    else:
        print("  Diary text details not found; viewer will show diary labels only.\n")
    write_viewer_shell(out, [])

    if args.replot_cache:
        summaries: list[str] = []
        bg_timings: list[dict] = []
        skipped: list[str] = []
        print(f"  Replot cache mode: min cached background items = {args.replot_min_items}\n")
        for i, bg_key in enumerate(bg_keys, 1):
            print(f"  [{i}/{n_bg}] replotting {bg_key} ...", end="", flush=True)
            try:
                summary, timing = replot_cached_background(
                    bg_key,
                    BACKGROUND_SPECS[bg_key],
                    diary,
                    diary_text_lookup,
                    cache_dir,
                    out,
                    args.replot_min_items,
                )
                if timing is None:
                    skipped.append(summary)
                    print(" skipped")
                else:
                    summaries.append(summary)
                    bg_timings.append(timing)
                    write_viewer_shell(out, bg_timings)
                    print(f" {_fmt_time(timing['total_s'])} ({timing['n_items']} items)")
            except Exception as exc:
                error_text = f"{bg_key}: {type(exc).__name__}: {exc}"
                skipped.append(error_text)
                error_dir = out / slugify(bg_key)
                error_dir.mkdir(parents=True, exist_ok=True)
                (error_dir / "ERROR.txt").write_text(error_text + "\n", encoding="utf-8")
                print(f" ERROR: {error_text}")

        if skipped:
            summaries.append("Skipped backgrounds:\n" + "\n".join(f"- {e}" for e in skipped))
        (out / "summary_all.txt").write_text(
            "\n\n---\n\n".join(summaries) + "\n", encoding="utf-8"
        )
        (out / "params.json").write_text(
            json.dumps(params_snapshot(), ensure_ascii=False, indent=2), encoding="utf-8"
        )
        write_viewer_shell(out, bg_timings)
        relation_status = write_background_relation_chart(
            out,
            cache_dir,
            [timing["bg_key"] for timing in bg_timings],
        )
        with (out / "summary_all.txt").open("a", encoding="utf-8") as f:
            f.write("\n---\n\n" + relation_status + "\n")
        print(f"  {relation_status}")

        print(f"\n{'═' * 62}")
        print(f"  Replotted: {len(bg_timings)}/{n_bg} backgrounds")
        if skipped:
            print(f"  Skipped  : {len(skipped)}")
            for item in skipped:
                print(f"    - {item}")
        print(f"  Output   : {out}")
        print(f"{'═' * 62}")
        print("\n  Generated files:")
        for fp in sorted(out.rglob("*")):
            if fp.is_file():
                print(f"    {fp}")
        return

    t_wall = time.perf_counter()
    summaries: list[str] = []
    errors: list[str] = []
    bg_timings: list[dict] = []

    for i, bg_key in enumerate(bg_keys, 1):
        try:
            summary, timing = build_one_background(
                bg_key,
                BACKGROUND_SPECS[bg_key],
                diary,
                diary_text_lookup,
                cache_dir,
                get_embedder,
                out,
                bg_index=i,
                bg_total=n_bg,
            )
            summaries.append(summary)
            bg_timings.append(timing)
            write_viewer_shell(out, bg_timings)
        except Exception as exc:
            error_text = f"{bg_key}: {type(exc).__name__}: {exc}"
            errors.append(error_text)
            error_dir = out / slugify(bg_key)
            error_dir.mkdir(parents=True, exist_ok=True)
            (error_dir / "ERROR.txt").write_text(error_text + "\n", encoding="utf-8")
            print(f"  ERROR: {error_text}", file=sys.stderr)
        gc.collect()
        if embedder is not None:
            embedder.clear_cache()

    total_wall = time.perf_counter() - t_wall

    if errors:
        summaries.append(
            "Background errors:\n" + "\n".join(f"- {e}" for e in errors)
        )

    (out / "summary_all.txt").write_text(
        "\n\n---\n\n".join(summaries) + "\n", encoding="utf-8"
    )
    (out / "params.json").write_text(
        json.dumps(params_snapshot(), ensure_ascii=False, indent=2), encoding="utf-8"
    )
    write_viewer_shell(out, bg_timings)
    relation_status = write_background_relation_chart(
        out,
        cache_dir,
        [timing["bg_key"] for timing in bg_timings],
    )
    with (out / "summary_all.txt").open("a", encoding="utf-8") as f:
        f.write("\n---\n\n" + relation_status + "\n")

    n_ok = n_bg - len(errors)
    print(f"\n{'═' * 62}")
    print(f"  Completed: {n_ok}/{n_bg} backgrounds OK" + (f", {len(errors)} error(s)" if errors else ""))
    print(f"  Total wall time     : {_fmt_time(total_wall)}")
    print(f"  {relation_status}")
    if bg_timings:
        avg = total_wall / len(bg_timings)
        print(f"  Avg per background  : {_fmt_time(avg)}")

    if test_mode and bg_timings:
        scale = full_limit / 100
        est = total_wall * scale
        print()
        print(f"  ── Full-run estimate ({full_limit:,} items/background) ──────────")
        print(f"  Scale factor  : ~{scale:.0f}× (100 → {full_limit:,} items)")
        print(f"  Estimated time: {_fmt_time(est)}")
        print(f"  Note: cached backgrounds skip vectorization on re-run.")

    print(f"{'═' * 62}")

    print("\n  Generated files:")
    for fp in sorted(out.rglob("*")):
        if fp.is_file():
            print(f"    {fp}")
