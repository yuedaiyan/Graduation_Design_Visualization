#!/usr/bin/env python3
"""Render Lombardi-style referential networks from diary entries.

Each output image is built from one diary entry's ``content`` field:
nodes are entities/references, and edges connect entities that co-occur
inside the same sentence.

Outputs:
  output_All/diary_ReferentialNetwork/try_N/*.png
  output_All/diary_ReferentialNetwork/svg_try_N/*.svg
  output_All/diary_ReferentialNetwork/try_N/summary.json
"""

from __future__ import annotations

import argparse
import hashlib
import html
import json
import math
import os
import random
import re
import sys
from collections import Counter
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any


# =========================
# Parameters you may edit
# =========================

ENTRIES_FILE = "diary_entries.merged.json"
OUTPUT_ROOT = "output_All/diary_ReferentialNetwork"

SAMPLE_SIZE = 0
RANDOM_SEED = 20260516
CANVAS_SIZE_INCH = 10.0
DPI = 180
OUTPUT_PNG = True
OUTPUT_SVG = True

MIN_NODE_COUNT = 3
MAX_NODES = 36
MAX_ENTITY_CHARS = 9
MIN_ENTITY_FREQ = 1
MAX_SENTENCE_ENTITIES = 12

SPRING_K = 1.65
SPRING_ITERATIONS = 180
LAYOUT_SCALE = 1.0

BACKGROUND_COLOR = "#F4EEDC"
INK_COLOR = "#171411"
PNG_EDGE_ALPHA = 0.55
PNG_EDGE_WIDTH = 0.58
PNG_NODE_EDGE_WIDTH = 0.68
NODE_FONT_SIZE = 5.5
SVG_EDGE_STROKE_RATIO = 0.0018
SVG_NODE_STROKE_RATIO = SVG_EDGE_STROKE_RATIO * 0.9
SVG_ARROW_LENGTH_RATIO = 0.014
SVG_ARROW_WIDTH_RATIO = 0.010
SVG_EDGE_COLOR = INK_COLOR
SVG_EDGE_BLEND_COLOR = BACKGROUND_COLOR
SVG_EDGE_BLEND_ALPHA = 0.72
BASE_CONTENT_SIZE = 1000
DEFAULT_CONTENT_SIZE = 1000
DEFAULT_RESOLUTION = 2160
SVG_CANVAS_SIZE = DEFAULT_CONTENT_SIZE
PNG_OUTPUT_SIZE = DEFAULT_RESOLUTION
SVG_DATA_LIMIT = 0.56
FONT_FAMILY = [
    "Source Han Sans CN VF",
    "Source Han Sans CN",
    "Hiragino Sans GB",
    "sans-serif",
]

CATEGORY_ORDER = ("person", "place", "object", "abstract")
CATEGORY_CODE = {
    "person": "A",
    "place": "B",
    "object": "C",
    "abstract": "D",
}
CATEGORY_NAME = {
    "person": "person",
    "place": "place",
    "object": "object",
    "abstract": "abstract",
}
CATEGORY_COLOR = {
    "person": "#8A1F1D",
    "place": "#1D5D62",
    "object": "#7A5522",
    "abstract": "#5B3F86",
}


# =========================
# Program
# =========================

ROOT = Path(__file__).resolve().parents[1]
SCRIPT_DIR = Path(__file__).resolve().parent

os.environ.setdefault("MPLCONFIGDIR", str(SCRIPT_DIR / ".plot_cache" / "mplconfig"))
os.environ.setdefault("XDG_CACHE_HOME", str(SCRIPT_DIR / ".plot_cache" / "xdg_cache"))
Path(os.environ["MPLCONFIGDIR"]).mkdir(parents=True, exist_ok=True)
Path(os.environ["XDG_CACHE_HOME"]).mkdir(parents=True, exist_ok=True)

import jieba.posseg as pseg
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import networkx as nx
from matplotlib.font_manager import FontProperties
from matplotlib.patches import Circle, FancyArrowPatch


@dataclass
class DiaryEntry:
    index: int
    entry_id: str
    stem: str
    date: str
    location: str
    time_of_day: str
    content: str


@dataclass
class EntityInfo:
    label: str
    categories: set[str] = field(default_factory=set)
    frequency: int = 0
    sentence_count: int = 0


@dataclass(frozen=True)
class NodeCode:
    code: str
    category: str


@dataclass
class EdgeInfo:
    count: int = 0
    min_gap: int = 999
    directions: Counter[tuple[str, str]] = field(default_factory=Counter)


@dataclass(frozen=True)
class RenderGeometry:
    pos: dict[str, tuple[float, float]]
    radii: dict[str, float]


ABSTRACT_TERMS = {
    "爱",
    "害怕",
    "恐惧",
    "孤独",
    "自由",
    "命运",
    "意义",
    "价值",
    "秩序",
    "阶段",
    "巅峰",
    "舒适区",
    "关系",
    "记忆",
    "理想",
    "现实",
    "未来",
    "过去",
    "现在",
    "权力",
    "政治",
    "社会",
    "制度",
    "国家",
    "家庭",
    "世界",
    "人生",
    "生活",
    "时间",
    "空间",
    "稳定",
    "成功",
    "失败",
    "焦虑",
    "痛苦",
    "快乐",
    "幸福",
    "悲观",
    "希望",
    "危险",
    "安全",
    "欲望",
    "道德",
    "审美",
    "艺术",
    "哲学",
    "科学",
    "宗教",
    "教育",
    "经济",
}

PERSON_REFERENCES = {
    "我",
    "我们",
    "你",
    "你们",
    "他",
    "她",
    "他们",
    "她们",
    "自己",
    "别人",
    "人",
    "大家",
    "人们",
    "爸爸",
    "妈妈",
    "父亲",
    "母亲",
    "爷爷",
    "奶奶",
    "姥姥",
    "姥爷",
    "哥哥",
    "姐姐",
    "弟弟",
    "妹妹",
    "老师",
    "同学",
    "朋友",
    "男友",
    "女友",
}

STOPWORDS = {
    "一个",
    "一种",
    "一些",
    "这个",
    "那个",
    "这些",
    "那些",
    "什么",
    "怎么",
    "可以",
    "可能",
    "应该",
    "因为",
    "所以",
    "但是",
    "如果",
    "还是",
    "只是",
    "已经",
    "没有",
    "不是",
    "就是",
    "时候",
    "事情",
    "东西",
    "问题",
    "感觉",
    "觉得",
    "一样",
    "很多",
    "非常",
    "真的",
    "今天",
    "昨天",
    "明天",
    "进行",
    "开始",
    "发现",
    "产生",
    "出现",
    "存在",
    "成为",
    "具有",
    "对于",
    "通过",
    "由于",
    "之中",
    "左右",
    "一下",
    "一点",
    "一切",
    "各种",
    "方面",
    "时候",
    "原因",
    "结果",
    "方式",
    "过程",
    "部分",
    "程度",
    "状态",
    "能力",
    "目标",
    "方法",
    "意义上",
}

NOUN_FLAGS = ("n", "nr", "ns", "nt", "nz")
PLACE_SUFFIXES = (
    "国",
    "省",
    "市",
    "区",
    "县",
    "镇",
    "村",
    "路",
    "街",
    "园",
    "校",
    "家",
    "楼",
    "馆",
    "站",
    "处",
    "地铁",
    "小区",
    "教室",
    "学校",
    "画室",
    "医院",
    "北京",
    "上海",
    "美国",
    "中国",
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Generate Lombardi-style referential networks from diary content"
    )
    parser.add_argument("--root", default=None, help="Project root. Defaults to this repo")
    parser.add_argument("--entries", default=ENTRIES_FILE, help="Diary JSON path relative to root")
    parser.add_argument("--out-dir", default=None, help="Output base dir. Defaults to output_All/diary_ReferentialNetwork")
    parser.add_argument("--sample-size", type=int, default=SAMPLE_SIZE, help="Random sample size. Use 0 for all entries")
    parser.add_argument("--seed", type=int, default=RANDOM_SEED)
    parser.add_argument("--max-nodes", type=int, default=MAX_NODES)
    parser.add_argument("--date", action="append", default=None, help="Optional date/stem/id filter. Repeatable")
    parser.add_argument("--png", action=argparse.BooleanOptionalAction, default=OUTPUT_PNG)
    parser.add_argument("--svg", action=argparse.BooleanOptionalAction, default=OUTPUT_SVG)
    parser.add_argument("--png-only", action="store_true", help="Only write PNG files under try_N")
    parser.add_argument("--svg-only", action="store_true", help="Only write SVG files under svg_try_N")
    parser.add_argument(
        "--content-size",
        type=int,
        default=DEFAULT_CONTENT_SIZE,
        help=f"Logical SVG content coordinate size. Default: {DEFAULT_CONTENT_SIZE}.",
    )
    parser.add_argument(
        "--resolution",
        "--size",
        dest="resolution",
        type=int,
        default=DEFAULT_RESOLUTION,
        help=f"Final square PNG pixel size. SVG display size follows --content-size. Default: {DEFAULT_RESOLUTION}.",
    )
    args = parser.parse_args()
    if args.png_only and args.svg_only:
        parser.error("--png-only cannot be combined with --svg-only.")
    if args.png_only:
        args.png = True
        args.svg = False
    if args.svg_only:
        args.png = False
        args.svg = True
    if not args.png and not args.svg:
        parser.error("At least one output is required. Use --png, --svg, or --svg-only.")
    if args.content_size < 1:
        parser.error("--content-size must be a positive integer")
    if args.resolution < 1:
        parser.error("--resolution must be a positive integer")
    return args


def set_render_sizes(content_size: int, resolution: int) -> None:
    global SVG_CANVAS_SIZE, PNG_OUTPUT_SIZE
    SVG_CANVAS_SIZE = int(content_size)
    PNG_OUTPUT_SIZE = int(resolution)


def raster_scale() -> float:
    return PNG_OUTPUT_SIZE / (CANVAS_SIZE_INCH * DPI)


def raster_s(value: float) -> float:
    return value * raster_scale()


def resolve_path(path: str, base: Path) -> Path:
    p = Path(path).expanduser()
    if p.is_absolute():
        return p.resolve()
    return (base / p).resolve()


def next_try_dirs(base: Path, export_png: bool = True, export_svg: bool = True) -> tuple[Path | None, Path | None]:
    base.mkdir(parents=True, exist_ok=True)
    max_idx = 0
    for child in base.iterdir():
        if not child.is_dir():
            continue
        match = re.fullmatch(r"(?:svg_)?try_(\d+)", child.name)
        if match:
            max_idx = max(max_idx, int(match.group(1)))
    next_idx = max_idx + 1
    png_dir = base / f"try_{next_idx}" if export_png else None
    svg_dir = base / f"svg_try_{next_idx}" if export_svg else None
    if png_dir is not None:
        png_dir.mkdir(parents=True, exist_ok=False)
    if svg_dir is not None:
        svg_dir.mkdir(parents=True, exist_ok=False)
    return png_dir, svg_dir


def safe_file_part(text: str) -> str:
    cleaned = re.sub(r'[\\/:*?"<>|\s]+', "_", str(text).strip())
    cleaned = cleaned.strip("._")
    return cleaned[:80] or "blank"


def safe_stem(date: str, count: int) -> str:
    base = re.sub(r"[^0-9A-Za-z_-]", "_", date.strip()) or "unknown_date"
    return base if count == 0 else f"{base}_{count + 1}"


def load_entries(path: Path) -> list[DiaryEntry]:
    with path.open("r", encoding="utf-8") as f:
        raw = json.load(f)
    if not isinstance(raw, list):
        raise SystemExit(f"Expected a JSON array: {path}")

    entries: list[DiaryEntry] = []
    seen_dates: dict[str, int] = {}
    for idx, item in enumerate(raw):
        if not isinstance(item, dict):
            continue
        content = item.get("content")
        if content is None:
            content = ""
        date = str(item.get("date") or "").strip()
        count = seen_dates.get(date, 0)
        seen_dates[date] = count + 1
        entry_id = str(item.get("id") or idx + 1)
        entries.append(
            DiaryEntry(
                index=idx,
                entry_id=entry_id,
                stem=safe_stem(date, count),
                date=date,
                location=str(item.get("location") or "").strip(),
                time_of_day=str(item.get("time_of_day") or "").strip(),
                content=str(content),
            )
        )
    return entries

def select_entries(entries: list[DiaryEntry], requested: list[str] | None, sample_size: int, seed: int) -> list[DiaryEntry]:
    if requested:
        wanted = set(requested)
        selected = [
            entry
            for entry in entries
            if entry.date in wanted or entry.stem in wanted or entry.entry_id in wanted
        ]
        if not selected:
            raise SystemExit("No diary entries matched the requested filters")
        return selected

    if sample_size <= 0 or sample_size >= len(entries):
        return entries
    rng = random.Random(seed)
    return sorted(rng.sample(entries, sample_size), key=lambda entry: entry.index)


def split_sentences(text: str) -> list[str]:
    text = re.sub(r"[ \t\r\f\v]+", " ", text)
    pieces = re.split(r"[。！？!?；;\n]+", text)
    return [piece.strip(" ·*-\u3000\t") for piece in pieces if piece.strip(" ·*-\u3000\t")]


def normalize_entity(token: str) -> str:
    token = token.strip()
    token = re.sub(r"^[“”\"'‘’《〈（(【\[]+", "", token)
    token = re.sub(r"[“”\"'‘’》〉）)】\]，、,.。！？!?；;：:]+$", "", token)
    return token.strip()


def is_noise(token: str) -> bool:
    if not token:
        return True
    if token in STOPWORDS:
        return True
    if len(token) > MAX_ENTITY_CHARS:
        return True
    if re.fullmatch(r"[\d.]+", token):
        return True
    if re.fullmatch(r"[A-Za-z]", token):
        return True
    if re.fullmatch(r"[\W_]+", token):
        return True
    return False


def classify_token(word: str, flag: str) -> set[str]:
    categories: set[str] = set()
    if word in PERSON_REFERENCES or flag == "nr":
        categories.add("person")
    if flag == "ns" or word.endswith(PLACE_SUFFIXES):
        categories.add("place")
    if word in ABSTRACT_TERMS:
        categories.add("abstract")
    if flag.startswith(NOUN_FLAGS) and not categories:
        categories.add("object")
    return categories


def regex_entities(sentence: str) -> list[tuple[str, str]]:
    found: list[tuple[str, str]] = []
    for match in re.finditer(r"\b\d{2,5}[A-Za-z]?\b", sentence):
        found.append((match.group(0), "object"))
    for match in re.finditer(r"\d+(?:\.\d+)?\s*(?:块|元|美元|人民币|斤|米|公里|分钟|小时|天|年|岁|分)", sentence):
        found.append((match.group(0).replace(" ", ""), "object"))
    return found


def extract_sentence_entities(sentence: str) -> list[tuple[str, set[str]]]:
    entities: list[tuple[str, set[str]]] = []

    for label, category in regex_entities(sentence):
        label = normalize_entity(label)
        if not is_noise(label):
            entities.append((label, {category}))

    for word, flag in pseg.cut(sentence):
        label = normalize_entity(word)
        if is_noise(label):
            continue
        categories = classify_token(label, flag)
        if not categories:
            continue
        if flag.startswith("n") and len(label) == 1 and label not in PERSON_REFERENCES and label not in ABSTRACT_TERMS:
            continue
        entities.append((label, categories))

    dedup: dict[str, set[str]] = {}
    for label, categories in entities:
        dedup.setdefault(label, set()).update(categories)
    return list(dedup.items())[:MAX_SENTENCE_ENTITIES]


def ordered_sentence_labels(sentence: str, extracted: list[tuple[str, set[str]]]) -> list[str]:
    seen: set[str] = set()
    positions: list[tuple[int, int, str]] = []
    for order, (label, _categories) in enumerate(extracted):
        if label in seen:
            continue
        seen.add(label)
        index = sentence.find(label)
        if index < 0:
            index = len(sentence) + order
        positions.append((index, order, label))
    return [label for _index, _order, label in sorted(positions)]


def build_graph(content: str, max_nodes: int) -> tuple[nx.Graph, dict[str, EntityInfo], dict[frozenset[str], EdgeInfo]]:
    entity_info: dict[str, EntityInfo] = {}
    edge_info: dict[frozenset[str], EdgeInfo] = {}

    for sentence in split_sentences(content):
        extracted = extract_sentence_entities(sentence)
        if not extracted:
            continue
        for label, categories in extracted:
            info = entity_info.setdefault(label, EntityInfo(label=label))
            info.frequency += 1
            info.categories.update(categories)
        unique_labels = ordered_sentence_labels(sentence, extracted)
        for label in unique_labels:
            entity_info[label].sentence_count += 1
        for i, source in enumerate(unique_labels):
            for j, target in enumerate(unique_labels[i + 1 :], i + 1):
                if source == target:
                    continue
                key = frozenset((source, target))
                info = edge_info.setdefault(key, EdgeInfo())
                info.count += 1
                info.min_gap = min(info.min_gap, j - i)
                info.directions[(source, target)] += 1

    if not entity_info:
        return nx.Graph(), entity_info, edge_info

    ranked = sorted(
        entity_info.values(),
        key=lambda info: (
            info.sentence_count,
            info.frequency,
            "person" in info.categories,
            "abstract" in info.categories,
            -len(info.label),
        ),
        reverse=True,
    )
    keep = {info.label for info in ranked if info.frequency >= MIN_ENTITY_FREQ}
    if len(keep) > max_nodes:
        keep = {info.label for info in ranked[:max_nodes]}

    graph = nx.Graph()
    for label in keep:
        graph.add_node(label)

    for key, info in edge_info.items():
        if len(key) != 2:
            continue
        source, target = tuple(key)
        if source in keep and target in keep:
            graph.add_edge(source, target, edge_info=info)

    isolated = [node for node in graph.nodes if graph.degree(node) == 0]
    if len(graph.nodes) - len(isolated) >= MIN_NODE_COUNT:
        graph.remove_nodes_from(isolated)

    return graph, {label: entity_info[label] for label in graph.nodes}, edge_info


def stable_int(text: str) -> int:
    return int(hashlib.sha256(text.encode("utf-8")).hexdigest()[:8], 16)


def graph_layout(graph: nx.Graph, seed: int) -> dict[str, tuple[float, float]]:
    if len(graph.nodes) == 1:
        node = next(iter(graph.nodes))
        return {node: (0.0, 0.0)}
    pos = nx.spring_layout(
        graph,
        seed=seed % (2**32 - 1),
        k=SPRING_K / math.sqrt(max(len(graph.nodes), 1)),
        iterations=SPRING_ITERATIONS,
        scale=LAYOUT_SCALE,
    )
    return {node: (float(coords[0]), float(coords[1])) for node, coords in pos.items()}


def normalize_positions(pos: dict[str, tuple[float, float]]) -> dict[str, tuple[float, float]]:
    if not pos:
        return {}
    xs = [xy[0] for xy in pos.values()]
    ys = [xy[1] for xy in pos.values()]
    min_x, max_x = min(xs), max(xs)
    min_y, max_y = min(ys), max(ys)
    width = max(max_x - min_x, 1e-6)
    height = max(max_y - min_y, 1e-6)
    scale = 0.82 / max(width, height)
    centered = {}
    for node, (x, y) in pos.items():
        centered[node] = ((x - (min_x + max_x) / 2) * scale, (y - (min_y + max_y) / 2) * scale)
    return centered


def choose_font() -> FontProperties:
    candidates = [
        Path("~/Library/Fonts/SourceHanSansCN-Normal.otf").expanduser(),
        Path("~/Library/Fonts/SourceHanSansCN-VF-2.otf").expanduser(),
        Path("~/Library/Fonts/SourceHanSansCN-Regular.otf").expanduser(),
        Path("/System/Library/Fonts/Hiragino Sans GB.ttc"),
    ]
    for path in candidates:
        if path.exists():
            return FontProperties(fname=str(path))
    return FontProperties(family=FONT_FAMILY)


def primary_category(info: EntityInfo) -> str:
    for category in CATEGORY_ORDER:
        if category in info.categories:
            return category
    return "object"


def make_node_codes(graph: nx.Graph, entity_info: dict[str, EntityInfo]) -> dict[str, NodeCode]:
    counters = {category: 0 for category in CATEGORY_ORDER}
    code_map: dict[str, NodeCode] = {}
    ordered_nodes = sorted(
        graph.nodes,
        key=lambda node: (
            CATEGORY_ORDER.index(primary_category(entity_info[node])),
            -graph.degree(node),
            -entity_info[node].sentence_count,
            -entity_info[node].frequency,
            stable_int(node),
        ),
    )
    for node in ordered_nodes:
        category = primary_category(entity_info[node])
        counters[category] += 1
        code_map[node] = NodeCode(
            code=f"{CATEGORY_CODE[category]}{counters[category]}",
            category=category,
        )
    return code_map


def node_radius(label: str, degree: int, max_degree: int) -> float:
    degree_part = 0.005 * math.sqrt(degree / max(max_degree, 1))
    label_part = min(max(len(label), 2), MAX_ENTITY_CHARS) * 0.0031
    return 0.019 + degree_part + label_part


def radius_to_points(ax: Any, radius: float) -> float:
    start = ax.transData.transform((0.0, 0.0))
    end = ax.transData.transform((radius, 0.0))
    pixels = abs(end[0] - start[0])
    return pixels * 72.0 / ax.figure.dpi


def choose_edge_direction(source: str, target: str, info: EdgeInfo) -> tuple[str, str]:
    if not info.directions:
        return source, target
    ranked = sorted(
        info.directions.items(),
        key=lambda item: (item[1], stable_int(f"{item[0][0]}->{item[0][1]}")),
        reverse=True,
    )
    return ranked[0][0]


def edge_is_solid(info: EdgeInfo) -> bool:
    return info.count >= 2 or info.min_gap <= 1


def draw_curved_arrow(
    ax: Any,
    p1: tuple[float, float],
    p2: tuple[float, float],
    r1: float,
    r2: float,
    edge_key: str,
    solid: bool,
) -> None:
    sign = -1 if stable_int(edge_key) % 2 else 1
    rad = sign * 0.18
    linestyle = "solid" if solid else (0, (4.0, 3.0))
    patch = FancyArrowPatch(
        p1,
        p2,
        arrowstyle="-|>",
        mutation_scale=raster_s(7.0),
        shrinkA=radius_to_points(ax, r1) + 1.0,
        shrinkB=radius_to_points(ax, r2) + 1.2,
        connectionstyle=f"arc3,rad={rad}",
        facecolor=INK_COLOR,
        edgecolor=INK_COLOR,
        lw=raster_s(PNG_EDGE_WIDTH),
        alpha=PNG_EDGE_ALPHA,
        linestyle=linestyle,
        capstyle="round",
        joinstyle="round",
        zorder=1,
    )
    ax.add_patch(patch)


def make_render_geometry(
    entry: DiaryEntry,
    graph: nx.Graph,
    node_codes: dict[str, NodeCode],
    seed: int,
) -> RenderGeometry:
    if len(graph.nodes) == 0:
        return RenderGeometry(pos={}, radii={})
    layout_seed = seed + stable_int(entry.entry_id)
    pos = normalize_positions(graph_layout(graph, layout_seed))
    max_degree = max((degree for _node, degree in graph.degree()), default=1)
    radii = {
        node: node_radius(node_codes[node].code, graph.degree(node), max_degree)
        for node in graph.nodes
    }
    return RenderGeometry(pos=pos, radii=radii)


def svg_num(value: float) -> str:
    text = f"{value:.3f}".rstrip("0").rstrip(".")
    return text if text != "-0" else "0"


def svg_point(point: tuple[float, float]) -> tuple[float, float]:
    size = float(SVG_CANVAS_SIZE)
    limit = float(SVG_DATA_LIMIT)
    x, y = point
    return ((x + limit) / (limit * 2.0) * size, (limit - y) / (limit * 2.0) * size)


def svg_radius(radius: float) -> float:
    return radius / (SVG_DATA_LIMIT * 2.0) * SVG_CANVAS_SIZE


def svg_stroke_width(ratio: float) -> float:
    return max(1.0, SVG_CANVAS_SIZE * ratio)


def blend_hex_color(foreground: str, background: str, alpha: float) -> str:
    alpha = max(0.0, min(1.0, alpha))

    def channel_pair(value: str) -> tuple[int, int, int]:
        cleaned = value.strip().lstrip("#")
        if len(cleaned) != 6:
            raise ValueError(f"Expected #RRGGBB color, got {value!r}")
        return (
            int(cleaned[0:2], 16),
            int(cleaned[2:4], 16),
            int(cleaned[4:6], 16),
        )

    fg = channel_pair(foreground)
    bg = channel_pair(background)
    mixed = tuple(round(f * alpha + b * (1.0 - alpha)) for f, b in zip(fg, bg))
    return f"#{mixed[0]:02X}{mixed[1]:02X}{mixed[2]:02X}"


def svg_edge_color() -> str:
    return blend_hex_color(SVG_EDGE_COLOR, SVG_EDGE_BLEND_COLOR, SVG_EDGE_BLEND_ALPHA)


def trim_edge_points(
    p1: tuple[float, float],
    p2: tuple[float, float],
    r1: float,
    r2: float,
) -> tuple[tuple[float, float], tuple[float, float]]:
    dx = p2[0] - p1[0]
    dy = p2[1] - p1[1]
    distance = math.hypot(dx, dy)
    if distance <= 1e-9:
        return p1, p2
    ux = dx / distance
    uy = dy / distance
    start = (p1[0] + ux * (r1 + 0.004), p1[1] + uy * (r1 + 0.004))
    end = (p2[0] - ux * (r2 + 0.006), p2[1] - uy * (r2 + 0.006))
    return start, end


def svg_arrow_points(
    tip: tuple[float, float],
    tangent_from: tuple[float, float],
) -> list[tuple[float, float]]:
    dx = tip[0] - tangent_from[0]
    dy = tip[1] - tangent_from[1]
    distance = math.hypot(dx, dy)
    if distance <= 1e-9:
        return []
    ux = dx / distance
    uy = dy / distance
    length = max(7.0, SVG_CANVAS_SIZE * SVG_ARROW_LENGTH_RATIO)
    width = max(5.0, SVG_CANVAS_SIZE * SVG_ARROW_WIDTH_RATIO)
    base_x = tip[0] - ux * length
    base_y = tip[1] - uy * length
    px = -uy
    py = ux
    half_width = width / 2.0
    return [
        tip,
        (base_x + px * half_width, base_y + py * half_width),
        (base_x - px * half_width, base_y - py * half_width),
    ]


def svg_edge_shape(
    p1: tuple[float, float],
    p2: tuple[float, float],
    r1: float,
    r2: float,
    edge_key: str,
) -> tuple[str, list[tuple[float, float]]]:
    start, end = trim_edge_points(p1, p2, r1, r2)
    dx = end[0] - start[0]
    dy = end[1] - start[1]
    distance = math.hypot(dx, dy)
    if distance <= 1e-9:
        sx, sy = svg_point(start)
        return f"M {svg_num(sx)} {svg_num(sy)}", []
    sign = -1 if stable_int(edge_key) % 2 else 1
    rad = sign * 0.18
    mx = (start[0] + end[0]) / 2.0
    my = (start[1] + end[1]) / 2.0
    ctrl = (mx - dy / distance * rad * distance, my + dx / distance * rad * distance)
    sx, sy = svg_point(start)
    cx, cy = svg_point(ctrl)
    ex, ey = svg_point(end)
    arrow_points = svg_arrow_points((ex, ey), (cx, cy))
    return (
        f"M {svg_num(sx)} {svg_num(sy)} Q {svg_num(cx)} {svg_num(cy)} {svg_num(ex)} {svg_num(ey)}",
        arrow_points,
    )


def write_svg_foreground(
    svg_path: Path,
    graph: nx.Graph,
    node_codes: dict[str, NodeCode],
    geometry: RenderGeometry,
) -> None:
    canvas_size = svg_num(SVG_CANVAS_SIZE)
    edge_stroke = svg_num(svg_stroke_width(SVG_EDGE_STROKE_RATIO))
    node_stroke = svg_num(svg_stroke_width(SVG_NODE_STROKE_RATIO))
    edge_color = svg_edge_color()
    lines: list[str] = [
        f'<svg xmlns="http://www.w3.org/2000/svg" width="{canvas_size}" height="{canvas_size}" viewBox="0 0 {canvas_size} {canvas_size}">',
        '  <g fill="none" stroke-linecap="round" stroke-linejoin="round">',
    ]

    for source, target in sorted(graph.edges):
        info = graph.edges[source, target]["edge_info"]
        arrow_source, arrow_target = choose_edge_direction(source, target, info)
        path, arrow_points = svg_edge_shape(
            geometry.pos[arrow_source],
            geometry.pos[arrow_target],
            geometry.radii[arrow_source],
            geometry.radii[arrow_target],
            f"{arrow_source}|{arrow_target}",
        )
        dash = "" if edge_is_solid(info) else ' stroke-dasharray="4 3"'
        lines.append(
            f'    <path d="{path}" fill="none" stroke="{edge_color}" '
            f'stroke-width="{edge_stroke}"{dash}/>'
        )
        if arrow_points:
            points = " ".join(f"{svg_num(x)},{svg_num(y)}" for x, y in arrow_points)
            lines.append(f'    <polygon points="{points}" fill="{edge_color}"/>')

    lines.append("  </g>")
    lines.append('  <g font-family="Source Han Sans CN VF, Source Han Sans CN, Hiragino Sans GB, sans-serif" text-anchor="middle" dominant-baseline="central">')
    for node in sorted(graph.nodes, key=lambda n: graph.degree(n)):
        x, y = svg_point(geometry.pos[node])
        node_code = node_codes[node]
        color = CATEGORY_COLOR[node_code.category]
        radius = svg_radius(geometry.radii[node])
        lines.append(
            f'    <circle cx="{svg_num(x)}" cy="{svg_num(y)}" r="{svg_num(radius)}" '
            f'stroke="{color}" stroke-width="{node_stroke}" fill="none"/>'
        )
        lines.append(
            f'    <text x="{svg_num(x)}" y="{svg_num(y)}" fill="{color}" '
            f'font-size="14">{html.escape(node_code.code)}</text>'
        )
    lines.append("  </g>")
    lines.append("</svg>")
    svg_path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def draw_png_foreground(
    ax: Any,
    graph: nx.Graph,
    node_codes: dict[str, NodeCode],
    geometry: RenderGeometry,
    font: FontProperties,
) -> None:
    if len(graph.nodes) > 0:
        for source, target in sorted(graph.edges):
            info = graph.edges[source, target]["edge_info"]
            arrow_source, arrow_target = choose_edge_direction(source, target, info)
            draw_curved_arrow(
                ax,
                geometry.pos[arrow_source],
                geometry.pos[arrow_target],
                geometry.radii[arrow_source],
                geometry.radii[arrow_target],
                f"{arrow_source}|{arrow_target}",
                edge_is_solid(info),
            )

        for node in sorted(graph.nodes, key=lambda n: graph.degree(n)):
            x, y = geometry.pos[node]
            node_code = node_codes[node]
            color = CATEGORY_COLOR[node_code.category]
            radius = geometry.radii[node]
            circle = Circle(
                (x, y),
                radius=radius,
                edgecolor=color,
                facecolor="none",
                lw=raster_s(PNG_NODE_EDGE_WIDTH),
                zorder=3,
            )
            ax.add_patch(circle)
            ax.text(
                x,
                y,
                node_code.code,
                ha="center",
                va="center",
                color=color,
                fontsize=raster_s(NODE_FONT_SIZE),
                fontproperties=font,
                zorder=4,
            )


def render_entry(
    entry: DiaryEntry,
    png_dir: Path | None,
    svg_dir: Path | None,
    args: argparse.Namespace,
    font: FontProperties,
) -> dict[str, Any]:
    graph, entity_info, _edge_info = build_graph(entry.content, args.max_nodes)
    node_codes = make_node_codes(graph, entity_info)
    geometry = make_render_geometry(entry, graph, node_codes, args.seed)
    out_stem = safe_file_part(entry.stem)

    written: list[str] = []
    if args.png and png_dir is not None:
        fig, ax = plt.subplots(figsize=(PNG_OUTPUT_SIZE / DPI, PNG_OUTPUT_SIZE / DPI), dpi=DPI)
        fig.patch.set_facecolor(BACKGROUND_COLOR)
        ax.set_facecolor(BACKGROUND_COLOR)
        ax.set_xlim(-0.56, 0.56)
        ax.set_ylim(-0.56, 0.56)
        ax.set_aspect("equal")
        ax.axis("off")
        ax.set_position([0.08, 0.08, 0.84, 0.84])
        draw_png_foreground(ax, graph, node_codes, geometry, font)
        png_path = png_dir / f"{out_stem}.png"
        fig.savefig(png_path, facecolor=fig.get_facecolor())
        written.append(png_path.name)
        plt.close(fig)
    if args.svg and svg_dir is not None:
        svg_path = svg_dir / f"{out_stem}.svg"
        write_svg_foreground(svg_path, graph, node_codes, geometry)
        written.append(svg_path.name)

    category_counts = Counter()
    for info in entity_info.values():
        category_counts.update([primary_category(info)])

    code_nodes = [
        {
            "code": node_codes[node].code,
            "category": CATEGORY_NAME[node_codes[node].category],
            "degree": graph.degree(node),
        }
        for node in sorted(graph.nodes, key=lambda item: node_codes[item].code)
    ]

    return {
        "entry_id": entry.entry_id,
        "date": entry.date,
        "location": entry.location,
        "node_count": graph.number_of_nodes(),
        "edge_count": graph.number_of_edges(),
        "categories": dict(sorted(category_counts.items())),
        "nodes": code_nodes,
        "files": written,
    }


def write_summary(out_dir: Path, rows: list[dict[str, Any]], entries_path: Path, args: argparse.Namespace) -> None:
    summary = {
        "logic": "referential_network",
        "entries_file": str(entries_path),
        "sample_size": len(rows),
        "seed": args.seed,
        "max_nodes": args.max_nodes,
        "outputs": {
            "png": args.png,
            "svg": args.svg,
        },
        "rows": rows,
    }
    (out_dir / "summary.json").write_text(
        json.dumps(summary, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )


def main() -> None:
    args = parse_args()
    set_render_sizes(args.content_size, args.resolution)
    root = Path(args.root).expanduser().resolve() if args.root else ROOT
    entries_path = resolve_path(args.entries, root)
    out_base = resolve_path(args.out_dir, root) if args.out_dir else resolve_path(OUTPUT_ROOT, root)
    png_dir, svg_dir = next_try_dirs(out_base, export_png=args.png, export_svg=args.svg)
    summary_dir = png_dir if png_dir is not None else svg_dir
    if summary_dir is None:
        raise SystemExit("No output directory was created")

    entries = load_entries(entries_path)
    selected = select_entries(entries, args.date, args.sample_size, args.seed)
    font = choose_font()

    rows = []
    for number, entry in enumerate(selected, 1):
        rows.append(render_entry(entry, png_dir, svg_dir, args, font))
        print(f"[{number}/{len(selected)}] {entry.entry_id} {entry.date}: nodes={rows[-1]['node_count']} edges={rows[-1]['edge_count']}")

    write_summary(summary_dir, rows, entries_path, args)
    outputs = []
    if png_dir is not None:
        outputs.append(f"PNG: {png_dir}")
    if svg_dir is not None:
        outputs.append(f"SVG: {svg_dir}")
    print(f"Output: {'; '.join(outputs)}")


if __name__ == "__main__":
    main()
