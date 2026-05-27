#!/usr/bin/env python3
"""Render keyword density maps from diary text.

Default behavior:
1) Read the top N keywords from diary_entries.merged.sentence_counts.json.
2) Simulate square-ish notebook pages: 56 chars per line, 28 lines per page.
3) Overlay every notebook page onto one square canvas.
4) Generate one PNG per keyword.

Outputs:
  output_All/diary_SingleCharacterDensityMap/try_N/<rank>_<keyword>.png
  output_All/diary_SingleCharacterDensityMap/svg_try_N/<rank>_<keyword>.svg
  output_All/diary_SingleCharacterDensityMap/try_N/summary.json
"""

from __future__ import annotations

import argparse
import html
import json
import math
import re
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from PIL import Image, ImageDraw, ImageFilter, ImageFont

# =========================
# Parameters you may edit
# =========================

KEYWORD_COUNTS_FILE = "diary_entries.merged.sentence_counts.json"
TOP_KEYWORDS = 100

SKIP_KEYWORD_RANKS = {16, 43, 92}

NOTEBOOK_CHARS_PER_LINE = 56
NOTEBOOK_LINES_PER_PAGE = 28

FONT_SIZE = 18
CHAR_STEP_PX = 18.0
LINE_HEIGHT_PX = 36.0  # Character height + one character-height line gap.
MARGIN_PX = 18

BACKGROUND_COLOR = "#E8DFD3"
TEXT_COLOR = "#171717"

FONT_FAMILY = "Source Han Sans CN VF"
FONT_PATH = "/Users/yuedaiyan/Library/Fonts/SourceHanSansCN-VF-2.otf"
BLUR_THRESHOLD = 8
MAX_BLUR_PX = 3.2
MAX_STROKE_PX = 10
MIN_ALPHA = 255
OUTPUT_SCALE = 2.0
LAYER_SPREAD_PER_COUNT_PX = 0.16
MAX_LAYER_SPREAD_PX = 9.0
OUTPUT_PNG = True
OUTPUT_SVG = True


# =========================
# Program
# =========================


@dataclass
class Entry:
    index: int
    stem: str
    date: str
    location: str
    time_of_day: str
    content: str
    first_page: int = 0
    page_count: int = 1


@dataclass(frozen=True)
class NotebookPosition:
    page: int
    row: int
    col: int


@dataclass
class RenderItem:
    char: str
    keyword: str
    entry_index: int
    entry_stem: str
    page: int
    row: int
    col: int
    x: float
    y: float


@dataclass
class AggregatedGlyph:
    char: str
    keyword: str
    row: int
    col: int
    x: float
    y: float
    count: int
    density: float
    blur: float
    stroke_width: int


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Generate one PNG density map per keyword from diary text"
    )
    parser.add_argument(
        "--root", default=None, help="Project root. Defaults to parent of this script"
    )
    parser.add_argument(
        "--entries",
        default="diary_entries.json",
        help="Diary JSON path relative to root",
    )
    parser.add_argument(
        "--keyword-counts",
        default=KEYWORD_COUNTS_FILE,
        help="Keyword frequency JSON path relative to root",
    )
    parser.add_argument(
        "--out-dir",
        default=None,
        help="Output base dir. Defaults to <root>/output_All/diary_SingleCharacterDensityMap",
    )
    parser.add_argument("--top-keywords", type=int, default=TOP_KEYWORDS)
    parser.add_argument(
        "--date",
        action="append",
        default=None,
        help="Optional date/stem filter. Repeatable",
    )
    parser.add_argument("--chars-per-line", type=int, default=NOTEBOOK_CHARS_PER_LINE)
    parser.add_argument("--lines-per-page", type=int, default=NOTEBOOK_LINES_PER_PAGE)
    parser.add_argument("--font-size", type=int, default=FONT_SIZE)
    parser.add_argument("--font-family", default=FONT_FAMILY)
    parser.add_argument("--font-path", default=FONT_PATH)
    parser.add_argument("--char-step", type=float, default=CHAR_STEP_PX)
    parser.add_argument("--line-height", type=float, default=LINE_HEIGHT_PX)
    parser.add_argument("--margin", type=int, default=MARGIN_PX)
    parser.add_argument("--background", default=BACKGROUND_COLOR)
    parser.add_argument("--text-color", default=TEXT_COLOR)
    parser.add_argument("--blur-threshold", type=int, default=BLUR_THRESHOLD)
    parser.add_argument("--max-blur", type=float, default=MAX_BLUR_PX)
    parser.add_argument("--max-stroke", type=int, default=MAX_STROKE_PX)
    parser.add_argument("--min-alpha", type=int, default=MIN_ALPHA)
    parser.add_argument("--output-scale", type=float, default=OUTPUT_SCALE)
    parser.add_argument("--layer-spread", type=float, default=LAYER_SPREAD_PER_COUNT_PX)
    parser.add_argument("--max-layer-spread", type=float, default=MAX_LAYER_SPREAD_PX)
    parser.add_argument(
        "--png", action=argparse.BooleanOptionalAction, default=OUTPUT_PNG
    )
    parser.add_argument(
        "--svg", action=argparse.BooleanOptionalAction, default=OUTPUT_SVG
    )
    parser.add_argument(
        "--svg-only", action="store_true", help="Only write SVG files under svg_try_N"
    )
    parser.add_argument(
        "--skip-rank",
        "--skip-entry",
        type=int,
        action="append",
        dest="skip_rank",
        default=None,
        help="Keyword source rank to skip. Repeatable; defaults are set in SKIP_KEYWORD_RANKS.",
    )
    parser.add_argument("--skip-empty", action="store_true")
    parser.add_argument("--no-summary", action="store_true")
    args = parser.parse_args()
    if args.svg_only:
        args.png = False
        args.svg = True
    if not args.png and not args.svg:
        parser.error(
            "At least one output is required. Use --png, --svg, or --svg-only."
        )
    return args


def safe_stem(date: str, count: int) -> str:
    base = re.sub(r"[^0-9A-Za-z_-]", "_", date.strip()) or "unknown_date"
    return base if count == 0 else f"{base}_{count + 1}"


def safe_file_part(text: str) -> str:
    cleaned = re.sub(r'[\\/:*?"<>|\s]+', "_", text.strip())
    cleaned = cleaned.strip("._")
    return cleaned or "blank"


def resolve_path(path: str, base: Path) -> Path:
    p = Path(path).expanduser()
    if p.is_absolute():
        return p.resolve()
    return (base / p).resolve()


def next_try_dirs(
    base: Path, export_png: bool = True, export_svg: bool = True
) -> tuple[Path | None, Path | None]:
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


def load_entries(path: Path) -> list[Entry]:
    with path.open("r", encoding="utf-8") as f:
        raw = json.load(f)
    if not isinstance(raw, list):
        raise SystemExit(f"Expected a JSON array: {path}")

    entries: list[Entry] = []
    seen_dates: dict[str, int] = {}
    for idx, item in enumerate(raw):
        if not isinstance(item, dict):
            continue
        date = str(item.get("date") or "").strip()
        content = item.get("content")
        if not date or content is None:
            continue
        count = seen_dates.get(date, 0)
        seen_dates[date] = count + 1
        entries.append(
            Entry(
                index=idx,
                stem=safe_stem(date, count),
                date=date,
                location=str(item.get("location") or ""),
                time_of_day=str(item.get("time_of_day") or ""),
                content=str(content),
            )
        )
    return entries


def select_entries(
    entries: list[Entry], requested_dates: list[str] | None
) -> list[Entry]:
    if not requested_dates:
        return entries
    requested = set(requested_dates)
    selected = [
        entry for entry in entries if entry.date in requested or entry.stem in requested
    ]
    if not selected:
        raise SystemExit("No diary entries matched the requested filters")
    return selected


def skip_keyword_ranks(args: argparse.Namespace) -> set[int]:
    skipped = set(SKIP_KEYWORD_RANKS)
    if args.skip_rank:
        skipped.update(args.skip_rank)
    return skipped


def load_top_keywords(
    path: Path, limit: int, skipped_ranks: set[int]
) -> list[tuple[int, str, int]]:
    with path.open("r", encoding="utf-8") as f:
        raw = json.load(f)
    if not isinstance(raw, dict):
        raise SystemExit(f"Expected a keyword-count dictionary: {path}")

    keywords: list[tuple[int, str, int]] = []
    for source_rank, (key, value) in enumerate(raw.items(), start=1):
        if source_rank in skipped_ranks:
            continue
        keyword = str(key)
        if not keyword:
            continue
        try:
            count = int(value)
        except (TypeError, ValueError):
            count = 0
        keywords.append((source_rank, keyword, count))
        if len(keywords) >= limit:
            break
    if not keywords:
        raise SystemExit(f"No keywords found in {path}")
    return keywords


def notebook_positions(
    content: str, chars_per_line: int, lines_per_page: int
) -> tuple[dict[int, NotebookPosition], int]:
    positions: dict[int, NotebookPosition] = {}
    page = 0
    row = 0
    col = 0

    for idx, ch in enumerate(content):
        if ch == "\n":
            row += 1
            col = 0
            if row >= lines_per_page:
                page += 1
                row = 0
            continue

        positions[idx] = NotebookPosition(page=page, row=row, col=col)
        col += 1
        if col >= chars_per_line:
            row += 1
            col = 0
            if row >= lines_per_page:
                page += 1
                row = 0

    return positions, page + 1


def assign_notebook_pages(
    entries: list[Entry], chars_per_line: int, lines_per_page: int
) -> dict[str, dict[int, NotebookPosition]]:
    by_stem: dict[str, dict[int, NotebookPosition]] = {}
    next_page = 0
    for entry in entries:
        positions, page_count = notebook_positions(
            entry.content,
            chars_per_line=chars_per_line,
            lines_per_page=lines_per_page,
        )
        entry.first_page = next_page
        entry.page_count = page_count
        by_stem[entry.stem] = positions
        next_page += page_count
    return by_stem


def page_size(args: argparse.Namespace) -> tuple[float, float]:
    width = args.margin * 2 + args.chars_per_line * args.char_step
    height = args.margin * 2 + args.lines_per_page * args.line_height
    return width, height


def canvas_size(args: argparse.Namespace) -> tuple[int, int]:
    page_w, page_h = page_size(args)
    side = math.ceil(max(page_w, page_h))
    return side, side


def page_origin(args: argparse.Namespace) -> tuple[float, float]:
    page_w, page_h = page_size(args)
    canvas_w, canvas_h = canvas_size(args)
    return (canvas_w - page_w) / 2, (canvas_h - page_h) / 2


def global_xy(
    global_page: int, row: int, col: int, args: argparse.Namespace
) -> tuple[float, float]:
    del global_page
    origin_x, origin_y = page_origin(args)
    return (
        origin_x + args.margin + col * args.char_step,
        origin_y + args.margin + row * args.line_height,
    )


def density_for_count(count: int, max_count: int) -> float:
    if max_count <= 0:
        return 0.0
    return max(0.0, min(1.0, count / max_count))


def blur_for_density(count: int, density: float, args: argparse.Namespace) -> float:
    threshold = max(1, args.blur_threshold)
    if count <= threshold:
        return 0.0
    return round(min(args.max_blur, max(0.25, density * args.max_blur)), 2)


def stroke_for_density(count: int, density: float, args: argparse.Namespace) -> int:
    if count <= 1:
        return 0
    stroke = round(density * args.max_stroke)
    return max(1, min(args.max_stroke, stroke))


def aggregate_items(
    keyword: str, items: list[RenderItem], args: argparse.Namespace
) -> list[AggregatedGlyph]:
    buckets: dict[tuple[str, int, int], tuple[int, float, float]] = {}
    for item in items:
        key = (item.char, item.row, item.col)
        count, x, y = buckets.get(key, (0, item.x, item.y))
        buckets[key] = (count + 1, x, y)

    max_count = max((count for count, _, _ in buckets.values()), default=0)
    glyphs: list[AggregatedGlyph] = []
    for (char, row, col), (count, x, y) in buckets.items():
        density = density_for_count(count, max_count)
        glyphs.append(
            AggregatedGlyph(
                char=char,
                keyword=keyword,
                row=row,
                col=col,
                x=x,
                y=y,
                count=count,
                density=density,
                blur=blur_for_density(count, density, args),
                stroke_width=stroke_for_density(count, density, args),
            )
        )
    glyphs.sort(key=lambda item: (item.row, item.col, item.char))
    return glyphs


def load_font(args: argparse.Namespace) -> ImageFont.FreeTypeFont:
    path = Path(args.font_path).expanduser()
    if not path.exists():
        raise SystemExit(f"Font file not found: {path}")
    try:
        return ImageFont.truetype(str(path), round(args.font_size * args.output_scale))
    except OSError as exc:
        raise SystemExit(f"Failed to load font: {path}") from exc


def text_alpha(density: float, args: argparse.Namespace) -> int:
    del density
    return max(0, min(255, round(args.min_alpha)))


def hex_to_rgb(color: str) -> tuple[int, int, int]:
    value = color.strip().lstrip("#")
    if len(value) != 6:
        raise SystemExit(f"Expected #RRGGBB color, got: {color}")
    return tuple(int(value[i : i + 2], 16) for i in (0, 2, 4))


def draw_offsets(
    glyph: AggregatedGlyph, args: argparse.Namespace
) -> list[tuple[float, float]]:
    if glyph.count <= 1:
        return [(0.0, 0.0)]

    max_spread = min(args.max_layer_spread, (glyph.count - 1) * args.layer_spread)
    offsets = [(0.0, 0.0)]
    angle_seed = (glyph.row * 37 + glyph.col * 101 + ord(glyph.char[0])) % 360
    angle_base = math.radians(angle_seed)
    golden_angle = math.pi * (3.0 - math.sqrt(5.0))
    for idx in range(1, glyph.count):
        radius = max_spread * math.sqrt(idx / max(1, glyph.count - 1))
        angle = angle_base + idx * golden_angle
        offsets.append((math.cos(angle) * radius, math.sin(angle) * radius * 0.82))
    return offsets


def draw_layered_text(
    draw: ImageDraw.ImageDraw,
    glyph: AggregatedGlyph,
    args: argparse.Namespace,
    font: ImageFont.FreeTypeFont,
    fill: tuple[int, int, int, int],
    stroke_width: int,
) -> None:
    scale = args.output_scale
    for dx, dy in draw_offsets(glyph, args):
        draw.text(
            ((glyph.x + dx) * scale, (glyph.y + dy) * scale),
            glyph.char,
            fill=fill,
            font=font,
            stroke_width=stroke_width,
            stroke_fill=fill,
        )


def save_png(
    path: Path,
    size: tuple[int, int],
    glyphs: list[AggregatedGlyph],
    args: argparse.Namespace,
    font: ImageFont.FreeTypeFont,
) -> None:
    scale = args.output_scale
    output_size = (round(size[0] * scale), round(size[1] * scale))
    image = Image.new("RGB", output_size, args.background).convert("RGBA")
    text_rgb = hex_to_rgb(args.text_color)

    sharp_layer = Image.new("RGBA", output_size, (0, 0, 0, 0))
    sharp_draw = ImageDraw.Draw(sharp_layer)
    blurred_layers: dict[float, Image.Image] = {}

    for glyph in glyphs:
        fill = (*text_rgb, text_alpha(glyph.density, args))
        stroke_width = round(glyph.stroke_width * scale)
        if glyph.blur <= 0:
            draw_layered_text(sharp_draw, glyph, args, font, fill, stroke_width)
            continue

        blur = round(glyph.blur * scale, 2)
        layer = blurred_layers.get(blur)
        if layer is None:
            layer = Image.new("RGBA", output_size, (0, 0, 0, 0))
            blurred_layers[blur] = layer
        draw_layered_text(ImageDraw.Draw(layer), glyph, args, font, fill, stroke_width)
        draw_layered_text(sharp_draw, glyph, args, font, fill, stroke_width)

    for blur, layer in sorted(blurred_layers.items()):
        image.alpha_composite(layer.filter(ImageFilter.GaussianBlur(radius=blur)))
    image.alpha_composite(sharp_layer)
    image.convert("RGB").save(path, quality=96)


def svg_num(value: float) -> str:
    text = f"{value:.3f}".rstrip("0").rstrip(".")
    return text if text != "-0" else "0"


def svg_filter_id(blur: float) -> str:
    return f"blur_{str(blur).replace('.', '_')}"


def svg_text_attributes(
    x: float,
    y: float,
    fill: str,
    alpha: int,
    stroke_width: int,
    args: argparse.Namespace,
    extra: str = "",
) -> str:
    opacity = max(0.0, min(1.0, alpha / 255.0))
    attrs = [
        f'x="{svg_num(x)}"',
        f'y="{svg_num(y)}"',
        f'fill="{fill}"',
        f'fill-opacity="{svg_num(opacity)}"',
        f'font-family="{html.escape(args.font_family)}"',
        f'font-size="{svg_num(args.font_size)}"',
    ]
    if stroke_width > 0:
        attrs.extend(
            [
                f'stroke="{fill}"',
                f'stroke-width="{svg_num(stroke_width)}"',
                f'stroke-opacity="{svg_num(opacity)}"',
                'paint-order="stroke fill"',
                'stroke-linejoin="round"',
            ]
        )
    if extra:
        attrs.append(extra)
    return " ".join(attrs)


def svg_text_element(
    glyph: AggregatedGlyph,
    dx: float,
    dy: float,
    args: argparse.Namespace,
    fill: str,
    alpha: int,
    stroke_width: int,
    extra: str = "",
) -> str:
    attrs = svg_text_attributes(
        glyph.x + dx,
        glyph.y + dy,
        fill,
        alpha,
        stroke_width,
        args,
        extra=extra,
    )
    return f"    <text {attrs}>{html.escape(glyph.char)}</text>"


def save_svg(
    path: Path,
    size: tuple[int, int],
    glyphs: list[AggregatedGlyph],
    args: argparse.Namespace,
) -> None:
    output_size = (
        round(size[0] * args.output_scale),
        round(size[1] * args.output_scale),
    )
    text_rgb = hex_to_rgb(args.text_color)
    fill = f"#{text_rgb[0]:02x}{text_rgb[1]:02x}{text_rgb[2]:02x}"
    blur_values = sorted({glyph.blur for glyph in glyphs if glyph.blur > 0})

    lines = [
        f'<svg xmlns="http://www.w3.org/2000/svg" width="{output_size[0]}" height="{output_size[1]}" viewBox="0 0 {svg_num(size[0])} {svg_num(size[1])}">',
    ]
    if blur_values:
        lines.append("  <defs>")
        for blur in blur_values:
            lines.append(
                f'    <filter id="{svg_filter_id(blur)}" x="-50%" y="-50%" width="200%" height="200%">'
            )
            lines.append(f'      <feGaussianBlur stdDeviation="{svg_num(blur)}"/>')
            lines.append("    </filter>")
        lines.append("  </defs>")
    lines.append('  <g dominant-baseline="text-before-edge">')

    for glyph in glyphs:
        alpha = text_alpha(glyph.density, args)
        stroke_width = glyph.stroke_width
        offsets = draw_offsets(glyph, args)
        if glyph.blur > 0:
            blur_attr = f'filter="url(#{svg_filter_id(glyph.blur)})"'
            for dx, dy in offsets:
                lines.append(
                    svg_text_element(
                        glyph,
                        dx,
                        dy,
                        args,
                        fill,
                        alpha,
                        stroke_width,
                        extra=blur_attr,
                    )
                )
        for dx, dy in offsets:
            lines.append(
                svg_text_element(glyph, dx, dy, args, fill, alpha, stroke_width)
            )

    lines.append("  </g>")
    lines.append("</svg>")
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def render_items_for_keyword(
    entries: list[Entry],
    entry_positions: dict[str, dict[int, NotebookPosition]],
    keyword: str,
    args: argparse.Namespace,
) -> list[RenderItem]:
    items: list[RenderItem] = []
    if not keyword:
        return items

    for entry in entries:
        positions = entry_positions[entry.stem]
        start = 0
        while True:
            idx = entry.content.find(keyword, start)
            if idx < 0:
                break
            for offset, ch in enumerate(keyword):
                pos = positions.get(idx + offset)
                if pos is None:
                    continue
                global_page = entry.first_page + pos.page
                x, y = global_xy(global_page, pos.row, pos.col, args)
                items.append(
                    RenderItem(
                        char=ch,
                        keyword=keyword,
                        entry_index=entry.index,
                        entry_stem=entry.stem,
                        page=global_page,
                        row=pos.row,
                        col=pos.col,
                        x=x,
                        y=y,
                    )
                )
            start = idx + 1
    return items


def main() -> None:
    args = parse_args()
    script_dir = Path(__file__).resolve().parent
    root = Path(args.root).expanduser().resolve() if args.root else script_dir.parent
    entries_path = resolve_path(args.entries, root)
    keyword_counts_path = resolve_path(args.keyword_counts, root)
    out_base = (
        Path(args.out_dir).expanduser().resolve()
        if args.out_dir
        else root / "output_All" / "diary_SingleCharacterDensityMap"
    )
    png_dir, svg_dir = next_try_dirs(out_base, export_png=args.png, export_svg=args.svg)
    summary_dir = png_dir if png_dir is not None else svg_dir
    if summary_dir is None:
        raise SystemExit("No output directory was created")

    font = load_font(args)
    entries = select_entries(load_entries(entries_path), args.date)
    if not entries:
        raise SystemExit("No diary entries matched the input filters")
    skipped_keyword_ranks = skip_keyword_ranks(args)
    keywords = load_top_keywords(
        keyword_counts_path,
        limit=args.top_keywords,
        skipped_ranks=skipped_keyword_ranks,
    )
    entry_positions = assign_notebook_pages(
        entries,
        chars_per_line=args.chars_per_line,
        lines_per_page=args.lines_per_page,
    )
    total_pages = sum(entry.page_count for entry in entries)
    logical_size = canvas_size(args)
    output_size = (
        round(logical_size[0] * args.output_scale),
        round(logical_size[1] * args.output_scale),
    )

    outputs = []
    if png_dir is not None:
        outputs.append(f"PNG: {png_dir}")
    if svg_dir is not None:
        outputs.append(f"SVG: {svg_dir}")
    print(f"输出目录: {'; '.join(outputs)}")
    print(f"关键词: 生成 {len(keywords)} 个 from {keyword_counts_path}")
    if skipped_keyword_ranks:
        skipped_text = ", ".join(str(rank) for rank in sorted(skipped_keyword_ranks))
        print(f"跳过关键词序号: {skipped_text}")
    print(
        f"日记: {len(entries)} 篇, notebook pages={total_pages}, "
        f"logical canvas={logical_size[0]}x{logical_size[1]}, "
        f"output canvas={output_size[0]}x{output_size[1]}"
    )
    print(
        f"版面: {args.chars_per_line} 字/行, {args.lines_per_page} 行/页, "
        "所有页面叠加到一张方形画布"
    )
    print(f"字体: {args.font_family} ({args.font_path})")

    summary_items: list[dict[str, Any]] = []
    for generated_rank, (source_rank, keyword, keyword_count) in enumerate(
        keywords, start=1
    ):
        items = render_items_for_keyword(entries, entry_positions, keyword, args)
        glyphs = aggregate_items(keyword, items, args)
        png_filename = f"{source_rank:03d}_{safe_file_part(keyword)}.png"
        svg_filename = f"{source_rank:03d}_{safe_file_part(keyword)}.svg"
        if args.skip_empty and not items:
            png_filename = None
            svg_filename = None
        else:
            if png_dir is not None:
                save_png(png_dir / png_filename, logical_size, glyphs, args, font)
            if svg_dir is not None:
                save_svg(svg_dir / svg_filename, logical_size, glyphs, args)
        summary_items.append(
            {
                "rank": source_rank,
                "generated_rank": generated_rank,
                "keyword": keyword,
                "keyword_count": keyword_count,
                "rendered_glyphs": len(items),
                "aggregated_glyphs": len(glyphs),
                "max_overlap": max((glyph.count for glyph in glyphs), default=0),
                "blurred_glyphs": sum(1 for glyph in glyphs if glyph.blur > 0),
                "thickened_glyphs": sum(
                    1 for glyph in glyphs if glyph.stroke_width > 0
                ),
                "density_levels": len({glyph.count for glyph in glyphs}),
                "stroke_levels": len({glyph.stroke_width for glyph in glyphs}),
                "file": png_filename if png_dir is not None else svg_filename,
                "png_file": png_filename if png_dir is not None else None,
                "svg_file": svg_filename if svg_dir is not None else None,
            }
        )
        print(
            f"  {source_rank:03d} {keyword}: count={keyword_count} "
            f"glyphs={len(items)} cells={len(glyphs)} "
            f"blurred={summary_items[-1]['blurred_glyphs']} "
            f"thickened={summary_items[-1]['thickened_glyphs']}"
        )

    if not args.no_summary:
        summary = {
            "try_dir": str(summary_dir),
            "png_dir": str(png_dir) if png_dir is not None else None,
            "svg_dir": str(svg_dir) if svg_dir is not None else None,
            "source": {
                "entries": str(entries_path),
                "keyword_counts": str(keyword_counts_path),
            },
            "entries": len(entries),
            "requested_keywords": args.top_keywords,
            "generated_keywords": len(keywords),
            "skipped_keyword_ranks": sorted(skipped_keyword_ranks),
            "total_notebook_pages": total_pages,
            "canvas": list(output_size),
            "logical_canvas": list(logical_size),
            "layout": {
                "chars_per_line": args.chars_per_line,
                "lines_per_page": args.lines_per_page,
                "square_canvas": True,
                "page_overlay": True,
                "font_size": args.font_size,
                "font_family": args.font_family,
                "font_path": args.font_path,
                "char_step": args.char_step,
                "line_height": args.line_height,
                "margin": args.margin,
                "background": args.background,
                "text_color": args.text_color,
                "blur_threshold": args.blur_threshold,
                "max_blur": args.max_blur,
                "max_stroke": args.max_stroke,
                "min_alpha": args.min_alpha,
                "output_scale": args.output_scale,
                "layer_spread": args.layer_spread,
                "max_layer_spread": args.max_layer_spread,
                "outputs": {
                    "png": args.png,
                    "svg": args.svg,
                },
            },
            "items": summary_items,
        }
        with (summary_dir / "summary.json").open("w", encoding="utf-8") as f:
            json.dump(summary, f, ensure_ascii=False, indent=2)

    png_count = sum(1 for item in summary_items if item["png_file"])
    svg_count = sum(1 for item in summary_items if item["svg_file"])
    parts = []
    if png_dir is not None:
        parts.append(f"{png_count} 个 PNG")
    if svg_dir is not None:
        parts.append(f"{svg_count} 个 SVG")
    print(f"完成: {', '.join(parts)}")


if __name__ == "__main__":
    main()
