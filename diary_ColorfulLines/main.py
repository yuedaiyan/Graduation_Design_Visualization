#!/usr/bin/env python3
"""Generate colorful pencil-hatching visuals from diary vectors.

Pipeline summary:
1) Re-embed diary text at char-level context granularity (Qwen3-Embedding-0.6B).
2) Fuse with existing event-level sentence vectors and diary-level vectors.
3) Render textured colorful hatching blocks inspired by colored pencil strokes.

Outputs:
  output_All/diary_ColorfulLines/try_N/<date_or_date_k>.png
  output_All/diary_ColorfulLines/svg_try_N/<date_or_date_k>.svg
"""

from __future__ import annotations

import argparse
import json
import math
import os
import re
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable
from xml.sax.saxutils import escape

ROOT_FOR_IMPORT = Path(__file__).resolve().parent.parent
if str(ROOT_FOR_IMPORT) not in sys.path:
    sys.path.insert(0, str(ROOT_FOR_IMPORT))

import numpy as np
from diary_core.embedding import QwenEmbedder
from PIL import Image, ImageDraw, ImageFilter


BASE_CONTENT_SIZE = 1000
DEFAULT_CONTENT_SIZE = 1000
DEFAULT_RESOLUTION = 2160

CANVAS_W = DEFAULT_CONTENT_SIZE
CANVAS_H = DEFAULT_CONTENT_SIZE
OUTPUT_W = DEFAULT_RESOLUTION
OUTPUT_H = DEFAULT_RESOLUTION
MARGIN = 32

BG_COLOR = (243, 237, 225)
FRAME_COLOR = (104, 92, 76)

# Earthy + blue/teal accents close to the reference image feeling.
PALETTE = np.array(
    [
        [196, 96, 58],   # brick red
        [174, 118, 55],  # ochre
        [127, 145, 73],  # olive green
        [69, 115, 59],   # deep green
        [66, 107, 152],  # muted blue
        [70, 142, 145],  # cyan teal
        [98, 84, 76],    # brown gray
        [42, 42, 42],    # near-black accent
        [171, 161, 148], # warm gray
        [215, 152, 82],  # sand orange
    ],
    dtype=np.float32,
)


@dataclass
class Entry:
    stem: str
    date: str
    content: str


@dataclass
class HatchLine:
    x0: float
    y0: float
    x1: float
    y1: float
    color: tuple[int, int, int]
    alpha: int
    width: int


@dataclass(frozen=True)
class RenderSpec:
    content_w: int
    content_h: int
    output_w: int
    output_h: int

    @property
    def raster_scale(self) -> float:
        return self.output_w / self.content_w

    def raster_point(self, x: float, y: float) -> tuple[float, float]:
        scale = self.raster_scale
        return x * scale, y * scale

    def raster_width(self, value: float) -> int:
        return max(1, int(round(value * self.raster_scale)))


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Generate colorful line blocks from diary vectors")
    parser.add_argument("--root", default=None, help="Project root. Defaults to parent of this script")
    parser.add_argument("--entries", default="diary_entries.json", help="Diary json path relative to root")
    parser.add_argument("--model-dir", default="Qwen3-Embedding-0.6B", help="Model dir relative to root")
    parser.add_argument(
        "--out-dir",
        default=None,
        help="Output base dir. Defaults to <root>/output_All/<this program folder>.",
    )
    parser.add_argument("--cache-dir", default="cache_char_vectors", help="Char vector cache dir (relative to this script dir)")
    parser.add_argument(
        "--char-vector-source",
        choices=["auto", "semantic", "cache", "model"],
        default="auto",
        help=(
            "auto/cache reuse cache if valid, then derive from diary_sentence_vectors; "
            "model explicitly rebuilds char-context embeddings."
        ),
    )
    parser.add_argument("--date", default=None, help="Optional single date stem, e.g. 2026-03-10")
    parser.add_argument("--max-chars", type=int, default=0, help="Optional cap for chars per entry (0 means all)")
    parser.add_argument("--png-only", action="store_true", help="Only render PNGs; skip svg_try_N output.")
    parser.add_argument("--seed", type=int, default=23)
    parser.add_argument("--dpi", type=int, default=220)
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
    return args


def safe_stem(date: str, count: int) -> str:
    base = re.sub(r"[^0-9A-Za-z_-]", "_", date.strip()) or "unknown_date"
    return base if count == 0 else f"{base}_{count + 1}"


def load_entries(entries_path: Path) -> list[Entry]:
    with entries_path.open("r", encoding="utf-8") as f:
        raw = json.load(f)

    filtered = [e for e in raw if e.get("date") and e.get("content")]
    counter: dict[str, int] = {}
    out: list[Entry] = []
    for e in filtered:
        date = str(e["date"]).strip()
        c = counter.get(date, 0)
        counter[date] = c + 1
        out.append(Entry(stem=safe_stem(date, c), date=date, content=str(e["content"])))
    return out


def next_try_dirs(base: Path, export_png: bool = True, export_svg: bool = True) -> tuple[Path | None, Path | None]:
    base.mkdir(parents=True, exist_ok=True)
    max_idx = 0
    for p in base.iterdir():
        if p.is_dir():
            m = re.fullmatch(r"(?:try|svg_try|SVG_try)_(\d+)", p.name)
            if m:
                max_idx = max(max_idx, int(m.group(1)))
    run_idx = max_idx + 1
    png_target = base / f"try_{run_idx}" if export_png else None
    svg_target = base / f"svg_try_{run_idx}" if export_svg else None
    if png_target is not None:
        png_target.mkdir(parents=True, exist_ok=False)
    if svg_target is not None:
        svg_target.mkdir(parents=True, exist_ok=False)
    return png_target, svg_target


def display_path(path: Path, start: Path) -> str:
    try:
        return str(path.relative_to(start))
    except ValueError:
        return str(path)


def display_entry_paths(item: dict, start: Path) -> dict:
    out = dict(item)
    out["output"] = display_path(Path(str(out["output"])), start)
    if out.get("svg_output"):
        out["svg_output"] = display_path(Path(str(out["svg_output"])), start)
    return out


def choose_batch_size(max_chars: int) -> int:
    if max_chars <= 24:
        return 128
    if max_chars <= 64:
        return 96
    if max_chars <= 128:
        return 64
    return 48


def dynamic_embed(texts: list[str], embedder: QwenEmbedder) -> np.ndarray:
    if not texts:
        return np.empty((0, 1024), dtype=np.float32)

    indexed = list(enumerate(texts))
    indexed.sort(key=lambda x: len(x[1]))
    vecs: list[np.ndarray | None] = [None] * len(texts)
    i = 0
    while i < len(indexed):
        candidate = indexed[i : i + 128]
        max_len = max(len(t) for _, t in candidate)
        bs = choose_batch_size(max_len)
        batch = indexed[i : i + bs]
        batch_texts = [t for _, t in batch]
        out = embedder.embed_texts(
            batch_texts,
            max_tokens=512,
            batch_size=bs,
        )
        for (orig_i, _), v in zip(batch, out):
            vecs[orig_i] = v
        i += len(batch)

    return np.asarray([v for v in vecs if v is not None], dtype=np.float32)


def char_context_units(content: str, radius: int = 2, max_chars: int = 0) -> tuple[list[str], list[int], list[str]]:
    chars = list(content)
    n = len(chars)
    snippets: list[str] = []
    pos: list[int] = []
    glyphs: list[str] = []
    for i, ch in enumerate(chars):
        if ch.isspace():
            continue
        lo = max(0, i - radius)
        hi = min(n, i + radius + 1)
        snippet = "".join(chars[lo:hi]).replace("\n", " ").strip()
        if not snippet:
            continue
        snippets.append(snippet)
        pos.append(i)
        glyphs.append(ch)

    if max_chars > 0 and len(snippets) > max_chars:
        sel = np.linspace(0, len(snippets) - 1, max_chars).astype(int)
        snippets = [snippets[i] for i in sel]
        pos = [pos[i] for i in sel]
        glyphs = [glyphs[i] for i in sel]

    return snippets, pos, glyphs


def robust_scale(x: np.ndarray, lo: float = 5.0, hi: float = 95.0) -> np.ndarray:
    if x.size == 0:
        return x
    a = np.percentile(x, lo)
    b = np.percentile(x, hi)
    if b - a < 1e-9:
        return np.zeros_like(x)
    y = (x - a) / (b - a)
    return np.clip(y, 0.0, 1.0)


def softsig(x: np.ndarray) -> np.ndarray:
    return 1.0 / (1.0 + np.exp(-x))


def normalized(v: np.ndarray, axis: int = -1) -> np.ndarray:
    n = np.linalg.norm(v, axis=axis, keepdims=True)
    n = np.clip(n, 1e-12, None)
    return v / n


def pca2(x: np.ndarray) -> np.ndarray:
    if x.shape[0] < 2:
        return np.zeros((x.shape[0], 2), dtype=np.float32)
    xc = x - x.mean(axis=0, keepdims=True)
    _, _, vt = np.linalg.svd(xc, full_matrices=False)
    comp = vt[:2].T
    z = xc @ comp
    return z.astype(np.float32)


def pick_palette_color(values: Iterable[float], rng: np.random.Generator, vivid: float = 1.2) -> tuple[int, int, int]:
    vals = np.asarray(list(values), dtype=np.float32)
    if vals.size == 0:
        return tuple(int(v) for v in PALETTE[0])
    vals = np.clip(vals, 0.0, 1.0)
    k = min(vals.size, PALETTE.shape[0])
    score = vals[:k] + rng.normal(0.0, 0.06, size=k).astype(np.float32)

    i1 = int(np.argmax(score))
    score2 = score.copy()
    score2[i1] = -1e9
    i2 = int(np.argmax(score2)) if k > 1 else i1

    r = float(0.68 + 0.22 * rng.random())
    base = r * PALETTE[i1] + (1.0 - r) * PALETTE[i2]

    # Simple saturation boost around gray center.
    center = float(base.mean())
    boosted = center + vivid * (base - center)
    return tuple(int(np.clip(c, 0, 255)) for c in boosted)


def draw_hatch_cluster(
    draw: ImageDraw.ImageDraw,
    rng: np.random.Generator,
    cx: float,
    cy: float,
    theta: float,
    length: float,
    width: float,
    color: tuple[int, int, int],
    alpha: int,
    collector: list[HatchLine] | None = None,
    draw_scale: float = 1.0,
) -> None:
    ux = math.cos(theta)
    uy = math.sin(theta)
    vx = -uy
    vy = ux

    hatch_count = max(4, int(width / 2.4))
    spacing = max(0.9, width / (hatch_count + 2))

    for h in range(-hatch_count // 2, hatch_count // 2 + 1):
        off = h * spacing + float(rng.normal(0.0, 0.55))
        ll = length * float(0.70 + 0.42 * rng.random())
        x0 = cx + vx * off - ux * ll * 0.5 + float(rng.normal(0.0, 0.8))
        y0 = cy + vy * off - uy * ll * 0.5 + float(rng.normal(0.0, 0.8))
        x1 = cx + vx * off + ux * ll * 0.5 + float(rng.normal(0.0, 0.8))
        y1 = cy + vy * off + uy * ll * 0.5 + float(rng.normal(0.0, 0.8))
        lw = max(1, int(round(0.9 + 1.8 * rng.random())))
        draw.line(
            [(x0 * draw_scale, y0 * draw_scale), (x1 * draw_scale, y1 * draw_scale)],
            fill=(color[0], color[1], color[2], alpha),
            width=max(1, int(round(lw * draw_scale))),
        )
        if collector is not None:
            collector.append(HatchLine(x0=x0, y0=y0, x1=x1, y1=y1, color=color, alpha=alpha, width=lw))


def rgb_hex(color: tuple[int, int, int]) -> str:
    return "#{:02x}{:02x}{:02x}".format(*color)


def svg_line_elements(lines: list[HatchLine]) -> str:
    parts: list[str] = []
    for line in lines:
        opacity = max(0.0, min(1.0, line.alpha / 255.0))
        parts.append(
            (
                f'<line x1="{line.x0:.3f}" y1="{line.y0:.3f}" '
                f'x2="{line.x1:.3f}" y2="{line.y1:.3f}" '
                f'stroke="{rgb_hex(line.color)}" stroke-opacity="{opacity:.3f}" '
                f'stroke-width="{line.width}" />'
            )
        )
    return "\n".join(parts)


def write_svg(
    svg_path: Path,
    date: str,
    layer_a: list[HatchLine],
    layer_b: list[HatchLine],
    footer_lines: list[HatchLine],
    spec: RenderSpec,
) -> None:
    title = escape(f"diary_ColorfulLines {date}")

    body = f'''<?xml version="1.0" encoding="UTF-8"?>
<svg xmlns="http://www.w3.org/2000/svg" width="{spec.output_w}" height="{spec.output_h}" viewBox="0 0 {spec.content_w} {spec.content_h}">
  <title>{title}</title>
  <defs>
    <filter id="soft-pencil" x="-4%" y="-4%" width="108%" height="108%">
      <feGaussianBlur stdDeviation="0.55" />
    </filter>
    <filter id="footer-soft-pencil" x="-4%" y="-4%" width="108%" height="108%">
      <feGaussianBlur stdDeviation="0.45" />
    </filter>
  </defs>
  <g fill="none" stroke-linecap="butt" stroke-linejoin="miter" filter="url(#soft-pencil)">
{svg_line_elements(layer_b)}
  </g>
  <g fill="none" stroke-linecap="butt" stroke-linejoin="miter">
{svg_line_elements(layer_a)}
  </g>
  <g fill="none" stroke-linecap="butt" stroke-linejoin="miter" filter="url(#footer-soft-pencil)">
{svg_line_elements(footer_lines)}
  </g>
</svg>
'''
    svg_path.parent.mkdir(parents=True, exist_ok=True)
    svg_path.write_text(body, encoding="utf-8")


def render_one(
    stem: str,
    date: str,
    glyphs: list[str],
    positions: list[int],
    char_vecs: np.ndarray,
    sentence_vecs: np.ndarray,
    window_vecs: np.ndarray,
    diary_vec: np.ndarray,
    out_path: Path,
    svg_path: Path | None,
    seed: int,
    spec: RenderSpec,
) -> dict:
    rng = np.random.default_rng(seed)
    scale = spec.raster_scale

    n = char_vecs.shape[0]
    z = pca2(char_vecs)
    zx = robust_scale(z[:, 0]) if n else np.array([])
    zy = robust_scale(z[:, 1]) if n else np.array([])

    d = normalized(diary_vec.reshape(1, -1))[0]
    c = normalized(char_vecs, axis=1)

    sim_diary = (c @ d).reshape(-1)
    sim_diary_n = robust_scale(sim_diary)

    # Event flow from sentence/window vectors.
    s_mean = normalized(sentence_vecs.mean(axis=0, keepdims=True))[0]
    w_mean = normalized(window_vecs.mean(axis=0, keepdims=True))[0]
    sim_s = robust_scale((c @ s_mean).reshape(-1))
    sim_w = robust_scale((c @ w_mean).reshape(-1))

    # Texture controls.
    tension = robust_scale(0.55 * sim_diary_n + 0.25 * sim_s + 0.20 * sim_w)
    density = robust_scale(0.45 * zx + 0.55 * (1.0 - zy))

    canvas = Image.new("RGBA", (spec.output_w, spec.output_h), BG_COLOR + (255,))

    # Slight paper grain underlay.
    grain = rng.normal(0.0, 8.5, size=(spec.output_h, spec.output_w, 1)).astype(np.float32)
    paper = np.full((spec.output_h, spec.output_w, 3), np.array(BG_COLOR, dtype=np.float32), dtype=np.float32)
    paper = np.clip(paper + grain, 0, 255).astype(np.uint8)
    paper_img = Image.fromarray(paper, mode="RGB").convert("RGBA")
    canvas = Image.blend(canvas, paper_img, alpha=0.12)

    layer_a = Image.new("RGBA", (spec.output_w, spec.output_h), (0, 0, 0, 0))
    layer_b = Image.new("RGBA", (spec.output_w, spec.output_h), (0, 0, 0, 0))
    draw_a = ImageDraw.Draw(layer_a, "RGBA")
    draw_b = ImageDraw.Draw(layer_b, "RGBA")
    svg_layer_a: list[HatchLine] = []
    svg_layer_b: list[HatchLine] = []
    svg_footer_lines: list[HatchLine] = []

    inner_w = spec.content_w - 2 * MARGIN
    inner_h = spec.content_h - 2 * MARGIN

    total_pos = max(max(positions, default=1), 1)

    for i in range(n):
        # Semantic x/y placement: mostly vertical ribbons with local distortions.
        px = float(softsig(np.array([2.1 * (zx[i] - 0.5) + 0.8 * (sim_s[i] - 0.5)]))[0])
        py = (positions[i] / total_pos) if total_pos else (i / max(n - 1, 1))
        py += 0.16 * (zy[i] - 0.5) + 0.06 * math.sin(10.0 * px + i * 0.06)
        py = py % 1.0

        cx = MARGIN + px * inner_w + float(rng.normal(0.0, 4.0))
        cy = MARGIN + py * inner_h + float(rng.normal(0.0, 4.5))

        # Orientation leans vertical, perturbed by semantic channels.
        theta = math.radians(83 + 28 * (sim_w[i] - 0.5) + 15 * (zy[i] - 0.5))

        # Block size and hatch behavior.
        base_len = 30 + 88 * density[i]
        base_wid = 10 + 36 * tension[i]
        alpha = int(110 + 120 * tension[i])

        # Palette mixing by multiple semantic factors.
        palette_scores = np.array(
            [
                sim_diary_n[i],
                sim_s[i],
                sim_w[i],
                zx[i],
                zy[i],
                1.0 - zx[i],
                1.0 - zy[i],
                density[i],
                tension[i],
                0.5 * (density[i] + tension[i]),
            ],
            dtype=np.float32,
        )
        color = pick_palette_color(palette_scores, rng, vivid=1.28)

        # Two-layer hatching for pencil buildup.
        draw_hatch_cluster(draw_a, rng, cx, cy, theta, base_len, base_wid, color, alpha, collector=svg_layer_a, draw_scale=scale)
        draw_hatch_cluster(
            draw_b,
            rng,
            cx + float(rng.normal(0.0, 1.8)),
            cy + float(rng.normal(0.0, 1.8)),
            theta + math.radians(float(rng.normal(0.0, 5.0))),
            base_len * (0.72 + 0.34 * rng.random()),
            base_wid * (0.68 + 0.28 * rng.random()),
            color,
            int(alpha * 0.72),
            collector=svg_layer_b,
            draw_scale=scale,
        )

    # Soften second layer then composite.
    layer_b = layer_b.filter(ImageFilter.GaussianBlur(radius=0.55 * scale))
    canvas = Image.alpha_composite(canvas, layer_b)
    canvas = Image.alpha_composite(canvas, layer_a)

    # Frame and footer strip, similar to reference composition.
    frame = ImageDraw.Draw(canvas, "RGBA")
    frame.rectangle(
        [6 * scale, 6 * scale, (spec.content_w - 6) * scale, (spec.content_h - 6) * scale],
        outline=FRAME_COLOR + (210,),
        width=max(1, int(round(3 * scale))),
    )
    frame.rectangle(
        [18 * scale, 18 * scale, (spec.content_w - 18) * scale, (spec.content_h - 18) * scale],
        outline=FRAME_COLOR + (180,),
        width=max(1, int(round(1 * scale))),
    )

    footer_h = int(spec.content_h * 0.18)
    y0 = spec.content_h - footer_h
    frame.rectangle([0, y0 * scale, spec.output_w, (y0 + 1) * scale], fill=FRAME_COLOR + (120,))

    # Add sparse low-density echoes in footer.
    footer_layer = Image.new("RGBA", (spec.output_w, spec.output_h), (0, 0, 0, 0))
    footer_draw = ImageDraw.Draw(footer_layer, "RGBA")
    pick = np.linspace(0, max(n - 1, 0), min(140, max(1, n // 3))).astype(int) if n else np.array([], dtype=int)
    for idx in pick:
        px = float(softsig(np.array([2.0 * (zx[idx] - 0.5)]))[0])
        cx = MARGIN + px * inner_w + float(rng.normal(0.0, 8.0))
        cy = y0 + 16 + float(rng.random()) * (footer_h - 28)
        theta = math.radians(84 + 20 * (zy[idx] - 0.5))
        color = pick_palette_color([sim_diary_n[idx], sim_s[idx], 1.0 - sim_w[idx], density[idx]], rng, vivid=1.15)
        draw_hatch_cluster(
            footer_draw,
            rng,
            cx,
            cy,
            theta,
            20 + 42 * density[idx],
            6 + 14 * tension[idx],
            color,
            110,
            collector=svg_footer_lines,
            draw_scale=scale,
        )
    footer_layer = footer_layer.filter(ImageFilter.GaussianBlur(radius=0.45 * scale))
    canvas = Image.alpha_composite(canvas, footer_layer)

    # Tiny grain on top for pencil roughness.
    arr = np.asarray(canvas.convert("RGB"), dtype=np.float32)
    top_noise = rng.normal(0.0, 4.0, size=arr.shape).astype(np.float32)
    arr = np.clip(arr + top_noise, 0, 255).astype(np.uint8)
    out = Image.fromarray(arr, mode="RGB")

    out_path.parent.mkdir(parents=True, exist_ok=True)
    out.save(out_path, quality=96)
    if svg_path is not None:
        write_svg(
            svg_path=svg_path,
            date=date,
            layer_a=svg_layer_a,
            layer_b=svg_layer_b,
            footer_lines=svg_footer_lines,
            spec=spec,
        )

    info = {
        "stem": stem,
        "date": date,
        "chars_used": int(n),
        "output": str(out_path),
        "footer_echoes": int(len(pick)),
    }
    if svg_path is not None:
        info["svg_output"] = str(svg_path)
    return info


def load_event_vectors(root: Path, stem: str) -> tuple[np.ndarray, np.ndarray, np.ndarray, dict]:
    diary_vec_file = root / "diary_vectors" / f"{stem}.npy"
    sent_dir = root / "diary_sentence_vectors" / stem
    if not diary_vec_file.exists():
        raise FileNotFoundError(f"Missing diary vector: {diary_vec_file}")
    if not sent_dir.exists():
        raise FileNotFoundError(f"Missing sentence vector dir: {sent_dir}")

    diary_vec = np.load(diary_vec_file).astype(np.float32).reshape(-1)
    sentence_vecs = np.load(sent_dir / "sentence_vectors.npy").astype(np.float32)
    window_vecs = np.load(sent_dir / "window_vectors.npy").astype(np.float32)
    with (sent_dir / "meta.json").open("r", encoding="utf-8") as f:
        meta = json.load(f)
    return diary_vec, sentence_vecs, window_vecs, meta


def nonspace_len(text: str) -> int:
    return sum(1 for ch in text if not ch.isspace())


def semantic_char_vectors(
    content: str,
    positions: list[int],
    sentence_vecs: np.ndarray,
    window_vecs: np.ndarray,
    diary_vec: np.ndarray,
    meta: dict,
) -> np.ndarray:
    """Approximate char-level vectors from existing sentence/window vectors.

    This keeps full-run generation light on 16GB machines and avoids embedding
    very long entries again. Small deterministic positional components prevent
    all characters in the same sentence from collapsing to one visual point.
    """

    if not positions:
        return np.empty((0, diary_vec.shape[0]), dtype=np.float32)

    order_by_pos: dict[int, int] = {}
    order = 0
    for i, ch in enumerate(content):
        if ch.isspace():
            continue
        order_by_pos[i] = order
        order += 1

    sentences = meta.get("sentences", [])
    sent_lengths = [max(1, nonspace_len(str(s.get("text", "")))) for s in sentences]
    if not sent_lengths:
        sent_lengths = [max(1, order)]
    sent_edges = np.cumsum(np.asarray(sent_lengths, dtype=np.int64))

    sentence_to_window: dict[int, int] = {}
    for win in meta.get("windows", []):
        win_idx = int(win.get("idx", 0))
        start = int(win.get("start_sentence_idx", 0))
        end = int(win.get("end_sentence_idx", start))
        for sent_idx in range(start, end + 1):
            sentence_to_window[sent_idx] = min(win_idx, max(0, len(window_vecs) - 1))

    d = normalized(diary_vec.reshape(1, -1))[0]
    sv = normalized(sentence_vecs, axis=1) if len(sentence_vecs) else d.reshape(1, -1)
    wv = normalized(window_vecs, axis=1) if len(window_vecs) else d.reshape(1, -1)

    dim = d.shape[0]
    out = np.empty((len(positions), dim), dtype=np.float32)
    for row, pos in enumerate(positions):
        char_order = order_by_pos.get(pos, row)
        sent_idx = int(np.searchsorted(sent_edges, char_order, side="right"))
        sent_idx = min(sent_idx, len(sv) - 1)
        win_idx = sentence_to_window.get(sent_idx, min(sent_idx, len(wv) - 1))

        vec = 0.64 * sv[sent_idx] + 0.26 * wv[win_idx] + 0.10 * d
        phase = char_order / max(order - 1, 1)
        wobble_dims = min(32, dim)
        wobble = np.sin(np.linspace(0.0, 2.0 * math.pi, wobble_dims, endpoint=False) + phase * math.tau)
        vec = vec.copy()
        vec[:wobble_dims] += (0.018 * wobble).astype(np.float32)
        out[row] = normalized(vec.reshape(1, -1))[0].astype(np.float32)

    return out


def load_or_build_char_vectors(
    cache_dir: Path,
    stem: str,
    snippets: list[str],
    content: str,
    positions: list[int],
    sentence_vecs: np.ndarray,
    window_vecs: np.ndarray,
    diary_vec: np.ndarray,
    meta: dict,
    source: str,
    embedder: QwenEmbedder | None,
) -> tuple[np.ndarray, str]:
    cache_dir.mkdir(parents=True, exist_ok=True)
    cache_file = cache_dir / f"{stem}.npz"

    if source in {"auto", "cache"} and cache_file.exists():
        try:
            data = np.load(cache_file, allow_pickle=False)
            cached_snippets = data["snippets"].astype(str)
            current_snippets = np.asarray(snippets, dtype=str)
            if len(cached_snippets) == len(snippets) and np.all(cached_snippets == current_snippets):
                return data["vectors"].astype(np.float32), "cache"
        except Exception:
            pass

    if source in {"auto", "semantic", "cache"}:
        return (
            semantic_char_vectors(
                content=content,
                positions=positions,
                sentence_vecs=sentence_vecs,
                window_vecs=window_vecs,
                diary_vec=diary_vec,
                meta=meta,
            ),
            "semantic",
        )

    if embedder is None:
        raise RuntimeError("Model char vectors requested, but the embedding model is not loaded.")

    # Deduplicate snippets to reduce model calls.
    uniq_map: dict[str, int] = {}
    uniq_texts: list[str] = []
    inv_idx: list[int] = []
    for s in snippets:
        j = uniq_map.get(s)
        if j is None:
            j = len(uniq_texts)
            uniq_map[s] = j
            uniq_texts.append(s)
        inv_idx.append(j)

    uniq_vecs = dynamic_embed(uniq_texts, embedder)
    full = uniq_vecs[np.asarray(inv_idx, dtype=np.int64)]

    np.savez_compressed(
        cache_file,
        snippets=np.asarray(snippets, dtype=str),
        vectors=full.astype(np.float32),
    )
    return full.astype(np.float32), "model"


def main() -> None:
    args = parse_args()
    spec = RenderSpec(
        content_w=args.content_size,
        content_h=args.content_size,
        output_w=args.resolution,
        output_h=args.resolution,
    )
    script_dir = Path(__file__).resolve().parent
    root = Path(args.root).expanduser().resolve() if args.root else script_dir.parent

    entries_path = root / args.entries
    model_dir = root / args.model_dir

    out_base = (
        Path(args.out_dir).expanduser().resolve()
        if args.out_dir
        else root / "output_All" / script_dir.name
    )
    out_run, svg_run = next_try_dirs(out_base, export_png=True, export_svg=not args.png_only)
    if out_run is None:
        raise RuntimeError("PNG output directory was not created.")

    cache_dir = Path(args.cache_dir)
    if not cache_dir.is_absolute():
        cache_dir = script_dir / cache_dir

    entries = load_entries(entries_path)
    if args.date:
        entries = [e for e in entries if e.stem == args.date or e.date == args.date]

    if not entries:
        raise RuntimeError("No diary entries matched the filter.")

    embedder = None
    if args.char_vector_source == "model":
        print(f"Loading model: {display_path(model_dir, root)}")
        embedder = QwenEmbedder(model_dir)
        print(f"Device: {embedder.device}")
    else:
        print(f"Char vectors: {args.char_vector_source} (reuse cache or diary_sentence_vectors)")

    summary: list[dict] = []
    total = len(entries)

    for idx, e in enumerate(entries, start=1):
        diary_vec, sentence_vecs, window_vecs, meta = load_event_vectors(root, e.stem)

        snippets, positions, glyphs = char_context_units(e.content, radius=2, max_chars=args.max_chars)
        if not snippets:
            print(f"[{idx}/{total}] {e.stem}: empty after char filtering, skipped")
            continue

        char_vecs, vector_source = load_or_build_char_vectors(
            cache_dir=cache_dir,
            stem=e.stem,
            snippets=snippets,
            content=e.content,
            positions=positions,
            sentence_vecs=sentence_vecs,
            window_vecs=window_vecs,
            diary_vec=diary_vec,
            meta=meta,
            source=args.char_vector_source,
            embedder=embedder,
        )

        out_file = out_run / f"{e.stem}.png"
        svg_file = svg_run / f"{e.stem}.svg" if svg_run is not None else None
        info = render_one(
            stem=e.stem,
            date=e.date,
            glyphs=glyphs,
            positions=positions,
            char_vecs=char_vecs,
            sentence_vecs=sentence_vecs,
            window_vecs=window_vecs,
            diary_vec=diary_vec,
            out_path=out_file,
            svg_path=svg_file,
            seed=args.seed + idx * 17,
            spec=spec,
        )
        info["vector_source"] = vector_source
        summary.append(info)
        saved_names = [out_file.name]
        if svg_file is not None:
            saved_names.append(svg_file.name)
        print(
            f"[{idx}/{total}] Saved {' + '.join(saved_names)} "
            f"(chars={info['chars_used']}, vectors={vector_source})"
        )

    with (out_run / "summary.json").open("w", encoding="utf-8") as f:
        json.dump(
            {
                "count": len(summary),
                "output_dir": display_path(out_run, root),
                "svg_output_dir": display_path(svg_run, root) if svg_run is not None else None,
                "seed": args.seed,
                "entries": [display_entry_paths(item, root) for item in summary],
            },
            f,
            ensure_ascii=False,
            indent=2,
        )

    print(f"Done. PNG output: {display_path(out_run, root)}")
    if svg_run is not None:
        print(f"Done. SVG output: {display_path(svg_run, root)}")


if __name__ == "__main__":
    main()
