import argparse
import colorsys
import json
import re
from dataclasses import dataclass
from pathlib import Path
from xml.sax.saxutils import escape

import numpy as np
from PIL import Image, ImageDraw, ImageFilter


BASE_CONTENT_SIZE = 1000
DEFAULT_CONTENT_SIZE = 1000
DEFAULT_RESOLUTION = 2160

CANVAS_SIZE = DEFAULT_CONTENT_SIZE
OUTPUT_SIZE = DEFAULT_RESOLUTION
HIGH_CANVAS_SIZE = 5000
LONG_DIARY_SENTENCE_THRESHOLD = 120
SENTENCE_GLOW_BATCH_SIZE = 10
PUNCTUATION = "，,。.!！？?；;：:、\"“”‘’（）()—-…"
WARM_HINTS = ("爱", "喜欢", "拥抱", "希望", "阳光", "笑", "温", "暖", "快乐", "亲")
COOL_HINTS = ("怕", "焦虑", "痛", "冷", "夜", "孤", "失", "难", "累", "哭", "死")


@dataclass(frozen=True)
class SvgLine:
    points: tuple[tuple[float, float], ...]
    color: tuple[int, int, int]
    alpha: int
    width: int


def _fract(x):
    return x - np.floor(x)


def _clip01(x):
    return np.clip(x, 0.0, 1.0)


def _fmt_num(value):
    return f"{float(value):.3f}".rstrip("0").rstrip(".")


def _rgb_hex(rgb):
    vals = [int(np.clip(round(float(c)), 0, 255)) for c in rgb]
    return "#{:02x}{:02x}{:02x}".format(*vals)


def _safe_unit(vec):
    norm = float(np.linalg.norm(vec))
    if norm < 1e-8:
        return vec.astype(np.float64), norm
    return vec.astype(np.float64) / norm, norm


def _semantic_signature(vec, count=10):
    unit, _ = _safe_unit(vec)
    n = unit.shape[0]
    idx = np.arange(n, dtype=np.float64) + 1.0
    feats = []
    for k in range(1, count + 1):
        phase = k * 0.731
        wave = np.sin(idx * (0.007 * k + 0.011) + phase)
        wave += 0.65 * np.cos(idx * (0.004 * k + 0.017) - phase * 1.37)
        wave /= np.linalg.norm(wave) + 1e-8
        feats.append(float(np.dot(unit, wave)))
    return np.array(feats, dtype=np.float64)


def split_sentences(content):
    return [s.strip() for s in re.split(r"[。！？!?；;\n]+", content) if s.strip()]


def split_phrases(content):
    return [s.strip() for s in re.split(r"[，,、；;：:\n]+", content) if s.strip()]


def sentence_sentiment_bias(text):
    warm = sum(text.count(token) for token in WARM_HINTS)
    cool = sum(text.count(token) for token in COOL_HINTS)
    if warm == 0 and cool == 0:
        return 0.0
    return np.tanh((warm - cool) / 3.0)


def vector_to_semantic_palette(vec):
    vec = vec.astype(np.float64)
    chunks = np.array_split(vec, 12)
    sig = _semantic_signature(vec, count=10)

    hue_seed = _fract((sig[0] * 11.3 + sig[1] * 7.7 + sig[2] * 5.9 + sig[8] * 3.1) * 3.7)
    base_hue = (hue_seed * 360 + sig[3] * 45) % 360
    sat = np.clip(0.30 + 0.26 * np.abs(sig[3]) + np.std(chunks[3]) * 1.1, 0.22, 0.92)
    light = np.clip(0.16 + 0.20 * np.abs(sig[4]) + np.abs(np.mean(chunks[4])) * 0.28, 0.1, 0.58)
    contrast = np.clip(0.28 + 0.34 * np.abs(sig[5]) + np.std(chunks[5]) * 0.9, 0.2, 0.95)
    temp = np.tanh((sig[6] * 1.4 + np.mean(chunks[6]) * 1.8))
    drift = np.clip(0.22 + 0.5 * np.abs(sig[7]) + np.std(chunks[7]) * 1.2, 0.12, 1.2)

    primary = np.array(colorsys.hsv_to_rgb(base_hue / 360.0, sat, light), dtype=np.float64)
    secondary_hue = (base_hue + 70 + temp * 55) % 360
    secondary = np.array(
        colorsys.hsv_to_rgb(
            secondary_hue / 360.0,
            np.clip(sat * (0.85 + 0.2 * contrast), 0.2, 1.0),
            np.clip(light * (0.75 + 0.5 * contrast), 0.1, 0.85),
        ),
        dtype=np.float64,
    )

    return {
        "hue": float(base_hue),
        "sat": float(sat),
        "light": float(light),
        "contrast": float(contrast),
        "temp": float(temp),
        "drift": float(drift),
        "primary": primary,
        "secondary": secondary,
        "signature": sig,
    }


def build_semantic_background(palette, diary_vec, width, height):
    yy, xx = np.mgrid[0:height, 0:width]
    x = xx / max(1, width - 1)
    y = yy / max(1, height - 1)

    chunks = np.array_split(diary_vec.astype(np.float64), 8)
    sig = palette["signature"]

    code1 = _fract((sig[0] * 13.1 + sig[4] * 7.3 + sig[8] * 5.7) * 2.9)
    code2 = _fract((sig[1] * 11.7 + sig[5] * 9.1 + sig[9] * 4.3) * 3.3)
    code3 = _fract((sig[2] * 15.7 + sig[6] * 6.1 + sig[7] * 5.1) * 2.7)
    code4 = _fract((sig[3] * 10.9 + sig[0] * 8.3 + sig[5] * 4.7) * 3.1)

    c1x = 0.08 + 0.84 * code1
    c1y = 0.08 + 0.84 * code2
    c2x = 0.08 + 0.84 * code3
    c2y = 0.08 + 0.84 * code4

    radial1 = np.exp(-(((x - c1x) ** 2) / (0.05 + 0.16 * palette["contrast"]) + ((y - c1y) ** 2) / (0.07 + 0.14 * palette["contrast"])))
    radial2 = np.exp(-(((x - c2x) ** 2) / (0.06 + 0.15 * palette["contrast"]) + ((y - c2y) ** 2) / (0.05 + 0.12 * palette["contrast"])))

    angle = (code1 * np.pi * 1.8) - np.pi * 0.4 + np.mean(chunks[2]) * np.pi * 0.35
    xr = x * np.cos(angle) - y * np.sin(angle)
    yr = x * np.sin(angle) + y * np.cos(angle)

    wave_a = 0.5 + 0.5 * np.sin((xr * (4.0 + 10.0 * code2) + yr * (3.2 + 8.0 * code3)) * (0.9 + palette["drift"] * 1.1) + palette["hue"] * 0.03)
    wave_b = 0.5 + 0.5 * np.cos((x * (2.8 + 9.0 * code4) - y * (3.0 + 8.0 * code1)) * (1.0 + palette["drift"] * 0.8) + sig[9] * 2.5)

    field = _clip01(0.10 + 0.42 * radial1 + 0.30 * radial2 + 0.23 * wave_a + 0.15 * wave_b)

    base = np.zeros((height, width, 3), dtype=np.float64)
    for c in range(3):
        base[:, :, c] = palette["primary"][c] * (0.35 + 0.70 * field) + palette["secondary"][c] * (0.20 + 0.75 * (1.0 - field))

    grain = _fract(np.sin((xx * 0.131 + yy * 0.173 + palette["hue"] * 0.01) * 113.17) * 43758.5453)
    base += (grain[:, :, None] - 0.5) * (0.03 + 0.05 * palette["drift"])

    return _clip01(base)


def align_sentence_text(sentences, idx, target_count):
    if not sentences:
        return ""
    mapped = int(round(idx * (len(sentences) - 1) / max(1, target_count - 1)))
    return sentences[mapped]


def build_sentence_glow_items(sentence_vectors, sentences):
    total = len(sentence_vectors)
    if total <= LONG_DIARY_SENTENCE_THRESHOLD:
        return [
            (i, sentence_vectors[i], align_sentence_text(sentences, i, total), 1)
            for i in range(total)
        ]

    items = []
    for start in range(0, total, SENTENCE_GLOW_BATCH_SIZE):
        end = min(total, start + SENTENCE_GLOW_BATCH_SIZE)
        chunk = sentence_vectors[start:end]
        weights = np.linalg.norm(chunk, axis=1)
        weights = weights / max(float(weights.sum()), 1e-8)
        vec = np.average(chunk, axis=0, weights=weights)
        center_idx = (start + end - 1) / 2.0
        text = "".join(
            align_sentence_text(sentences, i, total)
            for i in range(start, end)
        )
        items.append((center_idx, vec, text, end - start))
    return items


def blend_sentence_glows(base, sentence_vectors, diary_vector, sentences, palette, content_width=CANVAS_SIZE, content_height=CANVAS_SIZE):
    if sentence_vectors.size == 0:
        return base

    h, w, _ = base.shape
    yy, xx = np.mgrid[0:h, 0:w]
    sx_scale = w / content_width
    sy_scale = h / content_height

    diary_u, _ = _safe_unit(diary_vector)
    norms = np.linalg.norm(sentence_vectors, axis=1)
    norm_anchor = float(np.mean(norms) + 1e-6)
    glow_items = build_sentence_glow_items(sentence_vectors, sentences)

    for i, vec, sentence, group_size in glow_items:
        vec_u, vec_norm = _safe_unit(vec)
        sim = float(np.dot(vec_u, diary_u))
        energy = np.clip(vec_norm / norm_anchor, 0.6, 2.2)

        bias = sentence_sentiment_bias(sentence)
        punct_weight = sum(sentence.count(p) for p in "!?！？") * 0.12 + sentence.count("，") * 0.03

        px = _fract(np.mean(vec[::17]) * 13.37 + i * 0.6180339)
        py = _fract(np.mean(vec[5::19]) * 9.91 + i * 0.4142136)
        py = 0.15 + 0.7 * (0.55 * py + 0.45 * (i / max(1, len(sentence_vectors) - 1)))

        cx = int(px * (w - 1))
        cy = int(py * (h - 1))

        sx = (60 + 110 * energy + 40 * np.abs(sim)) * sx_scale
        sy = (45 + 90 * energy + 30 * np.abs(bias)) * sy_scale
        if group_size > 1:
            spread = 1.0 + 0.18 * np.log1p(group_size)
            sx *= spread
            sy *= spread

        hue = (palette["hue"] + sim * 95 + bias * 45 + i * 12) % 360
        sat = np.clip(0.25 + 0.35 * (0.5 + 0.5 * sim), 0.2, 0.75)
        val = np.clip(0.35 + 0.45 * energy / 2.2 + 0.12 * punct_weight, 0.2, 0.95)
        color = np.array(colorsys.hsv_to_rgb(hue / 360.0, sat, val), dtype=np.float64)

        mask = np.exp(-(((xx - cx) ** 2) / (2 * sx * sx) + ((yy - cy) ** 2) / (2 * sy * sy)))
        alpha = np.clip(0.06 + 0.16 * (0.5 + 0.5 * sim) + punct_weight, 0.04, 0.32)
        if group_size > 1:
            alpha *= min(1.8, 1.0 + 0.08 * (group_size - 1))
        base += mask[:, :, None] * color[None, None, :] * alpha

        # Small brighter core for sentence focus
        core = np.exp(-(((xx - cx) ** 2) / (2 * (sx * 0.25) ** 2) + ((yy - cy) ** 2) / (2 * (sy * 0.25) ** 2)))
        base += core[:, :, None] * color[None, None, :] * (alpha * 0.35)

    return _clip01(base)


def collect_sentence_glow_shapes(sentence_vectors, diary_vector, sentences, palette, width, height):
    if sentence_vectors.size == 0:
        return []

    diary_u, _ = _safe_unit(diary_vector)
    norms = np.linalg.norm(sentence_vectors, axis=1)
    norm_anchor = float(np.mean(norms) + 1e-6)
    glow_items = build_sentence_glow_items(sentence_vectors, sentences)
    shapes = []

    for i, vec, sentence, group_size in glow_items:
        vec_u, vec_norm = _safe_unit(vec)
        sim = float(np.dot(vec_u, diary_u))
        energy = np.clip(vec_norm / norm_anchor, 0.6, 2.2)

        bias = sentence_sentiment_bias(sentence)
        punct_weight = sum(sentence.count(p) for p in "!?！？") * 0.12 + sentence.count("，") * 0.03

        px = _fract(np.mean(vec[::17]) * 13.37 + i * 0.6180339)
        py = _fract(np.mean(vec[5::19]) * 9.91 + i * 0.4142136)
        py = 0.15 + 0.7 * (0.55 * py + 0.45 * (i / max(1, len(sentence_vectors) - 1)))

        sx = 60 + 110 * energy + 40 * np.abs(sim)
        sy = 45 + 90 * energy + 30 * np.abs(bias)
        if group_size > 1:
            spread = 1.0 + 0.18 * np.log1p(group_size)
            sx *= spread
            sy *= spread

        hue = (palette["hue"] + sim * 95 + bias * 45 + i * 12) % 360
        sat = np.clip(0.25 + 0.35 * (0.5 + 0.5 * sim), 0.2, 0.75)
        val = np.clip(0.35 + 0.45 * energy / 2.2 + 0.12 * punct_weight, 0.2, 0.95)
        color = tuple(int(c * 255) for c in colorsys.hsv_to_rgb(hue / 360.0, sat, val))

        alpha = np.clip(0.06 + 0.16 * (0.5 + 0.5 * sim) + punct_weight, 0.04, 0.32)
        if group_size > 1:
            alpha *= min(1.8, 1.0 + 0.08 * (group_size - 1))

        shapes.append(
            {
                "cx": float(px * (width - 1)),
                "cy": float(py * (height - 1)),
                "rx": float(sx),
                "ry": float(sy),
                "color": color,
                "alpha": float(np.clip(alpha, 0.0, 0.58)),
                "core_alpha": float(np.clip(alpha * 0.35, 0.0, 0.26)),
            }
        )

    return shapes


def text_stats(content):
    char_count = len(content)
    punct_count = sum(1 for ch in content if ch in PUNCTUATION)
    phrases = split_phrases(content)
    sentences = split_sentences(content)

    avg_phrase_len = float(np.mean([len(p) for p in phrases])) if phrases else 0.0
    digit_ratio = sum(ch.isdigit() for ch in content) / max(1, char_count)
    cjk_ratio = sum("\u4e00" <= ch <= "\u9fff" for ch in content) / max(1, char_count)

    return {
        "char_count": char_count,
        "punct_count": punct_count,
        "phrases": phrases,
        "sentences": sentences,
        "avg_phrase_len": avg_phrase_len,
        "digit_ratio": digit_ratio,
        "cjk_ratio": cjk_ratio,
        "comma_count": content.count("，") + content.count(","),
        "period_count": content.count("。") + content.count("."),
        "question_count": content.count("？") + content.count("?"),
        "exclaim_count": content.count("！") + content.count("!"),
        "semicolon_count": content.count("；") + content.count(";"),
    }


def collect_text_grid_lines(width, height, content, day_token, palette):
    stats = text_stats(content)
    density = stats["punct_count"] / max(1, stats["char_count"])
    seed = sum(ord(ch) for ch in day_token) + stats["char_count"]
    lines = []

    line_base = int(np.clip(40 + density * 360, 35, 158))
    grid_h_gap = int(np.clip(18 + stats["avg_phrase_len"] * 1.6, 16, 72))
    grid_v_gap = int(np.clip(20 + stats["char_count"] / max(2, len(stats["phrases"]) + 1) * 0.75, 18, 92))

    grid_hue = (palette["hue"] + 120 + stats["comma_count"] * 2 - stats["question_count"] * 5) % 360
    grid_rgb = tuple(int(c * 255) for c in colorsys.hsv_to_rgb(grid_hue / 360.0, 0.38, 0.94))

    h_width = 1 + (stats["period_count"] % 2)
    v_width = 1 + (1 if stats["semicolon_count"] > 2 else 0)

    for y in range(0, height, grid_h_gap):
        jitter = int(np.sin((y + seed) * 0.031) * (2 + stats["comma_count"] % 4))
        alpha = int(line_base * (0.63 + 0.45 * ((y // max(1, grid_h_gap)) % 3) / 2))
        lines.append(SvgLine(((0, y + jitter), (width, y + jitter)), grid_rgb, alpha, h_width))

    for x in range(0, width, grid_v_gap):
        jitter = int(np.cos((x + seed) * 0.027) * (2 + stats["period_count"] % 5))
        alpha = int(line_base * (0.52 + 0.5 * ((x // max(1, grid_v_gap)) % 4) / 3))
        lines.append(SvgLine(((x + jitter, 0), (x + jitter, height)), grid_rgb, alpha, v_width))

    burst_count = max(2, stats["question_count"] + stats["exclaim_count"])
    for i in range(burst_count):
        x = int(_fract(np.sin((seed + i * 17) * 0.13) * 13.7) * (width - 1))
        y1 = int(_fract(np.cos((seed + i * 29) * 0.11) * 7.3) * (height * 0.7))
        y2 = min(height - 1, y1 + int(height * (0.18 + 0.05 * (i % 3))))
        zig = 8 + (i % 3) * 4
        pts = []
        for y in range(y1, y2, zig):
            dx = (zig // 2) if ((y - y1) // max(1, zig)) % 2 == 0 else -(zig // 2)
            pts.append((x + dx, y))
        if len(pts) > 1:
            lines.append(SvgLine(tuple(pts), grid_rgb, int(line_base * 1.28), 1))

    phrase_candidates = [p for p in stats["phrases"] if 1 <= len(p) <= 16]
    for i, phrase in enumerate(phrase_candidates[:80]):
        code = sum(ord(ch) for ch in phrase)
        x = int((code * 1.37 + i * 41 + seed) % width)
        y = int((code * 0.91 + i * 67 + seed * 0.7) % height)
        length = int(np.clip(len(phrase) * 6 + (code % 23), 12, 180))
        angle = ((code % 90) - 45) * np.pi / 180.0
        x2 = int(np.clip(x + np.cos(angle) * length, 0, width - 1))
        y2 = int(np.clip(y + np.sin(angle) * length, 0, height - 1))

        alpha = int(np.clip(48 + len(phrase) * 6.5 + phrase.count("我") * 15, 40, 200))
        stroke = 2 if len(phrase) >= 12 else 1
        lines.append(SvgLine(((x, y), (x2, y2)), grid_rgb, alpha, stroke))

    return stats, density, lines


def draw_text_grid_layer(width, height, content, day_token, palette, output_width=None, output_height=None):
    output_width = int(output_width or width)
    output_height = int(output_height or height)
    scale_x = output_width / width
    scale_y = output_height / height
    line_scale = (scale_x + scale_y) / 2.0
    stats, density, lines = collect_text_grid_lines(width, height, content, day_token, palette)
    overlay = Image.new("RGBA", (output_width, output_height), (0, 0, 0, 0))
    draw = ImageDraw.Draw(overlay)

    for line in lines:
        pts = [(x * scale_x, y * scale_y) for x, y in line.points]
        draw.line(pts, fill=(*line.color, line.alpha), width=max(1, int(round(line.width * line_scale))))

    return overlay.filter(ImageFilter.GaussianBlur(radius=(0.58 + density * 1.8) * line_scale))


def compose_diary_image(
    diary_vec,
    sentence_vectors,
    content,
    date_text,
    width=CANVAS_SIZE,
    height=CANVAS_SIZE,
    output_width=None,
    output_height=None,
):
    output_width = int(output_width or width)
    output_height = int(output_height or height)
    line_scale = (output_width / width + output_height / height) / 2.0
    palette = vector_to_semantic_palette(diary_vec)
    sentences = split_sentences(content)

    base = build_semantic_background(palette, diary_vec, output_width, output_height)
    base = blend_sentence_glows(
        base,
        sentence_vectors,
        diary_vec,
        sentences,
        palette,
        content_width=width,
        content_height=height,
    )

    img = Image.fromarray((base * 255).astype(np.uint8), mode="RGB")
    grid_layer = draw_text_grid_layer(
        width,
        height,
        content,
        date_text,
        palette,
        output_width=output_width,
        output_height=output_height,
    )
    img = Image.alpha_composite(img.convert("RGBA"), grid_layer).convert("RGB")

    return img.filter(ImageFilter.GaussianBlur(radius=0.4 * line_scale))


def svg_line_elements(lines):
    out = []
    for line in lines:
        color = _rgb_hex(line.color)
        opacity = _fmt_num(np.clip(line.alpha / 255.0, 0.0, 1.0))
        width = _fmt_num(line.width)
        if len(line.points) == 2:
            (x1, y1), (x2, y2) = line.points
            out.append(
                f'    <line x1="{_fmt_num(x1)}" y1="{_fmt_num(y1)}" '
                f'x2="{_fmt_num(x2)}" y2="{_fmt_num(y2)}" '
                f'stroke="{color}" stroke-opacity="{opacity}" stroke-width="{width}" />'
            )
        else:
            pts = " ".join(f"{_fmt_num(x)},{_fmt_num(y)}" for x, y in line.points)
            out.append(
                f'    <polyline points="{pts}" stroke="{color}" '
                f'stroke-opacity="{opacity}" stroke-width="{width}" />'
            )
    return "\n".join(out)


def write_diary_svg(
    svg_path,
    date,
    diary_vec,
    sentence_vectors,
    content,
    width=CANVAS_SIZE,
    height=CANVAS_SIZE,
    output_width=OUTPUT_SIZE,
    output_height=OUTPUT_SIZE,
):
    palette = vector_to_semantic_palette(diary_vec)
    sentences = split_sentences(content)
    glows = collect_sentence_glow_shapes(sentence_vectors, diary_vec, sentences, palette, width, height)
    _, density, grid_lines = collect_text_grid_lines(width, height, content, date, palette)

    grid_blur = 0.58 + density * 1.8
    title = escape(f"diary_Glitch {date}")

    glow_elements = []
    for shape in glows:
        color = _rgb_hex(shape["color"])
        glow_elements.append(
            f'    <ellipse cx="{_fmt_num(shape["cx"])}" cy="{_fmt_num(shape["cy"])}" '
            f'rx="{_fmt_num(shape["rx"])}" ry="{_fmt_num(shape["ry"])}" '
            f'fill="{color}" opacity="{_fmt_num(shape["alpha"])}" />'
        )
        glow_elements.append(
            f'    <ellipse cx="{_fmt_num(shape["cx"])}" cy="{_fmt_num(shape["cy"])}" '
            f'rx="{_fmt_num(shape["rx"] * 0.25)}" ry="{_fmt_num(shape["ry"] * 0.25)}" '
            f'fill="{color}" opacity="{_fmt_num(shape["core_alpha"])}" />'
        )

    body = f'''<?xml version="1.0" encoding="UTF-8"?>
<svg xmlns="http://www.w3.org/2000/svg" width="{output_width}" height="{output_height}" viewBox="0 0 {width} {height}">
  <title>{title}</title>
  <defs>
    <filter id="glow-blur" x="-12%" y="-12%" width="124%" height="124%">
      <feGaussianBlur stdDeviation="{_fmt_num(18 + palette["drift"] * 8)}" />
    </filter>
    <filter id="grid-blur" x="-4%" y="-4%" width="108%" height="108%">
      <feGaussianBlur stdDeviation="{_fmt_num(grid_blur)}" />
    </filter>
  </defs>
  <g filter="url(#glow-blur)">
{chr(10).join(glow_elements)}
  </g>
  <g fill="none" stroke-linecap="square" stroke-linejoin="miter" filter="url(#grid-blur)">
{svg_line_elements(grid_lines)}
  </g>
</svg>
'''
    svg_path.parent.mkdir(parents=True, exist_ok=True)
    svg_path.write_text(body, encoding="utf-8")


def get_project_root():
    return Path(__file__).parent.parent


def get_script_dir():
    return Path(__file__).parent


def get_next_numbered_dir(output_root, prefix):
    output_root.mkdir(parents=True, exist_ok=True)
    indices = []
    for child in output_root.iterdir():
        if not child.is_dir() or not child.name.startswith(prefix):
            continue
        suffix = child.name[len(prefix):]
        if suffix.isdigit():
            indices.append(int(suffix))

    next_index = max(indices, default=0) + 1
    output_dir = output_root / f"{prefix}{next_index}"
    output_dir.mkdir(parents=True, exist_ok=False)
    return output_dir


def get_next_try_dirs(output_root, export_png=True, export_svg=True):
    output_root.mkdir(parents=True, exist_ok=True)
    max_index = 0
    for child in output_root.iterdir():
        if not child.is_dir():
            continue
        match = re.fullmatch(r"(?:svg_)?try_(\d+)", child.name)
        if match:
            max_index = max(max_index, int(match.group(1)))

    next_index = max_index + 1
    try_dir = output_root / f"try_{next_index}" if export_png else None
    svg_try_dir = output_root / f"svg_try_{next_index}" if export_svg else None
    if try_dir is not None:
        try_dir.mkdir(parents=True, exist_ok=False)
    if svg_try_dir is not None:
        svg_try_dir.mkdir(parents=True, exist_ok=False)
    return try_dir, svg_try_dir


def get_next_try_dir(output_root):
    return get_next_numbered_dir(output_root, "try_")


def get_next_high_try_dir(output_root):
    return get_next_numbered_dir(output_root, "high_try_")


def entry_stem(date, count):
    return date if count == 0 else f"{date}_{count + 1}"


def load_entries_by_date(entries_path):
    entries = json.loads(entries_path.read_text(encoding="utf-8"))
    if not isinstance(entries, list):
        raise ValueError(f"Expected a list in {entries_path}, got {type(entries).__name__}")

    mapping = {}
    seen_dates = {}
    for item in entries:
        if not isinstance(item, dict):
            continue
        date = item.get("date")
        content = item.get("content", "")
        if date:
            count = seen_dates.get(date, 0)
            seen_dates[date] = count + 1
            mapping[entry_stem(date, count)] = content

    return mapping


def ensure_day_alignment(dates, entries_by_date, sentence_root):
    missing_entries = [d for d in dates if d not in entries_by_date]
    missing_sentence_dirs = [d for d in dates if not (sentence_root / d).is_dir()]

    if missing_entries:
        raise FileNotFoundError(f"Missing diary entries for dates: {missing_entries}")
    if missing_sentence_dirs:
        raise FileNotFoundError(f"Missing sentence vector directories for dates: {missing_sentence_dirs}")


def parse_args():
    parser = argparse.ArgumentParser(
        description="Render diary glitch images from diary and sentence vectors."
    )
    parser.add_argument(
        "--date",
        help="Render only one date or entry stem, for example 2022-04-18.",
    )
    parser.add_argument(
        "--high",
        action="store_true",
        help="Render the selected --date as a 5000x5000 image into high_try_N.",
    )
    output_group = parser.add_mutually_exclusive_group()
    output_group.add_argument(
        "--png-only",
        action="store_true",
        help="Only render PNGs into try_N; skip SVG output.",
    )
    output_group.add_argument(
        "--svg-only",
        action="store_true",
        help="Only render SVGs into svg_try_N; skip PNG output.",
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
    return args


def load_render_inputs(date, diary_vector_dir, sentence_vector_root, entries_by_date):
    vector_file = diary_vector_dir / f"{date}.npy"
    sentence_vec_path = sentence_vector_root / date / "sentence_vectors.npy"

    if date not in entries_by_date:
        raise FileNotFoundError(f"Missing diary entry for date: {date}")
    if not vector_file.exists():
        raise FileNotFoundError(f"Diary vector file not found: {vector_file}")
    if not sentence_vec_path.exists():
        raise FileNotFoundError(f"Sentence vector file not found: {sentence_vec_path}")

    return (
        np.load(vector_file),
        np.load(sentence_vec_path),
        entries_by_date[date],
    )


def render_one_date(
    date,
    diary_vector_dir,
    sentence_vector_root,
    entries_by_date,
    width,
    height,
    output_dir=None,
    svg_output_dir=None,
    label=None,
    upscale_to=None,
    output_width=OUTPUT_SIZE,
    output_height=OUTPUT_SIZE,
):
    diary_vec, sentence_vectors, content = load_render_inputs(
        date,
        diary_vector_dir,
        sentence_vector_root,
        entries_by_date,
    )

    print(f"\n=== {label or date} ===")
    print(
        f"  size={width}x{height}, text chars={len(content)}, "
        f"sentence_vectors={sentence_vectors.shape[0]}"
    )

    output_file = None
    if output_dir is not None:
        img = compose_diary_image(
            diary_vec,
            sentence_vectors,
            content,
            date,
            width=width,
            height=height,
            output_width=width if upscale_to else output_width,
            output_height=height if upscale_to else output_height,
        )

        if upscale_to:
            img = img.resize(upscale_to, Image.Resampling.NEAREST)

        output_file = output_dir / f"{date}.png"
        img.save(output_file)
        print(f"  png -> {output_file}")

    svg_file = None
    if svg_output_dir is not None:
        svg_file = svg_output_dir / f"{date}.svg"
        write_diary_svg(
            svg_file,
            date,
            diary_vec,
            sentence_vectors,
            content,
            width=width,
            height=height,
            output_width=output_width,
            output_height=output_height,
        )
        print(f"  svg -> {svg_file}")

    return output_file, svg_file


def main():
    args = parse_args()
    project_root = get_project_root()
    script_dir = get_script_dir()

    diary_vector_dir = project_root / "diary_vectors"
    sentence_vector_root = project_root / "diary_sentence_vectors"
    diary_entries_path = project_root / "diary_entries.json"
    output_root = project_root / "output_All" / script_dir.name

    if not diary_vector_dir.exists():
        raise FileNotFoundError(f"Diary vector directory not found: {diary_vector_dir}")
    if not sentence_vector_root.exists():
        raise FileNotFoundError(f"Sentence vector directory not found: {sentence_vector_root}")
    if not diary_entries_path.exists():
        raise FileNotFoundError(f"Diary entries file not found: {diary_entries_path}")

    vector_files = sorted(diary_vector_dir.glob("*.npy"), key=lambda p: p.stem)
    if not vector_files:
        raise FileNotFoundError(f"No .npy vector files found in: {diary_vector_dir}")

    entries_by_date = load_entries_by_date(diary_entries_path)

    if args.high and not args.date:
        raise ValueError("--high requires --date so only one large image is rendered.")
    if args.high and args.svg_only:
        raise ValueError("--svg-only cannot be combined with --high because --high is a PNG upscale mode.")

    if args.date:
        if args.high:
            output_dir = get_next_high_try_dir(output_root)
            svg_output_dir = None
        else:
            output_dir, svg_output_dir = get_next_try_dirs(
                output_root,
                export_png=not args.svg_only,
                export_svg=not args.png_only,
            )
        if output_dir is not None:
            print(f"Saving PNG outputs to: {output_dir}")
        if svg_output_dir is not None:
            print(f"Saving SVG outputs to: {svg_output_dir}")
        render_one_date(
            args.date,
            diary_vector_dir,
            sentence_vector_root,
            entries_by_date,
            args.content_size,
            args.content_size,
            output_dir=output_dir,
            svg_output_dir=svg_output_dir,
            upscale_to=(HIGH_CANVAS_SIZE, HIGH_CANVAS_SIZE) if args.high else None,
            output_width=args.resolution,
            output_height=args.resolution,
        )
        if args.svg_only:
            print("\nDone! Generated 1 SVG.")
        elif args.high:
            print("\nDone! Generated 1 high-resolution PNG.")
        elif args.png_only:
            print("\nDone! Generated 1 PNG.")
        else:
            print("\nDone! Generated 1 PNG and 1 SVG.")
        return

    dates = [vf.stem for vf in vector_files]
    ensure_day_alignment(dates, entries_by_date, sentence_vector_root)

    try_dir, svg_try_dir = get_next_try_dirs(
        output_root,
        export_png=not args.svg_only,
        export_svg=not args.png_only,
    )
    if try_dir is not None:
        print(f"Saving PNG outputs to: {try_dir}")
    if svg_try_dir is not None:
        print(f"Saving SVG outputs to: {svg_try_dir}")

    for i, vector_file in enumerate(vector_files, start=1):
        date = vector_file.stem
        render_one_date(
            date,
            diary_vector_dir,
            sentence_vector_root,
            entries_by_date,
            args.content_size,
            args.content_size,
            output_dir=try_dir,
            svg_output_dir=svg_try_dir,
            label=f"[{i:02d}/{len(vector_files)}] {date}",
            output_width=args.resolution,
            output_height=args.resolution,
        )

    if args.svg_only:
        print(f"\nDone! Generated {len(vector_files)} aligned daily SVGs.")
    elif args.png_only:
        print(f"\nDone! Generated {len(vector_files)} aligned daily PNGs.")
    else:
        print(f"\nDone! Generated {len(vector_files)} aligned daily PNGs and SVGs.")


if __name__ == "__main__":
    main()
