"""
rendering.py — 视觉映射 + 绘制 + 导出
"""

import colorsys
import hashlib
import math
import os
import re

import numpy as np
from PIL import Image


def _add_common_cairo_library_paths() -> None:
    existing = [p for p in os.environ.get("DYLD_FALLBACK_LIBRARY_PATH", "").split(":") if p]
    candidates = [
        "/opt/homebrew/opt/cairo/lib",
        "/usr/local/opt/cairo/lib",
        "/opt/local/lib",
    ]
    additions = [
        p
        for p in candidates
        if p not in existing and os.path.exists(os.path.join(p, "libcairo.2.dylib"))
    ]
    if additions:
        os.environ["DYLD_FALLBACK_LIBRARY_PATH"] = ":".join(additions + existing)


try:
    import cairo
except ModuleNotFoundError:
    _add_common_cairo_library_paths()
    import cairocffi as cairo

from .config import (
    BEZIER_TOLERANCE,
    BLOCK_ASPECT_RATIOS,
    BLOCK_MAX_SIZE,
    BLOCK_MIN_SIZE,
    BLOCK_OPACITY_MAX,
    BLOCK_OPACITY_MIN,
    BLOCK_SIZE_POWER,
    BLOCK_SIZE_SCALE,
    BLOCK_SPREAD_CURVE,
    BLOCK_X_JITTER,
    BLOCK_X_RANGE,
    BLOCK_Y_JITTER,
    CANVAS_PADDING,
    CANVAS_SIZE,
    COLOR_HUE_JITTER,
    EMOTION_SYSTEM,
    EXISTENTIAL_OPACITY_CAP,
    LINE_LAYER_COVERAGE,
    METABALL_BASE_RADIUS,
    METABALL_EXTRA_BALLS,
    METABALL_EXTRA_RADIUS_SCALE,
    METABALL_GRID_RES,
    METABALL_RADIUS_MAX,
    METABALL_RADIUS_MIN,
    METABALL_THRESHOLDS,
    NOISE_STRENGTH,
    OUTPUT_SIZE,
    STROKE_WIDTH,
)


def set_render_sizes(content_size: int, output_size: int) -> None:
    global CANVAS_SIZE, OUTPUT_SIZE
    CANVAS_SIZE = int(content_size)
    OUTPUT_SIZE = int(output_size)


def hex_to_rgb(hex_color: str) -> tuple[float, float, float]:
    hex_color = hex_color.lstrip("#")
    return (
        int(hex_color[0:2], 16) / 255.0,
        int(hex_color[2:4], 16) / 255.0,
        int(hex_color[4:6], 16) / 255.0,
    )


def jitter_hue(hex_color: str, jitter_deg: float) -> tuple[float, float, float]:
    # 先注释掉随机色相抖动，保持纯确定性颜色输出
    _ = jitter_deg
    return hex_to_rgb(hex_color)


def _stable_hash_unit(text: str) -> float:
    digest = hashlib.sha256(text.encode("utf-8")).digest()
    val = int.from_bytes(digest[:8], "big", signed=False)
    return (val % 10_000_000) / 10_000_000.0


def _vec_unit(vec: np.ndarray, idx: int, default: float = 0.5) -> float:
    if vec is None or vec.size == 0:
        return default
    raw = float(vec[idx % vec.size])
    return max(0.0, min(1.0, 0.5 + 0.5 * math.tanh(raw)))


def _vec_signed(vec: np.ndarray, idx: int, default: float = 0.0) -> float:
    return _vec_unit(vec, idx, 0.5) * 2.0 - 1.0 if vec is not None and vec.size > 0 else default


def map_function_word_to_block(
    fw_token: dict,
    emotion: str,
    entry_vector: np.ndarray,
    token_index: int = 0,
    total_tokens: int = 1,
) -> dict:
    category = fw_token["category"]
    count = fw_token["count"]
    intensity = fw_token["intensity"]
    position = fw_token["position"]
    spread = fw_token.get("spread", 0.3)
    vec = np.asarray(entry_vector, dtype=np.float32).reshape(-1) if entry_vector is not None else np.array([], dtype=np.float32)

    count_norm = min(count / 50.0, 1.0)
    base_side = BLOCK_MIN_SIZE + (BLOCK_MAX_SIZE - BLOCK_MIN_SIZE) * (count_norm ** BLOCK_SIZE_POWER)
    side = base_side * BLOCK_SIZE_SCALE.get(category, 1.0)

    aspect = BLOCK_ASPECT_RATIOS.get(category, 1.0)
    area = side * side
    w = math.sqrt(area * aspect)
    h = math.sqrt(area / aspect)

    content_height = CANVAS_SIZE - 2 * CANVAS_PADDING
    token_hash = _stable_hash_unit(fw_token.get("word", ""))
    entry_phase = 2 * math.pi * _vec_unit(vec, 5)
    token_phase = 2 * math.pi * token_hash
    seq_phase = 2 * math.pi * (token_index / max(total_tokens, 1))
    y_wave_strength = BLOCK_Y_JITTER * (0.25 + 0.75 * spread)
    y_wave = math.sin(2 * math.pi * position + entry_phase + token_phase + seq_phase) * y_wave_strength
    y_center = CANVAS_PADDING + position * content_height + y_wave

    content_width = CANVAS_SIZE - 2 * CANVAS_PADDING
    base_spread_radius = content_width * BLOCK_X_RANGE / 2
    effective_radius = base_spread_radius * (0.4 + 0.6 * (spread ** BLOCK_SPREAD_CURVE))
    entry_spread_gain = 0.8 + 0.9 * abs(_vec_signed(vec, 11))
    x_from_position = (position - 0.5) * 0.65
    x_from_intensity = (intensity - 0.5) * 0.85
    x_from_hash = (token_hash - 0.5) * 1.1
    x_from_wave = math.cos(2 * math.pi * position * (1.0 + _vec_unit(vec, 17)) + entry_phase + token_phase) * 0.35
    x_signal = x_from_position + x_from_intensity + x_from_hash + x_from_wave
    x_center = CANVAS_SIZE / 2 + x_signal * effective_radius * entry_spread_gain
    x_center += math.cos(token_phase + seq_phase + entry_phase) * BLOCK_X_JITTER * (0.15 + 0.85 * spread)

    x = max(CANVAS_PADDING, min(CANVAS_SIZE - CANVAS_PADDING - w, x_center - w / 2))
    y = max(CANVAS_PADDING, min(CANVAS_SIZE - CANVAS_PADDING - h, y_center - h / 2))

    alpha = BLOCK_OPACITY_MIN + (BLOCK_OPACITY_MAX - BLOCK_OPACITY_MIN) * intensity
    if category == "existential":
        alpha = min(alpha, EXISTENTIAL_OPACITY_CAP)

    block_color_hex = EMOTION_SYSTEM.get(emotion, EMOTION_SYSTEM["calm"])["block_color"]
    r, g, b = jitter_hue(block_color_hex, COLOR_HUE_JITTER)
    return {"x": float(x), "y": float(y), "w": float(w), "h": float(h), "r": r, "g": g, "b": b, "alpha": float(alpha)}


def map_content_words_to_metaballs(clusters: dict, weights: dict, entry_vector: np.ndarray) -> list[dict]:
    if not clusters:
        return []
    cluster_word_lists = [clusters[cid] for cid in sorted(clusters.keys()) if clusters[cid]]
    if not cluster_word_lists:
        return []

    vec = np.asarray(entry_vector, dtype=np.float32).reshape(-1) if entry_vector is not None else np.array([], dtype=np.float32)
    coverage_margin = (1 - LINE_LAYER_COVERAGE) / 2 * CANVAS_SIZE
    canvas_usable = CANVAS_SIZE - 2 * coverage_margin
    max_words_in_cluster = max((len(wl) for wl in cluster_word_lists), default=1)
    center_x = CANVAS_SIZE / 2 + _vec_signed(vec, 3) * coverage_margin * 0.9
    center_y = CANVAS_SIZE / 2 + _vec_signed(vec, 7) * coverage_margin * 0.9
    global_rotation = 2 * math.pi * _vec_unit(vec, 9)
    radius_gain = 0.75 + 0.9 * _vec_unit(vec, 13)
    ellipse_x = 0.7 + 0.8 * _vec_unit(vec, 19)
    ellipse_y = 0.7 + 0.8 * _vec_unit(vec, 23)
    weight_gamma = 0.6 + 1.4 * _vec_unit(vec, 29)
    radius_scale = 0.85 + 0.7 * _vec_unit(vec, 31)

    metaballs = []
    cluster_centers_canvas = []
    n_clusters = max(len(cluster_word_lists), 1)
    for i, word_list in enumerate(cluster_word_lists):
        ordered_words = sorted(word_list)
        cluster_hash = _stable_hash_unit("|".join(ordered_words))
        angle = global_rotation + 2 * math.pi * (i / n_clusters + (cluster_hash - 0.5) * 0.25)
        ring_r = canvas_usable * (0.22 + 0.18 * (len(word_list) / max(max_words_in_cluster, 1))) * radius_gain
        local_rx = ring_r * ellipse_x * (0.8 + 0.4 * cluster_hash)
        local_ry = ring_r * ellipse_y * (1.2 - 0.4 * cluster_hash)
        cx = center_x + math.cos(angle) * local_rx
        cy = center_y + math.sin(angle) * local_ry
        cx = max(coverage_margin, min(CANVAS_SIZE - coverage_margin, cx))
        cy = max(coverage_margin, min(CANVAS_SIZE - coverage_margin, cy))
        cluster_centers_canvas.append((cx, cy, len(word_list)))

        n_words = max(len(ordered_words), 1)
        for wi, word in enumerate(ordered_words):
            w_weight = min(max(weights.get(word, 0.1), 0.0), 1.0)
            w_weight = w_weight ** weight_gamma
            radius = min(max(w_weight * METABALL_BASE_RADIUS * radius_scale, METABALL_RADIUS_MIN), METABALL_RADIUS_MAX)
            offset_scale = (0.18 + 0.16 * _vec_unit(vec, 37 + wi)) * canvas_usable / max(len(word_list), 1)
            theta = global_rotation + 2 * math.pi * wi / n_words + 2 * math.pi * cluster_hash
            metaballs.append(
                {
                    "x": float(cx + math.cos(theta) * offset_scale),
                    "y": float(cy + math.sin(theta) * offset_scale),
                    "radius": float(radius),
                    "weight": float(w_weight),
                }
            )

    for cx, cy, n_words in cluster_centers_canvas:
        center_radius = METABALL_BASE_RADIUS * (0.8 + 0.4 * n_words / max(max_words_in_cluster, 1))
        metaballs.append({"x": float(cx), "y": float(cy), "radius": float(min(center_radius, METABALL_RADIUS_MAX)), "weight": 1.0})

    extra_ratio = 0.4 + 0.9 * _vec_unit(vec, 41)
    extra_count = max(0, int(METABALL_EXTRA_BALLS * extra_ratio))
    for ei in range(extra_count):
        px = center_x + _vec_signed(vec, 43 + ei * 2) * canvas_usable * 0.45
        py = center_y + _vec_signed(vec, 44 + ei * 2) * canvas_usable * 0.45
        px = max(coverage_margin, min(CANVAS_SIZE - coverage_margin, px))
        py = max(coverage_margin, min(CANVAS_SIZE - coverage_margin, py))
        pr = METABALL_BASE_RADIUS * METABALL_EXTRA_RADIUS_SCALE * (0.55 + 0.9 * _vec_unit(vec, 80 + ei))
        metaballs.append(
            {
                "x": float(px),
                "y": float(py),
                "radius": float(min(max(pr, METABALL_RADIUS_MIN), METABALL_RADIUS_MAX)),
                "weight": 0.5,
            }
        )

    return metaballs


def _build_metaball_field(metaballs: list[dict]) -> np.ndarray:
    xs = np.linspace(0, CANVAS_SIZE, METABALL_GRID_RES, dtype=np.float32)
    ys = np.linspace(0, CANVAS_SIZE, METABALL_GRID_RES, dtype=np.float32)
    XX, YY = np.meshgrid(xs, ys)
    field = np.zeros((METABALL_GRID_RES, METABALL_GRID_RES), dtype=np.float32)
    for mb in metaballs:
        r2 = (XX - mb["x"]) ** 2 + (YY - mb["y"]) ** 2
        field += (mb["radius"] ** 2) / (r2 + 1.0)
    return field


def _rdp_simplify(points: np.ndarray, tolerance: float) -> np.ndarray:
    if len(points) < 3:
        return points

    def rdp(pts, eps):
        if len(pts) < 3:
            return pts
        d_max, idx, end = 0, 0, len(pts) - 1
        for i in range(1, end):
            x0, y0 = pts[i]
            x1, y1 = pts[0]
            x2, y2 = pts[end]
            dx, dy = x2 - x1, y2 - y1
            denom = max(np.sqrt(dx * dx + dy * dy), 1e-9)
            d = abs(dy * x0 - dx * y0 + x2 * y1 - y2 * x1) / denom
            if d > d_max:
                d_max, idx = d, i
        if d_max > eps:
            left = rdp(pts[: idx + 1], eps)
            right = rdp(pts[idx:], eps)
            return np.vstack([left[:-1], right])
        return np.array([pts[0], pts[-1]])

    return rdp(points, tolerance)


def _draw_smooth_contour(ctx: cairo.Context, points: np.ndarray, closed: bool = True) -> None:
    if len(points) < 2:
        return

    def catmull_rom_cp(p0, p1, p2, p3):
        cp1 = (p1[0] + (p2[0] - p0[0]) / 6, p1[1] + (p2[1] - p0[1]) / 6)
        cp2 = (p2[0] - (p3[0] - p1[0]) / 6, p2[1] - (p3[1] - p1[1]) / 6)
        return cp1, cp2

    n = len(points)
    ctx.move_to(points[0][0], points[0][1])
    for i in range(n - 1 if not closed else n):
        p0 = points[(i - 1) % n]
        p1 = points[i % n]
        p2 = points[(i + 1) % n]
        p3 = points[(i + 2) % n]
        cp1, cp2 = catmull_rom_cp(p0, p1, p2, p3)
        ctx.curve_to(cp1[0], cp1[1], cp2[0], cp2[1], p2[0], p2[1])
    if closed:
        ctx.close_path()


def _draw_line_layer(surface, metaball_params: list[dict], stroke_color_hex: str, surface_scale: float = 1.0) -> None:
    if not metaball_params:
        return
    from skimage import measure

    ctx = cairo.Context(surface)
    if surface_scale != 1.0:
        ctx.scale(surface_scale, surface_scale)
    sr, sg, sb = hex_to_rgb(stroke_color_hex)
    field = _build_metaball_field(metaball_params)

    t_count = max(len(METABALL_THRESHOLDS), 1)
    for ti, threshold in enumerate(METABALL_THRESHOLDS):
        contours = measure.find_contours(field, level=threshold)
        for contour in contours:
            if len(contour) < 6:
                continue
            canvas_pts = contour[:, ::-1] * (CANVAS_SIZE / METABALL_GRID_RES)
            simplified = _rdp_simplify(canvas_pts, BEZIER_TOLERANCE * 2)
            if len(simplified) < 3:
                continue
            layer_ratio = ti / max(t_count - 1, 1)
            sw = max(0.45, STROKE_WIDTH * (0.72 + 0.4 * layer_ratio))
            ctx.set_line_width(sw)
            ctx.set_source_rgba(sr, sg, sb, 0.38 + 0.24 * layer_ratio)
            ctx.set_line_cap(cairo.LINE_CAP_ROUND)
            ctx.set_line_join(cairo.LINE_JOIN_ROUND)
            dist_ends = np.linalg.norm(canvas_pts[0] - canvas_pts[-1])
            closed = dist_ends < (CANVAS_SIZE / METABALL_GRID_RES * 3)
            _draw_smooth_contour(ctx, simplified, closed=closed)
            ctx.stroke()


def _draw_block_layer(surface, block_params: list[dict], surface_scale: float = 1.0) -> None:
    ctx = cairo.Context(surface)
    if surface_scale != 1.0:
        ctx.scale(surface_scale, surface_scale)
    for bp in block_params:
        ctx.set_source_rgba(bp["r"], bp["g"], bp["b"], bp["alpha"])
        ctx.rectangle(bp["x"], bp["y"], bp["w"], bp["h"])
        ctx.fill()


def _add_noise(img_array: np.ndarray, strength: float, seed: int = 0) -> np.ndarray:
    rng = np.random.RandomState(seed)
    noise = rng.normal(0, strength * 255, img_array.shape[:2]).astype(np.float32)
    result = img_array.astype(np.float32).copy()
    result[:, :, 0] = np.clip(result[:, :, 0] + noise, 0, 255)
    result[:, :, 1] = np.clip(result[:, :, 1] + noise, 0, 255)
    result[:, :, 2] = np.clip(result[:, :, 2] + noise, 0, 255)
    return result.astype(np.uint8)


def _cairo_surface_to_pil(surface: cairo.ImageSurface) -> Image.Image:
    data = surface.get_data()
    img_array = np.frombuffer(data, dtype=np.uint8).reshape(surface.get_height(), surface.get_width(), 4)
    return Image.fromarray(img_array[:, :, [2, 1, 0, 3]].copy(), mode="RGBA")


def _paint_visual(
    surface,
    emotion: str,
    block_params: list[dict],
    metaball_params: list[dict],
    surface_scale: float = 1.0,
) -> None:
    emotion_cfg = EMOTION_SYSTEM.get(emotion, EMOTION_SYSTEM["calm"])
    bg_r, bg_g, bg_b = hex_to_rgb(emotion_cfg["bg"])
    stroke_hex = emotion_cfg["stroke"]

    ctx = cairo.Context(surface)
    ctx.set_source_rgb(bg_r, bg_g, bg_b)
    ctx.paint()

    _draw_line_layer(surface, metaball_params, stroke_hex, surface_scale=surface_scale)
    _draw_block_layer(surface, block_params, surface_scale=surface_scale)


def compose_and_save(entry: dict, emotion: str, block_params: list[dict], metaball_params: list[dict], output_path: str) -> None:
    surface = cairo.ImageSurface(cairo.FORMAT_ARGB32, OUTPUT_SIZE, OUTPUT_SIZE)
    surface_scale = OUTPUT_SIZE / CANVAS_SIZE
    _paint_visual(surface, emotion, block_params, metaball_params, surface_scale=surface_scale)

    img_array = np.array(_cairo_surface_to_pil(surface))
    # 先注释掉随机噪点
    # if NOISE_STRENGTH > 0:
    #     img_array = _add_noise(img_array, NOISE_STRENGTH, seed=hash(entry.get("date", "")) % 10000)

    os.makedirs(os.path.dirname(output_path) if os.path.dirname(output_path) else ".", exist_ok=True)
    image = Image.fromarray(img_array, mode="RGBA")
    image.save(output_path, "PNG", optimize=False)


def _set_svg_display_size(output_path: str) -> None:
    if OUTPUT_SIZE == CANVAS_SIZE:
        return
    text = open(output_path, "r", encoding="utf-8").read()
    text = re.sub(r'width="[^"]+"', f'width="{OUTPUT_SIZE}"', text, count=1)
    text = re.sub(r'height="[^"]+"', f'height="{OUTPUT_SIZE}"', text, count=1)
    open(output_path, "w", encoding="utf-8").write(text)


def compose_and_save_svg(emotion: str, block_params: list[dict], metaball_params: list[dict], output_path: str) -> None:
    os.makedirs(os.path.dirname(output_path) if os.path.dirname(output_path) else ".", exist_ok=True)
    surface = cairo.SVGSurface(output_path, CANVAS_SIZE, CANVAS_SIZE)
    _paint_visual(surface, emotion, block_params, metaball_params)
    surface.finish()
    _set_svg_display_size(output_path)
