"""Static interactive point viewer output for MultiBackground runs."""

from __future__ import annotations

import json
import re
from pathlib import Path
from typing import Any

import numpy as np


def _compact_value(value: Any, limit: int = 240) -> str:
    if value is None:
        return ""
    if isinstance(value, (list, tuple)):
        text = ", ".join(_compact_value(item, 80) for item in value if item is not None)
    elif isinstance(value, dict):
        text = json.dumps(value, ensure_ascii=False, sort_keys=True)
    else:
        text = str(value)
    return re.sub(r"\s+", " ", text).strip()[:limit]


def load_diary_text_lookup(path: Path, diary_labels: list[str]) -> dict[str, dict[str, Any]]:
    if not path.exists():
        return {}
    data = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(data, list):
        return {}

    lookup: dict[str, dict[str, Any]] = {}
    if len(data) == len(diary_labels):
        rows = zip(diary_labels, data)
    else:
        buckets: dict[str, list[dict[str, Any]]] = {}
        for row in data:
            if isinstance(row, dict):
                date = str(row.get("date") or "")
                if date:
                    buckets.setdefault(date, []).append(row)
        rows = []
        for label in diary_labels:
            date = label.split("_", 1)[0]
            bucket = buckets.get(date) or []
            row = bucket.pop(0) if bucket else {}
            rows.append((label, row))

    for label, row in rows:
        if not isinstance(row, dict):
            row = {}
        content = str(row.get("content") or "")
        lookup[label] = {
            "text": content,
            "date": str(row.get("date") or label),
            "location": _compact_value(row.get("location")),
            "time_of_day": _compact_value(row.get("time_of_day")),
            "tags": _compact_value(row.get("tags")),
            "people_tags": _compact_value(row.get("people_tags")),
        }
    return lookup


def _point_id(prefix: str, index: int) -> str:
    return f"{prefix}_{index + 1:05d}"


def _meta_preview(meta: dict[str, Any]) -> dict[str, str]:
    source_meta = meta.get("source_meta") if isinstance(meta.get("source_meta"), dict) else {}
    return {
        "stratum": _compact_value(meta.get("stratum")),
        "length_bucket": _compact_value(meta.get("length_bucket")),
        "chunk_length": _compact_value(meta.get("chunk_length")),
        "raw_length": _compact_value(meta.get("raw_length")),
        "chunk_index": _compact_value(meta.get("chunk_index")),
        "chunk_count": _compact_value(meta.get("chunk_count")),
        "source_name": _compact_value(source_meta.get("source_name")),
        "source_kind": _compact_value(source_meta.get("source_kind")),
        "file": _compact_value(source_meta.get("file"), 360),
        "url": _compact_value(source_meta.get("url"), 360),
        "author": _compact_value(source_meta.get("author")),
        "category": _compact_value(source_meta.get("category") or source_meta.get("label")),
    }


def write_viewer_background_data(
    out_root: Path,
    bg_key: str,
    title: str,
    background_xy: np.ndarray,
    diary_xy: np.ndarray,
    background_labels: list[str],
    background_texts: list[str],
    background_metas: list[dict[str, Any]],
    diary_labels: list[str],
    diary_text_lookup: dict[str, dict[str, Any]],
):
    data_dir = out_root / "viewer_data"
    data_dir.mkdir(parents=True, exist_ok=True)

    points: list[dict[str, Any]] = []
    for i, (label, xy) in enumerate(zip(background_labels, background_xy)):
        text = background_texts[i] if i < len(background_texts) else ""
        meta = background_metas[i] if i < len(background_metas) else {}
        if not isinstance(meta, dict):
            meta = {}
        points.append(
            {
                "id": _point_id("bg", i),
                "kind": "background",
                "label": str(label),
                "x": round(float(xy[0]), 8),
                "y": round(float(xy[1]), 8),
                "text": text,
                "meta": _meta_preview(meta),
            }
        )

    for i, (label, xy) in enumerate(zip(diary_labels, diary_xy)):
        diary_info = diary_text_lookup.get(label, {})
        points.append(
            {
                "id": _point_id("diary", i),
                "kind": "diary",
                "label": str(label),
                "x": round(float(xy[0]), 8),
                "y": round(float(xy[1]), 8),
                "text": str(diary_info.get("text") or ""),
                "meta": {
                    "date": _compact_value(diary_info.get("date") or label),
                    "location": _compact_value(diary_info.get("location")),
                    "time_of_day": _compact_value(diary_info.get("time_of_day")),
                    "tags": _compact_value(diary_info.get("tags")),
                    "people_tags": _compact_value(diary_info.get("people_tags")),
                },
            }
        )

    payload = {
        "key": bg_key,
        "title": title,
        "background_count": int(len(background_xy)),
        "diary_count": int(len(diary_xy)),
        "points": points,
    }
    js = (
        "window.MULTIBG_VIEWER_DATA = window.MULTIBG_VIEWER_DATA || {};\n"
        f"window.MULTIBG_VIEWER_DATA[{json.dumps(bg_key, ensure_ascii=False)}] = "
        f"{json.dumps(payload, ensure_ascii=False, separators=(',', ':'))};\n"
    )
    (data_dir / f"{bg_key}.js").write_text(js, encoding="utf-8")


def write_viewer_shell(out_root: Path, backgrounds: list[dict[str, Any]]):
    data_dir = out_root / "viewer_data"
    data_dir.mkdir(parents=True, exist_ok=True)
    manifest = [
        {
            "key": item["bg_key"],
            "title": item.get("title") or item["bg_key"],
            "n_items": int(item.get("n_items") or 0),
            "file": f"viewer_data/{item['bg_key']}.js",
        }
        for item in backgrounds
    ]
    manifest_js = (
        "window.MULTIBG_VIEWER_MANIFEST = "
        f"{json.dumps(manifest, ensure_ascii=False, indent=2)};\n"
    )
    (data_dir / "manifest.js").write_text(manifest_js, encoding="utf-8")
    (out_root / "viewer.html").write_text(_viewer_html(), encoding="utf-8")


def _viewer_html() -> str:
    return """<!doctype html>
<html lang="zh-CN">
<head>
  <meta charset="utf-8">
  <meta name="viewport" content="width=device-width, initial-scale=1">
  <title>MultiBackground Point Viewer</title>
  <script src="viewer_data/manifest.js"></script>
  <style>
    :root {
      color-scheme: light;
      --bg: #ffffff;
      --ink: #222426;
      --muted: #6d737a;
      --line: #d8dde2;
      --panel: #f6f7f8;
      --accent: #207a78;
      --diary: #242426;
      --context: #e28772;
    }
    * { box-sizing: border-box; }
    body {
      margin: 0;
      font-family: -apple-system, BlinkMacSystemFont, "Segoe UI", sans-serif;
      color: var(--ink);
      background: var(--bg);
      overflow: hidden;
    }
    .app {
      height: 100vh;
      display: grid;
      grid-template-columns: minmax(0, 1fr) 380px;
      grid-template-rows: 46px minmax(0, 1fr);
    }
    .toolbar {
      grid-column: 1 / -1;
      display: flex;
      align-items: center;
      gap: 10px;
      padding: 7px 10px;
      border-bottom: 1px solid var(--line);
      background: #fbfbfc;
    }
    select, input, button {
      height: 30px;
      border: 1px solid var(--line);
      background: white;
      color: var(--ink);
      border-radius: 4px;
      padding: 0 9px;
      font: inherit;
    }
    select { min-width: 220px; }
    input { min-width: 260px; }
    button { cursor: pointer; }
    .stat { color: var(--muted); font-size: 12px; white-space: nowrap; }
    .canvas-wrap {
      position: relative;
      min-width: 0;
      min-height: 0;
      background: white;
    }
    canvas {
      display: block;
      width: 100%;
      height: 100%;
      cursor: crosshair;
    }
    .tooltip {
      position: absolute;
      pointer-events: none;
      max-width: 360px;
      padding: 8px 9px;
      border: 1px solid var(--line);
      background: rgba(255, 255, 255, 0.96);
      box-shadow: 0 8px 22px rgba(25, 30, 35, 0.12);
      font-size: 12px;
      line-height: 1.45;
      display: none;
    }
    .side {
      min-width: 0;
      min-height: 0;
      border-left: 1px solid var(--line);
      background: var(--panel);
      display: grid;
      grid-template-rows: auto minmax(0, 1fr);
    }
    .side-head {
      padding: 12px 14px;
      border-bottom: 1px solid var(--line);
      background: #fff;
    }
    .kind {
      display: inline-block;
      margin-bottom: 8px;
      padding: 2px 6px;
      border-radius: 4px;
      color: white;
      background: var(--context);
      font-size: 12px;
    }
    .kind.diary { background: var(--diary); }
    h1 {
      margin: 0;
      font-size: 16px;
      line-height: 1.35;
      font-weight: 650;
      word-break: break-word;
    }
    .meta {
      padding: 12px 14px;
      overflow: auto;
      font-size: 13px;
      line-height: 1.55;
    }
    .meta-row {
      display: grid;
      grid-template-columns: 86px minmax(0, 1fr);
      gap: 8px;
      padding: 5px 0;
      border-bottom: 1px solid #e6eaee;
    }
    .meta-row b { color: var(--muted); font-weight: 500; }
    pre {
      margin: 12px 0 0;
      white-space: pre-wrap;
      word-break: break-word;
      font: 13px/1.62 ui-monospace, SFMono-Regular, Menlo, Consolas, monospace;
      background: white;
      border: 1px solid var(--line);
      padding: 10px;
      border-radius: 4px;
    }
    .empty { color: var(--muted); }
    @media (max-width: 900px) {
      .app { grid-template-columns: 1fr; grid-template-rows: auto minmax(45vh, 1fr) 45vh; }
      .toolbar { flex-wrap: wrap; height: auto; }
      .side { border-left: 0; border-top: 1px solid var(--line); }
    }
  </style>
</head>
<body>
  <div class="app">
    <div class="toolbar">
      <select id="bgSelect"></select>
      <input id="search" placeholder="搜索日期、标题、正文、来源">
      <button id="reset">重置视图</button>
      <span id="stat" class="stat"></span>
    </div>
    <div class="canvas-wrap" id="wrap">
      <canvas id="plot"></canvas>
      <div class="tooltip" id="tip"></div>
    </div>
    <aside class="side">
      <div class="side-head">
        <span id="kind" class="kind">未选择</span>
        <h1 id="title">点击一个点查看内容</h1>
      </div>
      <div class="meta" id="detail">
        <p class="empty">滚轮缩放，拖动画布平移。鼠标悬停可以预览，点击后右侧显示完整信息。</p>
      </div>
    </aside>
  </div>
  <script>
    window.MULTIBG_VIEWER_DATA = window.MULTIBG_VIEWER_DATA || {};
    const manifest = window.MULTIBG_VIEWER_MANIFEST || [];
    const select = document.getElementById('bgSelect');
    const search = document.getElementById('search');
    const resetBtn = document.getElementById('reset');
    const stat = document.getElementById('stat');
    const canvas = document.getElementById('plot');
    const wrap = document.getElementById('wrap');
    const ctx = canvas.getContext('2d');
    const tip = document.getElementById('tip');
    const kindEl = document.getElementById('kind');
    const titleEl = document.getElementById('title');
    const detailEl = document.getElementById('detail');
    let current = null;
    let view = { scale: 1, tx: 0, ty: 0 };
    let bounds = null;
    let filtered = [];
    let hovered = null;
    let selected = null;
    let dragging = false;
    let lastMouse = null;

    for (const item of manifest) {
      const option = document.createElement('option');
      option.value = item.key;
      option.textContent = `${item.key} (${item.n_items})`;
      select.appendChild(option);
    }

    function escapeHtml(value) {
      return String(value || '').replace(/[&<>"']/g, ch => ({
        '&': '&amp;', '<': '&lt;', '>': '&gt;', '"': '&quot;', "'": '&#39;'
      }[ch]));
    }

    function loadScript(src) {
      return new Promise((resolve, reject) => {
        const existing = document.querySelector(`script[data-src="${src}"]`);
        if (existing) return resolve();
        const script = document.createElement('script');
        script.src = src;
        script.dataset.src = src;
        script.onload = resolve;
        script.onerror = reject;
        document.head.appendChild(script);
      });
    }

    async function loadBackground(key) {
      const item = manifest.find(x => x.key === key);
      if (!item) return;
      stat.textContent = '加载中...';
      if (!window.MULTIBG_VIEWER_DATA[key]) {
        await loadScript(item.file);
      }
      current = window.MULTIBG_VIEWER_DATA[key];
      selected = null;
      hovered = null;
      computeBounds();
      resetView();
      applySearch();
    }

    function computeBounds() {
      const xs = current.points.map(p => p.x);
      const ys = current.points.map(p => p.y);
      bounds = {
        minX: Math.min(...xs), maxX: Math.max(...xs),
        minY: Math.min(...ys), maxY: Math.max(...ys)
      };
      const padX = (bounds.maxX - bounds.minX || 1) * 0.06;
      const padY = (bounds.maxY - bounds.minY || 1) * 0.06;
      bounds.minX -= padX; bounds.maxX += padX;
      bounds.minY -= padY; bounds.maxY += padY;
    }

    function resize() {
      const dpr = window.devicePixelRatio || 1;
      const rect = wrap.getBoundingClientRect();
      canvas.width = Math.max(1, Math.floor(rect.width * dpr));
      canvas.height = Math.max(1, Math.floor(rect.height * dpr));
      canvas.style.width = `${rect.width}px`;
      canvas.style.height = `${rect.height}px`;
      ctx.setTransform(dpr, 0, 0, dpr, 0, 0);
      draw();
    }

    function baseScale() {
      const w = canvas.clientWidth || 1;
      const h = canvas.clientHeight || 1;
      return Math.min(w / (bounds.maxX - bounds.minX || 1), h / (bounds.maxY - bounds.minY || 1));
    }

    function toScreen(point) {
      const s = baseScale() * view.scale;
      const w = canvas.clientWidth || 1;
      const h = canvas.clientHeight || 1;
      const x = (point.x - bounds.minX) * s + view.tx;
      const y = h - (point.y - bounds.minY) * s + view.ty;
      return [x, y];
    }

    function resetView() {
      view = { scale: 1, tx: 0, ty: 0 };
      draw();
    }

    function applySearch() {
      const q = search.value.trim().toLowerCase();
      if (!q) {
        filtered = current ? current.points : [];
      } else {
        filtered = current.points.filter(p => {
          const hay = `${p.kind} ${p.label} ${p.text} ${Object.values(p.meta || {}).join(' ')}`.toLowerCase();
          return hay.includes(q);
        });
      }
      stat.textContent = current
        ? `${current.title} | 背景 ${current.background_count} | 日记 ${current.diary_count} | 当前 ${filtered.length}`
        : '';
      draw();
    }

    function drawPoint(point, x, y, radius, color, alpha) {
      ctx.globalAlpha = alpha;
      ctx.fillStyle = color;
      ctx.beginPath();
      ctx.arc(x, y, radius, 0, Math.PI * 2);
      ctx.fill();
      ctx.globalAlpha = 1;
    }

    function draw() {
      if (!current || !bounds) return;
      const w = canvas.clientWidth || 1;
      const h = canvas.clientHeight || 1;
      ctx.clearRect(0, 0, w, h);
      ctx.fillStyle = '#fff';
      ctx.fillRect(0, 0, w, h);
      for (const p of filtered) {
        if (p.kind !== 'background') continue;
        const [x, y] = toScreen(p);
        drawPoint(p, x, y, 1.5, '#d88672', 0.95);
      }
      for (const p of filtered) {
        if (p.kind !== 'diary') continue;
        const [x, y] = toScreen(p);
        drawPoint(p, x, y, 2.2, '#242426', 0.96);
      }
      for (const p of [hovered, selected]) {
        if (!p) continue;
        const [x, y] = toScreen(p);
        ctx.strokeStyle = p.kind === 'diary' ? '#111' : '#a94f3c';
        ctx.lineWidth = 1.4;
        ctx.beginPath();
        ctx.arc(x, y, 6.5, 0, Math.PI * 2);
        ctx.stroke();
      }
    }

    function nearestPoint(clientX, clientY) {
      if (!current) return null;
      const rect = canvas.getBoundingClientRect();
      const mx = clientX - rect.left;
      const my = clientY - rect.top;
      let best = null;
      let bestD = Infinity;
      const limit = Math.max(8, 10 / Math.sqrt(view.scale));
      for (const p of filtered) {
        const [x, y] = toScreen(p);
        const d = Math.hypot(x - mx, y - my);
        if (d < bestD) {
          best = p;
          bestD = d;
        }
      }
      return bestD <= limit ? best : null;
    }

    function showTip(point, clientX, clientY) {
      if (!point) {
        tip.style.display = 'none';
        return;
      }
      const text = point.text ? point.text.slice(0, 160) : '无正文';
      tip.innerHTML = `<b>${escapeHtml(point.kind === 'diary' ? '日记' : '背景')}</b> ${escapeHtml(point.label)}<br>${escapeHtml(text)}`;
      const rect = wrap.getBoundingClientRect();
      tip.style.left = `${Math.min(clientX - rect.left + 14, rect.width - 370)}px`;
      tip.style.top = `${Math.min(clientY - rect.top + 14, rect.height - 120)}px`;
      tip.style.display = 'block';
    }

    function selectPoint(point) {
      selected = point;
      if (!point) return;
      kindEl.textContent = point.kind === 'diary' ? '日记点' : '背景语境点';
      kindEl.className = `kind ${point.kind === 'diary' ? 'diary' : ''}`;
      titleEl.textContent = point.label;
      const rows = Object.entries(point.meta || {})
        .filter(([, value]) => value)
        .map(([key, value]) => `<div class="meta-row"><b>${escapeHtml(key)}</b><span>${escapeHtml(value)}</span></div>`)
        .join('');
      detailEl.innerHTML = `${rows || '<p class="empty">没有额外元数据。</p>'}<pre>${escapeHtml(point.text || '没有正文。')}</pre>`;
      draw();
    }

    canvas.addEventListener('mousemove', event => {
      if (dragging && lastMouse) {
        view.tx += event.clientX - lastMouse.x;
        view.ty += event.clientY - lastMouse.y;
        lastMouse = { x: event.clientX, y: event.clientY };
        draw();
        return;
      }
      hovered = nearestPoint(event.clientX, event.clientY);
      showTip(hovered, event.clientX, event.clientY);
      draw();
    });
    canvas.addEventListener('mouseleave', () => {
      hovered = null;
      tip.style.display = 'none';
      draw();
    });
    canvas.addEventListener('mousedown', event => {
      dragging = true;
      lastMouse = { x: event.clientX, y: event.clientY };
    });
    window.addEventListener('mouseup', () => {
      dragging = false;
      lastMouse = null;
    });
    canvas.addEventListener('click', event => {
      const point = nearestPoint(event.clientX, event.clientY);
      if (point) selectPoint(point);
    });
    canvas.addEventListener('wheel', event => {
      event.preventDefault();
      const factor = event.deltaY < 0 ? 1.18 : 1 / 1.18;
      const rect = canvas.getBoundingClientRect();
      const mx = event.clientX - rect.left;
      const my = event.clientY - rect.top;
      view.tx = mx - (mx - view.tx) * factor;
      view.ty = my - (my - view.ty) * factor;
      view.scale = Math.max(0.25, Math.min(80, view.scale * factor));
      draw();
    }, { passive: false });
    select.addEventListener('change', () => loadBackground(select.value));
    search.addEventListener('input', applySearch);
    resetBtn.addEventListener('click', resetView);
    window.addEventListener('resize', resize);

    resize();
    if (manifest.length) {
      loadBackground(manifest[0].key);
    } else {
      stat.textContent = '没有可查看的背景数据';
    }
  </script>
</body>
</html>
"""
