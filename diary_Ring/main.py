#!/usr/bin/env python3
"""Generate diary ring visuals from sentence/window vectors.

Each diary becomes one perfectly circular ring. Semantic windows occupy
contiguous angular sectors. Focused windows render as stable line arcs, while
looser windows dissolve into square particles around the same circular path.

Outputs:
  output_All/diary_Ring/try_N/<date>.png
  output_All/diary_Ring/SVG_try_N/<date>.svg
  output_All/diary_Ring/try_N/summary.json
"""

from __future__ import annotations

import argparse
import json
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path
from typing import Any

from config import (
    BACKGROUND_EMOTION_VARIATION,
    BG_COLOR,
    CANVAS_PX,
    CONCRETE_DIVERGENT_IMPACT,
    EMOTION_COLORS,
    RING_COLOR,
    SUMMARY_NAME,
    WINDOW_TRANSITION_SMOOTHING,
)
from io_data import load_emotion_profiles, load_entries, next_try_dir
from metrics import build_window_metrics, load_diary_distinctiveness
from models import EmotionProfile, Entry
from renderer import render_ring, render_ring_svg
from utils import stable_seed

MAX_RENDER_WORKERS = 10


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Generate circular diary ring visuals")
    parser.add_argument(
        "--root", default=None, help="Project root. Defaults to parent of this script"
    )
    parser.add_argument(
        "--out-dir",
        default=None,
        help="Output base dir. Defaults to <root>/output_All/diary_Ring",
    )
    parser.add_argument(
        "--date", default=None, help="Optional date/stem filter, e.g. 2026-03-10"
    )
    parser.add_argument("--seed", type=int, default=71, help="Base random seed")
    parser.add_argument(
        "--size", type=int, default=CANVAS_PX, help="Square canvas size in pixels"
    )
    parser.add_argument(
        "--particles", type=float, default=1.0, help="Particle density multiplier"
    )
    parser.add_argument(
        "--impact",
        type=float,
        default=CONCRETE_DIVERGENT_IMPACT,
        help="Concrete/divergent visual impact. Lower values make the ring more line-like.",
    )
    parser.add_argument(
        "--smoothing",
        type=float,
        default=WINDOW_TRANSITION_SMOOTHING,
        help="Window transition smoothing from 0.0 hard boundaries to 1.0 full smoothing.",
    )
    parser.add_argument(
        "--background-variation",
        type=float,
        default=BACKGROUND_EMOTION_VARIATION,
        help="Emotion background variation strength. Lower values are calmer.",
    )
    parser.add_argument(
        "--workers",
        type=int,
        default=MAX_RENDER_WORKERS,
        help="Parallel diary renders. Defaults to 5.",
    )
    return parser.parse_args()


def render_entry(
    index: int,
    entry: Entry,
    root: Path,
    out_dir: Path,
    svg_out_dir: Path,
    emotion_profiles: dict[str, EmotionProfile],
    distinctiveness: dict[str, float],
    args: argparse.Namespace,
) -> tuple[int, dict[str, Any], str]:
    metrics, summary = build_window_metrics(root, entry)
    seed = stable_seed(entry.stem, args.seed)
    entry_distinctiveness = distinctiveness.get(entry.stem, 0.45)
    item = render_ring(
        out_path=out_dir / f"{entry.stem}.png",
        metrics=metrics,
        summary=summary,
        distinctiveness=entry_distinctiveness,
        emotion=emotion_profiles.get(
            entry.stem, EmotionProfile(primary="平静", arc="稳定")
        ),
        seed=seed,
        size_px=args.size,
        particle_multiplier=args.particles,
        visual_impact=args.impact,
        smoothing_strength=args.smoothing,
        background_variation=args.background_variation,
    )
    render_ring_svg(
        out_path=svg_out_dir / f"{entry.stem}.svg",
        metrics=metrics,
        summary=summary,
        distinctiveness=entry_distinctiveness,
        seed=seed,
        size_px=args.size,
        particle_multiplier=args.particles,
        visual_impact=args.impact,
        smoothing_strength=args.smoothing,
    )
    item["date"] = entry.date
    item["stem"] = entry.stem
    item["num_chars"] = len(entry.content)
    item["windows"] = [
        {
            "idx": m.idx,
            "start_sentence_idx": m.start,
            "end_sentence_idx": m.end,
            "chars": m.chars,
            "focus": m.focus,
            "looseness": m.looseness,
            "novelty": m.novelty,
        }
        for m in metrics
    ]
    message = (
        f"  {entry.stem}.png  windows={summary['num_windows']} "
        f"variations={item['variation_count']} focus={item['avg_focus']:.3f} "
        f"loose={item['avg_looseness']:.3f}"
    )
    return index, item, message


def main() -> None:
    args = parse_args()
    script_dir = Path(__file__).resolve().parent
    root = Path(args.root).expanduser().resolve() if args.root else script_dir.parent
    out_base = (
        Path(args.out_dir).expanduser().resolve()
        if args.out_dir
        else root / "output_All" / "diary_Ring"
    )
    out_dir = next_try_dir(out_base)
    svg_out_dir = out_base / f"SVG_{out_dir.name}"
    svg_out_dir.mkdir(parents=True, exist_ok=False)

    all_entries = load_entries(root / "diary_entries.json")
    emotion_profiles = load_emotion_profiles(root, all_entries)
    distinctiveness = load_diary_distinctiveness(all_entries, root)

    entries = all_entries
    if args.date:
        entries = [
            entry
            for entry in entries
            if entry.stem == args.date or entry.date == args.date
        ]
    if not entries:
        raise SystemExit(f"No diary entries matched {args.date!r}")

    worker_count = max(1, min(args.workers, MAX_RENDER_WORKERS, len(entries)))
    summary_items: list[dict[str, Any] | None] = [None] * len(entries)

    print(f"输出目录: {out_dir}")
    print(f"SVG输出目录: {svg_out_dir}")
    print(f"并发数量: {worker_count}")

    with ThreadPoolExecutor(max_workers=worker_count) as executor:
        futures = [
            executor.submit(
                render_entry,
                index,
                entry,
                root,
                out_dir,
                svg_out_dir,
                emotion_profiles,
                distinctiveness,
                args,
            )
            for index, entry in enumerate(entries)
        ]
        for future in as_completed(futures):
            index, item, message = future.result()
            summary_items[index] = item
            print(message)

    finalized_items = [item for item in summary_items if item is not None]
    run_summary = {
        "try_dir": str(out_dir),
        "svg_try_dir": str(svg_out_dir),
        "source": {
            "entries": str(root / "diary_entries.json"),
            "analysis": str(root / "diary_analysis_result.json"),
            "sentence_vectors": str(root / "diary_sentence_vectors"),
            "diary_vectors": str(root / "diary_vectors"),
        },
        "rendering": {
            "size_px": args.size,
            "seed": args.seed,
            "particle_multiplier": args.particles,
            "concrete_divergent_impact": args.impact,
            "window_transition_smoothing": args.smoothing,
            "background_emotion_variation": args.background_variation,
            "render_workers": worker_count,
            "fallback_background_rgba": BG_COLOR,
            "emotion_colors": {
                key: {"base": base, "accent": accent}
                for key, (base, accent) in EMOTION_COLORS.items()
            },
            "ring_color_rgb": RING_COLOR.astype(int).tolist(),
        },
        "items": finalized_items,
    }
    with (out_dir / SUMMARY_NAME).open("w", encoding="utf-8") as f:
        json.dump(run_summary, f, ensure_ascii=False, indent=2)

    print(f"完成: {len(finalized_items)} 张图, summary={out_dir / SUMMARY_NAME}")


if __name__ == "__main__":
    main()
