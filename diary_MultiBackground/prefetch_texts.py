"""Pre-download and cache all background text corpora without vectorizing.

Run this once to build all JSONL text caches.  The main pipeline will then
skip the download step and go straight to GPU embedding.

Usage:
    python3 diary_MultiBackground/prefetch_texts.py
"""

from __future__ import annotations

import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from diary_MultiBackground.config import (
    BACKGROUND_SPECS,
    CACHE_DIR,
    ENABLED_BACKGROUNDS,
    TEXT_CACHE_TARGET_ITEMS,
)
from diary_MultiBackground.pipeline import (
    SCRIPT_DIR,
    _fmt_time,
    collect_texts,
    enabled_background_keys,
)


def main() -> None:
    cache_dir = SCRIPT_DIR / CACHE_DIR
    bg_keys = enabled_background_keys()
    n = len(bg_keys)

    print(f"\n{'═' * 62}")
    print(f"  Prefetch texts — {TEXT_CACHE_TARGET_ITEMS:,} items/background")
    print(f"  Backgrounds : {n}")
    print(f"{'═' * 62}\n")

    ok, skipped, errors = 0, 0, 0
    t_wall = time.perf_counter()

    for i, bg_key in enumerate(bg_keys, 1):
        spec = BACKGROUND_SPECS[bg_key]
        print(f"[{i}/{n}] {bg_key} ...", end="", flush=True)
        t = time.perf_counter()
        try:
            tc = collect_texts(bg_key, spec, cache_dir)
            elapsed = time.perf_counter() - t
            print(f"  {len(tc.texts)} items  {tc.byte_size / 1024 / 1024:.1f} MB  {_fmt_time(elapsed)}")
            ok += 1
        except Exception as exc:
            elapsed = time.perf_counter() - t
            print(f"  FAIL ({_fmt_time(elapsed)}): {type(exc).__name__}: {exc}")
            errors += 1

    total = time.perf_counter() - t_wall
    print(f"\n{'═' * 62}")
    print(f"  Done: {ok} ok, {errors} errors  —  {_fmt_time(total)}")
    print(f"{'═' * 62}\n")


if __name__ == "__main__":
    main()
