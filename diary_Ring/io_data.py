from __future__ import annotations

import json
import re
from pathlib import Path

from models import EmotionProfile, Entry


def safe_stem(date: str, count: int) -> str:
    base = re.sub(r"[^0-9A-Za-z_-]", "_", date.strip()) or "unknown_date"
    return base if count == 0 else f"{base}_{count + 1}"


def load_entries(path: Path) -> list[Entry]:
    with path.open("r", encoding="utf-8") as f:
        raw = json.load(f)

    entries: list[Entry] = []
    seen: dict[str, int] = {}
    for item in raw:
        if not item.get("date") or not item.get("content"):
            continue
        date = str(item["date"]).strip()
        count = seen.get(date, 0)
        seen[date] = count + 1
        entries.append(
            Entry(stem=safe_stem(date, count), date=date, content=str(item["content"]))
        )
    return entries


def load_emotion_profiles(
    root: Path, entries: list[Entry]
) -> dict[str, EmotionProfile]:
    path = root / "diary_analysis_result.json"
    if not path.exists():
        return {}

    with path.open("r", encoding="utf-8") as f:
        raw = json.load(f)
    if not isinstance(raw, list):
        return {}

    # The current analysis file contains one leading record that is not present
    # in diary_entries.json, so the useful mapping starts at raw[1].
    offset = 1 if len(raw) == len(entries) + 1 else 0
    profiles: dict[str, EmotionProfile] = {}
    for idx, entry in enumerate(entries):
        analysis_idx = idx + offset
        if analysis_idx >= len(raw) or not isinstance(raw[analysis_idx], dict):
            continue
        item = raw[analysis_idx]
        primary = str(item.get("emotion_primary") or "平静")
        arc = str(item.get("emotion_arc") or "稳定")
        profiles[entry.stem] = EmotionProfile(primary=primary, arc=arc)
    return profiles


def next_try_dir(base: Path) -> Path:
    base.mkdir(parents=True, exist_ok=True)
    max_idx = 0
    for child in base.iterdir():
        if not child.is_dir():
            continue
        match = re.fullmatch(r"try_(\d+)", child.name)
        if match:
            max_idx = max(max_idx, int(match.group(1)))
    target = base / f"try_{max_idx + 1}"
    target.mkdir(parents=True, exist_ok=False)
    return target
