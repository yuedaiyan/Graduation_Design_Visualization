from __future__ import annotations

import hashlib

import numpy as np


def stable_seed(text: str, base_seed: int) -> int:
    digest = hashlib.sha256(f"{base_seed}:{text}".encode("utf-8")).digest()
    return int.from_bytes(digest[:8], "little", signed=False) % (2**32 - 1)


def clamp01(x: float | np.ndarray) -> float | np.ndarray:
    return np.clip(x, 0.0, 1.0)


def normalize_range(value: float, low: float, high: float) -> float:
    if high <= low:
        return 0.0
    return float(clamp01((value - low) / (high - low)))


def normalize_rows(x: np.ndarray) -> np.ndarray:
    norms = np.linalg.norm(x, axis=1, keepdims=True)
    return x / np.clip(norms, 1e-12, None)

def rgba(color: np.ndarray, alpha: float = 1.0) -> tuple[int, int, int, int]:
    rgb = np.clip(color, 0, 255).astype(int)
    return int(rgb[0]), int(rgb[1]), int(rgb[2]), int(clamp01(alpha) * 255)


def blend(a: np.ndarray, b: np.ndarray, t: float) -> np.ndarray:
    return a * (1.0 - t) + b * t
