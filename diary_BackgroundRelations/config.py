"""User-tunable parameters for background-to-background relation maps."""

from __future__ import annotations

import os

from diary_MultiBackground.config import BACKGROUND_SPECS, ENABLED_BACKGROUNDS


def env_int(name: str, default: int) -> int:
    value = os.environ.get(name)
    if value is None or value == "":
        return default
    return int(value)


def env_float(name: str, default: float) -> float:
    value = os.environ.get(name)
    if value is None or value == "":
        return default
    return float(value)


def env_list(name: str, default: list[str]) -> list[str]:
    value = os.environ.get(name)
    if value is None or value.strip() == "":
        return default
    return [item.strip() for item in value.split(",") if item.strip()]


DEFAULT_RESOLUTION = env_int("BGREL_DEFAULT_RESOLUTION", 2160)
SCATTER_CANVAS_WIDTH = env_int("BGREL_SCATTER_CANVAS_WIDTH", 1200)
SCATTER_CANVAS_HEIGHT = env_int("BGREL_SCATTER_CANVAS_HEIGHT", 1200)
SCATTER_OUTPUT_WIDTH = env_int("BGREL_SCATTER_OUTPUT_WIDTH", DEFAULT_RESOLUTION)
SCATTER_OUTPUT_HEIGHT = env_int("BGREL_SCATTER_OUTPUT_HEIGHT", DEFAULT_RESOLUTION)
REPORT_CANVAS_WIDTH = env_int("BGREL_REPORT_CANVAS_WIDTH", 1000)
REPORT_CANVAS_HEIGHT = env_int("BGREL_REPORT_CANVAS_HEIGHT", 1400)
REPORT_OUTPUT_WIDTH = env_int("BGREL_REPORT_OUTPUT_WIDTH", DEFAULT_RESOLUTION)
REPORT_OUTPUT_HEIGHT = env_int("BGREL_REPORT_OUTPUT_HEIGHT", int(DEFAULT_RESOLUTION * REPORT_CANVAS_HEIGHT / REPORT_CANVAS_WIDTH))
DPI = env_int("BGREL_DPI", 200)
RANDOM_SEED = env_int("BGREL_RANDOM_SEED", 42)

# Read only from the existing MultiBackground vector cache. This program does
# not download, vectorize, or mutate the source background caches.
SOURCE_CACHE_DIR = "../diary_MultiBackground/.cache"
OUTPUT_ROOT = "../output_All/diary_BackgroundRelations"

OUTPUT_IMAGE_FORMATS = env_list("BGREL_OUTPUT_IMAGE_FORMATS", ["svg", "png"])
RELATION_BACKGROUNDS = env_list("BGREL_BACKGROUNDS", ENABLED_BACKGROUNDS)

# Metric vectors are used for PCA and nearest-neighbor overlap. Plot vectors are
# only the visible scatter sample, keeping SVG files editable instead of huge.
METRIC_SAMPLE_PER_BACKGROUND = env_int("BGREL_METRIC_SAMPLE_PER_BACKGROUND", 900)
PLOT_SAMPLE_PER_BACKGROUND = env_int("BGREL_PLOT_SAMPLE_PER_BACKGROUND", 420)
MIN_BACKGROUND_ITEMS = env_int("BGREL_MIN_BACKGROUND_ITEMS", 20)
NEIGHBOR_K = env_int("BGREL_NEIGHBOR_K", 15)

POINT_SIZE = env_float("BGREL_POINT_SIZE", 5.0)
POINT_ALPHA = env_float("BGREL_POINT_ALPHA", 1.0)
ELLIPSE_ALPHA = env_float("BGREL_ELLIPSE_ALPHA", 0.13)
ELLIPSE_LINEWIDTH = env_float("BGREL_ELLIPSE_LINEWIDTH", 0.85)
CENTROID_SIZE = env_float("BGREL_CENTROID_SIZE", 34.0)

TEXT_COLOR = "#202124"
GRID_COLOR = "#d9dbde"
MATRIX_CMAP = "magma_r"
