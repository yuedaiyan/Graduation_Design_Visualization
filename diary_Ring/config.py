from __future__ import annotations

import math

import numpy as np

CANVAS_PX = 2160
SUPERSAMPLE = 2
BG_COLOR = (184, 222, 246, 255)
RING_COLOR = np.array([58, 40, 38], dtype=np.float32)
SUMMARY_NAME = "summary.json"
TAU = 2 * math.pi
DISTINCTIVENESS_CACHE_VERSION = "v2"

EMOTION_COLORS: dict[str, tuple[tuple[int, int, int], tuple[int, int, int]]] = {
    # 核心情绪群（冷紫蓝 / 深蓝）
    "孤独": ((132, 148, 196), (38, 52, 140)),
    "失落": ((148, 166, 202), (64, 82, 150)),
    "疲惫": ((162, 170, 192), (96, 100, 138)),
    "感慨": ((172, 160, 200), (168, 96, 134)),
    "愤懑": ((138, 154, 188), (140, 38, 52)),
    # 中性波动（蓝青 / 雾蓝）
    "困惑": ((150, 196, 212), (70, 140, 160)),
    "矛盾": ((178, 174, 220), (170, 82, 180)),
    "焦虑": ((158, 188, 224), (232, 108, 56)),
    # 远端释放（明亮天蓝）
    "平静": ((168, 218, 232), (56, 176, 200)),
    "兴奋": ((190, 222, 242), (248, 72, 52)),
    "满足": ((212, 232, 244), (108, 184, 86)),
}

# Visual tuning knobs:
# 调低，比如 0.5、0.7，整体会更偏线条。
# 调高，比如 1.2、1.5，发散窗口会更容易散成粒子。
CONCRETE_DIVERGENT_IMPACT = 1.0

# 0.0 gives hard boundaries between windows, 1.0 keeps the original smooth blend.
# 0.0 是窗口之间硬切。
# 1.0 是完整平滑过渡。
WINDOW_TRANSITION_SMOOTHING = 1.0

# 0.0 keeps the background nearly flat, 1.0 uses the emotion arc normally,
# and values above 1.0 make the background particles/gradients more active.
# emotion_arc 决定背景颗粒变化
BACKGROUND_EMOTION_VARIATION = 2.0
