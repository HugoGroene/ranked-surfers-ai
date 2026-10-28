"""V6.1.2 pre-key distance HUD mask for two *observed* screen layouts.

Keep the inherited V5 screenshot and full screen-state classifier unchanged.
The V6 preflight visual-gameplay corroboration requires MOVING TRACK + CHANGING
DISTANCE DIGITS. The inherited meter_mask sees fullscreen-only y=1.1%-4.8%,
but the 2026-10-28 real 1920x1080 browser screenshot has game HUD at
x=10.1%-15.1%, y=9%-13% owing to visible Chrome controls.

A positive pixel mask is never itself permission to act. screen_gate()
requires scene motion, textured track, HUD *changes*, and negative menu evidence.
Fail closed for layouts neither ROI can corroborate.
"""
from __future__ import annotations
import cv2
import numpy as np

# These normalized ROIs are fixed to two documented layouts, NOT a search over
# arbitrary yellow pixels that could be score banners or other overlays.
DISTANCE_X = (.101, .151)
FULLSCREEN_Y = (.011, .048)  # inherited V5 behaviour, unchanged
CHROME_VISIBLE_Y = (.090, .130)  # V1.7.4 diagnostic: 179 m at y=111..121/1080
HSV_LO = np.array([16, 40, 135], dtype=np.uint8)
HSV_HI = np.array([46, 255, 255], dtype=np.uint8)
MASK_SHAPE_WH = (64, 24)
MIN_PIXELS = 8  # screen_gate's existing threshold; not reduced


def _roi_mask(bgra: np.ndarray, y_range: tuple[float, float]) -> np.ndarray:
    if not isinstance(bgra, np.ndarray) or bgra.ndim != 3 or bgra.shape[2] != 4:
        raise ValueError('Expected raw BGRA screenshot')
    h, w = bgra.shape[:2]
    roi = bgra[int(h * y_range[0]):int(h * y_range[1]),
               int(w * DISTANCE_X[0]):int(w * DISTANCE_X[1]), :3]
    if roi.size == 0:
        return np.zeros((MASK_SHAPE_WH[1], MASK_SHAPE_WH[0]), np.uint8)
    hsv = cv2.cvtColor(roi, cv2.COLOR_BGR2HSV)
    mask = cv2.inRange(hsv, HSV_LO, HSV_HI)
    return cv2.resize(mask, MASK_SHAPE_WH, interpolation=cv2.INTER_NEAREST)


def meter_mask(bgra: np.ndarray) -> np.ndarray:
    """Return the original fixed-shape distance HUD mask, with browser fallback.

    Priority stays with original fullscreen HUD. When it has no confirmed
    yellow digit pixels, use the *observed browser-layout distance band*.
    If neither can be validated, return the zero fullscreen mask: the guard
    will BLOCK, exactly as before. Never mix pixels from unrelated regions.
    """
    fullscreen = _roi_mask(bgra, FULLSCREEN_Y)
    if cv2.countNonZero(fullscreen) >= MIN_PIXELS:
        return fullscreen
    browser = _roi_mask(bgra, CHROME_VISIBLE_Y)
    return browser if cv2.countNonZero(browser) >= MIN_PIXELS else fullscreen
