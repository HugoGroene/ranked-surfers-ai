#!/usr/bin/env python3
"""Observation-only evidence for a previously verified React Pause panel.

The screen classifier has no PAUSED class. Compare a small central crop of
new captures with the visual pause panel recorded after an independent frozen
Unity check; this catches a stale rendered Pause panel after React hides its
DOM node. This signal can ONLY reject a frame, never authorize gameplay.
"""
from __future__ import annotations

import numpy as np


def pause_panel_signature(bgra):
    """Small visual fingerprint of the central local Practice pause dialog.

    Returned data must come from a verified pause screenshot in the SAME run.
    An absent/invalid frame fails closed rather than skipping visual checks.
    """
    if not isinstance(bgra, np.ndarray) or bgra.ndim != 3 or bgra.shape[2] < 3:
        raise ValueError('Missing native BGRA screenshot for pause evidence')
    height, width = bgra.shape[:2]
    if width < 32 or height < 32:
        raise ValueError(f'Pause screenshot too small: {width}x{height}')
    # Local Practice panel is centered over the Unity canvas; 0.42..0.58 and
    # 0.45..0.55 cover the stable dialog and buttons, not the moving track.
    region = bgra[int(height*.45):int(height*.55),
                  int(width*.42):int(width*.58), :3]
    if region.size == 0:
        raise ValueError('Empty pause-panel crop')
    import cv2
    return cv2.resize(region, (96, 48), interpolation=cv2.INTER_AREA).copy()


def matches_prior_paused_panel(bgra, reference, *, max_mean_difference=18.0):
    """Return (matches, mean_pixel_delta), never a positive gameplay signal."""
    if (not isinstance(reference, np.ndarray) or
            reference.shape != (48, 96, 3) or reference.dtype != np.uint8):
        raise ValueError('Missing verified paused-panel visual reference')
    current = pause_panel_signature(bgra)
    mean_delta = float(np.abs(current.astype(np.int16) -
                              reference.astype(np.int16)).mean())
    return mean_delta <= max_mean_difference, mean_delta
