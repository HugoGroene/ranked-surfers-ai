#!/usr/bin/env python3
"""Nine-class V5.4 segmentation -> PPO numerical observation (not action rules).

Coordinates are normalized to the ORIGINAL cropped screenshot. 'upper/middle/lower'
are image bands only; they are NOT calibrated lanes, time-to-collision, or safety.
Each class/band has two slots. Empty or stale slots are zero-padded.
The raw screenshot history remains a separate PPO observation.
"""
from __future__ import annotations

from collections import defaultdict
import time
from typing import Mapping

import numpy as np

from rs_v5_shadow_segmentation import V5_CLASSES, distance_band

IMAGE_BANDS = ('upper_frame', 'middle_frame', 'lower_frame')
SLOTS_PER_BAND = 2
VALUES_PER_SLOT = 8
ACTION_COUNT = 5
HISTORY = 4
INTERVALS = 3
SLOT_COUNT = len(V5_CLASSES) * len(IMAGE_BANDS) * SLOTS_PER_BAND
HAZARD_DIM = SLOT_COUNT * VALUES_PER_SLOT + len(V5_CLASSES) + 1
AUX_DIM = HAZARD_DIM + HISTORY * ACTION_COUNT + INTERVALS + 2
FEATURE_VERSION = 'v6_seg9_3bands_2slots8_v1'


def _polygon_area(poly):
    """Area of a normalized simple polygon. Zero for absent/malformed masks."""
    try:
        points = np.asarray(poly, dtype=np.float32)
        if points.ndim != 2 or points.shape[1] != 2 or len(points) < 3:
            return 0.0
        if not np.isfinite(points).all():
            return 0.0
        x, y = points[:, 0], points[:, 1]
        return float(np.clip(abs(np.sum(x * np.roll(y, -1) - y * np.roll(x, -1))) * 0.5, 0., 1.))
    except (ValueError, TypeError):
        return 0.0


def _valid_box(obj):
    box = obj.get('box')
    if box is None or len(box) != 4:
        return None
    try:
        x1, y1, x2, y2 = [float(v) for v in box]
    except (ValueError, TypeError):
        return None
    if not np.isfinite([x1, y1, x2, y2]).all():
        return None
    x1, y1, x2, y2 = np.clip([x1, y1, x2, y2], 0, 1)
    if x2 <= x1 or y2 <= y1:
        return None
    return (float(x1), float(y1), float(x2), float(y2))


def encode_hazards(snapshot: Mapping | None, *, now: float | None = None,
                   episode_tag: int | None = None, max_stale_ms: float = 300.):
    """Returns (feature vector, detection age in ms or None, fresh flag).

    Layout: per class -> upper/middle/lower image band -> two objects ordered by
    descending bounding-box bottom y -> [x1,y1,x2,y2,confidence,polygon_area,
    box_area,present]; then class counts / 12 (clamped); then fresh flag.
    Polygons are used for area, while bounding boxes remain useful geometry.
    Never maps a class or location to an action, collision reward, or safe lane.
    """
    empty = np.zeros(HAZARD_DIM, dtype=np.float32)
    if now is None:
        now = time.perf_counter()
    if snapshot is None:
        return empty, None, False
    if episode_tag is not None and int(snapshot.get('episode', -1)) != int(episode_tag):
        return empty, None, False
    captured_at = float(snapshot['captured_at'])
    age_ms = max(0., (float(now) - captured_at) * 1000.)
    if not np.isfinite(age_ms) or age_ms > max_stale_ms:
        return empty, age_ms, False

    bins = defaultdict(list)
    counts = np.zeros(len(V5_CLASSES), dtype=np.float32)
    for obj in snapshot.get('detections', []):
        name = str(obj.get('class', ''))
        if name not in V5_CLASSES:
            raise ValueError(f'Unexpected V5 class {name!r}. Model/schema mismatch.')
        box = _valid_box(obj)
        if box is None:
            continue
        conf = float(obj.get('conf', 0.))
        if not np.isfinite(conf):
            continue
        conf = float(np.clip(conf, 0, 1))
        band = distance_band(box[3])
        bins[(name, band)].append((box, conf, obj))
        counts[V5_CLASSES.index(name)] += 1

    slots = np.zeros((SLOT_COUNT, VALUES_PER_SLOT), dtype=np.float32)
    index = 0
    for name in V5_CLASSES:
        for band in IMAGE_BANDS:
            # Nearest screen-bottom boxes first. At most two per class and band.
            objs = sorted(bins[(name, band)], key=lambda item: (item[0][3], item[1]),
                          reverse=True)
            for box, conf, obj in objs[:SLOTS_PER_BAND]:
                x1, y1, x2, y2 = box
                slots[index] = [x1, y1, x2, y2, conf,
                                _polygon_area(obj.get('polygon', [])),
                                (x2-x1)*(y2-y1), 1.]
                index += 1
            index += SLOTS_PER_BAND - min(SLOTS_PER_BAND, len(objs))
    vec = np.concatenate([slots.reshape(-1), np.clip(counts / 12., 0, 1),
                          np.array([1.], dtype=np.float32)])
    if vec.shape != (HAZARD_DIM,) or not np.isfinite(vec).all():
        raise RuntimeError(f'V6 hazard features are invalid: {vec.shape}')
    return vec, age_ms, True


def encode_aux(hazards: np.ndarray, action_history, episode_seconds: float,
               frame_intervals, detection_age_ms: float | None):
    """Add action/timing context; every value in [0,1]."""
    hazards = np.asarray(hazards, dtype=np.float32)
    if hazards.shape != (HAZARD_DIM,):
        raise ValueError(f'Expected {HAZARD_DIM} hazard values, got {hazards.shape}')
    history = list(action_history)[-HISTORY:]
    actions = np.zeros((HISTORY, ACTION_COUNT), dtype=np.float32)
    for i, action in enumerate(history):
        act = int(action)
        if not 0 <= act < ACTION_COUNT:
            raise ValueError(f'Bad action {act}')
        actions[HISTORY-len(history)+i, act] = 1.
    recent_intervals = list(frame_intervals)[-INTERVALS:]
    intervals = np.zeros(INTERVALS, dtype=np.float32)
    for i, seconds in enumerate(recent_intervals):
        intervals[INTERVALS-len(recent_intervals)+i] = float(np.clip(seconds / .5, 0, 1))
    timing = np.array([np.clip(episode_seconds / 60., 0, 1),
                       np.clip((600. if detection_age_ms is None else detection_age_ms) / 600., 0, 1)],
                      dtype=np.float32)
    result = np.concatenate((hazards, actions.ravel(), intervals, timing))
    if result.shape != (AUX_DIM,) or not np.isfinite(result).all():
        raise RuntimeError(f'V6 aux vector invalid: {result.shape}')
    return result


def shape_description():
    return {'feature_version': FEATURE_VERSION, 'classes': list(V5_CLASSES),
            'image_bands': list(IMAGE_BANDS), 'slots_per_band': SLOTS_PER_BAND,
            'values_per_slot': VALUES_PER_SLOT, 'hazard_dim': HAZARD_DIM,
            'aux_dim': AUX_DIM, 'raw_rgb_history_frames': HISTORY,
            'note': 'Image bands are not lanes or distances. No scripted movement or rewards.'}
