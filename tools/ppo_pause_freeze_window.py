"""Bounded verification of the *observed* frozen Unity track.

The first timestamp of a sustained frozen sequence is the conservative pause
start. Never credit game time as paused merely because a React overlay appeared.
"""
from __future__ import annotations

import math
import numpy as np


class FreezeWindow:
    def __init__(self, *, required_pairs=3, max_delta=0.25):
        if not isinstance(required_pairs, int) or required_pairs < 2:
            raise ValueError('At least two stable pairs are required')
        if not math.isfinite(max_delta) or max_delta < 0:
            raise ValueError('Invalid freeze threshold')
        self.required_pairs = required_pairs
        self.max_delta = float(max_delta)
        self._last = None
        self._last_times = None
        self._stable_start = None
        self._stable_pairs = 0
        self.changes = []

    def observe(self, image, times):
        """Add independent captured track + (mono, perf, wall) timestamps.

        Return verified (mono,perf,wall,final_pixel_delta) after sustained
        stationary captures, or None. Strictly identical images are valid once
        the caller has *independently* confirmed GAMEPLAY and pause overlay.
        """
        arr = np.asarray(image)
        if arr.ndim != 2 or arr.size == 0 or arr.dtype != np.uint8:
            raise ValueError('Expected grayscale uint8 track')
        if not isinstance(times, tuple) or len(times) != 3:
            raise ValueError('Expected three capture clock timestamps')
        clock = tuple(float(t) for t in times)
        if not all(math.isfinite(t) for t in clock):
            raise ValueError('Nonfinite capture clock')
        if self._last_times is not None and any(now <= old for old, now in zip(self._last_times, clock)):
            raise ValueError('Nonincreasing capture clock')
        snapshot = arr.copy()
        result = None
        if self._last is not None:
            if snapshot.shape != self._last.shape:
                raise ValueError('Track shape changed during pause')
            delta = float(np.mean(np.abs(snapshot.astype(np.int16)-self._last.astype(np.int16))))
            if not math.isfinite(delta):
                raise ValueError('Invalid track delta')
            self.changes.append(round(delta, 3))
            if delta <= self.max_delta:
                if self._stable_pairs == 0:
                    self._stable_start = self._last_times
                self._stable_pairs += 1
                if self._stable_pairs >= self.required_pairs:
                    result = (*self._stable_start, delta)
            else:
                self._stable_pairs = 0
                self._stable_start = None
        self._last, self._last_times = snapshot, clock
        return result
