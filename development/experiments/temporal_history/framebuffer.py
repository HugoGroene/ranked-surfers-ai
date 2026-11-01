"""Isolated temporal-observation prototype. No trainer imports or live controls."""
from collections import deque
from dataclasses import dataclass
import math

import numpy as np


class HistoryUnavailable(RuntimeError):
    """Caller must not dispatch a policy action using this observation."""


@dataclass(frozen=True)
class HistoryConfig:
    frames: int = 8
    span_seconds: float = 0.6
    max_latest_age_seconds: float = 0.020
    max_selection_error_seconds: float = 0.060
    max_frames: int = 256

    def __post_init__(self):
        if self.frames not in (4, 6, 7, 8, 9, 10, 11, 12):
            raise ValueError('Use 4 (baseline) or 6 through 12 frames')
        for value in (self.span_seconds, self.max_latest_age_seconds,
                      self.max_selection_error_seconds):
            if not math.isfinite(value) or value <= 0:
                raise ValueError('Time limits must be finite and positive')
        if self.max_frames < self.frames:
            raise ValueError('max_frames must hold at least one observation')

    @property
    def offsets(self):
        # Preserve the existing oldest-to-newest channel convention.
        return np.linspace(self.span_seconds, 0., self.frames)

    def contract(self):
        return dict(version='temporal_history_experiment_v1', frames=self.frames,
                    channels=3*self.frames, span_seconds=self.span_seconds,
                    order='oldest_to_newest', selection='latest_at_or_before_target',
                    max_latest_age_seconds=self.max_latest_age_seconds,
                    max_selection_error_seconds=self.max_selection_error_seconds,
                    startup='reject_until_full_history', temporal_fields='ages_seconds')


@dataclass(frozen=True)
class Observation:
    image: np.ndarray
    ages_seconds: np.ndarray
    timestamps: tuple
    segment: object


class TimestampedFrameBuffer:
    """Owns CHW uint8 RGB frames, sampled without sleeps or extra captures.

    All times must use ONE monotonic clock. Timestamp each source frame, not
    the end of preprocessing. Invalidate at every reset and pause/resume gap.
    Historical frames are context, never substitutes for live safety evidence.
    """

    def __init__(self, config=HistoryConfig()):
        self.config = config
        self._frames = deque()
        self._segment = None
        self._shape = None

    def reset(self, segment):
        self._frames.clear()
        self._shape = None
        self._segment = segment

    @property
    def stored_bytes(self):
        return sum(frame.nbytes for _, frame in self._frames)

    def append(self, frame, timestamp, *, segment):
        if segment != self._segment:
            raise ValueError('Explicit reset required before changing segment')
        if not math.isfinite(timestamp):
            raise ValueError('Invalid timestamp')
        if self._frames and timestamp <= self._frames[-1][0]:
            raise ValueError('Frame timestamps must strictly increase')
        if frame.dtype != np.uint8 or frame.ndim != 3 or frame.shape[0] != 3:
            raise ValueError('Expected CHW uint8 RGB frame')
        if self._shape is not None and frame.shape != self._shape:
            raise ValueError('Frame geometry changed within segment')
        owned = np.array(frame, copy=True, order='C')
        owned.flags.writeable = False
        self._shape = owned.shape
        self._frames.append((float(timestamp), owned))
        cutoff = timestamp - self.config.span_seconds - self.config.max_selection_error_seconds
        while self._frames and (self._frames[0][0] < cutoff or
                                len(self._frames) > self.config.max_frames):
            self._frames.popleft()

    def sample(self, now):
        if not math.isfinite(now) or not self._frames:
            raise HistoryUnavailable('No valid frame history')
        latest = self._frames[-1][0]
        if not 0 <= now-latest <= self.config.max_latest_age_seconds:
            raise HistoryUnavailable('Newest frame is future-dated or stale')
        frames = list(self._frames)
        times = np.array([t for t, _ in frames])
        targets = latest-self.config.offsets
        indices = np.searchsorted(times, targets, side='right')-1
        indices[-1] = len(frames)-1
        if np.any(indices < 0):
            raise HistoryUnavailable('Full requested history is not yet available')
        selected = times[indices]
        if len(set(indices)) != self.config.frames:
            raise HistoryUnavailable('Capture cadence cannot supply distinct history frames')
        if np.any(targets-selected > self.config.max_selection_error_seconds):
            raise HistoryUnavailable('Historical capture gap exceeds selection tolerance')
        return Observation(np.concatenate([frames[i][1] for i in indices], axis=0),
                           (now-selected).astype(np.float32), tuple(selected), self._segment)


def require_matching_contract(saved, requested):
    """Prevent an old checkpoint from silently accepting different input meaning."""
    if saved != requested:
        raise ValueError('Temporal input contract mismatch; explicit model adaptation required')
