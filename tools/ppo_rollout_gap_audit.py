"""Pure-Python timing diagnostics for a real-time PPO environment.

This module *only* records timestamps. It does not modify policy observations,
rollouts, rewards, steps, actions, or training state.
"""
from dataclasses import dataclass
from typing import Optional
import math


@dataclass
class RolloutGapTracker:
    last_step_at: Optional[float] = None
    last_capture_at: Optional[float] = None
    last_step_done: Optional[bool] = None
    last_rollout_end_at: Optional[float] = None
    n_boundaries: int = 0

    def record_step(self, now: float, *, capture_at=None, done=False) -> None:
        """Called after a completed environment step, not before input dispatch."""
        now = float(now)
        if not math.isfinite(now):
            raise ValueError('non-finite step clock')
        self.last_step_at = now
        self.last_step_done = bool(done)
        if capture_at is not None:
            capture_at = float(capture_at)
            if math.isfinite(capture_at) and capture_at <= now:
                self.last_capture_at = capture_at

    def finish_rollout(self, now: float) -> None:
        """Marks SB3 on_rollout_end: just before PPO optimizer updates."""
        now = float(now)
        if not math.isfinite(now):
            raise ValueError('non-finite rollout clock')
        self.last_rollout_end_at = now

    def begin_rollout(self, now: float, *, timesteps: int, warning_ms=250.0):
        """Called at next on_rollout_start, after PPO.train() has returned.

        The environment is deliberately NOT stepped or captured here. A high
        cached age flags a possible live-game mismatch, not proof of a death.
        """
        now = float(now)
        if not math.isfinite(now):
            raise ValueError('non-finite rollout clock')
        if self.last_rollout_end_at is None:
            return None  # first rollout, no preceding optimizer update
        end = self.last_rollout_end_at
        self.last_rollout_end_at = None
        self.n_boundaries += 1

        def elapsed(since):
            if since is None or since > now:
                return None
            return round((now - since) * 1000.0, 2)

        frame_age = elapsed(self.last_capture_at)
        record = {
            'event': 'ppo_rollout_resume_gap',
            'boundary': self.n_boundaries,
            'timesteps': int(timesteps),
            'rollout_end_to_next_start_ms': elapsed(end),
            'last_env_step_to_next_start_ms': elapsed(self.last_step_at),
            'cached_frame_to_next_start_ms': frame_age,
            'last_env_step_done': self.last_step_done,
            'potentially_stale': (frame_age is not None and frame_age >= warning_ms),
            'warning_threshold_ms': float(warning_ms),
            'interpretation': 'Timestamp age of cached observation at the next PPO rollout. '
                              'Does not establish that Unity died or a specific action was wrong.',
        }
        return record
