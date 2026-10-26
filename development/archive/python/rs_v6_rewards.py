#!/usr/bin/env python3
"""Action-neutral PPO training rewards from observed local Practice outcomes.

Survival time and confirmed death only: NO YOLO predictions, specific movement,
coin proximity, pseudo safe lanes, or invented game score contribute reward.
"""
from __future__ import annotations

from dataclasses import dataclass


@dataclass
class SurvivalRewards:
    per_second: float = .4
    milestone_bonus: float = .5
    milestone_seconds: float = 2.
    death_penalty: float = 1.2
    # Avoid counting long UI stalls as unbounded gameplay progress in one step.
    max_rewarded_step_seconds: float = .15

    def __post_init__(self):
        self.reset()

    def reset(self):
        self.last_duration = 0.
        self.last_milestone = 0

    def step(self, duration: float, died: bool):
        duration = max(0., float(duration))
        elapsed = max(0., min(self.max_rewarded_step_seconds,
                              duration-self.last_duration))
        self.last_duration = duration
        milestone = int(duration // self.milestone_seconds)
        reached = max(0, milestone-self.last_milestone)
        self.last_milestone = milestone
        pieces = {'survival': self.per_second * elapsed,
                  'milestones': self.milestone_bonus*reached if not died else 0.,
                  'death': -self.death_penalty if died else 0.}
        return float(sum(pieces.values())), pieces
