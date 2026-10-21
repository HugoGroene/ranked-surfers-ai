#!/usr/bin/env python3
"""Dense V6.1 reward based only on elapsed CONFIRMED local gameplay and death.

No detector-based safety targets, guessed HUD distance, lane-change rewards,
coin collection, or specific keypress rewards. Robust to death menu frame lag.
"""
from dataclasses import dataclass
from math import isfinite


@dataclass
class ProgressRewards:
    per_second: float = 0.8
    milestone_bonus: float = 0.15
    milestone_seconds: float = 0.5
    death_penalty: float = 1.0
    max_rewarded_step_seconds: float = 0.15

    def __post_init__(self):
        if self.per_second<0 or self.milestone_bonus<0 or self.death_penalty<0:
            raise ValueError('Reward coefficients must be nonnegative')
        if self.milestone_seconds<=0 or self.max_rewarded_step_seconds<=0:
            raise ValueError('Reward timing must be positive')
        self.reset()

    def reset(self):
        self.last_duration=0.
        self.last_milestone=0

    def step(self, duration, died):
        duration=float(duration)
        if not isfinite(duration):
            raise ValueError('Non-finite episode duration')
        duration=max(0.,duration)
        increment=max(0., duration-self.last_duration)
        capped_increment=min(self.max_rewarded_step_seconds, increment)
        self.last_duration=max(self.last_duration,duration)
        milestone=int((duration+1e-9)//self.milestone_seconds)
        newly=max(0,milestone-self.last_milestone)
        self.last_milestone=max(self.last_milestone,milestone)
        # Terminal death cancels milestone awarded only on the terminal step;
        # survival reward remains for real gameplay time actually experienced.
        pieces={'survival':self.per_second*capped_increment,
                'milestones':0. if died else self.milestone_bonus*newly,
                'death':-self.death_penalty if died else 0.}
        return float(sum(pieces.values())),pieces
