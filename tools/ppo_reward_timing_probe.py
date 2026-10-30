#!/usr/bin/env python3
"""Synthetic fixed-duration reward comparison. Does NOT launch Unity or PPO."""
# Support direct execution as `python tools/ppo_reward_timing_probe.py`.
from pathlib import Path
import sys
if __package__ in (None, ''):
    sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from rs_v61_rewards import ProgressRewards


def evaluate(duration_s, step_s, gamma=.995):
    """Same wall-clock survival; different policy-decision intervals."""
    rewarder = ProgressRewards()
    n = round(duration_s / step_s)
    if abs(n * step_s - duration_s) > 1e-8:
        raise ValueError('duration must be an exact multiple of step_s')
    rewards = []
    for k in range(n):
        reward, _ = rewarder.step((k + 1) * step_s, died=(k == n - 1))
        rewards.append(reward)
    return {'step_ms': round(1000 * step_s, 2), 'decision_steps': n,
            'total_reward': round(sum(rewards), 5),
            'discounted_step_return': round(sum((gamma ** i) * r for i, r in enumerate(rewards)), 5)}


if __name__ == '__main__':
    print('Synthetic 5-second episode, same death time, gamma=0.995:')
    for step_s in (.05, .10, .20, .25):
        print(evaluate(5.0, step_s))
    print('Interpretation: the per-step 150 ms cap and step-based discounting can '
          'make these totals differ even when survival time is identical. '
          'This is a diagnostic, not a recommendation to change rewards.')
