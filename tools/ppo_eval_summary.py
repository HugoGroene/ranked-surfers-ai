"""Explicit evaluation clocks and completion status, independent of rendering."""
import math


def episode_metrics(info, *, legacy=False):
    wall = float(info['duration'])
    verified = None if legacy else float(info['reward_duration_s'])
    if not math.isfinite(wall) or wall < 0 or (verified is not None and
            (not math.isfinite(verified) or verified < 0)):
        raise ValueError('Invalid evaluation survival clock')
    return (wall if legacy else verified), wall, verified


def completion(requested, completed, stop_reason):
    complete = completed == requested and stop_reason == 'episodes'
    return dict(status='COMPLETE' if complete else 'INCOMPLETE',
                requested_episodes=requested, completed_episodes=completed,
                completed_requested_episodes=complete, stop_reason=stop_reason)
