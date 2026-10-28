"""Pause-aware wall-clock compensation for the single-env PPO Practice trainer.

Adjust recorded clock origins after a verified game pause.  The game did not
advance, so neither episode survival duration nor time-since-action features
should advance during the pause.  No reward/step/observation is fabricated.
"""
import math


def _value(x, label):
    try:
        y = float(x)
    except (TypeError, ValueError, OverflowError) as exc:
        raise ValueError(f'Invalid {label}: {x!r}') from exc
    if not math.isfinite(y):
        raise ValueError(f'Nonfinite {label}')
    return y


def compensate_pause_clocks(env, elapsed_mono, elapsed_perf, *, monitor=None, elapsed_wall=None):
    """Validate everything first, then shift *existing* time origins atomically.

    Intentionally does NOT update captured_at on old YOLO detections: the
    detector's existing stale-result rejection remains conservative.
    """
    dm = _value(elapsed_mono, 'monotonic pause')
    dp = _value(elapsed_perf, 'perf_counter pause')
    if not (0.0 < dm <= 30.0 and 0.0 < dp <= 30.0):
        raise ValueError(f'Pause outside safe bound: {dm:.3f}/{dp:.3f}s')
    if abs(dm - dp) > 0.30:
        raise ValueError('Monotonic/perf pause clocks disagreed')
    if monitor is None or getattr(monitor, 'env', None) is not env:
        raise ValueError('Expected the exact SB3 Monitor wrapping this single raw environment')
    dw = _value(elapsed_wall, 'wall clock pause')
    if not (0.0 < dw <= 30.0) or abs(dw - dm) > 0.30:
        raise ValueError('Wall/monotonic pause clocks disagreed')
    monitor_origin = _value(getattr(monitor, 't_start', None), 'Monitor.t_start')

    planned = {}
    for name, delta in (('episode_start', dm), ('last_step', dm),
                        ('episode_started', dp)):
        if not hasattr(env, name):
            raise ValueError(f'Missing required PPO clock: {name}')
        planned[name] = _value(getattr(env, name), name) + delta
    for name, delta in (('suspected_since', dm),
                        ('previous_observation_at', dp),
                        ('_prev_step_start', dp)):
        if getattr(env, name, None) is not None:
            planned[name] = _value(getattr(env, name), name) + delta

    context = getattr(env, 'action_context', None)
    if context is None or not isinstance(getattr(context, 'last_action_times', None), list):
        raise ValueError('Missing V6.1 action clock history')
    old_actions = context.last_action_times
    if len(old_actions) != 5:
        raise ValueError('Action timing count changed')
    new_actions = [None if x is None else _value(x, 'action timestamp') + dp
                   for x in old_actions]

    # No mutation before all fields and action clocks pass validation.
    for name, value in planned.items():
        setattr(env, name, value)
    context.last_action_times[:] = new_actions
    monitor.t_start = monitor_origin + dw
    return {'paused_wall_seconds': round(dm, 3),
            'corrected_episode_timer': True,
            'corrected_action_timer': True,
            'corrected_monitor_timer': True}
