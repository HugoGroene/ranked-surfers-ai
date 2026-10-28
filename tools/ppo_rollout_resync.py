#!/usr/bin/env python3
"""Boundary-resynchronization helper for a one-environment Dict PPO.

Internal SB3 _last_obs is intentionally changed only in the experimental
rollout-start callback. No action, reward, or episode flag is modified here.
"""
import numpy as np


def validated_vec_observation(model, raw_obs):
    """Return a new, strictly shape/dtype-checked one-env observation dict.

    Validates completely before assignment, so callers cannot install a partial
    dictionary of images/aux. A returned copy is safe from subsequent env reuse.
    """
    old = getattr(model, '_last_obs', None)
    if not isinstance(old, dict) or not isinstance(raw_obs, dict):
        raise ValueError('PPO resync requires Dict observations')
    if set(old) != set(raw_obs):
        raise ValueError(f'PPO resync observation keys changed: {set(old)} != {set(raw_obs)}')
    if int(getattr(model, 'n_envs', -1)) != 1:
        raise ValueError('PPO resync supports exactly one environment')
    updated = {}
    for key, existing in old.items():
        existing = np.asarray(existing)
        incoming = np.asarray(raw_obs[key])
        if existing.shape[0] != 1 or incoming.shape != existing.shape[1:]:
            raise ValueError(f'PPO resync {key} shape changed: {incoming.shape} vs {existing.shape}')
        if incoming.dtype != existing.dtype:
            raise ValueError(f'PPO resync {key} dtype changed: {incoming.dtype} vs {existing.dtype}')
        if incoming.dtype.kind == 'f' and not np.isfinite(incoming).all():
            raise ValueError(f'PPO resync {key} contains non-finite features')
        updated[key] = incoming[np.newaxis].copy()
    return updated


def replace_cached_observation(model, raw_obs):
    """After all validation, update only SB3's next-action input."""
    model._last_obs = validated_vec_observation(model, raw_obs)
    # _last_episode_starts, rollout buffer, rewards and optimizer untouched.
