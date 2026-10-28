#!/usr/bin/env python3
"""Offline-validatable guards for experimental menu recovery at PPO update boundaries.

This helper cannot restart the game; the game-specific screen checks live in
PPOV61Env, and the callback only resets the monitored VecEnv on verified MENU.
"""
import numpy as np


def candidate_is_menu(reason):
    """Only screen-state MENU is eligible; never recover ambiguous/loading/focus loss."""
    return isinstance(reason, str) and reason.startswith('unsafe screen state=MENU menu_score=')


def accept_menu_checks(samples, *, threshold, focused):
    """Require 3 separate independent screenshots with both ML + template MENU.

    `samples` holds (state, template_similarity) tuples. Conservative by design.
    """
    if not focused or len(samples) != 3:
        return False
    for state, score in samples:
        try:
            value = float(score)
        except (ValueError, TypeError):
            return False
        if state != 'MENU' or not np.isfinite(value) or value < threshold:
            return False
    return True


def validated_new_vec_episode(previous_obs, new_vec_obs, n_envs):
    """Validate a new Dict VecEnv.reset() observation BEFORE changing PPO state.

    Returns owned copies of the reset observation and new episode-start flags.
    Both keys and data shapes must match the existing PPO policy observation.
    """
    if n_envs != 1 or not isinstance(previous_obs, dict) or not isinstance(new_vec_obs, dict):
        raise ValueError('Expected a single-environment Dict VecEnv')
    if set(previous_obs) != set(new_vec_obs):
        raise ValueError('Reset observation keys changed')
    validated = {}
    for key, prior in previous_obs.items():
        old = np.asarray(prior)
        new = np.asarray(new_vec_obs[key])
        if old.shape != new.shape or old.dtype != new.dtype or old.shape[0] != 1:
            raise ValueError(f'Unsafe VecEnv reset observation {key}: {new.shape}/{new.dtype}')
        if new.dtype.kind == 'f' and not np.isfinite(new).all():
            raise ValueError(f'Non-finite reset features in {key}')
        validated[key] = new.copy()
    return validated, np.ones((1,), dtype=bool)
