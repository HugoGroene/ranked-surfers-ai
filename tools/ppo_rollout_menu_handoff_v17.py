"""Strict eligibility for preserving a confirmed death across PPO rollout boundaries.

Only the pre-pause, pre-click GAMEPLAY check may authorize a MENU handoff.
No other pause errors (focus, overlay, freeze, uncertainty) may be recovered.
"""
import math
import re


_PATTERN = re.compile(r'^Not verified GAMEPLAY at rollout end: MENU, menu=([0-9]+(?:\.[0-9]+)?)$')


def boundary_menu_candidate(error):
    """True only for a precise pre-pause ML MENU result, subject to independent recheck."""
    match = _PATTERN.fullmatch(str(error))
    return bool(match and math.isfinite(float(match.group(1))))
