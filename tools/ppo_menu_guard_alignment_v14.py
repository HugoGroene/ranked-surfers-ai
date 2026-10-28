#!/usr/bin/env python3
"""Conservative rolling MENU confirmation following PPO optimizer pauses.

Only accept FOUR consecutive, independent corroborated MENU captures. We
allow a bounded number of additional captures so a single early menu-animation
transition does not permanently prevent recovery. No thresholds are lowered.
"""
from tools.ppo_menu_guard_alignment import accept_corroborated_menu

MAX_CONFIRM_CAPTURES = 8
REQUIRED_MENU_CAPTURES = 4


def accept_menu_after_transient(readings, *, focused):
    """True iff the last four readings meet the existing v1.3 safety guard.

    Caller must invoke this AFTER each NEW frame, no more than MAX_CONFIRM_CAPTURES
    times, and must require a fresh browser focus check before each capture.
    An earlier sequence cannot authorize restart if the newest frame differs.
    """
    if not focused or not REQUIRED_MENU_CAPTURES <= len(readings) <= MAX_CONFIRM_CAPTURES:
        return False
    return accept_corroborated_menu(readings[-REQUIRED_MENU_CAPTURES:], focused=True)
