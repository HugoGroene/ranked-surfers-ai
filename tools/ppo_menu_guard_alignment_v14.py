#!/usr/bin/env python3
"""Conservative rolling MENU confirmation following PPO optimizer pauses.

Only accept FOUR consecutive, independent corroborated MENU captures. We
allow a bounded number of additional captures so a single early menu-animation
transition does not permanently prevent recovery. No thresholds are lowered.
"""
from tools.ppo_menu_guard_alignment import accept_corroborated_menu

MAX_CONFIRM_CAPTURES = 8
MAX_EXTENDED_CONFIRM_CAPTURES = 12
REQUIRED_MENU_CAPTURES = 4


def late_menu_confirmation_pending(readings):
    """At the normal budget, a corroborated late MENU may request four more frames."""
    if len(readings)!=MAX_CONFIRM_CAPTURES:
        return False
    tail=readings[-3:]
    return (all(r.state=='MENU' and r.ml_menu and not r.visual_gameplay for r in tail)
            and sum(r.ml_template_corroborated for r in tail)>=2
            and any(r.menu_evidence for r in tail)
            and tail[-1].ml_template_corroborated)


def accept_menu_after_transient(readings, *, focused, max_captures=MAX_CONFIRM_CAPTURES):
    """True iff the last four readings meet the existing v1.3 safety guard.

    Caller must invoke this AFTER each NEW frame, within max_captures, and
    require a fresh browser focus check before each capture. The extended limit
    is for callers that verified late_menu_confirmation_pending at frame eight.
    An earlier sequence cannot authorize restart if the newest frame differs.
    """
    if (max_captures not in (MAX_CONFIRM_CAPTURES,MAX_EXTENDED_CONFIRM_CAPTURES) or
            not focused or not REQUIRED_MENU_CAPTURES <= len(readings) <= max_captures):
        return False
    return accept_corroborated_menu(readings[-REQUIRED_MENU_CAPTURES:], focused=True)
