"""Conservative, independently testable live-input gate for V6.1.1.

This is NOT a classifier. It evaluates the existing full-screen learned screen
state plus the independent menu-template score. Anything not positively
confirmed as gameplay suppresses movement; uncertain transitions fail closed.
"""
from __future__ import annotations

from dataclasses import dataclass
import math


@dataclass(frozen=True)
class GateReading:
    verdict: str  # PLAY, MENU, BLOCK
    state: str
    gameplay_probability: float
    menu_probability: float
    menu_score: float


def screen_gate(state, probabilities, menu_score, *, confidence, menu_threshold):
    """Return a one-frame verdict without treating 'not MENU' as gameplay.

    The actual live controller adds a second capture for any non-PLAY state.
    """
    if state not in ('MENU', 'LOADING', 'GAMEPLAY', 'OTHER', 'UNCERTAIN'):
        state = 'UNCERTAIN'
    probs = probabilities or {}
    try:
        p_game = float(probs.get('GAMEPLAY', 0.))
        p_menu = float(probs.get('MENU', 0.))
        score = float(menu_score)
    except (TypeError, ValueError):
        p_game, p_menu, score = 0., 0., float('nan')
    if not all(math.isfinite(x) for x in (p_game, p_menu, score)):
        return GateReading('BLOCK', state, p_game, p_menu, score)
    # A definite menu has priority even if the independent text template
    # misses an animation frame. A strong template prevents PLAY permission.
    if state == 'MENU' and p_menu >= confidence:
        verdict = 'MENU'
    elif state == 'GAMEPLAY' and p_game >= confidence and score < menu_threshold * .55:
        verdict = 'PLAY'
    else:
        verdict = 'BLOCK'
    return GateReading(verdict, state, p_game, p_menu, score)


def pair_decision(first: GateReading, second: GateReading):
    """Decision when the first immediate pre-key capture was NOT PLAY.

    MENU only when *both* independently captured screenshots say MENU.
    A single PLAY or MENU recovery requests a *third* screenshot to confirm.
    Do not silently replace an agent-requested action with a no-op: that would
    corrupt the on-policy PPO rollout. Otherwise abort the whole run.
    """
    if first.verdict == 'MENU' and second.verdict == 'MENU':
        return 'TERMINAL'
    if second.verdict == 'MENU':
        return 'CONFIRM_MENU'
    if second.verdict == 'PLAY':
        return 'CONFIRM_PLAY'
    return 'ABORT'
