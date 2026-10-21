"""V6.1.2 pre-key screenshot gate for local Ranked Surfers.

Requires independent, affirmative gameplay evidence before movement.
For uncertain transitions collects up to six separate readings and either
recovers to verified gameplay, confirms the game menu, or latches inputs off.
No unseen movement may be substituted for a PPO action.
"""
from dataclasses import dataclass
import math


@dataclass(frozen=True)
class GateReading:
    verdict: str  # PLAY, MENU, BLOCK
    state: str
    gameplay_probability: float
    menu_probability: float
    menu_score: float
    menu_evidence: bool


def screen_gate(state, probabilities, menu_score, *, confidence, menu_threshold):
    """Classify one capture, tracking independent menu-template evidence.

    MENU is a confirmed high-confidence ML menu. The template alone is not
    used to declare MENU from *one* screenshot; consecutive template hits
    can independently confirm MENU in sequence_decision().
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
    if not all(map(math.isfinite, (p_game, p_menu, score))):
        return GateReading('BLOCK', state, p_game, p_menu, score, False)

    ml_menu = state == 'MENU' and p_menu >= confidence
    template_menu = score >= menu_threshold
    if ml_menu:
        verdict = 'MENU'
    elif state == 'GAMEPLAY' and p_game >= confidence and score < menu_threshold * .55:
        verdict = 'PLAY'
    else:
        verdict = 'BLOCK'
    return GateReading(verdict, state, p_game, p_menu, score,
                       bool(ml_menu or template_menu))


def sequence_decision(readings, *, max_frames=6):
    """Return PLAY, TERMINAL, CONTINUE or ABORT from ordered screen captures.

    Two *consecutive* PLAY frames allow a movement after an uncertain frame.
    Two menu-evidence frames among the last three confirm an episode end;
    template-only evidence is useful when classifier returns UNCERTAIN.
    Otherwise wait, but never indefinitely and never send movement.
    """
    if not readings:
        return 'CONTINUE'
    if len(readings) >= 2:
        if readings[-1].verdict == readings[-2].verdict == 'PLAY':
            return 'PLAY'
        recent = readings[-3:]
        if readings[-1].menu_evidence and sum(r.menu_evidence for r in recent) >= 2:
            return 'TERMINAL'
    return 'ABORT' if len(readings) >= max_frames else 'CONTINUE'


class GuardUncertainError(RuntimeError):
    """Fail-closed UI guard aborted a rollout; no action was sent."""
