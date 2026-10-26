"""V6.1.2 pre-key screenshot gate for local Ranked Surfers.

An ML-only MENU prediction is not a confirmed death: the existing V4 classifier
can misclassify moving gameplay. Prefer agreement with the independent menu
template. When ML and template disagree, independent scene motion *and* a
changing gameplay-distance HUD can corroborate continued play. Without this
positive evidence, all movement stays blocked and the rollout aborts safely.
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
    ml_menu: bool = False
    visual_gameplay: bool = False
    track_motion: float = 0.
    track_texture: float = 0.
    hud_pixels: int = 0
    hud_changed_pixels: int = 0
    ml_template_corroborated: bool = False
    template_support: bool = False  # Independent moderate template evidence, even when ML is UNCERTAIN


def screen_gate(state, probabilities, menu_score, *, confidence, menu_threshold,
                track_motion=0., track_texture=0., hud_pixels=0,
                hud_changed_pixels=0, motion_threshold=2.2):
    """Classify one capture, tracking independent menu-template evidence.

    A strong ML MENU vote without a corroborating template remains BLOCK.
    Motion is considered independent positive gameplay evidence only with a
    textured moving track AND a visibly changing distance HUD. Mere absence
    of menu-template matches is never enough to allow a movement.
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
    # The inherited restart detector accepts two consecutive ML MENU votes
    # with a template score >= 80% of the main threshold. For in-game death
    # confirmation, require FOUR consecutive such corroborated captures.
    # This is still independent template evidence, unlike ML-only MENU.
    template_support = score >= menu_threshold * .80
    ml_template_corroborated = ml_menu and template_support
    try:
        motion = float(track_motion)
        texture = float(track_texture)
        hud_count = int(hud_pixels)
        hud_delta = int(hud_changed_pixels)
        visual = (all(map(math.isfinite, (motion, texture))) and
                  motion >= motion_threshold and texture >= 12. and
                  hud_count >= 8 and hud_delta >= 8 and
                  score < menu_threshold * .55)
    except (TypeError, ValueError, OverflowError):
        motion, texture, hud_count, hud_delta, visual = 0., 0., 0, 0, False

    if ml_menu and template_menu:
        verdict = 'MENU'
    elif state == 'GAMEPLAY' and p_game >= confidence and score < menu_threshold * .55:
        verdict = 'PLAY'
    else:
        verdict = 'BLOCK'
    return GateReading(verdict, state, p_game, p_menu, score,
                       bool(template_menu), bool(ml_menu), bool(visual),
                       motion, texture, hud_count, hud_delta,
                       bool(ml_template_corroborated), bool(template_support))


def sequence_decision(readings, *, max_frames=6):
    """Return PLAY, TERMINAL, CONTINUE or ABORT from ordered screen captures.

    Two consecutive ML PLAY frames allow a movement after uncertainty. So do
    two consecutive independent visual GAMEPLAY confirmations, derived from
    three changing captured frames. Two template-positive frames among the
    last three confirm MENU even when ML is UNCERTAIN. Four consecutive
    ML MENU + moderately matching template frames also confirm death. A
    longer sustained ML MENU sequence can accumulate TWO strong template
    matches spread across the six-capture window, provided at least four
    frames have independent moderate template corroboration and there are no
    intervening non-MENU or visual-gameplay frames. A separate six-frame rule
    handles a menu animation that begins UNCERTAIN and ends with confident
    MENU predictions, provided the independent template agrees repeatedly.
    ML-only MENU votes never confirm death. Otherwise wait and ultimately
    abort without movement.
    """
    if not readings:
        return 'CONTINUE'
    if len(readings) >= 2:
        recent = readings[-3:]
        if readings[-1].menu_evidence and sum(r.menu_evidence for r in recent) >= 2:
            return 'TERMINAL'
        if (len(readings) >= 4 and
                all(r.ml_template_corroborated for r in readings[-4:])):
            return 'TERMINAL'
        # Episode 0002: a genuine menu yielded template scores
        # [.65, 1.00, .66, .47, .58, .73]. The two strong matches were
        # separated by an animation dip, while V4 said MENU throughout.
        # Count only a *continuous* run of high-confidence ML MENU with no
        # positive visual-gameplay evidence; an intervening gameplay or
        # uncertain reading breaks the run. Never accept ML-only evidence.
        if readings[-1].menu_evidence and readings[-1].ml_menu:
            contiguous_menu = []
            for reading in reversed(readings[-max_frames:]):
                if not reading.ml_menu or reading.visual_gameplay:
                    break
                contiguous_menu.append(reading)
            if (len(contiguous_menu) >= 5 and
                    sum(r.menu_evidence for r in contiguous_menu) >= 2 and
                    sum(r.ml_template_corroborated for r in contiguous_menu) >= 4):
                return 'TERMINAL'
        # Episode 0058: title/menu animation began with three UNCERTAIN V4
        # frames, then three confident MENU frames. The menu template was
        # strong in captures 2 and 6 but dipped in between. The earlier
        # long-span rule intentionally required continuous ML MENU and missed
        # this real menu. Make this path deliberately stricter: use all six
        # fresh captures, require repeated independent template corroboration,
        # finish on a strong corroborated MENU, and reject ANY recognized
        # gameplay/loading or independently confirmed visual gameplay.
        if len(readings) >= max_frames:
            window = readings[-max_frames:]
            if (len(window) >= 6 and
                    window[-1].ml_menu and window[-1].menu_evidence and
                    sum(r.menu_evidence for r in window) >= 2 and
                    sum(r.template_support for r in window) >= 4 and
                    all(r.ml_menu for r in window[-2:]) and
                    not any(r.state in ('GAMEPLAY', 'LOADING', 'OTHER') or
                            r.verdict == 'PLAY' or r.visual_gameplay
                            for r in window)):
                return 'TERMINAL'
        if not any(r.menu_evidence for r in recent):
            if readings[-1].verdict == readings[-2].verdict == 'PLAY':
                return 'PLAY'
            if (len(readings) >= 3 and readings[-1].visual_gameplay and
                    readings[-2].visual_gameplay):
                return 'PLAY'
    return 'ABORT' if len(readings) >= max_frames else 'CONTINUE'


class GuardUncertainError(RuntimeError):
    """Fail-closed UI guard aborted a rollout; no action was sent."""
