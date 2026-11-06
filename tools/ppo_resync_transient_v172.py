#!/usr/bin/env python3
"""Conservative, observation-only transient window after the PPO optimizer.

PPO resumes only after four consecutive accepted fresh captures. The opt-in
stateful guard may corroborate UNCERTAIN using independent visual continuity.
"""
from __future__ import annotations
import math

MAX_RESYNC_CAPTURES = 8
MAX_TRANSIENT_SKIPS = 4
REQUIRED_GAMEPLAY_FRAMES = 4


def completion_tail_frames(accepted, enabled):
    """One bounded opportunity to finish an already accepted partial stack.

    No extra budget for zero evidence, and never enough for a new four-frame
    sequence if another rejection occurs. Does not authorize keys or a terminal.
    """
    return REQUIRED_GAMEPLAY_FRAMES-accepted if enabled and 0<accepted<REQUIRED_GAMEPLAY_FRAMES else 0


class ResyncWindow:
    def __init__(self):
        self.accepted = 0
        self.skipped = 0
        self.observed = 0
        self.discarded_before_last_retry = 0
        self.last_reason = None
        self.menu_candidate = False

    def observe(self, *, state, probs, menu_score, confidence, menu_threshold,
                continuity=False, continuity_pending=False):
        """Return ACCEPT/DONE/RETRY/ABORT. Never authorizes an action directly."""
        self.observed += 1
        self.menu_candidate = False
        try:
            score = float(menu_score)
            p_game = float((probs or {}).get('GAMEPLAY', 0.0))
            p_menu = float((probs or {}).get('MENU', 0.0))
            conf = float(confidence)
            threshold = float(menu_threshold)
        except (TypeError, ValueError, OverflowError):
            self.last_reason = 'invalid score/probabilities'
            return 'ABORT'
        if (not all(math.isfinite(v) for v in (score, p_game, p_menu, conf, threshold))
                or not .5 <= conf < 1 or not 0 < threshold <= 1
                or not 0 <= score <= 1 or not 0 <= p_game <= 1 or not 0 <= p_menu <= 1):
            self.last_reason = 'invalid score/probabilities'
            return 'ABORT'
        # Candidate means "request independent confirmation", never terminal
        # acceptance. Early menu animation may have a weak template score.
        self.menu_candidate = (
            (state == 'MENU' and p_menu >= conf) or
            # An animated menu can fall just short of the classifier threshold
            # (2026-11-01: UNCERTAIN, pm=.889, pg=.088, template=.656).
            # This ONLY requests the existing bounded independent confirmation.
            # It never accepts a terminal, packs a frame or permits an input.
            (state == 'UNCERTAIN' and p_menu >= max(.80, conf*.90)
             and p_game < .10 and score >= threshold*.80))
        if (state == 'GAMEPLAY' and p_game >= conf and score < threshold*.85) or (
                continuity and state=='UNCERTAIN' and score<threshold*.30):
            self.accepted += 1
            self.last_reason = None
            return 'DONE' if self.accepted == REQUIRED_GAMEPLAY_FRAMES else 'ACCEPT'
        # No relaxation of classifier confidence: uncertain captures are dropped,
        # not passed to _pack or counted toward the four safe GAMEPLAY frames.
        # A brief UNCERTAIN frame is not evidence of gameplay or death.
        # In particular, a tunnel/camera transition can push p_game below .70.
        # Passive retries may wait for independent proof, but DO NOT permit a
        # movement key: four consecutive high-confidence GAMEPLAY captures are
        # still required. Strong template evidence, other states and exhausted retries
        # still abort. Never count an UNCERTAIN frame toward the image stack.
        # A catch/camera transition may mix GAMEPLAY and MENU probabilities
        # (episode 1566: .649/.344, template .161). A majority GAMEPLAY vote
        # with a weak template may also wait for fresh proof. It is NOT accepted
        # as gameplay, and a subsequent confident MENU uses normal confirmation.
        retry_probabilities = (continuity_pending or p_menu < .10 or
                               (.50 < p_game < conf and p_menu < .50 and p_game > p_menu))
        if (state == 'UNCERTAIN' and p_game < conf and retry_probabilities
                and score < threshold * .30 and self.skipped < MAX_TRANSIENT_SKIPS):
            self.skipped += 1
            self.discarded_before_last_retry = self.accepted
            self.accepted = 0
            self.last_reason = 'transient UNCERTAIN; no corroborated menu evidence'
            return 'RETRY'
        self.last_reason = ('unsafe post-resume state=' + str(state) +
                            f' menu_score={score:.3f} ml_gameplay={p_game:.3f} '
                            f'ml_menu={p_menu:.3f} skips={self.skipped}')
        return 'ABORT'
