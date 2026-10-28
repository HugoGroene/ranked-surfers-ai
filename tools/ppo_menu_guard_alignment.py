#!/usr/bin/env python3
"""Pure offline-testable guard for post-PPO MENU recovery.

Same four-frame ML-plus-template threshold and terminal verdict as the
existing rs_v612_gameplay_guard. Never accept ML-only MENU predictions.
"""
from rs_v612_gameplay_guard import sequence_decision


def accept_corroborated_menu(readings, *, focused):
    """Only four high-confidence MENU readings with corroborating template.

    Called with GateReading objects built via screen_gate using the *same*
    threshold and classifier confidence as the live movement guard. Four
    consecutive moderate matches (>=0.80 x menu template threshold) are
    explicitly treated as terminal by the existing guard, even if their
    scores fall just below the strong threshold (>=0.85) used in v1.2.
    """
    if not focused or len(readings) != 4:
        return False
    if not all(r.state == 'MENU' and r.ml_template_corroborated and
               not r.visual_gameplay and r.verdict != 'PLAY' for r in readings):
        return False
    return sequence_decision(readings, max_frames=6) == 'TERMINAL'
