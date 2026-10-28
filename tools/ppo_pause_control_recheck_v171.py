#!/usr/bin/env python3
"""Fail-closed evidence window when React's Pause button disappears at a PPO boundary.

No browser calls or inputs are performed here. The caller provides one fresh
full-screen classifier/template observation per check. Rules deliberately
reuse the existing independent V6.1.2 MENU corroboration guard.
"""
from rs_v612_gameplay_guard import screen_gate
from tools.ppo_menu_guard_alignment_v14 import accept_menu_after_transient

MAX_RECHECK_CAPTURES = 8


class PauseControlRecheck:
    def __init__(self, *, confidence, threshold, motion_threshold):
        self.confidence = float(confidence)
        self.threshold = float(threshold)
        self.motion_threshold = float(motion_threshold)
        self.readings = []
        self.first_menu_observed_mono = None
        self.button_states = []

    def observe(self, *, state, probabilities, menu_score, observed_mono,
                focused, button_count, button_visible):
        """Return WAIT/READY/MENU/ABORT; never authorize a click from ML alone.

        READY needs a UNIQUE VISIBLE button and TWO latest independent PLAY
        frames. MENU needs four latest ML+independent-template corroborations.
        A missing button or ambiguous state never authorizes a click or reset.
        """
        if not focused or button_count not in (0, 1):
            return 'ABORT'
        button_ok = button_count == 1 and bool(button_visible)
        reading = screen_gate(
            state, probabilities, menu_score,
            confidence=self.confidence, menu_threshold=self.threshold,
            motion_threshold=self.motion_threshold)
        self.readings.append(reading)
        self.button_states.append(button_ok)
        if reading.ml_template_corroborated and self.first_menu_observed_mono is None:
            self.first_menu_observed_mono = float(observed_mono)
        if accept_menu_after_transient(self.readings, focused=True):
            return 'MENU'
        if (button_ok and len(self.readings) >= 2 and
                self.readings[-1].verdict == 'PLAY' and
                self.readings[-2].verdict == 'PLAY'):
            return 'READY'
        if len(self.readings) >= MAX_RECHECK_CAPTURES:
            return 'ABORT'
        return 'WAIT'

    def diagnostic(self):
        return {
            'checks': len(self.readings),
            'button_visible': self.button_states,
            'screen_states': [r.state for r in self.readings],
            'menu_scores': [round(float(r.menu_score), 3) for r in self.readings],
            'verdicts': [r.verdict for r in self.readings],
        }
