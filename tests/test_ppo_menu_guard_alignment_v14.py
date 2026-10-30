"""Offline tests for bounded MENU animation tolerance, no game required."""
import pathlib
import sys
import unittest

HERE = pathlib.Path(__file__).resolve().parents[1]
sys.path.insert(0, str(HERE))

from rs_v612_gameplay_guard import screen_gate
from tools.ppo_menu_guard_alignment_v14 import (
    accept_menu_after_transient, MAX_CONFIRM_CAPTURES)
from tools.extend_ppo_menu_recovery_window import V13_METHOD, V14_METHOD, patch


class RollingMenuTests(unittest.TestCase):
    def reading(self, score=.60, state='MENU', menu_prob=.999):
        return screen_gate(state, {'MENU':menu_prob, 'GAMEPLAY':1-menu_prob},score,
                           confidence=.90,menu_threshold=.68)

    def test_exact_observed_animation_recovers_after_fifth_good_frame(self):
        observed = [self.reading(score=s) for s in (.47064,.72788,.68168,.58733,.618)]
        self.assertFalse(accept_menu_after_transient(observed[:4],focused=True))
        self.assertTrue(accept_menu_after_transient(observed,focused=True))

    def test_three_corroborations_do_not_allow_restart(self):
        self.assertFalse(accept_menu_after_transient([
            self.reading(.47), self.reading(.73), self.reading(.68), self.reading(.59)],focused=True))

    def test_four_moderately_corroborated_allow_restart(self):
        self.assertTrue(accept_menu_after_transient([self.reading(.57719)]*4,focused=True))

    def test_intervening_gameplay_requires_four_new_consecutive_menus(self):
        rs=[self.reading(.65)]*3+[self.reading(.1,state='GAMEPLAY',menu_prob=.01)]
        self.assertFalse(accept_menu_after_transient(rs,focused=True))
        rs=rs+[self.reading(.65)]*3
        self.assertFalse(accept_menu_after_transient(rs,focused=True))
        self.assertTrue(accept_menu_after_transient(rs+[self.reading(.65)],focused=True))

    def test_ml_only_menu_is_never_accepted(self):
        self.assertFalse(accept_menu_after_transient([self.reading(.1)]*8,focused=True))

    def test_unconfident_predictions_cannot_authorize_reset(self):
        self.assertFalse(accept_menu_after_transient([self.reading(.7,menu_prob=.5)]*4,focused=True))

    def test_no_focus_cannot_authorize_reset(self):
        self.assertFalse(accept_menu_after_transient([self.reading(.7)]*4,focused=False))

    def test_rejects_nan(self):
        self.assertFalse(accept_menu_after_transient([self.reading(float('nan'))]*4,focused=True))

    def test_maximum_window_is_bounded(self):
        self.assertFalse(accept_menu_after_transient([self.reading(.7)]*9,focused=True))
        self.assertEqual(MAX_CONFIRM_CAPTURES,8)

    def test_requires_four(self):
        self.assertFalse(accept_menu_after_transient([self.reading(.7)]*3,focused=True))

    def test_patch_changes_only_v13_method(self):
        original='class FakeEnv:\n'+V13_METHOD+'def flags():\n    x="--resync-rollouts"\n    y="--recover-rollout-menu"\n'
        updated=patch(original)
        self.assertEqual(updated.replace(V14_METHOD,V13_METHOD),original)
        self.assertEqual(updated.count(V14_METHOD),1)
        with self.assertRaisesRegex(ValueError,'already installed'):
            patch(updated)
        with self.assertRaisesRegex(ValueError,'Expected exactly one'):
            patch(original.replace(V13_METHOD,'    def other():\n        pass\n'))

    def test_compiles_new_method(self):
        compile('class Dummy:\n'+V14_METHOD,'<test>','exec')


if __name__ == '__main__':
    unittest.main()
