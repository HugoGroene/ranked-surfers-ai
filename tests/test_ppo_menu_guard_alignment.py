"""Offline validation: no game, network, YOLO, model training or checkpoint writes."""
import pathlib
import sys
import unittest

HERE = pathlib.Path(__file__).resolve().parents[1]
sys.path.insert(0, str(HERE))
from rs_v612_gameplay_guard import screen_gate
from tools.ppo_menu_guard_alignment import accept_corroborated_menu
from tools.align_ppo_menu_recovery_guard import patch, OLD_METHOD, NEW_METHOD


class MenuAlignmentTests(unittest.TestCase):
    def reading(self, state='MENU', score=.5771913909581744, p_menu=.999, p_game=0.):
        return screen_gate(state, {'MENU': p_menu, 'GAMEPLAY': p_game}, score,
                           confidence=.90, menu_threshold=.68)

    def test_exact_recorded_borderline_template_is_accepted_after_four(self):
        samples = [self.reading() for _ in range(4)]
        self.assertTrue(all(r.ml_template_corroborated for r in samples))
        self.assertTrue(accept_corroborated_menu(samples, focused=True))

    def test_three_frames_do_not_authorize_restart(self):
        self.assertFalse(accept_corroborated_menu([self.reading() for _ in range(3)], focused=True))

    def test_focus_loss_is_not_recoverable(self):
        self.assertFalse(accept_corroborated_menu([self.reading() for _ in range(4)], focused=False))

    def test_ml_only_menu_is_not_enough(self):
        self.assertFalse(accept_corroborated_menu([self.reading(score=.1) for _ in range(4)], focused=True))

    def test_no_confident_menu_is_not_enough(self):
        self.assertFalse(accept_corroborated_menu([self.reading(p_menu=.89) for _ in range(4)], focused=True))

    def test_intervening_gameplay_blocks_recovery(self):
        samples=[self.reading(), self.reading(), self.reading(state='GAMEPLAY',p_menu=0.,p_game=.999,score=.1),self.reading()]
        self.assertFalse(accept_corroborated_menu(samples,focused=True))

    def test_one_weak_template_blocks_recovery(self):
        samples=[self.reading() for _ in range(3)]+[self.reading(score=.52)]
        self.assertFalse(accept_corroborated_menu(samples, focused=True))

    def test_rejects_nonfinite_data(self):
        samples=[self.reading() for _ in range(3)]+[self.reading(score=float('nan'))]
        self.assertFalse(accept_corroborated_menu(samples, focused=True))

    def test_valid_strong_menu_is_accepted(self):
        self.assertTrue(accept_corroborated_menu([self.reading(score=.992) for _ in range(4)],focused=True))

    def test_installer_replaces_only_exact_v12_method(self):
        original = 'def no_op():\n    pass\n\nclass Dummy:\n'+OLD_METHOD+\
                   'def flags():\n    x="--resync-rollouts"\n    y="--recover-rollout-menu"\n'
        updated=patch(original)
        self.assertIn(NEW_METHOD,updated)
        self.assertNotIn(OLD_METHOD,updated)
        self.assertEqual(updated.replace(NEW_METHOD,OLD_METHOD),original)
        compile(updated,'<test>', 'exec')
        with self.assertRaisesRegex(ValueError,'already installed'):
            patch(updated)
        with self.assertRaises(ValueError):
            patch(original.replace(OLD_METHOD, '    def something_other():\n        pass\n'))

if __name__=='__main__': unittest.main()
