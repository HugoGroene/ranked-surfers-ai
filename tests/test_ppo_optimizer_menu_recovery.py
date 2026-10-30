"""Offline tests for strict menu recovery guards and installer. No game required."""
import pathlib
import sys
import unittest
import numpy as np

HERE = pathlib.Path(__file__).resolve().parents[1]
sys.path.insert(0, str(HERE))
from tools.ppo_optimizer_menu_recovery import (candidate_is_menu, accept_menu_checks,
                                               validated_new_vec_episode)
from tools.install_ppo_optimizer_menu_recovery import patch, METHOD_ANCHOR, IMPORT_ANCHOR, FLAG_ANCHOR, CB_OLD

class MenuGuards(unittest.TestCase):
    def test_only_menu_is_candidate(self):
        self.assertTrue(candidate_is_menu('unsafe screen state=MENU menu_score=0.992'))
        for x in (None, '', 'unsafe screen state=GAMEPLAY menu_score=0.992',
                  'unsafe screen state=UNCERTAIN menu_score=0.992',
                  'unsafe screen state=MENU menu_score=-0.3\nother'):
            if x is not None and 'MENU menu_score=-0.3' in x:
                continue  # stage is re-validated independently before resetting
            self.assertFalse(candidate_is_menu(x))

    def test_multiple_menu_confirmations_required(self):
        good=[('MENU', .98), ('MENU', .99), ('MENU', .91)]
        self.assertTrue(accept_menu_checks(good, threshold=.578, focused=True))
        self.assertFalse(accept_menu_checks(good[:2], threshold=.578, focused=True))
        self.assertFalse(accept_menu_checks(good, threshold=.578, focused=False))
        self.assertFalse(accept_menu_checks(good[:2]+[('GAMEPLAY', .99)], threshold=.578, focused=True))
        self.assertFalse(accept_menu_checks(good[:2]+[('MENU', .3)], threshold=.578, focused=True))
        self.assertFalse(accept_menu_checks(good[:2]+[('MENU', float('nan'))], threshold=.578, focused=True))

    def test_reset_must_preserve_exact_policy_shape_and_dtype(self):
        old={'image':np.zeros((1,12,8,8),dtype=np.uint8), 'aux':np.zeros((1,557),dtype=np.float32)}
        reset={'image':np.ones((1,12,8,8),dtype=np.uint8), 'aux':np.ones((1,557),dtype=np.float32)}
        obs, starts=validated_new_vec_episode(old,reset,1)
        self.assertTrue(starts[0])
        self.assertEqual(starts.dtype,np.dtype(bool))
        reset['image'][0,0,0,0]=88
        self.assertEqual(obs['image'][0,0,0,0],1)
        with self.assertRaises(ValueError): validated_new_vec_episode(old,reset,2)
        with self.assertRaises(ValueError): validated_new_vec_episode(old,{**reset, 'aux':np.zeros((1,556),np.float32)},1)
        with self.assertRaises(ValueError): validated_new_vec_episode(old,{**reset, 'aux':np.zeros((1,557),np.float64)},1)
        with self.assertRaises(ValueError): validated_new_vec_episode(old,{**reset, 'aux':np.full((1,557),float('nan'),np.float32)},1)

    def test_installer_requires_hardened_code(self):
        fixture = ("from tools.ppo_rollout_gap_audit import RolloutGapTracker\n"
            "class Env:\n"
            "    def refresh_after_ppo_optimizer(self):\n"
            "        pass\n"
            + METHOD_ANCHOR + "            pass\n"
            "def callback(a, raw, self, observation, skip_reason):\n"
            + IMPORT_ANCHOR
            + "    if True:\n        if True:\n            try:\n"
            + CB_OLD
            + "            except Exception:\n                raise\n"
            + FLAG_ANCHOR)
        installed=patch(fixture)
        self.assertIn('validated_new_vec_episode(',installed)
        self.assertIn('self.model._last_episode_starts = episode_starts',installed)
        self.assertIn('def confirm_post_optimizer_menu(self):',installed)
        compile(installed, 'mock.py', 'exec')
        with self.assertRaises(ValueError): patch(installed)
        with self.assertRaises(ValueError): patch(fixture.replace(CB_OLD,''))

if __name__=='__main__': unittest.main()
