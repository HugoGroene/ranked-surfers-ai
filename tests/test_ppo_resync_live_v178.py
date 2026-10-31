#!/usr/bin/env python3
"""Current-runtime uncertain-frame tests using the shared pause-aware fixture."""
import unittest
from tests import test_ppo_pause_runtime_v179 as pause_runtime


def f(state='GAMEPLAY',pg=1.0,pm=0.0,score=0.0):
    return (state,{'GAMEPLAY':pg,'MENU':pm},score)


def run(sequence,focus=True):
    # v1.7.9 requires a verified Pause fingerprint and DOM checks. Reuse the
    # current fixture so these older uncertainty regressions reach the gate.
    frames=[pause_runtime.frame(state,pg=probs['GAMEPLAY'],pm=probs['MENU'],score=score)
            for state,probs,score in sequence]
    return pause_runtime.run(frames,focus=focus)


class CurrentRefreshTests(unittest.TestCase):
    def test_547_confidence_frame_recovers_only_after_four_gameplay(self):
        e,obs,reason,log=run([f('UNCERTAIN',.547,.004,0.)]+[f()]*4)
        self.assertIsNone(reason)
        self.assertEqual(obs,('img2','img3','img4','img5'))
        self.assertEqual(e.packed,4)
        self.assertEqual(log.count('[PPO RESYNC TRACE] frame='),4)
        self.assertEqual(e.keys,[])

    def test_retries_are_bounded_and_never_authorize_uncertain(self):
        e,obs,reason,log=run([f('UNCERTAIN',.547,.004,0.)]*5)
        self.assertIsNone(obs)
        self.assertIn('skips=4',reason)
        self.assertEqual(e.packed,0)
        self.assertEqual(e.keys,[])

    def test_menu_candidate_is_not_packed_as_gameplay(self):
        e,obs,reason,log=run([f('MENU',.01,.99,.70)])
        self.assertIsNone(obs)
        self.assertIn('state=MENU',reason)
        self.assertIsNotNone(e._ppo_post_resume_menu_candidate)
        self.assertEqual(e.keys,[])

    def test_suspicious_menu_ml_probability_refuses_retry(self):
        e,obs,reason,log=run([f('UNCERTAIN',.547,.11,0.)])
        self.assertIsNone(obs)
        self.assertEqual(e.index,1)

    def test_loss_of_focus_stops_without_capture(self):
        e,obs,reason,log=run([f()]*4,focus=False)
        self.assertIsNone(obs)
        self.assertEqual(e.index,0)

    def test_four_normal_gameplay_frames_unchanged(self):
        e,obs,reason,log=run([f()]*4)
        self.assertIsNone(reason)
        self.assertEqual(e.index,4)
        self.assertEqual(e.packed,4)
        self.assertEqual(e.keys,[])

if __name__=='__main__': unittest.main()
