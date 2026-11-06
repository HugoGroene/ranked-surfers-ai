"""Episode 1566 mixed classifier vote: fresh proof required, no live game."""
import unittest
from tests.test_ppo_pause_runtime_v179 import run, frame
from tools.ppo_resync_transient_v172 import ResyncWindow


def mixed():
    # Probabilities rounded as recorded in the saved worker log.
    return frame('UNCERTAIN', pg=.649, pm=.344, score=.161)


class MixedTransitionTests(unittest.TestCase):
    def test_pause_then_partial_stack_then_mixed_requires_four_new_valid_frames(self):
        e,obs,reason,log=run([frame(paused=True),frame(),frame(),mixed()]+[frame()]*4)
        self.assertIsNone(reason)
        self.assertEqual(obs,('img5','img6','img7','img8'))
        self.assertEqual(e.index,8)
        self.assertEqual(e.packed,6)  # Two earlier frames discarded, four new.
        self.assertEqual(e.keys,[])
        self.assertEqual(log.count('[PPO RESYNC TRACE] frame='),4)
        self.assertIn('discarded_valid=2',log)

    def test_catch_animation_can_request_existing_menu_confirmation(self):
        e,obs,reason,log=run([mixed(),frame('MENU',pg=.001,pm=.999,score=.70)])
        self.assertIsNone(obs)
        self.assertIsNotNone(e._ppo_post_resume_menu_candidate)
        self.assertIn('state=MENU',reason)
        self.assertEqual(e.packed,0)
        self.assertEqual(e.keys,[])

    def test_persistent_mixed_predictions_exhaust_skip_budget(self):
        e,obs,reason,_=run([mixed()]*5)
        self.assertIsNone(obs)
        self.assertIn('skips=4',reason)
        self.assertEqual(e.index,5)
        self.assertEqual(e.packed,0)
        self.assertEqual(e.keys,[])

    def test_total_capture_budget_still_applies_after_late_uncertainty(self):
        e,obs,reason,_=run([frame(paused=True)]+[frame()]*3+[mixed()]+[frame()]*3)
        self.assertIsNone(obs)
        self.assertIn('within 8 captures',reason)
        self.assertEqual(e.index,8)
        self.assertEqual(e.keys,[])

    def test_low_template_mixed_prediction_is_not_a_gameplay_vote(self):
        w=ResyncWindow()
        decision=w.observe(state='UNCERTAIN',probs={'GAMEPLAY':.649,'MENU':.344},
                           menu_score=.161,confidence=.90,menu_threshold=.68)
        self.assertEqual(decision,'RETRY')
        self.assertEqual(w.accepted,0)
        self.assertFalse(w.menu_candidate)

    def test_menu_dominance_template_conflict_invalid_and_other_states_stop(self):
        for item in [frame('UNCERTAIN',pg=.4,pm=.59,score=.161),
                     frame('UNCERTAIN',pg=.5,pm=.2,score=.161),
                     frame('UNCERTAIN',pg=.649,pm=.344,score=.35),
                     frame('UNCERTAIN',pg=float('nan'),pm=.344,score=.161),
                     frame('LOADING',pg=.649,pm=.344,score=.161)]:
            with self.subTest(frame=item):
                e,obs,reason,_=run([item])
                self.assertIsNone(obs)
                self.assertIsNotNone(reason)
                self.assertEqual(e.index,1)
                self.assertEqual(e.packed,0)
                self.assertEqual(e.keys,[])


if __name__=='__main__':unittest.main()
