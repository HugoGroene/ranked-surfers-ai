"""Replay the November 1 resync failure without browser/model side effects."""
import unittest
from tools.ppo_resync_transient_v172 import ResyncWindow
from tools.ppo_menu_guard_alignment_v14 import accept_menu_after_transient
from rs_v612_gameplay_guard import screen_gate
from tests.test_ppo_resync_live_v178 import run, f


class CandidateTests(unittest.TestCase):
    def observe(self,state='UNCERTAIN',pg=.088,pm=.889,score=.656):
        w=ResyncWindow()
        result=w.observe(state=state,probs={'GAMEPLAY':pg,'MENU':pm},
                         menu_score=score,confidence=.90,menu_threshold=.68)
        return w,result

    def test_exact_failure_only_requests_confirmation(self):
        w,result=self.observe()
        self.assertEqual(result,'ABORT')  # Stop refresh, not a gameplay vote.
        self.assertTrue(w.menu_candidate)
        self.assertEqual(w.accepted,0)

    def test_actual_refresh_sets_anchor_without_keys_or_packing(self):
        e,obs,reason,log=run([f('UNCERTAIN',.088,.889,.656)])
        self.assertIsNone(obs)
        self.assertIsNotNone(e._ppo_post_resume_menu_candidate)
        self.assertEqual(len(e._ppo_post_resume_menu_candidate),3)
        self.assertEqual(e.packed,0)
        self.assertEqual(e.keys,[])
        self.assertIn('Candidate requests independent confirmation',log)

    def test_ml_alone_weak_menu_and_other_states_do_not_request(self):
        for kwargs in ({'score':.1},{'pm':.7},{'pg':.2},
                       {'state':'OTHER'},{'state':'PAUSED'},{'state':'GAMEPLAY'},
                       {'score':float('nan')},{'pm':float('inf')}):
            with self.subTest(kwargs=kwargs):
                w,_=self.observe(**kwargs)
                self.assertFalse(w.menu_candidate)

    def test_uncertain_candidate_is_not_terminal_even_when_repeated(self):
        r=screen_gate('UNCERTAIN',{'GAMEPLAY':.088,'MENU':.889},.656,
                      confidence=.9,menu_threshold=.68)
        self.assertFalse(accept_menu_after_transient([r]*4,focused=True))

    def test_four_fresh_confident_corroborated_frames_still_required(self):
        good=screen_gate('MENU',{'GAMEPLAY':.001,'MENU':.999},.656,
                         confidence=.9,menu_threshold=.68)
        weak=screen_gate('MENU',{'GAMEPLAY':.001,'MENU':.999},.1,
                         confidence=.9,menu_threshold=.68)
        self.assertFalse(accept_menu_after_transient([good]*3,focused=True))
        self.assertFalse(accept_menu_after_transient([good]*3+[weak],focused=True))
        self.assertFalse(accept_menu_after_transient([good]*4,focused=False))
        self.assertTrue(accept_menu_after_transient([weak]+[good]*4,focused=True))


if __name__=='__main__':unittest.main()
