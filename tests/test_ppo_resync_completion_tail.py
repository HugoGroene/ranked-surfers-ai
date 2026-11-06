"""Replay the late-transient failure through the actual current refresh method."""
import unittest
from unittest.mock import patch
from tests.test_ppo_pause_runtime_v179 import frame,run


class CompletionTests(unittest.TestCase):
    def sequence(self):
        return [frame(paused=True)]+[frame()]*3+[frame('UNCERTAIN',pg=.408)]+[frame()]*3

    def test_recorded_pattern_finishes_one_extra_capture(self):
        with patch.dict('os.environ',{'RS_STATEFUL_GUARD':'1'}):
            env,obs,reason,log=run(self.sequence()+[frame()])
        self.assertIsNone(reason)
        self.assertEqual(obs,('img6','img7','img8','img9'))
        self.assertEqual(env.keys,[])
        self.assertIn('[PPO RESYNC TAIL]',log)

    def test_strict_mode_retains_eight_capture_limit(self):
        with patch.dict('os.environ',{'RS_STATEFUL_GUARD':'0'}):
            env,obs,reason,log=run(self.sequence())
        self.assertIsNone(obs)
        self.assertIn('within 8 captures',reason)
        self.assertEqual(env.keys,[])

    def test_rejection_in_tail_does_not_extend_again(self):
        with patch.dict('os.environ',{'RS_STATEFUL_GUARD':'1'}):
            env,obs,reason,log=run(self.sequence()+[frame('UNCERTAIN',pg=.408)])
        self.assertIsNone(obs)
        self.assertIn('within 9 captures',reason)
        self.assertEqual(env.keys,[])

    def test_menu_in_tail_still_stops(self):
        with patch.dict('os.environ',{'RS_STATEFUL_GUARD':'1'}):
            env,obs,reason,log=run(self.sequence()+[frame('MENU',pg=.01,pm=.99,score=.8)])
        self.assertIsNone(obs)
        self.assertIn('state=MENU',reason)
        self.assertEqual(env.keys,[])


if __name__=='__main__':unittest.main()
