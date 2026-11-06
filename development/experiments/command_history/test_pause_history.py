import unittest
import numpy as np
from command_adapter import history_env
from history import CommandHistory
from hybrid import HybridHistory
from pause_history import PauseHistory
from test_history import FakeBase


class PauseTests(unittest.TestCase):
    def fixture(self):
        history=HybridHistory(1,(.05,.25,.75))
        frame=np.ones((3,2,2),np.uint8)
        history.observation(frame,100.5)
        for t in [100.5,100.75,100.95,101.]:history.add(frame,t,history.token())
        commands=CommandHistory();commands.add(3,100.8);commands.add(4,101.)
        pause=PauseHistory();generation=history.token()
        pause.snapshot(history,commands,(1,1))
        history.clear();commands.clear()
        token=(101.1,101.1,101.1,1.)
        pause.paused(token,(1,1));pause.resumed(token,106.1,(1,1))
        return history,commands,pause,frame,generation

    def test_shift_only_verified_pause_keep_active_resume_elapsed(self):
        history,commands,pause,frame,old_generation=self.fixture()
        fresh=[(106.1+i*.05,frame) for i in range(4)]
        self.assertTrue(pause.restore(history,commands,fresh,106.25,(1,1)))
        np.testing.assert_allclose([t for t,_ in commands.events],[105.8,106.])
        vector=commands.vector(106.25).reshape(32,7)
        np.testing.assert_allclose(vector[-2:,5],[.45,.25])
        image,aux=history.observation(frame,106.25)
        self.assertEqual(aux[-4:].tolist(),[1.,1.,1.,1.])
        self.assertAlmostEqual(aux[0]*(.75+.06),.75,places=6)
        count=len(history.items)
        history.add(frame,106.3,old_generation)
        self.assertEqual(len(history.items),count)
        self.assertEqual(pause.restores,1)
        self.assertAlmostEqual(pause.shifted_seconds,5.)

    def test_different_episode_never_restores(self):
        history,commands,pause,frame,_=self.fixture()
        self.assertFalse(pause.restore(history,commands,[(106.25,frame)],106.25,(2,2)))
        self.assertFalse(commands.events)
    def test_invalid_pause_delta_discards(self):
        history,commands,pause,frame,_=self.fixture()
        pause.resumed((101.1,101.1,101.1,1.),200.,(1,1))
        self.assertFalse(pause.restore(history,commands,[(200.,frame)],200.,(1,1)))
    def test_uncertain_discard_never_restores(self):
        history,commands,pause,frame,_=self.fixture();pause.discard()
        self.assertFalse(pause.restore(history,commands,[(106.25,frame)],106.25,(1,1)))
    def test_missing_pause_token_discards(self):
        history,commands,pause,frame,_=self.fixture();pause.paused(None,(1,1))
        self.assertIsNone(pause.saved)
    def test_second_pause_does_not_double_count_first(self):
        h,c,p,f,_=self.fixture();p.restore(h,c,[(106.25,f)],106.25,(1,1))
        p.snapshot(h,c,(1,1));token=(106.3,106.3,106.3,1.)
        p.paused(token,(1,1));p.resumed(token,108.3,(1,1))
        h.clear();c.clear();p.restore(h,c,[(108.4,f)],108.4,(1,1))
        np.testing.assert_allclose([t for t,_ in c.events],[107.8,108.])


class FreezeBase(FakeBase):
    def __init__(self):
        super().__init__();self.episode_number=1;self.episode_tag=1
        self._temporal_resync=False;self.uncertain=False
    def pause_for_ppo_optimizer(self):
        self.temporal.clear();self._ppo_pause_pending=(10.6,10.6,10.6,1.)
        return 0.
    def resume_after_ppo_optimizer(self,monitor):
        self._ppo_pause_pending=None;self._ppo_resume_trace_start_perf=15.6
        return {'paused_wall_seconds':5.}
    def _read_screen_state(self):return ('UNCERTAIN' if self.uncertain else 'GAMEPLAY'),{}
    def refresh_after_ppo_optimizer(self):
        self.temporal.clear();self._temporal_resync=True
        try:
            self._read_screen_state()
            for t in [15.6,15.65,15.7,15.75]:
                self._headless_capture_timing={'request_started_at':t}
                obs=self._pack()[0]
            return obs,None,15.76
        finally:self._temporal_resync=False


class IntegrationTests(unittest.TestCase):
    def env(self):
        env=history_env(FreezeBase)()
        frame=np.ones((3,2,2),np.uint8)
        env.temporal.observation(frame,10.)
        env.temporal.add(frame,10.,env.temporal.token())
        env.command_timeline.add(3,10.2)
        return env
    def test_resume_rebuilds_extended_observation_and_preserves_guard_clock(self):
        env=self.env();env.pause_for_ppo_optimizer();env.resume_after_ppo_optimizer(None)
        obs,reason,stamp=env.refresh_after_ppo_optimizer()
        self.assertEqual(stamp,15.76);self.assertEqual(env._headless_capture_timing['request_started_at'],15.75)
        self.assertEqual(obs['aux'].shape,(789,))
        self.assertEqual(obs['aux'][557+4],1.) # preserved oldest frame validity
        tail=obs['aux'][565:].reshape(32,7)
        self.assertEqual(tail[-1,3],1.)
        self.assertAlmostEqual(float(tail[-1,5]),.55,places=6)
        self.assertEqual(env._pause_history.restores,1)
    def test_uncertain_resync_drops_history(self):
        env=self.env();env.pause_for_ppo_optimizer();env.resume_after_ppo_optimizer(None)
        env.uncertain=True
        obs,_,_=env.refresh_after_ppo_optimizer()
        self.assertFalse(obs['aux'][565:].any())
        self.assertEqual(env._pause_history.restores,0)
    def test_reset_invalidates_saved_snapshot(self):
        env=self.env();env.pause_for_ppo_optimizer();env.reset()
        self.assertIsNone(env._pause_history.saved)


if __name__=='__main__':unittest.main()
