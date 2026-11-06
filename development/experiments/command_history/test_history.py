import unittest
from types import SimpleNamespace
import numpy as np
from history import CommandHistory, SLOTS, WIDTH
from command_adapter import history_env


class HistoryTests(unittest.TestCase):
    def test_order_repeat_actions_and_age(self):
        h=CommandHistory()
        for action,t in [(3,10.),(4,10.2),(3,10.4)]:h.add(action,t)
        v=h.vector(10.5).reshape(SLOTS,WIDTH)
        self.assertTrue(np.all(v[:-3]==0))
        self.assertEqual(v[-3:,:5].argmax(1).tolist(),[3,4,3])
        np.testing.assert_allclose(v[-3:,5],[.5,.3,.1])
        self.assertTrue(np.all(v[-3:,6]==1))

    def test_anchor_excludes_future_and_expired_events(self):
        h=CommandHistory()
        h.add(1,10.);h.add(2,10.5)
        v=h.vector(10.25).reshape(SLOTS,WIDTH)
        self.assertEqual(v[:,6].sum(),1)
        self.assertEqual(v[-1,:5].argmax(),1)
        self.assertEqual(h.vector(12.).sum(),0)

    def test_capacity_under_cap_and_overflow_telemetry(self):
        h=CommandHistory()
        for i in range(31):h.add(i%5,10+i/30)
        self.assertEqual(h.overflows,0)
        self.assertEqual(h.vector(11.).reshape(SLOTS,WIDTH)[:,6].sum(),31)
        h.clear()
        for i in range(40):h.add(i%5,20+i*.02)
        self.assertEqual(len(h.events),32)
        self.assertEqual(h.overflows,8)

    def test_worker_isolation_and_reset(self):
        a,b=CommandHistory(),CommandHistory()
        a.add(3,10);b.add(4,10)
        a.clear()
        self.assertEqual(a.vector(10).sum(),0)
        self.assertEqual(b.vector(10).reshape(SLOTS,WIDTH)[-1,4],1)

    def test_invalid_clocks_and_actions(self):
        h=CommandHistory();h.add(0,10)
        for action,stamp in [(5,11),(True,11),(3.5,11),(3,9),(3,float('nan'))]:
            with self.assertRaises(ValueError):h.add(action,stamp)
        with self.assertRaises(ValueError):h.vector(float('inf'))


class FakeBase:
    def __init__(self):
        from gymnasium import spaces
        from hybrid import HybridHistory
        self.temporal=HybridHistory(1,(.05,.25,.75))
        self.observation_space=spaces.Dict(dict(image=spaces.Box(0,255,(12,2,2),np.uint8),
            aux=spaces.Box(0.,1.,(565,),np.float32)))
        self.step_timings=[];self.action_context=SimpleNamespace(last_action_times=[None]*5)
        self._headless_capture_timing={'request_started_at':10.5}
        self.block=False;self.suppress=False
    async def _step_browser(self,action):
        if not self.block:
            self.action_context.last_action_times[action]=10.
            self.step_timings.append(dict(action=action,press=action!=0 and not self.suppress,
                                          press_timestamp=10. if action else None))
        return 'captured'
    def _pack(self):
        return dict(image=np.zeros((12,2,2),np.uint8),aux=np.arange(565,dtype=np.float32)/565),False,None
    def reset(self):return self._pack()[0]
    def pause_for_ppo_optimizer(self):return 'paused'
    def refresh_after_ppo_optimizer(self):return self._pack()[0],None,10.5


class AdapterTests(unittest.IsolatedAsyncioTestCase):
    def env(self):return history_env(FakeBase)()
    async def test_post_dispatch_pack_preserves_prefix(self):
        env=self.env()
        await env._step_browser(3)
        obs,_,_=env._pack()
        self.assertEqual(obs['aux'].shape,(789,))
        np.testing.assert_array_equal(obs['aux'][:565],np.arange(565,dtype=np.float32)/565)
        self.assertEqual(obs['aux'][565:].reshape(SLOTS,WIDTH)[-1,3],1)
        self.assertIs(env._learning_current_observation,obs)
    async def test_blocked_and_suppressed_commands_are_not_events(self):
        env=self.env();env.block=True
        await env._step_browser(3)
        env.block=False;env.suppress=True
        await env._step_browser(2)
        self.assertEqual(env.command_timeline.total_events,0)
    async def test_noop_and_clear_before_reset_pause_resync(self):
        env=self.env();await env._step_browser(0)
        self.assertEqual(env.command_timeline.events[-1],(10.,0))
        self.assertFalse(env.reset()['aux'][565:].any())
        await env._step_browser(3);env.pause_for_ppo_optimizer()
        self.assertEqual(len(env.command_timeline.events),0)
        await env._step_browser(4)
        self.assertFalse(env.refresh_after_ppo_optimizer()[0]['aux'][565:].any())


if __name__=='__main__':unittest.main()
