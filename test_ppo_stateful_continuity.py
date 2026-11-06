import unittest
from tools.ppo_stateful_continuity import Continuity
from tools.ppo_resync_transient_v172 import ResyncWindow

class Tests(unittest.TestCase):
    def observe(self,c,**changes):
        args=dict(now=1.,fresh=True,focused=True,paused=False,state='UNCERTAIN',
                  score=0.,threshold=.68,hud_present=True)
        args.update(changes)
        return c.observe(**args)
    def test_unknown_not_play(self):self.assertFalse(self.observe(Continuity()))
    def test_verified_then_uncertain(self):
        c=Continuity()
        self.assertTrue(self.observe(c,confident=True,state='GAMEPLAY'))
        self.assertTrue(self.observe(c,now=1.3))
        self.assertFalse(self.observe(c,now=1.8))
    def test_visual_renewal(self):
        c=Continuity();self.assertTrue(self.observe(c,advancing=True))
        self.assertTrue(self.observe(c,now=1.7,advancing=True))
        self.assertTrue(self.observe(c,now=2.))
    def test_interruptions_revoke(self):
        for change in [dict(paused=True),dict(fresh=False),dict(focused=False),
                       dict(state='MENU'),dict(state='OTHER'),dict(state='LOADING'),
                       dict(score=.3),dict(hud_present=False),dict(now=.5)]:
            c=Continuity();self.observe(c,advancing=True)
            self.assertFalse(self.observe(c,**change))
            self.assertFalse(self.observe(c,now=1.1))
    def test_resync_requires_four(self):
        w=ResyncWindow()
        args=dict(state='UNCERTAIN',probs={'GAMEPLAY':.35,'MENU':.64},menu_score=.08,
                  confidence=.9,menu_threshold=.68,continuity=True)
        self.assertEqual([w.observe(**args) for _ in range(4)],['ACCEPT','ACCEPT','ACCEPT','DONE'])
    def test_menu_not_overridden(self):
        w=ResyncWindow()
        self.assertEqual(w.observe(state='MENU',probs={'MENU':.99},menu_score=.8,
             confidence=.9,menu_threshold=.68,continuity=True),'ABORT')
        self.assertTrue(w.menu_candidate)

    def test_completion_tail_requires_existing_evidence(self):
        from tools.ppo_resync_transient_v172 import completion_tail_frames
        self.assertEqual([completion_tail_frames(n,True) for n in range(5)],[0,3,2,1,0])
        self.assertEqual(completion_tail_frames(3,False),0)

    def test_late_transient_can_finish_but_second_rejection_exhausts_tail(self):
        from tools.ppo_resync_transient_v172 import completion_tail_frames
        args=dict(state='GAMEPLAY',probs={'GAMEPLAY':1.},menu_score=0.,confidence=.9,menu_threshold=.68)
        for reject_tail in (False,True):
            w=ResyncWindow()
            for _ in range(3):w.observe(**args)
            self.assertEqual(w.observe(**dict(args,state='UNCERTAIN',probs={'GAMEPLAY':.4},continuity_pending=True)),'RETRY')
            for _ in range(3):w.observe(**args)
            self.assertEqual(completion_tail_frames(w.accepted,True),1)
            result=w.observe(**(dict(args,state='UNCERTAIN',probs={'GAMEPLAY':.4},continuity_pending=True) if reject_tail else args))
            self.assertEqual(result,'RETRY' if reject_tail else 'DONE')

class AsyncTests(unittest.IsolatedAsyncioTestCase):
    def raw(self):
        import numpy as np
        from types import SimpleNamespace
        from unittest.mock import AsyncMock,Mock
        page=SimpleNamespace(evaluate=AsyncMock(return_value=True),
            locator=Mock(return_value=SimpleNamespace(is_visible=AsyncMock(return_value=False))))
        return SimpleNamespace(_stateful_episode_verified=True,page=page,
            state_confidence=.9,args=SimpleNamespace(threshold=.68),last_capture_end=10.,
            _v2_last_bgra=None,crop=None,startup_motion_threshold=2.2), (np.ones((8,8),dtype=np.uint8)*255,np.ones((8,8),dtype=np.uint8))
    async def test_confident_path_no_browser_calls(self):
        from tools.ppo_stateful_continuity import verify
        from unittest.mock import patch
        raw,views=self.raw()
        with patch('time.perf_counter',return_value=10.01):
            self.assertFalse(await verify(raw,'GAMEPLAY',{'GAMEPLAY':1.},0.,views=views))
        raw.page.evaluate.assert_not_awaited()
        raw.page.locator.assert_not_called()
        self.assertGreater(raw._stateful_continuity.until,10.01)
    async def test_uncertain_still_checks_pause(self):
        from tools.ppo_stateful_continuity import verify
        from unittest.mock import patch
        raw,views=self.raw();raw._stateful_continuity=Continuity();raw._stateful_continuity.until=11.
        raw.page.locator.return_value.is_visible.return_value=True
        with patch('time.perf_counter',return_value=10.01):
            self.assertFalse(await verify(raw,'UNCERTAIN',{'GAMEPLAY':.7},0.,views=views))
        raw.page.evaluate.assert_awaited_once()

    async def test_resync_confident_frame_seeds_continuity(self):
        from tools.ppo_stateful_continuity import verify
        from unittest.mock import patch,Mock
        raw,views=self.raw();raw._meter_mask=Mock(return_value=views[0]);raw._motion_view=Mock(return_value=views[1])
        with patch('time.perf_counter',return_value=10.01):
            self.assertFalse(await verify(raw,'GAMEPLAY',{'GAMEPLAY':1.},0.,resync=True))
            self.assertTrue(await verify(raw,'UNCERTAIN',{'GAMEPLAY':.4},0.,resync=True))
        self.assertIsNotNone(raw._stateful_evidence)
    async def test_rpc_delay_cannot_authorize_stale_frame(self):
        from tools.ppo_stateful_continuity import verify
        from unittest.mock import patch
        raw,views=self.raw();raw._stateful_continuity=Continuity();raw._stateful_continuity.until=11.
        with patch('time.perf_counter',side_effect=[10.01,10.2]):
            self.assertFalse(await verify(raw,'UNCERTAIN',{'GAMEPLAY':.7},0.,views=views))

if __name__=='__main__':unittest.main()
