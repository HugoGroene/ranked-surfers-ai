"""Offline first-Practice sequencing; no browser/game launched."""
import sys
from pathlib import Path
sys.path.insert(0,str(Path(__file__).resolve().parents[3]))
import unittest
from types import SimpleNamespace
from unittest.mock import AsyncMock, Mock, patch
from visible_browser import install_practice_start
from visible_startup import start_ready_practice
from rs_v612_gameplay_guard import GuardUncertainError


class StartupTests(unittest.IsolatedAsyncioTestCase):
    async def test_practice_is_not_clicked_during_environment_construction(self):
        import rs_practice_controller as ctrl
        button=SimpleNamespace(is_visible=AsyncMock(return_value=True),
            is_enabled=AsyncMock(return_value=True),evaluate=AsyncMock())
        handle=SimpleNamespace(evaluate=AsyncMock(),dispose=AsyncMock())
        button.element_handle=AsyncMock(return_value=handle)
        buttons=SimpleNamespace(count=AsyncMock(return_value=1),nth=Mock(return_value=button))
        page=SimpleNamespace(url='http://localhost:8000/',frames=[SimpleNamespace(
            get_by_role=Mock(return_value=buttons))])
        with patch.object(ctrl,'click_practice'):
            install_practice_start(Path('unused'),defer_first=True)
            await ctrl.click_practice(page)
            button.evaluate.assert_not_awaited()
            handle.evaluate.assert_not_awaited()
            # Simulate layout hiding it: resolving the old role locator again
            # would now time out. Activation must use the retained DOM handle.
            button.evaluate.side_effect=TimeoutError('hidden role locator')
            button.element_handle.side_effect=TimeoutError('hidden role locator')
            await page._ppo_pending_practice()
            handle.evaluate.assert_awaited_once()
            handle.dispose.assert_awaited_once()
            page.url='https://rankedsurfers.com/'
            with self.assertRaises(GuardUncertainError):await page._ppo_pending_practice()

    def env(self, states, score=0.):
        return SimpleNamespace(page=SimpleNamespace(), canvas=object(),
            _capture=AsyncMock(return_value=('pixels',score,None)),
            _read_screen_state=Mock(side_effect=[(s,{}) for s in states]),
            startup_timeout_s=1.,startup_grace_s=.1,args=SimpleNamespace(threshold=.68),
            startup_guard_ms=[],startup_failures=0,_encode_last_frame=Mock(return_value=b'preview'))

    async def test_adopt_first_game_only_after_warmup_and_two_votes(self):
        env=self.env(['MENU','GAMEPLAY','GAMEPLAY'])
        async def activate():
            self.assertEqual(env._read_screen_state.call_count,1)
        pending=AsyncMock(side_effect=activate);restart=AsyncMock()
        with patch('rs_practice_controller.focus_canvas',AsyncMock()), \
             patch('visible_startup.asyncio.sleep',AsyncMock()), \
             patch('visible_startup.time.perf_counter',side_effect=[0.,0.,.1,.1,.2]):
            result=await start_ready_practice(env,pending,restart)
        self.assertEqual(result,('pixels',b'preview'))
        restart.assert_not_awaited();pending.assert_awaited_once()
        self.assertIsNone(env.page._ppo_pending_practice)
        self.assertEqual(len(env.startup_guard_ms),1)

    async def test_one_gameplay_vote_then_uncertain_does_not_start(self):
        env=self.env(['MENU','GAMEPLAY','UNCERTAIN'])
        with patch('rs_practice_controller.focus_canvas',AsyncMock()), \
             patch('visible_startup.asyncio.sleep',AsyncMock()), \
             patch('visible_startup.time.perf_counter',side_effect=[0.,0.,.1,.1,.2,2.]):
            with self.assertRaises(GuardUncertainError):
                await start_ready_practice(env,AsyncMock(),AsyncMock())
        self.assertEqual(env.startup_failures,1)

    async def test_menu_routes_through_existing_guarded_restart(self):
        env=self.env(['MENU']*4,score=.9);restart=AsyncMock(return_value=('fresh',b'preview'))
        with patch('rs_practice_controller.focus_canvas',AsyncMock()), \
             patch('visible_startup.asyncio.sleep',AsyncMock()), \
             patch('visible_startup.time.perf_counter',side_effect=[0.,0.,.2,.2,.4,.4,.8]):
            self.assertEqual(await start_ready_practice(env,AsyncMock(),restart),('fresh',b'preview'))
        restart.assert_awaited_once()


if __name__=='__main__':unittest.main()
