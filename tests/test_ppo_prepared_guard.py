import unittest
from collections import deque
from types import SimpleNamespace
from unittest.mock import AsyncMock, Mock, patch

import numpy as np

from tools.ppo_prepared_guard import PreparedGameplayReading


class PreparedGuardTests(unittest.TestCase):
    def test_same_image_freshness_bound_and_future_timestamp(self):
        frame = object()
        reading = PreparedGameplayReading(id(frame), 10., 'GAMEPLAY', {'GAMEPLAY': 1.}, 0.)
        self.assertTrue(reading.valid_for(frame, 10., 10.019))
        self.assertFalse(reading.valid_for(frame, 10., 10.021))
        self.assertFalse(reading.valid_for(frame, 10., 9.999))

    def test_new_capture_invalidates_even_if_object_or_timestamp_matches(self):
        frame = object()
        reading = PreparedGameplayReading(id(frame), 10., 'GAMEPLAY', {}, 0.)
        self.assertFalse(reading.valid_for(object(), 10., 10.01))
        self.assertFalse(reading.valid_for(frame, 10.001, 10.01))


class ActualPreparedGuardTests(unittest.IsolatedAsyncioTestCase):
    async def test_recorded_two_dip_menu_uses_original_confirmation_without_movement(self):
        from tests.test_ppo_stream_menu_tail import two_dip_failure
        env=self.environment()
        env._headless_backend='electron'
        readings=two_dip_failure()
        scores=[r.menu_score for r in readings]+[.60]
        states=[r.state for r in readings]+['MENU']
        probabilities=[{'GAMEPLAY':r.gameplay_probability,'MENU':r.menu_probability}
                       for r in readings]+[{'MENU':1.}]
        env._capture=AsyncMock(side_effect=[(env._v2_last_chw,s,None) for s in scores])
        env._read_screen_state=Mock(side_effect=list(zip(states,probabilities)))
        with patch('time.perf_counter',return_value=10.05),patch('asyncio.sleep',new=AsyncMock()):
            await env._step_browser(1)
        self.assertEqual(env._capture.await_count,7)
        self.assertTrue(env._guard_forced_terminal)
        self.assertFalse(env._guard_hard_stopped)
        env.page.keyboard.press.assert_not_awaited()
        env.action_context.note_action.assert_not_called()

    async def test_recorded_two_dip_menu_still_stops_without_new_corroboration(self):
        from tests.test_ppo_stream_menu_tail import two_dip_failure
        from rs_v612_gameplay_guard import GuardUncertainError
        env=self.environment()
        env._headless_backend='electron'
        readings=two_dip_failure()
        scores=[r.menu_score for r in readings]+[.10]*6
        states=[r.state for r in readings]+['MENU']*6
        env._capture=AsyncMock(side_effect=[(env._v2_last_chw,s,None) for s in scores])
        env._read_screen_state=Mock(side_effect=[(s,{s:1.}) for s in states])
        with patch('time.perf_counter',return_value=10.05),patch('asyncio.sleep',new=AsyncMock()):
            with self.assertRaisesRegex(GuardUncertainError,'SAFETY STOP'):
                await env._step_browser(1)
        self.assertEqual(env._capture.await_count,12)
        self.assertTrue(env._guard_hard_stopped)
        self.assertFalse(env._guard_forced_terminal)
        env.page.keyboard.press.assert_not_awaited()

    async def test_electron_menu_animation_confirms_without_movement(self):
        env=self.environment()
        env._headless_backend='electron'
        scores=[.560689865987547,.9994769327335497,.6310866392463845,
            .5213169475469391,.7388194849442891,.6766063125611257,.60]
        states=['GAMEPLAY']+['MENU']*6
        env._capture=AsyncMock(side_effect=[(env._v2_last_chw,s,None) for s in scores])
        env._read_screen_state=Mock(side_effect=[(s,{s:1.}) for s in states])
        with patch('time.perf_counter',return_value=10.05),patch('asyncio.sleep',new=AsyncMock()):
            await env._step_browser(1)
        self.assertEqual(env._capture.await_count,7)
        self.assertTrue(env._guard_forced_terminal)
        env.page.keyboard.press.assert_not_awaited()

    async def test_electron_menu_extension_stops_on_gameplay_disagreement(self):
        from rs_v612_gameplay_guard import GuardUncertainError
        env=self.environment()
        env._headless_backend='electron'
        scores=[.560689865987547,.9994769327335497,.6310866392463845,
            .5213169475469391,.7388194849442891,.6766063125611257,0.]
        states=['GAMEPLAY']+['MENU']*5+['GAMEPLAY']
        env._capture=AsyncMock(side_effect=[(env._v2_last_chw,s,None) for s in scores])
        env._read_screen_state=Mock(side_effect=[(s,{s:1.}) for s in states])
        with patch('time.perf_counter',return_value=10.05),patch('asyncio.sleep',new=AsyncMock()):
            with self.assertRaises(GuardUncertainError):await env._step_browser(1)
        self.assertTrue(env._guard_hard_stopped)
        env.page.keyboard.press.assert_not_awaited()

    def environment(self, *, expired=False, next_menu=False):
        from train_rs_v612_ppo import build_env_class
        env = object.__new__(build_env_class())
        env._guard_hard_stopped = False
        env._guard_forced_terminal = False
        env.reuse_observation_guard = True
        env._v2_last_bgra = np.zeros((20, 20, 4), dtype=np.uint8)
        env._v2_last_chw = np.zeros((3, 144, 256), dtype=np.uint8)
        env.last_capture_end = 10.
        env._prepared_gameplay_reading = PreparedGameplayReading(
            id(env._v2_last_bgra), 10., 'GAMEPLAY', {'GAMEPLAY': 1.}, 0.)
        env.suspected_since = None
        env._ppo_boundary_menu_pending = False
        env.prepared_guard_hits = env.prepared_guard_fallbacks = 0
        env.prepared_guard_late_fallbacks = 0
        env.guard_allowed = env.guard_rechecks = env.guard_confirmed_menu = env.guard_aborts = 0
        env.guard_ml_menu_disagreements = env.guard_visual_recoveries = 0
        env.action_context = SimpleNamespace(note_action=Mock())
        env.page = SimpleNamespace(keyboard=SimpleNamespace(press=AsyncMock()))
        env.args = SimpleNamespace(threshold=.68, fps=30)
        env.state_confidence = .9
        env.startup_motion_threshold = 2.2
        env._motion_view = lambda *args: np.zeros((4, 4), dtype=np.uint8)
        env._meter_mask = lambda *args: np.zeros((4, 4), dtype=np.uint8)
        env.crop = None
        env._capture = AsyncMock(return_value=(env._v2_last_chw, .95 if next_menu else 0., None))
        env._read_screen_state = Mock(return_value=(
            ('MENU', {'MENU': 1.}) if next_menu else ('GAMEPLAY', {'GAMEPLAY': 1.})))
        env._guard_diagnostic = Mock(return_value='no filesystem diagnostic')
        env.menu_scores = deque()
        env.clear_streak = 0
        env.episode_number = 1
        env.probe_wait_mode = 'none'
        env.probe_press_delay_ms = 5
        env.step_timings = []
        return env

    async def test_actual_guard_consumes_reading_once_without_extra_pre_key_grab(self):
        env = self.environment()
        with patch('time.perf_counter', return_value=10.010):
            await env._step_browser(1)
        self.assertEqual(env._capture.await_count, 1)  # inherited post-key capture only
        env._read_screen_state.assert_not_called()
        env.page.keyboard.press.assert_awaited_once_with('ArrowLeft', delay=5)
        self.assertIsNone(env._prepared_gameplay_reading)
        self.assertEqual(env.prepared_guard_hits, 1)

    async def test_expired_reading_uses_fresh_capture_and_blocks_menu(self):
        env = self.environment(next_menu=True)
        with patch('time.perf_counter', return_value=10.021), patch('asyncio.sleep', new=AsyncMock()):
            await env._step_browser(1)
        env.page.keyboard.press.assert_not_awaited()
        self.assertTrue(env._guard_forced_terminal)
        self.assertGreaterEqual(env._capture.await_count, 2)

    async def test_changed_capture_uses_fresh_guard_even_with_recent_timestamp(self):
        env = self.environment(next_menu=True)
        env._v2_last_bgra = env._v2_last_bgra.copy()
        with patch('time.perf_counter', return_value=10.010), patch('asyncio.sleep', new=AsyncMock()):
            await env._step_browser(2)
        env.page.keyboard.press.assert_not_awaited()
        self.assertTrue(env._guard_forced_terminal)

    async def test_pending_menu_handoff_never_uses_prepared_gameplay_to_send_key(self):
        env = self.environment(next_menu=True)
        env._ppo_boundary_menu_pending = True
        with patch('time.perf_counter', return_value=10.010), patch('asyncio.sleep', new=AsyncMock()):
            await env._step_browser(1)
        env.page.keyboard.press.assert_not_awaited()
        self.assertTrue(env._ppo_menu_handoff_terminal_observed)

    async def test_scheduler_stall_after_reading_rechecks_before_dispatch(self):
        env = self.environment(next_menu=True)
        calls = iter([10.010])
        def clock():
            return next(calls, 10.050)
        with patch('time.perf_counter', side_effect=clock), patch('asyncio.sleep', new=AsyncMock()):
            await env._step_browser(1)
        env.page.keyboard.press.assert_not_awaited()
        self.assertEqual(env.prepared_guard_late_fallbacks, 1)
        self.assertTrue(env._guard_forced_terminal)
