"""Real coordinate/crop operations and isolated async browser bootstrap tests."""
import tempfile
import unittest
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import AsyncMock, Mock, patch

import numpy as np

from rs_v612_gameplay_guard import GuardUncertainError
from tools.ppo_background_display import (
    BackgroundDisplayMixin, canvas_from_display, contains, select_display, validate_window)


def monitors():
    return [dict(left=-1920, top=0, width=3840, height=1080),
            dict(left=0, top=0, width=1920, height=1080, is_primary=True),
            dict(left=-1920, top=0, width=1920, height=1080, is_primary=False)]


class DisplayGeometryTests(unittest.TestCase):
    def test_reject_primary_union_and_missing_displays(self):
        for index in (0, 1, 3):
            with self.subTest(index=index), self.assertRaises(ValueError):
                select_display(monitors(), index)
        self.assertEqual(select_display(monitors(), 2)['left'], -1920)

    def test_negative_coordinate_canvas_matches_actual_pixels(self):
        display = monitors()[2]
        shot = np.zeros((1080, 1920, 4), np.uint8)
        shot[87:1032, :, :] = (23, 45, 67, 255)
        region = dict(left=-1920, top=87, width=1920, height=945)
        canvas = canvas_from_display(shot, display, region)
        self.assertEqual(canvas.shape, (945, 1920, 4))
        self.assertTrue(np.all(canvas == (23, 45, 67, 255)))
        for key, value in [('left', -1921), ('height', 1080)]:
            with self.subTest(key=key), self.assertRaises(GuardUncertainError):
                canvas_from_display(shot, display, dict(region, **{key: value}))
        with self.assertRaises(GuardUncertainError):
            canvas_from_display(shot[:1000], display, region)

    def test_minimized_hidden_scaled_and_moved_windows_fail_closed(self):
        display = monitors()[2]
        bounds = dict(display, windowState='normal')
        metrics = dict(dpr=1., visibility='visible')
        validate_window(display, bounds, metrics)
        cases = [(dict(bounds, windowState='minimized'), metrics),
                 (dict(bounds, left=0), metrics),
                 (bounds, dict(metrics, dpr=1.25)),
                 (bounds, dict(metrics, visibility='hidden'))]
        for b, m in cases:
            with self.subTest(bounds=b, metrics=m), self.assertRaises(GuardUncertainError):
                validate_window(display, b, m)

    def test_work_area_preserves_selected_monitor_identity(self):
        from tools.ppo_background_display import window_work_area
        # Non-Windows fallback has no native work-area API; never substitutes
        # monitor 1 or shifts a negative-coordinate secondary to the primary.
        with patch('sys.platform', 'linux'):
            work = window_work_area(monitors()[2])
        self.assertTrue(contains(monitors()[2], work))
        self.assertEqual(work['left'], -1920)

    def test_canvas_policy_is_same_crop_and_letterbox_as_bc(self):
        import rs_v5_segmentation_live_benchmark as bench
        from collections import deque
        env = BackgroundDisplayMixin()
        env._background_display_index = 2
        env._background_canvas_policy = True
        env._background_previews_saved = False
        env.region = monitors()[2]
        env._background_canvas = dict(left=-1920, top=87, width=1920, height=945)
        rng = np.random.default_rng(12)
        shot = rng.integers(0, 256, (1080, 1920, 4), dtype=np.uint8)
        env._v2_last_bgra = shot
        env.crop, env.target_wh, env.resize_method = bench.DEFAULT_CROP, (512, 288), 'linear'
        env.game_crop_ms = deque()
        import cv2
        expected_native = cv2.cvtColor(bench.crop_game(shot[87:1032], env.crop), cv2.COLOR_BGRA2RGB)
        expected = bench.fit_rgb_to_target(expected_native, env.target_wh, cv2.INTER_LINEAR)
        with tempfile.TemporaryDirectory() as tmp:
            env.output_dir = Path(tmp)
            chw, native = env._native_frames()
            np.testing.assert_array_equal(chw.transpose(1, 2, 0), expected)
            np.testing.assert_array_equal(native, expected_native)
            self.assertIs(env._v2_last_bgra, shot)  # raw guard lineage intact
            self.assertTrue((env.output_dir/'background_policy.png').exists())


class BackgroundBrowserTests(unittest.IsolatedAsyncioTestCase):
    async def test_native_title_delay_is_bounded_and_never_activates(self):
        from tools.benchmark_ppo_background import identify_window
        page = SimpleNamespace(title=AsyncMock(return_value='Original'), evaluate=AsyncMock())
        native = Mock()
        native.find.side_effect = [RuntimeError('title not presented'), 123]
        with patch('asyncio.sleep', new_callable=AsyncMock):
            self.assertEqual(await identify_window(page, native), 123)
        native.activate.assert_not_called()
        self.assertEqual(page.evaluate.await_args.args[-1], 'Original')

    async def test_window_verification_never_activates_or_sends_keys(self):
        env = BackgroundDisplayMixin()
        env.region = monitors()[2]
        env._background_window_id = 7
        env._background_cdp = SimpleNamespace(send=AsyncMock(return_value={'bounds': dict(env.region, windowState='normal')}))
        env.page = SimpleNamespace(bring_to_front=AsyncMock(), keyboard=Mock())
        view = dict(metrics=dict(dpr=1., visibility='visible'), canvas=dict(x=0, y=0, width=1900, height=900))
        env._background_view = view
        env._background_geometry = AsyncMock(return_value=view)
        await env._background_verify_window()
        env.page.bring_to_front.assert_not_awaited()
        self.assertEqual(env.page.keyboard.mock_calls, [])
        env._background_geometry.return_value = dict(view, canvas=dict(x=1))
        with self.assertRaises(GuardUncertainError):
            await env._background_verify_window()

    async def test_display_disappears_before_capture_marks_safety_stop(self):
        env = BackgroundDisplayMixin()
        env._background_display_index = 2
        env._background_checked_at = 0.
        env._background_verify_window = AsyncMock()
        with patch('mss.MSS') as factory:
            factory.return_value.__enter__.return_value.monitors = monitors()[:2]
            with self.assertRaises(GuardUncertainError):
                await env._sync_capture_region()
        self.assertTrue(env._guard_hard_stopped)
        env._background_verify_window.assert_not_awaited()


class CliAndReuseTests(unittest.TestCase):
    def test_failed_worker_has_no_successful_speedup_metric(self):
        from tools.benchmark_ppo_background import combine
        passed = combine([{'steps': 900}, {'steps': 800}], 90.)
        self.assertAlmostEqual(passed['aggregate_decisions_per_second'], 1700/90)
        self.assertFalse(passed['shared_learner'])
        self.assertFalse(passed['model_updated'])
        self.assertFalse(passed['input_response_verified'])
        failed = combine([{'steps': 900}, {'steps': 100, 'guard_hard_stopped': True}], 90.)
        self.assertEqual(failed['status'], 'STOPPED')
        self.assertIsNone(failed['aggregate_decisions_per_second'])

    def test_duplicate_worker_displays_rejected_before_launch(self):
        from tools.benchmark_ppo_background import main
        import contextlib
        import io
        with contextlib.redirect_stderr(io.StringIO()), self.assertRaises(SystemExit):
            main(['--displays', '2', '2', '--checkpoint', 'missing.zip', '--report', 'unused'])

    def test_flags_require_background_fast_capture(self):
        from train_rs_v612_ppo import parse_args
        import contextlib
        import io
        for extra in (['--canvas-policy-input'], ['--background-display', '0'],
                      ['--background-display', '2', '--capture-mode', 'legacy']):
            with contextlib.redirect_stderr(io.StringIO()), self.assertRaises(SystemExit):
                parse_args(['--mode', 'check', *extra])
        args = parse_args(['--mode', 'check', '--background-display', '2', '--canvas-policy-input'])
        self.assertEqual(args.background_display, 2)
        self.assertTrue(args.canvas_policy_input)

    def test_session_rejects_changing_capture_display(self):
        from tools.ppo_browser_session import PersistentBrowserSession
        session = PersistentBrowserSession()
        env = SimpleNamespace(_background_display_index=2, _background_canvas_policy=False)
        session.owner = env
        session.resources['browser'] = Mock()
        session.capture_profile = (3, False)
        with self.assertRaisesRegex(RuntimeError, 'Capture mode/display changed'):
            session.restore(env)


if __name__ == '__main__':
    unittest.main()
