"""Offline integration tests against exact v1.7.3 pause callback, no browser needed."""
import ast
import asyncio
import copy
import hashlib
from pathlib import Path
import shutil
import sys
import tempfile
import types
import unittest
from unittest.mock import patch

import numpy as np
from playwright.async_api import TimeoutError as PlaywrightTimeoutError

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from tools.patch_ppo_pause_click_v174 import (
    EXPECTED_V173_SHA256, BACKUP_SUFFIX, apply, digest,
    prepare, only_pause_method_changed, payloads)
from rs_v612_gameplay_guard import GuardUncertainError
from tools.ppo_rollout_menu_handoff_v17 import boundary_menu_candidate


def baseline_source():
    target = ROOT / 'train_rs_v612_ppo.py'
    backup = target.with_name(target.name + BACKUP_SUFFIX)
    for p in (target, backup):
        if p.is_file() and digest(p.read_bytes()) == EXPECTED_V173_SHA256:
            return p.read_bytes()
    raise AssertionError('Need exact v1.7.3 trainer, or backup made by installer')


def updated_source():
    source = baseline_source()
    old, new = payloads()
    assert source.count(old) == 1
    return source.replace(old, new)


def pause_method(source):
    tree = ast.parse(source.decode('utf8'))
    outer = next(n for n in tree.body if isinstance(n, ast.FunctionDef)
                 and n.name == 'build_env_class')
    cls = next(n for n in outer.body if isinstance(n, ast.ClassDef)
               and n.name == 'PPOV61Env')
    fn = next(n for n in cls.body if isinstance(n, ast.FunctionDef)
              and n.name == 'pause_for_ppo_optimizer')
    mod = ast.fix_missing_locations(ast.Module(body=[ast.ClassDef(
        name='FakeEnv', bases=[], keywords=[], body=[copy.deepcopy(fn)],
        decorator_list=[])], type_ignores=[]))
    scope = {'GuardUncertainError': GuardUncertainError}
    exec(compile(mod, 'real_pause_method', 'exec'), scope)
    return scope['FakeEnv']


class FakeLocator:
    def __init__(self, env, kind):
        self.env, self.kind = env, kind

    async def count(self):
        if self.kind != 'button':
            return 1
        return 2 if self.env.duplicate else (1 if self.env.button else 0)

    async def is_visible(self):
        return self.env.overlay if self.kind == 'overlay' else self.env.button

    async def click(self, timeout):
        assert self.kind == 'button'
        self.env.clicks += 1
        script = self.env.click_results.pop(0)
        if script == 'timeout_menu':
            self.env.button = False
            raise PlaywrightTimeoutError('Locator.click: Timeout 1800ms exceeded; element detached')
        if script == 'timeout_play':
            self.env.button = False
            raise PlaywrightTimeoutError('Locator.click: Timeout 1800ms exceeded; element not stable')
        if script == 'timeout_overlay':
            self.env.overlay = True
            raise PlaywrightTimeoutError('Locator.click: Timeout 1800ms exceeded; overlay appeared')
        if script == 'timeout_keep_button':
            raise PlaywrightTimeoutError('Locator.click: Timeout 1800ms exceeded; still gameplay')
        if script == 'ordinary_error':
            raise RuntimeError('Unexpected unrelated click error')
        assert script == 'success', script
        self.env.overlay = True

    async def wait_for(self, state, timeout):
        assert self.kind == 'overlay' and state == 'visible' and self.env.overlay


class FakePage:
    def __init__(self, env):
        self.env = env

    async def evaluate(self, expression):
        return self.env.focus

    def locator(self, selector):
        if selector == '.game-pause-panel[aria-label="Game paused"]':
            return FakeLocator(self.env, 'overlay')
        if selector == 'button[aria-label="Pause game"]':
            return FakeLocator(self.env, 'button')
        raise AssertionError(selector)


def make_env(states, clicks, *, prebutton=True, button_on_capture=None, focus=True,
             duplicate=False, overlay=False):
    cls = pause_method(updated_source())
    env = cls()
    env.focus, env.button, env.duplicate, env.overlay = focus, prebutton, duplicate, overlay
    env.click_results = list(clicks)
    env.clicks = 0
    env.capture_idx = -1
    env.states = states
    env.button_on_capture = button_on_capture
    env._guard_hard_stopped = False
    env._ppo_pause_pending = None
    env._ppo_boundary_menu_duration_at = None
    env.suspected_since = None
    env.page = FakePage(env)
    env.args = types.SimpleNamespace(threshold=.68)
    env.state_confidence = .90
    env.startup_motion_threshold = 2.2
    import time
    env.episode_start = time.monotonic() - 2.0
    env.survival_rewards = types.SimpleNamespace(last_duration=.25)
    env.output_dir = Path(tempfile.mkdtemp(prefix='ppo_click_v174_'))
    env.episode_number = 1
    env._v2_last_bgra = np.zeros((24,24,4), np.uint8)
    env.crop = (.1, .9, .1, .9)
    env._motion_view = lambda *_: np.zeros((16,16), np.uint8)

    async def capture():
        env.capture_idx += 1
        if env.button_on_capture is not None and env.capture_idx in env.button_on_capture:
            env.button = env.button_on_capture[env.capture_idx]
        sample = env.states[env.capture_idx] if env.capture_idx < len(env.states) else ('UNCERTAIN', .0)
        env.last_state, env.last_score = sample
        return None, env.last_score, None

    def classify():
        state = env.last_state
        return state, {'GAMEPLAY': 1.0 if state == 'GAMEPLAY' else 0.,
                       'MENU': 1.0 if state == 'MENU' else 0.}

    env._capture = capture
    env._read_screen_state = classify
    env._call = lambda coroutine, timeout: asyncio.run(coroutine)
    return env


class IntegrationTests(unittest.TestCase):
    def run_pause(self, env):
        # Give all three clocks time to advance even on Windows. Patching
        # sleep to zero caused flaky non-increasing timestamp tests in v1.7.3.
        real_sleep = asyncio.sleep
        async def tiny_sleep(_):
            await real_sleep(.01)
        try:
            with patch('asyncio.sleep', new=tiny_sleep):
                return env.pause_for_ppo_optimizer()
        finally:
            shutil.rmtree(env.output_dir, ignore_errors=True)

    def test_detached_click_followed_by_confirmed_menu_handoffs(self):
        env = make_env([('GAMEPLAY',.03)] + [('MENU',.62)]*4,
                       ['timeout_menu'])
        with self.assertRaises(GuardUncertainError) as cm:
            self.run_pause(env)
        self.assertTrue(boundary_menu_candidate(cm.exception))
        self.assertEqual(env.clicks,1)
        self.assertIsNone(env._ppo_pause_pending)
        self.assertIsNotNone(env._ppo_boundary_menu_duration_at)
        self.assertGreater(env._ppo_boundary_menu_duration_at, .25)

    def test_timeout_then_overlay_means_no_retry_and_freeze_verified(self):
        env = make_env([('GAMEPLAY',.01)], ['timeout_overlay'])
        self.assertEqual(self.run_pause(env),0.0)
        self.assertEqual(env.clicks,1)
        self.assertIsNotNone(env._ppo_pause_pending)
        self.assertTrue(env.overlay)

    def test_timeout_then_valid_gameplay_and_one_reverified_retry(self):
        env = make_env([('GAMEPLAY',.01)] + [('GAMEPLAY',.02)]*3,
                       ['timeout_play','success'],
                       button_on_capture={2:True})
        self.assertEqual(self.run_pause(env),0.0)
        self.assertEqual(env.clicks,2)
        self.assertIsNotNone(env._ppo_pause_pending)

    def test_second_timeout_cannot_trigger_third_click(self):
        env = make_env([('GAMEPLAY',.01)] + [('GAMEPLAY',.02)]*12,
                       ['timeout_play','timeout_keep_button'],
                       button_on_capture={2:True})
        with self.assertRaisesRegex(GuardUncertainError,'timed out twice'):
            self.run_pause(env)
        self.assertEqual(env.clicks,2)
        self.assertIsNone(env._ppo_pause_pending)

    def test_second_timeout_followed_by_overlay_is_accepted_only_if_frozen(self):
        env = make_env([('GAMEPLAY',.01)] + [('GAMEPLAY',.02)]*3,
                       ['timeout_play','timeout_overlay'],
                       button_on_capture={2:True})
        self.assertEqual(self.run_pause(env),0.0)
        self.assertEqual(env.clicks,2)
        self.assertIsNotNone(env._ppo_pause_pending)

    def test_timeout_plus_ml_only_menu_never_handoffs(self):
        env = make_env([('GAMEPLAY',.01)] + [('MENU',.02)]*8,
                       ['timeout_menu'])
        with self.assertRaisesRegex(GuardUncertainError,'control unresolved'):
            self.run_pause(env)
        self.assertEqual(env.clicks,1)
        self.assertIsNone(env._ppo_boundary_menu_duration_at)

    def test_timeout_and_ambiguous_screen_stops(self):
        env = make_env([('GAMEPLAY',.01)] + [('UNCERTAIN',.02)]*8,
                       ['timeout_play'])
        with self.assertRaisesRegex(GuardUncertainError,'control unresolved'):
            self.run_pause(env)
        self.assertEqual(env.clicks,1)

    def test_timeout_plus_duplicate_controls_stops(self):
        env = make_env([('GAMEPLAY',.01),('GAMEPLAY',.02)], ['timeout_keep_button'])
        # duplicate only after the first (pre-click) count
        env.duplicate = False
        original_click = FakeLocator.click
        async def duplicated_click(locator,timeout):
            locator.env.duplicate=True
            return await original_click(locator,timeout)
        with patch.object(FakeLocator,'click',new=duplicated_click):
            with self.assertRaisesRegex(GuardUncertainError,'multiple site Pause'):
                self.run_pause(env)
        self.assertEqual(env.clicks,1)

    def test_timeout_lost_focus_never_clicks_again(self):
        env = make_env([('GAMEPLAY',.01)], ['timeout_play'])
        orig = FakeLocator.click
        async def lose_focus(locator,timeout):
            locator.env.focus=False
            return await orig(locator, timeout)
        with patch.object(FakeLocator,'click',new=lose_focus):
            with self.assertRaisesRegex(GuardUncertainError,'focus lost'):
                self.run_pause(env)
        self.assertEqual(env.clicks,1)

    def test_unrelated_click_error_not_retried(self):
        env = make_env([('GAMEPLAY',.01)], ['ordinary_error'])
        with self.assertRaisesRegex(RuntimeError,'Unexpected unrelated'):
            self.run_pause(env)
        self.assertEqual(env.clicks,1)

    def test_preclick_missing_control_still_handoffs(self):
        env = make_env([('GAMEPLAY',.01)]+[('MENU',.62)]*4,[],prebutton=False)
        with self.assertRaises(GuardUncertainError) as cm:
            self.run_pause(env)
        self.assertTrue(boundary_menu_candidate(cm.exception))
        self.assertEqual(env.clicks,0)

    def test_preclick_missing_control_can_be_reverified(self):
        env = make_env([('GAMEPLAY',.01)]+[('GAMEPLAY',.01)]*3,
                       ['success'],prebutton=False,button_on_capture={2:True})
        self.assertEqual(self.run_pause(env),0.0)
        self.assertEqual(env.clicks,1)

    def test_preclick_good_button_normal_click(self):
        env=make_env([('GAMEPLAY',.01)],['success'])
        self.assertEqual(self.run_pause(env),0.0)
        self.assertEqual(env.clicks,1)

    def test_click_timeout_after_menu_detection_does_not_fake_reward(self):
        env = make_env([('GAMEPLAY',.01)]+[('MENU',.62)]*4,['timeout_menu'])
        with self.assertRaises(GuardUncertainError):
            self.run_pause(env)
        # Pending handoff only. Actual -1 death penalty can only be emitted by
        # next guarded env.step in the unchanged rollout callback.
        self.assertFalse(getattr(env,'_ppo_boundary_menu_pending',False))


class InstallerTests(unittest.TestCase):
    def test_exact_hash_compile_and_method_scope(self):
        base = baseline_source()
        self.assertEqual(digest(base), EXPECTED_V173_SHA256)
        updated = updated_source()
        only_pause_method_changed(base.decode(),updated.decode())
        compile(updated.decode(), 'v174', 'exec')
        self.assertEqual(updated.count(b'await pause_button.click(timeout=1800)'),1)
        self.assertEqual(updated.count(b'await pause_button.click(timeout=1200)'),1)

    def test_source_matched_backup_and_second_install_refused(self):
        with tempfile.TemporaryDirectory() as t:
            root=Path(t)
            (root/'tools').mkdir()
            (root/'train_rs_v612_ppo.py').write_bytes(baseline_source())
            (root/'tools'/'ppo_pause_control_recheck_v171.py').write_text('helper')
            target,backup,old,updated=prepare(root)
            self.assertEqual(old,baseline_source())
            self.assertEqual(target,root/'train_rs_v612_ppo.py')
            apply(root)
            self.assertEqual(target.read_bytes(),updated)
            self.assertEqual(backup.read_bytes(),old)
            with self.assertRaisesRegex(RuntimeError,'does not match'):
                apply(root)

    def test_modified_source_refused_without_touching_bytes(self):
        with tempfile.TemporaryDirectory() as t:
            root=Path(t)
            (root/'tools').mkdir()
            (root/'tools'/'ppo_pause_control_recheck_v171.py').write_text('helper')
            target=root/'train_rs_v612_ppo.py'
            target.write_bytes(baseline_source()+b'\n# other edit\n')
            with self.assertRaisesRegex(RuntimeError,'does not match'):
                apply(root)
            self.assertFalse((root/('train_rs_v612_ppo.py'+BACKUP_SUFFIX)).exists())
            self.assertTrue(target.read_bytes().endswith(b'# other edit\n'))


if __name__ == '__main__':
    unittest.main()
