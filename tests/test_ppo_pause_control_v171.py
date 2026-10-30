"""Offline regression for missing Pause control, independent MENU handoff, and v1.7 installer."""
import ast
import asyncio
import copy
import hashlib
import importlib.util
from pathlib import Path
import shutil
import sys
import tempfile
import types
import unittest

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from tools.patch_ppo_pause_control_v171 import (EXPECTED_V17_SHA256, OLD, NEW,
                                                BACKUP_SUFFIX, prepare, apply)
from tools.ppo_pause_control_recheck_v171 import PauseControlRecheck, MAX_RECHECK_CAPTURES
from tools.ppo_rollout_menu_handoff_v17 import boundary_menu_candidate


def feedback(states, buttons, *, focused=True):
    window = PauseControlRecheck(confidence=.90, threshold=.68, motion_threshold=2.2)
    result = []
    for i, ((state, score), visible) in enumerate(zip(states, buttons)):
        probs = {'GAMEPLAY': 1.0 if state == 'GAMEPLAY' else 0.,
                 'MENU': 1.0 if state == 'MENU' else 0.}
        result.append(window.observe(state=state, probabilities=probs, menu_score=score,
                    observed_mono=100.+i*.07, focused=focused,
                    button_count=1 if visible else 0, button_visible=visible))
        if result[-1] != 'WAIT':
            break
    return window, result


class WindowTests(unittest.TestCase):
    def test_four_corroborated_menu_frames(self):
        w, out = feedback([('MENU', .62)]*4, [False]*4)
        self.assertEqual(out, ['WAIT']*3 + ['MENU'])
        self.assertEqual(w.first_menu_observed_mono, 100.0)

    def test_missing_control_remains_unsafe_during_gameplay(self):
        w, out = feedback([('GAMEPLAY', .10)]*8, [False]*8)
        self.assertEqual(out[-1], 'ABORT')
        self.assertNotIn('READY', out)

    def test_recovered_button_needs_two_new_play_frames(self):
        w, out = feedback([('GAMEPLAY', .12)]*5, [False, False, True, True, True])
        self.assertEqual(out[-1], 'READY')
        self.assertEqual(len(out), 3)
        self.assertEqual(out[:2], ['WAIT', 'WAIT'])

    def test_ml_only_menu_not_confirmed(self):
        w, out = feedback([('MENU', .02)]*8, [False]*8)
        self.assertEqual(out[-1], 'ABORT')
        self.assertNotIn('MENU', out)

    def test_animated_menu_requires_latest_four(self):
        w, out = feedback([('MENU', .62), ('MENU', .62), ('GAMEPLAY', .1),
                           ('MENU', .62), ('MENU', .62), ('MENU', .62), ('MENU', .62)],
                          [False]*7)
        self.assertEqual(out[-1], 'MENU')
        self.assertEqual(len(out), 7)

    def test_menu_then_gameplay_does_not_click_without_two_plays(self):
        w, out = feedback([('MENU', .62), ('GAMEPLAY', .1), ('MENU', .62),
                           ('GAMEPLAY', .1), ('GAMEPLAY', .1)], [False, True, True, True, True])
        self.assertEqual(out[-1], 'READY')
        self.assertNotIn('MENU', out)

    def test_multiple_controls_and_focus_loss_abort(self):
        window = PauseControlRecheck(confidence=.9, threshold=.68, motion_threshold=2.2)
        kwargs = dict(state='GAMEPLAY', probabilities={'GAMEPLAY':1.0}, menu_score=.1,
                      observed_mono=1., focused=True, button_visible=True)
        self.assertEqual(window.observe(button_count=2, **kwargs), 'ABORT')
        self.assertEqual(window.observe(button_count=1, **{**kwargs, 'focused':False}), 'ABORT')

    def test_never_ready_from_uncertain_or_high_template(self):
        w, out = feedback([('UNCERTAIN', .01)]*8, [True]*8)
        self.assertEqual(out[-1], 'ABORT')
        w, out = feedback([('GAMEPLAY', .6)]*8, [True]*8)
        self.assertEqual(out[-1], 'ABORT')


class StopAfterClick(Exception):
    pass


class FakeLocator:
    def __init__(self, env, control):
        self.env, self.control = env, control

    async def count(self):
        if self.control == 'button':
            # First count is the initial pre-recheck check.
            self.env.button_counts += 1
            if self.env.duplicate:
                return 2
            return 0 if self.env.capture_idx == 0 else int(self.env.button_visible)
        return 1

    async def is_visible(self):
        if self.control == 'overlay':
            return False
        return self.env.button_visible

    async def click(self, timeout):
        self.env.clicks += 1
        raise StopAfterClick


class FakePage:
    def __init__(self, env):
        self.env = env

    async def evaluate(self, expression):
        return True

    def locator(self, selector):
        if selector == '.game-pause-panel[aria-label="Game paused"]':
            return FakeLocator(self.env, 'overlay')
        if selector == 'button[aria-label="Pause game"]':
            return FakeLocator(self.env, 'button')
        raise AssertionError(selector)


def get_compiled_method(updated):
    tree = ast.parse(updated.decode('utf-8'))
    outer = next(x for x in tree.body if isinstance(x, ast.FunctionDef) and x.name == 'build_env_class')
    cls = next(x for x in outer.body if isinstance(x, ast.ClassDef) and x.name == 'PPOV61Env')
    method = next(x for x in cls.body if isinstance(x, ast.FunctionDef) and x.name == 'pause_for_ppo_optimizer')
    probe = ast.ClassDef(name='FakeEnv', bases=[], keywords=[], body=[copy.deepcopy(method)], decorator_list=[])
    module = ast.fix_missing_locations(ast.Module(body=[probe], type_ignores=[]))
    from rs_v612_gameplay_guard import GuardUncertainError
    scope = {'GuardUncertainError': GuardUncertainError}
    exec(compile(module, 'pause_for_ppo_optimizer', 'exec'), scope)
    return scope['FakeEnv'], GuardUncertainError


def make_env(updated, samples, visible_after=99, duplicate=False):
    cls, error = get_compiled_method(updated)
    env = cls()
    env.samples = samples
    env.capture_idx = -1
    env.button_visible = False
    env.visible_after = visible_after
    env.button_counts = env.clicks = 0
    env.duplicate = duplicate
    env.page = FakePage(env)
    env._guard_hard_stopped = False
    env._ppo_pause_pending = None
    env.suspected_since = None
    env._ppo_boundary_menu_duration_at = None
    env.args = types.SimpleNamespace(threshold=.68)
    env.state_confidence = .9
    env.startup_motion_threshold = 2.2
    env.episode_start = 0.0
    env.survival_rewards = types.SimpleNamespace(last_duration=.25)
    env.output_dir = Path(tempfile.mkdtemp(prefix='ppo_pause_v171_test_'))
    env.episode_number = 1
    env._v2_last_bgra = np.zeros((16, 16, 4), np.uint8)

    async def capture():
        env.capture_idx += 1
        env.button_visible = env.capture_idx >= env.visible_after
        if env.capture_idx >= len(env.samples):
            raise AssertionError('Probe took too many captures')
        env.last_state, env.last_score = env.samples[env.capture_idx]
        return None, env.last_score, None

    def classify():
        state = env.last_state
        return state, {'GAMEPLAY': 1.0 if state == 'GAMEPLAY' else 0.0,
                       'MENU': 1.0 if state == 'MENU' else 0.0}

    def call(coro, timeout):
        return asyncio.run(coro)

    env._capture = capture
    env._read_screen_state = classify
    env._call = call
    return env, error


class IntegratedTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        source = get_baseline()
        cls.updated = source.replace(OLD, NEW)

    def test_menu_after_missing_control_has_no_click_and_eligible_handoff(self):
        env, error = make_env(self.updated,
                              [('GAMEPLAY', .141)] + [('MENU', .62)]*4)
        try:
            with self.assertRaises(error) as exception:
                env.pause_for_ppo_optimizer()
            self.assertTrue(boundary_menu_candidate(exception.exception))
            self.assertEqual(env.clicks, 0)
            self.assertIsNotNone(env._ppo_boundary_menu_duration_at)
            self.assertEqual(env.capture_idx, 4)
        finally:
            shutil.rmtree(env.output_dir)

    def test_missing_control_reappears_only_clicks_after_two_new_play_votes(self):
        env, error = make_env(self.updated,
                              [('GAMEPLAY', .141)] + [('GAMEPLAY', .10)]*3,
                              visible_after=2)
        try:
            with self.assertRaises(StopAfterClick):
                env.pause_for_ppo_optimizer()
            self.assertEqual(env.clicks, 1)
            self.assertEqual(env.capture_idx, 3)
        finally:
            shutil.rmtree(env.output_dir)

    def test_ambiguous_missing_control_is_fail_closed_and_diagnostic_saved(self):
        env, error = make_env(self.updated,
                              [('GAMEPLAY', .141)] + [('UNCERTAIN', .02)]*8)
        try:
            with self.assertRaisesRegex(error, 'Pause control unavailable/ambiguous'):
                env.pause_for_ppo_optimizer()
            self.assertEqual(env.clicks, 0)
            self.assertTrue((env.output_dir/'ppo_pause_control_ep0001.jpg').exists())
        finally:
            shutil.rmtree(env.output_dir)

    def test_duplicate_control_stops_before_click(self):
        env, error = make_env(self.updated, [('GAMEPLAY', .141)], duplicate=True)
        try:
            with self.assertRaisesRegex(error, 'Ambiguous multiple site Pause game buttons'):
                env.pause_for_ppo_optimizer()
            self.assertEqual(env.clicks, 0)
        finally:
            shutil.rmtree(env.output_dir)


def get_baseline():
    root = Path(__file__).resolve().parents[1]
    trainer = root/'train_rs_v612_ppo.py'
    backup = trainer.with_name(trainer.name + BACKUP_SUFFIX)
    for candidate in (trainer, backup):
        if candidate.is_file() and hashlib.sha256(candidate.read_bytes()).hexdigest() == EXPECTED_V17_SHA256:
            return candidate.read_bytes()
    raise AssertionError('Tests require the exact installed v1.7 trainer or its v1.7.1 backup')


class InstallerTests(unittest.TestCase):
    def test_source_matched_and_one_block_only(self):
        old = get_baseline()
        self.assertEqual(old.count(OLD), 1)
        self.assertEqual(old.replace(OLD, NEW).count(NEW), 1)
        self.assertNotIn(OLD, old.replace(OLD, NEW))
        compile(old.replace(OLD, NEW).decode('utf8'), 'trainer', 'exec')

    def test_only_one_method_ast_changes(self):
        old = ast.parse(get_baseline().decode())
        new = ast.parse(get_baseline().replace(OLD, NEW).decode())
        def without_pause_check(tree):
            for node in ast.walk(tree):
                if isinstance(node, ast.FunctionDef) and node.name == 'pause_for_ppo_optimizer':
                    node.body = []
            return ast.dump(tree, include_attributes=False)
        self.assertEqual(without_pause_check(old), without_pause_check(new))

    def test_strict_installer_apply_and_refuses_second_install(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            (root/'tools').mkdir()
            (root/'train_rs_v612_ppo.py').write_bytes(get_baseline())
            (root/'tools'/'ppo_pause_control_recheck_v171.py').write_bytes(
                (ROOT/'tools'/'ppo_pause_control_recheck_v171.py').read_bytes())
            prepare(root)
            backup = apply(root)
            self.assertEqual(backup.read_bytes(), get_baseline())
            with self.assertRaisesRegex(RuntimeError, 'Unexpected trainer source hash'):
                prepare(root)

    def test_refuses_unexpected_trainer_and_existing_backup(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            (root/'tools').mkdir()
            (root/'train_rs_v612_ppo.py').write_bytes(get_baseline()+b'\n')
            (root/'tools'/'ppo_pause_control_recheck_v171.py').write_bytes(
                (ROOT/'tools'/'ppo_pause_control_recheck_v171.py').read_bytes())
            with self.assertRaisesRegex(RuntimeError, 'Unexpected trainer source hash'):
                prepare(root)
            (root/'train_rs_v612_ppo.py').write_bytes(get_baseline())
            (root/('train_rs_v612_ppo.py'+BACKUP_SUFFIX)).write_text('existing')
            with self.assertRaisesRegex(RuntimeError, 'Backup already exists'):
                prepare(root)


if __name__ == '__main__':
    unittest.main()
