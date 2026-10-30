#!/usr/bin/env python3
"""Offline regression tests for v1.7.2 post-optimizer observation gating.

No Unity/browser/YOLO/model weights are loaded; real trainer refresh method is
compiled via AST, driven by a fake async capture loop, and observed for keys.
"""
import ast
import asyncio
from collections import deque
from contextlib import redirect_stdout
import importlib.util
import io
from pathlib import Path
import shutil
import tempfile
import time
import types
import unittest

ROOT = Path(__file__).resolve().parents[1]
TOOL_DIR = ROOT / 'tools'
if str(ROOT) not in __import__('sys').path:
    __import__('sys').path.insert(0, str(ROOT))
from tools import patch_ppo_resync_transient_v172 as patch
from tools.ppo_resync_transient_v172 import (
    ResyncWindow, MAX_RESYNC_CAPTURES, MAX_TRANSIENT_SKIPS)


def frame(state='GAMEPLAY', pg=1.0, pm=0.0, score=0.0):
    return (state, {'GAMEPLAY': pg, 'MENU': pm}, score)


GOOD = frame()
TRANSIENT = frame('UNCERTAIN', .804, .003, 0.)
MENU = frame('MENU', .02, .98, .789)


def method_source():
    # The installer provides the candidate even before --apply. Tests are
    # safe to run before or after installing v1.7.2.
    path = ROOT / 'train_rs_v612_ppo.py'
    if path.is_file():
        b = path.read_bytes()
        if b.count(patch.NEW) == 1:
            return b.decode('utf8')
        if b.count(patch.OLD) == 1:
            return b.replace(patch.OLD, patch.NEW).decode('utf8')
    # Not in a repo; compile exactly the packaged function body instead.
    return 'class PPOV61Env:\n' + patch.NEW.decode('utf8')


def mock_method():
    tree = ast.parse(method_source())
    method = None
    for node in ast.walk(tree):
        if isinstance(node, ast.FunctionDef) and node.name == 'refresh_after_ppo_optimizer':
            method = node
            break
    assert method is not None
    cls = ast.ClassDef(name='Injected', bases=[], keywords=[],
                       body=[method], decorator_list=[])
    module = ast.fix_missing_locations(ast.Module(body=[cls], type_ignores=[]))
    ns = {'time': time, 'GuardUncertainError': RuntimeError}
    exec(compile(module, '<v172-refresh>', 'exec'), ns)
    return ns['Injected']


def run_frames(frames, *, focus=True, timestamps=True):
    base = mock_method()
    class Page:
        async def evaluate(self, script):
            assert 'hasFocus()' in script and 'visibilityState' in script
            return focus
    class Env(base):
        def __init__(self):
            self._guard_hard_stopped = False
            self.native_history = deque(maxlen=4)
            self.frame_intervals = deque(maxlen=12)
            self.previous_observation_at = None
            self._ppo_resume_trace_start_perf = time.perf_counter() if timestamps else None
            self.last_capture_end = None
            self.state_confidence = .90
            self.args = types.SimpleNamespace(threshold=.68)
            self.page = Page()
            self._v2_last_bgra = None
            self.output_dir = Path(tempfile.gettempdir())
            self.episode_number = 15
            self._items = list(frames)
            self._index = 0
            self.captures = 0
            self.packed = 0
            self.detector_starts = 0
            self.actions_sent = 0
        async def _capture(self):
            if self._index >= len(self._items):
                raise AssertionError('Unexpected extra capture')
            state, probs, score = self._items[self._index]
            self._current_state = (state, probs)
            self.last_capture_end = time.perf_counter()
            self._index += 1
            self.captures += 1
            return f'image{self.captures}', score, None
        def _read_screen_state(self):
            return self._current_state
        def _pack(self, legacy, *, start_detection):
            self.packed += 1
            self.native_history.append(legacy)
            if start_detection:
                self.detector_starts += 1
            return {'image': tuple(self.native_history)}, True, 0
        def _call(self, coro, timeout):
            assert timeout == 15
            return asyncio.run(coro)
    env = Env()
    out = io.StringIO()
    with redirect_stdout(out):
        obs, reason, captured_at = env.refresh_after_ppo_optimizer()
    return env, obs, reason, captured_at, out.getvalue()


class WindowTests(unittest.TestCase):
    def test_first_frame_uncertain_then_four_gameplay(self):
        env, obs, reason, captured_at, log = run_frames([TRANSIENT] + [GOOD]*4)
        self.assertIsNone(reason)
        self.assertEqual(obs['image'], ('image2','image3','image4','image5'))
        self.assertEqual(env.captures, 5)
        self.assertEqual(env.packed, 4)
        self.assertEqual(env.detector_starts, 1)
        self.assertIn('[PPO RESYNC TRANSIENT] skipped=1', log)
        self.assertEqual(log.count('[PPO RESYNC TRACE] frame='), 4)
        self.assertEqual(env.actions_sent, 0)
        self.assertLess(time.perf_counter()-captured_at, 0.1)

    def test_midstack_uncertainty_discards_whole_stack(self):
        env, obs, reason, _, log = run_frames([GOOD, GOOD, TRANSIENT] + [GOOD]*4)
        self.assertIsNone(reason)
        self.assertEqual(obs['image'], ('image4','image5','image6','image7'))
        self.assertEqual(env.captures, 7)
        self.assertEqual(env.actions_sent, 0)
        self.assertIn('discarded_valid=2',log)
        self.assertEqual(log.count('[PPO RESYNC TRACE] frame='), 4)
        # There were 6 accepted images processed; only latest four remain.
        self.assertEqual(env.packed, 6)

    def test_menu_is_not_retried_or_authorized(self):
        env, obs, reason, _, log = run_frames([MENU] + [GOOD]*4)
        self.assertIsNone(obs)
        self.assertIn('state=MENU',reason)
        self.assertEqual(env.captures, 1)
        self.assertEqual(env.actions_sent, 0)
        self.assertNotIn('[PPO RESYNC TRANSIENT]',log)

    def test_uncertain_with_menu_probability_aborts(self):
        suspect = frame('UNCERTAIN',.79,.11,.0)
        env, obs, reason, _, _ = run_frames([suspect] + [GOOD]*4)
        self.assertIsNone(obs)
        self.assertIn('unsafe post-resume',reason)
        self.assertEqual(env.captures,1)

    def test_unknown_or_loading_aborts(self):
        for state in ('OTHER','LOADING'):
            with self.subTest(state=state):
                env, obs, reason, _, _ = run_frames([frame(state,.6,.01,.0)] + [GOOD]*4)
                self.assertIsNone(obs)
                self.assertEqual(env.captures,1)

    def test_uncertain_with_template_menu_evidence_aborts(self):
        env, obs, reason, _, _ = run_frames([frame('UNCERTAIN',.80,.01,.35)] + [GOOD]*4)
        self.assertIsNone(obs)
        self.assertIn('unsafe',reason)
        self.assertEqual(env.captures,1)

    def test_gameplay_with_menu_template_aborts(self):
        env, obs, reason, _, _ = run_frames([frame('GAMEPLAY',.99,0,.68)] + [GOOD]*4)
        self.assertIsNone(obs)
        self.assertIn('menu_score',reason)
        self.assertEqual(env.captures,1)

    def test_max_four_uncertain_skips(self):
        env, obs, reason, _, out = run_frames([TRANSIENT]*5)
        self.assertIsNone(obs)
        self.assertEqual(env.captures, 5)
        self.assertIn('skips=4', reason)
        self.assertEqual(out.count('[PPO RESYNC TRANSIENT]'),4)

    def test_eight_captures_and_only_three_gameplay_fail(self):
        env, obs, reason, _, _ = run_frames([GOOD]*3 + [TRANSIENT] + [GOOD]*3 + [TRANSIENT])
        self.assertIsNone(obs)
        self.assertIn('within 8 captures',reason)
        self.assertEqual(env.captures,8)

    def test_no_browser_focus_must_fail_without_capture(self):
        env, obs, reason, _, _ = run_frames([GOOD]*4, focus=False)
        self.assertIsNone(obs)
        self.assertIn('focus/visibility lost',reason)
        self.assertEqual(env.captures,0)

    def test_all_valid_is_original_four_frame_path(self):
        env, obs, reason, _, out = run_frames([GOOD]*4)
        self.assertIsNone(reason)
        self.assertEqual(env.captures,4)
        self.assertEqual(env.packed,4)
        self.assertNotIn('[PPO RESYNC TRANSIENT]', out)
        self.assertEqual(out.count('[PPO RESYNC TRACE] frame='),4)
        self.assertEqual(env.detector_starts,1)

    def test_existing_v17_log_auditor_accepts_four_valid_traces_after_retry(self):
        try:
            from tools.ppo_reward_timing_audit import audit_training_log
        except ImportError:
            self.skipTest('Existing reward auditor not present in this test environment')
        env, obs, reason, _, trace = run_frames([GOOD, TRANSIENT] + [GOOD]*4)
        self.assertIsNone(reason)
        import json
        lines = [
            '[V612 TRAIN] PPO self-action training',
            '[PPO PAUSE TRACE] pre_pause_classifier=GAMEPLAY menu_score=0.000 ml_gameplay=1.000',
            '[PPO PAUSE] Freeze verified after 4 captures; track_deltas=[0.0, 0.0, 0.0]',
            '[PPO PAUSE] Verified frozen gameplay before optimizer; paused-track delta=0.000',
            '[PPO GAP] '+json.dumps({'boundary':1,'timesteps':128}),
            '[PPO PAUSE] Resumed after optimizer with pause-aware clocks: '+json.dumps({
                'corrected_episode_timer':True,'corrected_action_timer':True,
                'corrected_monitor_timer':True,'paused_wall_seconds':1.2,
                'active_episode_clock_drift_s':0.}),
            trace.strip(),
            '[PPO RESYNC] refreshed 4-frame stack; cached age now 8.0ms',
            '[V612 REPORT] '+json.dumps({'steps':128,'episodes':1,'completed_requested_steps':True}),
        ]
        result=audit_training_log('\n'.join(lines),min_resumes=1,min_steps=128)
        self.assertEqual(result['status'],'PASS_STABILITY_CANDIDATE',result['failures'])
        self.assertEqual(result['per_frame_gameplay_checks'],4)

    def test_uncertain_after_last_valid_resets_four_frame_requirement(self):
        env, obs, reason, _, _ = run_frames([GOOD]*3+[TRANSIENT]+[GOOD]*4)
        self.assertIsNone(reason)
        self.assertEqual(obs['image'],('image5','image6','image7','image8'))

    def test_malformed_uncertain_never_forgiven(self):
        w=ResyncWindow()
        self.assertEqual(w.observe(state='UNCERTAIN', probs={'GAMEPLAY':float('nan')},
                                   menu_score=0,confidence=.9,menu_threshold=.68),'ABORT')
        self.assertEqual(w.accepted,0)

    def test_helper_requires_four_consecutive(self):
        w=ResyncWindow()
        for _ in range(3):
            self.assertEqual(w.observe(state='GAMEPLAY',probs={'GAMEPLAY':1},
                                       menu_score=0,confidence=.9,menu_threshold=.68),'ACCEPT')
        self.assertEqual(w.observe(state='UNCERTAIN',probs={'GAMEPLAY':.804,'MENU':.003},
                                   menu_score=0,confidence=.9,menu_threshold=.68),'RETRY')
        for _ in range(3):
            self.assertEqual(w.observe(state='GAMEPLAY',probs={'GAMEPLAY':1},
                                       menu_score=0,confidence=.9,menu_threshold=.68),'ACCEPT')
        self.assertEqual(w.observe(state='GAMEPLAY',probs={'GAMEPLAY':1},
                                   menu_score=0,confidence=.9,menu_threshold=.68),'DONE')


class InstallerTests(unittest.TestCase):
    def test_compiles_patched_source(self):
        source = method_source()
        compile(source,'trainer','exec')
        self.assertIn('def refresh_after_ppo_optimizer',source)

    def test_exact_one_method_change(self):
        # No classifier or reward modifications hidden in the installer.
        if not (ROOT / 'train_rs_v612_ppo.py').is_file():
            self.skipTest('Not running in local repository')
        installed = (ROOT / 'train_rs_v612_ppo.py').read_bytes()
        if installed.count(patch.OLD) == 1:
            self.assertEqual(installed.count(patch.NEW),0)
            changed = installed.replace(patch.OLD, patch.NEW)
            self.assertEqual(changed.replace(patch.NEW, patch.OLD),installed)
        elif installed.count(patch.NEW) == 1:
            self.assertEqual(installed.count(patch.OLD),0)
        else:
            self.fail('Installed trainer has neither exact v1.7.1 nor v1.7.2 method')

    def test_refuse_modified_trainer(self):
        with tempfile.TemporaryDirectory() as d:
            root=Path(d)
            (root/'tools').mkdir()
            (root/'tools'/'ppo_resync_transient_v172.py').write_bytes(
                (ROOT/'tools'/'ppo_resync_transient_v172.py').read_bytes())
            (root/'train_rs_v612_ppo.py').write_text('print("different")\n')
            with self.assertRaisesRegex(RuntimeError,'Unexpected trainer hash'):
                patch.prepare(root)

    def test_applies_and_refuses_second_install(self):
        if not (ROOT/'train_rs_v612_ppo.py').is_file():
            self.skipTest('Not running in local repository')
        installed=(ROOT/'train_rs_v612_ppo.py').read_bytes()
        if installed.count(patch.OLD) != 1:
            self.skipTest('Installer test needs exact v1.7.1 input')
        with tempfile.TemporaryDirectory() as d:
            root=Path(d);(root/'tools').mkdir()
            (root/'train_rs_v612_ppo.py').write_bytes(installed)
            shutil.copy(ROOT/'tools'/'ppo_resync_transient_v172.py',root/'tools')
            _, _, old, updated=patch.prepare(root)
            backup=patch.apply(root)
            self.assertEqual((root/'train_rs_v612_ppo.py').read_bytes(),updated)
            self.assertEqual(backup.read_bytes(),old)
            with self.assertRaisesRegex(RuntimeError,'Unexpected trainer hash'):
                patch.apply(root)

    def test_no_saved_checkpoint_or_rewards_modified(self):
        text=patch.NEW.decode('utf8')
        self.assertNotIn('self.survival_rewards.step(',text)
        self.assertNotIn('self.page.keyboard',text)
        self.assertNotIn('self.model.save',text)
        self.assertNotIn('episode_start =',text)
        self.assertNotIn('click(',text)
        self.assertNotIn('reset(',text)


if __name__=='__main__':
    unittest.main()
