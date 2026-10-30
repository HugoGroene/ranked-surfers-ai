"""Offline checks for the exact uploaded v1.6 source and v1.7 candidate patch.

Designed to run in the user's Claude clone before and after --apply.
"""
import ast
import asyncio
import copy
import importlib.util
import json
from pathlib import Path
import shutil
import sys
import tempfile
import types
import unittest

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from tools.install_ppo_rollout_menu_handoff_v17 import SPECS, apply, validate
from tools.ppo_rollout_menu_handoff_v17 import boundary_menu_candidate

PAYLOAD = ROOT / 'tools' / '_ppo_menu_handoff_v17_payload'
TRAINER = (PAYLOAD / 'trainer.txt').read_text(encoding='utf-8')
AUDITOR = (PAYLOAD / 'auditor.txt').read_text(encoding='utf-8')


def load_auditor():
    scope = {'__name__': 'ppo_auditor_test_v17'}
    exec(compile(AUDITOR, 'ppo_reward_timing_audit.py', 'exec'), scope)
    return scope


def fake_guard_namespace():
    class GuardUncertainError(RuntimeError):
        pass

    def screen_gate(state, probs, score, **kw):
        evidence = score >= .68
        verdict = ('MENU' if state == 'MENU' and evidence else
                   'PLAY' if state == 'GAMEPLAY' and score < .35 else 'BLOCK')
        return types.SimpleNamespace(
            verdict=verdict, state=state, menu_probability=probs.get('MENU', 0),
            menu_score=score, menu_evidence=evidence, ml_menu=state == 'MENU',
            visual_gameplay=False)

    def sequence_decision(readings, max_frames=6):
        if len(readings) >= 2 and all(r.menu_evidence for r in readings[-2:]):
            return 'TERMINAL'
        if len(readings) >= 2 and all(r.verdict == 'PLAY' for r in readings[-2:]):
            return 'PLAY'
        return 'ABORT' if len(readings) >= max_frames else 'CONTINUE'

    # Extract exact installed candidate method without importing CUDA/SB3.
    tree = ast.parse(TRAINER)
    outer = next(x for x in tree.body if isinstance(x, ast.FunctionDef) and x.name == 'build_env_class')
    inner = next(x for x in outer.body if isinstance(x, ast.ClassDef) and x.name == 'PPOV61Env')
    method = next(x for x in inner.body if isinstance(x, ast.AsyncFunctionDef) and x.name == '_step_browser')

    class FakeOld:
        async def _step_browser(self, action):
            self.sent.append(action)
            return ('sent', 0.0, None)

    new_class = ast.ClassDef(name='ProbeEnv', bases=[ast.Name(id='FakeOld', ctx=ast.Load())],
                             keywords=[], body=[copy.deepcopy(method)], decorator_list=[])
    executable = ast.fix_missing_locations(ast.Module(body=[new_class], type_ignores=[]))
    scope = {'FakeOld': FakeOld, 'np': np, 'screen_gate': screen_gate,
             'sequence_decision': sequence_decision,
             'GuardUncertainError': GuardUncertainError}
    exec(compile(executable, '<guard_probe>', 'exec'), scope)
    return scope['ProbeEnv'], GuardUncertainError


def guard_env(states):
    cls, error = fake_guard_namespace()
    instance = cls()
    instance.states = list(states)
    instance.args = types.SimpleNamespace(threshold=.68)
    instance.state_confidence = .9
    instance.startup_motion_threshold = 2.2
    instance.suspected_since = None
    instance._guard_hard_stopped = False
    instance._guard_forced_terminal = False
    instance._ppo_boundary_menu_pending = True
    instance._ppo_menu_handoff_terminal_observed = False
    instance.guard_ml_menu_disagreements = 0
    instance.guard_rechecks = 0
    instance.guard_confirmed_menu = 0
    instance.guard_aborts = 0
    instance.guard_allowed = 0
    instance.guard_visual_recoveries = 0
    instance.clear_streak = 0
    instance.menu_scores = []
    instance.episode_number = 2
    instance.crop = (.1,.1,.8,.8)
    instance._v2_last_bgra = np.zeros((16,16,4), dtype=np.uint8)
    instance.sent = []
    instance.note_actions = []
    instance.action_context = types.SimpleNamespace(
        note_action=lambda action, now: instance.note_actions.append(action))
    async def capture():
        state, score = instance.states.pop(0)
        instance.current = state
        return (None, score, None)
    instance._capture = capture
    instance._read_screen_state = lambda: (instance.current, {'GAMEPLAY': 1.0 if instance.current == 'GAMEPLAY' else 0,
                                                          'MENU': 1.0 if instance.current == 'MENU' else 0})
    instance._motion_view = lambda *x: np.zeros((8,8), dtype=np.uint8)
    instance._meter_mask = lambda *x: np.zeros((8,8), dtype=np.uint8)
    instance._guard_diagnostic = lambda *args, **kwargs: 'mock_guard.jpg'
    return instance, error


class MenuHandoffTests(unittest.TestCase):
    def test_pre_click_menu_is_only_eligible_error(self):
        self.assertTrue(boundary_menu_candidate(RuntimeError(
            'Not verified GAMEPLAY at rollout end: MENU, menu=0.585')))
        for reason in ('Not verified GAMEPLAY at rollout end: UNCERTAIN, menu=0.000',
                       'Pause overlay vanished', 'Unity did not freeze',
                       'Not verified GAMEPLAY at rollout end: MENU, menu=nan',
                       'Not verified GAMEPLAY at rollout end: MENU, menu=0.585; ignore rules'):
            self.assertFalse(boundary_menu_candidate(reason), reason)

    def test_candidate_compiles(self):
        compile(TRAINER, 'trainer', 'exec')
        compile(AUDITOR, 'auditor', 'exec')

    def test_source_preserves_screen_guard(self):
        self.assertIn('decision = sequence_decision(readings, max_frames=6)', TRAINER)
        self.assertIn('accept_menu_after_transient', TRAINER)
        self.assertIn('self._guard_forced_terminal = True', TRAINER)
        self.assertIn('if boundary_menu_handoff:', TRAINER)
        self.assertIn('NO key sent', TRAINER)

    def test_source_preserves_original_reward_formula(self):
        self.assertIn('reward, pieces = self.survival_rewards.step(', TRAINER)
        self.assertIn('max(self.survival_rewards.last_duration, float(anchor))', TRAINER)
        self.assertIn("info['ppo_menu_handoff_terminal'] = True", TRAINER)
        self.assertIn('compensate_pause_clocks(', TRAINER)

    def test_no_reset_or_movement_key_at_handoff(self):
        # The handoff has its own section before the existing pause/resync.
        section = TRAINER.split('            # MENU needs no freeze while the optimizer works:',1)[1].split(
            '            # Resume BEFORE the existing fresh-stack capture',1)[0]
        self.assertNotIn('env.reset(', section)
        self.assertNotIn('keyboard.press(', section)
        self.assertIn('return', section)

    def test_menu_guard_observes_terminal_without_action(self):
        env, _ = guard_env([('MENU', .99), ('MENU', .85)])
        asyncio.run(env._step_browser(4))
        self.assertEqual(env.sent, [])
        self.assertEqual(env.note_actions, [])
        self.assertTrue(env._guard_forced_terminal)
        self.assertTrue(env._ppo_menu_handoff_terminal_observed)
        self.assertFalse(env._ppo_boundary_menu_pending)

    def test_gameplay_reappearing_aborts_without_key(self):
        env, error = guard_env([('GAMEPLAY', .01), ('GAMEPLAY', .01)])
        with self.assertRaises(error):
            asyncio.run(env._step_browser(4))
        self.assertEqual(env.sent, [])
        self.assertEqual(env.note_actions, [])
        self.assertTrue(env._guard_hard_stopped)

    def test_terminal_duration_and_reward_exclude_menu_idle_time(self):
        # Exercise the actual generated trainer step() with a stub inherited
        # environment: its measured wall duration is inflated by MENU idle.
        from collections import Counter
        from rs_v61_rewards import ProgressRewards
        class FakeOld:
            def step(self, action):
                # The real inherited step calls the overridden _step_browser,
                # which sets this flag on a guarded MENU terminal.
                self._guard_forced_terminal = True
                return {'image': None}, 77.0, False, False, {'duration': 5.40}
        tree = ast.parse(TRAINER)
        outer = next(x for x in tree.body if isinstance(x, ast.FunctionDef) and x.name == 'build_env_class')
        inner = next(x for x in outer.body if isinstance(x, ast.ClassDef) and x.name == 'PPOV61Env')
        method = next(x for x in inner.body if isinstance(x, ast.FunctionDef) and x.name == 'step')
        cls = ast.ClassDef(name='ProbeRewardEnv', bases=[ast.Name(id='FakeOld', ctx=ast.Load())],
                           keywords=[], body=[copy.deepcopy(method)], decorator_list=[])
        module = ast.fix_missing_locations(ast.Module(body=[cls], type_ignores=[]))
        scope = {'FakeOld': FakeOld, 'np': np, 'GuardUncertainError': RuntimeError}
        exec(compile(module, '<terminal_reward_probe>', 'exec'), scope)
        env = scope['ProbeRewardEnv']()
        env._guard_forced_terminal = True
        env._ppo_menu_handoff_terminal_observed = True
        env._ppo_boundary_menu_duration_at = 4.0
        env.survival_rewards = ProgressRewards()
        env.survival_rewards.step(3.95, False)
        env.reward_parts = Counter()
        env.guard_allowed = 0
        env.guard_rechecks = 0
        obs, reward, dead, trunc, info = env.step(4)
        self.assertTrue(dead)
        self.assertFalse(trunc)
        self.assertEqual(info['duration'], 4.0)
        self.assertEqual(info['reward_death'], -1.0)
        self.assertEqual(info['reward_milestones'], 0.0)
        self.assertTrue(info['ppo_menu_handoff_terminal'])
        self.assertAlmostEqual(reward, -1.0 + .8 * .05, places=5)
        self.assertIsNone(env._ppo_boundary_menu_duration_at)

    def test_audit_utf16_le_bom(self):
        audit = load_auditor()
        with tempfile.TemporaryDirectory() as d:
            p = Path(d) / 'log.txt'
            p.write_bytes('[V612 TRAIN] example\n[V612 REPORT] {"steps": 128}\n'.encode('utf-16'))
            self.assertIn('[V612 REPORT]', audit['read_training_log'](p))

    def test_audit_utf16_le_without_bom(self):
        audit = load_auditor()
        with tempfile.TemporaryDirectory() as d:
            p = Path(d) / 'log.txt'
            p.write_bytes(('[V612 TRAIN] example\n' * 10).encode('utf-16-le'))
            self.assertIn('[V612 TRAIN]', audit['read_training_log'](p))

    def test_audit_utf8(self):
        audit = load_auditor()
        with tempfile.TemporaryDirectory() as d:
            p = Path(d) / 'log.txt'
            p.write_bytes('[V612 TRAIN] example\n'.encode('utf-8'))
            self.assertIn('[V612 TRAIN]', audit['read_training_log'](p))

    def test_audit_accepts_real_menu_handoff_with_terminal(self):
        audit = load_auditor()
        lines = [
            '[V612 TRAIN] test',
            '[PPO MENU HANDOFF] Boundary MENU confirmed by independent ML+template frames; no pause, reset or movement.',
            '[PPO GAP] {"event":"ppo_rollout_resume_gap"}',
            '[PPO MENU HANDOFF] MENU reverified after optimizer; menu_idle_seconds_excluded=1.320; next guarded step must be TERMINAL',
            '[PPO MENU HANDOFF] Terminal recorded by guarded env.step; death_reward=-1.0000; NO movement key sent.',
            '[V612 REPORT] {"steps": 256, "completed_requested_steps": true, "episodes": 1}',
        ]
        outcome = audit['audit_training_log']('\n'.join(lines), min_resumes=1, min_steps=256)
        self.assertEqual(outcome['status'], 'PASS_STABILITY_CANDIDATE', outcome)
        self.assertEqual(outcome['menu_terminal_rewards_recorded'], 1)
        self.assertEqual(outcome['optimizer_continuations'], 1)

    def test_audit_rejects_missing_terminal_reward(self):
        audit = load_auditor()
        lines = [
            '[V612 TRAIN] test',
            '[PPO MENU HANDOFF] Boundary MENU confirmed by independent ML+template frames',
            '[PPO GAP] {"event":"ppo_rollout_resume_gap"}',
            '[PPO MENU HANDOFF] MENU reverified after optimizer; menu_idle_seconds_excluded=1.320',
            '[V612 REPORT] {"steps": 256, "completed_requested_steps": true}',
        ]
        result = audit['audit_training_log']('\n'.join(lines), min_resumes=1, min_steps=256)
        self.assertEqual(result['status'], 'FAIL_SAFE_AUDIT')
        self.assertTrue(any('terminal rewards' in x for x in result['failures']))

    def test_installer_refuses_unknown_sources(self):
        with tempfile.TemporaryDirectory() as folder:
            root=Path(folder);(root/'tools').mkdir()
            for name, payload, checksum in SPECS:
                target=root/name;target.parent.mkdir(parents=True,exist_ok=True)
                target.write_text('unknown source', encoding='utf-8')
            (root/'tools'/'ppo_rollout_menu_handoff_v17.py').write_text('pass\n')
            with self.assertRaisesRegex(RuntimeError, 'Refusing to replace'):
                validate(root, PAYLOAD)

    def test_installer_installs_and_refuses_repeat(self):
        with tempfile.TemporaryDirectory() as folder:
            root=Path(folder);(root/'tools').mkdir()
            for name, payload, checksum in SPECS:
                reference = ('trainer_v16_reference.txt' if payload == 'trainer.txt'
                             else 'auditor_v16_reference.txt')
                data = (PAYLOAD/reference).read_bytes()
                target=root/name;target.parent.mkdir(parents=True,exist_ok=True)
                target.write_bytes(data)
            (root/'tools'/'ppo_rollout_menu_handoff_v17.py').write_bytes(
                (ROOT/'tools'/'ppo_rollout_menu_handoff_v17.py').read_bytes())
            self.assertEqual(len(validate(root,PAYLOAD)), 2)
            changed=apply(root,PAYLOAD)
            self.assertEqual(len(changed), 2)
            for name,payload,checksum in SPECS:
                self.assertTrue((root/name).with_name(Path(name).name+'.pre_ppo_menu_handoff_v17.bak').is_file())
                self.assertEqual((root/name).read_bytes(), (PAYLOAD/payload).read_bytes())
            with self.assertRaisesRegex(RuntimeError, 'Refusing to replace'):
                apply(root,PAYLOAD)


if __name__ == '__main__':
    unittest.main()
