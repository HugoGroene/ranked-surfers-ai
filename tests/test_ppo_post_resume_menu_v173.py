#!/usr/bin/env python3
"""Offline tests of the actual patched v1.7.3 refresh/callback, audit and installer.
No Playwright, local Unity, model weights, browser or live actions are required.
"""
import ast
import asyncio
from collections import deque
from contextlib import redirect_stdout
import importlib.util
import io
import json
from pathlib import Path
import shutil
import sys
import tempfile
import time
import types
import unittest

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path: sys.path.insert(0,str(ROOT))
from tools import install_ppo_post_resume_menu_v173 as installer
# Audit the CANDIDATE implementation before installing it. Importing the
# local v1.7.2 auditor here tests the wrong version and causes false failures.
AUDITOR_NEW = ROOT / 'tools/_ppo_post_resume_menu_v173_payload/auditor.txt'
audit_ns = {'__name__': '_ppo_reward_timing_audit_v173_candidate',
            '__file__': str(AUDITOR_NEW)}
exec(compile(AUDITOR_NEW.read_text(encoding='utf-8'), str(AUDITOR_NEW), 'exec'), audit_ns)
audit_training_log = audit_ns['audit_training_log']

TRAINER_NEW = (ROOT/'tools/_ppo_post_resume_menu_v173_payload/trainer.txt').read_text()


def get_method(name):
    tree=ast.parse(TRAINER_NEW)
    owner=[n for n in ast.walk(tree) if isinstance(n,ast.ClassDef) and n.name=='PPOV61Env']
    assert len(owner)==1
    found=[n for n in owner[0].body if isinstance(n,(ast.FunctionDef,ast.AsyncFunctionDef)) and n.name==name]
    if not found and name=='_on_rollout_start':
        cb=[n for n in ast.walk(tree) if isinstance(n,ast.ClassDef) and n.name=='V61Callback']
        found=[n for n in cb[0].body if isinstance(n,ast.FunctionDef) and n.name==name]
    assert len(found)==1,(name,len(found))
    return found[0]


def injected_class(method, ns, base='object'):
    tree=ast.Module(body=[ast.ClassDef(name='Injected',bases=[ast.Name(id=base,ctx=ast.Load())],
                                   keywords=[],body=[method],decorator_list=[])],type_ignores=[])
    ns=dict(ns)
    exec(compile(ast.fix_missing_locations(tree),'<patched-actual-method>','exec'),ns)
    return ns['Injected']


def menu(score=.572,pm=.998):
    return ('MENU', {'GAMEPLAY':.0, 'MENU':pm},score)

def gameplay():
    return ('GAMEPLAY', {'GAMEPLAY':1.0, 'MENU':0.},0.)


def make_env(captures):
    from tools.ppo_resync_transient_v172 import MAX_RESYNC_CAPTURES
    assert len(captures)<=MAX_RESYNC_CAPTURES
    class Page:
        async def evaluate(self,script): return True
    Base = injected_class(get_method('refresh_after_ppo_optimizer'),
                          {'time':time,'GuardUncertainError':RuntimeError})
    class Env(Base):
        def __init__(self):
            self._guard_hard_stopped=False
            self.native_history=deque(maxlen=4)
            self.frame_intervals=deque(maxlen=12)
            self.previous_observation_at=None
            self._ppo_resume_trace_start_perf=time.perf_counter()
            self.last_capture_end=None
            self.state_confidence=.90
            self.args=types.SimpleNamespace(threshold=.68)
            self.page=Page()
            self._v2_last_bgra=None
            self.output_dir=Path(tempfile.gettempdir())
            self.episode_number=15
            self.items=list(captures)
            self.actions_sent=0
            self.packed=0
            self._ppo_post_resume_menu_candidate=('stale',)
            self.episode_start=time.monotonic()-4.5
            self.survival_rewards=types.SimpleNamespace(last_duration=4.0)
            self.action_context=types.SimpleNamespace(last_action_times=[None]*5)
            self.episode_started=time.perf_counter()-4.5
            self.last_step=time.monotonic()-0.01
            self._prev_step_start=None
            self.suspected_since=None
            self._ppo_boundary_menu_pending=False
            self._ppo_boundary_menu_duration_at=None
            self._ppo_pause_pending=None
            self._confirm=(True, {'checks':[{'state':'MENU'}]*4})
        async def _capture(self):
            if not self.items: raise AssertionError('unexpected extra capture')
            state,probs,score=self.items.pop(0)
            self.current=(state,probs)
            self.last_capture_end=time.perf_counter()
            return 'observation',score,None
        def _read_screen_state(self):return self.current
        def _pack(self,legacy,*,start_detection):
            self.packed+=1
            self.native_history.append(legacy)
            return {'image': tuple(self.native_history)}, True,0
        def _call(self,coro,timeout):
            self.timeout=timeout
            return asyncio.run(coro)
        def confirm_post_optimizer_menu(self):
            # Advance the callback's simulated *current clock*, not the
            # captured candidate tuple.  The real callback holds a local
            # reference to the original immutable tuple before confirming;
            # replacing self._ppo_post_resume_menu_candidate has no effect.
            # This models 210 ms of MENU confirmation without sleeping and
            # without depending on Windows timer granularity.
            if self._confirm[0] and hasattr(self, '_test_clock'):
                self._test_clock.advance(.210)
            return self._confirm
        def resume_after_ppo_optimizer(self, monitor):
            assert monitor.env is self
            return dict(paused_wall_seconds=1.17, corrected_episode_timer=True,
                        corrected_action_timer=True, corrected_monitor_timer=True,
                        active_episode_clock_drift_s=0.0)
    return Env()


def refresh(env):
    with redirect_stdout(io.StringIO()) as out:
        obs,reason,when=env.refresh_after_ppo_optimizer()
    return obs,reason,out.getvalue()


class RefreshTests(unittest.TestCase):
    def test_exact_failure_frame_becomes_candidate_no_keys(self):
        e=make_env([gameplay()]*3+[menu()])
        obs,reason,txt=refresh(e)
        self.assertIsNone(obs)
        self.assertIn('state=MENU',reason)
        self.assertIsNotNone(e._ppo_post_resume_menu_candidate)
        self.assertEqual(len(e._ppo_post_resume_menu_candidate),3)
        self.assertEqual(e.actions_sent,0)
        self.assertNotIn('[PPO RESYNC TRACE] frame=',txt)
    def test_unconfirmed_menu_no_candidate(self):
        e=make_env([menu(.2)])
        obs,reason,_=refresh(e)
        self.assertIsNone(obs)
        self.assertIsNone(e._ppo_post_resume_menu_candidate)
    def test_ml_only_menu_no_candidate(self):
        e=make_env([menu(0.,pm=1.)])
        obs,reason,_=refresh(e)
        self.assertIsNone(obs)
        self.assertIsNone(e._ppo_post_resume_menu_candidate)
    def test_non_menu_other_never_candidate(self):
        e=make_env([('OTHER',{'GAMEPLAY':0.,'MENU':1.},1.)])
        obs,_,_=refresh(e)
        self.assertIsNone(obs)
        self.assertIsNone(e._ppo_post_resume_menu_candidate)
    def test_four_good_frames_reset_stale_candidate(self):
        e=make_env([gameplay()]*4)
        obs,reason,txt=refresh(e)
        self.assertIsNone(reason)
        self.assertIsNotNone(obs)
        self.assertIsNone(e._ppo_post_resume_menu_candidate)
        self.assertEqual(txt.count('[PPO RESYNC TRACE] frame='),4)
        self.assertEqual(e.actions_sent,0)


def callback_for(env):
    cb_method=get_method('_on_rollout_start')
    import numpy as np
    class Gap:
        def begin_rollout(self,now,timesteps): return {'boundary':1,'timesteps':128}
    # Only callback _on_rollout_start is extracted, and its production branches
    # are executed unmodified. No SB3 machinery is needed for this offline test.
    class Monitor:
        def __init__(self,env): self.env=env;self.t_start=time.time()-4.5
    monitor=Monitor(env)
    class Vec:
        envs=[monitor]
    model=types.SimpleNamespace(env=Vec(),n_envs=1,_last_obs={'image':'previous'})
    a=types.SimpleNamespace(pause_ppo_optimizer=True,resync_rollouts=True,
                            recover_rollout_menu=False)
    class SimulatedClock:
        # Use the same offset for monotonic, perf_counter and wall time so
        # compensate_pause_clocks receives consistent nonzero intervals.
        def __init__(self): self.offset=0.0
        def monotonic(self): return time.monotonic()+self.offset
        def perf_counter(self): return time.perf_counter()+self.offset
        def time(self): return time.time()+self.offset
        def advance(self, seconds): self.offset += seconds
    clock=SimulatedClock()
    env._test_clock=clock
    ns={'time':clock,'json':json,'np':np,'a':a,'raw':env,'GuardUncertainError':RuntimeError,
        'candidate_is_menu':lambda reason:False,
        'replace_cached_observation':lambda model,obs: model.__setattr__('_last_obs',obs)}
    cb_cls=injected_class(cb_method,ns)
    cb=cb_cls()
    cb.model=model
    cb.num_timesteps=128
    cb.rollout_gap_tracker=Gap()
    cb.rollout_gap_log_path=Path(tempfile.gettempdir())/'v173_temporary_gap_test.jsonl'
    return cb,monitor


class CallbackTests(unittest.TestCase):
    def test_post_resume_menu_records_pending_real_terminal_and_clocks(self):
        e=make_env([gameplay()]*3+[menu()])
        cb,monitor=callback_for(e)
        old_origin=e.episode_start
        old_monitor_origin=monitor.t_start
        prev_obs=cb.model._last_obs
        with redirect_stdout(io.StringIO()) as out:
            cb._on_rollout_start()
        self.assertTrue(e._ppo_boundary_menu_pending)
        self.assertIsNone(e._ppo_post_resume_menu_candidate)
        self.assertGreater(e._ppo_boundary_menu_duration_at,4.0)
        self.assertLess(e._ppo_boundary_menu_duration_at,6.0)
        self.assertGreater(e.episode_start,old_origin)
        self.assertGreater(monitor.t_start,old_monitor_origin)
        self.assertEqual(cb.model._last_obs,prev_obs)
        self.assertEqual(e.actions_sent,0)
        self.assertIn('deferred guarded terminal with NO movement.',out.getvalue())
    def test_menu_confirmation_failure_stops_and_does_not_arm(self):
        e=make_env([menu()]);e._confirm=(False,{'checks':[]})
        cb,_=callback_for(e)
        with self.assertRaisesRegex(RuntimeError,'not independently confirmed'):
            cb._on_rollout_start()
        self.assertFalse(e._ppo_boundary_menu_pending)
        self.assertEqual(e.actions_sent,0)
    def test_ambiguous_ml_only_menu_stops(self):
        e=make_env([menu(0.05)])
        cb,_=callback_for(e)
        with self.assertRaisesRegex(RuntimeError,'Cannot verify gameplay'):
            cb._on_rollout_start()
        self.assertFalse(e._ppo_boundary_menu_pending)
        self.assertEqual(e.actions_sent,0)
    def test_steady_gameplay_refresh_updates_cached_frame(self):
        e=make_env([gameplay()]*4)
        cb,_=callback_for(e)
        with redirect_stdout(io.StringIO()) as out:
            cb._on_rollout_start()
        self.assertFalse(e._ppo_boundary_menu_pending)
        self.assertEqual(cb.model._last_obs,{'image':('observation',)*4})
        self.assertIn('refreshed 4-frame stack',out.getvalue())


def log_lines(*,handoff=True,reward=True):
    lines=['[V612 TRAIN] PPO executing its OWN actions',
           '[PPO PAUSE TRACE] pre_pause_classifier=GAMEPLAY menu_score=0.000 ml_gameplay=1.000',
           '[PPO PAUSE] Freeze verified after 4 captures; track_deltas=[0.0, 0.0, 0.0]',
           '[PPO PAUSE] Verified frozen gameplay before optimizer; paused-track delta=0.000',
           '[PPO GAP] '+json.dumps({'boundary':1,'timesteps':128}),
           '[PPO PAUSE] Resumed after optimizer with pause-aware clocks: '+json.dumps({
                'corrected_episode_timer':True,'corrected_action_timer':True,
                'corrected_monitor_timer':True,'paused_wall_seconds':1.172,
                'active_episode_clock_drift_s':0.0})]
    if handoff:
        lines.append('[PPO POST RESUME MENU] Confirmed MENU after 4 fresh ML+template captures; menu_idle_seconds_excluded=0.210; deferred guarded terminal with NO movement.')
    else:
        for i in range(1,5):
            lines.append(f'[PPO RESYNC TRACE] frame={i}/4 since_resume_ms=20.0 state=GAMEPLAY menu_score=0.000 ml_gameplay=1.000 ml_menu=0.000')
        lines.append('[PPO RESYNC] refreshed 4-frame stack; cached age now 8.0ms')
    if reward:
        lines.append('[PPO MENU HANDOFF] Terminal recorded by guarded env.step; death_reward=-1.0000; NO movement key sent.')
    lines.append('[V612 REPORT] '+json.dumps({'steps':128,'episodes':1,'completed_requested_steps':True}))
    return '\n'.join(lines)

class AuditTests(unittest.TestCase):
    def test_verified_post_menu_counts_instead_of_fresh_stack(self):
        report=audit_training_log(log_lines(),min_steps=128,min_resumes=1)
        self.assertEqual(report['status'],'PASS_STABILITY_CANDIDATE',report['failures'])
        self.assertEqual(report['post_resume_menu_handoffs'],1)
        self.assertEqual(report['fresh_stacks'],0)
        self.assertEqual(report['menu_terminal_rewards_recorded'],1)
    def test_handoff_without_actual_terminal_reward_fails(self):
        report=audit_training_log(log_lines(reward=False),min_steps=128,min_resumes=1)
        self.assertEqual(report['status'],'FAIL_SAFE_AUDIT')
        self.assertTrue(any('terminal rewards' in f for f in report['failures']))
    def test_handoff_without_confirmation_fails(self):
        raw=log_lines().replace('after 4 fresh','after 2 fresh')
        report=audit_training_log(raw,min_steps=128,min_resumes=1)
        self.assertEqual(report['status'],'FAIL_SAFE_AUDIT')
    def test_nonhandoff_legacy_four_stack_path_still_passes(self):
        report=audit_training_log(log_lines(handoff=False,reward=False),min_steps=128,min_resumes=1)
        self.assertEqual(report['status'],'PASS_STABILITY_CANDIDATE',report['failures'])
        self.assertEqual(report['fresh_stacks'],1)
    def test_missing_both_handoff_and_fresh_stack_fails(self):
        raw=log_lines(handoff=False,reward=False).replace('[PPO RESYNC] refreshed 4-frame stack; cached age now 8.0ms','')
        report=audit_training_log(raw,min_steps=128,min_resumes=1)
        self.assertEqual(report['status'],'FAIL_SAFE_AUDIT')

class TerminalSafetyTests(unittest.TestCase):
    def test_existing_handoff_prekey_guard_never_sends_movement(self):
        # Run the REAL guarded browser method against four independently
        # corroborated MENU frames. The parent method would send a key if hit.
        from rs_v612_gameplay_guard import screen_gate, sequence_decision
        class Parent:
            async def _step_browser(self, action):
                self.movement_keys += 1
                return 'movement dispatched'
        Guard = injected_class(
            get_method('_step_browser'),
            {'screen_gate':screen_gate,'sequence_decision':sequence_decision,
             'GuardUncertainError':RuntimeError,'Parent':Parent}, base='Parent')
        class GuardEnv(Guard):
            def __init__(self):
                self._guard_hard_stopped=False
                self._ppo_boundary_menu_pending=True
                self._ppo_menu_handoff_terminal_observed=False
                self._guard_forced_terminal=False
                self.guard_ml_menu_disagreements=0
                self.guard_rechecks=0
                self.guard_confirmed_menu=0
                self.guard_allowed=0
                self.episode_number=15
                self.suspected_since=None
                self.state_confidence=.9
                self.args=types.SimpleNamespace(threshold=.68)
                self.startup_motion_threshold=2.2
                self._v2_last_bgra=None
                self.movement_keys=0
                self.action_context=types.SimpleNamespace(note_action=lambda *args: self._raise_action())
            def _raise_action(self):
                raise AssertionError('Action must not be noted on MENU')
            async def _capture(self):
                self._current=menu(score=.572)
                return None,.572,None
            def _read_screen_state(self):return self._current[0],self._current[1]
            def _guard_diagnostic(self,readings,reason,**kwargs):return 'mock_diagnostic'
        env=GuardEnv()
        with redirect_stdout(io.StringIO()):
            asyncio.run(env._step_browser(3))
        self.assertEqual(env.movement_keys,0)
        self.assertEqual(env.guard_confirmed_menu,1)
        self.assertTrue(env._guard_forced_terminal)
        self.assertTrue(env._ppo_menu_handoff_terminal_observed)
        self.assertFalse(env._ppo_boundary_menu_pending)

    def test_real_reward_method_uses_first_menu_time_not_menu_idle(self):
        from rs_v61_rewards import ProgressRewards
        from collections import Counter
        class OldEnv:
            def step(self,action):
                self._guard_forced_terminal=True
                self._ppo_menu_handoff_terminal_observed=True
                return {'image':'menu'},0.0,False,False,{'duration':999.0}
        Step= injected_class(get_method('step'),
                             {'np':__import__('numpy'), 'GuardUncertainError':RuntimeError, 'OldEnv':OldEnv},
                             base='OldEnv')
        env=Step()
        env._ppo_menu_handoff_terminal_observed=False
        env._ppo_boundary_menu_duration_at=4.50
        env.survival_rewards=ProgressRewards()
        env.survival_rewards.last_duration=4.40
        env.survival_rewards.last_milestone=8
        env.reward_parts=Counter()
        env.guard_allowed=0
        env.guard_rechecks=0
        with redirect_stdout(io.StringIO()) as trace:
            _,reward,died,truncated,info=env.step(3)
        self.assertTrue(died)
        self.assertFalse(truncated)
        self.assertEqual(info['duration'],4.50)
        self.assertEqual(info['reward_death'],-1.0)
        self.assertEqual(info['reward_milestones'],0.0)
        self.assertAlmostEqual(info['reward_survival'],.08)
        self.assertAlmostEqual(reward,-.92)
        self.assertTrue(info['ppo_menu_handoff_terminal'])
        self.assertIn('NO movement key sent',trace.getvalue())


class InstallTests(unittest.TestCase):
    def test_unknown_source_refused(self):
        with tempfile.TemporaryDirectory() as td:
            root=Path(td);(root/'tools').mkdir()
            (root/'train_rs_v612_ppo.py').write_text('bad')
            (root/'tools/ppo_reward_timing_audit.py').write_text('bad')
            with self.assertRaisesRegex(RuntimeError,'Unexpected installed'):
                installer.validate(root)
    def test_exact_v172_apply_backup_and_reject_second(self):
        # Tests in installed user repo need no external dev directory; source
        # base is reverse-constructed only for the strict installation test.
        current=(ROOT/'train_rs_v612_ppo.py').read_bytes() if (ROOT/'train_rs_v612_ppo.py').is_file() else None
        expected=installer.SPECS[0][2]
        if current is None or __import__('hashlib').sha256(current).hexdigest()!=expected:
            self.skipTest('Requires exact installed v1.7.2 to exercise --apply')
        with tempfile.TemporaryDirectory() as td:
            root=Path(td);(root/'tools').mkdir()
            (root/'train_rs_v612_ppo.py').write_bytes(current)
            (root/'tools/ppo_reward_timing_audit.py').write_bytes((ROOT/'tools/ppo_reward_timing_audit.py').read_bytes())
            written=installer.apply(root)
            self.assertEqual(len(written),2)
            self.assertEqual((root/'train_rs_v612_ppo.py').read_bytes(),(installer.PAYLOAD/'trainer.txt').read_bytes())
            self.assertTrue((root/'train_rs_v612_ppo.py.pre_ppo_post_resume_menu_v173.bak').is_file())
            with self.assertRaisesRegex(RuntimeError,'Unexpected installed'):
                installer.apply(root)
    def test_unchanged_guard_rewards_action_interface(self):
        source=TRAINER_NEW
        self.assertIn('if boundary_menu_handoff:',source)
        self.assertIn('NO key sent',source)
        self.assertIn('self.survival_rewards.step(',source)
        self.assertIn('self.action_context.note_action',source)
        self.assertNotIn('self.page.keyboard.press("r")',source)

if __name__=='__main__':unittest.main()
