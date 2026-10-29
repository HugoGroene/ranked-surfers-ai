#!/usr/bin/env python3
"""Clean-clone PPO v1.7.6 runtime regressions using actual checked-out trainer.

Never imports historical patch payloads or installer backups.
Run with: python -B -m unittest discover -s tests -p test_ppo_current_runtime.py -v
"""
import ast
import asyncio
import copy
from pathlib import Path
import sys
import tempfile
import time
import types
import unittest
from unittest.mock import patch
import zipfile

import numpy as np
from playwright.async_api import TimeoutError as PlaywrightTimeoutError

ROOT=Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:sys.path.insert(0,str(ROOT))
from rs_v612_gameplay_guard import GuardUncertainError
from tools.ppo_atomic_checkpoint_v176 import save_atomic_checkpoint
from tools.ppo_rollout_menu_handoff_v17 import boundary_menu_candidate


def current_code():
    return (ROOT / "train_rs_v612_ppo.py").read_text(encoding="utf-8")


def actual_class_method(name):
    tree=ast.parse(current_code())
    owner=next(n for n in ast.walk(tree) if isinstance(n,ast.ClassDef) and n.name=='PPOV61Env')
    method=next(n for n in owner.body if isinstance(n,ast.FunctionDef) and n.name==name)
    mod=ast.fix_missing_locations(ast.Module(body=[ast.ClassDef(name='Actual',bases=[],keywords=[],
                        body=[copy.deepcopy(method)],decorator_list=[])],type_ignores=[]))
    scope={'GuardUncertainError':GuardUncertainError}
    exec(compile(mod, '<CURRENT TRAINER '+name+'>','exec'),scope)
    return scope['Actual']


class Locator:
    def __init__(self,e,kind):self.e,self.kind=e,kind
    async def is_visible(self):
        return self.e.overlay if self.kind=='overlay' else self.e.button
    async def count(self):
        if self.kind=='overlay':return 1
        return (2 if self.e.duplicate else 1) if self.e.button else 0
    async def click(self,timeout):
        self.e.clicks+=1
        act=self.e.click_script.pop(0)
        if act=='timeout_hidden':
            self.e.overlay=False
            raise PlaywrightTimeoutError('Element detached after successful click')
        if act=='timeout_visible':
            raise PlaywrightTimeoutError('Resume target detached')
        if act=='timeout_menu':
            self.e.button=False
            raise PlaywrightTimeoutError('Pause target detached')
        if act=='pause_success':
            self.e.overlay=True
            return
        if act=='resume_success':
            self.e.overlay=False
            return
        raise AssertionError(act)
    async def wait_for(self,state,timeout):
        if state=='visible' and self.e.overlay:return
        if state=='hidden' and not self.e.overlay:return
        raise PlaywrightTimeoutError('overlay not in expected state')
    def get_by_role(self,role,name,exact):
        assert self.kind=='overlay' and role=='button' and name=='Resume' and exact
        return Locator(self.e,'button')


class Page:
    def __init__(self,e):self.e=e
    async def evaluate(self,script):return self.e.focus
    def locator(self,selector):
        if selector=='.game-pause-panel[aria-label="Game paused"]':return Locator(self.e,'overlay')
        if selector=='button[aria-label="Pause game"]':return Locator(self.e,'button')
        raise AssertionError(selector)


def env_for_pause(states, script, *, suspected=False):
    e=actual_class_method('pause_for_ppo_optimizer')()
    e.focus=True;e.duplicate=False;e.overlay=False;e.button=True;e.clicks=0
    e.click_script=list(script);e.page=Page(e)
    e.suspected_since=time.monotonic()-.05 if suspected else None
    e.menu_scores=[];e.clear_streak=0
    e._guard_hard_stopped=False;e._ppo_pause_pending=None
    e._ppo_boundary_menu_duration_at=None
    e.args=types.SimpleNamespace(threshold=.68)
    e.state_confidence=.90;e.startup_motion_threshold=2.2
    e.episode_start=time.monotonic()-2.0
    e.survival_rewards=types.SimpleNamespace(last_duration=.4)
    e.output_dir=Path(tempfile.gettempdir());e.episode_number=4
    e._v2_last_bgra=np.zeros((64,64,4),np.uint8)
    e.crop=(.1,.9,.1,.9)
    e._motion_view=lambda *_:np.zeros((16,16),np.uint8)
    e.samples=list(states);e.last=('UNCERTAIN',0)
    async def capture():
        e.last=e.samples.pop(0) if e.samples else ('GAMEPLAY',0.)
        return None,e.last[1],None
    def classify():
        state,score=e.last
        return state,{'GAMEPLAY':1. if state=='GAMEPLAY' else 0.,
                      'MENU':1. if state=='MENU' else 0.}
    e._capture=capture;e._read_screen_state=classify
    e._call=lambda coro,timeout:asyncio.run(coro)
    return e


def env_for_resume(script):
    e=actual_class_method('resume_after_ppo_optimizer')()
    e.focus=True;e.duplicate=False;e.overlay=True;e.button=True;e.clicks=0
    e.click_script=list(script);e.page=Page(e)
    e.args=types.SimpleNamespace(threshold=.68)
    e._v2_last_bgra=np.zeros((64,64,4),np.uint8)
    e.crop=(.1,.9,.1,.9)
    e._motion_view=lambda *_:np.zeros((16,16),np.uint8)
    e._capture=async_capture=lambda_captured(e)
    e._call=lambda coro,timeout:asyncio.run(coro)
    e.episode_start=time.monotonic()-2.5
    token=(time.monotonic()-1.2,time.perf_counter()-1.2,time.time()-1.2,1.3)
    # episode start chosen exactly from the token as the original active clock
    e.episode_start=token[0]-token[3]
    e._ppo_pause_pending=token
    e._ppo_resume_trace_start_perf=None
    return e,types.SimpleNamespace(env=e)


def lambda_captured(e):
    async def cap():return None,0.,None
    return cap


class BoundaryTests(unittest.TestCase):
    @staticmethod
    def run_pause(e):
        real_sleep=asyncio.sleep
        # 8 ms can be below Windows' monotonic clock tick. Real production
        # capture waits are >=33 ms; keep the shortened test wait above a tick.
        async def short_sleep(_):await real_sleep(.020)
        with patch('asyncio.sleep',new=short_sleep):
            return e.pause_for_ppo_optimizer()

    def test_transient_camera_angle_two_fresh_play_frames(self):
        e=env_for_pause([('UNCERTAIN',0.),('GAMEPLAY',0.),('GAMEPLAY',0.)],['pause_success'])
        self.assertEqual(self.run_pause(e),0.)
        self.assertEqual(e.clicks,1)
        self.assertIsNotNone(e._ppo_pause_pending)

    def test_pending_suspicion_requires_two_fresh_gameplay_frames(self):
        e=env_for_pause([('GAMEPLAY',0.),('GAMEPLAY',0.),('GAMEPLAY',0.)],
                        ['pause_success'],suspected=True)
        self.run_pause(e)
        self.assertIsNone(e.suspected_since)
        self.assertEqual(e.clicks,1)

    def test_confirmed_death_handoff_does_not_click(self):
        e=env_for_pause([('UNCERTAIN',0.)]+[('MENU',.62)]*4,['pause_success'])
        with self.assertRaises(GuardUncertainError) as ctx:
            self.run_pause(e)
        self.assertTrue(boundary_menu_candidate(ctx.exception))
        self.assertEqual(e.clicks,0)
        self.assertGreater(e._ppo_boundary_menu_duration_at,1.5)

    def test_uncertain_eight_frames_safe_stop(self):
        e=env_for_pause([('UNCERTAIN',0.)]*9,['pause_success'])
        with self.assertRaisesRegex(GuardUncertainError,'control unresolved'):
            self.run_pause(e)
        self.assertEqual(e.clicks,0)

    def test_unconfirmed_menu_never_handoffs(self):
        e=env_for_pause([('MENU',.0)]*9,['pause_success'])
        with self.assertRaisesRegex(GuardUncertainError,'control unresolved'):
            self.run_pause(e)
        self.assertIsNone(e._ppo_boundary_menu_duration_at)
        self.assertEqual(e.clicks,0)

    def test_pending_suspicion_unresolved_must_not_click(self):
        e=env_for_pause([('GAMEPLAY',0.)]+[('UNCERTAIN',0.)]*8,
                        ['pause_success'],suspected=True)
        with self.assertRaisesRegex(GuardUncertainError,'control unresolved'):
            self.run_pause(e)
        self.assertIsNotNone(e.suspected_since)
        self.assertEqual(e.clicks,0)

    def test_regular_pause_still_clicks_once(self):
        e=env_for_pause([('GAMEPLAY',0.)],['pause_success'])
        self.assertEqual(self.run_pause(e),0.)
        self.assertEqual(e.clicks,1)


class ResumeTests(unittest.TestCase):
    def run_resume(self,e,m):
        real_sleep=asyncio.sleep
        async def short_sleep(_):await real_sleep(.020)
        def compensated(raw,dm,dp,*,monitor,elapsed_wall):
            self.assertIs(raw,e)
            self.assertIs(monitor,m)
            raw.episode_start+=dm
            return {'paused_wall_seconds':dm,'corrected_episode_timer':True}
        with patch('tools.ppo_optimizer_pause_clock.compensate_pause_clocks',side_effect=compensated),patch('asyncio.sleep',new=short_sleep):
            return e.resume_after_ppo_optimizer(m)

    def test_normal_resume(self):
        e,m=env_for_resume(['resume_success'])
        self.assertIn('paused_wall_seconds',self.run_resume(e,m))
        self.assertEqual(e.clicks,1)
        self.assertIsNone(e._ppo_pause_pending)

    def test_timeout_but_overlay_hidden_does_not_retry(self):
        e,m=env_for_resume(['timeout_hidden'])
        self.run_resume(e,m)
        self.assertEqual(e.clicks,1)
        self.assertIsNone(e._ppo_pause_pending)

    def test_timeout_overlay_still_visible_one_frozen_retry(self):
        e,m=env_for_resume(['timeout_visible','resume_success'])
        self.run_resume(e,m)
        self.assertEqual(e.clicks,2)

    def test_timeout_twice_no_third_click(self):
        e,m=env_for_resume(['timeout_visible','timeout_visible'])
        with self.assertRaisesRegex(GuardUncertainError,'timed out twice'):
            self.run_resume(e,m)
        self.assertEqual(e.clicks,2)
        self.assertIsNotNone(e._ppo_pause_pending)

    def test_button_disappeared_stops_no_retry(self):
        e,m=env_for_resume(['timeout_visible','resume_success'])
        old_click=Locator.click
        async def disappear_button(loc,timeout):
            loc.e.button=False
            return await old_click(loc,timeout)
        with patch.object(Locator,'click',new=disappear_button):
            with self.assertRaisesRegex(GuardUncertainError,'not unique and visible'):
                self.run_resume(e,m)
        self.assertEqual(e.clicks,1)

    def test_resume_paused_track_moving_cannot_retry(self):
        e,m=env_for_resume(['timeout_visible','resume_success'])
        n=[0]
        def motion(*_):
            n[0]+=1
            return np.full((16,16), n[0]*10,dtype=np.uint8)
        e._motion_view=motion
        with self.assertRaisesRegex(GuardUncertainError,'frozen Unity not independently verified'):
            self.run_resume(e,m)
        self.assertEqual(e.clicks,1)


class FakeModel:
    def __init__(self,payload,fail=False):self.payload=payload;self.fail=fail
    def save(self,path):
        with zipfile.ZipFile(path,'w') as z:
            z.writestr('model.txt',self.payload)
        if self.fail:
            with open(path,'wb') as f:f.write(b'incomplete zip')
            raise RuntimeError('simulated interruption during save')


class CallbackSaveTests(unittest.TestCase):
    def test_source_periodic_save_is_deferred_under_optimizer_pause(self):
        tree=ast.parse(current_code())
        cls=next(n for n in ast.walk(tree) if isinstance(n,ast.ClassDef) and n.name=='V61Callback')
        step=next(n for n in cls.body if isinstance(n,ast.FunctionDef) and n.name=='_on_step')
        end=next(n for n in cls.body if isinstance(n,ast.FunctionDef) and n.name=='_on_rollout_end')
        step_code=ast.unparse(step)
        end_code=ast.unparse(end)
        self.assertIn('self._checkpoint_due_at_boundary = True',step_code)
        self.assertIn('if self._checkpoint_due_at_boundary:',end_code)
        self.assertIn("save_atomic_checkpoint(self.model, output / 'latest.zip')",end_code)
        self.assertNotIn("self.model.save(output / 'latest')",step_code)
        top_train=next(n for n in tree.body if isinstance(n,ast.FunctionDef) and n.name=='train')
        self.assertIn("save_atomic_checkpoint(model, dest / 'latest.zip')",ast.unparse(top_train))


class AtomicTests(unittest.TestCase):
    def test_previous_checkpoint_preserved_on_write_failure(self):
        with tempfile.TemporaryDirectory() as d:
            target=Path(d)/'latest.zip'
            save_atomic_checkpoint(FakeModel('old'),target)
            before=target.read_bytes()
            with self.assertRaisesRegex(RuntimeError,'simulated interruption'):
                save_atomic_checkpoint(FakeModel('new',fail=True),target)
            self.assertEqual(target.read_bytes(),before)
            self.assertFalse(list(Path(d).glob('.latest.v176.*')))

    def test_success_keeps_previous_copy(self):
        with tempfile.TemporaryDirectory() as d:
            target=Path(d)/'latest.zip'
            save_atomic_checkpoint(FakeModel('old'),target)
            save_atomic_checkpoint(FakeModel('new'),target)
            with zipfile.ZipFile(target) as z:self.assertEqual(z.read('model.txt'),b'new')
            with zipfile.ZipFile(Path(d)/'latest.prev.zip') as z:self.assertEqual(z.read('model.txt'),b'old')

    def test_invalid_zip_does_not_replace(self):
        class Invalid:
            def save(self,path):Path(path).write_text('not a ZIP')
        with tempfile.TemporaryDirectory() as d:
            target=Path(d)/'latest.zip'
            save_atomic_checkpoint(FakeModel('valid'),target)
            with self.assertRaises(zipfile.BadZipFile):save_atomic_checkpoint(Invalid(),target)
            with zipfile.ZipFile(target) as z:self.assertEqual(z.read('model.txt'),b'valid')

    def test_original_checkpoint_path_is_rejected(self):
        with tempfile.TemporaryDirectory() as d:
            with self.assertRaisesRegex(ValueError,'only permits local latest'):
                save_atomic_checkpoint(FakeModel('not safe'),Path(d)/'original.zip')



if __name__=='__main__':unittest.main()
