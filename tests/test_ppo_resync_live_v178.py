#!/usr/bin/env python3
"""Clean-clone current-runtime PPO post-resume uncertain-frame gating tests.

No installer payload or historical backup needed. Run AFTER installing v1.7.8.
"""
import ast
import asyncio
import copy
from collections import deque
from contextlib import redirect_stdout
import io
from pathlib import Path
import tempfile
import time
import types
import unittest

ROOT=Path(__file__).resolve().parents[1]


def current_refresh():
    tree=ast.parse((ROOT/'train_rs_v612_ppo.py').read_text(encoding='utf8'))
    owner=next(n for n in ast.walk(tree) if isinstance(n,ast.ClassDef) and n.name=='PPOV61Env')
    method=next(n for n in owner.body if isinstance(n,ast.FunctionDef) and n.name=='refresh_after_ppo_optimizer')
    module=ast.fix_missing_locations(ast.Module(body=[ast.ClassDef(name='Actual',bases=[],keywords=[],body=[copy.deepcopy(method)],decorator_list=[])],type_ignores=[]))
    ns={'time':time,'GuardUncertainError':RuntimeError}
    exec(compile(module,'<current post resume refresh>','exec'),ns)
    return ns['Actual']


def f(state='GAMEPLAY',pg=1.0,pm=0.0,score=0.0):
    return (state,{'GAMEPLAY':pg,'MENU':pm},score)


def run(sequence,focus=True):
    class Page:
        async def evaluate(self,script):
            return focus
    class E(current_refresh()):
        def __init__(self):
            self._guard_hard_stopped=False
            self.native_history=deque(maxlen=4)
            self.frame_intervals=deque(maxlen=16)
            self.previous_observation_at=None
            self._ppo_resume_trace_start_perf=time.perf_counter()
            self._ppo_post_resume_menu_candidate=None
            self.last_capture_end=None
            self.state_confidence=.90
            self.args=types.SimpleNamespace(threshold=.68)
            self.page=Page()
            self._v2_last_bgra=None
            self.output_dir=Path(tempfile.gettempdir())
            self.episode_number=278
            self.sequence=list(sequence)
            self.index=0
            self.keys=[]
            self.packed=0
        async def _capture(self):
            if self.index>=len(self.sequence):raise AssertionError('Unexpected screenshot')
            state,p,score=self.sequence[self.index]
            self.current=state,p
            self.index+=1
            self.last_capture_end=time.perf_counter()
            return 'image'+str(self.index),score,None
        def _read_screen_state(self):return self.current
        def _pack(self,legacy,*,start_detection):
            self.packed+=1
            self.native_history.append(legacy)
            return tuple(self.native_history),True,0
        def _call(self,coro,timeout):return asyncio.run(coro)
    e=E()
    buf=io.StringIO()
    with redirect_stdout(buf):
        observation,reason,capture=e.refresh_after_ppo_optimizer()
    return e,observation,reason,buf.getvalue()


class CurrentRefreshTests(unittest.TestCase):
    def test_547_confidence_frame_recovers_only_after_four_gameplay(self):
        e,obs,reason,log=run([f('UNCERTAIN',.547,.004,0.)]+[f()]*4)
        self.assertIsNone(reason)
        self.assertEqual(obs,('image2','image3','image4','image5'))
        self.assertEqual(e.packed,4)
        self.assertEqual(log.count('[PPO RESYNC TRACE] frame='),4)
        self.assertEqual(e.keys,[])

    def test_retries_are_bounded_and_never_authorize_uncertain(self):
        e,obs,reason,log=run([f('UNCERTAIN',.547,.004,0.)]*5)
        self.assertIsNone(obs)
        self.assertIn('skips=4',reason)
        self.assertEqual(e.packed,0)
        self.assertEqual(e.keys,[])

    def test_menu_candidate_is_not_packed_as_gameplay(self):
        e,obs,reason,log=run([f('MENU',.01,.99,.70)])
        self.assertIsNone(obs)
        self.assertIn('state=MENU',reason)
        self.assertIsNotNone(e._ppo_post_resume_menu_candidate)
        self.assertEqual(e.keys,[])

    def test_suspicious_menu_ml_probability_refuses_retry(self):
        e,obs,reason,log=run([f('UNCERTAIN',.547,.11,0.)])
        self.assertIsNone(obs)
        self.assertEqual(e.index,1)

    def test_loss_of_focus_stops_without_capture(self):
        e,obs,reason,log=run([f()]*4,focus=False)
        self.assertIsNone(obs)
        self.assertEqual(e.index,0)

    def test_four_normal_gameplay_frames_unchanged(self):
        e,obs,reason,log=run([f()]*4)
        self.assertIsNone(reason)
        self.assertEqual(e.index,4)
        self.assertEqual(e.packed,4)
        self.assertEqual(e.keys,[])

if __name__=='__main__': unittest.main()
