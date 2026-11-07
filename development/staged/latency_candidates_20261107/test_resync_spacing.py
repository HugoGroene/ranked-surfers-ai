"""Resume cadence, actual verifier branches, and conservative timestamp fallback."""
import ast
import asyncio
from collections import deque
import copy
import os
from pathlib import Path
import tempfile
import time
from types import SimpleNamespace
import unittest
from unittest.mock import patch
from development.experiments.latency_tail_20261107.resync_spacing import CaptureSpacing, replace_spacing


class SpacingTests(unittest.IsolatedAsyncioTestCase):
    async def test_processing_counts_toward_interval_and_early_wake_rechecked(self):
        now = [10.]; waits = []
        async def sleep(seconds):
            waits.append(seconds)
            now[0] += seconds/2 if len(waits)==1 else seconds
        cadence = CaptureSpacing(lambda: now[0],sleep)
        await cadence.before();cadence.captured({'request_started_at':10.})
        now[0] += .020
        await cadence.before()
        self.assertAlmostEqual(waits[0],.013)
        self.assertGreaterEqual(now[0],10.033)
        cadence.captured({'request_started_at':now[0]})
        now[0] += .050
        before=len(waits);await cadence.before()
        self.assertEqual(len(waits),before)

    async def test_missing_stale_future_nonfinite_metadata_keep_full_gap(self):
        for stamp in (None,9.,11.,float('nan'),True):
            now=[10.]
            async def sleep(seconds):now[0]+=seconds
            cadence=CaptureSpacing(lambda:now[0],sleep)
            await cadence.before();now[0]=10.020
            cadence.captured({'request_started_at':stamp})
            await cadence.before()
            self.assertGreaterEqual(now[0],10.053-1e-12)
            self.assertEqual(cadence.fallbacks,1)


def actual_method():
    source=Path(__file__).resolve().parents[3]/'train_rs_v612_ppo.py'
    tree=ast.parse(source.read_text(encoding='utf8'))
    node=next(n for n in ast.walk(tree) if isinstance(n,ast.FunctionDef)
              and n.name=='refresh_after_ppo_optimizer')
    namespace=dict(time=time,GuardUncertainError=RuntimeError)
    exec(compile(ast.Module(body=[copy.deepcopy(node)],type_ignores=[]),str(source),'exec'),namespace)
    return namespace[node.name]


class FakeEnv:
    def __init__(self,folder,states=None,paused=False,focused=True):
        self._guard_hard_stopped=False;self._ppo_pause_visual_reference=object()
        self._v2_last_bgra=None;self.last_capture_end=time.perf_counter()
        self.native_history=deque();self.frame_intervals=deque()
        self.output_dir=Path(folder);self.episode_number=1
        self.state_confidence=.9;self.args=SimpleNamespace(threshold=.68)
        self.states=iter(states or ['GAMEPLAY']*4);self.state=None;self.packed=[];self.captures=0
        async def visible():return paused
        async def evaluate(_):return focused
        self.page=SimpleNamespace(locator=lambda _:SimpleNamespace(is_visible=visible),evaluate=evaluate)
    def _call(self,coro,timeout):return asyncio.run(coro)
    async def _capture(self):
        self.captures+=1;self.state=next(self.states,'GAMEPLAY')
        self._headless_capture_timing={'request_started_at':time.perf_counter()}
        await asyncio.sleep(.001)
        self.last_capture_end=time.perf_counter()
        return object(),.95 if self.state=='MENU' else 0.,None
    def _read_screen_state(self):return self.state,{self.state:1.}
    def _pack(self,legacy,start_detection):
        self.packed.append(self.state)
        return legacy,True,0.


class VerifierTests(unittest.TestCase):
    def run_case(self,**kwargs):
        with tempfile.TemporaryDirectory() as folder,patch.dict(os.environ,{'RS_STATEFUL_GUARD':'0'}),\
                patch('tools.ppo_pause_visual_v179.matches_prior_paused_panel',return_value=(False,99.)):
            env=FakeEnv(folder,**kwargs)
            result=replace_spacing(actual_method())(env)
            return result,env
    def test_actual_verifier_still_requires_four_fresh_frames(self):
        result,env=self.run_case()
        self.assertIsNotNone(result[0]);self.assertIsNone(result[1])
        self.assertEqual(env.captures,4);self.assertEqual(env.packed,['GAMEPLAY']*4)
        self.assertTrue(all(t>=33-1e-6 for t in env._candidate_resync_spacing.report()['request_intervals_ms']))
    def test_menu_does_not_authorize_observation(self):
        result,env=self.run_case(states=['GAMEPLAY','MENU'])
        self.assertIsNone(result[0]);self.assertIsNotNone(result[1])
        self.assertEqual(env.packed,['GAMEPLAY'])
        self.assertIsNotNone(env._ppo_post_resume_menu_candidate)
    def test_pause_excluded_even_with_gameplay_classifier(self):
        result,env=self.run_case(paused=True)
        self.assertIsNone(result[0]);self.assertIn('PAUSED',result[1])
        self.assertEqual(env.packed,[])
    def test_focus_loss_stops_before_capture(self):
        result,env=self.run_case(focused=False)
        self.assertIsNone(result[0]);self.assertEqual(env.captures,0)
    def test_source_drift_refused(self):
        with self.assertRaises(RuntimeError):replace_spacing(FakeEnv._capture)


if __name__=='__main__':unittest.main()
