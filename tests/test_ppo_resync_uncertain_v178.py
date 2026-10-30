#!/usr/bin/env python3
"""Candidate-first/live-source PPO resync regression tests; no Unity or keys.

The exact live trainer refresh method is compiled by AST. Only the conservative
ResyncWindow helper is injected, from the packaged candidate before installation
or from the installed file after installation. Other trainer logic is unmodified.
"""
import ast
import asyncio
import copy
from collections import deque
from contextlib import redirect_stdout
import importlib.util
import io
from pathlib import Path
import shutil
import sys
import tempfile
import time
import types
import unittest
from unittest.mock import patch

ROOT=Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:sys.path.insert(0,str(ROOT))
from tools.patch_ppo_resync_uncertain_v178 import (prepare,apply,PAYLOAD,ORIGINAL_SHA,sha)


def load_candidate():
    b=PAYLOAD.read_bytes()
    module=types.ModuleType('tools.ppo_resync_transient_v172')
    module.__file__=str(PAYLOAD)
    exec(compile(b,str(PAYLOAD),'exec'),module.__dict__)
    return module


def actual_refresh():
    # Read current trainer, not an earlier installer or patch payload.
    tree=ast.parse((ROOT/'train_rs_v612_ppo.py').read_text(encoding='utf8'))
    owner=next(n for n in ast.walk(tree) if isinstance(n,ast.ClassDef) and n.name=='PPOV61Env')
    method=next(n for n in owner.body if isinstance(n,ast.FunctionDef) and n.name=='refresh_after_ppo_optimizer')
    new=ast.fix_missing_locations(ast.Module(body=[ast.ClassDef(name='Actual',bases=[],keywords=[],body=[copy.deepcopy(method)],decorator_list=[])],type_ignores=[]))
    namespace={'time':time,'GuardUncertainError':RuntimeError}
    exec(compile(new,'<actual CURRENT trainer refresh>','exec'),namespace)
    return namespace['Actual']


def frame(state='GAMEPLAY',pg=1.0,pm=0.0,score=0.0):
    return state,{'GAMEPLAY':pg,'MENU':pm},score

GOOD=frame()
FIRST_FAILED=frame('UNCERTAIN',.547,.004,0.)
MENU=frame('MENU',.02,.99,.75)


def simulate(frames,*,focus=True):
    base=actual_refresh()
    class Page:
        async def evaluate(self,script):
            assert 'document.hasFocus()' in script
            return focus
    class Fake(base):
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
            self.items=list(frames)
            self.index=0
            self.captures=0
            self.packed=0
            self.sent_keys=[]
        async def _capture(self):
            if self.index>=len(self.items):
                raise AssertionError('Unexpected extra screenshot: '+str(self.index))
            state,probs,score=self.items[self.index]
            self.current=(state,probs)
            self.index+=1
            self.captures+=1
            self.last_capture_end=time.perf_counter()
            return f'image{self.captures}',score,None
        def _read_screen_state(self):return self.current
        def _pack(self,legacy,*,start_detection):
            self.packed+=1
            self.native_history.append(legacy)
            return {'images':tuple(self.native_history)},True,0
        def _call(self,coro,timeout):
            assert timeout==15
            return asyncio.run(coro)
    e=Fake()
    out=io.StringIO()
    with patch.dict(sys.modules,{'tools.ppo_resync_transient_v172':load_candidate()}),redirect_stdout(out):
        obs,reason,capture=e.refresh_after_ppo_optimizer()
    return e,obs,reason,capture,out.getvalue()


class WindowTests(unittest.TestCase):
    def test_real_failure_frame_547_then_four_good_frames_recovers(self):
        e,obs,reason,capture,out=simulate([FIRST_FAILED]+[GOOD]*4)
        self.assertIsNone(reason)
        self.assertEqual(obs['images'],('image2','image3','image4','image5'))
        self.assertEqual(e.packed,4)
        self.assertEqual(e.captures,5)
        self.assertEqual(len(e.sent_keys),0)
        self.assertIn('[PPO RESYNC TRANSIENT] skipped=1',out)
        self.assertEqual(out.count('[PPO RESYNC TRACE] frame='),4)
        self.assertLess(time.perf_counter()-capture,.2)

    def test_low_confidence_midstack_drops_partial_frames(self):
        e,obs,reason,_,out=simulate([GOOD,GOOD,FIRST_FAILED]+[GOOD]*4)
        self.assertIsNone(reason)
        self.assertEqual(obs['images'],('image4','image5','image6','image7'))
        self.assertIn('discarded_valid=2',out)
        self.assertEqual(e.packed,6)
        self.assertEqual(e.sent_keys,[])
        self.assertEqual(out.count('[PPO RESYNC TRACE] frame='),4)

    def test_four_ambiguous_then_four_valid_is_bounded(self):
        e,obs,reason,_,out=simulate([FIRST_FAILED]*4+[GOOD]*4)
        self.assertIsNone(reason)
        self.assertEqual(e.captures,8)
        self.assertEqual(obs['images'],('image5','image6','image7','image8'))
        self.assertEqual(out.count('[PPO RESYNC TRANSIENT]'),4)

    def test_fifth_uncertain_refuses_and_does_not_send_keys(self):
        e,obs,reason,_,out=simulate([FIRST_FAILED]*5)
        self.assertIsNone(obs)
        self.assertIn('skips=4',reason)
        self.assertEqual(e.captures,5)
        self.assertEqual(e.packed,0)
        self.assertEqual(e.sent_keys,[])

    def test_uncertain_followed_by_insufficient_valid_stops(self):
        e,obs,reason,_,out=simulate([GOOD]*3+[FIRST_FAILED]+[GOOD]*3+[FIRST_FAILED])
        self.assertIsNone(obs)
        self.assertIn('within 8 captures',reason)
        self.assertEqual(e.sent_keys,[])
        self.assertEqual(out.count('[PPO RESYNC TRACE] frame='),0)

    def test_probable_menu_ml_only_is_never_retried_as_gameplay(self):
        e,obs,reason,_,_=simulate([frame('UNCERTAIN',.547,.11,0.)]+[GOOD]*4)
        self.assertIsNone(obs)
        self.assertIn('unsafe post-resume',reason)
        self.assertEqual(e.captures,1)

    def test_menu_template_evidence_is_never_passively_ignored(self):
        e,obs,reason,_,_=simulate([frame('UNCERTAIN',.547,.004,.35)]+[GOOD]*4)
        self.assertIsNone(obs)
        self.assertIn('unsafe post-resume',reason)
        self.assertEqual(e.captures,1)

    def test_corroborated_menu_sets_candidate_but_cannot_move(self):
        e,obs,reason,_,_=simulate([MENU]+[GOOD]*4)
        self.assertIsNone(obs)
        self.assertIn('state=MENU',reason)
        self.assertIsNotNone(e._ppo_post_resume_menu_candidate)
        self.assertEqual(e.captures,1)
        self.assertEqual(e.sent_keys,[])

    def test_other_screen_or_loading_still_aborts(self):
        for name in ['LOADING','OTHER']:
            with self.subTest(name=name):
                e,obs,reason,_,_=simulate([frame(name,.547,.004,0.)]+[GOOD]*4)
                self.assertIsNone(obs)
                self.assertEqual(e.captures,1)

    def test_uncertain_after_menuish_gameplay_still_aborts(self):
        e,obs,reason,_,_=simulate([frame('GAMEPLAY',1.0,0.,.68)]+[GOOD]*4)
        self.assertIsNone(obs)
        self.assertEqual(e.captures,1)

    def test_all_four_good_unchanged(self):
        e,obs,reason,_,out=simulate([GOOD]*4)
        self.assertIsNone(reason)
        self.assertEqual(e.captures,4)
        self.assertEqual(e.packed,4)
        self.assertEqual(e.sent_keys,[])
        self.assertNotIn('[PPO RESYNC TRANSIENT]',out)

    def test_focus_loss_aborts_before_any_capture(self):
        e,obs,reason,_,_=simulate([GOOD]*4,focus=False)
        self.assertIsNone(obs)
        self.assertIn('focus/visibility lost',reason)
        self.assertEqual(e.captures,0)

    def test_nonfinite_uncertain_still_aborts(self):
        e,obs,reason,_,_=simulate([frame('UNCERTAIN',float('nan'),.004,0.)]+[GOOD]*4)
        self.assertIsNone(obs)
        self.assertIn('invalid score',reason)

    def test_zero_gameplay_probability_is_not_positive_gameplay(self):
        e,obs,reason,_,out=simulate([frame('UNCERTAIN',0.,.004,0.)]+[GOOD]*4)
        self.assertIsNone(reason)
        self.assertEqual(e.packed,4)
        self.assertEqual(e.captures,5)
        self.assertIn('skipped=1',out)


class InstallerTests(unittest.TestCase):
    def test_exact_source_and_refuse_second_install(self):
        with tempfile.TemporaryDirectory() as d:
            repo=Path(d)
            (repo/'tools').mkdir()
            (repo/'train_rs_v612_ppo.py').write_bytes((ROOT/'train_rs_v612_ppo.py').read_bytes())
            orig=(ROOT/'tools/_ppo_resync_uncertain_v178_payload/ppo_resync_transient_v172_original.txt').read_bytes()
            (repo/'tools/ppo_resync_transient_v172.py').write_bytes(orig)
            self.assertEqual(sha(orig),ORIGINAL_SHA)
            prepare(repo)
            backup=apply(repo)
            self.assertEqual(backup.read_bytes(),orig)
            self.assertEqual((repo/'tools/ppo_resync_transient_v172.py').read_bytes(),PAYLOAD.read_bytes())
            with self.assertRaisesRegex(RuntimeError,'Unexpected helper normalized SHA256'):
                apply(repo)

    def test_crlf_windows_checkout_preserved_and_supported(self):
        with tempfile.TemporaryDirectory() as d:
            repo=Path(d)
            (repo/'tools').mkdir()
            (repo/'train_rs_v612_ppo.py').write_bytes((ROOT/'train_rs_v612_ppo.py').read_bytes())
            orig=(ROOT/'tools/_ppo_resync_uncertain_v178_payload/ppo_resync_transient_v172_original.txt').read_bytes()
            win_original=orig.replace(b'\n',b'\r\n')
            (repo/'tools/ppo_resync_transient_v172.py').write_bytes(win_original)
            prepare(repo)
            backup=apply(repo)
            self.assertEqual(backup.read_bytes(),win_original)
            patched=(repo/'tools/ppo_resync_transient_v172.py').read_bytes()
            self.assertIn(b'\r\n',patched)
            self.assertEqual(patched.replace(b'\r\n',b'\n'),PAYLOAD.read_bytes())

    def test_unknown_source_refused(self):
        with tempfile.TemporaryDirectory() as d:
            repo=Path(d)
            (repo/'tools').mkdir()
            (repo/'train_rs_v612_ppo.py').write_bytes((ROOT/'train_rs_v612_ppo.py').read_bytes())
            (repo/'tools/ppo_resync_transient_v172.py').write_text('other source')
            with self.assertRaisesRegex(RuntimeError,'Unexpected helper normalized SHA256'):
                prepare(repo)


if __name__=='__main__':unittest.main()
