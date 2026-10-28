#!/usr/bin/env python3
"""Tests current trainer runtime PAUSED exclusion with actual AST method code."""
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

import numpy as np
import sys
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from tools.ppo_pause_visual_v179 import (
    pause_panel_signature, matches_prior_paused_panel)

ROOT=Path(__file__).resolve().parents[1]


def candidate_source():
    # Exercise the checked-out runtime; no installer, payload or backup needed.
    return (ROOT/'train_rs_v612_ppo.py').read_text(encoding='utf8')


def actual_refresh():
    source=candidate_source()
    tree=ast.parse(source)
    owner=next(n for n in ast.walk(tree) if isinstance(n,ast.ClassDef) and n.name=='PPOV61Env')
    method=next(n for n in owner.body if isinstance(n,ast.FunctionDef) and n.name=='refresh_after_ppo_optimizer')
    module=ast.fix_missing_locations(ast.Module(body=[ast.ClassDef(
        name='Actual',bases=[],keywords=[],body=[copy.deepcopy(method)],decorator_list=[]
    )],type_ignores=[]))
    ns={'time':time,'GuardUncertainError':RuntimeError}
    exec(compile(module,'<current trainer refresh>','exec'),ns)
    return ns['Actual']


def screen(*,paused=False,variant=0):
    # Synthetic screenshot changes background and controls panel independently.
    arr=np.zeros((270,480,4),dtype=np.uint8)
    arr[:,:,:3]=np.array([24+variant,83+variant,62+variant],np.uint8)
    arr[:,:,3]=255
    if paused:
        arr[120:160,202:278,:3]=220
        arr[133:145,208:235,:3]=[30,192,245]   # yellow-ish Resume button
        arr[133:145,240:275,:3]=[185,191,190]
    return arr


def frame(state='GAMEPLAY', pg=1., pm=0., score=0., *, paused=False, dom_before=False, dom_after=False):
    return dict(state=state,pg=pg,pm=pm,score=score,paused=paused,
                dom_before=dom_before,dom_after=dom_after)


def run(frames, *, anchor=True, focus=True):
    class Overlay:
        def __init__(self,e):self.e=e
        async def is_visible(self):
            pre=self.e.dom_check % 2 == 0
            i=self.e.index if pre else self.e.index-1
            if i >= len(self.e.frames): raise AssertionError('Unexpected DOM check')
            f=self.e.frames[i]
            result=f['dom_before'] if pre else f['dom_after']
            self.e.dom_check+=1
            return result
    class Page:
        def __init__(self,e):self.e=e
        async def evaluate(self,script):return focus
        def locator(self,sel):
            assert sel=='.game-pause-panel[aria-label="Game paused"]'
            return Overlay(self.e)
    class E(actual_refresh()):
        def __init__(self):
            self._guard_hard_stopped=False
            self.native_history=deque(maxlen=4)
            self.frame_intervals=deque(maxlen=16)
            self.previous_observation_at=None
            self._ppo_resume_trace_start_perf=time.perf_counter()
            self._ppo_post_resume_menu_candidate=None
            self._ppo_pause_visual_reference=(pause_panel_signature(screen(paused=True)) if anchor else None)
            self.last_capture_end=None
            self.state_confidence=.9
            self.args=types.SimpleNamespace(threshold=.68)
            self.page=Page(self)
            self._v2_last_bgra=None
            self.output_dir=Path(tempfile.gettempdir())
            self.episode_number=278
            self.frames=list(frames)
            self.index=0
            self.dom_check=0
            self.packed=0
            self.keys=[]
        async def _capture(self):
            if self.index>=len(self.frames):raise AssertionError('Unexpected screenshot')
            f=self.frames[self.index]
            self._v2_last_bgra=screen(paused=f['paused'],variant=self.index)
            self.current=f
            self.index+=1
            self.last_capture_end=time.perf_counter()
            return f'img{self.index}',f['score'],None
        def _read_screen_state(self):
            f=self.current
            return f['state'],{'GAMEPLAY':f['pg'],'MENU':f['pm']}
        def _pack(self,legacy,*,start_detection):
            self.packed+=1
            self.native_history.append(legacy)
            return tuple(self.native_history),True,0
        def _call(self,coro,timeout):return asyncio.run(coro)
    e=E()
    buffer=io.StringIO()
    with redirect_stdout(buffer):
        obs,reason,at=e.refresh_after_ppo_optimizer()
    return e,obs,reason,buffer.getvalue()


class PauseVisualEvidenceTests(unittest.TestCase):
    def test_matches_same_rendered_pause_with_minor_noise(self):
        base=screen(paused=True)
        noisy=base.copy()
        noisy[:,:,:3]=np.clip(noisy[:,:,:3].astype(np.int16)+3,0,255).astype(np.uint8)
        match,delta=matches_prior_paused_panel(noisy,pause_panel_signature(base))
        self.assertTrue(match)
        self.assertLess(delta,5)

    def test_does_not_match_normal_gameplay(self):
        match,delta=matches_prior_paused_panel(screen(paused=False),pause_panel_signature(screen(paused=True)))
        self.assertFalse(match)
        self.assertGreater(delta,18)

    def test_missing_reference_fails_closed(self):
        with self.assertRaisesRegex(ValueError,'Missing verified'):
            matches_prior_paused_panel(screen(),None)

    def test_actual_trainer_stores_reference_only_after_freeze(self):
        code=candidate_source()
        tree=ast.parse(code)
        owner=next(n for n in ast.walk(tree) if isinstance(n,ast.ClassDef) and n.name=='PPOV61Env')
        pause=next(n for n in owner.body if isinstance(n,ast.FunctionDef) and n.name=='pause_for_ppo_optimizer')
        src=ast.unparse(pause)
        self.assertLess(src.index('self._call(pause(), timeout=15)'),src.index('pause_panel_signature(self._v2_last_bgra)'))
        self.assertLess(src.index('pause_panel_signature(self._v2_last_bgra)'),src.index('self._ppo_pause_pending ='))


class CurrentRefreshTests(unittest.TestCase):
    def test_four_normal_frames_still_pass(self):
        e,obs,reason,log=run([frame()]*4)
        self.assertIsNone(reason)
        self.assertEqual(obs,('img1','img2','img3','img4'))
        self.assertEqual(e.packed,4)
        self.assertEqual(e.keys,[])
        self.assertIsNone(e._ppo_pause_visual_reference)

    def test_dom_pause_is_not_gameplay_even_with_confident_ml(self):
        e,obs,reason,log=run([frame(dom_before=True)]+[frame()]*4)
        self.assertIsNone(reason)
        self.assertEqual(obs,('img2','img3','img4','img5'))
        self.assertEqual(e.packed,4)
        self.assertIn('runtime_state=PAUSED',log)
        self.assertIn('source=DOM',log)

    def test_dom_overlay_appears_after_capture(self):
        e,obs,reason,log=run([frame(dom_after=True)]+[frame()]*4)
        self.assertIsNone(reason)
        self.assertEqual(e.packed,4)
        self.assertIn('source=DOM',log)

    def test_stale_rendered_pause_dom_hidden_is_not_gameplay(self):
        e,obs,reason,log=run([frame(paused=True)]+[frame()]*4)
        self.assertIsNone(reason)
        self.assertEqual(obs,('img2','img3','img4','img5'))
        self.assertEqual(e.packed,4)
        self.assertIn('source=RENDERED_FRAME',log)
        self.assertEqual(e.keys,[])

    def test_overlay_midstack_discards_prior_positive_frames(self):
        frames=[frame(),frame(),frame(paused=True)]+[frame()]*4
        e,obs,reason,log=run(frames)
        self.assertIsNone(reason)
        self.assertEqual(obs,('img4','img5','img6','img7'))
        self.assertEqual(e.packed,6) # 2 old+4 new packs, but only last 4 used
        self.assertEqual(log.count('[PPO RESYNC TRACE] frame='),4)

    def test_overlay_persistent_fails_without_keys(self):
        e,obs,reason,log=run([frame(paused=True)]*4)
        self.assertIsNone(obs)
        self.assertIn('PAUSED overlay persisted',reason)
        self.assertEqual(e.packed,0)
        self.assertEqual(e.keys,[])

    def test_uncertain_then_pause_then_four_good_frames(self):
        frames=[frame('UNCERTAIN',.547,.004),frame(paused=True)]+[frame()]*4
        e,obs,reason,log=run(frames)
        self.assertIsNone(reason)
        self.assertEqual(obs,('img3','img4','img5','img6'))
        self.assertEqual(e.packed,4)

    def test_missing_anchor_safe_stop_before_capture(self):
        e,obs,reason,log=run([frame()]*4,anchor=False)
        self.assertIsNone(obs)
        self.assertIn('Missing verified PAUSED visual reference',reason)
        self.assertEqual(e.index,0)

    def test_menu_remains_terminal_not_pause(self):
        e,obs,reason,log=run([frame('MENU',.01,.99,.75)])
        self.assertIsNone(obs)
        self.assertIn('state=MENU',reason)
        self.assertIsNotNone(e._ppo_post_resume_menu_candidate)
        self.assertEqual(e.keys,[])

    def test_focus_loss_stops_without_key(self):
        e,obs,reason,log=run([frame()]*4,focus=False)
        self.assertIsNone(obs)
        self.assertIn('focus/visibility',reason)
        self.assertEqual(e.packed,0)

class TransientWindowTests(unittest.TestCase):
    def observe(self, window, state='UNCERTAIN', pg=.547, pm=.004, score=0.):
        return window.observe(state=state, probs={'GAMEPLAY':pg,'MENU':pm},
                              menu_score=score,confidence=.90,menu_threshold=.68)

    def test_four_uncertain_skips_then_four_good_frames(self):
        from tools.ppo_resync_transient_v172 import ResyncWindow
        w=ResyncWindow()
        self.assertEqual([self.observe(w) for _ in range(4)],['RETRY']*4)
        self.assertEqual([self.observe(w,'GAMEPLAY',1.,0.) for _ in range(4)],
                         ['ACCEPT','ACCEPT','ACCEPT','DONE'])

    def test_fifth_uncertain_frame_aborts(self):
        from tools.ppo_resync_transient_v172 import ResyncWindow
        w=ResyncWindow()
        for _ in range(4): self.observe(w)
        self.assertEqual(self.observe(w),'ABORT')

    def test_uncertain_invalidates_partial_gameplay_stack(self):
        from tools.ppo_resync_transient_v172 import ResyncWindow
        w=ResyncWindow()
        self.observe(w,'GAMEPLAY',1.,0.)
        self.assertEqual(self.observe(w),'RETRY')
        self.assertEqual(w.accepted,0)

    def test_menu_loading_and_invalid_values_abort(self):
        from tools.ppo_resync_transient_v172 import ResyncWindow
        for state,pg,pm,score in [('MENU',.01,.99,.75),('LOADING',.1,0.,0.),
                                   ('UNCERTAIN',.5,.2,0.),('GAMEPLAY',float('nan'),0.,0.)]:
            with self.subTest(state=state,pg=pg,pm=pm):
                self.assertEqual(self.observe(ResyncWindow(),state,pg,pm,score),'ABORT')

if __name__=='__main__':unittest.main()
