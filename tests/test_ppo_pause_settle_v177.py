#!/usr/bin/env python3
"""Regression of v1.7.7 overlay settling race against current/candidate method.

Tests import the checked-in clean-clone test harness and AST-extract the actual
v1.7.7 pause method. They never reconstruct historical installer patches.
"""
import hashlib
from pathlib import Path
import tempfile
import types
import unittest
from unittest.mock import patch
import sys

ROOT=Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path: sys.path.insert(0,str(ROOT))
import test_ppo_current_runtime as runtime
from rs_v612_gameplay_guard import GuardUncertainError
from tools.ppo_rollout_menu_handoff_v17 import boundary_menu_candidate
from tools.patch_ppo_pause_settle_v177 import SOURCE_SHA, PAYLOAD, prepare, apply, sha, _only_pause_changed


def candidate_code():
    original=(ROOT/'train_rs_v612_ppo.py').read_bytes()
    new=PAYLOAD.read_bytes()
    if sha(original)==SOURCE_SHA:return new.decode('utf-8')
    if original==new:return original.decode('utf-8')
    raise AssertionError('Unexpected trainer source; refusing to test unrelated method')


class IntermittentOverlay(runtime.Locator):
    async def is_visible(self):
        if self.kind=='overlay':
            self.e.overlay_checks += 1
            if self.e.overlay_checks in self.e.missing_at:
                if self.e.temporary:
                    return False
                self.e.overlay=False
                if getattr(self.e, "duplicate_after_vanish", False):
                    self.e.duplicate=True
        return await super().is_visible()


class ChangingPage(runtime.Page):
    def locator(self,selector):
        if selector=='.game-pause-panel[aria-label="Game paused"]':
            return IntermittentOverlay(self.e,'overlay')
        return super().locator(selector)


def env(states,script,*,missing_at=(2,),temporary=False,duplicate=False):
    with patch.object(runtime,'current_code',side_effect=candidate_code):
        e=runtime.env_for_pause(states,script)
    e.page=ChangingPage(e)
    e.overlay_checks=0
    e.missing_at=set(missing_at)
    e.temporary=temporary
    e.duplicate=False
    e.duplicate_after_vanish=duplicate
    return e


class PauseOverlaySettleTests(unittest.TestCase):
    def test_one_reverified_second_click_after_overlay_vanishes(self):
        e=env([('GAMEPLAY',0.),('GAMEPLAY',0.),('GAMEPLAY',0.)],
              ['pause_success','pause_success'])
        self.assertEqual(runtime.BoundaryTests.run_pause(e),0.)
        self.assertEqual(e.clicks,2)
        self.assertIsNotNone(e._ppo_pause_pending)
        self.assertTrue(e.overlay)

    def test_menu_during_disappearance_becomes_handoff_no_second_click(self):
        e=env([('GAMEPLAY',0.)]+[('MENU',.65)]*4,['pause_success'])
        with self.assertRaises(GuardUncertainError) as x:
            runtime.BoundaryTests.run_pause(e)
        self.assertTrue(boundary_menu_candidate(x.exception))
        self.assertEqual(e.clicks,1)
        self.assertIsNotNone(e._ppo_boundary_menu_duration_at)
        self.assertIsNone(e._ppo_pause_pending)

    def test_uncertain_disappearance_saves_no_pause_token(self):
        e=env([('GAMEPLAY',0.)]+[('UNCERTAIN',0.)]*8,['pause_success'])
        with self.assertRaisesRegex(GuardUncertainError,'control unresolved'):
            runtime.BoundaryTests.run_pause(e)
        self.assertEqual(e.clicks,1)
        self.assertIsNone(e._ppo_pause_pending)

    def test_overlay_flicker_returns_without_second_click(self):
        e=env([('GAMEPLAY',0.)],['pause_success'],temporary=True)
        runtime.BoundaryTests.run_pause(e)
        self.assertEqual(e.clicks,1)
        self.assertIsNotNone(e._ppo_pause_pending)

    def test_duplicate_button_refuses_reclick(self):
        e=env([('GAMEPLAY',0.)]+[('GAMEPLAY',0.)]*8,
              ['pause_success','pause_success'],duplicate=True)
        with self.assertRaisesRegex(GuardUncertainError,'multiple site Pause game buttons'):
            runtime.BoundaryTests.run_pause(e)
        self.assertEqual(e.clicks,1)

    def test_two_prior_clicks_no_third_if_overlay_drops(self):
        # Initial Pause click times out, two PLAY frames permit one retry.
        # Overlay then vanishes in settling: cannot click a third time.
        e=env([('GAMEPLAY',0.)]+[('GAMEPLAY',0.)]*4,
              ['timeout_visible','pause_success','pause_success'],missing_at=(5,))
        with self.assertRaisesRegex(GuardUncertainError,'NO third click'):
            runtime.BoundaryTests.run_pause(e)
        self.assertEqual(e.clicks,2)

    def test_overlay_vanishes_just_after_freeze_reverify_from_scratch(self):
        # Before the final overlay check of the 4th frozen capture.
        e=env([('GAMEPLAY',0.)]+[('GAMEPLAY',0.)]*2,
              ['pause_success','pause_success'],missing_at=(10,))
        runtime.BoundaryTests.run_pause(e)
        self.assertEqual(e.clicks,2)
        self.assertIsNotNone(e._ppo_pause_pending)

    def test_overlay_flaps_twice_fail_closed(self):
        e=env([('GAMEPLAY',0.)]+[('GAMEPLAY',0.)]*8,
              ['pause_success','pause_success'],missing_at=(2,6))
        with self.assertRaisesRegex(GuardUncertainError,'NO more retries'):
            runtime.BoundaryTests.run_pause(e)
        self.assertLessEqual(e.clicks,2)

    def test_static_track_without_visible_overlay_never_verifies_freeze(self):
        e=env([('GAMEPLAY',0.)]+[('UNCERTAIN',0.)]*8,['pause_success'])
        with self.assertRaises(GuardUncertainError):
            runtime.BoundaryTests.run_pause(e)
        self.assertIsNone(e._ppo_pause_pending)

    def test_regular_verified_freeze_unchanged(self):
        e=env([('GAMEPLAY',0.)],['pause_success'],missing_at=())
        self.assertEqual(runtime.BoundaryTests.run_pause(e),0.)
        self.assertEqual(e.clicks,1)


class InstallerTests(unittest.TestCase):
    def test_only_pause_method_changes(self):
        old=ROOT/'tools'/'_ppo_pause_settle_v177_payload'/'v176_trainer.txt'
        _only_pause_changed(old.read_bytes(),PAYLOAD.read_bytes())

    def test_exact_v176_before_apply_and_reject_reinstall(self):
        import shutil
        with tempfile.TemporaryDirectory() as temp:
            root=Path(temp)
            (root/'tools').mkdir()
            for module in ('ppo_pause_control_recheck_v171.py',
                           'ppo_pause_freeze_window.py','ppo_atomic_checkpoint_v176.py'):
                shutil.copyfile(ROOT/'tools'/module,root/'tools'/module)
            before=ROOT/'tools'/'_ppo_pause_settle_v177_payload'/'v176_trainer.txt'
            (root/'train_rs_v612_ppo.py').write_bytes(before.read_bytes())
            self.assertEqual(sha((root/'train_rs_v612_ppo.py').read_bytes()),SOURCE_SHA)
            prepare(root)
            backup=apply(root)
            self.assertTrue(backup.exists())
            self.assertEqual((root/'train_rs_v612_ppo.py').read_bytes(),PAYLOAD.read_bytes())
            with self.assertRaisesRegex(RuntimeError,'Unexpected trainer SHA256'):
                apply(root)


if __name__=='__main__':unittest.main()
