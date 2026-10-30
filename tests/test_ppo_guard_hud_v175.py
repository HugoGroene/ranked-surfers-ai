"""Offline tests: actual 2026-10-28 cropped distance HUD, gate and installer.

No browser, keypress, model, or network calls.
"""
import ast
import cv2
import importlib.util
from pathlib import Path
import shutil
import sys
import tempfile
import unittest
import numpy as np

ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT))
from tools.ppo_guard_hud_v175 import meter_mask, _roi_mask, FULLSCREEN_Y, CHROME_VISIBLE_Y
from tools.patch_ppo_guard_hud_v175 import apply, prepare, sha, SOURCE_SHA, BACKUP_SUFFIX
from rs_v612_gameplay_guard import screen_gate, sequence_decision

FIXTURE=ROOT/'tests'/'fixtures'/'v174_hud_browser_crop.png'


def image_with_observed_hud():
    # Only a small crop of the uploaded JPEG is bundled; browser/person data excluded.
    x,y=194,97
    frame=np.zeros((1080,1920,4),np.uint8)
    tile=cv2.imread(str(FIXTURE),cv2.IMREAD_COLOR)
    assert tile is not None and tile.shape==(44,96,3)
    frame[y:y+44,x:x+96,:3]=tile
    frame[:,:,3]=255
    return frame


class HudTests(unittest.TestCase):
    def test_exact_case_browser_hud_recovers_but_fullscreen_only_fails(self):
        shot=image_with_observed_hud()
        self.assertEqual(cv2.countNonZero(_roi_mask(shot,FULLSCREEN_Y)),0)
        self.assertGreaterEqual(cv2.countNonZero(_roi_mask(shot,CHROME_VISIBLE_Y)),8)
        self.assertGreaterEqual(cv2.countNonZero(meter_mask(shot)),8)
        self.assertEqual(meter_mask(shot).shape,(24,64))

    def test_original_fullscreen_layout_still_works(self):
        shot=np.zeros((1080,1920,4),np.uint8)
        cv2.rectangle(shot,(211,19),(220,26),(0,255,255,255),-1)
        mask=meter_mask(shot)
        self.assertGreaterEqual(cv2.countNonZero(mask),8)
        self.assertTrue(np.array_equal(mask,_roi_mask(shot,FULLSCREEN_Y)))

    def test_no_hud_does_not_create_false_positive(self):
        self.assertEqual(cv2.countNonZero(meter_mask(np.zeros((1080,1920,4),np.uint8))),0)

    def test_wrong_number_layout_blocks(self):
        frame=np.zeros((1080,1920,4),np.uint8)
        cv2.rectangle(frame,(500,110),(522,120),(0,255,255,255),-1)
        self.assertEqual(cv2.countNonZero(meter_mask(frame)),0)

    def test_static_browser_hud_never_suffices_to_unblock_uncertain(self):
        mask=meter_mask(image_with_observed_hud())
        first=mask.copy()
        readings=[]
        for idx in range(6):
            r=screen_gate('UNCERTAIN',{'GAMEPLAY':.85,'MENU':.002},0.,
                confidence=.90,menu_threshold=.68,track_motion=0. if idx==0 else 10.,
                track_texture=32.,hud_pixels=cv2.countNonZero(mask),
                hud_changed_pixels=cv2.countNonZero(cv2.absdiff(mask,first)))
            readings.append(r)
        self.assertEqual(sequence_decision(readings,max_frames=6),'ABORT')

    def test_changing_browser_digits_and_moving_track_support_uncertain(self):
        shot=image_with_observed_hud()
        first=meter_mask(shot)
        # Shift only pixels from the digit image. No changes to the ML confidence threshold.
        shifted=shot.copy()
        shifted[97:141,194:290]=np.roll(shifted[97:141,194:290],7,axis=1)
        later=meter_mask(shifted)
        changed=cv2.countNonZero(cv2.absdiff(first,later))
        self.assertGreaterEqual(changed,8)
        readings=[]
        for idx,mask in enumerate((first,later,later)):
            readings.append(screen_gate('UNCERTAIN',{'GAMEPLAY':.85,'MENU':.002},0.,
                confidence=.90,menu_threshold=.68,track_motion=0. if idx==0 else 10.,
                track_texture=32.,hud_pixels=cv2.countNonZero(mask),
                hud_changed_pixels=cv2.countNonZero(cv2.absdiff(mask,first))))
        self.assertEqual(sequence_decision(readings,max_frames=6),'PLAY')

    def test_menu_evidence_cannot_be_overridden_by_moving_hud(self):
        readings=[screen_gate('MENU',{'GAMEPLAY':.0,'MENU':.999},.8,
            confidence=.90,menu_threshold=.68,track_motion=10.,track_texture=32.,
            hud_pixels=60,hud_changed_pixels=40) for _ in range(2)]
        self.assertEqual(sequence_decision(readings,max_frames=6),'TERMINAL')

    def test_bad_source_shape_rejected(self):
        with self.assertRaises(ValueError):
            meter_mask(np.zeros((100,100,3),np.uint8))


class InstallerTests(unittest.TestCase):
    @staticmethod
    def baseline():
        target=ROOT/'train_rs_v612_ppo.py'
        backup=target.with_name(target.name+BACKUP_SUFFIX)
        for item in (target, backup):
            if item.is_file() and sha(item.read_bytes())==SOURCE_SHA:
                return item.read_bytes()
        # Unit tests should function after installation, because the original is backed up.
        raise AssertionError('Requires exact v1.7.4 trainer or its .v175 backup')

    def setUp(self):
        self.tmp=tempfile.TemporaryDirectory()
        self.root=Path(self.tmp.name)
        (self.root/'tools').mkdir()
        (self.root/'train_rs_v612_ppo.py').write_bytes(self.baseline())
        shutil.copy2(ROOT/'tools'/'ppo_guard_hud_v175.py',self.root/'tools'/'ppo_guard_hud_v175.py')

    def tearDown(self):
        self.tmp.cleanup()

    def test_exact_v174_source_hash_and_override_only(self):
        self.assertEqual(sha(self.baseline()),SOURCE_SHA)
        target,bak,old,updated=prepare(self.root)
        self.assertFalse(bak.exists())
        self.assertIn(b'def _meter_mask(bgra):',updated)
        self.assertEqual(updated.count(b'def _meter_mask(bgra):'),1)
        self.assertEqual(old,self.baseline())

    def test_apply_creates_backup_and_refuses_second_install(self):
        backup=apply(self.root)
        self.assertEqual(backup.read_bytes(),self.baseline())
        self.assertNotEqual((self.root/'train_rs_v612_ppo.py').read_bytes(),self.baseline())
        with self.assertRaises(RuntimeError):
            apply(self.root)

    def test_unknown_source_refused_without_backup(self):
        trainer=self.root/'train_rs_v612_ppo.py'
        trainer.write_bytes(self.baseline()+b'\n# unexpected edit')
        with self.assertRaises(RuntimeError):
            prepare(self.root)
        self.assertFalse(trainer.with_name(trainer.name+BACKUP_SUFFIX).exists())

    def test_impl_is_actual_v6_override(self):
        trainer=self.root/'train_rs_v612_ppo.py'
        apply(self.root)
        tree=ast.parse(trainer.read_text(encoding='utf-8'))
        build=next(n for n in tree.body if isinstance(n,ast.FunctionDef) and n.name=='build_env_class')
        cls=next(n for n in build.body if isinstance(n,ast.ClassDef) and n.name=='PPOV61Env')
        masks=[n for n in cls.body if isinstance(n,ast.FunctionDef) and n.name=='_meter_mask']
        self.assertEqual(len(masks),1)
        self.assertTrue(any(isinstance(d,ast.Name) and d.id=='staticmethod' for d in masks[0].decorator_list))


if __name__=='__main__':
    unittest.main()
