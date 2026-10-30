"""Tests on frozen-scene verification, conservative pause time and installation."""
import unittest
import numpy as np
from tools.ppo_pause_freeze_window import FreezeWindow
from tools.patch_ppo_optimizer_pause_settle_v151 import patch, NEW, OLD
from tools.install_ppo_optimizer_pause import patch as patch_v15
from test_ppo_optimizer_pause_v15 import SYNTHETIC


def frame(value):
    return np.full((24,32), value, dtype=np.uint8)


class FreezeWindowTests(unittest.TestCase):
    def test_early_overlay_does_not_count_as_freeze(self):
        w=FreezeWindow(required_pairs=3,max_delta=.25)
        vals=[0,50,100,150,190,190,190,190]
        results=[w.observe(frame(v),(i*.1,i*.1+.01,i*.1+.02)) for i,v in enumerate(vals)]
        self.assertTrue(all(x is None for x in results[:-1]))
        self.assertIsNotNone(results[-1])
        self.assertAlmostEqual(results[-1][0], .4)  # First verified stationary capture, not overlay
        self.assertEqual(results[-1][3],0.)

    def test_moving_forever_never_passes(self):
        w=FreezeWindow()
        out=[w.observe(frame(i*10),(i+1.,i+1.,i+1.)) for i in range(16)]
        self.assertTrue(all(x is None for x in out))

    def test_transient_one_stable_pair_not_enough(self):
        w=FreezeWindow()
        out=[w.observe(frame(v),(i+1.,i+1.,i+1.)) for i,v in enumerate([0,0,10,10,10])]
        self.assertTrue(all(x is None for x in out))

    def test_three_frozen_pairs_need_four_captures(self):
        w=FreezeWindow()
        for i in range(3):self.assertIsNone(w.observe(frame(80),(i+1.,i+1.,i+1.)))
        r=w.observe(frame(80),(4.,4.,4.))
        self.assertEqual(r[:3],(1.,1.,1.))

    def test_restarts_after_motion(self):
        w=FreezeWindow()
        values=[1,1,1,2,2,2,2]
        out=[w.observe(frame(v),(i+1.,i+1.,i+1.)) for i,v in enumerate(values)]
        self.assertIsNotNone(out[-1])
        self.assertEqual(out[-1][0],4.)

    def test_rejects_bad_shapes_clocks_and_dtype(self):
        for im,tm in [(np.ones((3,3),dtype=np.float32),(1.,1.,1.)),
                      (np.zeros((1,),dtype=np.uint8),(1.,1.,1.)),
                      (frame(1),(1.,float('nan'),1.)),
                      (frame(1),(1.,2.)),
                      (frame(1),(0.,0.,0.))]:
            with self.subTest(im=im.shape,tm=tm):
                w=FreezeWindow()
                if tm == (0.,0.,0.):
                    w.observe(im, tm)
                    with self.assertRaises(ValueError):w.observe(im, tm)
                else:
                    with self.assertRaises(ValueError):w.observe(im, tm)

    def test_rejects_changed_shape(self):
        w=FreezeWindow()
        w.observe(frame(0),(1.,1.,1.))
        with self.assertRaises(ValueError):w.observe(np.zeros((30,30),dtype=np.uint8),(2.,2.,2.))


class InstallerTests(unittest.TestCase):
    def test_only_exact_v15_patch(self):
        v15=patch_v15(SYNTHETIC)
        revised=patch(v15)
        self.assertEqual(revised.replace(NEW,OLD),v15)
        self.assertIn('FreezeWindow(required_pairs=3',revised)
        compile(revised,'train_rs_v612_ppo.py','exec')

    def test_double_install_rejected(self):
        with self.assertRaises(ValueError):patch(patch(patch_v15(SYNTHETIC)))

    def test_without_original_rejected(self):
        with self.assertRaises(ValueError):patch(SYNTHETIC)

    def test_pause_starts_after_verified_stable_frames(self):
        src=patch(patch_v15(SYNTHETIC))
        seg=src[src.index('def pause_for_ppo_optimizer(self):'):src.index('def resume_after_ppo_optimizer(self, monitor):')]
        self.assertLess(seg.index('await pause_button.click'),seg.index('FreezeWindow(required_pairs=3'))
        self.assertLess(seg.index('verified = window.observe'),seg.index('start_mono, start_perf, start_wall, mean_change = verified'))
        self.assertIn('self._read_screen_state()',seg)
        self.assertIn('if verified is None:',seg)
        self.assertIn('raise GuardUncertainError',seg)
        self.assertNotIn('force=True',seg)


if __name__=='__main__':unittest.main()
