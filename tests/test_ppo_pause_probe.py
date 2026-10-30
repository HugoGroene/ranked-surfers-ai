"""Offline tests; never open Unity or start PPO."""
import unittest

import numpy as np

from tools.probe_unity_pause import frame_motion, pause_freeze_support


class PauseProbeOfflineTests(unittest.TestCase):
    def test_exactly_identical_images_have_zero_change(self):
        a = np.zeros((20, 30), dtype=np.uint8)
        r = frame_motion(a, a.copy())
        self.assertEqual(r['mean_absolute_pixel_change'], 0)

    def test_animated_track_has_measurable_change(self):
        a = np.zeros((20, 30), dtype=np.uint8)
        b = a.copy()
        b[:, :15] = 60
        r = frame_motion(a, b)
        self.assertEqual(r['mean_absolute_pixel_change'], 30)
        self.assertAlmostEqual(r['pixel_fraction_changed_gt_8'], .5)

    def test_changed_dtype_does_not_wrap_subtraction(self):
        a = np.full((20, 30), 255, dtype=np.uint8)
        b = np.zeros_like(a)
        self.assertEqual(frame_motion(a,b)['mean_absolute_pixel_change'],255)

    def test_mismatched_dimensions_rejected(self):
        with self.assertRaises(ValueError):
            frame_motion(np.zeros((2,2)),np.zeros((3,3)))

    def test_static_before_is_not_evidence_of_success(self):
        before={'mean_absolute_pixel_change':0}
        paused={'mean_absolute_pixel_change':0}
        self.assertFalse(pause_freeze_support(before,paused))

    def test_large_paused_change_rejected(self):
        before={'mean_absolute_pixel_change':14}
        paused={'mean_absolute_pixel_change':3}
        self.assertFalse(pause_freeze_support(before,paused))

    def test_strong_freeze_accepted_as_candidate(self):
        before={'mean_absolute_pixel_change':14}
        paused={'mean_absolute_pixel_change':0.1}
        self.assertTrue(pause_freeze_support(before,paused))

    def test_compile_probe(self):
        import pathlib
        source=pathlib.Path(__file__).parents[1]/'tools'/'probe_unity_pause.py'
        compile(source.read_text(encoding='utf-8'),str(source),'exec')


if __name__ == '__main__':
    unittest.main()
