"""Offline tests for observation-only Unity pause/HUD/clock probe."""
import ast
import importlib.util
from pathlib import Path
import unittest

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
PATH = ROOT / 'tools/probe_unity_pause_clock.py'
spec = importlib.util.spec_from_file_location('ppo_pause_clock_probe', PATH)
probe = importlib.util.module_from_spec(spec)
spec.loader.exec_module(probe)


class MeterTests(unittest.TestCase):
    def test_compiles(self):
        ast.parse(PATH.read_text(encoding='utf-8'))

    def test_meter_detects_change_without_uint8_wrap(self):
        a = np.zeros((3, 8), dtype=np.uint8)
        b = np.zeros((3, 8), dtype=np.uint8)
        a[0, :6] = 255
        b[1, :6] = 255
        d = probe.meter_activity(a, b)
        self.assertEqual(d['yellow_pixels_first'], 6)
        self.assertEqual(d['yellow_pixels_second'], 6)
        self.assertEqual(d['changed_pixels'], 12)

    def test_meter_rejects_mismatched_shapes(self):
        with self.assertRaises(ValueError):
            probe.meter_activity(np.zeros((4, 5)), np.zeros((4, 6)))

    def test_meter_rejects_empty(self):
        with self.assertRaises(ValueError):
            probe.meter_activity(np.zeros((0, 4)), np.zeros((0, 4)))

    def test_meter_rejects_not_2d(self):
        with self.assertRaises(ValueError):
            probe.meter_activity(np.zeros((4, 4, 3)), np.zeros((4, 4, 3)))

    def test_hud_candidate_with_motion_freeze(self):
        moving = {'yellow_pixels_first': 20, 'yellow_pixels_second': 20, 'changed_pixels': 7}
        paused = {'yellow_pixels_first': 20, 'yellow_pixels_second': 20, 'changed_pixels': 0}
        self.assertTrue(probe.hud_freeze_support(moving, paused, moving))

    def test_hud_static_before_is_not_success(self):
        moving = {'yellow_pixels_first': 20, 'yellow_pixels_second': 20, 'changed_pixels': 7}
        paused = {'yellow_pixels_first': 20, 'yellow_pixels_second': 20, 'changed_pixels': 0}
        self.assertFalse(probe.hud_freeze_support(paused, paused, moving))

    def test_hud_static_after_is_not_success(self):
        moving = {'yellow_pixels_first': 20, 'yellow_pixels_second': 20, 'changed_pixels': 7}
        paused = {'yellow_pixels_first': 20, 'yellow_pixels_second': 20, 'changed_pixels': 0}
        self.assertFalse(probe.hud_freeze_support(moving, paused, paused))

    def test_hud_motion_during_pause_fails(self):
        moving = {'yellow_pixels_first': 20, 'yellow_pixels_second': 20, 'changed_pixels': 7}
        self.assertFalse(probe.hud_freeze_support(moving, moving, moving))

    def test_hud_missing_cannot_prove_freeze(self):
        moving = {'yellow_pixels_first': 0, 'yellow_pixels_second': 0, 'changed_pixels': 0}
        self.assertFalse(probe.hud_freeze_support(moving, moving, moving))

    def test_tracking_freeze_needs_running_motion(self):
        self.assertFalse(probe.pause_freeze_support(
            {'mean_absolute_pixel_change': 0}, {'mean_absolute_pixel_change': 0}))

    def test_scoped_resume_selector_and_no_force(self):
        code = PATH.read_text(encoding='utf-8')
        self.assertIn("overlay.get_by_role('button', name='Resume', exact=True)", code)
        self.assertNotIn('force=True', code.split('def main():')[1])

    def test_no_trainer_checkpoint_policy_updates(self):
        code = PATH.read_text(encoding='utf-8')
        self.assertNotIn('model.learn(', code)
        self.assertNotIn('model.save(', code)
        self.assertNotIn('PPO.load(', code)
        self.assertNotIn('trainer.train(', code)
        self.assertIn('trainer.create_live_env(a, output)', code)
        self.assertIn('monitored.reset()', code)
        self.assertNotIn('monitored.step(', code)

    def test_status_rejects_observed_hud_motion(self):
        code = PATH.read_text(encoding='utf-8')
        self.assertIn("'REJECT_HUD_MOVED_DURING_PAUSE'", code)
        self.assertIn("hud_paused['changed_pixels'] > 0", code)


if __name__ == '__main__':
    unittest.main()
