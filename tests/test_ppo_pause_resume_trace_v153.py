"""Read-only trace patch regression tests."""
import unittest
from tools.trace_ppo_pause_resume_v153 import (
    patch, RESYNC_OLD, RESYNC_NEW, RESUME_OLD, RESUME_NEW,
    PRE_OLD, PRE_NEW, MARKER,
)

# A minimal syntactically valid trainer with the three exact v1.5.2 anchors.
SYNTHETIC = '''import time
class GuardUncertainError(RuntimeError): pass
class Dummy:
    def pause_for_ppo_optimizer(self):
        async def pause():
                state = 'GAMEPLAY'; probs = {}; menu_score = 0.0
                pause_button = None
''' + PRE_OLD + '''                    return None
        return pause()
    def resume_after_ppo_optimizer(self, monitor):
        if True:
            async def resume(): pass
            token = (0., 0.)
            self._call = lambda *a, **k: (1.,1.,1.)
''' + RESUME_OLD + '''            return dm
    def refresh_after_ppo_optimizer(self):
        async def capture_fresh_stack():
            for i in range(4):
''' + RESYNC_OLD + '''            return None
        return capture_fresh_stack()
    def compensate(self):
        compensate_pause_clocks(None)
# Existing flags and functions
# --pause-ppo-optimizer
# FreezeWindow(required_pairs=3, max_delta=.25)
# MENU template detected during pause
# refresh_after_ppo_optimizer()
# replace_cached_observation
'''


class PatchTests(unittest.TestCase):
    def test_inserts_only_three_expected_blocks(self):
        revised = patch(SYNTHETIC)
        self.assertEqual(revised.replace(RESYNC_NEW, RESYNC_OLD)
                                .replace(RESUME_NEW, RESUME_OLD)
                                .replace(PRE_NEW, PRE_OLD), SYNTHETIC)

    def test_code_compiles(self):
        compile(patch(SYNTHETIC), 'mock_train_rs_v612_ppo.py', 'exec')

    def test_refuses_double_install(self):
        with self.assertRaises(ValueError):
            patch(patch(SYNTHETIC))

    def test_refuses_missing_integration(self):
        for marker in ('--pause-ppo-optimizer', 'MENU template detected during pause',
                       'replace_cached_observation'):
            with self.subTest(marker=marker), self.assertRaises(ValueError):
                patch(SYNTHETIC.replace(marker, 'missing'))

    def test_refuses_changed_capture_anchor(self):
        with self.assertRaises(ValueError):
            patch(SYNTHETIC.replace('state, _probs = self._read_screen_state()',
                                    'state = "GAMEPLAY"'))

    def test_trace_uses_no_extra_capture(self):
        revised = patch(SYNTHETIC)
        self.assertEqual(revised.count('await self._capture()'),
                         SYNTHETIC.count('await self._capture()'))
        self.assertIn('frame={i+1}/4', revised)
        self.assertIn('since_resume_ms', revised)

    def test_keeps_exact_reject_condition(self):
        s = patch(SYNTHETIC)
        self.assertEqual(s.count("if state != 'GAMEPLAY' or menu_score >= self.args.threshold * .85:"), 2)
        self.assertIn('return None, (f\'unsafe screen state={state}', s)

    def test_no_action_reward_or_restart_changes(self):
        s = patch(SYNTHETIC)
        for banned in ('page.keyboard', 'recover_rollout_menu',
                       'survival_rewards.step', 'model.save', 'model._last_obs ='):
            self.assertNotIn(banned, s)
        self.assertNotIn('force=True', s)

    def test_uses_overlay_disappearance_timestamp(self):
        s = patch(SYNTHETIC)
        self.assertIn('_ppo_resume_trace_start_perf = finished_perf', s)
        self.assertIn('dm = finished_mono - token[0]', s)


if __name__ == '__main__':
    unittest.main()
