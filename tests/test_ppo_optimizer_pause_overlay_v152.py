"""Offline regression: do not demand unoccluded ML GAMEPLAY behind pause overlay."""
import unittest
from tools.patch_ppo_optimizer_pause_overlay_v152 import OLD, NEW, patch
from tools.install_ppo_optimizer_pause import patch as install_v15
from tools.patch_ppo_optimizer_pause_settle_v151 import patch as install_v151
from test_ppo_optimizer_pause_v15 import SYNTHETIC
from tools.ppo_pause_freeze_window import FreezeWindow
import numpy as np


def source():
    # Synthetic trainer already has resync call; add import as anchor only.
    # Real --check will validate the full locally installed training script.
    return install_v151(install_v15(SYNTHETIC + "\nfrom tools.ppo_rollout_resync import replace_cached_observation\n"))


class PauseOverlayRegression(unittest.TestCase):
    def test_exact_installed_v151_is_accepted(self):
        modified = patch(source())
        self.assertIn(NEW, modified)
        self.assertEqual(modified.replace(NEW, OLD), source())

    def test_compiles(self):
        compile(patch(source()), 'train_rs_v612_ppo.py', 'exec')

    def test_refuses_base_script(self):
        with self.assertRaises(ValueError):
            patch(SYNTHETIC)

    def test_refuses_double_install(self):
        with self.assertRaises(ValueError):
            patch(patch(source()))

    def test_gameplay_gate_before_pause_is_retained(self):
        s = patch(source())
        m = s.split('def pause_for_ppo_optimizer(self):')[1].split('def resume_after_ppo_optimizer(self, monitor):')[0]
        self.assertIn("if state != 'GAMEPLAY' or menu_score >= self.args.threshold * .85:", m)
        self.assertLess(m.index("if state != 'GAMEPLAY'"), m.index('await pause_button.click'))

    def test_paused_classifier_check_is_removed(self):
        s = patch(source())
        m = s.split('def pause_for_ppo_optimizer(self):')[1].split('def resume_after_ppo_optimizer(self, monitor):')[0]
        self.assertNotIn('Not verified GAMEPLAY during pause', m)
        self.assertNotIn('state, _probs = self._read_screen_state()', m)
        self.assertIn('MENU template detected during pause', m)

    def test_pause_overlay_must_stay_visible(self):
        m = patch(source()).split('def pause_for_ppo_optimizer(self):')[1].split('def resume_after_ppo_optimizer(self, monitor):')[0]
        self.assertIn("if not await overlay.is_visible():", m)
        self.assertIn("await overlay.wait_for(state='visible'", m)
        self.assertIn('if verified is None:', m)

    def test_post_resume_classifier_based_refresh_retained(self):
        s=patch(source())
        self.assertIn('refresh_after_ppo_optimizer()',s)
        self.assertIn('replace_cached_observation',s)
        self.assertIn('Safe stop',s)

    def test_all_changes_limited_to_one_exact_block(self):
        old=source()
        new=patch(old)
        self.assertEqual(new.replace(NEW,OLD),old)

    def test_refuse_changed_anchor(self):
        s=source().replace("if state != 'GAMEPLAY' or menu_score >= self.args.threshold * .85:", 'if True:')
        with self.assertRaises(ValueError):patch(s)

    def test_freeze_window_requires_three_stable_pairs(self):
        w=FreezeWindow(required_pairs=3,max_delta=.25)
        frames=[10,15,20,20,20,20]
        results=[w.observe(np.full((12,12),x,dtype=np.uint8), (i+1.,i+1.,i+1.)) for i,x in enumerate(frames)]
        self.assertTrue(all(x is None for x in results[:-1]))
        self.assertIsNotNone(results[-1])

    def test_runtime_logic_does_not_change_other_flags(self):
        m=patch(source())
        self.assertIn("p.add_argument('--pause-ppo-optimizer'", m)
        self.assertIn('compensate_pause_clocks(',m)
        self.assertIn('active_episode_clock_drift_s', m)


if __name__=='__main__':
    unittest.main()
