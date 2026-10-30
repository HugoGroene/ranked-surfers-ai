import ast
import unittest
from pathlib import Path
from types import SimpleNamespace
import numpy as np
from tools.ppo_rollout_resync import replace_cached_observation, validated_vec_observation
from tools.apply_ppo_rollout_resync import patch, MARKER


class ObservationReplacementTests(unittest.TestCase):
    def setUp(self):
        self.model = SimpleNamespace(n_envs=1, _last_obs={
            'image': np.zeros((1,12,4,8), dtype=np.uint8),
            'aux': np.zeros((1,557), dtype=np.float32),
        }, _last_episode_starts=np.asarray([False]))

    def test_replace_both_fields(self):
        img = np.ones((12,4,8), np.uint8)
        aux = np.ones(557, np.float32)
        replace_cached_observation(self.model, {'image':img, 'aux':aux})
        self.assertTrue(np.array_equal(self.model._last_obs['image'][0],img))
        self.assertTrue(np.array_equal(self.model._last_obs['aux'][0],aux))
        self.assertFalse(bool(self.model._last_episode_starts[0]))
        img[:]=0
        self.assertEqual(int(self.model._last_obs['image'][0].sum()),12*4*8)

    def test_refuses_bad_images_without_partial_change(self):
        old = self.model._last_obs
        with self.assertRaises(ValueError):
            replace_cached_observation(self.model, {'image':np.zeros((12,4,7),np.uint8),
                'aux':np.zeros(557,np.float32)})
        self.assertIs(self.model._last_obs,old)

    def test_refuses_wrong_dtype(self):
        with self.assertRaises(ValueError):
            replace_cached_observation(self.model, {'image':np.zeros((12,4,8),np.float32),
                'aux':np.zeros(557,np.float32)})

    def test_refuses_nan(self):
        aux=np.zeros(557,np.float32);aux[11]=np.nan
        with self.assertRaises(ValueError):
            replace_cached_observation(self.model, {'image':np.zeros((12,4,8),np.uint8),
                'aux':aux})

    def test_refuses_multiple_envs(self):
        self.model.n_envs=2
        with self.assertRaises(ValueError):
            validated_vec_observation(self.model, {'image':np.zeros((12,4,8),np.uint8),
                'aux':np.zeros(557,np.float32)})


class PatchTests(unittest.TestCase):
    def setUp(self):
        # Representative instrumented v1 anchors, not a copy of user code.
        self.source = """def outer():
    class Environment:
        def _guard_diagnostic(self, readings, *, reason, name=None):
            return None
    return Environment

def build_callback(a, output, raw=None):
    from stable_baselines3.common.callbacks import BaseCallback
    from tools.ppo_rollout_gap_audit import RolloutGapTracker
    class C:
        def _on_rollout_start(self):
            try:
                record = self.rollout_gap_tracker.begin_rollout(1, timesteps=int(self.num_timesteps))
            except Exception as exc:
                print(f"[PPO GAP] rollout-start diagnostic failed: {exc}", flush=True)

        def _on_step(self):
            pass
    return C

def parse_args():
    p.add_argument('--rate-mode',choices=['uncapped','capped'],default='uncapped')
"""

    def test_patch_installed_instrumented_script(self):
        out = patch(self.source)
        ast.parse(out)
        self.assertIn(MARKER, out)
        self.assertIn('--resync-rollouts', out)
        self.assertEqual(out.count('refresh_after_ppo_optimizer(self)'), 1)

    def test_patch_rejects_double_install(self):
        out = patch(self.source)
        with self.assertRaises(ValueError):
            patch(out)

    def test_patch_rejects_uninstrumented_source(self):
        with self.assertRaises(ValueError):
            patch(self.source.replace('from tools.ppo_rollout_gap_audit import RolloutGapTracker',''))


if __name__=='__main__':
    unittest.main()
