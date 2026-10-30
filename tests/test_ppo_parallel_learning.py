import tempfile
import unittest
from pathlib import Path

import gymnasium as gym
import numpy as np
from stable_baselines3 import PPO

from tools.ppo_parallel_learning import (
    BUFFER_FIELDS, configure_learner, install_policy, merge_rollouts,
    policy_arrays, policy_hash, rollout_packet)
from tools.ppo_settled_learning import TimedDictBuffer


class SmallEnv(gym.Env):
    observation_space = gym.spaces.Dict({
        'image': gym.spaces.Box(-1000, 1000, (2,), dtype=np.float32),
        'aux': gym.spaces.Box(-1000, 1000, (1,), dtype=np.float32)})
    action_space = gym.spaces.Discrete(5)
    def reset(self, *, seed=None, options=None):
        return {'image': np.zeros(2, np.float32), 'aux': np.zeros(1, np.float32)}, {}


def local_packet(index, mask=1.):
    env = SmallEnv()
    buffer = TimedDictBuffer(16, env.observation_space, env.action_space, n_envs=1, device='cpu')
    for value in buffer.observations.values():
        value[:] = index
    for key in BUFFER_FIELDS:
        getattr(buffer, key)[:] = 0
    buffer.returns[:] = index+1.
    buffer.advantages[:] = index+3.
    buffer.active_seconds[:] = .05
    buffer.actor_masks[:] = mask
    buffer.full = True
    return rollout_packet(buffer, index, 7, 'policy')


class ParallelBufferTests(unittest.TestCase):
    def buffer(self):
        env = SmallEnv()
        return TimedDictBuffer(32, env.observation_space, env.action_space, device='cpu', n_envs=1)

    def test_merge_preserves_independent_returns_without_recomputing_gae(self):
        buffer = self.buffer()
        merge_rollouts(buffer, [local_packet(1), local_packet(0)], 16, 7, 'policy')
        np.testing.assert_array_equal(buffer.returns[:16], np.ones((16, 1)))
        np.testing.assert_array_equal(buffer.returns[16:], np.full((16, 1), 2.))
        np.testing.assert_array_equal(buffer.advantages[:16], np.full((16, 1), 3.))
        self.assertTrue(buffer.full)
        self.assertFalse(buffer.generator_ready)

    def test_mixed_versions_duplicate_workers_and_invalid_masks_rejected(self):
        for change in ('version', 'duplicate', 'mask', 'nan', 'shape'):
            packets = [local_packet(0), local_packet(1)]
            if change == 'version': packets[1]['version'] = 8
            if change == 'duplicate': packets[1]['worker'] = 0
            if change == 'mask': packets[1]['actor_masks'][:] = .5
            if change == 'nan': packets[1]['values'][0] = np.nan
            if change == 'shape': packets[1]['observations']['aux'] = np.zeros((15, 1, 1), np.float32)
            with self.subTest(change=change), self.assertRaises(ValueError):
                merge_rollouts(self.buffer(), packets, 16, 7, 'policy')

    def test_actual_shared_optimizer_masks_passive_actor_and_keeps_saved_worker_steps(self):
        model = PPO('MultiInputPolicy', SmallEnv(), n_steps=16, batch_size=8, n_epochs=1,
                    learning_rate=.0003, device='cpu', seed=3)
        original = {key: value.detach().clone() for key, value in model.policy.state_dict().items()}
        with tempfile.TemporaryDirectory() as folder:
            configure_learner(model, Path(folder), 2, .05)
            try:
                merge_rollouts(model.rollout_buffer, [local_packet(0, 0.), local_packet(1, 0.)], 16, 7, 'policy')
                model.num_timesteps = 32
                model.train()
                self.assertEqual(model.n_steps, 16)
                self.assertEqual(model._settled_stats['completed_rollout_transitions'], 32)
                self.assertEqual(len(model.parallel_used_indices), 32)
                actor = ['mlp_extractor.policy_net', 'action_net']
                for key, value in model.policy.state_dict().items():
                    if any(key.startswith(prefix) for prefix in actor):
                        self.assertTrue(value.equal(original[key]), key)
                self.assertFalse(model.policy.value_net.weight.equal(original['value_net.weight']))
            finally:
                model.logger.close()

    def test_policy_hash_is_checked_before_loading_any_weights(self):
        model = PPO('MultiInputPolicy', SmallEnv(), n_steps=16, batch_size=8, n_epochs=1, device='cpu')
        arrays = policy_arrays(model.policy)
        digest = policy_hash(arrays)
        install_policy(model.policy, arrays, digest)
        next(iter(arrays.values())).flat[0] += 1
        with self.assertRaises(ValueError):
            install_policy(model.policy, arrays, digest)


class PreviewTests(unittest.TestCase):
    def test_preview_uses_existing_policy_frame_and_preserves_input(self):
        from tools.ppo_worker_preview import write_preview
        image = np.full((12, 288, 512), 42, np.uint8)
        before = image.copy()
        with tempfile.TemporaryDirectory() as folder:
            self.assertTrue(write_preview(folder, image, dict(phase='collecting', steps=12)))
            self.assertTrue((Path(folder)/'live.jpg').exists())
            self.assertTrue((Path(folder)/'live.json').exists())
        np.testing.assert_array_equal(image, before)


if __name__ == '__main__':
    unittest.main()
