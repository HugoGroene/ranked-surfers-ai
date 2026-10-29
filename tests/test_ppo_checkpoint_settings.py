import hashlib
from pathlib import Path
import tempfile
from types import SimpleNamespace
import unittest

import gymnasium as gym
import numpy as np
import torch
from stable_baselines3 import PPO

from tools.ppo_checkpoint_settings import load_training_checkpoint


class Toy(gym.Env):
    observation_space = gym.spaces.Box(-1., 1., (4,), dtype=np.float32)
    action_space = gym.spaces.Discrete(2)

    def reset(self, *, seed=None, options=None):
        self.t = 0
        return np.zeros(4, dtype=np.float32), {}

    def step(self, action):
        self.t += 1
        return np.full(4, self.t/10, dtype=np.float32), float(action), self.t == 8, False, {}


class CheckpointSettingsTests(unittest.TestCase):
    def test_override_rebuilds_buffer_without_changing_weights_or_source(self):
        torch.set_num_threads(1)
        with tempfile.TemporaryDirectory() as temp:
            source = PPO('MlpPolicy', Toy(), n_steps=16, batch_size=8, n_epochs=1,
                         gamma=.995, learning_rate=.0003, device='cpu', seed=123)
            path = Path(temp)/'source.zip'
            source.save(path)
            digest = hashlib.sha256(path.read_bytes()).hexdigest()
            args = SimpleNamespace(resume=path, steps=32, n_steps=32, gamma=.99,
                target_kl=.03, learning_rate=.0001,
                override_ppo_settings=['n_steps', 'gamma', 'target_kl', 'learning_rate'])
            model, metadata = load_training_checkpoint(args, Toy(), device='cpu')
            self.assertEqual(model.n_steps, 32)
            self.assertEqual(model.rollout_buffer.buffer_size, 32)
            self.assertEqual(model.rollout_buffer.gamma, .99)
            self.assertEqual(model.target_kl, .03)
            for key, value in source.policy.state_dict().items():
                torch.testing.assert_close(value, model.policy.state_dict()[key], rtol=0, atol=0)
            model.learn(32)
            self.assertEqual(model.policy.optimizer.param_groups[0]['lr'], .0001)
            self.assertEqual(hashlib.sha256(path.read_bytes()).hexdigest(), digest)
            self.assertEqual(metadata['effective_settings']['learning_rate_at_start'], .0001)

    def test_no_override_preserves_saved_settings(self):
        with tempfile.TemporaryDirectory() as temp:
            source = PPO('MlpPolicy', Toy(), n_steps=16, batch_size=8, n_epochs=1, gamma=.97, device='cpu')
            path = Path(temp)/'source.zip'
            source.save(path)
            args = SimpleNamespace(resume=path, steps=32, n_steps=32, gamma=.99, override_ppo_settings=[])
            model, _ = load_training_checkpoint(args, Toy(), device='cpu')
            self.assertEqual(model.n_steps, 16)
            self.assertEqual(model.gamma, .97)

    def test_effective_rollout_must_divide_requested_step_budget(self):
        with tempfile.TemporaryDirectory() as temp:
            source = PPO('MlpPolicy', Toy(), n_steps=16, batch_size=8, n_epochs=1, device='cpu')
            path = Path(temp)/'source.zip'
            source.save(path)
            with self.assertRaisesRegex(ValueError, 'not divisible'):
                load_training_checkpoint(SimpleNamespace(resume=path, steps=24), Toy(), device='cpu')
