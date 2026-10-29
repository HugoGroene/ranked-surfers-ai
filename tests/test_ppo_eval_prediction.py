import unittest
from unittest.mock import patch

import gymnasium as gym
import numpy as np
import torch
from stable_baselines3 import PPO

from tools.ppo_eval_prediction import predict_with_optional_probabilities


class EvaluationPredictionTests(unittest.TestCase):
    def setUp(self):
        torch.set_num_threads(1)
        self.env = gym.make('CartPole-v1')
        self.model = PPO('MlpPolicy', self.env, n_steps=16, batch_size=8, device='cpu', seed=123)
        self.obs, _ = self.env.reset(seed=123)

    def tearDown(self):
        self.env.close()

    def test_same_action_and_probabilities_with_exactly_one_extractor_forward(self):
        reference, _ = self.model.predict(self.obs, deterministic=True)
        calls = []
        hook = self.model.policy.features_extractor.register_forward_hook(lambda *args: calls.append(1))
        try:
            action, statistics = predict_with_optional_probabilities(self.model, self.obs,
                deterministic=True, sample_probabilities=True)
        finally:
            hook.remove()
        np.testing.assert_array_equal(action, reference)
        self.assertEqual(len(calls), 1)
        with torch.no_grad():
            obs, _ = self.model.policy.obs_to_tensor(self.obs)
            expected = self.model.policy.get_distribution(obs).distribution.probs.cpu().numpy()[0]
        np.testing.assert_allclose(statistics['probabilities'], expected, atol=0.000006, rtol=0)
        self.assertNotIn('get_distribution', self.model.policy.__dict__)

    def test_stochastic_rng_and_action_are_preserved(self):
        torch.manual_seed(99)
        reference, _ = self.model.predict(self.obs, deterministic=False)
        expected_rng = torch.get_rng_state()
        torch.manual_seed(99)
        action, _ = predict_with_optional_probabilities(self.model, self.obs,
            deterministic=False, sample_probabilities=True)
        np.testing.assert_array_equal(action, reference)
        torch.testing.assert_close(torch.get_rng_state(), expected_rng, rtol=0, atol=0)

    def test_prediction_failure_restores_original_method(self):
        with patch.object(self.model, 'predict', side_effect=RuntimeError('injected')):
            with self.assertRaises(RuntimeError):
                predict_with_optional_probabilities(self.model, self.obs,
                    deterministic=True, sample_probabilities=True)
        self.assertNotIn('get_distribution', self.model.policy.__dict__)
        self.model.predict(self.obs, deterministic=True)
