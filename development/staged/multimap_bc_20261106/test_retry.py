import json
from pathlib import Path
import tempfile
from types import SimpleNamespace
import unittest
from unittest.mock import patch
import numpy as np
from retry_data import BalancedSampler, BoundedDataset, candidate_gate
from train_balanced import evaluate, fit


class TinyData:
    def __init__(self, split):
        self.samples = [(0, i, 'halloween' if i < 5 else 'tokyo', split+str(i//5)) for i in range(10)]
        self.arrays = [dict(actions=np.arange(10)%5)]
        self.max_frame_maps = 2

    def sample(self, k):
        aux = np.zeros(789, np.float32); aux[k%5] = 1
        return dict(image=np.zeros((12, 16, 16), np.uint8), aux=aux), k%5

    def drop_frames(self):
        pass


def good_metrics(loss=2.):
    return dict(selection_loss=loss, maps={name:dict(macro_cross_entropy=loss,
        per_action_recall=[.5]*5, mean_movement_recall=.5, prediction_counts=[4, 2, 2, 2, 2],
        max_probability_range=.2) for name in ('halloween', 'tokyo')})


class RetryTests(unittest.TestCase):
    def test_exact_sampling_per_map_class(self):
        data = TinyData('train'); sampler = BalancedSampler(data)
        rng = np.random.default_rng(19)
        for size in (16, 32, 64):
            ids = sampler.batch(rng, size)
            for name in ('halloween', 'tokyo'):
                labels = [data.sample(i)[1] for i in ids if data.samples[i][2] == name]
                np.testing.assert_array_equal(np.bincount(labels, minlength=5), np.array([4, 1, 1, 1, 1])*(size//16))

    def test_sampling_reproducible(self):
        sampler = BalancedSampler(TinyData('train'))
        self.assertEqual(sampler.batch(np.random.default_rng(5)), sampler.batch(np.random.default_rng(5)))
        with self.assertRaises(ValueError): sampler.batch(np.random.default_rng(), 15)

    def test_missing_movement_group_rejected(self):
        data = TinyData('train'); data.arrays[0]['actions'][4] = 0
        with self.assertRaises(ValueError): BalancedSampler(data)

    def test_none_collapse_rejected(self):
        metrics = good_metrics()
        for m in metrics['maps'].values():
            m.update(per_action_recall=[1., 0., 0., 0., 0.], prediction_counts=[100, 0, 0, 0, 0], max_probability_range=0.)
        passed, reasons = candidate_gate(metrics)
        self.assertFalse(passed)
        self.assertTrue(any('movement' in r for r in reasons))
        self.assertTrue(any('constant' in r for r in reasons))

    def test_constant_movement_rejected(self):
        metrics = good_metrics()
        metrics['maps']['tokyo'].update(per_action_recall=[0., 0., 0., 1., 0.], prediction_counts=[0, 0, 0, 100, 0])
        self.assertFalse(candidate_gate(metrics)[0])
        self.assertTrue(candidate_gate(good_metrics())[0])

    def test_validation_metrics_preserve_all_labels(self):
        import torch
        torch.set_num_threads(1)
        class Policy:
            device = 'cpu'
            def set_training_mode(self, mode): pass
            def get_distribution(self, obs):
                dist = torch.distributions.Categorical(logits=obs['aux'][:, :5]*4)
                return SimpleNamespace(distribution=dist, log_prob=dist.log_prob)
        metrics = evaluate(Policy(), TinyData('validation'), 4)
        for m in metrics['maps'].values():
            self.assertEqual(m['per_action_counts'], [1]*5)
            self.assertEqual(m['per_action_recall'], [1.]*5)
        self.assertTrue(candidate_gate(metrics)[0])

    def test_frame_mappings_evicted_without_mutating_samples(self):
        from common import digest
        with tempfile.TemporaryDirectory() as tmp:
            folder = Path(tmp); runs = []
            for i in range(3):
                p = folder/f'run_{i}'; p.mkdir()
                np.save(p/'frames.npy', np.full((2, 3, 16, 16), i+1, np.uint8))
                np.save(p/'indices.npy', np.array([[-1, -1, 0, 1]], np.int64))
                np.save(p/'aux.npy', np.zeros((1, 789), np.float32))
                np.save(p/'actions.npy', np.array([i], np.int64))
                (p/'samples.json').write_text('[]')
                runs.append(dict(path=p.name, split='train', count=1, map='halloween', session='train',
                    hashes={name:digest(p/name) for name in ('frames.npy','indices.npy','aux.npy','actions.npy','samples.json')}))
            (folder/'dataset.json').write_text(json.dumps(dict(status='COMPLETE',schema='multimap-bc-fixed4-v1',
                contract=dict(image_shape=[12,16,16]), runs=runs)))
            data = BoundedDataset(folder, 'train')
            try:
                first, _ = data.sample(0); saved = first['image'].copy()
                data.sample(1); data.sample(2)
                self.assertNotIn(0, data.frame_maps)
                self.assertEqual(data.peak_frame_mappings, 2)
                np.testing.assert_array_equal(first['image'], saved)
                np.testing.assert_array_equal(data.sample(0)[0]['image'], saved)
                data.drop_frames(); self.assertFalse(data.frame_maps)
            finally: data.close()

    def test_fit_updates_encoder_freezes_critic_and_saves_separate_candidate(self):
        import torch
        import gymnasium as gym
        from stable_baselines3 import PPO
        from stable_baselines3.common.torch_layers import BaseFeaturesExtractor
        torch.set_num_threads(1)
        class Features(BaseFeaturesExtractor):
            def __init__(self, space):
                super().__init__(space, 8); self.aux = torch.nn.Linear(789, 8)
            def forward(self, obs): return torch.tanh(self.aux(obs['aux']))
        class Env(gym.Env):
            observation_space = gym.spaces.Dict(image=gym.spaces.Box(0,255,(12,16,16),np.uint8),
                                                aux=gym.spaces.Box(0,1,(789,),np.float32))
            action_space = gym.spaces.Discrete(5)
        model = PPO('MultiInputPolicy', Env(), n_steps=8, batch_size=4, device='cpu',
                    policy_kwargs=dict(features_extractor_class=Features, net_arch=dict(pi=[8], vf=[8])))
        old_features = {k:v.clone() for k,v in model.policy.features_extractor.state_dict().items()}
        old_value = {k:v.clone() for k,v in model.policy.value_net.state_dict().items()}
        with tempfile.TemporaryDirectory() as tmp:
            output = Path(tmp)
            with patch('train_balanced.evaluate', side_effect=[good_metrics(3.), good_metrics(2.)]):
                status = fit(model, TinyData('train'), TinyData('val'), output,
                    updates=2, validate_every=2, actor_lr=.001, encoder_lr=.0001, use_mirror=False)
            self.assertEqual(status, 'CANDIDATE_SAVED')
            self.assertTrue((output/'bc_best.zip').is_file())
            saved = PPO.load(output/'bc_best.zip', device='cpu')
            self.assertTrue(any(not torch.equal(v, saved.policy.features_extractor.state_dict()[k]) for k,v in old_features.items()))
            for k,v in old_value.items(): self.assertTrue(torch.equal(v,saved.policy.value_net.state_dict()[k]))
            self.assertFalse(saved.policy.optimizer.state)
            self.assertTrue(all(p.requires_grad for p in saved.policy.parameters()))


if __name__ == '__main__':
    unittest.main()
