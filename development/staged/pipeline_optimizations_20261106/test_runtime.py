"""Run via check.py only, once BC has stopped. Tiny synthetic CPU models."""
import copy
from collections import deque
from types import SimpleNamespace
import unittest
import numpy as np
import torch
from gymnasium import spaces
from tools.ppo_settled_learning import SettledPPO, TimedDictBuffer
from optimizer import deferred_train
from transfers import UpdateTransfers
from observation import latest_only

CUDA = False
torch.set_num_threads(1)


class Policy(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.features = torch.nn.Linear(8, 8)
        self.actor = torch.nn.Linear(8, 5)
        self.value = torch.nn.Linear(8, 1)
        self.optimizer = torch.optim.Adam(self.parameters(), lr=3e-4)

    def set_training_mode(self, mode):
        self.train(mode)

    def evaluate_actions(self, observations, actions):
        x = torch.cat((observations['image'].float().flatten(1)/255., observations['aux']), dim=1)
        features = torch.tanh(self.features(x))
        dist = torch.distributions.Categorical(logits=self.actor(features))
        return self.value(features), dist.log_prob(actions), dist.entropy()


def buffer(device='cpu'):
    obs = spaces.Dict(image=spaces.Box(0, 255, (1, 2, 3), np.uint8),
                      aux=spaces.Box(0., 1., (2,), np.float32))
    result = TimedDictBuffer(8, obs, spaces.Discrete(5), device=device, n_envs=1)
    rng = np.random.default_rng(4)
    for value in result.observations.values():
        value[:] = rng.random(value.shape)*100
    result.actions[:] = rng.integers(0, 5, size=result.actions.shape)
    for name in ('values', 'returns', 'advantages'):
        value = getattr(result, name)
        value[:] = rng.normal(size=value.shape)
    result.log_probs[:] = -np.log(5.)
    result.full = True
    result.pos = 8
    return result


class Learner:
    def __init__(self, masks):
        torch.manual_seed(2)
        self.policy = Policy()
        self.rollout_buffer = buffer()
        self.rollout_buffer.actor_masks[:, 0] = masks
        self._settled_stats = dict(optimizer_sample_presentations=0, optimized_transitions=0)
        self._settled_trained_this_run = 0
        self._settled_checkpoint_due = False
        self._settled_deadline = float('inf')
        self._current_progress_remaining = 1.
        self.n_steps, self.n_epochs, self.batch_size, self._n_updates = 8, 2, 2, 0
        self.normalize_advantage = True
        self.clip_range = lambda _: .2
        self.clip_range_vf = None
        self.ent_coef, self.vf_coef, self.max_grad_norm = .002, .5, .5
        self.target_kl = None
        self.num_timesteps = 8
        self.logs = {}
        self.logger = SimpleNamespace(record=lambda key, value, **kwargs: self.logs.__setitem__(key, value))
        self.records = []

    def _update_learning_rate(self, optimizer):
        pass

    def _record(self, row):
        self.records.append(row)


class RuntimeTests(unittest.TestCase):
    def compare(self, masks, clip_vf=False):
        old, new = Learner(masks), Learner(masks)
        if clip_vf:
            old.clip_range_vf = new.clip_range_vf = lambda _: .2
        # Exercise existing Adam momentum, including all-passive batches.
        for learner in (old, new):
            for parameter in learner.policy.parameters():
                parameter.grad = torch.full_like(parameter, .01)
            learner.policy.optimizer.step()
            learner.policy.optimizer.zero_grad()
        old_actor = copy.deepcopy(old.policy.actor.state_dict())
        np.random.seed(75)
        SettledPPO.train(old)
        np.random.seed(75)
        deferred_train(new)
        for key, value in old.policy.state_dict().items():
            torch.testing.assert_close(value, new.policy.state_dict()[key], rtol=1e-6, atol=1e-7)
        self.assertEqual(old.records, new.records)
        self.assertEqual(old._settled_stats, new._settled_stats)
        self.assertEqual(old._n_updates, new._n_updates)
        for key in old.logs:
            np.testing.assert_allclose(old.logs[key], new.logs[key], rtol=1e-6, atol=1e-7)
        if not any(masks):
            for key, value in old_actor.items():
                self.assertTrue(torch.equal(value, new.policy.actor.state_dict()[key]))

    def test_all_actor_updates_match(self):
        self.compare([1]*8)

    def test_mixed_actor_passive_match(self):
        self.compare([0, 1, 0, 1, 1, 0, 1, 0])

    def test_all_passive_keeps_actor_momentum_frozen(self):
        self.compare([0]*8)

    def test_value_clipping_match(self):
        self.compare([0, 1, 1, 1, 1, 0, 1, 0], clip_vf=True)

    def test_cpu_transfer_fallback(self):
        for mode in ('gpu_cache', 'pinned'):
            original = buffer()
            wrapped = UpdateTransfers(buffer(), mode, 2)
            self.assertEqual(wrapped.mode, 'standard')
            np.random.seed(22)
            expected = list(original.get(2))
            np.random.seed(22)
            actual = list(wrapped.get(2))
            for left, right in zip(expected, actual):
                self.assert_samples(left, right)
            wrapped.close()

    def assert_samples(self, left, right):
        np.testing.assert_array_equal(left.sample_indices, right.sample_indices)
        for key in left.observations:
            torch.testing.assert_close(left.observations[key], right.observations[key], rtol=0, atol=0)
        for name in ('actions', 'old_values', 'old_log_prob', 'advantages', 'returns', 'actor_mask'):
            torch.testing.assert_close(getattr(left, name), getattr(right, name), rtol=0, atol=0)

    def test_cuda_transfer_order_and_lifetime(self):
        if not CUDA:
            self.skipTest('Run check.py --cuda when the GPU is idle')
        self.assertTrue(torch.cuda.is_available(), 'CUDA explicitly requested but unavailable')
        for mode in ('gpu_cache', 'pinned'):
            baseline = buffer('cuda')
            source = buffer('cuda')
            wrapped = UpdateTransfers(source, mode, 3, reserve_mib=32)
            try:
                self.assertEqual(wrapped.mode, mode, wrapped.reason)
                for epoch in range(2):
                    np.random.seed(epoch)
                    expected = list(baseline.get(3))
                    np.random.seed(epoch)
                    actual = list(wrapped.get(3))
                    # Compare AFTER all yields: later transfers must not overwrite
                    # device observations belonging to earlier minibatches.
                    for left, right in zip(expected, actual):
                        self.assert_samples(left, right)
            finally:
                wrapped.close()
            self.assertFalse(wrapped.cache)
            self.assertFalse(wrapped.host)

    def test_real_pack_final_temporal_observations_match(self):
        import train_rs_v612_ppo as trainer
        from development.staged.hybrid_history_20261102.hybrid import HybridHistory
        cls = trainer.build_env_class()
        original = cls._pack
        modified = latest_only(original)
        frame = np.arange(3*4*4, dtype=np.uint8).reshape(3, 4, 4)
        def raw():
            return SimpleNamespace(_poll_result=lambda: None, _native_frames=lambda: (frame, frame),
                native_history=deque(maxlen=4), _background_present_at=lambda: None,
                previous_observation_at=None, last_capture_end=10., frame_intervals=deque(maxlen=4),
                detector_pool=None, _detector_feature_cache=None, last_detection=None,
                episode_tag=1, max_stale_s=.5, hazard_observation_count=0, stale_observation_count=0,
                hazard_nonzero_observations=0, fresh_steps=0, decision_ages_ms=[], all_steps=0,
                action_history=deque(maxlen=4), episode_started=0., action_context=trainer.ActionContext(assume_center=False),
                column_nonzero_observations=0, pack_ms=[], reuse_observation_guard=False,
                verified_progress=False, _headless_capture_timing={'request_started_at':10.})
        # Pin the aux clock so the two executions receive identical inputs.
        from unittest.mock import patch
        with patch.object(trainer.time, 'perf_counter', return_value=10.):
            left, right = raw(), raw()
            a = original(left, None)[0]
            b = modified(right, None)[0]
        np.testing.assert_array_equal(a['aux'], b['aux'])
        np.testing.assert_array_equal(a['image'][-3:], b['image'])
        h1, h2 = HybridHistory(1, [.05, .25, .75]), HybridHistory(1, [.05, .25, .75])
        for stamp in (10., 10.05, 10.25, 10.75, 11.):
            for history in (h1, h2):
                history.add(frame, stamp-.001, history.token())
            x, tx = h1.observation(a['image'][-3:], stamp)
            y, ty = h2.observation(b['image'][-3:], stamp)
            np.testing.assert_array_equal(x, y)
            np.testing.assert_array_equal(tx, ty)
            saved = y.copy()
            h2.observation(np.zeros_like(frame), stamp+.0001)
            np.testing.assert_array_equal(saved, y)
            # Keep both state machines synchronized for the next iteration.
            h1.observation(np.zeros_like(frame), stamp+.0001)

    def test_z_factory_contract_and_pack_owner(self):
        import sys
        from pathlib import Path
        root = Path(__file__).resolve().parents[3]
        sys.path.insert(0, str(root/'development/experiments/command_history'))
        import command_adapter
        trainer = command_adapter.install()
        before = command_adapter.contract()
        from observation import install
        install(trainer)
        cls = trainer.build_env_class()
        owner = next(c for c in cls.__mro__ if c.__name__ == 'PPOV61Env')
        self.assertTrue(owner._pack._latest_only)
        self.assertEqual(before, command_adapter.contract())
        self.assertEqual(before['image_shape'], [12, 384, 384])
        self.assertEqual(before['aux_dim'], 789)
