"""One masked PPO optimizer over independently settled worker trajectories.

Workers retain the validated one-environment collector. GAE is computed within
each worker before merge; trajectories never bootstrap across worker IDs.
"""
import hashlib
from pathlib import Path

import numpy as np

from tools.ppo_settled_learning import SettledPPO, TimedDictBuffer


BUFFER_FIELDS = ('actions', 'rewards', 'episode_starts', 'values', 'log_probs',
                 'advantages', 'returns', 'active_seconds', 'actor_masks')


def policy_arrays(policy):
    return {key: value.detach().cpu().numpy().copy() for key, value in policy.state_dict().items()}


def policy_hash(arrays):
    digest = hashlib.sha256()
    for key, value in sorted(arrays.items()):
        digest.update(key.encode())
        digest.update(str(value.shape).encode())
        digest.update(str(value.dtype).encode())
        digest.update(np.ascontiguousarray(value).tobytes())
    return digest.hexdigest()


def install_policy(policy, arrays, expected):
    import torch
    if policy_hash(arrays) != expected:
        raise ValueError('Policy transport hash mismatch')
    policy.load_state_dict({key: torch.from_numpy(value) for key, value in arrays.items()}, strict=True)


def rollout_packet(buffer, worker_id, version, digest):
    if not buffer.full or buffer.generator_ready or buffer.n_envs != 1:
        raise ValueError('Need a complete, unflattened one-worker rollout')
    return dict(worker=worker_id, version=version, policy_hash=digest,
                observations={key: value.copy() for key, value in buffer.observations.items()},
                **{key: getattr(buffer, key).copy() for key in BUFFER_FIELDS})


def merge_rollouts(buffer, packets, worker_steps, version, digest):
    if sorted(p['worker'] for p in packets) != list(range(len(packets))):
        raise ValueError('Missing or duplicate worker trajectory')
    if buffer.n_envs != 1 or buffer.buffer_size != worker_steps*len(packets):
        raise ValueError('Aggregate buffer size mismatch')
    buffer.reset()
    for packet in packets:
        if packet['version'] != version or packet['policy_hash'] != digest:
            raise ValueError('Mixed/stale policy versions in a PPO update')
        begin = packet['worker']*worker_steps
        section = slice(begin, begin+worker_steps)
        if set(packet['observations']) != set(buffer.observations):
            raise ValueError('Observation schema mismatch')
        for key, target in buffer.observations.items():
            source = packet['observations'][key]
            if source.shape != target[section].shape or source.dtype != target.dtype:
                raise ValueError('Observation shape/dtype mismatch')
            target[section] = source
        for key in BUFFER_FIELDS:
            target, source = getattr(buffer, key), packet[key]
            if source.shape != target[section].shape or not np.isfinite(source).all():
                raise ValueError(f'Invalid {key} payload')
            target[section] = source
        if not np.isin(packet['actor_masks'], (0., 1.)).all() or (packet['active_seconds'] < 0).any():
            raise ValueError('Invalid passive mask/duration')
    # Returns and advantages already contain per-worker terminal/timeout GAE.
    # Recomputing GAE here would invent transitions between unrelated games.
    buffer.full, buffer.pos, buffer.generator_ready = True, buffer.buffer_size, False


class SharedRolloutPPO(SettledPPO):
    """Optimizer-only coordinator; never directly controls a live environment."""
    def _record(self, row):
        super()._record(row)
        if row['event'] == 'optimized':
            self.parallel_used_indices = row['used_indices']

    def train(self):
        worker_steps = self.n_steps
        self.n_steps = self.rollout_buffer.buffer_size
        try:
            return super().train()
        finally:
            self.n_steps = worker_steps  # checkpoints retain steps per worker


def configure_learner(model, output, worker_count, reference_seconds):
    from stable_baselines3.common.logger import configure
    if model.n_envs != 1 or model.use_sde:
        raise ValueError('Source must be a one-environment discrete PPO checkpoint')
    if model.target_kl is not None:
        raise ValueError('Prototype requires the existing no-KL-stop setting for exact worker sample accounting')
    model.__class__ = SharedRolloutPPO
    model.rollout_buffer = TimedDictBuffer(model.n_steps*worker_count,
        model.observation_space, model.action_space, device=model.device,
        gamma=model.gamma, gae_lambda=model.gae_lambda, n_envs=1)
    model.rollout_buffer.reference_seconds = reference_seconds
    model.settled_reference_seconds = reference_seconds
    model._settled_output = Path(output)
    model._settled_ledger = Path(output)/'optimizer.jsonl'
    model._settled_trained_this_run = 0
    model._settled_checkpoint_due = False
    model._settled_deadline = float('inf')
    model._settled_stats = dict(collected_transitions=0, optimized_transitions=0,
        actor_transitions=0, passive_transitions=0, boundary_settlements=0,
        completed_rollout_transitions=0, optimizer_sample_presentations=0,
        discarded_partial_transitions=0, stop_reason=None)
    model.parallel_used_indices = []
    model._current_progress_remaining = 1.
    model.set_logger(configure(str(output), ['stdout', 'csv']))
    return model
