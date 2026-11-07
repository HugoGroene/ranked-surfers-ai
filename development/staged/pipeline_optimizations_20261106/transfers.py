"""Bounded, update-scoped transfer alternatives. No permanent rollout GPU cache.

Each get() shuffles exactly once using NumPy, as DictRolloutBuffer does. Images
stay uint8 until the existing policy preprocessor. No changes to GAE or masks.
"""
from types import SimpleNamespace


def cache_fits(required, free, limit, reserve):
    return required > 0 and required <= limit and required + reserve <= free


class UpdateTransfers:
    def __init__(self, buffer, mode, batch_size, *, gpu_limit_mib=1024, reserve_mib=2048,
                 pinned_limit_mib=96):
        import numpy as np
        import torch
        if mode not in ('gpu_cache', 'pinned'):
            raise ValueError('Unknown transfer mode')
        if buffer.n_envs != 1 or not buffer.full or buffer.generator_ready:
            raise ValueError('Need a fresh full aggregate one-env rollout')
        if batch_size < 1 or min(gpu_limit_mib, reserve_mib, pinned_limit_mib) < 0:
            raise ValueError('Invalid buffer budgets')
        self.buffer, self.device = buffer, torch.device(buffer.device)
        self.cache, self.host, self.event = {}, {}, None
        self.event_recorded = False
        self.mode, self.reason = 'standard', 'not a CUDA buffer'
        self.obs = {}
        if self.device.type != 'cuda':
            return
        if any(not isinstance(v, np.ndarray) for v in buffer.observations.values()):
            raise ValueError('Dense observations required; compact storage must use its own iterator')
        self.obs = {k: v.reshape((-1, *v.shape[2:])) for k, v in buffer.observations.items()}
        required = sum(v.nbytes for v in self.obs.values())
        mib = 1024**2
        if mode == 'gpu_cache':
            free, _ = torch.cuda.mem_get_info(self.device)
            if not cache_fits(required, free, gpu_limit_mib*mib, reserve_mib*mib):
                self.reason = 'GPU budget/headroom check declined cache'
                return
            # Existing CPU rollout is immutable throughout this update. Copies
            # use the current stream; all indexing/forward work follows them.
            try:
                self.cache = {k: torch.as_tensor(v, device=self.device) for k, v in self.obs.items()}
            except torch.cuda.OutOfMemoryError:
                self.cache.clear()
                self.reason = 'CUDA allocation declined cache'
                return
        else:
            needed = sum(v[0].nbytes for v in self.obs.values()) * batch_size
            if needed > pinned_limit_mib*mib:
                self.reason = 'Pinned buffer budget exceeded'
                return
            try:
                self.host = {k: torch.empty((batch_size, *v.shape[1:]),
                             dtype=torch.from_numpy(v[:1]).dtype, pin_memory=True)
                             for k, v in self.obs.items()}
            except RuntimeError as exc:
                self.host.clear()
                self.reason = 'Pinned allocation unavailable: '+str(exc)
                return
            self.event = torch.cuda.Event()
        self.mode, self.reason = mode, 'enabled for this update only'

    def __getattr__(self, name):
        return getattr(self.buffer, name)

    def get(self, batch_size=None):
        import numpy as np
        import torch
        if self.mode == 'standard':
            yield from self.buffer.get(batch_size)
            return
        batch_size = batch_size or self.buffer.buffer_size
        indices = np.random.permutation(self.buffer.buffer_size)
        for start in range(0, len(indices), batch_size):
            ids = indices[start:start+batch_size]
            if self.mode == 'gpu_cache':
                gpu_ids = torch.as_tensor(ids, device=self.device)
                obs = {k: value[gpu_ids] for k, value in self.cache.items()}
            else:
                if any(len(value) < len(ids) for value in self.host.values()):
                    raise ValueError('Minibatch exceeds allocated pinned buffer')
                # Never overwrite source pages while the previous DMA reads them.
                if self.event_recorded:
                    self.event.synchronize()
                obs = {}
                for k, value in self.obs.items():
                    host = self.host[k][:len(ids)]
                    np.take(value, ids, axis=0, out=host.numpy())
                    # Fresh device tensor: do not mutate an earlier autograd input.
                    obs[k] = host.to(self.device, non_blocking=True)
                self.event.record(torch.cuda.current_stream(self.device))
                self.event_recorded = True
            def field(name):
                value = getattr(self.buffer, name)
                # n_envs is explicitly one; matches swap_and_flatten ordering.
                selected = value.reshape((-1, *value.shape[2:]))[ids]
                if name == 'actions':
                    selected = selected.astype(np.float32, copy=False)
                return self.buffer.to_torch(selected)
            yield SimpleNamespace(observations=obs, actions=field('actions'),
                old_values=field('values').flatten(), old_log_prob=field('log_probs').flatten(),
                advantages=field('advantages').flatten(), returns=field('returns').flatten(),
                actor_mask=field('actor_masks').flatten(), sample_indices=ids)

    def close(self):
        if self.event_recorded:
            self.event.synchronize()
        self.cache.clear()
        self.host.clear()
        self.obs.clear()
