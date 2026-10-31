"""Exact four-frame storage; reconstruct only the requested PPO minibatch."""
import hashlib
from types import SimpleNamespace
import numpy as np
from stable_baselines3.common.buffers import BaseBuffer
from tools.ppo_settled_learning import TimedDictBuffer


class CompactImages:
    def __init__(self, shape, *, frames=None, refs=None):
        if len(shape) != 5 or shape[1] != 1 or shape[2] != 12:
            raise ValueError('Compact storage requires one environment and four RGB frames')
        self.shape, self.dtype = tuple(shape), np.dtype('uint8')
        self.frames = frames if frames is not None else []
        self.capacity=shape[0]*4
        self.refs = refs if refs is not None else np.full((shape[0], 4), -1, np.int32)
        self.count = 0
        self.lookup = {}
        self.last = None
        self.sources = {}

    def _intern(self, frame):
        digest = hashlib.blake2b(frame.tobytes(), digest_size=16).digest()
        for index in self.lookup.get(digest, []):
            if np.array_equal(self.frames[index], frame):
                return index
        if self.count >= self.capacity:
            raise ValueError('Compact frame pool exhausted')
        index = self.count
        if isinstance(self.frames,list):self.frames.append(frame.copy())
        else:self.frames[index] = frame
        self.lookup.setdefault(digest, []).append(index)
        self.count += 1
        return index

    def __setitem__(self, index, stack):
        if not isinstance(index, (int, np.integer)):
            raise ValueError('Compact collection writes one transition at a time')
        stack = np.asarray(stack)
        if stack.shape != self.shape[1:] or stack.dtype != self.dtype:
            raise ValueError('Compact observation shape/dtype mismatch')
        pieces = stack[0].reshape(4, 3, *self.shape[-2:])
        if self.last is not None and all(np.array_equal(pieces[i], self.frames[self.last[i+1]]) for i in range(3)):
            refs = [*self.last[1:], self._intern(pieces[-1])]
        else:
            refs = [self._intern(frame) for frame in pieces]
        self.refs[index] = refs
        self.last = refs

    def __getitem__(self, index):
        scalar = isinstance(index, (int, np.integer))
        indices = np.arange(self.shape[0])[index]
        indices = np.atleast_1d(indices)
        result = np.empty((len(indices), *self.shape[1:]), np.uint8)
        for out, i in enumerate(indices):
            self._copy_row(int(i),result[out])
        return result[0] if scalar else result

    def _copy_row(self,index,target):
        if index in self.sources:
            source,source_index=self.sources[index]
            if hasattr(source,'_copy_row'):source._copy_row(source_index,target)
            else:np.copyto(target,source[source_index])
            return
        refs=self.refs[index]
        if (refs<0).any() or (refs>=self.count).any():raise ValueError('Unpublished compact observation')
        for i,ref in enumerate(refs):np.copyto(target[0,i*3:(i+1)*3],self.frames[int(ref)])

    def borrow(self, begin, source):
        """The coordinator reads published slots only until optimizer commit."""
        if source.shape[1:] != self.shape[1:] or source.dtype != self.dtype:
            raise ValueError('Borrowed compact observation schema mismatch')
        for i in range(source.shape[0]):
            self.sources[begin+i] = (source, i)

    def copy(self):
        return self[:]

    @property
    def stored_bytes(self):
        return self.count * 3 * self.shape[-2] * self.shape[-1] + self.refs.nbytes


class CompactTimedDictBuffer(TimedDictBuffer):
    def add(self,obs,action,reward,episode_start,value,log_prob):
        # Frame interning owns new pixels; avoid SB3's additional full-stack copy.
        for key in self.observations:self.observations[key][self.pos]=np.asarray(obs[key])
        self.actions[self.pos]=action.reshape((self.n_envs,self.action_dim))
        self.rewards[self.pos]=reward
        self.episode_starts[self.pos]=episode_start
        self.values[self.pos]=value.clone().cpu().numpy().flatten()
        self.log_probs[self.pos]=log_prob.clone().cpu().numpy().reshape(-1)
        self.pos+=1
        self.full=self.pos==self.buffer_size

    def reset(self):
        if self.n_envs != 1 or self.obs_shape['image'][0] != 12:
            raise ValueError('Compact buffer requires the existing four-frame observation')
        self.observations = {key: (CompactImages((self.buffer_size, 1, *shape)) if key == 'image'
                                  else np.zeros((self.buffer_size, 1, *shape), self.observation_space[key].dtype))
                             for key, shape in self.obs_shape.items()}
        self.actions = np.zeros((self.buffer_size, 1, self.action_dim), self.action_space.dtype)
        for name in ('rewards', 'returns', 'episode_starts', 'values', 'log_probs', 'advantages'):
            setattr(self, name, np.zeros((self.buffer_size, 1), np.float32))
        self.active_seconds = np.zeros((self.buffer_size, 1), np.float64)
        self.actor_masks = np.ones((self.buffer_size, 1), np.float32)
        self.generator_ready = False
        BaseBuffer.reset(self)

    def get(self, batch_size=None):
        if not self.full:
            raise ValueError('Cannot train on an incomplete compact rollout')
        self.generator_ready=True
        indices = np.random.permutation(self.buffer_size)
        batch_size = batch_size or self.buffer_size
        for start in range(0, len(indices), batch_size):
            yield self._get_samples(indices[start:start+batch_size])

    def _get_samples(self, batch_inds, env=None):
        return SimpleNamespace(
            observations={key:self.to_torch(value[batch_inds][:, 0]) for key,value in self.observations.items()},
            actions=self.to_torch(self.actions[batch_inds, 0].astype(np.float32, copy=False)),
            old_values=self.to_torch(self.values[batch_inds].flatten()),
            old_log_prob=self.to_torch(self.log_probs[batch_inds].flatten()),
            advantages=self.to_torch(self.advantages[batch_inds].flatten()),
            returns=self.to_torch(self.returns[batch_inds].flatten()),
            sample_indices=batch_inds,
            actor_mask=self.to_torch(self.actor_masks.reshape(-1)[batch_inds]))
