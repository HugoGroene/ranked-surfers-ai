"""Balanced BC sampling and bounded frame mappings over the existing export."""
from collections import OrderedDict
from pathlib import Path
import numpy as np
from common import read, digest
from dataset import stacked


class BoundedDataset:
    def __init__(self, folder, split, *, verify=True, max_frame_maps=2):
        self.folder = Path(folder).resolve()
        self.meta = read(self.folder/'dataset.json')
        self.arrays, self.samples, self.paths = [], [], []
        self.frame_maps = OrderedDict()
        self.max_frame_maps = max_frame_maps
        self.frame_bytes_limit = 2*1024**3
        self.peak_frame_mappings = 0
        if max_frame_maps < 1:
            raise ValueError('Positive frame mapping limit required')
        if self.meta.get('status') != 'COMPLETE' or self.meta.get('schema') != 'multimap-bc-fixed4-v1':
            raise ValueError('Incomplete/incompatible export')
        runs = [r for r in self.meta['runs'] if r['split'] == split]
        try:
            for number, run in enumerate(runs):
                path = (self.folder/run['path']).resolve()
                if not path.is_relative_to(self.folder):
                    raise ValueError('Dataset path escapes export')
                expected_names = {'frames.npy', 'indices.npy', 'aux.npy', 'actions.npy', 'samples.json'}
                if set(run['hashes']) != expected_names:
                    raise ValueError('Missing/unexpected dataset hashes')
                if verify:
                    print(f'[BC VERIFY] {split} run {number+1}/{len(runs)}: {run["path"]}', flush=True)
                    for name, sha in run['hashes'].items():
                        if digest(path/name) != sha:
                            raise ValueError('Dataset checksum mismatch: '+str(path/name))
                arrays = {}
                self.arrays.append(arrays)
                self.paths.append(path)
                for name in ('indices', 'aux', 'actions'):
                    arrays[name] = np.load(path/(name+'.npy'), mmap_mode='r', allow_pickle=False)
                n = run['count']
                frames = np.load(path/'frames.npy', mmap_mode='r', allow_pickle=False)
                try:
                    if (frames.dtype != np.uint8 or frames.shape[1:] != (3, *self.meta['contract']['image_shape'][-2:])
                            or arrays['indices'].shape != (n, 4) or arrays['indices'].dtype.kind not in 'iu'
                            or arrays['aux'].shape != (n, 789) or arrays['aux'].dtype != np.float32
                            or arrays['actions'].shape != (n,) or arrays['actions'].dtype.kind not in 'iu'
                            or not np.isin(arrays['actions'], range(5)).all()
                            or not np.isfinite(arrays['aux']).all() or np.any(arrays['aux'] < 0)
                            or np.any(arrays['aux'] > 1) or np.any(arrays['indices'] < -1)
                            or np.any(arrays['indices'] >= len(frames))):
                        raise ValueError('Dataset shape/range mismatch: '+str(path))
                    if frames.nbytes > self.frame_bytes_limit:
                        raise ValueError('Individual frame file exceeds bounded mapping limit')
                finally:
                    frames._mmap.close()
                self.samples.extend((number, i, run['map'], run['session']) for i in range(n))
            if not self.samples:
                raise ValueError('No '+split+' samples')
        except BaseException:
            self.close()
            raise

    def sample(self, k):
        r, i, _, _ = self.samples[k]
        if r in self.frame_maps:
            frames = self.frame_maps.pop(r)
        else:
            size = (self.paths[r]/'frames.npy').stat().st_size
            while self.frame_maps and (len(self.frame_maps) >= self.max_frame_maps
                    or size + sum(a.nbytes for a in self.frame_maps.values()) > self.frame_bytes_limit):
                _, old = self.frame_maps.popitem(last=False)
                old._mmap.close()
            frames = np.load(self.paths[r]/'frames.npy', mmap_mode='r', allow_pickle=False)
        self.frame_maps[r] = frames
        self.peak_frame_mappings = max(self.peak_frame_mappings, len(self.frame_maps))
        a = self.arrays[r]
        # stacked() and aux.copy() own their memory before an LRU eviction.
        return dict(image=stacked(frames, a['indices'][i]), aux=a['aux'][i].copy()), int(a['actions'][i])

    def drop_frames(self):
        for array in self.frame_maps.values():
            array._mmap.close()
        self.frame_maps.clear()

    def close(self):
        self.drop_frames()
        for arrays in self.arrays:
            for array in arrays.values():
                array._mmap.close()
        self.arrays.clear()


class BalancedSampler:
    """Per 16 samples: 8 NONE, 2 each movement; equal maps within every label."""
    def __init__(self, data):
        self.groups = {}
        for k, (r, i, name, _) in enumerate(data.samples):
            action = int(data.arrays[r]['actions'][i])
            self.groups.setdefault((name, action), []).append(k)
        expected = {(name, a) for name in ('halloween', 'tokyo') for a in range(5)}
        if set(self.groups) != expected:
            raise ValueError('Need all five actions on each training map')
        self.groups = {key: np.asarray(ids, dtype=np.int64) for key, ids in self.groups.items()}

    def batch(self, rng, batch_size=16):
        if batch_size < 16 or batch_size % 16:
            raise ValueError('Balanced batch size must be a multiple of 16')
        result = []
        for _ in range(batch_size//16):
            for name in ('halloween', 'tokyo'):
                for action in (0, 0, 0, 0, 1, 2, 3, 4):
                    pool = self.groups[name, action]
                    result.append(int(pool[rng.integers(len(pool))]))
        rng.shuffle(result)
        return result


def candidate_gate(metrics):
    """Offline eligibility only; gameplay evaluation is still required."""
    reasons = []
    for name, m in metrics['maps'].items():
        recall = m['per_action_recall']
        movement = float(np.mean([v for v in recall[1:] if v is not None]))
        if not np.isfinite(m['macro_cross_entropy']):
            reasons.append(name+': nonfinite validation loss')
        if recall[0] is None or recall[0] < .1:
            reasons.append(name+': NONE recall below 10%')
        if movement < .05:
            reasons.append(name+': mean movement recall below 5%')
        if sum(v > 0 for v in m['prediction_counts']) < 2:
            reasons.append(name+': only one predicted action')
        if m['max_probability_range'] < 1e-5:
            reasons.append(name+': nearly constant probabilities')
    return not reasons, reasons
