#!/usr/bin/env python3
"""Reversible actor-only behavioral cloning of approved V6.1.2 human replays.

Usage: python rs_v612_demo_bc.py --checkpoint ai_data/.../latest.zip
Writes a NEW checkpoint under ai_data/demonstrations_v1/bc_runs; never modifies input.
"""
from __future__ import annotations

import argparse
from collections import Counter
from datetime import datetime
import json
from pathlib import Path
import random

import numpy as np

from rs_v61_features import AUX_DIM, FEATURE_VERSION, ACTION_NAMES


def approved_runs(root):
    result = []
    for folder in sorted((root/'runs').glob('demo_*')):
        path = folder/'manifest.json'
        if not path.exists():
            continue
        m = json.loads(path.read_text(encoding='utf-8'))
        if m.get('status') != 'approved':
            continue
        if m.get('schema') != 1 or m.get('feature_version') != FEATURE_VERSION or m.get('aux_dim') != AUX_DIM:
            raise RuntimeError(f'Incompatible approved demo: {folder}')
        result.append((folder, m))
    return result


class Replay:
    """Lazy per-run loader; only manifest-approved and hand-selected intervals."""
    def __init__(self, folder, manifest):
        self.folder = folder
        self.m = manifest
        self.actions = np.load(folder/'actions.npy', mmap_mode='r', allow_pickle=False)
        self.times = np.load(folder/'timestamps.npy', mmap_mode='r', allow_pickle=False)
        self.valid = np.load(folder/'valid.npy', mmap_mode='r', allow_pickle=False)
        selected = np.zeros(len(self.actions), dtype=bool)
        for t0, t1 in manifest['valid_segments_seconds']:
            # Upper boundary exclusive: no cut-point sample crossing excluded interval.
            selected |= (self.times >= t0) & (self.times < t1)
        self.indices = np.flatnonzero(selected & self.valid)
        if not np.any(self.actions[self.indices] != 0):
            raise ValueError(f'Approved replay has only NONE actions; likely missing keyboard events: '
                             f'{folder}. Mark as candidate or reject and record with fixed collector.')
        self._frames = None
        self._aux = None
        if self.indices.size == 0:
            raise ValueError(f'Approved replay contains no valid samples: {folder}')

    def load(self):
        if self._frames is None:
            with np.load(self.folder/'frames.npz', allow_pickle=False) as z:
                self._frames = z['frames']
            self._aux = np.load(self.folder/'aux.npy', mmap_mode='r', allow_pickle=False)
            if self._frames.ndim != 4 or self._frames.shape[1:] != (3,288,512):
                raise ValueError(f'Unexpected frame resolution: {self.folder}: {self._frames.shape}')
            if self._aux.shape != (len(self.actions), AUX_DIM):
                raise ValueError(f'Unexpected aux size: {self.folder}: {self._aux.shape}')
        return self

    def release(self):
        self._frames = None
        self._aux = None

    def sample(self, i):
        self.load()
        return (np.concatenate([self._frames[max(0,i-j)] for j in (3,2,1,0)], axis=0),
                np.asarray(self._aux[i], dtype=np.float32), int(self.actions[i]))


def sample_weights(replays, *, none_fraction=.35, seed=77):
    """Preserve all real action events and bounded NONE subsampling, per train runs."""
    randomizer = np.random.default_rng(seed)
    positives = []
    nones = []
    for k, replay in enumerate(replays):
        for i in replay.indices:
            label = int(replay.actions[i])
            (nones if label == 0 else positives).append((k, int(i)))
    # Never oversample rare actions into mislabeled situations.
    keep_none = min(len(nones), max(50, round(len(positives)*none_fraction/max(.01,1.-none_fraction))))
    if len(nones) > keep_none:
        inds = randomizer.choice(len(nones), keep_none, replace=False)
        nones = [nones[k] for k in inds]
    samples = positives + nones
    randomizer.shuffle(samples)
    return samples


def batches(replays, indices, batch_size, device, rng):
    import torch
    # Shuffling within a run reduces repeated compressed-file loads.
    per_run = {}
    for k, i in indices:
        per_run.setdefault(k, []).append(i)
    blocks = []
    for k, ids in per_run.items():
        rng.shuffle(ids)
        for off in range(0, len(ids), batch_size):
            blocks.append((k, ids[off:off+batch_size]))
    rng.shuffle(blocks)
    for k, ids in blocks:
        data = [replays[k].sample(i) for i in ids]
        images = np.stack([x[0] for x in data], axis=0)
        aux = np.stack([x[1] for x in data], axis=0)
        acts = np.asarray([x[2] for x in data], dtype=np.int64)
        yield ({'image':torch.as_tensor(images, device=device),
                'aux':torch.as_tensor(aux, device=device)},
               torch.as_tensor(acts, dtype=torch.long, device=device))
        replays[k].release()


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--checkpoint', type=Path, required=True)
    p.add_argument('--root', type=Path, default=Path('ai_data/demonstrations_v1'))
    p.add_argument('--epochs', type=int, default=3)
    p.add_argument('--batch-size', type=int, default=12)
    p.add_argument('--lr', type=float, default=2e-5)
    p.add_argument('--holdout-fraction', type=float, default=.20)
    p.add_argument('--none-fraction', type=float, default=.35)
    p.add_argument('--seed', type=int, default=77)
    a = p.parse_args()
    if not a.checkpoint.is_file():
        p.error('Checkpoint not found')
    if not (1 <= a.epochs <= 20 and 1 <= a.batch_size <= 64 and 0 < a.lr <= 1e-3
            and .05 <= a.holdout_fraction <= .5 and 0 < a.none_fraction < 1):
        p.error('Invalid BC configuration')
    found = approved_runs(a.root)
    if len(found) < 3:
        p.error('Collect at least 3 explicitly approved runs to train with a run-level holdout')
    rng = random.Random(a.seed)
    rng.shuffle(found)
    n_val = max(1, round(len(found)*a.holdout_fraction))
    validation = [Replay(*x) for x in found[:n_val]]
    training = [Replay(*x) for x in found[n_val:]]
    samples = sample_weights(training, none_fraction=a.none_fraction, seed=a.seed)
    validation_indices = [(j, int(i)) for j, r in enumerate(validation) for i in r.indices]
    counts = Counter(int(training[j].actions[i]) for j,i in samples)
    if len(samples) < 20 or not counts[1] or not counts[2]:
        p.error('Need >=20 selected training samples and both LEFT and RIGHT examples')
    print('Training runs:', len(training), 'validation runs:', len(validation))
    print('Training samples:', len(samples), 'actions:', {ACTION_NAMES[k]:v for k,v in counts.items()})
    print('Validation samples:', len(validation_indices), '(natural action proportions)')

    import torch
    from stable_baselines3 import PPO
    import train_rs_v3_ppo  # noqa: F401; needed for model feature extractor deserialization.
    torch.manual_seed(a.seed)
    np.random.seed(a.seed)
    # Do not change input file. Actor-only BC preserves the visual feature extractor and critic.
    model = PPO.load(str(a.checkpoint), device='cuda' if torch.cuda.is_available() else 'cpu')
    policy = model.policy
    for param in policy.parameters():
        param.requires_grad_(False)
    actor = list(policy.mlp_extractor.policy_net.parameters()) + list(policy.action_net.parameters())
    for param in actor:
        param.requires_grad_(True)
    optimizer = torch.optim.Adam(actor, lr=a.lr)
    device = policy.device
    report = {'source_checkpoint':str(a.checkpoint), 'train_folders':[str(x.folder) for x in training],
              'validation_folders':[str(x.folder) for x in validation], 'samples':len(samples),
              'training_class_counts':{ACTION_NAMES[k]:v for k,v in counts.items()}, 'epochs':[]}

    def eval_loss(replays, ids):
        total, hits, used = 0., 0, 0
        policy.eval()
        with torch.no_grad():
            for observations, acts in batches(replays, ids, a.batch_size, device, rng):
                distribution = policy.get_distribution(observations)
                log_prob = distribution.log_prob(acts)
                total += float((-log_prob).sum().item())
                hits += int((distribution.get_actions(deterministic=True) == acts).sum().item())
                used += len(acts)
        return {'cross_entropy':round(total/max(1,used),5), 'accuracy':round(hits/max(1,used),4)}

    baseline = eval_loss(validation, validation_indices)
    report['validation_before'] = baseline
    best = float('inf')
    best_actor = None
    for epoch in range(a.epochs):
        policy.train()
        total_loss, total = 0., 0
        for observations, acts in batches(training, samples, a.batch_size, device, rng):
            distribution = policy.get_distribution(observations)
            loss = -distribution.log_prob(acts).mean()
            optimizer.zero_grad(set_to_none=True)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(actor, 1.)
            optimizer.step()
            total_loss += float(loss.detach().item())*len(acts)
            total += len(acts)
        score = eval_loss(validation, validation_indices)
        report['epochs'].append({'epoch':epoch+1, 'train_loss':round(total_loss/max(1,total),5), **score})
        print(f'BC epoch {epoch+1}/{a.epochs}: training loss {total_loss/max(1,total):.4f}; validation {score}')
        if score['cross_entropy'] < best:
            best = score['cross_entropy']
            best_actor = [p.detach().cpu().clone() for p in actor]
    if best_actor is not None:
        with torch.no_grad():
            for p, value in zip(actor, best_actor):
                p.copy_(value.to(p.device))
    # Re-enable normal SB3 PPO training, and reset Adam statistics after BC.
    for param in policy.parameters():
        param.requires_grad_(True)
    policy.optimizer = policy.optimizer_class(policy.parameters(),
                                               lr=model.lr_schedule(1.0),
                                               **policy.optimizer_kwargs)
    dest = a.root/'bc_runs'/('bc_'+datetime.now().strftime('%Y%m%d_%H%M%S'))
    dest.mkdir(parents=True, exist_ok=False)
    model.save(str(dest/'model'))
    report['validation_after'] = eval_loss(validation, validation_indices)
    report['saved_model'] = str(dest/'model.zip')
    (dest/'bc_report.json').write_text(json.dumps(report, indent=2)+'\n', encoding='utf-8')
    print('Saved new independent BC checkpoint:', dest/'model.zip')
    print('Use your normal deterministic 100-episode --mode eval; only continue PPO if gameplay improves.')


if __name__ == '__main__':
    main()
