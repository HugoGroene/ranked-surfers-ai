"""Actor-only BC with fixed feature caching, whole-run holdout and early stopping.

Raw replay files and source checkpoint are never modified. Cache entries are
outputs of the unchanged frozen extractor, not a new observation schema.
"""
import argparse
from collections import Counter
import hashlib
import json
from pathlib import Path
import random
import time

import numpy as np

from rs_v612_demo_bc import Replay, approved_runs, sample_weights

NAMES = ['NONE', 'LEFT', 'RIGHT', 'JUMP', 'ROLL']


def selection_loss(score, metric):
    if metric == 'macro':
        values = list(score['per_action_cross_entropy'].values())
        if any(value is None for value in values):
            raise ValueError('Macro selection requires all five holdout actions')
        return sum(values)/len(values)
    return score['cross_entropy']


def actor_distribution(policy, features):
    latent = policy.mlp_extractor.forward_actor(features)
    return policy._get_action_dist_from_latent(latent)


def make_split(root, seed, holdout):
    found = approved_runs(root)
    if len(found) < 3:
        raise ValueError('At least three explicitly approved runs are required')
    random.Random(seed).shuffle(found)
    count = max(1, round(len(found)*holdout))
    return [Replay(*x) for x in found[count:]], [Replay(*x) for x in found[:count]]


def cache_features(model, runs, indices, batch_size, label):
    import torch
    features, labels, origins = [], [], []
    parity_checked = False
    began = time.perf_counter()
    by_run = {}
    for j, i in indices:
        by_run.setdefault(j, []).append(i)
    for count, (j, ids) in enumerate(by_run.items(), 1):
        replay = runs[j]
        replay.load()
        try:
            for start in range(0, len(ids), batch_size):
                batch_ids = ids[start:start+batch_size]
                data = [replay.sample(i) for i in batch_ids]
                obs = {'image': torch.as_tensor(np.stack([x[0] for x in data]), device=model.device),
                       'aux': torch.as_tensor(np.stack([x[1] for x in data]), device=model.device)}
                with torch.inference_mode():
                    extracted = model.policy.extract_features(obs)
                    if not parity_checked:
                        original = model.policy.get_distribution(obs).distribution.probs
                        cached = actor_distribution(model.policy, extracted).distribution.probs
                        torch.testing.assert_close(original, cached, rtol=1e-6, atol=1e-7)
                        parity_checked = True
                    features.append(extracted.cpu().numpy().copy())
                labels.extend(int(x[2]) for x in data)
                origins.extend((j, i) for i in batch_ids)
        finally:
            replay.release()
        if count % 10 == 0 or count == len(by_run):
            print(f'[BC CACHE] {label}: {count}/{len(by_run)} runs; '
                  f'{len(labels)} examples; {time.perf_counter()-began:.1f}s', flush=True)
    if not features or not parity_checked:
        raise ValueError('Empty feature cache')
    return np.concatenate(features), np.asarray(labels, dtype=np.int64), np.asarray(origins)


def read_feature_cache(directory, filename, runs, indices):
    """Reuse a completed cache only for the same run/sample selection."""
    with np.load(directory/filename, allow_pickle=False) as archive:
        x, y, origins = archive['features'], archive['labels'], archive['origins']
    if (len(x) != len(y) or len(y) != len(indices) or x.ndim != 2
            or x.shape[1] != 192 or not np.isfinite(x).all()
            or origins.shape != (len(indices), 2)):
        raise ValueError('Feature cache dimensions or values changed')
    if sorted(map(tuple, origins.tolist())) != sorted(indices):
        raise ValueError('Feature cache sample selection changed')
    if any(int(runs[int(j)].actions[int(i)]) != int(label)
           for (j, i), label in zip(origins, y)):
        raise ValueError('Feature cache labels changed')
    return x, y, origins


def metrics(policy, features, labels, batch_size=2048):
    import torch
    confusion = np.zeros((5, 5), dtype=np.int64)
    nll = np.zeros(5, dtype=np.float64)
    counts = np.bincount(labels.cpu().numpy(), minlength=5)
    policy.eval()
    with torch.inference_mode():
        for start in range(0, len(labels), batch_size):
            y = labels[start:start+batch_size]
            dist = actor_distribution(policy, features[start:start+batch_size])
            losses = -dist.log_prob(y)
            pred = dist.distribution.probs.argmax(dim=1).cpu().numpy()
            truth = y.cpu().numpy()
            np.add.at(confusion, (truth, pred), 1)
            for action in range(5):
                nll[action] += float(losses[y == action].sum().item())
    return {
        'cross_entropy': float(nll.sum()/len(labels)),
        'accuracy': float(confusion.trace()/len(labels)),
        'support': dict(zip(NAMES, map(int, counts))),
        'recall': {name: float(confusion[i, i]/counts[i]) if counts[i] else None
                   for i, name in enumerate(NAMES)},
        'per_action_cross_entropy': {name: float(nll[i]/counts[i]) if counts[i] else None
                                     for i, name in enumerate(NAMES)},
        'predicted_counts': dict(zip(NAMES, map(int, confusion.sum(axis=0)))),
        'confusion_matrix': confusion.tolist()}


def train(args):
    import torch
    from stable_baselines3 import PPO
    import train_rs_v3_ppo  # noqa: F401 - saved extractor class
    if args.output.exists():
        raise ValueError('Output directory already exists; use a unique name')
    torch.set_num_threads(1)
    torch.manual_seed(args.seed)
    np.random.seed(args.seed)
    source_sha = hashlib.sha256(args.checkpoint.read_bytes()).hexdigest()
    model = PPO.load(str(args.checkpoint), device='cuda' if torch.cuda.is_available() else 'cpu')
    policy = model.policy
    if not policy.share_features_extractor:
        raise ValueError('Only the existing shared frozen extractor is supported')
    if any(isinstance(layer, (torch.nn.modules.batchnorm._BatchNorm, torch.nn.Dropout))
           for layer in policy.modules()):
        raise ValueError('Stateful/stochastic layers require a separate cache-equivalence audit')
    original = {key: value.detach().cpu().clone() for key, value in policy.state_dict().items()}
    training, validation = make_split(args.root, args.seed, args.holdout_fraction)
    train_ids = sample_weights(training, none_fraction=args.none_fraction, seed=args.seed)
    validation_ids = [(j, int(i)) for j, replay in enumerate(validation) for i in replay.indices]
    counts = Counter(int(training[j].actions[i]) for j, i in train_ids)
    if not counts[1] or not counts[2]:
        raise ValueError('Training needs LEFT and RIGHT examples')
    validation_counts = Counter(int(validation[j].actions[i]) for j, i in validation_ids)
    if not validation_counts[1] or not validation_counts[2]:
        raise ValueError('Whole-run holdout needs LEFT and RIGHT examples')
    args.output.mkdir(parents=True)
    config = dict(vars(args), source_sha256=source_sha,
                  training_sampler='Globally shuffled cached examples each epoch',
                  checkpoint_selection=f'Lowest {args.selection_metric} holdout cross-entropy, including epoch zero',
                  training_runs=[str(r.folder) for r in training],
                  validation_runs=[str(r.folder) for r in validation])
    (args.output/'config.json').write_text(json.dumps(config, indent=2, default=str), encoding='utf8')
    print(f'[BC] {len(training)} training / {len(validation)} holdout runs; '
          f'{len(train_ids)} training / {len(validation_ids)} validation observations; device={model.device}', flush=True)
    for param in policy.parameters():
        param.requires_grad_(False)
    actor = list(policy.mlp_extractor.policy_net.parameters()) + list(policy.action_net.parameters())
    for param in actor:
        param.requires_grad_(True)
    policy.eval()
    if args.feature_cache:
        old_config = json.loads((args.feature_cache/'config.json').read_text())
        for key in ('source_sha256', 'training_runs', 'validation_runs'):
            if old_config[key] != config[key]:
                raise ValueError('Feature cache source/split changed: '+key)
        x_train, y_train, origins = read_feature_cache(
            args.feature_cache, 'training_features.npz', training, train_ids)
    else:
        x_train, y_train, origins = cache_features(model, training, train_ids, args.batch_size, 'train')
    np.savez(args.output/'training_features.npz', features=x_train, labels=y_train, origins=origins)
    if args.feature_cache:
        x_val, y_val, origins = read_feature_cache(
            args.feature_cache, 'validation_features.npz', validation, validation_ids)
        print('[BC CACHE] Reused verified source/split/sample feature caches.', flush=True)
    else:
        x_val, y_val, origins = cache_features(model, validation, validation_ids, args.batch_size, 'holdout')
    np.savez(args.output/'validation_features.npz', features=x_val, labels=y_val, origins=origins)
    x_train = torch.as_tensor(x_train, device=model.device)
    y_train = torch.as_tensor(y_train, device=model.device)
    x_val = torch.as_tensor(x_val, device=model.device)
    y_val = torch.as_tensor(y_val, device=model.device)
    baseline = metrics(policy, x_val, y_val)
    optimizer = torch.optim.Adam(actor, lr=args.lr)
    best_loss = selection_loss(baseline, args.selection_metric)
    best_actor = [p.detach().cpu().clone() for p in actor]
    best_epoch, stale = 0, 0
    report = {'source_sha256': source_sha, 'cache_probability_parity_verified': True,
              'training_before': metrics(policy, x_train, y_train),
              'validation_before': baseline, 'epochs': []}
    for epoch in range(1, args.max_epochs+1):
        policy.train()
        order = torch.randperm(len(y_train), device=model.device)
        total = 0.
        for start in range(0, len(order), args.batch_size):
            ids = order[start:start+args.batch_size]
            loss = -actor_distribution(policy, x_train[ids]).log_prob(y_train[ids]).mean()
            optimizer.zero_grad(set_to_none=True)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(actor, 1.)
            optimizer.step()
            total += float(loss.detach().item())*len(ids)
        score = metrics(policy, x_val, y_val)
        row = {'epoch': epoch, 'training_loss': total/len(y_train), 'validation': score}
        report['epochs'].append(row)
        objective = selection_loss(score, args.selection_metric)
        if objective < best_loss-1e-5:
            best_loss, best_epoch, stale = objective, epoch, 0
            best_actor = [p.detach().cpu().clone() for p in actor]
        else:
            stale += 1
        print(f'[BC EPOCH] {epoch}/{args.max_epochs} train={row["training_loss"]:.4f} '
              f'val={score["cross_entropy"]:.4f} best={best_epoch} '
              f'L_recall={score["recall"]["LEFT"]:.3f} R_recall={score["recall"]["RIGHT"]:.3f}', flush=True)
        (args.output/'progress.json').write_text(json.dumps(report, indent=2), encoding='utf8')
        if epoch >= args.min_epochs and stale >= args.patience:
            print('[BC] Early stopping: validation loss stopped improving.', flush=True)
            break
    with torch.no_grad():
        for param, value in zip(actor, best_actor):
            param.copy_(value.to(model.device))
    changed = [key for key, value in policy.state_dict().items()
               if not torch.equal(value.detach().cpu(), original[key])]
    if any(not key.startswith(('mlp_extractor.policy_net.', 'action_net.')) for key in changed):
        raise RuntimeError('A frozen policy tensor changed; refusing to save')
    if hashlib.sha256(args.checkpoint.read_bytes()).hexdigest() != source_sha:
        raise RuntimeError('Source checkpoint changed during training')
    report.update(best_epoch=best_epoch, training_after=metrics(policy, x_train, y_train),
                  validation_after=metrics(policy, x_val, y_val), changed_tensors=changed,
                  unchanged_tensors=len(original)-len(changed), actor_only_verified=True)
    for param in policy.parameters():
        param.requires_grad_(True)
    policy.optimizer = policy.optimizer_class(policy.parameters(), lr=model.lr_schedule(1.0),
                                             **policy.optimizer_kwargs)
    policy.eval()
    from tools.ppo_atomic_checkpoint_v176 import save_atomic_checkpoint
    save_atomic_checkpoint(model, args.output/'latest.zip')
    report['saved_model'] = str(args.output/'latest.zip')
    report['output_sha256'] = hashlib.sha256((args.output/'latest.zip').read_bytes()).hexdigest()
    (args.output/'bc_report.json').write_text(json.dumps(report, indent=2), encoding='utf8')
    print('[BC SAVED]', args.output/'latest.zip', flush=True)
    print('[BC RESULT]', json.dumps({key: report[key] for key in
          ['best_epoch', 'training_after', 'validation_after', 'actor_only_verified']}), flush=True)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--root', type=Path, required=True)
    parser.add_argument('--checkpoint', type=Path, required=True)
    parser.add_argument('--output', type=Path, required=True)
    parser.add_argument('--feature-cache', type=Path, default=None)
    parser.add_argument('--max-epochs', type=int, default=100)
    parser.add_argument('--min-epochs', type=int, default=10)
    parser.add_argument('--patience', type=int, default=10)
    parser.add_argument('--batch-size', type=int, default=12)
    parser.add_argument('--lr', type=float, default=2e-5)
    parser.add_argument('--selection-metric', choices=['natural', 'macro'], default='natural')
    parser.add_argument('--holdout-fraction', type=float, default=.20)
    parser.add_argument('--none-fraction', type=float, default=.35)
    parser.add_argument('--seed', type=int, default=77)
    args = parser.parse_args()
    if not (1 <= args.min_epochs <= args.max_epochs <= 200 and args.patience >= 1
            and 1 <= args.batch_size <= 64 and 0 < args.lr <= 1e-3
            and .05 <= args.holdout_fraction <= .5 and 0 < args.none_fraction < 1):
        parser.error('Invalid BC optimization settings')
    train(args)
