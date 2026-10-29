"""RAM-only balanced actor fitting diagnostic; never saves policy weights."""
import argparse
import hashlib
import json
from pathlib import Path

import numpy as np
import torch

from tools.train_demo_bc_cached import NAMES, actor_distribution, metrics


def digest(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def main():
    from stable_baselines3 import PPO
    import train_rs_v3_ppo  # noqa: F401 - saved extractor class

    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--experiment', type=Path, required=True)
    parser.add_argument('--output', type=Path, required=True)
    parser.add_argument('--lr', type=float, default=0.001)
    parser.add_argument('--max-updates', type=int, default=3000)
    args = parser.parse_args()
    if args.lr <= 0 or args.max_updates < 1:
        raise ValueError('Learning rate and update budget must be positive')
    if args.output.exists():
        raise ValueError('Use a new output directory')
    torch.set_num_threads(1)
    checkpoint = args.experiment/'latest.zip'
    cache = args.experiment/'training_features.npz'
    config = json.loads((args.experiment/'config.json').read_text())
    report = json.loads((args.experiment/'bc_report.json').read_text())
    source = Path(config['checkpoint'])
    hashes = {str(p): digest(p) for p in (checkpoint, cache, source)}
    assert digest(checkpoint) == report['output_sha256']
    assert digest(source) == config['source_sha256']
    with np.load(cache, allow_pickle=False) as archive:
        x, y, origins = archive['features'], archive['labels'], archive['origins']
    if x.shape != (len(y), 192) or not np.isfinite(x).all():
        raise ValueError('Invalid feature cache')
    if set(y.tolist()) != set(range(5)):
        raise ValueError('All five actions are required')
    args.output.mkdir(parents=True)
    result = {'purpose': 'Balanced training-subset fitting only; no generalization or gameplay claim',
              'checkpoint': str(checkpoint), 'input_sha256': hashes,
              'examples_per_action': 16, 'learning_rate': args.lr,
              'max_updates': args.max_updates, 'pass_accuracy': 0.95, 'pass_minimum_recall': 0.8,
              'weights_saved': False, 'trials': []}
    for seed in (77, 78, 79):
        torch.manual_seed(seed)
        rng = np.random.default_rng(seed)
        ids = np.concatenate([rng.choice(np.flatnonzero(y == action), 16, replace=False)
                              for action in range(5)])
        model = PPO.load(str(checkpoint), device='cpu')
        policy = model.policy
        original = {k: v.detach().clone() for k, v in policy.state_dict().items()}
        for parameter in policy.parameters():
            parameter.requires_grad_(False)
        actor = list(policy.mlp_extractor.policy_net.parameters()) + list(policy.action_net.parameters())
        for parameter in actor:
            parameter.requires_grad_(True)
        policy.eval()
        features = torch.as_tensor(x[ids])
        labels = torch.as_tensor(y[ids], dtype=torch.long)
        optimizer = torch.optim.Adam(actor, lr=args.lr)
        trial = {'seed': seed, 'cache_indices': ids.tolist(),
                 'origins': origins[ids].tolist(), 'before': metrics(policy, features, labels),
                 'history': []}
        for update in range(1, args.max_updates+1):
            loss = -actor_distribution(policy, features).log_prob(labels).mean()
            optimizer.zero_grad(set_to_none=True)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(actor, 1.)
            optimizer.step()
            if update == 1 or update % 100 == 0 or update == args.max_updates:
                score = metrics(policy, features, labels)
                trial['history'].append({'update': update, **score})
                passed = score['accuracy'] >= 0.95 and min(score['recall'].values()) >= 0.8
                print(f'[FIT] seed={seed} update={update} accuracy={score["accuracy"]:.3f} '
                      f'recall={score["recall"]}', flush=True)
                if passed:
                    break
        trial.update(updates=update, passed=passed, after=score)
        changed = [k for k, v in policy.state_dict().items() if not torch.equal(v, original[k])]
        assert all(k.startswith(('mlp_extractor.policy_net.', 'action_net.')) for k in changed)
        trial['changed_tensors'] = changed
        trial['frozen_tensors_unchanged'] = True
        result['trials'].append(trial)
        (args.output/'result.json').write_text(json.dumps(result, indent=2)+'\n')
        del model, policy, optimizer
    assert all(digest(Path(p)) == expected for p, expected in hashes.items())
    result['input_hashes_unchanged'] = True
    result['all_trials_passed'] = all(t['passed'] for t in result['trials'])
    (args.output/'result.json').write_text(json.dumps(result, indent=2)+'\n')
    print('[FIT RESULT]', result['all_trials_passed'], flush=True)


if __name__ == '__main__':
    main()
