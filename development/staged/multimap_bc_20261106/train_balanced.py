"""Short, bounded-memory BC retry. Existing exports/checkpoints are read-only."""
import argparse
from datetime import datetime
import json
from pathlib import Path
import time
import numpy as np
from common import ROOT, read, write_new, digest, contract, require_idle
from train import mirror, split_check
from retry_data import BoundedDataset, BalancedSampler, candidate_gate


def evaluate(policy, data, batch_size, *, progress=False):
    import torch
    metrics = {name: dict(confusion=np.zeros((5, 5), np.int64), loss=np.zeros(5),
                         pmin=np.ones(5), pmax=np.zeros(5)) for name in ('halloween', 'tokyo')}
    policy.set_training_mode(False)
    started = last_log = time.monotonic()
    with torch.no_grad():
        for start in range(0, len(data.samples), batch_size):
            ids = list(range(start, min(start+batch_size, len(data.samples))))
            samples = [data.sample(i) for i in ids]
            obs = {key: torch.as_tensor(np.stack([s[0][key] for s in samples]), device=policy.device)
                   for key in ('image', 'aux')}
            labels = torch.as_tensor([s[1] for s in samples], device=policy.device)
            dist = policy.get_distribution(obs)
            loss = -dist.log_prob(labels).cpu().numpy()
            probs = dist.distribution.probs.cpu().numpy()
            if not np.isfinite(loss).all() or not np.isfinite(probs).all():
                raise ValueError('Nonfinite validation output')
            for j, index in enumerate(ids):
                m = metrics[data.samples[index][2]]
                label = samples[j][1]
                m['confusion'][label, int(probs[j].argmax())] += 1
                m['loss'][label] += loss[j]
                m['pmin'] = np.minimum(m['pmin'], probs[j])
                m['pmax'] = np.maximum(m['pmax'], probs[j])
            if progress and time.monotonic()-last_log >= 10:
                print(f'[BC VALIDATE] {start+len(ids)}/{len(data.samples)} samples; '
                      f'{time.monotonic()-started:.0f}s', flush=True)
                last_log = time.monotonic()
    maps = {}
    for name, m in metrics.items():
        confusion = m['confusion']; counts = confusion.sum(1); present = counts > 0
        recalls = [float(confusion[i, i]/counts[i]) if counts[i] else None for i in range(5)]
        maps[name] = dict(samples=int(counts.sum()), accuracy=float(np.trace(confusion)/counts.sum()),
            per_action_counts=counts.tolist(), per_action_recall=recalls,
            prediction_counts=confusion.sum(0).tolist(), confusion=confusion.tolist(),
            mean_movement_recall=float(np.mean([v for v in recalls[1:] if v is not None])),
            cross_entropy=float(m['loss'].sum()/counts.sum()),
            macro_cross_entropy=float(np.mean(m['loss'][present]/counts[present])),
            max_probability_range=float(np.max(m['pmax']-m['pmin'])))
    return dict(maps=maps, selection_loss=float(np.mean([m['macro_cross_entropy'] for m in maps.values()])),
                validation_seconds=time.monotonic()-started)


def parameter_groups(policy, actor_lr, encoder_lr):
    groups = {'actor': [], 'encoder': []}; frozen = {}
    encoder_ids = {id(p) for p in policy.features_extractor.parameters()}
    for name, parameter in policy.named_parameters():
        critic = name.startswith(('value_net.', 'mlp_extractor.value_net.'))
        parameter.requires_grad_(not critic)
        if critic:
            frozen[name] = parameter.detach().cpu().clone()
        else:
            groups['encoder' if id(parameter) in encoder_ids else 'actor'].append(parameter)
    if not all(groups.values()):
        raise ValueError('Expected shared encoder and actor parameters')
    return [dict(params=groups['actor'], lr=actor_lr), dict(params=groups['encoder'], lr=encoder_lr)], frozen


def fit(model, training, validation, output, *, updates=1200, batch_size=16, minutes=10,
        validate_every=200, actor_lr=1e-5, encoder_lr=1e-6, seed=123, use_mirror=True):
    import torch
    split_check(training, validation)
    sampler = BalancedSampler(training)
    rng = np.random.default_rng(seed); torch.manual_seed(seed)
    policy = model.policy
    groups, frozen = parameter_groups(policy, actor_lr, encoder_lr)
    parameters = [p for group in groups for p in group['params']]
    optimizer = torch.optim.Adam(groups)
    print('[BC BASELINE] Evaluating unchanged PPO on the full validation sessions', flush=True)
    baseline = evaluate(policy, validation, batch_size, progress=True)
    validation.drop_frames()
    write_new(output/'baseline.json', baseline)
    best = baseline['selection_loss']; best_update = None
    history = []; used_labels = np.zeros(5, np.int64); completed = 0; total_loss = 0.
    started = time.monotonic(); last_print = started; status = 'FAILED'; reason = None
    print(f'[BC TRAIN] max {updates} updates / {minutes:g} minutes; actor lr={actor_lr:g}, '
          f'encoder lr={encoder_lr:g}; 50% NONE, 12.5% each movement; equal maps', flush=True)
    try:
        for step in range(1, updates+1):
            if time.monotonic()-started >= minutes*60:
                reason = 'time_budget'; break
            policy.set_training_mode(True)
            ids = sampler.batch(rng, batch_size)
            data = []
            for index in ids:
                sample = training.sample(index)
                if use_mirror and rng.random() < .5:
                    sample = mirror(*sample)
                data.append(sample)
            obs = {key: torch.as_tensor(np.stack([s[0][key] for s in data]), device=policy.device)
                   for key in ('image', 'aux')}
            labels = torch.as_tensor([s[1] for s in data], device=policy.device)
            # Sampling already balances actions; do not also multiply inverse-frequency weights.
            loss = -policy.get_distribution(obs).log_prob(labels).mean()
            optimizer.zero_grad(set_to_none=True); loss.backward()
            norm = torch.nn.utils.clip_grad_norm_(parameters, 1.)
            if not torch.isfinite(loss) or not torch.isfinite(norm):
                raise ValueError('Nonfinite BC gradient/loss')
            optimizer.step()
            completed = step; total_loss += float(loss.detach())
            used_labels += np.bincount([s[1] for s in data], minlength=5)
            if step == 1 or time.monotonic()-last_print >= 10:
                elapsed = time.monotonic()-started
                print(f'[BC PROGRESS] updates={step}/{updates}; elapsed={elapsed/60:.1f}m; '
                      f'budget_left={max(0,minutes-elapsed/60):.1f}m; '
                      f'mean_loss={total_loss/step:.4f}', flush=True)
                last_print = time.monotonic()
            if step % validate_every == 0 or step == updates or time.monotonic()-started >= minutes*60:
                training.drop_frames()
                metrics = evaluate(policy, validation, batch_size, progress=True)
                validation.drop_frames()
                eligible, failures = candidate_gate(metrics)
                improved = eligible and metrics['selection_loss'] < best-1e-5
                if improved:
                    best = metrics['selection_loss']; best_update = step
                    model._last_obs = None
                    policy.optimizer.state.clear()
                    # Save with normal PPO trainability; restore BC head freeze afterward.
                    for p in policy.parameters(): p.requires_grad_(True)
                    try:
                        model.save(output/'best.partial.zip')
                        (output/'best.partial.zip').replace(output/'bc_best.zip')
                    finally:
                        for name, p in policy.named_parameters(): p.requires_grad_(name not in frozen)
                row = dict(update=step, training_loss=total_loss/step, eligible=eligible,
                           rejection_reasons=failures, saved=improved, **metrics)
                history.append(row)
                with (output/'checks.jsonl').open('a', encoding='utf8') as f:
                    f.write(json.dumps(row)+'\n')
                movement = ' '.join(f'{name} movement_recall={m["mean_movement_recall"]:.1%}'
                                    for name, m in metrics['maps'].items())
                print(f'[BC CHECK] update={step}; loss={metrics["selection_loss"]:.4f}; '
                      f'{movement}; eligible={eligible}; saved={improved}', flush=True)
                if failures: print('[BC REJECT] '+'; '.join(failures), flush=True)
                constant = all(m['max_probability_range'] < 1e-5 for m in metrics['maps'].values())
                if constant:
                    reason = 'constant_policy_detected'; break
        status = 'CANDIDATE_SAVED' if best_update is not None else 'NO_ACCEPTABLE_CANDIDATE'
        reason = reason or ('update_budget' if completed == updates else 'time_budget')
    finally:
        for name, parameter in policy.named_parameters():
            if name in frozen and not torch.equal(parameter.detach().cpu(), frozen[name]):
                raise RuntimeError('Critic head changed during BC')
            parameter.requires_grad_(True)
        policy.optimizer.state.clear()
        write_new(output/'bc_report.json', dict(status=status, stop_reason=reason, updates=completed,
            elapsed_training_and_validation_seconds=time.monotonic()-started,
            baseline=baseline, best_update=best_update, best_loss=best if best_update is not None else None,
            training_action_counts=used_labels.tolist(), checks=history,
            validation_unchanged=True, shared_encoder_updated=True, critic_head_weights_fixed=True,
            frame_mapping_limit_per_split=training.max_frame_maps,
            caveat='Offline eligibility is not proof of better gameplay. Shared features change value predictions.',
            ppo_continuation='Evaluate both maps before ordinary PPO; no BC replay during PPO.'))
    return status


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--dataset', type=Path, required=True)
    p.add_argument('--checkpoint', type=Path, required=True)
    p.add_argument('--minutes', type=float, default=10)
    p.add_argument('--updates', type=int, default=1200)
    p.add_argument('--validate-every', type=int, default=200)
    p.add_argument('--batch-size', type=int, default=16)
    p.add_argument('--actor-lr', type=float, default=1e-5)
    p.add_argument('--encoder-lr', type=float, default=1e-6)
    p.add_argument('--no-mirror', action='store_true')
    p.add_argument('--device', choices=('cuda', 'cpu'), default='cuda')
    p.add_argument('--output', type=Path)
    a = p.parse_args(); require_idle()
    if (not 1 <= a.minutes <= 60 or not 1 <= a.updates <= 10000
            or not 1 <= a.validate_every <= a.updates
            or a.batch_size not in (16, 32, 64)
            or not 0 < a.encoder_lr <= a.actor_lr <= 1e-4):
        p.error('Invalid retry bounds')
    output = a.output or ROOT/'reports'/datetime.now().strftime('multimap_bc_balanced_%Y%m%d_%H%M%S_%f')
    output.mkdir(parents=True, exist_ok=False)
    print('BC RETRY OUTPUT:', output.resolve(), flush=True)
    print('Verifying existing export; no new recordings or export copy. Training budget starts after baseline validation.', flush=True)
    meta = read(a.dataset/'dataset.json')
    if meta['contract'] != contract(a.checkpoint):
        p.error('Checkpoint/export contract mismatch')
    source_hash = digest(a.checkpoint)
    write_new(output/'plan.json', dict(checkpoint=str(a.checkpoint.resolve()), source_sha256=source_hash,
        dataset=str(a.dataset.resolve()), dataset_sha256=digest(a.dataset/'dataset.json'),
        arguments={k:str(v) for k,v in vars(a).items()},
        sources={name:digest(Path(__file__).with_name(name)) for name in
                 ('train_balanced.py','retry_data.py','audit_retry.py','train.py','dataset.py','common.py')}))
    from audit_retry import audit
    audit_result = audit(a.dataset)
    write_new(output/'label_audit.json', audit_result)
    if not audit_result['passed']:
        raise ValueError('Label/frame audit failed')
    training = validation = None
    try:
        training = BoundedDataset(a.dataset, 'train')
        validation = BoundedDataset(a.dataset, 'validation')
        split_check(training, validation)
        import sys
        sys.path.insert(0, str(ROOT/'development/experiments/command_history'))
        import command_adapter
        command_adapter.install()
        import torch
        from stable_baselines3 import PPO
        torch.set_num_threads(1)
        model = PPO.load(a.checkpoint, device=a.device)
        status = fit(model, training, validation, output, updates=a.updates, minutes=a.minutes,
            batch_size=a.batch_size, validate_every=a.validate_every,
            actor_lr=a.actor_lr, encoder_lr=a.encoder_lr, use_mirror=not a.no_mirror)
    finally:
        if training is not None: training.close()
        if validation is not None: validation.close()
        if digest(a.checkpoint) != source_hash:
            raise RuntimeError('Source checkpoint changed')
    if status == 'CANDIDATE_SAVED':
        print('BC CANDIDATE:', output/'bc_best.zip', flush=True)
        print('Offline checks passed. Evaluate both maps before PPO; do not promote automatically.', flush=True)
    else:
        print('NO ACCEPTABLE BC CANDIDATE. Original PPO remains unchanged. Review:', output/'bc_report.json', flush=True)


if __name__ == '__main__':
    main()
