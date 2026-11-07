"""Conservative actor-only BC; diagnostics never save a playable checkpoint."""
import argparse
import copy
from datetime import datetime
from pathlib import Path
import time
import numpy as np
from common import ROOT, ACTIONS, contract, digest, read, write_new, require_idle
from retry_data import BoundedDataset, BalancedSampler
from train import split_check


def freeze(policy):
    if not policy.share_features_extractor:
        raise ValueError('Expected shared feature extractor')
    actor = []
    for name, p in policy.named_parameters():
        enabled = name.startswith(('mlp_extractor.policy_net.', 'action_net.'))
        p.requires_grad_(enabled)
        if enabled: actor.append(p)
    if not actor: raise ValueError('No actor parameters')
    policy.set_training_mode(False)
    return actor


def logits(policy, features):
    return policy.action_net(policy.mlp_extractor.forward_actor(features))


def kl_old_new(old_log, new_log):
    return (old_log.exp() * (old_log-new_log)).sum(-1).clamp_min(0)


def features(policy, data, ids, deadline, output=None):
    import torch
    cached, labels, maps = [], [], []
    last = time.monotonic()
    with torch.no_grad():
        for offset in range(0, len(ids), 16):
            if time.monotonic() >= deadline: raise TimeoutError('Feature preparation deadline reached')
            selected = ids[offset:offset+16]
            samples = [data.sample(int(k)) for k in selected]
            obs = {key: torch.as_tensor(np.stack([s[0][key] for s in samples]), device=policy.device)
                   for key in ('image', 'aux')}
            cached.append(policy.extract_features(obs).detach().cpu())
            labels.extend(s[1] for s in samples)
            maps.extend(0 if data.samples[k][2] == 'halloween' else 1 for k in selected)
            if time.monotonic()-last > 10:
                print(f'[ACTOR CACHE] {offset+len(selected)}/{len(ids)}', flush=True)
                last = time.monotonic()
    data.drop_frames()
    result = dict(x=torch.cat(cached).to(policy.device),
                  y=torch.tensor(labels, device=policy.device),
                  maps=np.asarray(maps), ids=np.asarray(ids))
    with torch.no_grad():
        result['old_log'] = torch.cat([logits(policy, x).log_softmax(-1)
                                      for x in result['x'].split(1024)])
    return result


def metrics(policy, cache):
    import torch
    with torch.no_grad():
        log = torch.cat([logits(policy, x).log_softmax(-1) for x in cache['x'].split(1024)])
        loss = -log.gather(1, cache['y'][:, None]).squeeze(1).cpu().numpy()
        kl = kl_old_new(cache['old_log'], log).cpu().numpy()
        pred = log.argmax(-1).cpu().numpy(); labels = cache['y'].cpu().numpy()
    result = {}
    for number, name in enumerate(('halloween', 'tokyo')):
        choose = cache['maps'] == number
        confusion = np.zeros((5, 5), np.int64)
        np.add.at(confusion, (labels[choose], pred[choose]), 1)
        counts = confusion.sum(1)
        if np.any(counts == 0): raise ValueError('All actions required in validation')
        result[name] = dict(confusion=confusion.tolist(), recall=(confusion.diagonal()/counts).tolist(),
            counts=counts.tolist(), prediction_counts=confusion.sum(0).tolist(),
            macro_loss=float(np.mean([loss[choose & (labels == a)].mean() for a in range(5)])),
            natural_loss=float(loss[choose].mean()), mean_kl=float(kl[choose].mean()),
            p95_kl=float(np.percentile(kl[choose], 95)))
    return result


def gate(current, baseline, max_kl=.02, p95_kl=.1):
    """Conservative offline filters, never a claim of gameplay improvement."""
    reasons = []
    for name, m in current.items():
        b = baseline[name]
        if not np.isfinite([m['macro_loss'], m['natural_loss'], m['mean_kl'], m['p95_kl'], *m['recall']]).all():
            reasons.append(name+': nonfinite metrics'); continue
        if m['mean_kl'] > max_kl or m['p95_kl'] > p95_kl:
            reasons.append(name+': policy change exceeds KL budget')
        if m['macro_loss'] > b['macro_loss'] * .95:
            reasons.append(name+': less than 5% macro-loss improvement')
        for a, recall in enumerate(m['recall']):
            minimum = max(.1 if a == 0 else .05, b['recall'][a] - (.05 if a == 0 else .1))
            if recall < minimum:
                reasons.append(f'{name}: {ACTIONS[a]} recall {recall:.1%} below {minimum:.1%}')
    return not reasons, reasons


def importance_weights(data, movement_weight=3.):
    # Balanced proposal (50% NONE) for coverage, corrected back to each map's
    # actual label frequencies. Movement receives an explicit capped multiplier.
    sampler = BalancedSampler(data)
    result = {}
    proposal = np.array([.5, .125, .125, .125, .125])
    for name in ('halloween', 'tokyo'):
        counts = np.array([len(sampler.groups[name, a]) for a in range(5)], float)
        target = counts/counts.sum() * np.array([1., *([movement_weight]*4)])
        target /= target.sum()
        for a in range(5): result[name, a] = float(target[a]/proposal[a])
    return sampler, result


def contact(data, ids, output):
    """Small review sheet of the exact four RGB inputs, not resized source guesses."""
    from PIL import Image, ImageDraw
    canvas = Image.new('RGB', (720, len(ids)*148), 'white'); draw = ImageDraw.Draw(canvas)
    rows = []
    for row, k in enumerate(ids):
        obs, action = data.sample(k); run, index, name, session = data.samples[k]
        for frame in range(4):
            rgb = obs['image'][frame*3:frame*3+3].transpose(1, 2, 0)
            canvas.paste(Image.fromarray(rgb).resize((128,128)), (200+frame*128, row*148+18))
        draw.text((4, row*148+20), f'{name} {ACTIONS[action]}\nrun {run}, sample {index}', fill='black')
        ages = obs['aux'][565:].reshape(32,7)
        rows.append(dict(map=name, session=session, export_run=str(data.paths[run]), sample=index,
                         label=ACTIONS[action], history=ages[ages[:,6]>0].tolist()))
    canvas.save(output/'input_review.jpg')
    write_new(output/'input_review.json', rows)
    data.drop_frames()


def diagnostic(policy, training, output, deadline):
    import torch
    rng = np.random.default_rng(123); sampler = BalancedSampler(training)
    ids = [int(k) for pool in sampler.groups.values() for k in rng.choice(pool, 2, replace=False)]
    # Verify every pixel/aux file used by this short probe, without rereading 50GB.
    used = {training.samples[k][0] for k in ids}
    runs = [r for r in training.meta['runs'] if r['split'] == 'train']
    for r in used:
        for name, sha in runs[r]['hashes'].items():
            if time.monotonic() >= deadline: raise TimeoutError('Diagnostic verification deadline')
            if digest(training.paths[r]/name) != sha: raise ValueError('Dataset hash mismatch')
    contact(training, ids, output)
    cache = features(policy, training, ids, deadline)
    before = metrics(policy, cache)
    with torch.no_grad():
        activations=[]; x=cache['x']
        feature_summary=dict(mean_abs=float(x.abs().mean()),max_abs=float(x.abs().max()),
                             mean_std_across_examples=float(x.std(0).mean()))
        for layer in policy.mlp_extractor.policy_net:
            x=layer(x)
            activations.append(dict(layer=type(layer).__name__,mean_abs=float(x.abs().mean()),
                                    fraction_abs_over_099=float((x.abs()>.99).float().mean())))
    optimizer = torch.optim.Adam(freeze(policy), lr=1e-3)
    initial = float(torch.nn.functional.cross_entropy(logits(policy, cache['x']), cache['y']).detach())
    done = 0
    for step in range(5000):
        if time.monotonic() >= deadline: break
        loss = torch.nn.functional.cross_entropy(logits(policy, cache['x']), cache['y'])
        optimizer.zero_grad(set_to_none=True); loss.backward(); optimizer.step(); done = step+1
        if step % 1000 == 0:
            print(f'[MEMORIZATION ONLY] update={done}; loss={float(loss.detach()):.4f}', flush=True)
    after = metrics(policy, cache)
    with torch.no_grad():
        out = logits(policy, cache['x']); accuracy = float((out.argmax(-1)==cache['y']).float().mean())
        final = float(torch.nn.functional.cross_entropy(out, cache['y']))
    passed = accuracy >= .9 and final < initial*.2
    # Independent control: can these frozen features be memorized by a small,
    # well-scaled actor at all? This classifier is NEVER copied into the policy.
    torch.manual_seed(123)
    probe=torch.nn.Sequential(copy.deepcopy(policy.mlp_extractor.policy_net),
                              copy.deepcopy(policy.action_net)).to(policy.device)
    for layer in probe.modules():
        if isinstance(layer,torch.nn.Linear):
            torch.nn.init.orthogonal_(layer.weight, gain=1.)
            torch.nn.init.zeros_(layer.bias)
    normalized=(cache['x']-cache['x'].mean(0))/cache['x'].std(0).clamp_min(.1)
    probe_optimizer=torch.optim.Adam(probe.parameters(),lr=1e-3)
    probe_updates=0
    for step in range(1000):
        if time.monotonic()>=deadline: break
        loss=torch.nn.functional.cross_entropy(probe(normalized),cache['y'])
        probe_optimizer.zero_grad(set_to_none=True);loss.backward();probe_optimizer.step()
        probe_updates=step+1
    with torch.no_grad():
        output_logits=probe(normalized)
        probe_result=dict(updates=probe_updates,
            accuracy=float((output_logits.argmax(-1)==cache['y']).float().mean()),
            loss=float(torch.nn.functional.cross_entropy(output_logits,cache['y'])),
            scope='Fresh head plus standardized frozen features, memorization only; never saved as policy.')
    return dict(status='MEMORIZATION_PASS' if passed else 'MEMORIZATION_INCONCLUSIVE',
                updates=done, initial_loss=initial, final_loss=final, accuracy=accuracy,
                before=before, after=after, checkpoint_saved=False,
                initial_features=feature_summary, initial_actor_activations=activations,
                fresh_head_control=probe_result,
                note='Unconstrained memorization of 20 training examples; not generalization or gameplay evidence.')


def trial(model, training, validation, output, deadline, updates, movement_weight):
    import torch
    policy=model.policy
    print('[ACTOR CACHE] Frozen encoder: prepare features once, then train only small actor layers.', flush=True)
    train = features(policy, training, list(range(len(training.samples))), deadline)
    val = features(policy, validation, list(range(len(validation.samples))), deadline)
    baseline = metrics(policy, val); write_new(output/'baseline.json', baseline)
    sampler, weights = importance_weights(training, movement_weight)
    write_new(output/'sampling.json', {f'{m}/{ACTIONS[a]}':v for (m,a),v in weights.items()})
    actor=freeze(policy); optimizer=torch.optim.Adam(actor, lr=2e-5)
    rng=np.random.default_rng(123); history=[]; best=float('inf'); best_step=None
    anchor_pools={m:np.flatnonzero(train['maps']==m) for m in (0,1)}
    train_labels=train['y'].cpu().numpy()
    started=time.monotonic()
    for step in range(1, updates+1):
        if time.monotonic()>=deadline: break
        ids=sampler.batch(rng, 64)
        w=torch.tensor([weights[training.samples[k][2], int(train_labels[k])] for k in ids], device=policy.device)
        log=logits(policy, train['x'][ids]).log_softmax(-1)
        ce=-log.gather(1,train['y'][ids,None]).squeeze(1)
        # KL on an independent natural-frequency, equal-map anchor batch.
        anchors=[]
        for map_id in (0,1):
            anchors.extend(rng.choice(anchor_pools[map_id],32).tolist())
        now=logits(policy, train['x'][anchors]).log_softmax(-1)
        kl=kl_old_new(train['old_log'][anchors],now).mean()
        loss=(ce*w).mean()+kl
        if not torch.isfinite(loss): raise ValueError('Nonfinite objective')
        optimizer.zero_grad(set_to_none=True); loss.backward()
        torch.nn.utils.clip_grad_norm_(actor,1.); optimizer.step()
        if step%100==0 or step==updates or time.monotonic()>=deadline:
            current=metrics(policy,val); eligible,reasons=gate(current,baseline)
            score=sum(m['macro_loss'] for m in current.values())/2
            saved=eligible and score<best
            if saved:
                best=score; best_step=step
                for p in policy.parameters(): p.requires_grad_(True)
                # Only actor Adam moments are invalidated; frozen weights/moments stay intact.
                for p in actor: policy.optimizer.state.pop(p,None)
                model._last_obs=None
                model.save(output/'bc_best.partial.zip')
                (output/'bc_best.partial.zip').replace(output/'bc_best.zip')
                freeze(policy)
            history.append(dict(step=step, metrics=current, eligible=eligible, reasons=reasons, saved=saved))
            print(f'[ACTOR CHECK] {step}/{updates}; {time.monotonic()-started:.1f}s; '
                  f'macro_loss={score:.4f}; saved={saved}; '+ '; '.join(reasons), flush=True)
            if any(m['mean_kl']>.02 or m['p95_kl']>.1 for m in current.values()):
                print('[ACTOR STOP] Policy-change budget reached; no further updates.', flush=True); break
    return dict(status='CANDIDATE_SAVED' if best_step else 'NO_ACCEPTABLE_CANDIDATE',
                best_update=best_step, checks=history, baseline=baseline,
                note='Offline candidate only. Compare gameplay against unchanged source on both maps before PPO.')


def main():
    started=time.monotonic()
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--dataset',type=Path,required=True); p.add_argument('--checkpoint',type=Path,required=True)
    p.add_argument('--diagnostic',action='store_true'); p.add_argument('--seconds',type=float,default=100)
    p.add_argument('--minutes',type=float,default=10); p.add_argument('--updates',type=int,default=2000)
    p.add_argument('--movement-weight',type=float,default=3.)
    p.add_argument('--diagnostic-report',type=Path,
                   help='Required for a full trial: passing memorization report from the same source/export')
    a=p.parse_args(); require_idle()
    if not 1<=a.minutes<=30 or not 1<=a.updates<=5000 or not 1<=a.movement_weight<=4 or not 10<=a.seconds<=110:
        p.error('Invalid trial bounds')
    output=ROOT/'reports'/datetime.now().strftime('bc_actor_%Y%m%d_%H%M%S_%f')
    output.mkdir(); print('ACTOR BC REPORT:',output,flush=True)
    source=digest(a.checkpoint)
    if contract(a.checkpoint)!=read(a.dataset/'dataset.json')['contract']: raise ValueError('Contract mismatch')
    if not a.diagnostic:
        if a.diagnostic_report is None:
            p.error('Run --diagnostic first; a passing --diagnostic-report is required before the larger trial')
        prior=read(a.diagnostic_report);plan=read(a.diagnostic_report.parent/'plan.json')
        if (prior.get('status')!='MEMORIZATION_PASS' or not prior.get('frozen_state_unchanged')
                or not prior.get('source_unchanged') or plan['source_sha256']!=source
                or plan['dataset_sha256']!=digest(a.dataset/'dataset.json')):
            p.error('Diagnostic failed or belongs to another source/export; do not launch a longer BC trial')
    write_new(output/'plan.json',dict(arguments={k:str(v) for k,v in vars(a).items()},source_sha256=source,
        script_sha256=digest(Path(__file__)),dataset_sha256=digest(a.dataset/'dataset.json')))
    training=validation=None; model=None; result={}
    deadline=started+(a.seconds if a.diagnostic else a.minutes*60)
    try:
        training=BoundedDataset(a.dataset,'train',verify=not a.diagnostic)
        validation=BoundedDataset(a.dataset,'validation',verify=not a.diagnostic)
        split_check(training,validation)
        import sys
        sys.path.insert(0,str(ROOT/'development/experiments/command_history'))
        import command_adapter
        command_adapter.install()
        import torch
        from stable_baselines3 import PPO
        torch.set_num_threads(1); torch.manual_seed(123)
        model=PPO.load(a.checkpoint,device='cuda' if torch.cuda.is_available() else 'cpu')
        freeze(model.policy)
        fixed={n:t.detach().cpu().clone() for n,t in model.policy.state_dict().items()
               if not n.startswith(('mlp_extractor.policy_net.','action_net.'))}
        result=(diagnostic(model.policy,training,output,deadline) if a.diagnostic else
                trial(model,training,validation,output,deadline,a.updates,a.movement_weight))
        result['frozen_state_unchanged']=all(torch.equal(t,model.policy.state_dict()[n].detach().cpu()) for n,t in fixed.items())
        if not result['frozen_state_unchanged']: raise RuntimeError('Frozen encoder/critic changed')
    except TimeoutError as exc:
        result=dict(status='BUDGET_REACHED',reason=str(exc),checkpoint_saved=False)
    finally:
        if training is not None: training.close()
        if validation is not None: validation.close()
        if digest(a.checkpoint)!=source: raise RuntimeError('Source checkpoint changed')
        result.update(source_unchanged=True,elapsed_seconds=time.monotonic()-started)
        write_new(output/'actor_report.json',result)
    print(result['status'],flush=True)
    print('REPORT:',output/'actor_report.json',flush=True)
    if (output/'bc_best.zip').exists(): print('BC CANDIDATE:',output/'bc_best.zip',flush=True)


if __name__=='__main__': main()
