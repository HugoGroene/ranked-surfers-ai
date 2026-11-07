"""Bounded fresh-actor trial on frozen encoders; held-out recordings stay held out."""
import argparse
from datetime import datetime
from pathlib import Path
import time
import numpy as np
from common import ROOT, read, digest, write_new, contract, require_idle
from actor_patch import features, freeze, logits, metrics, importance_weights
from retry_data import BoundedDataset, BalancedSampler
from train import split_check
from development.staged.multimap_bc_20261106.normalized_actor import replace_actor


def select_training(data, per_group, seed=123, none_per_map=None):
    rng=np.random.default_rng(seed); sampler=BalancedSampler(data)
    # Keep all four movement classes where possible; cap NONE to keep the short
    # diagnostic bounded. Statistics for normalization use TRAINING only.
    return sorted(int(k) for (_,action),pool in sampler.groups.items()
                  for k in rng.choice(pool,min(none_per_map if action==0 and none_per_map is not None else per_group,
                                               len(pool)),replace=False))


def cache_groups(cache):
    labels=cache['y'].cpu().numpy()
    groups={(m,a):np.flatnonzero((cache['maps']==m)&(labels==a)) for m in (0,1) for a in range(5)}
    if any(not len(v) for v in groups.values()):raise ValueError('Both maps need all actions')
    return groups


def batch(groups,rng):
    # Exact 64-sample proposal: half NONE, 1/8 each movement, equal maps.
    ids=[]
    for m in (0,1):
        for a in range(5):ids.extend(rng.choice(groups[m,a],16 if a==0 else 4).tolist())
    rng.shuffle(ids)
    return ids


def acceptance(current,source,loss_weights=None):
    reasons=[]
    # Beat a constant classifier matching the training class prior, not merely
    # the degraded source. Also reject single-direction or always-NONE outputs.
    for name,m in current.items():
        prior=np.array([.5,.125,.125,.125,.125])
        if loss_weights is not None:prior*=np.array([loss_weights[name,a] for a in range(5)])
        prior/=prior.sum()
        constant_macro=float(-np.log(prior).mean())
        if not np.isfinite([m['macro_loss'],m['natural_loss'],*m['recall']]).all():
            reasons.append(name+': nonfinite');continue
        if m['macro_loss']>=min(source[name]['macro_loss'],constant_macro*.95):
            reasons.append(name+': insufficient held-out loss improvement')
        if m['recall'][0]<.5:reasons.append(name+': NONE recall below 50%')
        if min(m['recall'][1:])<.1:reasons.append(name+': at least one movement recall below 10%')
        c=np.asarray(m['confusion']); movement_predictions=c[:,1:].sum()
        precision=float(np.trace(c[1:,1:])/movement_predictions) if movement_predictions else 0.
        if precision<.05:reasons.append(name+': movement precision below 5%')
    return not reasons,reasons


def saturation(policy,cache):
    import torch
    result=[]
    with torch.no_grad():
        x=cache['x'][:2048]
        for layer in policy.mlp_extractor.policy_net:
            x=layer(x)
            if isinstance(layer,torch.nn.Tanh):
                result.append(float((x.abs()>.99).float().mean()))
    return result


def verify_used(data,ids,deadline):
    split='train' if data.samples[0][3] in {r['session'] for r in data.meta['runs'] if r['split']=='train'} else 'validation'
    runs=[r for r in data.meta['runs'] if r['split']==split]
    for n,r in enumerate(sorted({data.samples[k][0] for k in ids})):
        if n%25==0:print(f'[REPAIR VERIFY] {split} run {n+1}',flush=True)
        for name,expected in runs[r]['hashes'].items():
            if time.monotonic()>=deadline:raise TimeoutError('Verification budget reached')
            if digest(data.paths[r]/name)!=expected:raise ValueError('Export checksum mismatch')


def fit(model,train,val,output,deadline,updates=3000,loss_weights=None):
    import torch
    source=metrics(model.policy,val);source_saturation=saturation(model.policy,val)
    write_new(output/'source_validation.json',source)
    keep=replace_actor(model,train['x'])
    actor=freeze(model.policy)
    optimizer=torch.optim.Adam(actor,lr=1e-3)
    groups=cache_groups(train);rng=np.random.default_rng(123)
    label_array=train['y'].cpu().numpy()
    rows=[];best=float('inf');best_step=None;step=0
    try:
        for step in range(1,updates+1):
            if time.monotonic()>=deadline:break
            ids=batch(groups,rng)
            ce=torch.nn.functional.cross_entropy(logits(model.policy,train['x'][ids]),train['y'][ids],reduction='none')
            if loss_weights is not None:
                weight=torch.tensor([loss_weights['halloween' if train['maps'][k]==0 else 'tokyo',int(label_array[k])]
                                     for k in ids],device=model.device)
                loss=(ce*weight).mean()
            else:loss=ce.mean()
            optimizer.zero_grad(set_to_none=True);loss.backward()
            norm=torch.nn.utils.clip_grad_norm_(actor,1.)
            if not torch.isfinite(loss) or not torch.isfinite(norm):raise ValueError('Nonfinite update')
            optimizer.step()
            if step%250==0 or step==updates or time.monotonic()>=deadline:
                validation=metrics(model.policy,val);eligible,reasons=acceptance(validation,source,loss_weights)
                score=float(np.mean([v['macro_loss'] for v in validation.values()]))
                saved=bool(eligible and score<best)
                if saved:
                    best=float(score);best_step=step
                    for k,v in keep.items():
                        if not torch.equal(v,model.policy.state_dict()[k]):raise RuntimeError('Frozen state changed')
                    for p in model.policy.parameters():p.requires_grad_(True)
                    model.policy.optimizer.state.clear()
                    model.actor_repair=dict(version=1,training_update=step,normalization='fixed_training_feature_mean_std',
                        encoder_frozen_during_bc=True,critic_preserved=True,
                        requires_gameplay_evaluation=True)
                    model.save(output/'bc_best.partial.zip')
                    (output/'bc_best.partial.zip').replace(output/'bc_best.zip')
                    freeze(model.policy)
                rows.append(dict(update=step,metrics=validation,saved=saved,eligible=eligible,reasons=reasons))
                print(f'[REPAIR CHECK] update={step}; macro_loss={score:.4f}; saved={saved}; '+ '; '.join(reasons),flush=True)
    finally:
        unchanged=all(torch.equal(v,model.policy.state_dict()[k]) for k,v in keep.items())
        if not unchanged:raise RuntimeError('Encoder/critic changed')
    return dict(status='CANDIDATE_SAVED' if best_step else 'NO_ACCEPTABLE_CANDIDATE',
        updates=step,best_update=best_step,checks=rows,source_validation=source,
        source_tanh_saturation=source_saturation,final_tanh_saturation=saturation(model.policy,val),
        train_metrics=metrics(model.policy,train),frozen_state_unchanged=unchanged,
        train_samples=len(train['y']),validation_samples=len(val['y']),
        note='Fresh actor, no constraint to imitate the old policy. Same encoders/critic. '
             'Validation selects candidates and is not an independent test. Gameplay evaluation still required.')


def main():
    started=time.monotonic()
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--dataset',type=Path,required=True);p.add_argument('--checkpoint',type=Path,required=True)
    p.add_argument('--seconds',type=float,default=105);p.add_argument('--per-group',type=int,default=400)
    p.add_argument('--updates',type=int,default=3000)
    p.add_argument('--none-per-map',type=int,default=3000)
    p.add_argument('--movement-weight',type=float,default=3.,help='Multiply natural movement frequencies; 0 reproduces uncorrected balanced loss')
    a=p.parse_args();require_idle()
    if (not 15<=a.seconds<=3600 or not 2<=a.per_group<=100000 or not 1<=a.updates<=10000
            or not 2<=a.none_per_map<=100000 or not 0<=a.movement_weight<=4):p.error('Invalid limits')
    deadline=started+a.seconds
    output=ROOT/'reports'/datetime.now().strftime('bc_actor_repair_%Y%m%d_%H%M%S_%f')
    output.mkdir();print('REPAIR REPORT:',output,flush=True)
    source_sha=digest(a.checkpoint);meta=read(a.dataset/'dataset.json')
    if meta['contract']!=contract(a.checkpoint):raise ValueError('Observation contract mismatch')
    write_new(output/'plan.json',dict(arguments={k:str(v) for k,v in vars(a).items()},
        source_sha256=source_sha,dataset_sha256=digest(a.dataset/'dataset.json'),
        code_sha256={n:digest(Path(__file__).with_name(n)) for n in
                     ('repair_actor.py','normalized_actor.py','actor_patch.py','retry_data.py')}))
    training=validation=None;result=dict(status='FAILED')
    try:
        training=BoundedDataset(a.dataset,'train',verify=False)
        validation=BoundedDataset(a.dataset,'validation',verify=False);split_check(training,validation)
        ids=select_training(training,a.per_group,none_per_map=a.none_per_map);val_ids=list(range(len(validation.samples)))
        verify_used(training,ids,deadline);verify_used(validation,val_ids,deadline)
        import sys
        sys.path.insert(0,str(ROOT/'development/experiments/command_history'))
        import command_adapter
        command_adapter.install()
        import torch
        from stable_baselines3 import PPO
        torch.set_num_threads(1);torch.manual_seed(123)
        model=PPO.load(a.checkpoint,device='cuda' if torch.cuda.is_available() else 'cpu')
        freeze(model.policy)
        print(f'[REPAIR FEATURES] train={len(ids)} validation={len(val_ids)}',flush=True)
        train=features(model.policy,training,ids,deadline)
        val=features(model.policy,validation,val_ids,deadline)
        write_new(output/'selection.json',dict(train_ids=ids,validation='all separate-session examples'))
        weights=importance_weights(training,a.movement_weight)[1] if a.movement_weight else None
        write_new(output/'loss_weights.json', None if weights is None else {f'{m}/{act}':w for (m,act),w in weights.items()})
        result=fit(model,train,val,output,deadline,a.updates,weights)
        if (output/'bc_best.zip').exists():
            # Real checkpoint round trip with actor normalization and trainability.
            saved=PPO.load(output/'bc_best.zip',device=model.device)
            saved_metrics=metrics(saved.policy,val)
            if not acceptance(saved_metrics,result['source_validation'],weights)[0]:raise RuntimeError('Saved candidate failed gate')
            if not all(p.requires_grad for p in saved.policy.parameters()):raise RuntimeError('Saved PPO parameters frozen')
            result['saved_validation']=saved_metrics;result['checkpoint_reload_passed']=True
    except TimeoutError as exc:
        result=dict(status='BUDGET_REACHED',reason=str(exc))
    finally:
        if training is not None:training.close()
        if validation is not None:validation.close()
        if digest(a.checkpoint)!=source_sha:raise RuntimeError('Source checkpoint changed')
        result.update(source_unchanged=True,elapsed_seconds=time.monotonic()-started)
        write_new(output/'repair_report.json',result)
    print(result['status'],flush=True)
    print('REPORT:',output/'repair_report.json',flush=True)
    if result['status']=='CANDIDATE_SAVED':print('BC CANDIDATE:',output/'bc_best.zip',flush=True)


if __name__=='__main__':main()
