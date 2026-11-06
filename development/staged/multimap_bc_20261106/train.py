"""Supervised BC phase on a checkpoint COPY; subsequent PPO has no BC loss."""
import argparse
from datetime import datetime
import json
from pathlib import Path
import time
import numpy as np
from common import ROOT,DATA,read,write_new,digest,contract,require_idle
from dataset import stacked


def mirror(obs,action):
    from development.experiments.road_vision.road_bc import mirror as old_mirror,SWAP
    result,label=old_mirror(obs,action)
    history=result['aux'][565:].reshape(32,7)
    history[:,:5]=history[:,SWAP].copy()
    return result,label


class Dataset:
    def __init__(self,folder,split):
        self.meta=read(folder/'dataset.json');self.arrays=[];self.samples=[]
        if self.meta.get('status')!='COMPLETE' or self.meta.get('schema')!='multimap-bc-fixed4-v1':
            raise ValueError('Incomplete/incompatible dataset')
        for run in self.meta['runs']:
            if run['split']!=split:continue
            path=folder/run['path']
            if not path.resolve().is_relative_to(folder.resolve()):raise ValueError('Invalid dataset path')
            for name,sha in run['hashes'].items():
                if name not in ('frames.npy','indices.npy','aux.npy','actions.npy','samples.json') or digest(path/name)!=sha:
                    raise ValueError('Dataset checksum mismatch')
            arrays={name:np.load(path/(name+'.npy'),mmap_mode='r',allow_pickle=False)
                    for name in ('frames','indices','aux','actions')}
            n=run['count'];c=self.meta['contract']
            if (arrays['frames'].dtype!=np.uint8 or arrays['frames'].shape[1:]!=(3,*c['image_shape'][-2:])
                    or arrays['indices'].dtype.kind not in 'iu' or arrays['actions'].dtype.kind not in 'iu'
                    or arrays['indices'].shape!=(n,4) or arrays['aux'].shape!=(n,789)
                    or arrays['actions'].shape!=(n,) or not np.isin(arrays['actions'],range(5)).all()
                    or not np.isfinite(arrays['aux']).all() or np.any(arrays['aux']<0) or np.any(arrays['aux']>1)
                    or np.any(arrays['indices'] < -1) or np.any(arrays['indices']>=len(arrays['frames']))):
                raise ValueError('Dataset observation shape/range mismatch')
            number=len(self.arrays);self.arrays.append(arrays)
            self.samples.extend((number,i,run['map'],run['session']) for i in range(n))
        if not self.samples:raise ValueError('No '+split+' samples')

    def sample(self,k):
        r,i,_,_=self.samples[k];a=self.arrays[r]
        return dict(image=stacked(a['frames'],a['indices'][i]),aux=a['aux'][i].copy()),int(a['actions'][i])

    def close(self):
        for run in self.arrays:
            for a in run.values():a._mmap.close()


def split_check(training,validation):
    if {s[3] for s in training.samples}&{s[3] for s in validation.samples}:
        raise ValueError('Recording session appears in both splits')
    for dataset in (training,validation):
        if {s[2] for s in dataset.samples}!={'halloween','tokyo'}:
            raise ValueError('Both maps require separately recorded training AND validation data')


def evaluate(policy,dataset,batch_size):
    import torch
    metrics={name:dict(counts=np.zeros(5,np.int64),correct=np.zeros(5,np.int64),loss=np.zeros(5))
             for name in ('halloween','tokyo')}
    policy.set_training_mode(False)
    with torch.no_grad():
        for start in range(0,len(dataset.samples),batch_size):
            ids=list(range(start,min(start+batch_size,len(dataset.samples))))
            samples=[dataset.sample(i) for i in ids]
            obs={k:torch.as_tensor(np.stack([s[0][k] for s in samples]),device=policy.device) for k in ('image','aux')}
            labels=torch.as_tensor([s[1] for s in samples],device=policy.device)
            dist=policy.get_distribution(obs);loss=-dist.log_prob(labels).cpu().numpy()
            pred=dist.distribution.probs.argmax(dim=1).cpu().numpy()
            for j,i in enumerate(ids):
                action=samples[j][1];m=metrics[dataset.samples[i][2]]
                m['counts'][action]+=1;m['correct'][action]+=int(pred[j]==action);m['loss'][action]+=loss[j]
    result={}
    for name,m in metrics.items():
        count=m['counts'];present=count>0
        result[name]=dict(samples=int(count.sum()),accuracy=float(m['correct'].sum()/count.sum()),
            per_action_counts=count.tolist(),per_action_recall=[float(m['correct'][i]/count[i]) if count[i] else None for i in range(5)],
            cross_entropy=float(m['loss'].sum()/count.sum()),
            macro_cross_entropy=float(np.mean(m['loss'][present]/count[present])))
    return dict(maps=result,selection_loss=float(np.mean([m['macro_cross_entropy'] for m in result.values()])))


def fit(model,training,validation,output,*,epochs=20,batch_size=16,lr=1e-4,patience=5,seed=123,minutes=60,mirror_probability=.5):
    import torch
    split_check(training,validation);rng=np.random.default_rng(seed);torch.manual_seed(seed)
    policy=model.policy
    frozen={}
    for name,p in policy.named_parameters():
        critic=name.startswith(('value_net.','mlp_extractor.value_net.'))
        p.requires_grad_(not critic)
        if critic:frozen[name]=p.detach().cpu().clone()
    optimizer=torch.optim.Adam([p for p in policy.parameters() if p.requires_grad],lr=lr)
    baseline=evaluate(policy,validation,batch_size);best=baseline['selection_loss'];best_epoch=0;stale=0
    best_state={k:v.detach().cpu().clone() for k,v in policy.state_dict().items()}
    history=[];began=time.monotonic()
    groups={name:[i for i,s in enumerate(training.samples) if s[2]==name] for name in ('halloween','tokyo')}
    counts=np.bincount([int(a) for run in training.arrays for a in run['actions']],minlength=5)
    if np.any(counts==0):raise ValueError('Training data must include all five actions')
    # Moderate inverse-square-root weighting, capped at 5x. Natural validation
    # metrics are always reported alongside macro metrics so NONE cannot hide errors.
    weights=np.minimum(5.,np.sqrt(counts.max()/counts));weights/=weights.mean()
    class_weights=torch.tensor(weights,dtype=torch.float32,device=policy.device)
    def save_best():
        current={k:v.detach().cpu().clone() for k,v in policy.state_dict().items()}
        policy.load_state_dict(best_state)
        model._last_obs=None
        # A distinct BC optimizer trained features; stale PPO moments are discarded.
        policy.optimizer.state.clear()
        model.save(output/'best.partial.zip')
        (output/'best.partial.zip').replace(output/'bc_best.zip')
        policy.load_state_dict(current)
    save_best();status='INTERRUPTED_OR_FAILED'
    try:
        for epoch in range(epochs):
            if time.monotonic()-began>=minutes*60:break
            policy.set_training_mode(True);total=0.;updates=0
            for _ in range((len(training.samples)+batch_size-1)//batch_size):
                if time.monotonic()-began>=minutes*60:break
                data=[]
                for j in range(batch_size):
                    group=groups['halloween' if j%2==0 else 'tokyo']
                    sample=training.sample(int(rng.choice(group)))
                    if rng.random()<mirror_probability:sample=mirror(*sample)
                    data.append(sample)
                obs={k:torch.as_tensor(np.stack([s[0][k] for s in data]),device=policy.device) for k in ('image','aux')}
                labels=torch.as_tensor([s[1] for s in data],device=policy.device)
                loss=(-policy.get_distribution(obs).log_prob(labels)*class_weights[labels]).mean()
                optimizer.zero_grad(set_to_none=True);loss.backward()
                torch.nn.utils.clip_grad_norm_([p for p in policy.parameters() if p.requires_grad],1.)
                optimizer.step();total+=float(loss.detach());updates+=1
            metrics=evaluate(policy,validation,batch_size)
            if metrics['selection_loss']<best-1e-5:
                best=metrics['selection_loss'];best_epoch=epoch+1;stale=0
                best_state={k:v.detach().cpu().clone() for k,v in policy.state_dict().items()};save_best()
            else:stale+=1
            row=dict(epoch=epoch+1,updates=updates,training_loss=total/max(1,updates),**metrics)
            history.append(row)
            with (output/'epochs.jsonl').open('a') as f:f.write(json.dumps(row)+'\n')
            print(f'[BC] epoch {epoch+1}/{epochs}; validation={metrics["selection_loss"]:.4f}; best={best:.4f}; stale={stale}/{patience}',flush=True)
            if stale>=patience:break
        status='COMPLETE'
    finally:
        policy.load_state_dict(best_state)
        for name,p in policy.named_parameters():
            if name in frozen and not torch.equal(p.detach().cpu(),frozen[name]):raise RuntimeError('Critic head changed in BC')
            p.requires_grad_(True)
        policy.optimizer.state.clear()
        write_new(output/'bc_report.json',dict(status=status,baseline=baseline,best_epoch=best_epoch,best_loss=best,
            epochs=history,training_action_counts=counts.tolist(),class_weights=weights.tolist(),
            shared_encoder_updated=True,critic_head_weights_fixed=True,
            caveat='Shared features change critic predictions despite fixed critic-head weights; evaluate before PPO.',
            ppo_continuation='ordinary PPO only; no BC loss or demo replay; optimizer moments reset'))


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--dataset',type=Path,required=True);p.add_argument('--checkpoint',type=Path,required=True)
    p.add_argument('--epochs',type=int,default=20);p.add_argument('--minutes',type=float,default=60)
    p.add_argument('--batch-size',type=int,default=16);p.add_argument('--lr',type=float,default=1e-4)
    p.add_argument('--patience',type=int,default=5);p.add_argument('--no-mirror',action='store_true')
    p.add_argument('--device',choices=['cuda','cpu'],default='cuda');p.add_argument('--output',type=Path)
    a=p.parse_args();require_idle()
    if not(1<=a.epochs<=1000 and 1<=a.minutes<=1440 and 2<=a.batch_size<=128 and 0<a.lr<=.001 and a.patience>=1):p.error('Invalid training bounds')
    dataset_meta=read(a.dataset/'dataset.json')
    if dataset_meta['contract']!=contract(a.checkpoint):p.error('Dataset/checkpoint contract mismatch')
    import sys
    sys.path.insert(0,str(ROOT/'development/experiments/command_history'))
    import command_adapter
    command_adapter.install()
    import torch
    from stable_baselines3 import PPO
    torch.set_num_threads(1)
    source_hash=digest(a.checkpoint)
    training=Dataset(a.dataset,'train');validation=Dataset(a.dataset,'validation')
    output=a.output or ROOT/'reports'/f'multimap_bc_{datetime.now():%Y%m%d_%H%M%S_%f}'
    output.mkdir(parents=True,exist_ok=False)
    write_new(output/'plan.json',dict(checkpoint=str(a.checkpoint.resolve()),source_sha256=source_hash,
        dataset=str(a.dataset.resolve()),dataset_sha256=digest(a.dataset/'dataset.json'),arguments={k:str(v) for k,v in vars(a).items()}))
    try:
        model=PPO.load(a.checkpoint,device=a.device)
        fit(model,training,validation,output,epochs=a.epochs,batch_size=a.batch_size,lr=a.lr,
            patience=a.patience,minutes=a.minutes,mirror_probability=0 if a.no_mirror else .5)
    finally:
        training.close();validation.close()
        if digest(a.checkpoint)!=source_hash:raise RuntimeError('Source checkpoint changed')
    print('BC CHECKPOINT:',output/'bc_best.zip')
    print('Evaluate both maps first; then pass this checkpoint to the normal multimap resume.ps1. No BC runs during PPO.')


if __name__=='__main__':main()
