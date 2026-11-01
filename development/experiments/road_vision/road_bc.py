"""Optional whole-encoder BC on explicitly approved, compatible demonstrations."""
import json
import math
from pathlib import Path
import time
import numpy as np
from road_config import digest,write_json

SWAP=np.array([0,2,1,3,4],np.int64)  # Repository action order, NOT subwAI's order.

def inspect_demos(root,allow_downsampled=False):
    if not allow_downsampled:
        raise ValueError('BC requires --allow-downsampled-replays: stored 512x288 pixels cannot recover native detail')
    from rs_v61_features import FEATURE_VERSION,AUX_DIM,ACTION_NAMES
    found=[]
    for path in sorted((Path(root)/'runs').glob('demo_*/manifest.json')):
        m=json.loads(path.read_text(encoding='utf8'))
        if m.get('status')!='approved':continue
        if m.get('capture',{}).get('framing_profile','legacy')!='canvas_v1':continue
        if (m.get('schema')!=1 or m.get('feature_version')!=FEATURE_VERSION or m.get('aux_dim')!=AUX_DIM
            or m.get('image_shape')!=[3,288,512] or m.get('action_names')!=list(ACTION_NAMES)):
            raise ValueError(f'Incompatible approved recording: {path}')
        if not m.get('valid_segments_seconds'):raise ValueError('Approved run has no curated intervals')
        for name in ('frames.npz','aux.npy','actions.npy','timestamps.npy','valid.npy'):
            if not (path.parent/name).is_file():raise ValueError(f'Missing replay data: {path.parent/name}')
        found.append((path.parent,m))
    if len(found)<3:raise ValueError('Need at least three approved canvas_v1 runs; legacy crops are not automatically migrated')
    return found

def select_history(times,valid,segments,frames,span):
    """Indices/ages matching live selection, without crossing curated boundaries."""
    times=np.asarray(times,dtype=np.float64);valid=np.asarray(valid,dtype=bool)
    if times.ndim!=1 or times.shape!=valid.shape or not np.isfinite(times).all() or np.any(np.diff(times)<=0):
        raise ValueError('Invalid replay timestamps')
    indices=np.full((len(times),frames),-1,np.int64)
    ages=np.ones((len(times),frames),np.float32)
    chosen=np.zeros(len(times),bool)
    previous=-float('inf')
    for lo,hi in sorted(segments):
        if not np.isfinite([lo,hi]).all() or lo>=hi or lo<previous:raise ValueError('Invalid/overlapping curated intervals')
        previous=hi
        ids=np.flatnonzero(valid & (times>=lo) & (times<hi))
        for pos,i in enumerate(ids):
            past=ids[:pos+1];target=times[i]-np.linspace(span,0,frames)
            picks=np.searchsorted(times[past],target,side='right')-1;picks[-1]=pos
            used=set()
            for j,k in enumerate(picks):
                if k<0:continue
                selected=int(past[k])
                if selected in used or target[j]-times[selected]>.060000001:continue
                indices[i,j]=selected;used.add(selected)
                ages[i,j]=min(1.,max(0.,(times[i]-times[selected])/(span+.06)))
            chosen[i]=True
    return indices,ages,chosen

def mirror(obs,action):
    """Flip all temporal images and directional aux; timing/validity stay intact."""
    from rs_v6_ppo_features import HAZARD_DIM
    from rs_v61_features import V6_AUX_DIM,SCREEN_COLUMN_DIM
    aux=np.array(obs['aux'],copy=True)
    if aux.shape[0]<557 or np.any(aux[:HAZARD_DIM]) or np.any(aux[V6_AUX_DIM:V6_AUX_DIM+SCREEN_COLUMN_DIM]):
        raise ValueError('Mirroring requires detector-off auxiliary features')
    history=aux[HAZARD_DIM:HAZARD_DIM+20].reshape(4,5)
    history[:]=history[:,SWAP].copy()
    lane=V6_AUX_DIM+SCREEN_COLUMN_DIM
    aux[[lane,lane+2]]=aux[[lane+2,lane]]
    aux[[lane+5,lane+6]]=aux[[lane+6,lane+5]]  # five action ages after four lane fields
    return {'image':obs['image'][...,::-1].copy(),'aux':aux},int(SWAP[action])

class Dataset:
    def __init__(self,runs,config,cache):
        from road_crop import recorded_observation
        from tools.ppo_detector_off_features import detector_off_aux
        cache=Path(cache);cache.mkdir(parents=True,exist_ok=False)
        self.runs=[];self.samples=[];self.config=config;self.hashes={}
        for run_id,(folder,manifest) in enumerate(runs):
            for name in ('manifest.json','frames.npz','aux.npy','actions.npy','timestamps.npy','valid.npy'):
                self.hashes[str(folder/name)]=digest(folder/name)
            times=np.load(folder/'timestamps.npy',allow_pickle=False)
            valid=np.load(folder/'valid.npy',allow_pickle=False)
            actions=np.load(folder/'actions.npy',allow_pickle=False)
            aux=np.load(folder/'aux.npy',allow_pickle=False)
            if actions.shape!=times.shape or not np.issubdtype(actions.dtype,np.integer) or np.any((actions<0)|(actions>4)):
                raise ValueError('Invalid action labels')
            if aux.shape!=(len(times),557):raise ValueError('Invalid aux shape')
            aux=detector_off_aux(aux)
            if np.any((aux<0)|(aux>1)):raise ValueError('Aux features outside observation range')
            if valid.shape!=times.shape or np.any((valid!=0)&(valid!=1)):raise ValueError('Invalid validity mask')
            t=config['temporal'];indices,ages,chosen=select_history(times,valid,manifest['valid_segments_seconds'],t['frames'],t['span_ms']/1000)
            with np.load(folder/'frames.npz',allow_pickle=False) as z:frames=z['frames']
            if frames.shape!=(len(times),3,288,512) or frames.dtype!=np.uint8:raise ValueError('Invalid replay image schema')
            w,h=config['size'];dest=cache/f'{run_id}_frames.npy'
            small=np.lib.format.open_memmap(dest,mode='w+',dtype=np.uint8,shape=(len(times),3,h,w))
            for i in range(len(times)):small[i]=recorded_observation(frames[i],config)
            small.flush();del small,frames
            self.runs.append(dict(images=np.load(dest,mmap_mode='r'),aux=aux,indices=indices,ages=ages,actions=actions))
            self.samples.extend((run_id,int(i)) for i in np.flatnonzero(chosen))
        if not self.samples:raise ValueError('No valid curated samples')

    def sample(self,key):
        r,i=key;run=self.runs[r];idx=run['indices'][i];mask=idx>=0
        w,h=self.config['size'];frames=np.zeros((len(idx),3,h,w),np.uint8)
        frames[mask]=run['images'][idx[mask]]
        aux=np.concatenate((run['aux'][i],run['ages'][i],mask.astype(np.float32)))
        return {'image':frames.reshape(-1,h,w),'aux':aux},int(run['actions'][i])

    def label(self,key):
        r,i=key;return int(self.runs[r]['actions'][i])

    def close(self):
        for r in self.runs:r['images']._mmap.close()

def evaluate(policy,dataset,batch_size=32):
    import torch
    counts=np.zeros(5,np.int64);correct=np.zeros(5,np.int64);losses=np.zeros(5)
    policy.eval()
    with torch.no_grad():
        for start in range(0,len(dataset.samples),batch_size):
            data=[dataset.sample(k) for k in dataset.samples[start:start+batch_size]]
            obs={k:torch.as_tensor(np.stack([x[0][k] for x in data]),device=policy.device) for k in ('image','aux')}
            labels=torch.as_tensor([x[1] for x in data],device=policy.device)
            dist=policy.get_distribution(obs);nll=-dist.log_prob(labels)
            predicted=dist.distribution.probs.argmax(dim=1)
            for a in range(5):
                mask=labels==a;counts[a]+=int(mask.sum());correct[a]+=int(((predicted==labels)&mask).sum())
                losses[a]+=float(nll[mask].sum())
    return dict(samples=int(counts.sum()),accuracy=float(correct.sum()/counts.sum()),counts=counts.tolist(),
        recall=[float(correct[a]/counts[a]) if counts[a] else None for a in range(5)],
        cross_entropy=float(losses.sum()/counts.sum()),macro_cross_entropy=float(np.mean(losses[counts>0]/counts[counts>0])))

def fit(model,training,validation,epochs,seed,batch_size=32,lr=2e-4):
    """BC updates shared encoder + action head; critic head weights stay fixed."""
    import torch
    policy=model.policy;rng=np.random.default_rng(seed)
    for name,param in policy.named_parameters():
        param.requires_grad_(not (name.startswith('value_net.') or name.startswith('mlp_extractor.value_net.')))
    optimizer=torch.optim.Adam([p for p in policy.parameters() if p.requires_grad],lr=lr)
    groups=[[] for _ in range(5)]
    for key in training.samples:groups[training.label(key)].append(key)
    if any(not g for g in groups):raise ValueError('Training demonstrations must cover all five actions')
    baseline=evaluate(policy,validation,batch_size);best=baseline['macro_cross_entropy']
    best_state={k:v.detach().cpu().clone() for k,v in policy.state_dict().items()};history=[]
    for epoch in range(epochs):
        policy.train()
        for _ in range(math.ceil(len(training.samples)/batch_size)):
            data=[]
            for _ in range(batch_size):
                group=groups[int(rng.integers(5))];sample=training.sample(group[int(rng.integers(len(group)))])
                if rng.random()<.5:sample=mirror(*sample)
                data.append(sample)
            obs={k:torch.as_tensor(np.stack([x[0][k] for x in data]),device=policy.device) for k in ('image','aux')}
            labels=torch.as_tensor([x[1] for x in data],device=policy.device)
            loss=-policy.get_distribution(obs).log_prob(labels).mean()
            optimizer.zero_grad();loss.backward();torch.nn.utils.clip_grad_norm_(policy.parameters(),1.);optimizer.step()
        metrics=evaluate(policy,validation,batch_size);history.append(dict(epoch=epoch+1,**metrics))
        if metrics['macro_cross_entropy']<best:
            best=metrics['macro_cross_entropy'];best_state={k:v.detach().cpu().clone() for k,v in policy.state_dict().items()}
    policy.load_state_dict(best_state)
    for param in policy.parameters():param.requires_grad_(True)
    return best_state,dict(baseline=baseline,epochs=history,selected=evaluate(policy,validation,batch_size),
                          objective='uniform action sampling; train-only mirror; macro NLL holdout selection')

def pretrain(trainer,args,config,contract,root,output,epochs,allow_downsampled):
    import gymnasium as gym
    import torch
    from tools.ppo_atomic_checkpoint_v176 import save_atomic_checkpoint
    from tools.ppo_observation_contract import attach_contract
    runs=inspect_demos(root,allow_downsampled);rng=np.random.default_rng(args.seed);rng.shuffle(runs)
    n=max(1,round(len(runs)*.2));validation_runs,training_runs=runs[:n],runs[n:]
    output.mkdir(parents=True,exist_ok=False);started=time.perf_counter()
    torch.set_num_threads(1)
    class ShapeEnv(gym.Env):
        observation_space=gym.spaces.Dict({'image':gym.spaces.Box(0,255,tuple(contract['image_shape']),np.uint8),
            'aux':gym.spaces.Box(0,1,(contract['aux_dim'],),np.float32)})
        action_space=gym.spaces.Discrete(5)
    train=val=None
    try:
        train=Dataset(training_runs,config,output/'training_cache');val=Dataset(validation_runs,config,output/'holdout_cache')
        model=trainer.make_model(ShapeEnv(),args);attach_contract(model,contract)
        state,report=fit(model,train,val,epochs,args.seed)
        hashes={**train.hashes,**val.hashes}
        if any(digest(p)!=h for p,h in hashes.items()):raise RuntimeError('Demonstration changed during BC')
        report.update(status='COMPLETE',wall_seconds=time.perf_counter()-started,source_hashes=hashes,
            train_runs=[str(p) for p,m in training_runs],holdout_runs=[str(p) for p,m in validation_runs],
            trained_visual_encoder=True,critic_note='Head weights frozen; shared encoder changes can change value predictions',
            pixel_source='Already downsampled canvas_v1; not equivalent to native-resolution crop')
        save_atomic_checkpoint(model,output/'initialization.zip');write_json(output/'bc_report.json',report)
        return state
    finally:
        if train:train.close()
        if val:val.close()
