"""Bounded offline real-policy test: ordinary transfers vs reused pinned buffers."""
import argparse
from datetime import datetime
import hashlib
import json
from pathlib import Path
import sys
import time
import numpy as np

ROOT=Path(__file__).resolve().parents[3]
sys.path[:0]=[str(ROOT),str(ROOT/'development/experiments/command_history')]


def main():
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('--checkpoint',type=Path,required=True)
    a=p.parse_args();began=time.perf_counter()
    from command_adapter import install
    install()
    import torch
    from stable_baselines3 import PPO
    from stable_baselines3.common.utils import obs_as_tensor
    torch.set_num_threads(1)
    digest=hashlib.sha256(a.checkpoint.read_bytes()).hexdigest()
    model=PPO.load(a.checkpoint,device='cuda');model.policy.set_training_mode(False)
    rng=np.random.default_rng(73)
    obs=dict(image=rng.integers(0,256,(1,*model.observation_space['image'].shape),dtype=np.uint8),
             aux=rng.random((1,*model.observation_space['aux'].shape),dtype=np.float32))
    pinned={k:torch.empty(v.shape,dtype=torch.from_numpy(v).dtype,pin_memory=True) for k,v in obs.items()}
    arrays={k:v.numpy() for k,v in pinned.items()}
    device={k:torch.empty(v.shape,dtype=torch.from_numpy(v).dtype,device='cuda') for k,v in obs.items()}
    def standard():return obs_as_tensor(obs,'cuda')
    def reuse():
        for k in obs:
            np.copyto(arrays[k],obs[k]);device[k].copy_(pinned[k],non_blocking=True)
        return device
    with torch.no_grad():
        first=[v.cpu().numpy() for v in model.policy(standard(),deterministic=True)]
        other=[v.cpu().numpy() for v in model.policy(reuse(),deterministic=True)]
        exact=all(np.array_equal(x,y) for x,y in zip(first,other))
        if not exact:raise RuntimeError('Policy output parity failed')
        for _ in range(25):model.policy(standard(),deterministic=True)[0].cpu()
        blocks=[]
        for name in ('standard','pinned_reuse','pinned_reuse','standard'):
            fn=standard if name=='standard' else reuse;times=[]
            torch.cuda.synchronize()
            for i in range(200):
                if time.perf_counter()-began>100:raise TimeoutError('Offline benchmark time budget')
                start=time.perf_counter();result=model.policy(fn(),deterministic=True)
                result[0].cpu().numpy() # Existing decision synchronization, no extra per-step fence.
                times.append((time.perf_counter()-start)*1000)
            blocks.append(dict(method=name,count=len(times),median_ms=float(np.median(times)),
                               p95_ms=float(np.percentile(times,95))))
    if hashlib.sha256(a.checkpoint.read_bytes()).hexdigest()!=digest:raise RuntimeError('Checkpoint changed')
    report=dict(blocks=blocks,policy_outputs_exact=exact,source_unchanged=True,checkpoint=str(a.checkpoint),
        seconds=time.perf_counter()-began,pinned_bytes=sum(v.nbytes for v in obs.values()),
        note='One CUDA model, synthetic valid-shape inputs; complete transfer+actor/critic+action retrieval. '
             'No game, optimizer, workers or gameplay latency measured. No runtime change installed.')
    output=ROOT/'reports'/f'transfer_buffers_{datetime.now():%Y%m%d_%H%M%S_%f}.json'
    output.write_text(json.dumps(report,indent=2));print('REPORT:',output);print(json.dumps(report,indent=2))


if __name__=='__main__':main()
