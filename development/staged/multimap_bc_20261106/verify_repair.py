"""Independent read-only checkpoint/held-out verification, including reporting recovery."""
import argparse
from pathlib import Path
import time
import numpy as np
from common import ROOT,read,digest,write_new,require_idle,contract
from retry_data import BoundedDataset
from actor_patch import features,metrics
from repair_actor import verify_used,acceptance,saturation
from development.staged.multimap_bc_20261106.normalized_actor import NormalizedActorPolicy


def main():
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('--run',type=Path,required=True)
    a=p.parse_args();require_idle();started=time.monotonic();deadline=started+100
    plan=read(a.run/'plan.json');source=Path(plan['arguments']['checkpoint']);dataset=Path(plan['arguments']['dataset'])
    candidate=a.run/'bc_best.zip'
    if digest(source)!=plan['source_sha256'] or digest(dataset/'dataset.json')!=plan['dataset_sha256']:
        raise ValueError('Source/export provenance mismatch')
    candidate_sha=digest(candidate)
    if contract(source)!=contract(candidate):raise ValueError('Observation contract changed')
    validation=BoundedDataset(dataset,'validation',verify=False)
    try:
        ids=list(range(len(validation.samples)));verify_used(validation,ids,deadline)
        import sys
        sys.path.insert(0,str(ROOT/'development/experiments/command_history'))
        import command_adapter
        command_adapter.install()
        import torch
        from stable_baselines3 import PPO
        torch.set_num_threads(1)
        device='cuda' if torch.cuda.is_available() else 'cpu'
        original=PPO.load(source,device=device);model=PPO.load(candidate,device=device)
        if not isinstance(model.policy,NormalizedActorPolicy):raise ValueError('Wrong candidate policy class')
        unchanged=all(torch.equal(t,model.policy.state_dict()[k]) for k,t in original.policy.state_dict().items()
                      if not k.startswith(('mlp_extractor.policy_net.','action_net.')))
        if not unchanged:raise ValueError('Encoder/critic changed')
        if not all(p.requires_grad for p in model.policy.parameters()):raise ValueError('PPO parameters frozen')
        cache=features(original.policy,validation,ids,deadline)
        base=metrics(original.policy,cache);current=metrics(model.policy,cache)
        raw_weights=read(a.run/'loss_weights.json')
        weights=None if raw_weights is None else {(k.split('/')[0],int(k.split('/')[1])):v for k,v in raw_weights.items()}
        passed,reasons=acceptance(current,base,weights)
        samples=[validation.sample(k)[0] for k in range(16)]
        obs={k:torch.as_tensor(np.stack([s[k] for s in samples]),device=device) for k in ('image','aux')}
        with torch.no_grad():
            critic_same=torch.equal(original.policy.predict_values(obs),model.policy.predict_values(obs))
            actual=model.policy.get_distribution(obs).distribution.probs
            from actor_patch import logits
            expected=logits(model.policy,cache['x'][:16]).softmax(-1)
            parity=bool(torch.allclose(actual,expected,atol=1e-6,rtol=1e-5))
        if not critic_same or not parity:raise ValueError('Live/cached actor or critic mismatch')
        result=dict(status='VERIFIED_FOR_GAMEPLAY_EVALUATION' if passed else 'REJECTED',
            candidate=str(candidate.resolve()),candidate_sha256=candidate_sha,source_sha256=plan['source_sha256'],
            source_validation=base,candidate_validation=current,reasons=reasons,
            frozen_state_unchanged=unchanged,critic_output_identical=critic_same,live_cached_policy_parity=parity,
            tanh_saturation=saturation(model.policy,cache),optimizer_state_empty=not model.policy.optimizer.state,
            validation_samples=len(ids),elapsed_seconds=time.monotonic()-started,
            note='Independent recomputation of the saved checkpoint; validation was used for selection. No gameplay tested.')
        if digest(candidate)!=candidate_sha or digest(source)!=plan['source_sha256']:raise ValueError('Checkpoint changed')
        write_new(a.run/'verified_candidate.json',result);print(result['status'],flush=True)
        print('REPORT:',a.run/'verified_candidate.json',flush=True)
        if not passed:raise SystemExit(1)
    finally:validation.close()


if __name__=='__main__':main()
