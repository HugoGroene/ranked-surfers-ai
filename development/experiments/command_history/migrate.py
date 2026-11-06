"""Expand a COPY of fixed4, retaining policy and Adam state; CPU-only verification."""
import argparse
import copy
import hashlib
import json
from pathlib import Path
import zipfile

from command_adapter import ROOT, base, install, contract
from history import EXTRA_DIM


def expanded_tensor(value, shape, name):
    import torch
    if tuple(value.shape) == tuple(shape):
        return value.clone()
    if not name.endswith('aux.0.weight') or tuple(value.shape) != (96,565) or tuple(shape) != (96,789):
        raise ValueError(f'Unexpected shape change: {name} {value.shape} -> {shape}')
    result = torch.zeros(shape, dtype=value.dtype, device=value.device)
    result[:,:565] = value
    return result


def adapt(model):
    import torch
    from gymnasium import spaces
    import numpy as np
    old_policy = model.policy
    old_params = list(old_policy.named_parameters())
    old_optimizer = copy.deepcopy(old_policy.optimizer.state_dict())
    old_state = old_policy.state_dict()
    old_space = model.observation_space
    assert old_space['aux'].shape == (565,) and old_space['image'].shape == (12,384,384)
    counters = (model.num_timesteps, model._n_updates)
    model.observation_space = spaces.Dict(dict(image=old_space['image'],
        aux=spaces.Box(0.,1.,(565+EXTRA_DIM,),np.float32)))
    model._setup_model()
    target = model.policy.state_dict()
    if set(target) != set(old_state):
        raise ValueError('Policy structure changed beyond input expansion')
    model.policy.load_state_dict({k:expanded_tensor(old_state[k],v.shape,k) for k,v in target.items()},strict=True)
    new_params = list(model.policy.named_parameters())
    if [n for n,_ in old_params] != [n for n,_ in new_params]:
        raise ValueError('Parameter order changed')
    # Verify optimizer IDs follow exactly the parameter order being migrated.
    flat_ids = [i for g in old_optimizer['param_groups'] for i in g['params']]
    flat_parameters = [p for g in old_policy.optimizer.param_groups for p in g['params']]
    if len(flat_ids) != len(old_params) or any(p is not old_params[i][1] for i,p in enumerate(flat_parameters)):
        raise ValueError('Unsupported optimizer parameter grouping')
    changed_moments = 0
    for ident,(name,param) in zip(flat_ids,new_params):
        for key,value in list(old_optimizer['state'].get(ident,{}).items()):
            if torch.is_tensor(value) and value.ndim:
                if value.shape != param.shape:
                    old_optimizer['state'][ident][key] = expanded_tensor(value,param.shape,name)
                    changed_moments += 1
    model.policy.optimizer.load_state_dict(old_optimizer)
    if counters != (model.num_timesteps, model._n_updates):
        raise ValueError('Training counters changed')
    model._last_obs = model._last_original_obs = model._last_episode_starts = None
    return old_policy, changed_moments


def prediction_check(old_policy, new_policy, observations):
    import numpy as np
    import torch
    from stable_baselines3.common.utils import obs_as_tensor
    old_policy.set_training_mode(False); new_policy.set_training_mode(False)
    rng = np.random.default_rng(123)
    max_probability = max_value = 0.
    with torch.no_grad():
        for obs in observations:
            old = {k:v[None] for k,v in obs.items()}
            # Nonzero arbitrary new inputs must still be ignored initially.
            extended = dict(old, aux=np.concatenate((old['aux'],rng.random((1,EXTRA_DIM),dtype=np.float32)),axis=1))
            t_old = obs_as_tensor(old,old_policy.device)
            t_new = obs_as_tensor(extended,new_policy.device)
            p = old_policy.get_distribution(t_old).distribution.probs
            q = new_policy.get_distribution(t_new).distribution.probs
            v = old_policy.predict_values(t_old); w = new_policy.predict_values(t_new)
            torch.testing.assert_close(p,q,rtol=1e-5,atol=1e-6)
            torch.testing.assert_close(v,w,rtol=1e-5,atol=1e-6)
            assert torch.equal(p.argmax(-1),q.argmax(-1))
            max_probability=max(max_probability,float((p-q).abs().max()))
            max_value=max(max_value,float((v-w).abs().max()))
    return dict(observations=len(observations),max_probability_error=max_probability,
        max_value_error=max_value,deterministic_actions_identical=True,
        device='cpu',note='Sampled observations, not a proof over every possible input')


def migrate(source, output, evidence):
    import numpy as np
    from stable_baselines3 import PPO
    from tools.ppo_observation_contract import validate_checkpoint, attach_contract
    from tools.ppo_atomic_checkpoint_v176 import save_atomic_checkpoint
    from tools.ppo_run_integrity import atomic_json
    if output.name != 'latest.zip' or output.exists() or not output.is_relative_to(ROOT/'reports'):
        raise ValueError('Use a new latest.zip under reports')
    before = hashlib.sha256(source.read_bytes()).hexdigest()
    base.install()
    validate_checkpoint(source, base.contract())
    model = PPO.load(source,device='cpu')
    install()
    old, moment_count = adapt(model)
    attach_contract(model,contract())
    observations=[]
    for file in sorted(evidence.rglob('*.npz')):
        with np.load(file) as archive:
            observations.append(dict(image=archive['image'].copy(),aux=archive['aux'].copy()))
    if not observations:
        raise ValueError('No real saved actor inputs found for parity checks')
    parity = prediction_check(old,model.policy,observations)
    output.parent.mkdir(parents=True,exist_ok=True)
    model.command_history_migration = dict(source=str(source),source_sha256=before,
        strategy='append_zero_weight_columns_preserve_adam_and_counters',new_aux_dim=789)
    save_atomic_checkpoint(model,output)
    validate_checkpoint(output,contract())
    loaded=PPO.load(output,device='cpu')
    reloaded_parity=prediction_check(old,loaded.policy,observations)
    with zipfile.ZipFile(output) as archive:
        if archive.testzip() is not None:raise ValueError('Saved checkpoint CRC failed')
    if hashlib.sha256(source.read_bytes()).hexdigest()!=before:
        raise RuntimeError('Source checkpoint changed')
    report=dict(status='PASS',**model.command_history_migration,
        source_unchanged=True,parity=parity,reloaded_parity=reloaded_parity,
        expanded_optimizer_moment_tensors=moment_count,
        num_timesteps=loaded.num_timesteps,n_updates=loaded._n_updates,
        additional_parameters=96*EXTRA_DIM,checkpoint=str(output),live_tested=False)
    atomic_json(output.with_suffix('.migration.json'),report)
    print(json.dumps(report,indent=2))


if __name__=='__main__':
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--source',type=Path,required=True)
    p.add_argument('--output',type=Path,required=True)
    p.add_argument('--evidence',type=Path,required=True)
    a=p.parse_args();migrate(a.source.resolve(),a.output.resolve(),a.evidence.resolve())
