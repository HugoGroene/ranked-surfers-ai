"""Explicit v1-to-v2 pause semantics migration; no tensor/optimizer changes."""
import argparse
import copy
import hashlib
import json
from pathlib import Path
from command_adapter import ROOT,install,contract


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--source',type=Path,required=True);p.add_argument('--output',type=Path,required=True)
    a=p.parse_args();source=a.source.resolve();output=a.output.resolve()
    if output.name!='latest.zip' or output.exists() or not output.is_relative_to(ROOT/'reports'):
        p.error('Use a new reports/.../latest.zip')
    install()
    from tools.ppo_observation_contract import validate_checkpoint,attach_contract
    from tools.ppo_atomic_checkpoint_v176 import save_atomic_checkpoint
    from tools.ppo_run_integrity import atomic_json
    from stable_baselines3 import PPO
    import torch
    requested=contract();previous=copy.deepcopy(requested)
    previous.pop('history_pause')
    previous['command_history'].update(version=1,resets='episode_and_optimizer_pause_resync')
    previous['command_history'].pop('pause')
    validate_checkpoint(source,previous)
    digest=hashlib.sha256(source.read_bytes()).hexdigest()
    model=PPO.load(source,device='cpu')
    weights=copy.deepcopy(model.policy.state_dict());optimizer=copy.deepcopy(model.policy.optimizer.state_dict())
    attach_contract(model,requested)
    model.history_pause_migration=dict(source=str(source),sha256=digest,change='verified_pause_history_retention_only')
    output.parent.mkdir(parents=True,exist_ok=True)
    save_atomic_checkpoint(model,output);validate_checkpoint(output,requested)
    loaded=PPO.load(output,device='cpu')
    for key,value in weights.items():torch.testing.assert_close(value,loaded.policy.state_dict()[key],rtol=0,atol=0)
    new=loaded.policy.optimizer.state_dict()
    assert optimizer['param_groups']==new['param_groups']
    for ident,state in optimizer['state'].items():
        for key,value in state.items():
            if torch.is_tensor(value):torch.testing.assert_close(value,new['state'][ident][key],rtol=0,atol=0)
            else:assert value==new['state'][ident][key]
    assert loaded.num_timesteps==model.num_timesteps and loaded._n_updates==model._n_updates
    assert hashlib.sha256(source.read_bytes()).hexdigest()==digest
    report=dict(status='PASS',**model.history_pause_migration,checkpoint=str(output),
        weights_and_optimizer_identical=True,num_timesteps=model.num_timesteps,n_updates=model._n_updates,
        source_unchanged=True,live_tested=False)
    atomic_json(output.with_suffix('.migration.json'),report);print(json.dumps(report,indent=2))


if __name__=='__main__':main()
