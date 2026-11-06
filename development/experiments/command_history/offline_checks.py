"""Real checkpoint/Adam/shared-rollout/spawn tests without gameplay or GPU work."""
import argparse
import multiprocessing as mp
from pathlib import Path
import tempfile

from command_adapter import install, contract, base


def spawned(checkpoint,connection):
    try:
        # Same entry module imported by live Windows worker spawn.
        import os
        os.environ['RS_COMMAND_HISTORY_ACTIVE']='1'
        import run_training
        trainer=install()
        from tools.ppo_observation_contract import validate_checkpoint
        validate_checkpoint(checkpoint,contract())
        from tools.ppo_headless_env import install as headless
        headless(trainer,'electron',electron_transport='shared_memory')
        assert sum(c.__name__=='HeadlessEnv' for c in trainer.build_env_class().__mro__)==1
        assert contract()['aux_dim']==789
        connection.send(dict(status='PASS',contract=contract()))
    except BaseException as exc:connection.send(dict(status='FAIL',error=repr(exc)))
    finally:connection.close()


def check(source,checkpoint):
    import numpy as np
    import torch
    from stable_baselines3 import PPO
    from tools.ppo_observation_contract import validate_checkpoint
    from tools.ppo_settled_learning import TimedDictBuffer
    from tools.ppo_rollout_transport import RolloutSlot
    from tools.ppo_parallel_learning import merge_rollouts,configure_learner
    from tools.ppo_atomic_checkpoint_v176 import save_atomic_checkpoint
    from history import EXTRA_DIM
    base.install();old=PPO.load(source,device='cpu')
    install();validate_checkpoint(checkpoint,contract());model=PPO.load(checkpoint,device='cpu')
    assert model.num_timesteps==old.num_timesteps and model._n_updates==old._n_updates
    assert model.policy.pi_features_extractor is model.policy.vf_features_extractor
    assert model.policy.share_features_extractor
    for (name,op),(new_name,np_) in zip(old.policy.named_parameters(),model.policy.named_parameters()):
        assert name==new_name
        if op.shape==np_.shape:torch.testing.assert_close(op,np_,rtol=0,atol=0)
        else:
            assert name.endswith('aux.0.weight')
            torch.testing.assert_close(op,np_[:,:565],rtol=0,atol=0)
            assert not np_[:,565:].detach().any()
        old_state=old.policy.optimizer.state[op];new_state=model.policy.optimizer.state[np_]
        assert set(old_state)==set(new_state)
        for key,value in old_state.items():
            new=new_state[key]
            if torch.is_tensor(value):
                if value.shape!=new.shape:
                    torch.testing.assert_close(value,new[:,:565],rtol=0,atol=0)
                    assert not new[:,565:].any()
                else:torch.testing.assert_close(value,new,rtol=0,atol=0)
            else:assert value==new
    try:validate_checkpoint(source,contract())
    except ValueError:pass
    else:raise AssertionError('Old checkpoint incorrectly accepted new input contract')
    owners=[];packets=[]
    with tempfile.TemporaryDirectory() as tmp:
        model.n_steps=2;model.batch_size=2;model.n_epochs=1
        configure_learner(model,Path(tmp),4,.05)
        try:
            for worker in range(4):
                buffer=TimedDictBuffer(2,model.observation_space,model.action_space,device='cpu')
                buffer.observations['image'][:]=worker+1
                buffer.observations['aux'][:]=.1*(worker+1)
                buffer.active_seconds[:]=.05;buffer.actor_masks[:]=1
                buffer.returns[:]=.5;buffer.advantages[:]=.25
                buffer.full=True;buffer.pos=2
                owner=RolloutSlot(buffer);client=RolloutSlot(descriptor=owner.descriptor)
                owners.append((owner,client))
                message=client.publish(buffer,worker,0,'offline')
                packets.append(owner.packet(message,0,'offline'))
            merge_rollouts(model.rollout_buffer,packets,2,0,'offline')
            for worker in range(4):
                np.testing.assert_allclose(model.rollout_buffer.observations['aux'][2*worker:2*worker+2],.1*(worker+1))
            model.train()
            assert model._settled_stats['optimized_transitions']==8
            assert model.policy.features_extractor.aux[0].weight[:,565:].detach().abs().max()>0
            saved=Path(tmp)/'latest.zip';save_atomic_checkpoint(model,saved)
            loaded=PPO.load(saved,device='cpu');validate_checkpoint(saved,contract())
            assert loaded.observation_space['aux'].shape==(565+EXTRA_DIM,)
            assert all(torch.isfinite(t).all() for t in loaded.policy.state_dict().values())
        finally:
            model.logger.close();packets.clear()
            for owner,client in owners:client.acknowledge(0);client.close();owner.close()
    parent,child=mp.get_context('spawn').Pipe()
    process=mp.get_context('spawn').Process(target=spawned,args=(str(checkpoint),child))
    process.start();child.close()
    try:
        if not parent.poll(60):raise RuntimeError('Offline spawn timeout')
        result=parent.recv();process.join(10)
        assert result['status']=='PASS',result
        assert process.exitcode==0 and result['contract']==contract()
    finally:
        if process.is_alive():process.terminate();process.join(5)
        parent.close()
    print('OFFLINE PASS: exact policy/Adam preservation; rejects old contract; four-worker shared '
          'rollout merge; synthetic PPO update trains new columns; save/reload; Windows spawn. No game launched.')


if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--source',type=Path,required=True)
    p.add_argument('--checkpoint',type=Path,required=True)
    a=p.parse_args();check(a.source.resolve(),a.checkpoint.resolve())
