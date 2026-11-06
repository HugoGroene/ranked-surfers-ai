"""CPU checkpoint, spawned adapter and exact comparison-shape shared-buffer checks."""
import argparse
import multiprocessing as mp
from pathlib import Path
import os
import tempfile

from parallel_adapter import install,contract,inspect_spawn,selected
import run_workers  # Re-imported by Windows spawn; never calls its live main.


def check(checkpoint):
    import numpy as np
    import torch
    from stable_baselines3 import PPO
    from tools.ppo_observation_contract import validate_checkpoint
    from tools.ppo_settled_learning import TimedDictBuffer
    from tools.ppo_rollout_transport import RolloutSlot
    from tools.ppo_parallel_learning import merge_rollouts,configure_learner
    from road_models import SpatialExtractor
    install();validate_checkpoint(checkpoint,contract())
    _,spec=selected();frames=spec['recent']+len(spec['context'])
    shape=(frames*3,384,384);aux=557+frames*2
    model=PPO.load(checkpoint,device='cpu')
    assert isinstance(model.policy.features_extractor,SpatialExtractor)
    assert model.observation_space['image'].shape==shape
    assert model.observation_space['aux'].shape==(aux,) and model.n_steps==128
    assert model.policy.share_features_extractor
    assert model.policy.pi_features_extractor is model.policy.vf_features_extractor
    assert model.n_epochs==5 and model.batch_size==32 and model.target_kl is None
    assert model.gamma==.995 and model.ent_coef==.002
    assert float(model.lr_schedule(1.))==.0003
    observation={'image':np.zeros(shape,np.uint8),'aux':np.zeros(aux,np.float32)}
    action,_=model.predict(observation,deterministic=True)
    assert 0<=int(action)<5
    for count in range(1,6):
        target=TimedDictBuffer(2*count,model.observation_space,model.action_space,device='cpu')
        slots=[];packets=[]
        try:
            for i in range(count):
                source=TimedDictBuffer(2,model.observation_space,model.action_space,device='cpu')
                source.observations['image'][:]=i+1;source.observations['aux'][:]=(i+1)/10
                source.full=True;source.pos=2;source.active_seconds[:]=.05
                owner=RolloutSlot(source);client=RolloutSlot(descriptor=owner.descriptor)
                slots.append((owner,client))
                message=client.publish(source,i,0,'offline')
                packets.append(owner.packet(message,0,'offline'))
                del source
            merge_rollouts(target,packets,2,0,'offline')
            for i in range(count):
                assert np.all(target.observations['image'][2*i:2*i+2]==i+1)
                np.testing.assert_allclose(target.observations['aux'][2*i:2*i+2],(i+1)/10)
            if count==5:
                with tempfile.TemporaryDirectory() as tmp:
                    model.n_steps=2;model.batch_size=2;model.n_epochs=1
                    configure_learner(model,Path(tmp),5,.05)
                    try:
                        merge_rollouts(model.rollout_buffer,packets,2,0,'offline')
                        model.train()
                        assert model.n_steps==2 and model._settled_stats['optimized_transitions']==10
                        assert len(model.parallel_used_indices)==10
                        saved=Path(tmp)/'offline.zip';model.save(saved)
                        validate_checkpoint(saved,contract())
                        loaded=PPO.load(saved,device='cpu')
                        assert loaded.observation_space['image'].shape==shape
                        assert all(torch.isfinite(t).all() for t in loaded.policy.state_dict().values())
                    finally:model.logger.close()
        finally:
            packets.clear()
            for owner,client in slots:client.acknowledge(0);client.close();owner.close()
    # Verify process-local hooks actually survive the Windows spawn entry path.
    os.environ['RS_COMPARISON_ACTIVE']='1'
    parent,child=mp.get_context('spawn').Pipe()
    process=mp.get_context('spawn').Process(target=inspect_spawn,args=(str(checkpoint),child))
    process.start();child.close()
    try:
        if not parent.poll(60):raise RuntimeError('Offline spawn check timed out')
        result=parent.recv()
        process.join(10)
        assert result['status']=='PASS',result
        assert result['contract']==contract() and process.exitcode==0
    finally:
        if process.is_alive():process.terminate();process.join(5)
        parent.close()
    print('OFFLINE PASS: checkpoint contract, shared actor/critic features, 1-5 exact shared buffers, CPU update/save/reload, spawned adapter; no game launched.')


if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--checkpoint',type=Path,required=True)
    check(p.parse_args().checkpoint.resolve())
