"""Exercise real adapter, PPO construction and checkpoint reload on CPU only."""
import copy
from pathlib import Path
import sys
import tempfile
from unittest.mock import patch

import run_probe
import run_road_trial as trial


def check():
    import numpy as np
    import torch
    import gymnasium as gym
    from gymnasium import spaces
    from stable_baselines3 import PPO
    import train_rs_v612_ppo as trainer
    import road_adapter
    import live_adapter
    from road_models import SpatialExtractor
    from tools import ppo_observation_contract as contracts
    torch.set_num_threads(1)
    recent,old=run_probe.LAYOUTS[layout]
    n=recent+len(old);channels=3*n;aux_dim=557+2*n
    config=trial.configuration('road384_spatial',n,1000,60,30)
    road_adapter.install(trainer,config)
    contract=contracts.requested_contract(type('Args',(),{'resolution':(384,384)})())
    assert contract['image_shape']==[channels,384,384]
    assert contract['aux_dim']==aux_dim
    assert contract['temporal']['context_offsets_seconds']==list(old)
    assert contract['temporal']['recent_frames']==recent
    assert issubclass(trainer.build_env_class(),gym.Env)
    history=live_adapter.ContextHistory(n,1.)
    image,aux=history.observation(np.zeros((3,384,384),np.uint8),1.)
    assert image.shape==(channels,384,384) and aux[-n:].sum()==1

    class Dummy(gym.Env):
        observation_space=spaces.Dict({'image':spaces.Box(0,255,(channels,384,384),np.uint8),
                                       'aux':spaces.Box(0,1,(aux_dim,),np.float32)})
        action_space=spaces.Discrete(5)
        def reset(self,*,seed=None,options=None):
            super().reset(seed=seed)
            return {'image':image,'aux':np.zeros(aux_dim,np.float32)},{}
        def step(self,action):return self.reset()[0],0.,False,False,{}

    args=type('Args',(),dict(n_steps=2,batch_size=2,n_epochs=1,learning_rate=.0003,
                           gamma=.995,ent_coef=.002,seed=123))()
    with patch('torch.cuda.is_available',return_value=False):
        model=trainer.make_model(Dummy(),args)
    assert isinstance(model.policy.features_extractor,SpatialExtractor)
    contracts.attach_contract(model,contract)
    model.learn(2)
    with tempfile.TemporaryDirectory() as tmp:
        path=Path(tmp)/'model.zip';model.save(path)
        contracts.validate_checkpoint(path,contract)
        loaded=PPO.load(path,device='cpu')
        before=model.predict(Dummy().reset()[0],deterministic=True)[0]
        after=loaded.predict(Dummy().reset()[0],deterministic=True)[0]
        np.testing.assert_array_equal(before,after)
        wrong=copy.deepcopy(contract);wrong['temporal']['context_offsets_seconds']=[.3,.5,.8,1.]
        try:contracts.validate_checkpoint(path,wrong)
        except ValueError:pass
        else:raise AssertionError('Different history semantics accepted')
    # Forward shapes for every candidate, with no browser or GPU allocation.
    for size in (256,384):
        for recent,old in run_probe.LAYOUTS.values():
            n=recent+len(old)
            space=spaces.Dict({'image':spaces.Box(0,255,(3*n,size,size),np.uint8),
                              'aux':spaces.Box(0,1,(557+2*n,),np.float32)})
            encoder=SpatialExtractor(space)
            with torch.no_grad():
                output=encoder({'image':torch.zeros(1,3*n,size,size),'aux':torch.zeros(1,557+2*n)})
            assert output.shape==(1,192) and torch.isfinite(output).all()
    print('CPU integration PASS: all shapes, real PPO update, save/reload, contract rejection')


if __name__=='__main__':
    # run_probe installs its normal process-local hooks; replace only the final
    # live entry point, so this test cannot start a browser.
    import argparse
    p=argparse.ArgumentParser();p.add_argument('--layout',choices=list(run_probe.LAYOUTS),default='recent4_context4')
    layout=p.parse_args().layout
    trial.main=check
    sys.argv=['offline','--layout',layout,'--size','384','--output','unused']
    run_probe.main()
