"""Create a fresh CPU-only spatial PPO checkpoint; never opens a game."""
import argparse
import json
import os
from pathlib import Path

from parallel_adapter import LAYOUTS,install,contract,configuration


def create(output,layout,seed=123):
    if output.exists():raise ValueError('Refusing to overwrite a checkpoint')
    os.environ['RS_COMPARISON_LAYOUT']=layout
    install()
    import gymnasium as gym
    import numpy as np
    from gymnasium import spaces
    from stable_baselines3 import PPO
    from road_models import SpatialExtractor
    from tools.ppo_observation_contract import attach_contract,validate_checkpoint
    c=contract()
    class ShapeOnlyEnv(gym.Env):
        observation_space=spaces.Dict({
            'image':spaces.Box(0,255,tuple(c['image_shape']),np.uint8),
            'aux':spaces.Box(0,1,(c['aux_dim'],),np.float32)})
        action_space=spaces.Discrete(5)
        def reset(self,**kwargs):raise RuntimeError('Shape-only environment cannot collect samples')
        def step(self,action):raise RuntimeError('Shape-only environment cannot collect samples')
    model=PPO('MultiInputPolicy',ShapeOnlyEnv(),device='cpu',seed=seed,
        n_steps=128,batch_size=32,n_epochs=5,learning_rate=.0003,gamma=.995,
        gae_lambda=.95,ent_coef=.002,clip_range=.2,verbose=0,
        policy_kwargs=dict(features_extractor_class=SpatialExtractor,
            features_extractor_kwargs={'features_dim':192},net_arch={'pi':[64],'vf':[64]}))
    attach_contract(model,c)
    output.parent.mkdir(parents=True,exist_ok=True)
    model.save(output);validate_checkpoint(output,c)
    output.with_suffix('.json').write_text(json.dumps(dict(initialization='scratch',seed=seed,
        configuration=configuration(),num_timesteps=model.num_timesteps,
        share_features_extractor=model.policy.share_features_extractor),indent=2))
    print('FRESH CHECKPOINT:',output,flush=True)


if __name__=='__main__':
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--output',type=Path,required=True)
    p.add_argument('--layout',choices=LAYOUTS,required=True)
    a=p.parse_args();create(a.output.resolve(),a.layout)
