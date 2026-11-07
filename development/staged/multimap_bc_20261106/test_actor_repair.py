from pathlib import Path
import json
import tempfile
import time
import unittest
from unittest.mock import patch
import numpy as np
import torch
import gymnasium as gym
from stable_baselines3 import PPO
from stable_baselines3.common.torch_layers import BaseFeaturesExtractor
from common import ROOT
from development.staged.multimap_bc_20261106.normalized_actor import replace_actor, NormalizedActorPolicy
from actor_patch import freeze, logits
from repair_actor import acceptance,fit


class Extractor(BaseFeaturesExtractor):
    def __init__(self,space):
        super().__init__(space,8);self.linear=torch.nn.Linear(9,8)
    def forward(self,obs):return self.linear(obs['aux'])*100


class Env(gym.Env):
    observation_space=gym.spaces.Dict(image=gym.spaces.Box(0,255,(3,16,16),np.uint8),
                                     aux=gym.spaces.Box(0,1,(9,),np.float32))
    action_space=gym.spaces.Discrete(5)
    def reset(self,seed=None,options=None):
        super().reset(seed=seed);self.steps=0
        return self.observation_space.sample(),{}
    def step(self,action):
        self.steps+=1
        return self.observation_space.sample(),float(action==1),self.steps>=5,False,{}


class RepairTests(unittest.TestCase):
    def test_successful_real_fit_report_is_json_serializable(self):
        torch.set_num_threads(1)
        model=PPO('MultiInputPolicy',Env(),n_steps=8,batch_size=4,n_epochs=1,device='cpu',
                  policy_kwargs=dict(features_extractor_class=Extractor,net_arch=dict(pi=[8],vf=[8])))
        obs={'image':torch.zeros(20,3,16,16),'aux':torch.rand(20,9)}
        with torch.no_grad():
            x=model.policy.extract_features(obs)
            cache=dict(x=x,y=torch.arange(20)%5,maps=np.repeat([0,1],10),
                       old_log=logits(model.policy,x).log_softmax(-1))
        with tempfile.TemporaryDirectory() as tmp:
            with patch('repair_actor.acceptance',return_value=(True,[])):
                result=fit(model,cache,cache,Path(tmp),time.monotonic()+20,updates=2)
            self.assertEqual(result['status'],'CANDIDATE_SAVED')
            json.dumps(result,allow_nan=False)
            self.assertTrue((Path(tmp)/'bc_best.zip').is_file())

    def test_actor_only_normalization_roundtrip_and_ppo_update(self):
        torch.set_num_threads(1);torch.manual_seed(7)
        model=PPO('MultiInputPolicy',Env(),n_steps=8,batch_size=4,n_epochs=1,device='cpu',
                  policy_kwargs=dict(features_extractor_class=Extractor,net_arch=dict(pi=[8],vf=[8])))
        obs={'image':torch.zeros(10,3,16,16),'aux':torch.rand(10,9)}
        with torch.no_grad():
            x=model.policy.extract_features(obs);values=model.policy.predict_values(obs).clone()
        keep=replace_actor(model,x);params=freeze(model.policy)
        norm=model.policy.mlp_extractor.policy_net[0]
        mean=norm.mean.clone();scale=norm.scale.clone()
        with torch.no_grad():
            torch.testing.assert_close(values,model.policy.predict_values(obs),rtol=0,atol=0)
            torch.testing.assert_close(norm(x).mean(0),torch.zeros(8),atol=1e-5,rtol=0)
        opt=torch.optim.Adam(params,lr=.01)
        for _ in range(10):
            loss=torch.nn.functional.cross_entropy(logits(model.policy,x),torch.arange(10)%5)
            opt.zero_grad();loss.backward();opt.step()
        for k,v in keep.items():self.assertTrue(torch.equal(v,model.policy.state_dict()[k]))
        with torch.no_grad():expected=model.policy.get_distribution(obs).distribution.probs.clone()
        for p in model.policy.parameters():p.requires_grad_(True)
        with tempfile.TemporaryDirectory() as tmp:
            checkpoint=Path(tmp)/'test.zip';model.save(checkpoint)
            loaded=PPO.load(checkpoint,env=Env(),device='cpu')
            self.assertIsInstance(loaded.policy,NormalizedActorPolicy)
            with torch.no_grad():
                torch.testing.assert_close(expected,loaded.policy.get_distribution(obs).distribution.probs,rtol=0,atol=0)
                torch.testing.assert_close(values,loaded.policy.predict_values(obs),rtol=0,atol=0)
            # PPO's combined actor/critic path, not just the BC helper, can backpropagate.
            loaded.learn(16)
            norm2=loaded.policy.mlp_extractor.policy_net[0]
            self.assertTrue(torch.equal(mean,norm2.mean));self.assertTrue(torch.equal(scale,norm2.scale))
            self.assertTrue(all(torch.isfinite(p).all() for p in loaded.policy.parameters()))

    def test_quality_gate_rejects_prior_only_and_direction_collapse(self):
        source={m:dict(macro_loss=4.) for m in ('halloween','tokyo')}
        current={m:dict(macro_loss=1.4,natural_loss=.5,recall=[.8,.3,.3,.3,.3],
            confusion=[[80,5,5,5,5],[1,3,0,0,0],[1,0,3,0,0],[1,0,0,3,0],[1,0,0,0,3]]) for m in source}
        self.assertTrue(acceptance(current,source)[0])
        current['tokyo']['recall'][1]=0
        self.assertFalse(acceptance(current,source)[0])
        current['tokyo']['recall'][1]=.3;current['tokyo']['macro_loss']=1.8022
        self.assertFalse(acceptance(current,source)[0])
        # A different loss prior needs its own constant-classifier baseline.
        prior=np.array([.92,.02,.02,.02,.02]);proposal=np.array([.5,.125,.125,.125,.125])
        weights={(m,a):float(prior[a]/proposal[a]) for m in source for a in range(5)}
        self.assertTrue(acceptance(current,source,weights)[0])
        current['tokyo']['macro_loss']=float(-np.log(prior).mean())
        self.assertFalse(acceptance(current,source,weights)[0])


if __name__=='__main__':unittest.main()
