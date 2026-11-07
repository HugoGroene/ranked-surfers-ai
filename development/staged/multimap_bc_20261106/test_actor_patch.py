import copy
from pathlib import Path
import tempfile
import time
import unittest
from unittest.mock import patch
import numpy as np
import torch
from actor_patch import freeze, features, logits, kl_old_new, gate, importance_weights, trial
from test_retry import TinyData


def measurement(loss=2.):
    return {name:dict(macro_loss=loss, natural_loss=loss, mean_kl=.001,p95_kl=.01,
                     recall=[.7,.3,.3,.3,.3]) for name in ('halloween','tokyo')}


class ActorTests(unittest.TestCase):
    def test_change_budget_and_each_action_rejection(self):
        baseline=measurement(3.)
        self.assertTrue(gate(measurement(),baseline)[0])
        for key, value in [('mean_kl',.021),('p95_kl',.101),('macro_loss',2.99)]:
            m=measurement();m['tokyo'][key]=value
            self.assertFalse(gate(m,baseline)[0])
        for a in range(5):
            m=measurement();m['halloween']['recall'][a]=0
            self.assertFalse(gate(m,baseline)[0])
        m=measurement();m['tokyo']['macro_loss']=float('nan')
        self.assertFalse(gate(m,baseline)[0])

    def test_kl_direction_and_zero(self):
        p=torch.tensor([[.8,.2]]).log();q=torch.tensor([[.5,.5]]).log()
        self.assertTrue(torch.equal(kl_old_new(p,p),torch.zeros(1)))
        self.assertAlmostEqual(float(kl_old_new(p,q)),.8*np.log(1.6)+.2*np.log(.4),places=6)

    def test_sampling_importance_corrects_proposal(self):
        data=TinyData('train')
        _,weights=importance_weights(data,3.)
        for m in ('halloween','tokyo'):
            implied=np.array([weights[m,a] for a in range(5)])*np.array([.5,.125,.125,.125,.125])
            np.testing.assert_allclose(implied,np.array([1,3,3,3,3])/13)

    def test_frozen_features_critic_and_ppo_reload(self):
        import gymnasium as gym
        from stable_baselines3 import PPO
        from stable_baselines3.common.torch_layers import BaseFeaturesExtractor
        torch.set_num_threads(1)
        class Extractor(BaseFeaturesExtractor):
            def __init__(self,space):
                super().__init__(space,8);self.linear=torch.nn.Linear(789,8)
            def forward(self,obs):return torch.tanh(self.linear(obs['aux']))
        class Env(gym.Env):
            observation_space=gym.spaces.Dict(image=gym.spaces.Box(0,255,(12,16,16),np.uint8),
                                             aux=gym.spaces.Box(0,1,(789,),np.float32))
            action_space=gym.spaces.Discrete(5)
        model=PPO('MultiInputPolicy',Env(),device='cpu',n_steps=8,batch_size=4,
                  policy_kwargs=dict(features_extractor_class=Extractor,net_arch=dict(pi=[8],vf=[8])))
        actor=freeze(model.policy)
        before={n:t.clone() for n,t in model.policy.state_dict().items()}
        data=TinyData('train');val=TinyData('validation')
        cache=features(model.policy,data,list(range(10)),time.monotonic()+30)
        obs={'image':torch.zeros((10,12,16,16)), 'aux':torch.eye(789)[:10]}
        # Features used by the cache are exactly the live policy extraction path.
        with torch.no_grad():
            for i in range(10):obs['aux'][i].zero_();obs['aux'][i,i%5]=1
            direct=model.policy.get_distribution(obs).distribution.logits
            torch.testing.assert_close(logits(model.policy,cache['x']).log_softmax(-1),direct)
            values_before=model.policy.predict_values(obs).clone()
        with tempfile.TemporaryDirectory() as tmp:
            with patch('actor_patch.metrics',side_effect=[measurement(3.),measurement(2.)]):
                result=trial(model,data,val,Path(tmp),time.monotonic()+30,2,3.)
            self.assertEqual(result['status'],'CANDIDATE_SAVED')
            saved=PPO.load(Path(tmp)/'bc_best.zip',device='cpu')
            self.assertTrue(all(p.requires_grad for p in saved.policy.parameters()))
            changed=False
            for name,t in before.items():
                if name.startswith(('mlp_extractor.policy_net.','action_net.')):
                    changed |= not torch.equal(t,saved.policy.state_dict()[name])
                else:self.assertTrue(torch.equal(t,saved.policy.state_dict()[name]),name)
            self.assertTrue(changed)
            with torch.no_grad():torch.testing.assert_close(values_before,saved.policy.predict_values(obs),rtol=0,atol=0)
            with patch('actor_patch.metrics',side_effect=[measurement(3.),measurement(4.)]):
                rejected=Path(tmp)/'rejected';rejected.mkdir()
                result=trial(model,data,val,rejected,time.monotonic()+30,2,3.)
            self.assertEqual(result['status'],'NO_ACCEPTABLE_CANDIDATE')
            self.assertFalse((rejected/'bc_best.zip').exists())


if __name__=='__main__':unittest.main()
