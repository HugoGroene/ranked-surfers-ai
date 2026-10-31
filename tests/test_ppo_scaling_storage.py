import tempfile
from pathlib import Path
import unittest
import gymnasium as gym
import numpy as np
import torch
from stable_baselines3 import PPO
from stable_baselines3.common.torch_layers import BaseFeaturesExtractor
from tools.ppo_compact_buffer import CompactTimedDictBuffer,CompactImages
from tools.ppo_settled_learning import TimedDictBuffer
from tools.ppo_parallel_learning import BUFFER_FIELDS,merge_rollouts,configure_learner
from tools.ppo_rollout_transport import RolloutSlot


class TinyImageEnv(gym.Env):
    observation_space=gym.spaces.Dict({'image':gym.spaces.Box(0,255,(12,16,16),np.uint8),
                                     'aux':gym.spaces.Box(0,1,(3,),np.float32)})
    action_space=gym.spaces.Discrete(5)
    def reset(self,**kwargs):return {'image':np.zeros((12,16,16),np.uint8),'aux':np.zeros(3,np.float32)},{}


class TinyFeatures(BaseFeaturesExtractor):
    def __init__(self,space):super().__init__(space,3075)
    def forward(self,obs):return torch.cat([obs['image'].flatten(1),obs['aux']],dim=1)


def buffer(cls=TimedDictBuffer):
    result=cls(16,TinyImageEnv.observation_space,TinyImageEnv.action_space,device='cpu',n_envs=1)
    result.reference_seconds=.05
    return result


def fill(target):
    rng=np.random.default_rng(15)
    frames=[rng.integers(0,256,(3,16,16),dtype=np.uint8) for _ in range(20)]
    for i in range(16):
        # Include reset duplicates and a wholly fresh resynchronization stack.
        pieces=([frames[0]]*4 if i==0 else frames[i:i+4])
        if i==8:pieces=list(reversed(frames[:4]))
        obs={'image':np.concatenate(pieces)[None],'aux':np.full((1,3),i/16,np.float32)}
        target.active_seconds[i]=.04+i*.001
        target.actor_masks[i]=i%3!=0
        target.add(obs,np.array([[i%5]]),np.array([i/10]),np.array([i in (0,8)]),
                   torch.tensor([[i/20]]),torch.tensor([-.5]))
    target.compute_returns_and_advantage(torch.tensor([[.3]]),np.array([False]))


class ScalingStorageTests(unittest.TestCase):
    def test_minibatches_exactly_match_dense_storage_across_resets_and_multiple_epochs(self):
        dense,compact=buffer(),buffer(CompactTimedDictBuffer)
        fill(dense);fill(compact)
        np.testing.assert_array_equal(dense.observations['image'],compact.observations['image'][:])
        self.assertLess(compact.observations['image'].stored_bytes,dense.observations['image'].nbytes)
        for epoch in range(3):
            np.random.seed(epoch);a=list(dense.get(4))
            np.random.seed(epoch);b=list(compact.get(4))
            for x,y in zip(a,b):
                for key in x.observations:torch.testing.assert_close(x.observations[key],y.observations[key],rtol=0,atol=0)
                for key in ['actions','old_values','old_log_prob','advantages','returns','actor_mask']:
                    torch.testing.assert_close(getattr(x,key),getattr(y,key),rtol=0,atol=0)
                np.testing.assert_array_equal(x.sample_indices,y.sample_indices)

    def test_compact_storage_owns_frames_and_rejects_unwritten_rows(self):
        storage=CompactImages((2,1,12,16,16))
        source=np.full((1,12,16,16),10,np.uint8)
        storage[0]=source;source[:]=99
        self.assertTrue(np.all(storage[0]==10))
        self.assertEqual(storage.count,1)
        with self.assertRaises(ValueError):storage[1]

    def test_shared_slot_has_no_arrays_in_message_and_requires_commit_before_overwrite(self):
        for cls in [TimedDictBuffer,CompactTimedDictBuffer]:
            with self.subTest(buffer=cls.__name__):
                source=buffer(cls);fill(source)
                owner=RolloutSlot(source)
                writer=RolloutSlot(descriptor=owner.descriptor)
                try:
                    message=writer.publish(source,0,0,'digest')
                    self.assertFalse(any(isinstance(v,np.ndarray) for v in message.values()))
                    with self.assertRaises(ValueError):writer.publish(source,0,1,'digest')
                    with self.assertRaises(ValueError):writer.acknowledge(1)
                    packet=owner.packet(message,0,'digest')
                    with self.assertRaises(ValueError):owner.packet(message,1,'digest')
                    with self.assertRaises(ValueError):owner.packet(message,0,'other-policy')
                    for key in BUFFER_FIELDS:np.testing.assert_array_equal(packet[key],getattr(source,key))
                    image=packet['observations']['image']
                    np.testing.assert_array_equal(image[:] if hasattr(image,'refs') else image,
                                                  source.observations['image'][:] if cls==CompactTimedDictBuffer else source.observations['image'])
                    writer.acknowledge(0)
                    writer.publish(source,0,1,'digest')
                finally:writer.close();owner.close()

    def test_compact_shared_merge_keeps_workers_separate_and_matches_real_optimizer(self):
        torch.set_num_threads(1)
        sources=[buffer(CompactTimedDictBuffer),buffer(CompactTimedDictBuffer)]
        for source in sources:fill(source)
        owners=[RolloutSlot(source) for source in sources]
        writers=[RolloutSlot(descriptor=owner.descriptor) for owner in owners]
        packets=[owner.packet(writer.publish(source,i,0,'d'),0,'d') for i,(source,owner,writer) in enumerate(zip(sources,owners,writers))]
        models=[]
        directory=tempfile.TemporaryDirectory()
        try:
            for i,cls in enumerate([TimedDictBuffer,CompactTimedDictBuffer]):
                model=PPO('MultiInputPolicy',TinyImageEnv(),n_steps=16,batch_size=8,n_epochs=2,
                          device='cpu',seed=19,policy_kwargs={'features_extractor_class':TinyFeatures})
                configure_learner(model,Path(directory.name)/str(i),2,.05,buffer_class=cls)
                models.append(model)
                merge_rollouts(model.rollout_buffer,packets,16,0,'d')
                np.random.seed(22);torch.manual_seed(22);model.train()
            for key,value in models[0].policy.state_dict().items():
                torch.testing.assert_close(value,models[1].policy.state_dict()[key],rtol=0,atol=0)
            self.assertEqual(models[1]._settled_stats['optimized_transitions'],32)
            self.assertEqual(models[1].n_steps,16)
        finally:
            for model in models:model.logger.close();model.rollout_buffer.reset()
            for writer in writers:writer.close()
            for owner in owners:owner.close()
            directory.cleanup()
