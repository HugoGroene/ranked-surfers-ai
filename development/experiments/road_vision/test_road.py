"""CPU-only regression tests; no browser, recording, GPU or current-run edits."""
import json
from pathlib import Path
import tempfile
import time
from types import SimpleNamespace
import unittest
from unittest.mock import patch
import zipfile
import numpy as np
from road_config import configuration,PRESETS,DEFAULT_ARMS
from road_crop import crop_rgb,observation,recorded_observation
from road_bc import mirror,select_history,inspect_demos

class CropTests(unittest.TestCase):
    def test_full_native_guard_pixels_preserved(self):
        frame=np.zeros((945,1920,4),np.uint8);frame[:,:,3]=255
        frame[:,:250,2]=255;frame[:,900:1020,1]=255
        original=frame.copy()
        policy,native=observation(frame,configuration('road128_tiny'))
        np.testing.assert_array_equal(frame,original)
        self.assertEqual(native.shape,(945,1920,3))
        self.assertEqual(policy.shape,(3,128,128))
        self.assertEqual(policy[0].max(),0)
        self.assertEqual(native[0,0,0],255)
        self.assertGreater(policy[1].max(),0)

    def test_all_candidates_preserve_depth_and_symmetric_crop(self):
        for name in PRESETS:
            c=configuration(name);x0,y0,x1,y1=c['roi']
            self.assertEqual((y0,y1),(0,1));self.assertAlmostEqual(x0+x1,1)
            self.assertEqual(crop_rgb(np.zeros((94,192,3),np.uint8),c).shape,(3,c['size'][1],c['size'][0]))

    def test_replay_letterbox_excluded(self):
        frame=np.zeros((3,288,512),np.uint8);frame[:,18:270]=77
        c=configuration('full512_tiny');small=recorded_observation(frame,c)
        np.testing.assert_array_equal(frame,small)

    def test_bad_geometry_rejected(self):
        with self.assertRaises(ValueError):observation(np.zeros((1080,1920,4),np.uint8),configuration('road128_tiny'))

class DemoTests(unittest.TestCase):
    def test_directional_mirroring_is_involution(self):
        from rs_v6_ppo_features import HAZARD_DIM
        from rs_v61_features import V6_AUX_DIM,SCREEN_COLUMN_DIM
        aux=np.zeros(565,np.float32)
        aux[HAZARD_DIM:HAZARD_DIM+20]=np.arange(20)/20
        lane=V6_AUX_DIM+SCREEN_COLUMN_DIM;aux[lane:lane+9]=np.arange(9)/9
        aux[557:]=np.arange(8)/8
        obs={'image':np.arange(12*4*4,dtype=np.uint8).reshape(12,4,4),'aux':aux}
        flipped,label=mirror(obs,1)
        self.assertEqual(label,2);self.assertEqual(flipped['aux'][lane],aux[lane+2])
        back,label=mirror(flipped,label)
        for key in obs:np.testing.assert_array_equal(back[key],obs[key])
        self.assertEqual(label,1)

    def test_detector_inputs_cannot_be_silently_mirrored(self):
        aux=np.zeros(565,np.float32);aux[0]=1
        with self.assertRaises(ValueError):mirror({'image':np.zeros((12,4,4),np.uint8),'aux':aux},0)

    def test_history_matches_live_selection_and_never_crosses_segments(self):
        from live_adapter import ContextHistory
        times=np.arange(61)/60
        idx,ages,valid=select_history(times,np.ones(61,bool),[[0,1.1]],8,.5)
        h=ContextHistory(8,.5)
        for i,t in enumerate(times):
            latest=np.full((3,1,1),i+1,np.uint8)
            image,aux=h.observation(latest,t)
            expected=np.where(idx[i]>=0,idx[i]+1,0)
            np.testing.assert_array_equal(image[:,0,0].reshape(8,3)[:,0],expected)
            np.testing.assert_allclose(aux[:8],ages[i],atol=1e-6)
            h.add(latest,t,h.token())
        idx,_,chosen=select_history(times,np.ones(61,bool),[[0,.3],[.6,1.1]],4,.5)
        self.assertTrue(np.all(idx[36,:-1]==-1));self.assertFalse(chosen[25])

    def test_invalid_timestamps_and_overlaps_rejected(self):
        with self.assertRaises(ValueError):select_history([0,0],[1,1],[[0,1]],4,.5)
        with self.assertRaises(ValueError):select_history([0,.1],[1,1],[[0,1],[.2,2]],4,.5)

    def test_legacy_demos_never_auto_approved_or_migrated(self):
        with tempfile.TemporaryDirectory() as t:
            root=Path(t);folder=root/'runs'/'demo_1';folder.mkdir(parents=True)
            path=folder/'manifest.json';path.write_text(json.dumps({'status':'approved','capture':{'framing_profile':'legacy'}}))
            before=path.read_bytes()
            with self.assertRaisesRegex(ValueError,'canvas_v1'):inspect_demos(root,True)
            self.assertEqual(before,path.read_bytes())
            with self.assertRaisesRegex(ValueError,'allow-downsampled'):inspect_demos(root)

class IntegrationTests(unittest.TestCase):
    def test_process_local_adapter_latest_and_buffer_frames_have_same_crop(self):
        import gymnasium as gym
        import tools.ppo_headless_env as headless
        import tools.ppo_framing as framing
        import tools.ppo_observation_contract as contracts
        from road_adapter import install
        c=configuration('road128_spatial',6)
        frame=np.zeros((945,1920,4),np.uint8);frame[:,800:1000,1]=200
        with tempfile.TemporaryDirectory() as tmp:
            class Base:
                def __init__(self):
                    self.observation_space=gym.spaces.Dict({'image':gym.spaces.Box(0,255,(12,128,128),np.uint8),
                        'aux':gym.spaces.Box(0,1,(557,),np.float32)})
                    self.output_dir=Path(tmp);self.training_steps=0
                    self._headless_capture_timing={'request_started_at':time.perf_counter()}
                def _native_frames(self):return framing.canvas_observation(frame)
                def _pack(self):
                    small,self.last_source_native=self._native_frames()
                    return {'image':np.tile(small,(4,1,1)),'aux':np.zeros(557,np.float32)},False,0
            trainer=SimpleNamespace(build_env_class=lambda:Base,shape_description=lambda:{},make_model=lambda *a:None)
            with patch.object(headless,'install',lambda *a,**k:None),patch.object(framing,'canvas_observation',framing.canvas_observation),patch.object(contracts,'requested_contract',lambda a:{'version':1,'image_shape':[12,128,128],'aux_dim':557}):
                install(trainer,c);raw=trainer.build_env_class()();obs,_,_=raw._pack()
                self.assertTrue(raw.observation_space.contains(obs))
                expected,_=framing.canvas_observation(frame)
                np.testing.assert_array_equal(obs['image'][-3:],expected)
                self.assertEqual(contracts.requested_contract(None)['road_vision'],c)

    def test_crop_contract_rejects_different_roi_and_encoder(self):
        from tools.ppo_observation_contract import validate_checkpoint
        with tempfile.TemporaryDirectory() as tmp:
            path=Path(tmp)/'checkpoint.zip'
            c={'version':1,'road_vision':configuration('road128_tiny')}
            with zipfile.ZipFile(path,'w') as z:z.writestr('data',json.dumps({'observation_contract':c}))
            validate_checkpoint(path,c)
            with self.assertRaises(ValueError):validate_checkpoint(path,{'version':1,'road_vision':configuration('road128_spatial')})

    def test_jobs_match_inputs_and_all_training_precedes_evaluation(self):
        from road_batch import jobs
        t={'frames':6,'span_ms':500,'capture_hz':30,'decision_hz':30}
        data=jobs(DEFAULT_ARMS,[123],30,100,60,Path('output'),t)
        n=len(DEFAULT_ARMS)
        self.assertEqual(len(data),2*n)
        self.assertTrue(all(j['mode']=='train' for j in data[:n]))
        self.assertTrue(all(j['mode']=='eval' for j in data[n:]))
        for train,evaluation in zip(data[:n],data[n:]):
            self.assertEqual(train['name'],evaluation['name'])
            self.assertEqual(train['command'][train['command'].index('--capture-backend')+1],'gpu_100')
            for flag in ('--arm','--frames','--span-ms','--capture-hz','--decision-hz','--seed','--capture-backend'):
                self.assertEqual(train['command'][train['command'].index(flag)+1],evaluation['command'][evaluation['command'].index(flag)+1])
        with_bc=jobs(['road128_spatial'],[123],30,100,60,Path('output'),t,Path('demos'))
        self.assertEqual(len(with_bc),4)

    def test_cli_arguments_override_only_policy_size_after_legacy_parser(self):
        from run_road_trial import args_for
        import train_rs_v612_ppo as trainer
        o=SimpleNamespace(mode='train',output=Path('trial'),minutes=30,seed=123,state_model=Path('model.pt'),
                          checkpoint=None,episodes=100,decision_hz=30)
        with patch.object(Path,'is_file',return_value=True):a=args_for(trainer,o,configuration('road192_spatial'))
        self.assertEqual(a.resolution,(192,192));self.assertEqual(a.resize_method,'area')
        self.assertEqual(a.press_delay_ms,5);self.assertEqual(a.detector_mode,'off');self.assertTrue(a.settled_learning)

class ModelTests(unittest.TestCase):
    def test_larger_road_inputs_support_forward_and_gradient(self):
        import torch
        import gymnasium as gym
        from road_models import SpatialExtractor
        torch.set_num_threads(1)
        for size in (256,384):
            with self.subTest(size=size):
                name=f'road{size}_spatial'
                self.assertIn(name,DEFAULT_ARMS)
                c=configuration(name)
                image,_=observation(np.full((945,1920,4),100,np.uint8),c)
                space=gym.spaces.Dict({'image':gym.spaces.Box(0,255,(12,size,size),np.uint8),
                                      'aux':gym.spaces.Box(0,1,(565,),np.float32)})
                model=SpatialExtractor(space)
                x={'image':torch.from_numpy(np.tile(image,(4,1,1))).unsqueeze(0).float()/255,
                   'aux':torch.zeros(1,565)}
                features=model(x)
                self.assertEqual(tuple(features.shape),(1,192))
                self.assertTrue(torch.isfinite(features).all())
                features.sum().backward()
                self.assertGreater(float(model.rgb[0].weight.grad.abs().sum()),0)

    def test_recorded_dataset_pipeline_and_source_immutability(self):
        from road_bc import Dataset
        from road_config import digest
        from rs_v61_features import FEATURE_VERSION,AUX_DIM,ACTION_NAMES
        with tempfile.TemporaryDirectory() as tmp:
            root=Path(tmp)
            for i in range(3):
                folder=root/'runs'/f'demo_{i}';folder.mkdir(parents=True)
                manifest=dict(status='approved',capture={'framing_profile':'canvas_v1'},schema=1,
                    feature_version=FEATURE_VERSION,aux_dim=AUX_DIM,image_shape=[3,288,512],
                    action_names=list(ACTION_NAMES),valid_segments_seconds=[[0.,1.]])
                (folder/'manifest.json').write_text(json.dumps(manifest))
                frames=np.zeros((5,3,288,512),np.uint8)
                frames[:,:,18:270]=100
                np.savez(folder/'frames.npz',frames=frames)
                for name,array in dict(aux=np.zeros((5,557),np.float32),actions=np.arange(5),
                    timestamps=np.arange(5)/30,valid=np.ones(5,bool)).items():np.save(folder/f'{name}.npy',array)
            runs=inspect_demos(root,True)
            data=Dataset(runs[:2],configuration('road96_spatial',6),root/'cache')
            try:
                self.assertEqual(len(data.samples),10)
                obs,label=data.sample(data.samples[-1])
                self.assertEqual(obs['image'].shape,(18,96,96))
                self.assertEqual(obs['aux'].shape,(569,))
                self.assertEqual(label,4)
                self.assertEqual(obs['aux'][-1],1.)
                self.assertEqual(obs['image'][-3:].max(),100)
                for path,h in data.hashes.items():self.assertEqual(digest(path),h)
                self.assertFalse(any(str(runs[2][0]) in p for p in data.hashes))
            finally:data.close()

    def test_production_model_factory_uses_experimental_encoder_on_cpu(self):
        import torch
        torch.set_num_threads(1)
        import gymnasium as gym
        import train_rs_v612_ppo as trainer
        import tools.ppo_headless_env as headless
        import tools.ppo_framing as framing
        import tools.ppo_observation_contract as contracts
        from road_adapter import install
        from road_models import SpatialExtractor
        class Env(gym.Env):
            observation_space=gym.spaces.Dict({'image':gym.spaces.Box(0,255,(18,96,96),np.uint8),
                'aux':gym.spaces.Box(0,1,(569,),np.float32)})
            action_space=gym.spaces.Discrete(5)
        a=SimpleNamespace(n_steps=2,batch_size=2,n_epochs=1,learning_rate=.0003,gamma=.995,ent_coef=.002,seed=123)
        with patch.object(torch.cuda,'is_available',return_value=False), \
             patch.object(headless,'install',lambda *a,**k:None), \
             patch.object(framing,'canvas_observation',framing.canvas_observation), \
             patch.object(contracts,'requested_contract',contracts.requested_contract), \
             patch.object(trainer,'make_model',trainer.make_model), \
             patch.object(trainer,'build_env_class',trainer.build_env_class), \
             patch.object(trainer,'shape_description',trainer.shape_description):
            install(trainer,configuration('road96_spatial',6))
            model=trainer.make_model(Env(),a)
            self.assertIsInstance(model.policy.features_extractor,SpatialExtractor)
            self.assertEqual(model.policy.features_extractor.rgb[0].in_channels,18)
            self.assertEqual(str(model.device),'cpu')

    def test_spatial_encoder_gradient_and_checkpoint_roundtrip(self):
        import torch
        torch.set_num_threads(1)
        import gymnasium as gym
        from stable_baselines3 import PPO
        from road_models import SpatialExtractor
        class Env(gym.Env):
            observation_space=gym.spaces.Dict({'image':gym.spaces.Box(0,255,(12,96,96),np.uint8),
                'aux':gym.spaces.Box(0,1,(565,),np.float32)})
            action_space=gym.spaces.Discrete(5)
        model=PPO('MultiInputPolicy',Env(),device='cpu',n_steps=2,batch_size=2,seed=123,
                  policy_kwargs={'features_extractor_class':SpatialExtractor,'net_arch':{'pi':[64],'vf':[64]}})
        obs={'image':torch.full((2,12,96,96),100,dtype=torch.uint8),'aux':torch.zeros(2,565)}
        loss=-model.policy.get_distribution(obs).log_prob(torch.tensor([1,2])).mean();loss.backward()
        self.assertGreater(float(model.policy.features_extractor.rgb[0].weight.grad.abs().sum()),0)
        with tempfile.TemporaryDirectory() as tmp:
            path=Path(tmp)/'model.zip';model.save(path);loaded=PPO.load(path,device='cpu')
            with torch.no_grad():
                torch.testing.assert_close(model.policy.get_distribution(obs).distribution.probs,loaded.policy.get_distribution(obs).distribution.probs)

    def test_end_to_end_bc_reaches_encoder_but_not_critic_head(self):
        import torch
        torch.set_num_threads(1)
        import gymnasium as gym
        from stable_baselines3 import PPO
        from road_models import SpatialExtractor
        from road_bc import fit
        class Env(gym.Env):
            observation_space=gym.spaces.Dict({'image':gym.spaces.Box(0,255,(12,32,32),np.uint8),
                'aux':gym.spaces.Box(0,1,(565,),np.float32)})
            action_space=gym.spaces.Discrete(5)
        class Data:
            samples=list(range(10))
            def label(self,key):return key%5
            def sample(self,key):
                return {'image':np.full((12,32,32),30+40*(key%5),np.uint8),'aux':np.zeros(565,np.float32)},key%5
        model=PPO('MultiInputPolicy',Env(),device='cpu',n_steps=2,batch_size=2,seed=123,
                  policy_kwargs={'features_extractor_class':SpatialExtractor,'net_arch':{'pi':[64],'vf':[64]}})
        before={k:v.clone() for k,v in model.policy.state_dict().items() if k.startswith(('value_net.','mlp_extractor.value_net.'))}
        state,report=fit(model,Data(),Data(),1,123,batch_size=5)
        self.assertGreater(float(model.policy.features_extractor.rgb[0].weight.grad.abs().sum()),0)
        for k,v in before.items():torch.testing.assert_close(v,state[k])
        self.assertEqual(len(report['epochs']),1);self.assertEqual(report['selected']['samples'],10)

if __name__=='__main__':unittest.main()
