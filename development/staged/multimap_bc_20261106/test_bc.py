import json
from pathlib import Path
import tempfile
import threading
import unittest
from unittest.mock import patch
import numpy as np
from common import read,write_new
from dataset import labels,reconstruct,stacked
from train import mirror,split_check
from review import approve,latest
from record import FrameWriter


def frames(n=40):
    return [dict(at=10+i*.05,end=10+i*.05+.001,state='GAMEPLAY') for i in range(n)]


def key(at,code='ArrowUp'):
    return [dict(at=at,code=code,type='keydown'),dict(at=at+.005,code=code,type='keyup')]


class BCTests(unittest.TestCase):
    def test_duration_selection_is_explicit_and_not_claimed_visual_review(self):
        from review import authorized
        with tempfile.TemporaryDirectory() as folder:
            run=Path(folder)
            for name,value in [('manifest.json',dict(duration=20)),('frames.json',[]),('events.json',[])]:write_new(run/name,value)
            path=approve(run,'0:15',False,duration_rule=(12,5))
            revision=read(path)
            self.assertFalse(revision['reviewed'])
            self.assertTrue(authorized(revision,dict(duration=20)))
            revision['intervals']=[[0,19]]
            self.assertFalse(authorized(revision,dict(duration=20)))
            with self.assertRaises(ValueError):approve(run,'0:19',False,duration_rule=(12,5))
            with self.assertRaises(ValueError):approve(run,'0:15',False)

    def test_wasd_has_identical_targets_and_causal_history_to_arrows(self):
        for alias,arrow in [('KeyW','ArrowUp'),('KeyA','ArrowLeft'),('KeyS','ArrowDown'),('KeyD','ArrowRight')]:
            a=reconstruct(frames(),key(10.025,alias),[[0,1.8]],20)
            b=reconstruct(frames(),key(10.025,arrow),[[0,1.8]],20)
            self.assertEqual([s['action'] for s in a],[s['action'] for s in b])
            for x,y in zip(a,b):np.testing.assert_array_equal(x['aux'],y['aux'])

    def test_writer_lossless_pixel_roundtrip(self):
        import cv2
        source=np.random.default_rng(12).integers(0,256,(64,128,4),dtype=np.uint8)
        with tempfile.TemporaryDirectory() as folder:
            file=Path(folder)/'frame.png';writer=FrameWriter()
            try:self.assertTrue(writer.submit(file,source))
            finally:writer.close()
            np.testing.assert_array_equal(cv2.imread(str(file)),source[:,:,:3])

    def test_target_never_leaks_into_own_command_history(self):
        rows=frames();events=key(10.025)
        samples=reconstruct(rows,events,[[0,1.8]],20)
        self.assertEqual(samples[0]['action'],3)
        self.assertFalse(samples[0]['aux'][565:].any())
        history=samples[1]['aux'][565:].reshape(32,7)
        self.assertEqual(history[-1,3],1)
        self.assertAlmostEqual(history[-1,5],.025,places=5)
        self.assertEqual(samples[0]['aux'].shape,(789,))
        self.assertTrue(np.all(samples[0]['indices'][:-1]==-1))

    def test_no_crossing_review_segments_or_episodes(self):
        samples=reconstruct(frames(),key(10.925),[[0,.8],[1,1.8]],20)
        first=next(s for s in samples if s['segment']==1)
        self.assertFalse(first['aux'][565:].any())
        self.assertTrue(np.all(first['indices'][:-1]==-1))

    def test_live_history_selection_and_exact_stack_pixels(self):
        rows=frames();samples=reconstruct(rows,[],[[0,1.8]],20)
        sample=next(s for s in samples if s['index']==16)
        self.assertTrue(np.all(sample['indices']>=0))
        ages=(rows[16]['at']-np.array([rows[i]['at'] for i in sample['indices']]))/.81
        np.testing.assert_allclose(sample['aux'][557:561],ages,atol=1e-6)
        pixels=np.stack([np.full((3,4,4),i,np.uint8) for i in range(len(rows))])
        result=stacked(pixels,sample['indices'])
        np.testing.assert_equal(result[::3,0,0],sample['indices'])

    def test_exposure_conflicts_hold_and_slow_capture_are_invalid(self):
        rows=frames()
        for events in (key(10.0005),key(10.02)+key(10.03,'ArrowDown'),
                       [dict(at=10.02,code='ArrowUp',type='keydown'),dict(at=10.3,code='ArrowUp',type='keyup')]):
            targets,_=labels(rows,events,20)
            self.assertFalse(targets[0]['valid'])
        rows[1]['at']=10.14;rows[1]['end']=10.141;rows=rows[:1]+rows[3:]
        targets,_=labels(rows,[],20);self.assertFalse(targets[0]['valid'])

    def test_menu_images_never_become_context(self):
        rows=frames();rows[11]['state']='MENU'
        samples=reconstruct(rows,[],[[0,1.8]],20)
        self.assertTrue(all(11 not in s['indices'] for s in samples))

    def test_mirroring_swaps_every_directional_history_and_is_involution(self):
        obs=dict(image=np.arange(12*16*16,dtype=np.uint8).reshape(12,16,16),aux=np.zeros(789,np.float32))
        obs['aux'][565:].reshape(32,7)[-1]=[0,1,0,0,0,.2,1]
        mirrored,label=mirror(obs,1)
        self.assertEqual(label,2);self.assertEqual(mirrored['aux'][565:].reshape(32,7)[-1,2],1)
        restored,label=mirror(mirrored,label)
        for k in obs:np.testing.assert_array_equal(restored[k],obs[k])
        self.assertEqual(label,1)

    def test_review_requires_explicit_intervals_and_preserves_revisions(self):
        with tempfile.TemporaryDirectory() as folder:
            run=Path(folder)
            for name,value in [('manifest.json',dict(duration=10)),('frames.json',[]),('events.json',[])]:write_new(run/name,value)
            with self.assertRaises(ValueError):approve(run,'0:9',False)
            with self.assertRaises(ValueError):approve(run,'0:11',True)
            first=approve(run,'0.2:9',True)
            second=approve(run,None,True,'rejected')
            self.assertTrue(first.exists());self.assertTrue(second.exists())
            self.assertEqual(latest(run)[0]['status'],'rejected')

    def test_session_split_rejects_leakage(self):
        from types import SimpleNamespace as NS
        a=NS(samples=[(0,0,'halloween','same'),(1,0,'tokyo','t')])
        b=NS(samples=[(0,0,'halloween','same'),(1,0,'tokyo','v')])
        with self.assertRaisesRegex(ValueError,'session'):split_check(a,b)

    def test_native_export_roundtrip_and_checksums(self):
        import cv2
        from common import digest
        from dataset import export
        from train import Dataset
        c=dict(image_shape=[12,16,16],road_vision=dict())
        with tempfile.TemporaryDirectory() as folder:
            root=Path(folder);run=root/'sessions'/'halloween_train_fixture'/'run_0001'
            (run/'frames').mkdir(parents=True)
            rows=frames(8)
            for i,row in enumerate(rows):
                row['file']=f'{i:06d}.png'
                native=np.full((945,1920,3),10+i,np.uint8)
                self.assertTrue(cv2.imwrite(str(run/'frames'/row['file']),native))
                row['sha256']=digest(run/'frames'/row['file'])
            write_new(run/'manifest.json',dict(duration=.35,map='halloween',split='train',session='halloween_train_fixture'))
            write_new(run/'frames.json',rows);write_new(run/'events.json',key(10.025))
            approve(run,'0:0.35',True)
            checkpoint=root/'checkpoint';checkpoint.write_bytes(b'fixture')
            def crop(image,cfg):return cv2.resize(image,(16,16)).transpose(2,0,1)
            with patch('dataset.contract',return_value=c),patch('dataset.crop_rgb',side_effect=crop):
                exported=export(root,checkpoint,root/'export',20)
                before=digest(exported)
                export(root,checkpoint,root/'export',20,resume=True)
                self.assertEqual(digest(exported),before)
            self.assertEqual(read(exported)['status'],'COMPLETE')
            data=Dataset(exported.parent,'train')
            try:
                obs,action=data.sample(0)
                self.assertEqual(action,3);self.assertEqual(obs['image'].shape,(12,16,16))
                self.assertTrue((obs['image'][-3:]==10).all())
                self.assertFalse(obs['image'][:-3].any());self.assertFalse(obs['aux'][565:].any())
            finally:data.close()
            with (exported.parent/'run_0000'/'actions.npy').open('ab') as stream:stream.write(b'corrupt')
            with self.assertRaisesRegex(ValueError,'checksum'):Dataset(exported.parent,'train')

    def test_compaction_preserves_input_and_padding(self):
        from export_support import compact_indices
        from dataset import stacked
        images=np.arange(10*3*2*2,dtype=np.uint8).reshape(10,3,2,2)
        samples=[dict(indices=np.array([-1,2,5,8])),dict(indices=np.array([2,5,8,9]))]
        used,indices=compact_indices(samples)
        for sample,compact in zip(samples,indices):
            np.testing.assert_array_equal(stacked(images,sample['indices']),stacked(images[used],compact))

    def test_reviewed_states_are_causal_source_checked_and_do_not_mutate(self):
        from export_support import apply_reviewed
        run=Path.cwd()/'fixture';rows=[dict(file='000000.png',sha256='abc',state='UNCERTAIN')]
        reviewed={str((run/'frames/000000.png').resolve()):dict(sha256='abc',state='GAMEPLAY',split='train')}
        result,counts=apply_reviewed(rows,run,'train',reviewed)
        self.assertEqual(result[0]['state'],'GAMEPLAY');self.assertEqual(rows[0]['state'],'UNCERTAIN')
        self.assertEqual(counts,{'UNCERTAIN->GAMEPLAY':1})
        with self.assertRaises(ValueError):apply_reviewed(rows,run,'validation',reviewed)

    def test_writer_is_bounded_and_owns_pixels(self):
        entered=threading.Event();release=threading.Event();seen=[]
        def slow(path,image,*args):
            entered.set();release.wait(2);seen.append(image.copy());return True
        with patch('record.cv2.imwrite',slow):
            writer=FrameWriter();source=np.full((2,2,4),7,np.uint8)
            try:
                for i in range(4):self.assertTrue(writer.submit(str(i),source))
                self.assertTrue(entered.wait(1));source.fill(0)
                self.assertFalse(writer.submit('extra',source))
            finally:release.set();writer.close()
            self.assertTrue(all(np.all(image==7) for image in seen))

    def test_writer_does_not_hide_failed_saves(self):
        with patch('record.cv2.imwrite',return_value=False):
            writer=FrameWriter();writer.submit('x',np.zeros((2,2,4),np.uint8))
            with self.assertRaises(OSError):writer.close()

    def test_bc_updates_shared_features_preserves_value_head_and_saves_ppo(self):
        import torch
        torch.set_num_threads(1)
        import gymnasium as gym
        from stable_baselines3 import PPO
        from stable_baselines3.common.torch_layers import BaseFeaturesExtractor
        from train import fit
        class Features(BaseFeaturesExtractor):
            def __init__(self,space):
                super().__init__(space,8);self.aux=torch.nn.Linear(789,8)
            def forward(self,obs):return torch.tanh(self.aux(obs['aux']))
        class Env(gym.Env):
            observation_space=gym.spaces.Dict(dict(image=gym.spaces.Box(0,255,(12,16,16),np.uint8),
                                                    aux=gym.spaces.Box(0,1,(789,),np.float32)))
            action_space=gym.spaces.Discrete(5)
        class Data:
            def __init__(self,prefix):
                self.samples=[(0,i,'halloween' if i<5 else 'tokyo',prefix+str(i//5)) for i in range(10)]
                self.arrays=[dict(actions=np.arange(10)%5)]
            def sample(self,i):
                aux=np.zeros(789,np.float32);aux[500+i%5]=1
                return dict(image=np.zeros((12,16,16),np.uint8),aux=aux),i%5
        model=PPO('MultiInputPolicy',Env(),n_steps=4,batch_size=4,device='cpu',seed=7,
                  policy_kwargs=dict(features_extractor_class=Features,net_arch=dict(pi=[8],vf=[8])))
        initial=model.policy.features_extractor.aux.weight.detach().clone()
        value=model.policy.value_net.weight.detach().clone()
        with tempfile.TemporaryDirectory() as folder:
            fit(model,Data('train'),Data('val'),Path(folder),epochs=2,batch_size=10,lr=.001,mirror_probability=0)
            loaded=PPO.load(Path(folder)/'bc_best.zip',device='cpu')
            self.assertTrue(torch.equal(loaded.policy.value_net.weight,value))
            self.assertFalse(torch.equal(loaded.policy.features_extractor.aux.weight,initial))
            self.assertEqual(loaded.observation_space['aux'].shape,(789,))
            self.assertFalse(loaded.policy.optimizer.state)
            self.assertTrue(all(p.requires_grad for p in loaded.policy.parameters()))


if __name__=='__main__':unittest.main()
