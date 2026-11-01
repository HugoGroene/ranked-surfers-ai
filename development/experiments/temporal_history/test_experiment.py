"""CPU-only adapter, protocol, planning and model tests; no live browser."""
import asyncio
import json
from pathlib import Path
import tempfile
import time
import unittest
from unittest.mock import patch
import numpy as np

from experiment_config import configuration,PRESETS,ROOT
from live_adapter import ContextHistory,CapturePump
from batch import jobs_for,checked_result,summarize


class RateIntegrationTests(unittest.TestCase):
    def test_pump_stops_when_capture_swallows_cancellation(self):
        async def check():
            from types import SimpleNamespace
            started=asyncio.Event()
            class Source:
                last_timing={}
                async def capture(self):
                    started.set()
                    try:await asyncio.sleep(100)
                    except asyncio.CancelledError:pass
                    return np.zeros((2,2,4),np.uint8)
            pump=CapturePump(Source(),lambda *a:None,60,lambda:1)
            await started.wait()
            await asyncio.wait_for(pump.close(),1)
            self.assertTrue(pump.task.done())
            self.assertEqual(pump.count,0)
        asyncio.run(check())

    def test_real_inherited_parser_accepts_all_presets_and_experimental_rates(self):
        import argparse
        import sys
        import train_rs_v3_ppo as trainer
        from run_trial import install_rate_bridge
        original=trainer.v2_args
        original_argv=sys.argv[:]
        with patch.object(trainer,'v2_args',original):
            install_rate_bridge(trainer)
            once=trainer.v2_args
            install_rate_bridge(trainer)
            self.assertIs(once,trainer.v2_args)
            for rate in [c['decision_hz'] for c in PRESETS.values()]+[1.,12.5,60.]:
                with self.subTest(rate=rate):
                    a=argparse.Namespace(url='http://localhost:8000',fps=rate,max_minutes=1.,fullscreen=True)
                    cfg=trainer.v2_args(a)  # Real V3 -> V2 argparse, no environment/browser.
                    self.assertEqual(cfg.fps,rate)
                    self.assertEqual(a.fps,rate)
                    self.assertEqual(cfg.pretrain_updates,0)
                    self.assertEqual(cfg.bc_updates,0)
                    self.assertEqual(sys.argv,original_argv)

    def test_rate_bridge_rejects_invalid_values_before_parser(self):
        from types import SimpleNamespace
        from run_trial import install_rate_bridge
        from unittest.mock import Mock
        original=Mock()
        trainer=SimpleNamespace(v2_args=original)
        install_rate_bridge(trainer)
        for rate in [0,61,float('nan'),float('inf')]:
            with self.assertRaises(ValueError):trainer.v2_args(SimpleNamespace(fps=rate))
        original.assert_not_called()


class ContextTests(unittest.TestCase):
    def test_installed_adapter_updates_spaces_contract_and_endpoints(self):
        from types import SimpleNamespace
        import gymnasium as gym
        import tools.ppo_headless_env as headless
        import tools.ppo_observation_contract as contracts
        from live_adapter import install
        class Base:
            def __init__(self):
                self.observation_space=gym.spaces.Dict({
                    'image':gym.spaces.Box(0,255,(12,2,2),np.uint8),
                    'aux':gym.spaces.Box(0,1,(557,),np.float32)})
                self._headless_capture_timing={'request_started_at':time.perf_counter()}
            def _pack(self):return dict(image=np.full((12,2,2),9,np.uint8),aux=np.zeros(557,np.float32)),False,0
            def pause_for_ppo_optimizer(self):self._pack();return 0
            def refresh_after_ppo_optimizer(self):return self._pack()[0],None,time.perf_counter()
        trainer=SimpleNamespace(build_env_class=lambda:Base,shape_description=lambda:{'image_frames':4})
        with patch.object(headless,'install',lambda *a,**k:None),patch.object(contracts,'requested_contract',
                lambda a:dict(version=1,image_shape=[12,2,2],aux_dim=557)):
            install(trainer,configuration(12,500,60,30))
            raw=trainer.build_env_class()()
            obs,_,_=raw._pack()
            self.assertTrue(raw.observation_space.contains(obs))
            self.assertIs(raw._learning_current_observation,obs)
            self.assertEqual(contracts.requested_contract(None)['image_shape'][0],36)
            self.assertEqual(trainer.shape_description()['image_frames'],12)
            raw.pause_for_ppo_optimizer()
            self.assertIsNone(raw.temporal.token())
            refreshed,_,_=raw.refresh_after_ppo_optimizer()
            self.assertTrue(raw.observation_space.contains(refreshed))
            self.assertIsNone(raw.temporal.token())

    def test_missing_context_has_valid_latest_not_fake_copies(self):
        h=ContextHistory(8,.5)
        newest=np.full((3,2,2),7,np.uint8)
        image,aux=h.observation(newest,1.)
        np.testing.assert_array_equal(aux[8:],[0]*7+[1])
        self.assertFalse(image[:-3].any())
        np.testing.assert_array_equal(image[-3:],newest)

    def test_no_future_frames_and_exact_latest(self):
        h=ContextHistory(6,.5)
        h.observation(np.zeros((3,2,2),np.uint8),0.)
        for i in range(61):h.add(np.full((3,2,2),i,np.uint8),i/60,h.token())
        image,aux=h.observation(np.full((3,2,2),99,np.uint8),.98)
        self.assertEqual(image[-1,0,0],99)
        self.assertLessEqual(image[:-3].max(),58)
        self.assertTrue(aux[6:].all())

    def test_reset_rejects_inflight_previous_generation(self):
        h=ContextHistory(4,.1)
        h.observation(np.zeros((3,2,2),np.uint8),time.perf_counter())
        old=h.token()
        h.clear()
        h.observation(np.zeros((3,2,2),np.uint8),time.perf_counter())
        h.add(np.ones((3,2,2),np.uint8),time.perf_counter(),old)
        self.assertEqual(len(h.items),0)

    def test_no_promotion_during_resync(self):
        h=ContextHistory(4,.5)
        h.observation(np.zeros((3,2,2),np.uint8),1.,enable=False)
        self.assertIsNone(h.token())
        h.add(np.ones((3,2,2),np.uint8),2.,None)
        self.assertEqual(len(h.items),0)

    def test_owned_history_and_bounded_memory(self):
        h=ContextHistory(12,.5)
        f=np.zeros((3,2,2),np.uint8)
        h.observation(f,0)
        for i in range(600):
            f.fill(i%255);h.add(f,i/60,h.token())
        f.fill(255)
        self.assertLess(len(h.items),40)
        self.assertNotEqual(h.items[-1][1][0,0,0],255)


class FakeSource:
    def __init__(self):
        self.serial=0
        self.last_timing={}
        self.active=0
        self.max_active=0
    async def capture(self):
        self.active+=1;self.max_active=max(self.max_active,self.active)
        started=time.perf_counter()
        await asyncio.sleep(.001)
        self.serial+=1
        self.last_timing=dict(request_started_at=started,electron_serial=self.serial)
        self.active-=1
        return np.full((2,2,4),self.serial,np.uint8)


class PumpTests(unittest.IsolatedAsyncioTestCase):
    async def test_serial_and_timing_belong_to_delivered_pixels(self):
        source=FakeSource()
        pump=CapturePump(source,lambda *args:None,120,lambda:1)
        try:
            frame=await pump.capture()
            timing=dict(pump.last_timing)
            self.assertEqual(frame[0,0,0],timing['electron_serial'])
            await asyncio.sleep(.025)
            self.assertEqual(pump.last_timing,timing)
            second=await pump.capture()
            self.assertGreater(int(second[0,0,0]),int(frame[0,0,0]))
            self.assertEqual(source.max_active,1)
        finally:await pump.close()
        self.assertTrue(pump.task.done())

    async def test_idle_during_pause_until_guard_requests_capture(self):
        source=FakeSource()
        pump=CapturePump(source,lambda *args:None,120)
        try:
            await asyncio.sleep(.02)
            self.assertEqual(source.serial,0)
            await pump.capture()
            count=source.serial
            await asyncio.sleep(.025)
            self.assertEqual(source.serial,count)
        finally:await pump.close()

    async def test_error_propagates_and_does_not_return_cached_frame(self):
        source=FakeSource()
        async def fail():raise RuntimeError('source failed')
        source.capture=fail
        pump=CapturePump(source,lambda *args:None,60)
        try:
            with self.assertRaisesRegex(RuntimeError,'source failed'):await pump.capture()
        finally:await pump.close()


class BatchTests(unittest.TestCase):
    def test_real_cli_builds_scratch_and_evaluation_args_without_launch(self):
        from types import SimpleNamespace
        import train_rs_v612_ppo as trainer
        from run_trial import trainer_args
        options=SimpleNamespace(mode='train',output=Path('train'),minutes=30,seed=123,
            state_model=Path('dummy.pt'),checkpoint=None,episodes=100,decision_hz=60)
        with patch.object(Path,'is_file',return_value=True):
            a=trainer_args(trainer,options)
            self.assertIsNone(a.resume)
            self.assertTrue(a.settled_learning)
            self.assertEqual(a.fps,60)
            self.assertEqual(a.press_delay_ms,5)
            options.mode='eval';options.checkpoint=Path('latest.zip')
            a=trainer_args(trainer,options)
            self.assertFalse(a.settled_learning)
            self.assertEqual(a.eval_episodes,100)
            self.assertEqual(a.eval_policy,'deterministic')

    def test_settings_and_training_before_evaluations(self):
        jobs=jobs_for(list(PRESETS),[123,456],30,100,60,Path('batch'),42)
        self.assertEqual(len(jobs),24)
        self.assertTrue(all(j['mode']=='train' for j in jobs[:12]))
        self.assertTrue(all(j['mode']=='eval' for j in jobs[12:]))
        self.assertEqual(len({j['output'] for j in jobs}),24)
        for train,evaluation in zip(jobs[:12],jobs[12:]):
            self.assertEqual(train['name'],evaluation['name'])
            for flag in ('--frames','--span-ms','--capture-hz','--decision-hz','--seed'):
                self.assertEqual(train['command'][train['command'].index(flag)+1],
                                 evaluation['command'][evaluation['command'].index(flag)+1])
            self.assertNotIn('--checkpoint',train['command'])

    def test_configuration_validation(self):
        for args in ((12,100,30,30),(5,500,60,30),(8,500,0,30),(8,500,60,61)):
            with self.assertRaises(ValueError):configuration(*args)

    def test_failed_trial_never_eligible(self):
        with tempfile.TemporaryDirectory() as temp:
            root=Path(temp);train=root/'train';train.mkdir()
            (train/'trial_report.json').write_text(json.dumps(dict(status='STOPPED')))
            job=dict(name='test',mode='train',output=str(train))
            with self.assertRaises(RuntimeError):checked_result(job)
            report=summarize(dict(jobs=[job]))
            self.assertFalse(report['rows'][0]['eligible_for_comparison'])


class ModelTests(unittest.TestCase):
    def test_setup_failure_still_closes_environment_and_reports(self):
        import contextlib
        import io
        from types import SimpleNamespace
        from unittest.mock import Mock,patch
        from run_trial import train
        closed=Mock()
        trainer=SimpleNamespace(create_live_env=lambda *a:(SimpleNamespace(close=closed),SimpleNamespace(training_steps=0)),
                                make_model=lambda *a:SimpleNamespace(),build_report=lambda *a,**k:{})
        with tempfile.TemporaryDirectory() as temp,contextlib.redirect_stderr(io.StringIO()), \
             patch('tools.ppo_frame_state_cache.install_frame_state_cache'), \
             patch('tools.ppo_observation_contract.attach_contract',side_effect=ValueError('setup failure')):
            output=Path(temp)
            self.assertFalse(train(trainer,SimpleNamespace(),output,{},1))
            report=json.loads((output/'trial_report.json').read_text())
            self.assertEqual(report['status'],'STOPPED')
            self.assertIn('setup failure',report['error'])
            self.assertEqual(report['finalization_errors'],[])
            closed.assert_called_once()

    def test_trial_runner_saves_and_audits_complete_rollout(self):
        import contextlib
        import io
        from types import SimpleNamespace
        import torch
        from stable_baselines3 import PPO
        from stable_baselines3.common.monitor import Monitor
        from tests.test_ppo_settled_learning import ProtocolEnv
        from train_rs_v612_ppo import build_callback
        from run_trial import train
        torch.set_num_threads(1)
        raw=ProtocolEnv('ordinary')
        raw._guard_hard_stopped=False
        raw._read_screen_state=lambda:('GAMEPLAY',{'GAMEPLAY':1.})
        raw.training_steps=0
        monitored=Monitor(raw)
        trainer=SimpleNamespace(create_live_env=lambda *args:(monitored,raw),
            make_model=lambda env,a:PPO('MultiInputPolicy',env,n_steps=4,batch_size=4,n_epochs=1,
                policy_kwargs={'net_arch':[8]},device='cpu',seed=123),
            build_callback=build_callback,build_report=lambda *args,**kwargs:{'synthetic':True})
        args=SimpleNamespace(pause_ppo_optimizer=True,resync_rollouts=True,recover_rollout_menu=False,
            audit_reward_timing=True,log_every_seconds=1e9,save_every_minutes=1e9,max_minutes=100,
            settled_learning=True,steps=128)
        with tempfile.TemporaryDirectory() as temp,contextlib.redirect_stdout(io.StringIO()):
            output=Path(temp)
            self.assertTrue(train(trainer,args,output,
                dict(version=1,framing_profile='canvas_v1',detector_mode='off'),.000001))
            report=json.loads((output/'trial_report.json').read_text())
            self.assertEqual(report['optimized_transitions'],4)
            self.assertEqual(report['audit']['status'],'PASS_SETTLED_LEARNING')
            self.assertTrue((output/'initial.zip').exists())
            restored=PPO.load(output/'latest.zip',device='cpu')
            self.assertEqual(restored.num_timesteps,4)

    def test_cpu_gradient_save_load_all_channel_counts(self):
        import gymnasium as gym
        import torch
        from stable_baselines3 import PPO
        from train_rs_v3_ppo import RGBHazardExtractor
        from tools.ppo_observation_contract import validate_checkpoint
        torch.set_num_threads(1)
        class Fake(gym.Env):
            def __init__(self,n):
                self.observation_space=gym.spaces.Dict({
                    'image':gym.spaces.Box(0,255,(n*3,64,96),np.uint8),
                    'aux':gym.spaces.Box(0,1,(557+n*2,),np.float32)})
                self.action_space=gym.spaces.Discrete(5)
                self.t=0
            def obs(self):return {'image':np.full(self.observation_space['image'].shape,80+self.t%30,np.uint8),
                                  'aux':np.full(self.observation_space['aux'].shape,.2,np.float32)}
            def reset(self,*,seed=None,options=None):
                super().reset(seed=seed);self.t=0;return self.obs(),{}
            def step(self,action):
                self.t+=1;return self.obs(),float(action==self.t%5),self.t%8==0,False,{}
        for n in (4,6,8,12):
            with self.subTest(frames=n),tempfile.TemporaryDirectory() as temp:
                env=Fake(n)
                model=PPO('MultiInputPolicy',env,device='cpu',n_steps=16,batch_size=8,n_epochs=1,seed=123,
                          policy_kwargs=dict(features_extractor_class=RGBHazardExtractor))
                self.assertEqual(model.observation_space['image'].shape,(n*3,64,96))
                before=model.policy.features_extractor.rgb.state_dict()
                before={k:v.clone() for k,v in before.items()}
                model.learn(16)
                self.assertTrue(any(not torch.equal(before[k],v) for k,v in model.policy.features_extractor.rgb.state_dict().items()))
                model.observation_contract=dict(version=1,frames=n,span_ms=500)
                path=Path(temp)/'model.zip';model.save(path)
                validate_checkpoint(path,model.observation_contract)
                with self.assertRaises(ValueError):validate_checkpoint(path,dict(version=1,frames=n,span_ms=100))
                restored=PPO.load(path,device='cpu')
                obs,_=env.reset()
                np.testing.assert_array_equal(model.predict(obs,deterministic=True)[0],restored.predict(obs,deterministic=True)[0])
                env.close()


if __name__=='__main__':unittest.main()
