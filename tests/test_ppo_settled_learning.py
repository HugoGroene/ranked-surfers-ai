"""Real SB3 collector/optimizer with synthetic local-environment protocol."""
import importlib.util
import io
import contextlib
import json
from pathlib import Path
import tempfile
import time
from types import SimpleNamespace
import unittest
import zipfile
from unittest.mock import patch

import gymnasium as gym
import numpy as np
import torch
from stable_baselines3 import PPO
from stable_baselines3.common.monitor import Monitor
from stable_baselines3.common.vec_env import DummyVecEnv

from tools.ppo_settled_learning import (SettledPPO,TimedDictBuffer,VerifiedProgressRewards,
    audit_settled_ledger,reference_seconds_for)
from train_rs_v612_ppo import build_callback,parse_args


class ProtocolEnv(gym.Env):
    def __init__(self,mode):
        self.mode=mode
        self.action_space=gym.spaces.Discrete(5)
        self.observation_space=gym.spaces.Dict({'image':gym.spaces.Box(-1000,1000,(2,),np.float32),
            'aux':gym.spaces.Box(-1000,1000,(1,),np.float32)})
        self.survival_rewards=VerifiedProgressRewards()
        self.steps=0
        self.boundaries=0
        self.episode_number=0
        self._ppo_boundary_menu_pending=False
        self._ppo_boundary_menu_duration_at=None
        self._ppo_boundary_menu_started_at=None
        self._ppo_pause_pending=None
        self._learning_passive_kind=None
        self.action_context=SimpleNamespace(last_action_times=[None]*5)

    def obs(self):
        return {'image':np.array([self.code,0],np.float32),'aux':np.array([self.duration],np.float32)}

    def reset(self,*,seed=None,options=None):
        super().reset(seed=seed)
        self.episode_number+=1
        self.duration=0.
        self.code=0
        self.survival_rewards.reset()
        self.episode_start=self.last_step=time.monotonic()
        self.episode_started=self.last_capture_end=time.perf_counter()
        return self.obs(),{}

    def step(self,action):
        kind=self._learning_passive_kind
        self._learning_passive_kind=None
        if kind=='boundary' and not self._ppo_boundary_menu_pending:
            return self.obs(),0.,False,False,self.info(0.,dict(survival=0.,milestones=0.,death=0.),False,None)
        terminal=self._ppo_boundary_menu_pending
        if not kind:
            self.steps+=1
            terminal=terminal or self.mode=='death' and self.steps==2
        truncated=self.mode=='timeout' and self.steps==2 and not kind
        self._ppo_boundary_menu_pending=False
        self.duration+=.25 if kind=='resync' else .05
        self.code+=1
        reward,parts=self.survival_rewards.step(self.duration,terminal)
        self.last_capture_end=time.perf_counter()
        return self.obs(),reward,terminal,truncated,self.info(reward,parts,
            bool(kind or terminal or self.mode=='critic_only'),
            None if kind or terminal or action==0 or self.mode=='critic_only' else 'movement')

    def info(self,reward,parts,suppressed,key):
        return dict(duration=self.duration,reward_duration_s=self.duration,
            reward_survival=parts['survival'],reward_milestones=parts['milestones'],
            reward_death=parts['death'],reward_v61=reward,
            learning_action_suppressed=suppressed,learning_dispatched_key=key)

    def pause_for_ppo_optimizer(self):
        self.boundaries+=1
        if ((self.mode=='boundary_death' and self.boundaries==1) or
                (self.mode=='final_boundary_death' and self.boundaries==2)):
            self._ppo_boundary_menu_duration_at=self.duration
            raise ValueError('Not verified GAMEPLAY at rollout end: MENU, menu=0.900')
        self._ppo_pause_pending=True
        self._learning_boundary_observation=self.obs()
        return 0.

    def confirm_post_optimizer_menu(self):
        return True,{'checks':[{}]*4}

    def resume_after_ppo_optimizer(self,monitor):
        self._ppo_pause_pending=None
        return dict(paused_wall_seconds=.01,corrected_episode_timer=True,
            corrected_action_timer=True,corrected_monitor_timer=True,active_episode_clock_drift_s=0.)

    def refresh_after_ppo_optimizer(self):
        if self.mode=='post_resume_death' and self.boundaries==1:
            self._ppo_post_resume_menu_candidate=(time.monotonic()-.001,time.perf_counter()-.001,time.time()-.001)
            return None,'MENU',None
        self.code+=100
        self.last_capture_end=time.perf_counter()
        return self.obs(),None,self.last_capture_end


class SettledTests(unittest.TestCase):
    def test_pause_aware_cli_defaults_to_corrected_and_legacy_is_explicit(self):
        args=['--mode','train','--resume','source.zip','--pause-ppo-optimizer','--resync-rollouts']
        with patch('pathlib.Path.is_file',return_value=True):
            self.assertTrue(parse_args(args).settled_learning)
            self.assertFalse(parse_args(args+['--legacy-learning']).settled_learning)

    def test_corrected_cli_rejects_abandoned_episode_recovery(self):
        args=['--mode','train','--resume','source.zip','--pause-ppo-optimizer',
            '--resync-rollouts','--recover-rollout-menu']
        with patch('pathlib.Path.is_file',return_value=True),contextlib.redirect_stderr(io.StringIO()):
            with self.assertRaises(SystemExit):parse_args(args)

    def run_case(self,mode,deadline=False):
        raw=ProtocolEnv(mode)
        vec=DummyVecEnv([lambda:Monitor(raw)])
        try:
            with tempfile.TemporaryDirectory() as temp:
                model=PPO('MultiInputPolicy',vec,n_steps=4,batch_size=4,n_epochs=1,
                    policy_kwargs={'net_arch':[8]},device='cpu',seed=123)
                model.__class__=SettledPPO
                model._settled_last_endpoint_capture={'serial':-1,'captured_at':-1}
                model.configure_settled(raw,Path(temp),.05,100)
                self.assertIsNone(model._settled_last_endpoint_capture)
                if mode=='checkpoint':model._settled_checkpoint_due=True
                before_head=[p.detach().clone() for p in model.policy.action_net.parameters()]
                if mode=='critic_only':
                    for p in model.policy.action_net.parameters():
                        model.policy.optimizer.state[p]=dict(step=torch.tensor(1.),
                            exp_avg=torch.ones_like(p),exp_avg_sq=torch.ones_like(p))
                if deadline:model._settled_deadline=0
                args=SimpleNamespace(pause_ppo_optimizer=True,resync_rollouts=True,
                    recover_rollout_menu=False,audit_reward_timing=True,
                    log_every_seconds=1e9,save_every_minutes=1e9,max_minutes=100,
                    settled_learning=True)
                callback=build_callback(args,Path(temp),raw)
                with contextlib.redirect_stdout(io.StringIO()):
                    model.learn(total_timesteps=8,callback=callback)
                rows=[json.loads(x) for x in model._settled_ledger.read_text().splitlines()]
                result=audit_settled_ledger(model._settled_ledger,minimum=4,gamma=model.gamma)
                self.assertEqual(result['failures'],[])
                stats=dict(model._settled_stats)
                stats['actor_head_unchanged']=all(torch.equal(a,b) for a,b in
                    zip(before_head,model.policy.action_net.parameters()))
                if mode=='checkpoint':
                    with zipfile.ZipFile(Path(temp)/'latest.zip') as z:
                        stats['periodic_checkpoint_optimized']=json.loads(z.read('data'))['_settled_stats']['optimized_transitions']
                return rows,stats
        finally:
            vec.close()

    def test_boundary_death_is_in_preceding_rollout_before_optimization(self):
        rows,stats=self.run_case('boundary_death')
        collected=[r for r in rows if r['event']=='collected']
        self.assertTrue(collected[3]['done'])
        self.assertLess(collected[3]['environment_reward'],0)
        self.assertEqual(sum(s['pieces']['death'] for s in collected[3]['reward_substeps']),-1)
        self.assertEqual(rows[4]['event'],'optimized')
        self.assertFalse(any(s.get('pieces',{}).get('death') for s in collected[4]['reward_substeps']))
        self.assertEqual(stats['optimized_transitions'],8)

    def test_resync_has_no_sampled_action_or_actor_gradient(self):
        rows,_=self.run_case('normal')
        bridge=[r for r in rows if r.get('kind')=='passive_resync']
        self.assertEqual(len(bridge),1)
        self.assertIsNone(bridge[0]['sampled_action'])
        self.assertFalse(bridge[0]['actor_mask'])
        self.assertGreaterEqual(bridge[0]['active_seconds'],.25)
        self.assertIsNone(bridge[0]['command'])

    def test_final_boundary_penalty_is_not_deferred_beyond_training(self):
        rows,stats=self.run_case('final_boundary_death')
        last=[r for r in rows if r['event']=='collected'][-1]
        self.assertTrue(last['done'])
        self.assertEqual(sum(s['pieces']['death'] for s in last['reward_substeps']),-1.)
        self.assertEqual(stats['optimized_transitions'],8)

    def test_wall_time_stop_finishes_and_optimizes_full_rollout(self):
        rows,stats=self.run_case('normal',True)
        self.assertEqual(stats['collected_transitions'],4)
        self.assertEqual(stats['optimized_transitions'],4)
        self.assertEqual(stats['discarded_partial_transitions'],0)

    def test_passive_only_batches_do_not_move_actor_head_even_with_adam_momentum(self):
        _,stats=self.run_case('critic_only')
        self.assertTrue(stats['actor_head_unchanged'])
        self.assertEqual(stats['actor_transitions'],0)

    def test_post_resume_death_is_passive_terminal_without_actor(self):
        rows,_=self.run_case('post_resume_death')
        terminal=[r for r in rows if r.get('kind')=='passive_resync'][0]
        self.assertTrue(terminal['done'])
        self.assertIsNone(terminal['sampled_action'])
        self.assertFalse(terminal['actor_mask'])

    def test_periodic_checkpoint_is_written_after_completed_optimization(self):
        _,stats=self.run_case('checkpoint')
        self.assertEqual(stats['periodic_checkpoint_optimized'],4)

    def test_timeout_bootstraps_but_death_does_not(self):
        rows,_=self.run_case('timeout')
        terminal=[r for r in rows if r.get('truncated')]
        self.assertEqual(len(terminal),1)
        self.assertNotEqual(terminal[0]['environment_reward'],terminal[0]['buffer_reward'])
        rows,_=self.run_case('death')
        terminal=[r for r in rows if r.get('done')]
        self.assertEqual(terminal[0]['environment_reward'],terminal[0]['buffer_reward'])
        self.assertFalse(terminal[0]['actor_mask'])

    def test_verified_rewards_are_cadence_invariant_and_terminal_does_not_add_time(self):
        totals=[]
        for dt in (.05,.04):
            reward=VerifiedProgressRewards()
            totals.append(sum(reward.step(round(i*dt,8),False)[0] for i in range(1,int(1/dt)+1)))
            value,parts=reward.step(1.,True)
            self.assertEqual(parts['survival'],0)
            self.assertEqual(value,-1)
        self.assertAlmostEqual(totals[0],totals[1])

    def test_duration_discount_composes_across_subdivided_time(self):
        gamma,lam=.995,.95
        self.assertAlmostEqual(gamma**(.1/.05),(gamma**(.04/.05))*(gamma**(.06/.05)))
        self.assertAlmostEqual((gamma*lam)**2,(gamma*lam)**(.04/.05)*(gamma*lam)**(.06/.05))

    def test_resumed_time_unit_is_inherited_unless_explicitly_overridden(self):
        self.assertEqual(reference_seconds_for(SimpleNamespace(),None),.05)
        model=SimpleNamespace(settled_reference_seconds=.075)
        self.assertEqual(reference_seconds_for(model,None),.075)
        self.assertEqual(reference_seconds_for(model,.05),.05)

    def test_buffer_time_gae_matches_hand_computation(self):
        obs=gym.spaces.Dict({'x':gym.spaces.Box(-1,1,(1,),np.float32)})
        buf=TimedDictBuffer(2,obs,gym.spaces.Discrete(5),gamma=.9,gae_lambda=.8,device='cpu')
        buf.reference_seconds=.05
        buf.rewards[:]=[[1.],[2.]]
        buf.values[:]=0
        buf.active_seconds[:]=[[.1],[.05]]
        buf.compute_returns_and_advantage(torch.tensor([0.]),np.array([True]))
        self.assertAlmostEqual(float(buf.returns[1,0]),2)
        self.assertAlmostEqual(float(buf.returns[0,0]),1+(.9*.8)**2*2,places=6)


if __name__=='__main__':unittest.main()
