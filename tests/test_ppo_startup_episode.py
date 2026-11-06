import json
from pathlib import Path
import tempfile
import threading
import unittest
import gymnasium as gym
import numpy as np
import torch
from stable_baselines3 import PPO
from stable_baselines3.common.monitor import Monitor
from tools.ppo_startup_episode import discard_first_episode


class EpisodeEnv(gym.Env):
    observation_space=gym.spaces.Box(0,100,shape=(1,),dtype=np.float32)
    action_space=gym.spaces.Discrete(2)

    def __init__(self,*,truncate=False,fail=False):
        self.episode_number=0
        self.history=[]
        self.truncate=truncate
        self.fail=fail

    def reset(self,**kwargs):
        super().reset(**kwargs)
        self.episode_number+=1
        self.history.clear()
        return np.array([self.episode_number],np.float32),{}

    def step(self,action):
        if self.fail:raise RuntimeError('guard uncertain')
        self.history.append(action)
        done=len(self.history)==3
        return (np.array([self.episode_number],np.float32),
                1000. if self.episode_number==1 else 1.,
                done and not self.truncate,done and self.truncate,
                dict(duration=len(self.history),termination_reason='limit' if self.truncate else 'death'))


class StartupTests(unittest.TestCase):
    def test_excluded_from_real_ppo_monitor_and_next_episode_resets(self):
        with tempfile.TemporaryDirectory() as folder:
            raw=EpisodeEnv()
            monitor=Monitor(raw)
            model=PPO('MlpPolicy',monitor,n_steps=4,batch_size=4,n_epochs=1,device='cpu',seed=7)
            weights={key:value.clone() for key,value in model.policy.state_dict().items()}
            report=discard_first_episode(raw,lambda obs:model.predict(obs)[0],threading.Event(),Path(folder))
            self.assertEqual(report['reward'],3000.)
            self.assertEqual(report['status'],'EXCLUDED')
            self.assertEqual(model.num_timesteps,0)
            self.assertEqual(model.rollout_buffer.pos,0)
            self.assertEqual(monitor.get_episode_rewards(),[])
            self.assertTrue(monitor.needs_reset)
            for key,value in model.policy.state_dict().items():self.assertTrue(torch.equal(value,weights[key]))
            # The ordinary SB3 setup resets through Monitor, clearing the raw history.
            _,callback=model._setup_learn(4,reset_num_timesteps=False)
            self.assertEqual(raw.episode_number,2)
            self.assertEqual(raw.history,[])
            self.assertTrue(model._last_episode_starts[0])
            callback.on_training_start({}, {})
            model.collect_rollouts(model.env,callback,model.rollout_buffer,4)
            self.assertEqual(model.num_timesteps,4)
            self.assertTrue(np.all(model.rollout_buffer.observations>=2))
            self.assertTrue(np.all(model.rollout_buffer.rewards==1))
            self.assertEqual([row['r'] for row in model.ep_info_buffer],[3.])
            self.assertEqual(monitor.get_episode_rewards(),[3.])
            model.env.close()

    def test_independent_workers_and_truncated_first_episode(self):
        with tempfile.TemporaryDirectory() as folder:
            for i in range(2):
                output=Path(folder)/str(i);output.mkdir()
                raw=EpisodeEnv(truncate=True)
                result=discard_first_episode(raw,lambda obs:0,threading.Event(),output)
                self.assertEqual(result['episode'],1)
                self.assertTrue(result['truncated'])
                self.assertFalse(raw._training_warmup_active)

    def test_guard_failure_propagates_and_is_not_a_completed_warmup(self):
        with tempfile.TemporaryDirectory() as folder:
            raw=EpisodeEnv(fail=True)
            with self.assertRaisesRegex(RuntimeError,'guard uncertain'):
                discard_first_episode(raw,lambda obs:0,threading.Event(),Path(folder))
            report=json.loads((Path(folder)/'startup_episode.json').read_text())
            self.assertEqual(report['status'],'FAILED')
            self.assertEqual(report['steps'],0)
            self.assertFalse(raw._training_warmup_active)

    def test_abort_and_timeout_never_silently_start_training(self):
        with tempfile.TemporaryDirectory() as folder:
            abort=threading.Event();abort.set()
            raw=EpisodeEnv()
            with self.assertRaisesRegex(RuntimeError,'aborted'):
                discard_first_episode(raw,lambda obs:0,abort,Path(folder))
            self.assertEqual(raw.episode_number,0)
            with self.assertRaises(TimeoutError):
                discard_first_episode(raw,lambda obs:0,threading.Event(),Path(folder),max_seconds=0)
            self.assertEqual(raw.history,[])


if __name__=='__main__':unittest.main()
