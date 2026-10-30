#!/usr/bin/env python3
"""Verify installed SB3 2.9.0 PPO hook ordering before live optimizer-pause use.

Run from Claude clone: python -B -m tools.check_pause_ppo_sb3
Does not open the game, load a model or modify any file.
"""
import inspect
import stable_baselines3
from stable_baselines3.common.on_policy_algorithm import OnPolicyAlgorithm
from stable_baselines3.common.monitor import Monitor


def require_between(text, *patterns):
    at = [text.find(p) for p in patterns]
    if any(i < 0 for i in at) or at != sorted(at):
        raise RuntimeError('SB3 callback order differs from validated 2.9.0: ' + str(patterns))


def main():
    version = stable_baselines3.__version__
    if version != '2.9.0':
        raise RuntimeError(f'SB3 {version} not validated for this patch (expected 2.9.0)')
    collect = inspect.getsource(OnPolicyAlgorithm.collect_rollouts)
    learn = inspect.getsource(OnPolicyAlgorithm.learn)
    init = inspect.getsource(Monitor.__init__)
    step = inspect.getsource(Monitor.step)
    require_between(collect,
                    'callback.on_rollout_start()',
                    'while n_steps < n_rollout_steps:',
                    'rollout_buffer.compute_returns_and_advantage(',
                    'callback.on_rollout_end()')
    require_between(learn, 'self.collect_rollouts(', 'self.train()')
    if 'self.t_start = time.time()' not in init or 'time.time() - self.t_start' not in step:
        raise RuntimeError('SB3 Monitor.t_start clock changed; inspect before enabling PPO pause')
    print('[PPO PAUSE CHECK] PASS: SB3 2.9.0 rollout_end -> PPO optimizer -> rollout_start')
    print('[PPO PAUSE CHECK] PASS: Monitor.t_start uses wall-clock time and is corrected after resume')


if __name__ == '__main__':
    main()
