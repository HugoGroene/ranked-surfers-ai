from collections import deque
from types import SimpleNamespace
import unittest
from unittest.mock import AsyncMock, patch

import gymnasium as gym
import numpy as np
from stable_baselines3.common.vec_env import DummyVecEnv

from train_rs_rl_early_death import LocalPracticeEnv
from train_rs_v612_ppo import parse_args
from tools.ppo_episode_reset import reload_after_time_limit
from rs_v612_gameplay_guard import GuardUncertainError


class CapturedController(gym.Env):
    step = LocalPracticeEnv.step
    _menu_votes = LocalPracticeEnv._menu_votes

    def __init__(self, seconds, death=False):
        self.action_space = gym.spaces.Discrete(5)
        self.observation_space = gym.spaces.Box(0, 255, (4, 2, 2), dtype=np.uint8)
        self.args = SimpleNamespace(episode_max_seconds=300., fps=30, threshold=.68, debug_every_seconds=99999.)
        self.frames = deque([np.zeros((2, 2), dtype=np.uint8)]*4, maxlen=4)
        self.menu_scores = deque([1.] if death else [], maxlen=8)
        self.suspected_since = None
        self.clear_streak = 0
        self.last_step = seconds-.05
        self.last_debug = seconds
        self.episode_start = 0.
        self.episode_number = 1
        self.last_raw = None
        self.death = death

    async def _step_browser(self, action):
        pass

    def _call(self, coroutine, **kwargs):
        coroutine.close()
        return np.ones((2, 2), dtype=np.uint8), 1. if self.death else 0., None

    def reset(self, *, seed=None, options=None):
        return np.zeros((4, 2, 2), dtype=np.uint8), {}


class LongEpisodeTests(unittest.TestCase):
    def test_default_300_and_explicit_historical_limit(self):
        self.assertEqual(parse_args(['--mode', 'check']).episode_max_seconds, 300.)
        self.assertEqual(parse_args(['--mode', 'check', '--episode-max-seconds', '15']).episode_max_seconds, 15.)

    def test_actual_controller_does_not_stop_at_15_or_240_seconds(self):
        for seconds in (15.1, 60., 240.):
            env = CapturedController(seconds)
            with patch('train_rs_rl_early_death.time.monotonic', return_value=seconds):
                _, _, death, truncated, _ = env.step(0)
            self.assertFalse(death)
            self.assertFalse(truncated)

    def test_sb3_receives_timeout_bootstrap_and_terminal_observation(self):
        vector = DummyVecEnv([lambda: CapturedController(300.1)])
        vector.reset()
        try:
            with patch('train_rs_rl_early_death.time.monotonic', return_value=300.1):
                obs, _, done, infos = vector.step([0])
            self.assertTrue(done[0])
            self.assertTrue(infos[0]['TimeLimit.truncated'])
            self.assertTrue(infos[0]['terminal_observation'][-1].all())
            self.assertFalse(obs.any())
        finally:
            vector.close()

    def test_real_death_wins_over_simultaneous_time_limit(self):
        env = CapturedController(300.1, death=True)
        with patch('train_rs_rl_early_death.time.monotonic', return_value=300.1):
            _, _, death, truncated, _ = env.step(0)
        self.assertTrue(death)
        self.assertFalse(truncated)


class TimeLimitReloadTests(unittest.IsolatedAsyncioTestCase):
    async def test_reload_then_normal_startup_without_key_injection(self):
        page = SimpleNamespace(evaluate=AsyncMock(return_value=True), goto=AsyncMock())
        env = SimpleNamespace(args=SimpleNamespace(url='http://localhost:8000', fullscreen=True),
                              page=page, _anchor_capture_region=AsyncMock(), _enter_game_fullscreen=AsyncMock())
        with patch('rs_practice_controller.find_canvas', new=AsyncMock(return_value='new canvas')):
            await reload_after_time_limit(env)
        self.assertEqual(env.canvas, 'new canvas')
        page.goto.assert_awaited_once()
        env._enter_game_fullscreen.assert_awaited_once()

    async def test_focus_loss_blocks_reload(self):
        page = SimpleNamespace(evaluate=AsyncMock(return_value=False), goto=AsyncMock())
        env = SimpleNamespace(args=SimpleNamespace(url='http://localhost:8000'), page=page)
        with self.assertRaises(GuardUncertainError):
            await reload_after_time_limit(env)
        page.goto.assert_not_awaited()
