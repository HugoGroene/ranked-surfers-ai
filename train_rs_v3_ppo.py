#!/usr/bin/env python3
"""Ranked Surfers LOCAL Practice: PPO with RGB history + optional async YOLO.

Files required in the same folder (from the earlier tested setup):
  train_rs_v2_verify.py, rs_v2_timing_probe_ab.py,
  rs_v2_live_hazards_v2.py, and their existing dependencies.

Examples:
  python -u train_rs_v3_ppo.py --mode check
  python -u train_rs_v3_ppo.py --mode smoke --fullscreen --smoke-seconds 45
  python -u train_rs_v3_ppo.py --mode train --fullscreen --max-minutes 12 \
    --run-name rgb_hazards_trial --steps 6000 \
    --detector ai_data/v2_yolo_dataset/runs/hazards/weights/best.pt
  python -u train_rs_v3_ppo.py --mode eval --fullscreen --run-name eval_trial \
    --resume ai_data/rl_v2_runs/rgb_hazards_trial/latest.zip --eval-episodes 20

IMPORTANT: Uses only http://localhost:8000 (local Practice). This is a
short experimental trainer, not evidence that PPO has learned to play.
Existing V2 checkpoints have a different observation space and CANNOT be resumed.
"""
from __future__ import annotations

import argparse
from collections import Counter, deque
from concurrent.futures import ThreadPoolExecutor
from datetime import datetime
import json
from pathlib import Path
import sys
import time

import cv2
import gymnasium as gym
from gymnasium import spaces
import numpy as np
import torch
from torch import nn
from stable_baselines3 import PPO
from stable_baselines3.common.callbacks import BaseCallback
from stable_baselines3.common.monitor import Monitor
from stable_baselines3.common.torch_layers import BaseFeaturesExtractor

import train_rs_v2_verify as v2
from rs_v2_timing_probe_ab import TimedEnv
from rs_v2_live_hazards_v2 import (FEATURE_COUNT, detect, hazard_features)
from train_rs_rl_early_death import TinyCNN

HISTORY = 4
ACTIONS = 5
FRAME_INTERVALS = 3  # Actual capture intervals among the four stacked RGB frames
AUX_DIM = FEATURE_COUNT + HISTORY * ACTIONS + FRAME_INTERVALS + 2


class RGBHazardExtractor(BaseFeaturesExtractor):
    """Separate visual and numerical branches; no collision rules encoded."""
    def __init__(self, observation_space: spaces.Dict, features_dim: int = 192):
        super().__init__(observation_space, features_dim)
        if features_dim != 192:
            raise ValueError('The RGB+aux extractor currently uses 192 features.')
        image_space = observation_space.spaces['image']
        aux_size = int(np.prod(observation_space.spaces['aux'].shape))
        self.rgb = TinyCNN(image_space, features_dim=128)
        self.aux = nn.Sequential(
            nn.Linear(aux_size, 96), nn.ReLU(),
            nn.Linear(96, 64), nn.ReLU(),
        )

    def forward(self, obs: dict[str, torch.Tensor]) -> torch.Tensor:
        return torch.cat((self.rgb(obs['image']), self.aux(obs['aux'])), dim=1)


def build_observation_space() -> spaces.Dict:
    return spaces.Dict({
        'image': spaces.Box(low=0, high=255,
                            shape=(v2.vision.CHANNELS, v2.vision.HEIGHT, v2.vision.WIDTH),
                            dtype=np.uint8),
        # Includes padded YOLO slots, confidence, age, recent actions, time data.
        'aux': spaces.Box(low=0., high=1., shape=(AUX_DIM,), dtype=np.float32),
    })


def form_aux(hazard_vec: np.ndarray, history, episode_seconds: float,
             frame_intervals, detection_age_ms: float | None) -> np.ndarray:
    """Normalize numerical context to [0, 1] without asserting lane safety."""
    if hazard_vec.shape != (FEATURE_COUNT,):
        raise ValueError(f'Hazard vector wrong shape: {hazard_vec.shape}')
    recent = np.zeros((HISTORY, ACTIONS), dtype=np.float32)
    for i, act in enumerate(list(history)[-HISTORY:]):
        recent[HISTORY - min(len(history), HISTORY) + i, int(act)] = 1.
    intervals = np.zeros(FRAME_INTERVALS, dtype=np.float32)
    for i, interval in enumerate(list(frame_intervals)[-FRAME_INTERVALS:]):
        intervals[FRAME_INTERVALS-min(len(frame_intervals), FRAME_INTERVALS)+i] = np.clip(interval / .5, 0, 1)
    info = np.array([
        np.clip(episode_seconds / 60., 0, 1),
        np.clip((detection_age_ms if detection_age_ms is not None else 600.) / 600., 0, 1),
    ], dtype=np.float32)
    combined = np.concatenate([hazard_vec, recent.reshape(-1), intervals, info]).astype(np.float32)
    if combined.shape != (AUX_DIM,) or not np.isfinite(combined).all():
        raise RuntimeError('Invalid PPO auxiliary features')
    return combined


class RGBHazardPracticeEnv(TimedEnv):
    """Local Practice with a one-item, asynchronous YOLO queue.

    The RGB frames are always returned. YOLO boxes are optional contextual
    hints (never rewards or a hard-coded collision/avoidance policy).
    """
    def __init__(self, args, output, demos, detector_model, *, detect_every=3,
                 imgsz=640, conf=.20, max_stale_ms=600, press_delay_ms=20):
        self.detector_model = detector_model
        self.detect_every = int(detect_every)
        self.imgsz = int(imgsz)
        self.conf = float(conf)
        self.max_stale_s = float(max_stale_ms) / 1000
        self.episode_tag = 0
        self.detector_pool = (ThreadPoolExecutor(max_workers=1, thread_name_prefix='rs-v3-yolo')
                              if detector_model is not None else None)
        self.job = None
        self.last_detection = None
        self.detector_requests = 0
        self.detector_completed = 0
        self.detector_skipped_busy = 0
        self.detector_discarded_episode = 0
        self.detector_latency_ms = deque(maxlen=1000)
        self.decision_ages_ms = deque(maxlen=1000)
        self.fresh_steps = 0
        self.all_steps = 0
        self.total_detections = Counter()
        self.detect_tick = 0
        self.action_history = deque(maxlen=HISTORY)
        self.episode_started = time.perf_counter()
        self.previous_observation_at = None
        self.frame_intervals = deque(maxlen=FRAME_INTERVALS)
        self.probe_press_delay_ms = int(press_delay_ms)
        self.probe_wait_mode = 'none'
        try:
            super().__init__(args, output, demos)
        except BaseException:
            if self.detector_pool is not None:
                self.detector_pool.shutdown(wait=False, cancel_futures=True)
            raise
        self.observation_space = build_observation_space()

    def _poll_result(self):
        if self.job is None or not self.job.done():
            return
        result = self.job.result()  # Surface genuine YOLO/runtime errors.
        self.job = None
        self.detector_completed += 1
        self.detector_latency_ms.append(result['inference_ms'])
        if result['episode'] != self.episode_tag:
            self.detector_discarded_episode += 1
            return
        self.last_detection = result
        self.total_detections.update(o['class'] for o in result['detections'])

    def _submit_frame(self, rgb_obs):
        if self.detector_pool is None:
            return
        if self.job is not None:
            self.detector_skipped_busy += 1
            return
        # Copy the latest frame so the background detector cannot read a
        # buffer that another step modifies. Timestamp is the SOURCE capture.
        rgb = v2.vision.latest_rgb(rgb_obs)
        captured_at = self.last_capture_end or time.perf_counter()
        self.job = self.detector_pool.submit(
            detect, self.detector_model, rgb, captured_at, self.episode_tag,
            self.detector_requests, self.imgsz, self.conf)
        self.detector_requests += 1

    def _pack(self, rgb_obs, *, start_detection=False):
        self._poll_result()
        now = time.perf_counter()
        source_at = self.last_capture_end or now
        if self.previous_observation_at is not None:
            self.frame_intervals.append(max(0., source_at - self.previous_observation_at))
        self.previous_observation_at = source_at
        if start_detection:
            self._submit_frame(rgb_obs)
        h, age_ms, fresh = hazard_features(self.last_detection, now=now,
                                            max_age_s=self.max_stale_s)
        if fresh:
            self.fresh_steps += 1
            self.decision_ages_ms.append(age_ms)
        self.all_steps += 1
        aux = form_aux(h, self.action_history, now-self.episode_started,
                       self.frame_intervals, age_ms)
        return {'image': np.asarray(rgb_obs, dtype=np.uint8), 'aux': aux}, fresh, age_ms

    def reset(self, *, seed=None, options=None):
        rgb_obs, info = super().reset(seed=seed, options=options)
        self.episode_tag += 1
        self.action_history.clear()
        self.previous_observation_at = None
        self.frame_intervals.clear()
        self.episode_started = time.perf_counter()
        self.last_detection = None  # Never leak previous run's hazards.
        self.detect_tick = 0
        obs, fresh, age = self._pack(rgb_obs, start_detection=True)
        return obs, info

    def step(self, action):
        rgb_obs, reward, dead, truncated, info = super().step(action)
        self.action_history.append(int(action))
        self.detect_tick += 1
        obs, fresh, age = self._pack(
            rgb_obs, start_detection=(not (dead or truncated) and
                                      self.detect_tick % self.detect_every == 0))
        info['hazards_fresh'] = bool(fresh)
        info['hazards_age_ms'] = age
        return obs, reward, dead, truncated, info

    def close(self):
        try:
            super().close()
        finally:
            if self.detector_pool is not None:
                self.detector_pool.shutdown(wait=True, cancel_futures=True)
                self.detector_pool = None

    def hazard_report(self):
        latencies = list(self.detector_latency_ms)
        ages = list(self.decision_ages_ms)
        def dist(v):
            return {'count': len(v), 'median': round(float(np.median(v)), 1),
                    'p95': round(float(np.percentile(v, 95)), 1)} if v else {'count': 0}
        return {
            'requests': self.detector_requests,
            'completed': self.detector_completed,
            'discarded_previous_episode': self.detector_discarded_episode,
            'busy_skips': self.detector_skipped_busy,
            'fresh_fraction': round(self.fresh_steps/max(1, self.all_steps), 3),
            'inference_ms': dist(latencies),
            'age_at_decision_ms': dist(ages),
            'detected_objects': dict(self.total_detections),
        }


class TrainingCallback(BaseCallback):
    def __init__(self, output: Path, max_minutes: float, save_every_minutes: float = 4.0):
        super().__init__()
        self.output = output
        self.limit_s = max_minutes*60
        self.save_s = save_every_minutes*60
        self.started = time.perf_counter()
        self.last_save = self.started
        self.last_report = self.started
        self.recent_duration = deque(maxlen=25)
        self.all_durations = []
        self.action_counts = Counter()
        self.training_log = (output/'train_progress.jsonl').open('w', encoding='utf8')

    def _on_step(self):
        now = time.perf_counter()
        for action in np.asarray(self.locals.get('actions', []), dtype=np.int64).ravel():
            self.action_counts[v2.vision.ACTIONS[int(action)]] += 1
        for info in self.locals.get('infos', []):
            if 'episode' in info:  # Added by Monitor when the episode ends.
                duration = float(info.get('duration', info['episode']['t']))
                self.recent_duration.append(duration)
                self.all_durations.append(duration)
        if now-self.last_report >= 30:
            recent = list(self.recent_duration)
            report = {
                'seconds': round(now-self.started, 1),
                'steps': int(self.num_timesteps),
                'episodes': len(self.all_durations),
                'mean_last_25_seconds': round(float(np.mean(recent)), 2) if recent else None,
                'median_last_25_seconds': round(float(np.median(recent)), 2) if recent else None,
                'max_episode_seconds': round(max(self.all_durations), 2) if self.all_durations else None,
                'action_counts': dict(self.action_counts),
            }
            self.training_log.write(json.dumps(report)+'\n')
            self.training_log.flush()
            print('[V3] ' + json.dumps(report), flush=True)
            self.last_report = now
        if now-self.last_save >= self.save_s:
            self.model.save(self.output/'latest')
            print(f'[V3] Saved checkpoint at {self.num_timesteps} steps', flush=True)
            self.last_save = now
        return now-self.started < self.limit_s

    def _on_training_end(self):
        self.training_log.close()


class FakeInputEnv(gym.Env):
    """Offline-only schema test. It does not represent real gameplay."""
    def __init__(self):
        super().__init__()
        self.observation_space = build_observation_space()
        self.action_space = spaces.Discrete(ACTIONS)
        self.rng = np.random.default_rng(12)
        self.tick = 0

    def _obs(self):
        # Low-resolution uniform spatial patterns are sufficient to exercise
        # CNN input shape and feature path without pretending it's gameplay.
        x = self.rng.integers(0, 256, (12, 144, 256), dtype=np.uint8)
        aux = np.zeros(AUX_DIM, np.float32)
        aux[-1] = min(self.tick/50, 1)
        return {'image': x, 'aux': aux}

    def reset(self, *, seed=None, options=None):
        self.tick = 0
        return self._obs(), {}

    def step(self, action):
        self.tick += 1
        return self._obs(), (1. if action == self.tick % ACTIONS else -.1), self.tick >= 8, False, {}


def make_model(env, args, *, device=None):
    device = device or ('cuda' if torch.cuda.is_available() else 'cpu')
    return PPO('MultiInputPolicy', env, device=device,
               n_steps=args.n_steps, batch_size=args.batch_size, n_epochs=args.n_epochs,
               learning_rate=args.learning_rate, gamma=.99, gae_lambda=.95,
               ent_coef=args.ent_coef, clip_range=.2, verbose=1, seed=args.seed,
               policy_kwargs={
                   'features_extractor_class': RGBHazardExtractor,
                   'features_extractor_kwargs': {'features_dim': 192},
                   'net_arch': {'pi': [64], 'vf': [64]},
               })


def check_offline(args):
    """Check MultiInputPolicy training, gradients, serialization; no browser."""
    print('[CHECK] Building multimodal PPO with synthetic observations', flush=True)
    env = FakeInputEnv()
    # Avoid resizing `n_steps` with high RAM footprint on check.
    args.n_steps, args.batch_size, args.n_epochs = 16, 8, 1
    check_device = 'cuda' if torch.cuda.is_available() else 'cpu'
    model = make_model(env, args, device=check_device)
    obs, _ = env.reset()
    out, _ = model.predict(obs, deterministic=False)
    if not env.action_space.contains(int(out)):
        raise AssertionError('Bad policy output')
    prior = {k: v.clone() for k, v in model.policy.features_extractor.state_dict().items()}
    model.learn(total_timesteps=32)
    update = any(not torch.equal(prior[k], v) for k, v in
                 model.policy.features_extractor.state_dict().items())
    if not update:
        raise AssertionError('Feature extractor weights did not change')
    import tempfile
    with tempfile.TemporaryDirectory() as temporary:
        checkpoint = Path(temporary)/'model'
        model.save(checkpoint)
        restored = PPO.load(checkpoint.with_suffix('.zip'), device=check_device)
        with torch.no_grad():
            x = {'image': torch.as_tensor(obs['image'][None]),
                 'aux': torch.as_tensor(obs['aux'][None])}
            left = model.policy.obs_to_tensor(obs)[0]
            right = restored.policy.obs_to_tensor(obs)[0]
            a = model.policy.get_distribution(left).distribution.probs
            b = restored.policy.get_distribution(right).distribution.probs
            if not torch.allclose(a, b, atol=1e-5):
                raise AssertionError('Checkpoint changed action probabilities')
    print('[CHECK] PASS: dict observations, multimodal CNN, PPO update and '
          'checkpoint roundtrip. Does NOT establish useful game learning.', flush=True)


def options():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument('--mode', choices=('check', 'smoke', 'train', 'eval'), default='check')
    p.add_argument('--url', default='http://localhost:8000')
    p.add_argument('--fullscreen', action='store_true')
    p.add_argument('--run-name', default=None)
    p.add_argument('--detector', type=Path, default=Path('ai_data/v2_yolo_dataset/runs/hazards/weights/best.pt'))
    p.add_argument('--rgb-only', action='store_true', help='Ablation: disable YOLO while keeping time/action history')
    p.add_argument('--imgsz', type=int, default=640)
    p.add_argument('--conf', type=float, default=.20)
    p.add_argument('--detect-every', type=int, default=3)
    p.add_argument('--max-stale-ms', type=int, default=600)
    p.add_argument('--press-delay-ms', type=int, default=20)
    p.add_argument('--fps', type=int, default=20)
    p.add_argument('--steps', type=int, default=6000)
    p.add_argument('--max-minutes', type=float, default=12)
    p.add_argument('--save-every-minutes', type=float, default=4)
    p.add_argument('--resume', type=Path, default=None)
    p.add_argument('--smoke-seconds', type=float, default=45)
    p.add_argument('--eval-episodes', type=int, default=20)
    p.add_argument('--eval-policy', choices=('model', 'random'), default='model')
    p.add_argument('--eval-max-minutes', type=float, default=20)
    p.add_argument('--n-steps', type=int, default=128)
    p.add_argument('--batch-size', type=int, default=32)
    p.add_argument('--n-epochs', type=int, default=4)
    p.add_argument('--learning-rate', type=float, default=3e-4)
    p.add_argument('--ent-coef', type=float, default=.02)
    p.add_argument('--seed', type=int, default=123)
    a = p.parse_args()
    if not a.url.startswith(('http://localhost:', 'http://127.0.0.1:')):
        p.error('LOCAL Practice only; URL must start with localhost or 127.0.0.1')
    if a.mode != 'check' and not a.rgb_only and not a.detector.is_file():
        p.error(f'YOLO checkpoint missing: {a.detector}')
    if a.resume and not a.resume.is_file():
        p.error(f'Model checkpoint missing: {a.resume}')
    if a.mode == 'eval' and a.eval_policy == 'model' and not a.resume:
        p.error('Evaluation of learned policy requires --resume.')
    if a.mode == 'train' and a.eval_policy != 'model':
        p.error('--eval-policy applies to evaluation only')
    if a.mode == 'train' and a.steps < 128:
        p.error('--steps must be at least 128')
    if a.n_steps < 16 or a.batch_size < 1 or a.n_steps % a.batch_size != 0:
        p.error('n-steps must be >=16 and divisible by batch-size')
    if a.steps < 1 or a.max_minutes <= 0 or a.eval_max_minutes <= 0 or a.smoke_seconds <= 0:
        p.error('Invalid steps or time limit')
    if a.detect_every < 1 or a.press_delay_ms < 1 or a.imgsz < 160 or not 0 < a.conf < 1:
        p.error('Invalid detector or input timing values')
    if not (0 < a.max_stale_ms <= 5000 and 1 <= a.eval_episodes <= 1000):
        p.error('Invalid freshness age or evaluation episode count')
    if a.run_name and (Path(a.run_name).name != a.run_name or a.run_name in ('.', '..')):
        p.error('--run-name must be a simple folder name')
    if a.resume and a.mode == 'check':
        p.error('No --resume in check mode')
    return a


def v2_args(a):
    argv = sys.argv[:]
    try:
        sys.argv = [str(Path(v2.__file__)), '--mode', 'smoke', '--url', a.url,
                    '--fps', str(a.fps), '--max-minutes', str(a.max_minutes+2),
                    '--episode-max-seconds', '60', '--r-timeout', '1.15',
                    '--profile-seconds', '30']
        if a.fullscreen:
            sys.argv.append('--fullscreen')
        args = v2.parser_args()
        args.pretrain_updates = 0
        args.bc_updates = 0
        return args
    finally:
        sys.argv = argv



def live_smoke(env, args, output):
    """Short random-action check of real image/aux/time/restart plumbing."""
    rng = np.random.default_rng(args.seed)
    started = time.perf_counter()
    steps = 0
    deaths = 0
    first = True
    while time.perf_counter()-started < args.smoke_seconds:
        obs, _ = env.reset()
        if first:
            first = False
            if not env.observation_space.contains(obs):
                raise AssertionError('Live reset observation does not match multimodal space')
            sample = v2.vision.latest_rgb(obs['image'])
            cv2.imwrite(str(output/'smoke_gameplay_frame.png'),
                        cv2.cvtColor(sample, cv2.COLOR_RGB2BGR))
            print('[SMOKE] RGB', obs['image'].shape, obs['image'].dtype,
                  '| aux', obs['aux'].shape, obs['aux'].dtype,
                  '| saved smoke_gameplay_frame.png', flush=True)
        for _ in range(1000):
            if time.perf_counter()-started >= args.smoke_seconds:
                break
            action = int(rng.choice(5, p=[.60,.10,.10,.10,.10]))
            obs, reward, dead, truncated, info = env.step(action)
            if not env.observation_space.contains(obs):
                raise AssertionError('Live step observation is invalid')
            steps += 1
            if dead or truncated:
                deaths += 1
                break
    duration = max(time.perf_counter()-started, .001)
    report = {'wall_seconds': round(duration, 2), 'steps': steps,
              'steps_per_second_with_resets': round(steps/duration, 2),
              'completed_episodes': deaths, 'perception': env.hazard_report()}
    (output/'smoke_report.json').write_text(json.dumps(report, indent=2))
    print('[SMOKE] PASS (live observation + action loop):', json.dumps(report), flush=True)
    print('[SMOKE] This does NOT establish learned gameplay skill.', flush=True)


def evaluate(model, env, args, output):
    rng = np.random.default_rng(args.seed)
    durations = []
    action_counts = Counter()
    started = time.perf_counter()
    last_obs = None
    for ep in range(args.eval_episodes):
        if time.perf_counter()-started >= args.eval_max_minutes*60:
            break
        obs, info = env.reset()
        while True:
            if args.eval_policy == 'random':
                action = int(rng.choice(5, p=[.60, .10, .10, .10, .10]))
            else:
                action, _ = model.predict(obs, deterministic=True)
                action = int(action)
            action_counts[v2.vision.ACTIONS[action]] += 1
            obs, reward, ended, truncated, info = env.step(action)
            if ended or truncated:
                durations.append(float(info['duration']))
                print(f'[EVAL] episode {ep+1}/{args.eval_episodes}: '
                      f'{durations[-1]:.2f}s, action={v2.vision.ACTIONS[action]}', flush=True)
                break
    report = {
        'policy': args.eval_policy, 'episodes': len(durations),
        'seconds': durations,
        'mean_seconds': round(float(np.mean(durations)), 2) if durations else None,
        'median_seconds': round(float(np.median(durations)), 2) if durations else None,
        'best_seconds': round(float(max(durations)), 2) if durations else None,
        'action_counts': dict(action_counts),
        'perception': env.hazard_report(),
        'note': 'Unseen local Practice episodes. Compare with same-settings random evaluation.',
    }
    (output/'eval_report.json').write_text(json.dumps(report, indent=2))
    print('[EVAL] ' + json.dumps(report, indent=2), flush=True)


def main():
    args = options()
    if args.mode == 'check':
        check_offline(args)
        return
    name = args.run_name or ('v3_'+args.mode+'_'+datetime.now().strftime('%Y%m%d_%H%M%S'))
    output = v2.RUN_ROOT / name
    if output.exists() and any(output.iterdir()):
        raise SystemExit(f'Output folder already has files: {output}; use another --run-name')
    output.mkdir(parents=True, exist_ok=True)
    (output/'v3_settings.json').write_text(json.dumps(vars(args), indent=2, default=str))
    model = None
    env = None
    detector = None
    raw_env = None
    started = time.perf_counter()
    callback = None
    try:
        if not args.rgb_only:
            from ultralytics import YOLO
            print('[V3] Loading YOLO from', args.detector, flush=True)
            detector = YOLO(str(args.detector))
            # Warm up before opening Chromium (exclude cold-start inference).
            _ = detect(detector, np.zeros((144, 256, 3), np.uint8),
                       time.perf_counter(), 0, 0, args.imgsz, args.conf)
        demos = v2.DemoStore(v2.DEMO_ROOT, max_samples=100, load_existing=False)
        cfg = v2_args(args)
        raw_env = RGBHazardPracticeEnv(
            cfg, output, demos, detector,
            detect_every=args.detect_every, imgsz=args.imgsz, conf=args.conf,
            max_stale_ms=args.max_stale_ms, press_delay_ms=args.press_delay_ms)
        env = Monitor(raw_env, filename=str(output/'episodes.monitor.csv'))
        if args.mode == 'smoke':
            live_smoke(env, args, output)
            return
        if args.mode == 'eval':
            if args.eval_policy == 'model':
                model = PPO.load(str(args.resume), device='cuda' if torch.cuda.is_available() else 'cpu')
                if model.observation_space != env.observation_space:
                    raise RuntimeError('Model observation-space mismatch. Need a V3 checkpoint.')
            evaluate(model, env, args, output)
            return
        if args.resume:
            model = PPO.load(str(args.resume), env=env,
                             device='cuda' if torch.cuda.is_available() else 'cpu')
            print(f'[V3] Resumed {args.resume} (V3 checkpoints ONLY)', flush=True)
        else:
            model = make_model(env, args)
            print('[V3] Training new multimodal PPO. NO demonstrations; '
                  'YOLO boxes are auxiliary observations, NOT rewards.', flush=True)
        callback = TrainingCallback(output, args.max_minutes, args.save_every_minutes)
        model.learn(total_timesteps=args.steps, callback=callback,
                    reset_num_timesteps=not bool(args.resume))
    except KeyboardInterrupt:
        print('[V3] Ctrl+C received. Saving current model if available.', flush=True)
    finally:
        if model is not None and args.mode == 'train':
            model.save(output/'latest')
            print('[V3] Model saved:', output/'latest.zip', flush=True)
        if callback is not None:
            all_durations = callback.all_durations
            first_25 = all_durations[:25]
            last_25 = all_durations[-25:]
            summary = {
                'episodes': len(all_durations),
                'first_25_mean_seconds': round(float(np.mean(first_25)), 2) if first_25 else None,
                'last_25_mean_seconds': round(float(np.mean(last_25)), 2) if last_25 else None,
                'best_episode_seconds': round(float(max(all_durations)), 2) if all_durations else None,
                'action_counts': dict(callback.action_counts),
                'note': 'In-training changes are NOT an independent evaluation. Use --mode eval.',
            }
            (output/'training_summary.json').write_text(json.dumps(summary, indent=2))
            print('[V3] Training summary:', json.dumps(summary), flush=True)
        if raw_env is not None:
            try:
                report = raw_env.hazard_report()
                (output/'perception_report.json').write_text(json.dumps(report, indent=2))
                print('[V3] Perception:', json.dumps(report), flush=True)
            except Exception as e:
                print('[V3] Could not report hazards:', e, flush=True)
        if env is not None:
            env.close()
        elif raw_env is not None:
            raw_env.close()
        if callback is not None and not callback.training_log.closed:
            callback.training_log.close()
        print(f'[V3] Done. Run folder: {output} '
              f'elapsed={(time.perf_counter()-started)/60:.1f}min', flush=True)


if __name__ == '__main__':
    main()
