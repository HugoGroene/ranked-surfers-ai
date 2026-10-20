#!/usr/bin/env python3
"""LOCAL-only Ranked Surfers RGB PPO + human demonstrations.

Requires train_rs_rl_early_death.py and rs_practice_controller.py alongside this
file, and the same calibrated start_templates directory as the old trainer.

Workflow:
  1. Record examples (no AI):
       python train_rs_color_human.py --collect-only --max-minutes 7
  2. Start a NEW 4-frame RGB PPO model and learn from those examples:
       python train_rs_color_human.py --max-minutes 60 --fps 20 \
           --pretrain-updates 200 --bc-updates 4 --save-every-minutes 5
  3. Continue this RGB model later:
       python train_rs_color_human.py --resume ai_data/rl_color_runs/<RUN>/latest.zip \
           --max-minutes 60

During AI training press F8 while Chromium/canvas has keyboard focus to take
control. Use arrow keys normally, then press F8 to resume the AI (on a NEW
Practice attempt). Human samples are saved as .npz chunks in ai_data/rl_color_demos
and used to update the policy via an imitation (behavior cloning) loss.

Important:
 * Old GRAYSCALE checkpoints are not compatible with RGB observations.
 * Uses only localhost Practice; never enters online ranked matches.
 * Human input is NEVER recorded as a PPO-selected action.
 * Requires X11-compatible visible desktop for MSS; do not minimize Chromium.
 * F8 works when Chromium has focus, even if the Unity canvas is inside an iframe.
"""
import argparse
import asyncio
from collections import Counter, deque
from datetime import datetime
import json
from pathlib import Path
import time

import cv2
import numpy as np
from PIL import Image
import torch
from gymnasium import spaces
from stable_baselines3 import PPO
from stable_baselines3.common.callbacks import BaseCallback
from stable_baselines3.common.monitor import Monitor

import rs_practice_controller as ctrl
from train_rs_rl_early_death import LocalPracticeEnv, TinyCNN, IMAGE_W, IMAGE_H, ACTIONS

STACK = 4
CHANNELS = 3 * STACK
DEMO_FOLDER = ctrl.DATA / 'rl_color_demos'
RUN_FOLDER = ctrl.DATA / 'rl_color_runs'
KEY_TO_ACTION = {'ArrowLeft': 1, 'ArrowRight': 2, 'ArrowUp': 3, 'ArrowDown': 4}

# Script is installed in both top document and the Unity iframe; it observes
# REAL keydowns from the same browser window, with no global keylogger.
KEY_LISTENER_JS = r'''() => {
    if (window.__rsHumanInstalled) return;
    window.__rsHumanInstalled = true;
    window.__rsHumanQueue = [];
    window.addEventListener('keydown', e => {
        if (e.repeat) return;
        if (!['F8','ArrowLeft','ArrowRight','ArrowUp','ArrowDown'].includes(e.code)) return;
        const q = window.__rsHumanQueue;
        q.push(e.code);
        if (q.length > 80) q.splice(0, q.length - 80);
        if (e.code === 'F8') { e.preventDefault(); e.stopImmediatePropagation(); }
    }, true);
    window.addEventListener('keyup', e => {
        if (['ArrowLeft','ArrowRight','ArrowUp','ArrowDown'].includes(e.code)) {
            window.__rsHumanQueue.push('UP:' + e.code);
        }
    }, true);
}'''
DRAIN_JS = '() => { const q = window.__rsHumanQueue || []; return q.splice(0, q.length); }'


class HumanDemoStore:
    """Bounded RAM replay + durable, compressed per-takeover demonstration files."""
    def __init__(self, folder, max_items=1200):
        self.folder = Path(folder)
        self.folder.mkdir(parents=True, exist_ok=True)
        self.samples = deque(maxlen=max_items)
        self.actions = deque(maxlen=max_items)
        self.new_obs = []
        self.new_actions = []
        self.total_saved = 0
        self._load_previous()

    def _load_previous(self):
        files = sorted(self.folder.glob('demo_*.npz'), reverse=True)
        for path in files:
            if len(self.samples) >= self.samples.maxlen:
                break
            try:
                with np.load(path) as loaded:
                    obs, actions = loaded['observations'], loaded['actions']
                    if obs.ndim != 4 or obs.shape[1:] != (CHANNELS, IMAGE_H, IMAGE_W):
                        print('[DEMO] Skipping incompatible sample file:', path.name, flush=True)
                        continue
                    for sample, action in zip(obs[::-1], actions[::-1]):
                        if len(self.samples) >= self.samples.maxlen:
                            break
                        if 0 <= int(action) < len(ACTIONS):
                            self.samples.appendleft(np.array(sample, copy=True))
                            self.actions.appendleft(int(action))
            except Exception as exc:
                print('[DEMO] Could not read', path, ':', str(exc)[:120], flush=True)
        print(f'[DEMO] Loaded {len(self.samples)} saved RGB examples from {self.folder}', flush=True)

    def add(self, observation, action):
        assert observation.shape == (CHANNELS, IMAGE_H, IMAGE_W)
        sample = np.ascontiguousarray(observation, dtype=np.uint8)
        self.samples.append(sample)
        self.actions.append(int(action))
        self.new_obs.append(sample)
        self.new_actions.append(int(action))

    def save_pending(self):
        if not self.new_obs:
            return
        filename = self.folder / ('demo_' + datetime.now().strftime('%Y%m%d_%H%M%S_%f') + '.npz')
        np.savez_compressed(filename, observations=np.stack(self.new_obs),
                            actions=np.asarray(self.new_actions, dtype=np.int64))
        self.total_saved += len(self.new_obs)
        counts = dict(Counter(self.new_actions))
        print(f'[DEMO] Saved {len(self.new_obs)} examples: {filename.name}; actions={counts}', flush=True)
        self.new_obs.clear()
        self.new_actions.clear()

    def batch(self, batch_size, rng):
        if len(self.samples) < 8:
            return None
        acts = np.asarray(self.actions, dtype=np.int64)
        nonzero = np.flatnonzero(acts != 0)
        if len(nonzero) < 8:
            return None  # Avoid teaching the model to do absolutely nothing.
        zeros = np.flatnonzero(acts == 0)
        # 50% actual movement demonstrations, 50% non-movement (if available).
        # Without balancing, idle screenshots can overwhelm rare left/right inputs.
        nmove = batch_size if len(zeros) == 0 else batch_size // 2
        move_idxs = rng.choice(nonzero, size=nmove, replace=True)
        if len(zeros):
            no_idxs = rng.choice(zeros, size=batch_size - nmove, replace=True)
            indexes = np.concatenate([move_idxs, no_idxs])
            rng.shuffle(indexes)
        else:
            indexes = move_idxs
        return np.stack([self.samples[int(i)] for i in indexes]), acts[indexes]


class ColorPracticeEnv(LocalPracticeEnv):
    """Same tested restart/death logic; input is 4 RGB frames (12 channels)."""
    def __init__(self, args, output_dir, demos):
        self.demos = demos
        self.js_ready = False
        super().__init__(args, output_dir)
        self.observation_space = spaces.Box(0, 255, (CHANNELS, IMAGE_H, IMAGE_W), dtype=np.uint8)

    async def _open_browser(self):
        await super()._open_browser()
        await self._install_key_listeners()
        print('[HUMAN] Press F8 while Chromium is focused to take control.', flush=True)

    async def _install_key_listeners(self):
        installed = 0
        for frame in self.page.frames:
            try:
                await frame.evaluate(KEY_LISTENER_JS)
                installed += 1
            except Exception:
                pass
        if not installed:
            print('[HUMAN] WARNING: no keyboard listeners installed!', flush=True)
        self.js_ready = installed > 0

    async def _start_episode(self):
        image, raw = await super()._start_episode()
        # Practice sometimes recreates arena.html iframe, so install listeners
        # after every restart. The JS guard prevents duplicate listeners.
        await self._install_key_listeners()
        return image, raw

    async def _capture(self):
        await self._sync_capture_region()
        shot = np.asarray(self.grabber.grab(self.region))
        rgb = cv2.cvtColor(shot, cv2.COLOR_BGRA2RGB)
        score = self.menu_detector.score(Image.fromarray(rgb))
        small = cv2.resize(rgb, (IMAGE_W, IMAGE_H), interpolation=cv2.INTER_AREA)
        chw = np.ascontiguousarray(small.transpose(2, 0, 1))
        # Keep original diagnostic screenshots, but don't JPEG-roundtrip the
        # image used as neural-network input (preserves the exact RGB colors).
        ok, encoded = cv2.imencode('.jpg', cv2.cvtColor(rgb, cv2.COLOR_RGB2BGR),
                                   [cv2.IMWRITE_JPEG_QUALITY, 85])
        if not ok:
            raise RuntimeError('Could not encode diagnostic screenshot')
        raw = encoded.tobytes()
        self.last_raw = raw
        return chw, float(score), raw

    @staticmethod
    def _rgb_obs(stacked):
        # Base environment returns (4,3,H,W); SB3 CNN expects (12,H,W).
        return np.ascontiguousarray(stacked.reshape(CHANNELS, IMAGE_H, IMAGE_W))

    def reset(self, *, seed=None, options=None):
        stacked, info = super().reset(seed=seed, options=options)
        return self._rgb_obs(stacked), info

    async def _drain_keys(self):
        codes = []
        for frame in self.page.frames:
            try:
                codes.extend(await frame.evaluate(DRAIN_JS))
            except Exception:
                continue
        return codes

    async def _human_loop(self, end_time=None, require_f8_to_exit=True):
        """Browser keys go to Unity itself; we only OBSERVE and label them.

        The PPO actor is stopped while we are here. Observations are captured
        before processing newly observed key presses, approximating the state
        the human saw when choosing the action (poll interval ~80 ms).
        """
        print('[HUMAN] You control the game: arrows to move, F8 to return to AI.', flush=True)
        await ctrl.focus_canvas(self.canvas)
        self.frames = deque(self.frames, maxlen=STACK)
        await self._drain_keys()
        count_before = len(self.demos.samples)
        idle_ticks = 0
        menu_streak = 0
        held_keys = set()
        move_events = 0
        last_status = time.monotonic()
        started = time.monotonic()
        while True:
            if end_time is not None and time.monotonic() >= end_time:
                break
            if time.monotonic() - started >= self.args.human_max_seconds:
                print('[HUMAN] Takeover time limit reached.', flush=True)
                break
            await asyncio.sleep(1.0 / self.args.human_fps)
            keys = await self._drain_keys()
            if require_f8_to_exit and 'F8' in keys:
                print('[HUMAN] F8 received; returning control to PPO on next Practice run.', flush=True)
                break
            obs_before = self._rgb_obs(np.stack(self.frames))
            arrows = []
            for k in keys:
                if k in KEY_TO_ACTION:
                    held_keys.add(k)
                    arrows.append(KEY_TO_ACTION[k])
                elif k.startswith('UP:'):
                    held_keys.discard(k[3:])
            if arrows:
                for action in arrows:
                    self.demos.add(obs_before, action)
                move_events += len(arrows)
                idle_ticks = 0
                if move_events % 10 < len(arrows):
                    print(f'[HUMAN] Captured {move_events} arrow keydowns', flush=True)
            elif not held_keys:
                idle_ticks += 1
                # Never label a HELD arrow as 'do nothing'.
                if idle_ticks % 5 == 0:
                    self.demos.add(obs_before, 0)
            rgb, score, _ = await self._capture()
            self.frames.append(rgb)
            menu_streak = menu_streak + 1 if score >= self.args.threshold * 0.85 else 0
            if menu_streak >= 3:
                print('[HUMAN] Character died / menu detected; ending human attempt.', flush=True)
                break
            if time.monotonic() - last_status >= 10:
                print(f'[HUMAN] Examples this attempt: {len(self.demos.samples) - count_before}; '
                      f'arrow presses: {move_events}', flush=True)
                if not move_events:
                    print('[HUMAN] If you pressed arrows, check Chromium has keyboard focus.', flush=True)
                last_status = time.monotonic()
        self.demos.save_pending()
        await self._drain_keys()
        print('[HUMAN] Takeover complete.', flush=True)

    def collect_one_episode(self, end_time):
        self.reset()
        # This deliberately does not require F8: entire run is a demonstration.
        self._call(self._human_loop(end_time, require_f8_to_exit=False),
                   timeout=self.args.human_max_seconds + 30)

    def step(self, action):
        # The parent executes exactly one PPO-selected action and returns the
        # resulting transition BEFORE we start accepting human corrections.
        stacked, reward, terminated, truncated, info = super().step(action)
        obs = self._rgb_obs(stacked)
        # Keep survival reward independent of requested --fps. At 20 FPS this
        # matches the old reward scale (0.02 * seconds * 20).
        death_penalty = 1.0 if terminated else 0.0
        reward = (reward + death_penalty) * (20.0 / self.args.fps) - death_penalty
        if terminated or truncated:
            self._call(self._drain_keys(), timeout=10)
            return obs, reward, terminated, truncated, info
        keys = self._call(self._drain_keys(), timeout=10)
        if 'F8' not in keys:
            return obs, reward, False, False, info
        # PPO's prior action has completed and its resulting observation is
        # captured. Do NOT emit human moves as if PPO sampled those actions.
        self._call(self._human_loop(require_f8_to_exit=True),
                   timeout=self.args.human_max_seconds + 30)
        info['human_takeover'] = True
        info['end_reason'] = 'human_takeover'
        print('[RL] Human intervention: ending this PPO episode cleanly.', flush=True)
        return obs, reward, True, False, info


class DemoAndCheckpointCallback(BaseCallback):
    def __init__(self, output_dir, max_minutes, save_minutes, demos, args):
        super().__init__()
        self.folder = Path(output_dir)
        self.max_minutes = max_minutes
        self.save_interval = save_minutes * 60
        self.demos = demos
        self.args = args
        self.started = time.monotonic()
        self.saved = self.started
        self.rng = np.random.default_rng(args.seed + 1)
        self.rollouts = 0

    def _on_step(self):
        now = time.monotonic()
        if now - self.saved >= self.save_interval:
            self.model.save(self.folder / 'latest')
            self.demos.save_pending()
            self.saved = now
            print(f'[RL] Saved RGB checkpoint at {self.model.num_timesteps} steps', flush=True)
        return now - self.started < self.max_minutes * 60

    def _on_rollout_start(self):
        # Apply imitation BEFORE collecting a rollout. All PPO samples then
        # come from the resulting policy, preserving on-policy behavior.
        self.rollouts += 1
        if self.args.bc_updates > 0 and self.demos.samples:
            bc_update(self.model, self.demos, self.args.bc_updates,
                      self.args.bc_batch_size, self.rng, self.args.bc_weight)


def bc_update(model, demos, updates, batch_size, rng, weight=1.0):
    """Behavior cloning regularizer: -log pi(human_action | human_RGB_frames).

    Runs BEFORE a new PPO rollout, NEVER inside PPO's rollout collection. Reuses
    the same policy and optimizer, so subsequent PPO updates continue normally.
    """
    policy = model.policy
    was_training = policy.training
    policy.train(True)
    values = []
    for _ in range(updates):
        batch = demos.batch(batch_size, rng)
        if batch is None:
            break
        obs, actions = batch
        obs_tensor = torch.as_tensor(obs, device=model.device)
        action_tensor = torch.as_tensor(actions, device=model.device, dtype=torch.long)
        dist = policy.get_distribution(obs_tensor)
        loss = -dist.log_prob(action_tensor).mean() * weight
        policy.optimizer.zero_grad(set_to_none=True)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(policy.parameters(), 0.5)
        policy.optimizer.step()
        values.append(float(loss.detach().cpu()))
    policy.train(was_training)
    if values:
        print(f'[BC] {len(values)} supervised updates, loss={np.mean(values):.3f}; '
              f'{len(demos.samples)} RGB examples', flush=True)
    return len(values)


def build_args():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument('--url', default='http://localhost:8000')
    parser.add_argument('--collect-only', action='store_true',
                        help='YOU play every Practice run; record RGB + arrow examples; no PPO model')
    parser.add_argument('--run-name', type=str, default=None,
                        help='Save into ai_data/rl_color_runs/NAME instead of a timestamped folder')
    parser.add_argument('--resume', type=Path, default=None,
                        help='Only RGB checkpoints from THIS script; grayscale .zip is incompatible')
    parser.add_argument('--fps', type=int, default=20)
    parser.add_argument('--human-fps', type=int, default=12,
                        help='Capture frequency while you manually control the character')
    parser.add_argument('--human-max-seconds', type=float, default=180,
                        help='Maximum time spent in one F8 takeover before AI resumes')
    parser.add_argument('--max-minutes', type=float, default=60)
    parser.add_argument('--steps', type=int, default=250000)
    parser.add_argument('--save-every-minutes', type=float, default=5)
    parser.add_argument('--episode-max-seconds', type=float, default=60)
    parser.add_argument('--threshold', type=float, default=0.68)
    parser.add_argument('--load-timeout', type=float, default=25)
    parser.add_argument('--debug-every-seconds', type=float, default=12)
    parser.add_argument('--screen-x-offset', type=int, default=0)
    parser.add_argument('--screen-y-offset', type=int, default=0)
    parser.add_argument('--demo-max', type=int, default=1200, help='Max demonstration samples in RAM')
    parser.add_argument('--pretrain-updates', type=int, default=200,
                        help='One-time supervised batches for a NEW RGB model only')
    parser.add_argument('--bc-updates', type=int, default=4,
                        help='Supervised batches between PPO rollouts; 0 disables further BC')
    parser.add_argument('--bc-batch-size', type=int, default=32)
    parser.add_argument('--bc-weight', type=float, default=0.5,
                        help='Scale human imitation loss (0.5 recommended)')
    parser.add_argument('--seed', type=int, default=123)
    return parser, parser.parse_args()


def main():
    parser, args = build_args()
    if not args.url.startswith(('http://localhost:', 'http://127.0.0.1:')):
        parser.error('Local Practice only; official site / ranked matchmaking not supported')
    if not (2 <= args.fps <= 30 and 2 <= args.human_fps <= 30 and
            args.human_max_seconds >= 5 and args.max_minutes > 0 and
            args.steps >= 10 and args.save_every_minutes > 0 and
            args.demo_max >= 50 and args.bc_batch_size >= 8 and
            0 <= args.bc_updates <= 100 and 0 <= args.pretrain_updates <= 20000 and
            0 <= args.bc_weight <= 2 and 0.35 <= args.threshold <= 0.98):
        parser.error('One or more numeric settings are out of range')
    if args.resume and not args.resume.is_file():
        parser.error(f'Checkpoint does not exist: {args.resume}')
    if args.run_name and (args.run_name in ('.', '..') or
                          Path(args.run_name).name != args.run_name or
                          '/' in args.run_name or '\\' in args.run_name):
        parser.error('--run-name must be a single folder name, e.g. model')
    ctrl.DATA.mkdir(parents=True, exist_ok=True)
    DEMO_FOLDER.mkdir(parents=True, exist_ok=True)
    RUN_FOLDER.mkdir(parents=True, exist_ok=True)
    output = RUN_FOLDER / (args.run_name or datetime.now().strftime('%Y%m%d_%H%M%S'))
    if not args.resume and (output / 'latest.zip').exists():
        parser.error(f"Existing model at {output / 'latest.zip'}; use --resume to continue or choose another --run-name")
    output.mkdir(parents=True, exist_ok=True)
    (output / 'settings.json').write_text(json.dumps(vars(args), indent=2, default=str))
    demos = HumanDemoStore(DEMO_FOLDER, max_items=args.demo_max)
    env = None
    model = None
    try:
        if args.collect_only:
            print('[HUMAN] Recording manual play (no AI). Focus Chromium, use arrow keys.', flush=True)
            env = ColorPracticeEnv(args, output, demos)
            deadline = time.monotonic() + args.max_minutes * 60
            while time.monotonic() < deadline:
                env.collect_one_episode(deadline)
            return
        device = 'cuda' if torch.cuda.is_available() else 'cpu'
        print(f'[RL] RGB observation = ({CHANNELS},{IMAGE_H},{IMAGE_W}) uint8; '
              f'4 color frames, using {device}', flush=True)
        env = Monitor(ColorPracticeEnv(args, output, demos),
                      filename=str(output / 'episodes.monitor.csv'))
        if args.resume:
            model = PPO.load(str(args.resume), env=env, device=device)
            print('[RL] Resumed RGB checkpoint:', args.resume, flush=True)
            if tuple(model.observation_space.shape) != (CHANNELS, IMAGE_H, IMAGE_W):
                raise RuntimeError('Wrong observation shape. Only use RGB model checkpoints.')
        else:
            model = PPO('CnnPolicy', env, device=device, verbose=1, seed=args.seed,
                        n_steps=128, batch_size=32, n_epochs=4,
                        learning_rate=3e-4, gamma=0.99, gae_lambda=0.95,
                        ent_coef=0.02, clip_range=0.2,
                        policy_kwargs={
                            'features_extractor_class': TinyCNN,
                            'features_extractor_kwargs': {'features_dim': 128},
                            'net_arch': dict(pi=[64], vf=[64]),
                        })
            with torch.no_grad():
                model.policy.action_net.bias[0] += 1.0
            if args.pretrain_updates and demos.samples:
                print(f'[BC] Pretraining new RGB policy on {len(demos.samples)} human examples', flush=True)
                n = bc_update(model, demos, args.pretrain_updates,
                              args.bc_batch_size, np.random.default_rng(args.seed),
                              args.bc_weight)
                if not n:
                    print('[BC] Not enough movement examples; play some runs with --collect-only.', flush=True)
            else:
                print('[RL] No pretraining examples; consider --collect-only first.', flush=True)
        callback = DemoAndCheckpointCallback(output, args.max_minutes,
                                             args.save_every_minutes, demos, args)
        print('[RL] Training started. F8 in Chromium = takeover; F8 again = PPO.', flush=True)
        model.learn(total_timesteps=args.steps, callback=callback,
                    reset_num_timesteps=not bool(args.resume))
    except KeyboardInterrupt:
        print('\n[RL] Stopped by Ctrl+C; saving checkpoint and demos.', flush=True)
    except Exception as exc:
        if ('Target page, context or browser has been closed' in str(exc) or
                'TargetClosedError' in type(exc).__name__):
            print('[RL] Chromium closed. Saving progress.', flush=True)
        else:
            raise
    finally:
        demos.save_pending()
        if model is not None:
            model.save(output / 'latest')
            print('[RL] Checkpoint:', output / 'latest.zip', flush=True)
        if env is not None:
            env.close()
        print('[RL] Demo folder:', DEMO_FOLDER, flush=True)
        print('[RL] Session folder:', output, flush=True)


if __name__ == '__main__':
    main()
