#!/usr/bin/env python3
"""Ranked Surfers LOCAL PRACTICE v2: fullscreen, R restart, RGB 256x144,
timestamp-aligned human demonstrations, PPO + BC, visual diagnostics and
conservative action-neutral milestone rewards.

REQUIRES beside this file (from your existing project):
  rs_v2_vision.py, train_rs_color_anchor_cache.py,
  train_rs_color_quick_restart.py, train_rs_color_fast.py,
  train_rs_color_human_named.py, train_rs_rl_early_death.py,
  rs_practice_controller.py

Examples:
  python train_rs_v2_verify.py --mode smoke --fullscreen --max-minutes 3
  python train_rs_v2_verify.py --mode verify --fullscreen --max-minutes 5 --steps 512
  python train_rs_v2.py --mode collect --fullscreen --max-minutes 30
  python train_rs_v2.py --mode train --fullscreen --run-name v2_model \\
      --pretrain-updates 500 --max-minutes 10
  python train_rs_v2.py --mode train --fullscreen --run-name v2_more \\
      --resume ai_data/rl_v2_runs/v2_model/latest.zip --max-minutes 60

R restart is attempted FROM THE FIRST EPISODE, after activating the site's
fullscreen. If R fails several times, the script leaves fullscreen and falls
back to HTML Practice, then returns to fullscreen.

Unlike previous versions, timestamps of human keydown events are matched to
an observation captured *before* the keydown. See [DEMO] lag metrics.

Candidate coin detections are DIAGNOSTIC ONLY; do not trust them as rewards.
"""
import argparse
import asyncio
from collections import Counter, deque
from datetime import datetime
import hashlib
import json
from pathlib import Path
import sys
import time

import cv2
import numpy as np
import torch
from gymnasium import spaces
from stable_baselines3 import PPO
from stable_baselines3.common.callbacks import BaseCallback
from stable_baselines3.common.monitor import Monitor

import rs_practice_controller as ctrl
import rs_v2_vision as vision
from train_rs_rl_early_death import LocalPracticeEnv, TinyCNN
import train_rs_color_anchor_cache as anchors

DATA = ctrl.DATA
RUN_ROOT = DATA / 'rl_v2_runs'
DEMO_ROOT = DATA / 'rl_v2_demos'  # NEW SHAPE: do not mix with old 90x160 datasets
KEYS = {'ArrowLeft': 1, 'ArrowRight': 2, 'ArrowUp': 3, 'ArrowDown': 4}

EVENT_JS = r'''() => {
 if (window.__rsV2Installed) return;
 window.__rsV2Installed = true;
 window.__rsV2Queue = [];
 window.addEventListener('keydown', e => {
   if (e.repeat || !['ArrowLeft','ArrowRight','ArrowUp','ArrowDown'].includes(e.code)) return;
   window.__rsV2Queue.push({code:e.code, down:true, t:Date.now()/1000});
 }, true);
 window.addEventListener('keyup', e => {
   if (!['ArrowLeft','ArrowRight','ArrowUp','ArrowDown'].includes(e.code)) return;
   window.__rsV2Queue.push({code:e.code, down:false, t:Date.now()/1000});
 }, true);
}'''
DRAIN_JS = '() => (window.__rsV2Queue || []).splice(0,200)'


class DemoStore:
    """Store full (12,144,256) uint8 observations; hold out whole run files."""
    def __init__(self, root, max_samples=2200, seed=123, load_existing=True):
        self.root = Path(root)
        self.root.mkdir(parents=True, exist_ok=True)
        self.max_samples = max_samples
        self.seed = seed
        self.train_obs = []
        self.train_action = []
        self.val_obs = []
        self.val_action = []
        self.new_obs = []
        self.new_actions = []
        self.new_lags = []
        if load_existing:
            self.load()

    @staticmethod
    def compatible(obs):
        return obs.ndim == 4 and obs.shape[1:] == (vision.CHANNELS, vision.HEIGHT, vision.WIDTH)

    def load(self):
        paths = sorted(self.root.glob('v2_demo_*.npz'), reverse=True)
        invalid = 0
        for path in paths:
            if len(self.train_obs) + len(self.val_obs) >= self.max_samples:
                break
            try:
                with np.load(path) as data:
                    obs = data['observations']
                    acts = data['actions']
                    if not self.compatible(obs):
                        invalid += 1
                        continue
                    # Hold out whole episodes: no adjacent frames leak into validation.
                    val = (len(paths) >= 5 and int(hashlib.sha1(path.name.encode()).hexdigest(), 16) % 5 == 0)
                    a, b = (self.val_obs, self.val_action) if val else (self.train_obs, self.train_action)
                    budget = self.max_samples - len(self.train_obs) - len(self.val_obs)
                    for frame, action in zip(obs[:budget], acts[:budget]):
                        if int(action) in range(5):
                            a.append(frame.copy())
                            b.append(int(action))
            except Exception as e:
                print(f'[DEMO] Cannot read {path.name}: {str(e)[:100]}', flush=True)
        print(f'[DEMO] {len(self.train_obs)} training samples, {len(self.val_obs)} '
              f'holdout samples; {invalid} incompatible file(s) skipped; dir={self.root}', flush=True)

    def add(self, obs, action, lag):
        assert obs.shape == (vision.CHANNELS, vision.HEIGHT, vision.WIDTH), obs.shape
        if action not in range(5):
            return
        self.new_obs.append(np.array(obs, dtype=np.uint8, copy=True))
        self.new_actions.append(int(action))
        self.new_lags.append(float(lag))

    def flush(self):
        if not self.new_obs:
            return
        name = self.root / ('v2_demo_' + datetime.now().strftime('%Y%m%d_%H%M%S_%f') + '.npz')
        np.savez_compressed(name, observations=np.stack(self.new_obs),
                            actions=np.asarray(self.new_actions, dtype=np.uint8),
                            lag_ms=np.asarray(self.new_lags, dtype=np.float32)*1000)
        counts = {vision.ACTIONS[k]: n for k, n in Counter(self.new_actions).items()}
        print(f'[DEMO] Saved {len(self.new_obs)} examples to {name.name}; '
              f'actions={counts}; mean_alignment_lag={1000*np.mean(self.new_lags):.0f}ms', flush=True)
        self.new_obs.clear()
        self.new_actions.clear()
        self.new_lags.clear()
        # Fresh samples become training samples for THIS process too.
        # Reload on restart will assign per-episode holdout groups.

    def batch(self, size, rng, validation=False):
        arr, acts = (self.val_obs, self.val_action) if validation else (self.train_obs, self.train_action)
        if len(arr) < 8:
            return None
        labels = np.asarray(acts, dtype=np.int64)
        # Avoid over-weighting NONE; keep examples of all actual actions.
        groups = [np.flatnonzero(labels == i) for i in range(5)]
        occupied = [i for i, group in enumerate(groups) if len(group)]
        picks = []
        for _ in range(size):
            label = int(rng.choice(occupied, p=(np.array([.30 if i == 0 else .175 for i in occupied]) /
                                                   sum(.30 if i == 0 else .175 for i in occupied))))
            picks.append(int(rng.choice(groups[label])))
        return np.stack([arr[i] for i in picks]), labels[picks]


def bc_updates(model, demos, n, batch_size, weight, seed=123):
    if not n or len(demos.train_obs) < 8:
        return 0
    rng = np.random.default_rng(seed)
    policy = model.policy
    policy.train(True)
    losses = []
    for k in range(n):
        batch = demos.batch(batch_size, rng)
        if batch is None:
            break
        obs, labels = batch
        tensor = torch.as_tensor(obs, device=model.device)
        target = torch.as_tensor(labels, device=model.device, dtype=torch.long)
        dist = policy.get_distribution(tensor)
        loss = -dist.log_prob(target).mean() * weight
        policy.optimizer.zero_grad(set_to_none=True)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(policy.parameters(), .5)
        policy.optimizer.step()
        losses.append(float(loss.detach().cpu()))
        if (k+1) % 100 == 0 or (k+1) == n:
            print(f'[BC] {k+1}/{n} updates; recent loss={np.mean(losses[-50:]):.3f}', flush=True)
    policy.train(False)
    return len(losses)


def validation_stats(model, demos, batches=8, size=32):
    if len(demos.val_obs) < 8:
        print('[BC] No held-out demo episodes available; record more takes for validation.', flush=True)
        return
    rng = np.random.default_rng(999)
    correct, total, losses = 0, 0, []
    with torch.no_grad():
        for _ in range(batches):
            obs, targets = demos.batch(size, rng, validation=True)
            distribution = model.policy.get_distribution(torch.as_tensor(obs, device=model.device))
            probs = distribution.distribution.probs.cpu().numpy()
            correct += int((probs.argmax(axis=1) == targets).sum())
            total += len(targets)
            losses.append(float(-np.log(np.clip(probs[np.arange(len(targets)), targets], 1e-8, 1)).mean()))
    print(f'[BC] Episode-held-out accuracy={correct/total:.1%}, cross_entropy={np.mean(losses):.3f} '
          f'({len(demos.val_obs)} validation images). Imitation accuracy != game skill.', flush=True)


class V2PracticeEnv(anchors.CachedAnchorEnv):
    """Reuse proven restart/death logic, with higher res input and fullscreen/R."""
    def __init__(self, args, output_dir, demos):
        self._game_fullscreen = False
        self._v2_last_chw = None
        self._v2_last_bgra = None
        self._v2_prev_image = None
        self._v2_last_milestone = 0
        self.demos = demos
        super().__init__(args, output_dir, demos)
        self.observation_space = spaces.Box(0, 255,
            (vision.CHANNELS, vision.HEIGHT, vision.WIDTH), dtype=np.uint8)

    async def _open_browser(self):
        # First open and locate the Unity canvas in ordinary page mode.
        # Then activate SITE fullscreen BEFORE starting the first Practice run.
        # R works on the initial menu as well as after subsequent deaths.
        await super()._open_browser()
        await self._save_capture_debug('startup_before_fullscreen.jpg')
        if self.args.fullscreen:
            print('[SCREEN] Entering SITE fullscreen before the first R.', flush=True)
            await self._enter_game_fullscreen()
            await self._save_capture_debug('startup_fullscreen_before_r.jpg')

    async def _save_capture_debug(self, filename):
        """Always leave startup evidence, even if episode 1 never starts."""
        if self.grabber is None or self.region is None:
            return
        try:
            frame = np.asarray(self.grabber.grab(self.region))
            dest = self.output_dir / filename
            dest.parent.mkdir(parents=True, exist_ok=True)
            cv2.imwrite(str(dest), cv2.cvtColor(frame, cv2.COLOR_BGRA2BGR))
            print(f'[SCREEN] Diagnostic screenshot: {dest}', flush=True)
        except Exception as exc:
            print(f'[SCREEN] Could not save {filename}: {str(exc)[:120]}', flush=True)

    async def _fullscreen_detected(self):
        for f in self.page.frames:
            try:
                if await f.evaluate('!!document.fullscreenElement'):
                    return True
            except Exception:
                pass
        try:
            box = await self.canvas.bounding_box()
            monitor = self.grabber.monitors[1]
            return bool(box and box['width'] >= monitor['width']*.94 and
                        box['height'] >= monitor['height']*.90)
        except Exception:
            return False

    async def _enter_game_fullscreen(self):
        if await self._fullscreen_detected():
            self._game_fullscreen = True
            await self._anchor_capture_region()
            print('[SCREEN] Game already fullscreen.', flush=True)
            return True
        # Click the site's own fullscreen control, NOT Chrome F11.
        # First search accessible controls, then the bottom-right blue icon.
        try:
            for f in self.page.frames:
                for selector in ('button[title*="full" i]', '[aria-label*="full" i]',
                                 'button[id*="full" i]', '[class*="fullscreen" i]'):
                    loc = f.locator(selector).first
                    try:
                        if await loc.count() and await loc.is_visible():
                            await loc.click(timeout=1400)
                            await asyncio.sleep(.40)
                            if await self._fullscreen_detected():
                                self._game_fullscreen = True
                                await self._anchor_capture_region()
                                print('[SCREEN] Site fullscreen control activated.', flush=True)
                                return True
                    except Exception:
                        pass
            box = await self.canvas.bounding_box()
            if box:
                # Screenshot shows the small blue icon at bottom-right.
                await self.page.mouse.click(box['x'] + box['width'] - 25,
                                            box['y'] + box['height'] - 25)
                await asyncio.sleep(.5)
                if await self._fullscreen_detected():
                    self._game_fullscreen = True
                    await self._anchor_capture_region()
                    print('[SCREEN] Site fullscreen activated using bottom-right control.', flush=True)
                    return True
        except Exception as exc:
            print(f'[SCREEN] Fullscreen click error: {str(exc)[:150]}', flush=True)
        print('[SCREEN] Could not confirm game fullscreen. Continuing in windowed mode. '
              'Check capture_preview.jpg; DO NOT use Chrome F11.', flush=True)
        return False

    async def _anchor_capture_region(self):
        if self._game_fullscreen:
            # The game's fullscreen element occupies an entire monitor; no
            # Chromium tab/address offset should be added in this mode.
            monitor = self.grabber.monitors[1]
            if not await self._fullscreen_detected():
                self._game_fullscreen = False
                self._anchor_forced = True
                print('[SCREEN] Fullscreen exited; re-anchoring windowed canvas.', flush=True)
            else:
                self.region = {'left': monitor['left'], 'top': monitor['top'],
                               'width': monitor['width'], 'height': monitor['height']}
                box = await self.canvas.bounding_box()
                self._capture_anchor_box = box or {
                    'x': 0, 'y': 0, 'width': monitor['width'], 'height': monitor['height']}
                self._capture_anchor_region = dict(self.region)
                self._last_alignment_check = time.monotonic()
                self._fast_alignment_last = time.monotonic()
                self._anchor_forced = False
                self._anchor_full_count += 1
                return
        await super()._anchor_capture_region()

    async def _sync_capture_region(self):
        if self._game_fullscreen:
            # Fixed desktop rect while actual site fullscreen remains active.
            return
        await super()._sync_capture_region()

    async def _capture(self):
        await self._sync_capture_region()
        start = time.perf_counter()
        shot = np.asarray(self.grabber.grab(self.region))
        self._metric('mss_grab', time.perf_counter() - start)
        self._last_bgra = shot
        self._v2_last_bgra = shot
        start = time.perf_counter()
        score = self._fast_menu_score(shot)
        self._metric('menu_detect', time.perf_counter() - start)
        start = time.perf_counter()
        chw = vision.bgra_to_chw(shot)
        self._metric('image_resize', time.perf_counter() - start)
        self._v2_last_chw = chw
        return chw, float(score), None

    @staticmethod
    def _rgb_obs(stacked):
        return vision.frames_to_obs(stacked)

    def reset(self, *, seed=None, options=None):
        # Bypass the 90x160 RGB reshape hardcoded in ColorPracticeEnv.reset.
        stacked, info = LocalPracticeEnv.reset(self, seed=seed, options=options)
        self._v2_last_milestone = 0
        self._perf_resets += 1
        return self._rgb_obs(stacked), info

    async def _press_r_restart(self):
        await ctrl.focus_canvas(self.canvas)
        await self.page.keyboard.press('r', delay=25)
        print('[RESTART] Sent R for next Practice attempt.', flush=True)
        good = 0
        end = time.monotonic() + self.args.r_timeout
        while time.monotonic() < end:
            image, score, _ = await self._capture()
            good = good + 1 if (score < self.args.threshold*.40 and
                                float(np.std(image)) >= 9.0) else 0
            if good >= 2:
                return image
            await asyncio.sleep(.035)
        return None

    async def _start_episode(self):
        start_time = time.perf_counter()
        # R is the same start control for episode 1 and every later episode.
        # It avoids hunting for HTML Practice (which is hidden in fullscreen).
        if self.args.r_restart:
            for attempt in range(1, self.args.r_attempts + 1):
                self._anchor_start_attempts += 1
                try:
                    image = await self._press_r_restart()
                    if image is not None:
                        self._metric('r_restart', time.perf_counter() - start_time)
                        print(f'[RESTART] R succeeded for episode {self.episode_number} '
                              f'(attempt {attempt}).', flush=True)
                        if self.episode_number == 1:
                            await self._save_capture_debug('startup_after_r.jpg')
                        preview = self._encode_last_frame() if self.episode_number == 1 else None
                        return image, preview
                except Exception as exc:
                    print(f'[RESTART] R attempt {attempt} failed: '
                          f'{str(exc)[:150]}', flush=True)
                if attempt < self.args.r_attempts:
                    print(f'[RESTART] R not yet confirmed (attempt {attempt}); retrying.',
                          flush=True)
                    await asyncio.sleep(.12)
            print('[RESTART] R did not confirm gameplay; using HTML fallback.', flush=True)

        if self._game_fullscreen:
            # The HTML Practice control is outside the fullscreen Unity element.
            await self.page.keyboard.press('Escape')
            self._game_fullscreen = False
            self._anchor_forced = True
            self._capture_anchor_region = None
            await asyncio.sleep(.2)

        try:
            image, raw = await super()._start_episode()
        except Exception:
            await self._save_capture_debug(
                f'startup_failed_episode_{self.episode_number:04d}.jpg')
            raise
        if self.args.fullscreen and not self._game_fullscreen:
            print('[SCREEN] HTML fallback started Practice; restoring site fullscreen.', flush=True)
            await self._enter_game_fullscreen()
            if self._game_fullscreen:
                image, score, _ = await self._capture()
                print(f'[SCREEN] Fullscreen live capture: menu_score={score:.3f}, '
                      f'pixels_std={float(np.std(image)):.1f}, '
                      f'region={self.region}', flush=True)
                await self._save_capture_debug('startup_after_fullscreen.jpg')
                if self.episode_number == 1:
                    raw = self._encode_last_frame()
        return image, raw

    def step(self, action):
        start = time.perf_counter()
        stacked, reward, terminated, truncated, info = LocalPracticeEnv.step(self, int(action))
        obs = self._rgb_obs(stacked)
        # Original reward is 0.02 * real_dt * fps - death_penalty.
        # Preserve approximately the old 20fps scale independently of fps.
        death = 1.0 if terminated else 0.0
        reward = (reward + death) * (20.0 / self.args.fps) - death
        # Action-NEUTRAL milestone reward. No incentive to spam a movement.
        elapsed = float(info.get('duration', 0.0))
        milestone = int(elapsed // self.args.milestone_seconds)
        if not terminated and milestone > self._v2_last_milestone:
            reward += self.args.milestone_bonus * (milestone - self._v2_last_milestone)
        self._v2_last_milestone = max(self._v2_last_milestone, milestone)
        info['milestones'] = milestone
        if truncated and self._last_bgra is not None:
            p = self.output_dir / f'watchdog_{self.episode_number:05d}.jpg'
            cv2.imwrite(str(p), cv2.cvtColor(self._last_bgra, cv2.COLOR_BGRA2BGR))
        self._metric('env_step', time.perf_counter() - start)
        self._perf_steps += 1
        self._report_if_due()
        return obs, float(reward), terminated, truncated, info

    async def _install_demo_hooks(self):
        success = 0
        for f in self.page.frames:
            try:
                await f.evaluate(EVENT_JS)
                success += 1
            except Exception:
                pass
        if not success:
            raise RuntimeError('Could not install keyboard event listener in browser frames')
        return success

    async def _drain_demo_events(self):
        result = []
        for f in self.page.frames:
            try:
                result.extend(await f.evaluate(DRAIN_JS))
            except Exception:
                continue
        return result

    async def _collect_episode(self, deadline):
        await self._install_demo_hooks()
        await ctrl.focus_canvas(self.canvas)
        await self._drain_demo_events()
        recent = deque(maxlen=16)
        recent.append((time.time(), self._rgb_obs(np.stack(self.frames))))
        pressed = set()
        idle_since = time.monotonic()
        samples = 0
        skipped = 0
        menu_streak = 0
        previous_events = set()
        assigned_at_frame = {}
        print('[HUMAN] Play normally using arrows. R restarts after death automatically.', flush=True)
        while time.monotonic() < deadline:
            await asyncio.sleep(1.0/self.args.human_fps)
            # Capture before draining events; use timestamp to select a frame
            # from BEFORE the user's actual keypress, never one from after it.
            frame, score, _ = await self._capture()
            self.frames.append(frame)
            recent.append((time.time(), self._rgb_obs(np.stack(self.frames))))
            events = await self._drain_demo_events()
            for ev in sorted(events, key=lambda e: e.get('t', 0)):
                code = ev.get('code', '')
                if code not in KEYS:
                    continue
                if not ev.get('down', False):
                    pressed.discard(code)
                    continue
                pressed.add(code)
                t = float(ev.get('t', 0))
                signature = (code, round(t, 3))
                if signature in previous_events:
                    continue
                previous_events.add(signature)
                previous_events = set(list(previous_events)[-100:])
                candidates = [(ts, o) for ts, o in recent if ts <= t]
                if not candidates:
                    skipped += 1
                    continue
                ts, older_obs = candidates[-1]
                lag = t - ts
                if not 0 <= lag <= self.args.max_demo_lag:
                    skipped += 1
                    continue
                # One discrete policy decision per captured frame: if the user
                # presses two DIFFERENT arrows faster than we can take a new
                # screenshot, assigning both to identical pixels creates
                # contradictory imitation labels. Keep the first and skip
                # subsequent ambiguous keydowns for that observation.
                if ts in assigned_at_frame and assigned_at_frame[ts] != KEYS[code]:
                    skipped += 1
                    continue
                assigned_at_frame[ts] = KEYS[code]
                while len(assigned_at_frame) > 20:
                    assigned_at_frame.pop(next(iter(assigned_at_frame)))
                self.demos.add(older_obs, KEYS[code], lag)
                samples += 1
                idle_since = time.monotonic()
            if not pressed and time.monotonic() - idle_since >= .27:
                # Record genuinely idle decisions, not held keys.
                self.demos.add(recent[-1][1], 0, 0.0)
                samples += 1
                idle_since = time.monotonic()
            menu_streak = menu_streak + 1 if score >= self.args.threshold*.85 else 0
            if menu_streak >= 3:
                break
        self.demos.flush()
        print(f'[HUMAN] Run complete: {samples} labeled frames; '
              f'{skipped} poorly aligned keydowns skipped.', flush=True)
        return samples

    def collect_one(self, deadline):
        self.reset()
        self._call(self._collect_episode(deadline), timeout=max(20, deadline - time.monotonic() + 30))


class V2Callback(BaseCallback):
    def __init__(self, output, args, demos):
        super().__init__()
        self.output = Path(output)
        self.args = args
        self.demos = demos
        self.started = time.monotonic()
        self.last_save = self.started
        self.last_diag = self.started
        self.rollouts = 0

    def _on_rollout_start(self):
        self.rollouts += 1
        if self.args.bc_updates:
            n = bc_updates(self.model, self.demos, self.args.bc_updates,
                           self.args.bc_batch_size, self.args.bc_weight,
                           seed=self.args.seed+self.rollouts)
            if n and self.rollouts % 15 == 0:
                validation_stats(self.model, self.demos, batches=3)

    def _save_audit(self):
        try:
            obs = self.locals.get('new_obs')
            if obs is None:
                return
            sample = np.asarray(obs)[0]
            with torch.no_grad():
                probs = self.model.policy.get_distribution(
                    torch.as_tensor(sample[None], device=self.model.device)).distribution.probs
            probs = probs[0].cpu().numpy()
            rgb = vision.latest_rgb(sample)
            label = ' '.join(f'{name}:{p:.2f}' for name,p in zip(vision.ACTIONS, probs))
            image = vision.annotate(rgb, text=label)
            out = self.output / 'audits'
            out.mkdir(exist_ok=True)
            path = out / f'step_{self.model.num_timesteps:08d}.png'
            cv2.imwrite(str(path), cv2.resize(image, (vision.WIDTH*4,vision.HEIGHT*4),
                                             interpolation=cv2.INTER_NEAREST))
            (out / f'step_{self.model.num_timesteps:08d}.json').write_text(json.dumps({
                'probabilities': dict(zip(vision.ACTIONS, map(float, probs))),
                'argmax': vision.ACTIONS[int(probs.argmax())],
                'candidate_coins': vision.candidate_coins(rgb),
                'note': 'Candidate coin circles are NOT validated and not used as PPO rewards.'
            }, indent=2))
            print(f'[VISION] Policy audit saved: {path}; {label}', flush=True)
        except Exception as exc:
            print(f'[VISION] Audit skipped: {str(exc)[:110]}', flush=True)

    def _on_step(self):
        now = time.monotonic()
        if now - self.last_save >= self.args.save_every_minutes * 60:
            self.model.save(self.output / 'latest')
            self.demos.flush()
            self.last_save = now
            print(f'[RL] Saved {self.model.num_timesteps} steps', flush=True)
        if self.args.audit_seconds > 0 and now - self.last_diag >= self.args.audit_seconds:
            self._save_audit()
            self.last_diag = now
        return now - self.started < self.args.max_minutes*60


class PipelineVerifyCallback(V2Callback):
    """Check real scene variability, executed actions, rewards and PPO updates."""
    def __init__(self, output, args, demos):
        super().__init__(output, args, demos)
        self.action_counts = Counter()
        self.reward_total = 0.0
        self.reward_min = float('inf')
        self.reward_max = -float('inf')
        self.terminal_count = 0
        self.observation_digests = set()
        self.pixel_stds = []
        self.policy_probs = []
        self.last_obs = None
        self.step_count = 0
        self.first_obs = None
        self.last_obs_rgb = None

    def _on_step(self):
        self.step_count += 1
        actions = np.asarray(self.locals.get('actions', [0])).reshape(-1)
        rewards = np.asarray(self.locals.get('rewards', [0])).reshape(-1)
        dones = np.asarray(self.locals.get('dones', [False])).reshape(-1)
        self.action_counts.update(vision.ACTIONS[int(x)] for x in actions)
        for r in rewards:
            self.reward_total += float(r)
            self.reward_min = min(self.reward_min, float(r))
            self.reward_max = max(self.reward_max, float(r))
        self.terminal_count += sum(bool(x) for x in dones)
        if self.step_count % 8 == 0:
            obs = np.asarray(self.locals.get('new_obs'))
            if obs.ndim == 4 and obs.shape[1:] == (vision.CHANNELS, vision.HEIGHT, vision.WIDTH):
                single = np.array(obs[0], copy=True)
                if self.first_obs is None:
                    self.first_obs = single
                self.last_obs = single
                self.observation_digests.add(hashlib.sha256(single.tobytes()).hexdigest())
                self.pixel_stds.append(float(np.std(single)))
                with torch.no_grad():
                    probs = self.model.policy.get_distribution(
                        torch.as_tensor(single[None], device=self.model.device))
                    self.policy_probs.append(probs.distribution.probs[0].detach().cpu().numpy())
        return super()._on_step()


def actor_tensors(model):
    # Separate encoder and action head to prove the VISUAL POLICY changed,
    # rather than just its value function or optimizer bookkeeping.
    p = model.policy
    return {
        'visual_encoder': [t.detach().cpu().clone() for t in p.features_extractor.parameters()],
        'actor_head': [t.detach().cpu().clone() for t in p.action_net.parameters()],
        'value_head': [t.detach().cpu().clone() for t in p.value_net.parameters()],
    }


def weight_change(before, after):
    result = {}
    for name in before:
        squared = 0.; elements = 0
        for x, y in zip(before[name], after[name]):
            delta = (y.double() - x.double())
            squared += float((delta * delta).sum())
            elements += int(delta.numel())
        result[name] = {'l2_change': round(squared**.5, 8), 'parameters': elements}
    return result


def finish_verification(model, callback, initial_weights, output):
    weights = weight_change(initial_weights, actor_tensors(model))
    sample = callback.last_obs
    probs_before = None
    probs_reloaded = None
    reload_match = None
    if sample is not None:
        with torch.no_grad():
            p = model.policy.get_distribution(
                torch.as_tensor(sample[None], device=model.device))
            probs_before = p.distribution.probs[0].cpu().numpy()
    model.save(output / 'latest')
    reloaded = PPO.load(str(output/'latest.zip'), device='cpu')
    if sample is not None:
        with torch.no_grad():
            p2 = reloaded.policy.get_distribution(torch.as_tensor(sample[None]))
            probs_reloaded = p2.distribution.probs[0].cpu().numpy()
        reload_match = bool(np.allclose(probs_before, probs_reloaded, atol=1e-5, rtol=1e-5))
    variation = np.stack(callback.policy_probs) if callback.policy_probs else None
    summary = {
        'steps': callback.step_count,
        'episodes_ended': callback.terminal_count,
        'action_counts': dict(callback.action_counts),
        'reward_mean': callback.reward_total/max(1, callback.step_count),
        'reward_min': callback.reward_min if callback.step_count else None,
        'reward_max': callback.reward_max if callback.step_count else None,
        'unique_images_sampled': len(callback.observation_digests),
        'sampled_images': len(callback.pixel_stds),
        'mean_image_std': float(np.mean(callback.pixel_stds)) if callback.pixel_stds else None,
        'action_probability_std_per_action': dict(zip(vision.ACTIONS,
             np.std(variation, axis=0).round(6).tolist())) if variation is not None else {},
        'action_probability_means': dict(zip(vision.ACTIONS,
             np.mean(variation, axis=0).round(5).tolist())) if variation is not None else {},
        'gradient_updates': int(model._n_updates),
        'weights': weights,
        'checkpoint_probability_match': reload_match,
        'note': 'Checks that the pipeline works, NOT whether it plays well or recognizes hazards.'
    }
    (output/'pipeline_verification.json').write_text(json.dumps(summary, indent=2))
    for group, results in weights.items():
        print(f'[VERIFY] {group} L2 weight change={results["l2_change"]:.8f}', flush=True)
    print(f'[VERIFY] steps={callback.step_count}, PPO updates={model._n_updates}, '
          f'deaths={callback.terminal_count}, distinct game images={len(callback.observation_digests)}, '
          f'probability checkpoint reload match={reload_match}', flush=True)
    print(f'[VERIFY] Action counts: {dict(callback.action_counts)}', flush=True)
    print(f'[VERIFY] Probability standard deviations: '
          f'{summary["action_probability_std_per_action"]}', flush=True)
    print(f'[VERIFY] Report: {output/"pipeline_verification.json"}', flush=True)
    if not (model._n_updates > 0 and weights['actor_head']['l2_change'] > 1e-8 and
            weights['visual_encoder']['l2_change'] > 1e-8 and
            len(callback.observation_digests) > 1 and reload_match):
        print('[VERIFY] INCOMPLETE: one or more pipeline checks failed. '
              'Do NOT collect demonstrations yet.', flush=True)
    else:
        print('[VERIFY] PASS: real screenshots -> actions -> rewards -> PPO '
              'weight updates -> saved/reloaded checkpoint. '
              'This does NOT yet demonstrate useful gameplay learning.', flush=True)


def parser_args():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument('--mode', choices=('smoke','collect','train','verify'), default='smoke')
    p.add_argument('--url', default='http://localhost:8000')
    p.add_argument('--run-name', default=None)
    p.add_argument('--resume', type=Path)
    p.add_argument('--fullscreen', action='store_true', help='Click site fullscreen icon, never Chrome F11')
    p.add_argument('--r-restart', action=argparse.BooleanOptionalAction, default=True)
    p.add_argument('--r-timeout', type=float, default=1.15)
    p.add_argument('--r-attempts', type=int, default=3, help='Try R this many times before HTML fallback')
    p.add_argument('--fps', type=int, default=20)
    p.add_argument('--human-fps', type=int, default=25)
    p.add_argument('--max-demo-lag', type=float, default=.23)
    p.add_argument('--max-minutes', type=float, default=3)
    p.add_argument('--steps', type=int, default=100000)
    p.add_argument('--pretrain-updates', type=int, default=500)
    p.add_argument('--bc-updates', type=int, default=1)
    p.add_argument('--bc-weight', type=float, default=.5)
    p.add_argument('--bc-batch-size', type=int, default=32)
    p.add_argument('--demo-max', type=int, default=2200)
    p.add_argument('--save-every-minutes', type=float, default=5)
    p.add_argument('--audit-seconds', type=float, default=45)
    p.add_argument('--milestone-seconds', type=float, default=2)
    p.add_argument('--milestone-bonus', type=float, default=.10)
    p.add_argument('--alignment-seconds', type=float, default=10)
    p.add_argument('--full-anchor-every', type=int, default=8)
    p.add_argument('--profile-seconds', type=float, default=30)
    p.add_argument('--threshold', type=float, default=.68)
    p.add_argument('--episode-max-seconds', type=float, default=60)
    p.add_argument('--load-timeout', type=float, default=25)
    p.add_argument('--debug-every-seconds', type=float, default=12)
    p.add_argument('--screen-x-offset', type=int, default=0)
    p.add_argument('--screen-y-offset', type=int, default=0)
    p.add_argument('--seed', type=int, default=123)
    args = p.parse_args()
    if not args.url.startswith(('http://localhost:', 'http://127.0.0.1:')):
        p.error('V2 only works with your LOCAL Practice game.')
    if not (2 <= args.fps <= 30 and 5 <= args.human_fps <= 30 and
            .2 <= args.threshold <= .99 and args.max_minutes > 0 and
            0 < args.r_timeout <= 8 and 1 <= args.r_attempts <= 6 and args.demo_max >= 100 and
            args.pretrain_updates >= 0 and args.bc_updates >= 0 and
            0 <= args.milestone_bonus <= 2 and args.milestone_seconds >= 1):
        p.error('Invalid numerical settings')
    if args.resume and not args.resume.is_file():
        p.error(f'Checkpoint does not exist: {args.resume}')
    if args.run_name and (args.run_name in ('.', '..') or Path(args.run_name).name != args.run_name):
        p.error('--run-name must be one folder name')
    if args.mode == 'verify':
        # Independent PPO learning test, deliberately NO demonstrations.
        # Never take a long training session by accident in this mode.
        args.steps = min(args.steps, 768)
        args.pretrain_updates = 0
        args.bc_updates = 0
        args.audit_seconds = min(args.audit_seconds, 15.)
        if args.resume:
            p.error('--mode verify must start fresh; do not pass --resume')
    # Attributes expected by the inherited, proven fast/quick restart classes.
    args.enable_f8 = False
    args.legacy_detector = False
    args.legacy_restart = False
    args.collect_only = args.mode == 'collect'
    args.human_max_seconds = max(180, args.max_minutes*60)
    return args


def run():
    args = parser_args()
    RUN_ROOT.mkdir(parents=True, exist_ok=True)
    DEMO_ROOT.mkdir(parents=True, exist_ok=True)
    run_name = args.run_name or datetime.now().strftime('%Y%m%d_%H%M%S')
    output = RUN_ROOT / run_name
    if args.mode in ('train','verify') and not args.resume and (output/'latest.zip').exists():
        raise SystemExit('Refusing to overwrite an existing model; choose a new --run-name')
    output.mkdir(parents=True, exist_ok=True)
    (output / 'settings.json').write_text(json.dumps(vars(args), default=str, indent=2))
    demos = DemoStore(DEMO_ROOT, max_samples=args.demo_max, seed=args.seed,
                      load_existing=(args.mode == 'train'))
    if (args.mode == 'train' and not args.resume and args.pretrain_updates and
            len(demos.train_obs) < 8):
        raise SystemExit('No usable V2 demonstrations found. First run: '
                         'python train_rs_v2.py --mode collect --fullscreen --max-minutes 15')
    env = None
    model = None
    try:
        if args.mode in ('smoke','collect'):
            env = V2PracticeEnv(args, output, demos)
            until = time.monotonic() + args.max_minutes*60
            if args.mode == 'collect':
                while time.monotonic() < until:
                    env.collect_one(until)
            else:
                # Random actions for validation, NOT PPO training.
                rng = np.random.default_rng(args.seed)
                for run_number in range(1, 7):
                    if time.monotonic() >= until: break
                    obs, _ = env.reset()
                    print(f'[SMOKE] Episode {run_number}, screenshot shape={obs.shape}, '
                          f'fullscreen={env._game_fullscreen}', flush=True)
                    if run_number == 1:
                        # The PRECISE network input, not just a full-size preview.
                        cv2.imwrite(str(output / 'actual_model_input.png'),
                                    cv2.cvtColor(vision.latest_rgb(obs), cv2.COLOR_RGB2BGR))
                    last_smoke_report = time.monotonic()
                    for _ in range(300):
                        if time.monotonic() >= until: break
                        action = int(rng.choice(5, p=[.60,.10,.10,.10,.10]))
                        obs, reward, dead, truncated, info = env.step(action)
                        if time.monotonic() - last_smoke_report >= 3.0:
                            print(f'[SMOKE] Active: elapsed={info.get("duration")}, '
                                  f'menu_score={info.get("menu_similarity")}, '
                                  f'suspected={env.suspected_since is not None}', flush=True)
                            last_smoke_report = time.monotonic()
                        if dead or truncated:
                            print(f'[SMOKE] End: {info}; dead={dead}, timeout={truncated}', flush=True)
                            break
            return
        device = 'cuda' if torch.cuda.is_available() else 'cpu'
        print('[RL] NEW model: 4 RGB frames x 256x144, 12 channels; '
              f'device={device}; image shape={(vision.CHANNELS,vision.HEIGHT,vision.WIDTH)}', flush=True)
        env = Monitor(V2PracticeEnv(args, output, demos), filename=str(output/'episodes.monitor.csv'))
        if args.resume:
            model = PPO.load(str(args.resume), env=env, device=device)
            if tuple(model.observation_space.shape) != (vision.CHANNELS,vision.HEIGHT,vision.WIDTH):
                raise RuntimeError('Checkpoint input size differs! Must use a V2 checkpoint.')
            print(f'[RL] Resumed {args.resume}', flush=True)
        else:
            model = PPO('CnnPolicy', env, device=device, verbose=1,
                        n_steps=128, batch_size=32, n_epochs=4, seed=args.seed,
                        learning_rate=3e-4, gamma=.99, gae_lambda=.95,
                        ent_coef=.02, clip_range=.2,
                        policy_kwargs={
                            'features_extractor_class': TinyCNN,
                            'features_extractor_kwargs': {'features_dim': 128},
                            'net_arch': dict(pi=[64], vf=[64])})
            if args.pretrain_updates:
                print(f'[BC] New model pretraining: {args.pretrain_updates} updates', flush=True)
                count = bc_updates(model, demos, args.pretrain_updates,
                                   args.bc_batch_size, args.bc_weight)
                if count == 0:
                    print('[BC] NO demonstrations loaded. Run --mode collect first!', flush=True)
                else:
                    validation_stats(model, demos)
                model.save(output / 'after_pretraining')
        print('[RL] Starting independent PPO (local Practice only)', flush=True)
        if args.mode == 'verify':
            print('[VERIFY] No human demonstrations. Training a TEMPORARY model to '
                  'check the complete pipeline; do not treat it as your final model.', flush=True)
            initial_weights = actor_tensors(model)
            callback = PipelineVerifyCallback(output, args, demos)
        else:
            callback = V2Callback(output, args, demos)
        model.learn(total_timesteps=args.steps, callback=callback,
                    reset_num_timesteps=not bool(args.resume))
        if args.mode == 'verify':
            finish_verification(model, callback, initial_weights, output)
    except KeyboardInterrupt:
        print('[V2] Ctrl+C: saving available data.', flush=True)
    finally:
        demos.flush()
        if model is not None:
            model.save(output/'latest')
            print(f'[V2] Saved model: {output}/latest.zip', flush=True)
        if env is not None:
            env.close()
        print(f'[V2] Output: {output}', flush=True)

if __name__ == '__main__':
    run()
