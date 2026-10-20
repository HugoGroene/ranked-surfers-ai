#!/usr/bin/env python3
"""Experimental PPO trainer for a LOCAL Unity practice copy.

Reuses rs_practice_controller.py for HTML Practice clicking, MSS screenshot
capture, and a text-color/shape matcher for the 'FIND RANKED MATCH' menu. This only accepts
localhost; it never visits ranked matchmaking or the public website.

Install into your venv:
  python -m pip install "stable-baselines3>=2.3,<3" gymnasium mss opencv-python-headless

Smoke test (leave Chromium visible):
  python train_rs_rl_text.py --max-minutes 3 --steps 500 --fps 10 --threshold 0.68 --episode-max-seconds 25

Longer run:
  python train_rs_rl_text.py --max-minutes 60 --steps 25000 --fps 10 --threshold 0.68

Resume a previous saved model:
  python train_rs_rl_text.py --resume ai_data/rl_runs/<RUN>/latest.zip --max-minutes 60

This is an EXPERIMENTAL trainer, not a proven winning agent. RL from pixels
with a terminal-only survival reward is highly sample-inefficient. This version
tracks desktop capture alignment after scrolling and retries very fast deaths.
Watch the first smoke test and verify it never clicks the Unity multiplayer start screen.
"""
import argparse
import asyncio
from collections import deque
from datetime import datetime
import io
import json
from pathlib import Path
import shutil
import threading
import time

import cv2
import gymnasium as gym
import mss
from PIL import Image
import numpy as np
from gymnasium import spaces
import torch
from torch import nn
from stable_baselines3 import PPO
from stable_baselines3.common.callbacks import BaseCallback
from stable_baselines3.common.monitor import Monitor
from stable_baselines3.common.torch_layers import BaseFeaturesExtractor

import rs_practice_controller as ctrl

ACTIONS = [None, "ArrowLeft", "ArrowRight", "ArrowUp", "ArrowDown"]
IMAGE_W, IMAGE_H = 160, 90


class MenuTextDetector:
    """Recognize stationary light-colored 'FIND RANKED MATCH' glyphs, not animated scenery.

    Reference images are the cropped 256x160 PNGs created by the existing
    rs_practice_controller.py --mode calibrate, saved under ai_data/start_templates.
    Bright cream/yellow characters survive changes to the moving train and runner.
    """
    def __init__(self, paths, roi):
        self.roi = roi
        self.kernel = np.ones((3, 3), dtype=np.uint8)
        self.refs = []
        for path in paths:
            with Image.open(path) as src:
                mask = self._foreground(src)
            if np.count_nonzero(mask) >= 250:
                self.refs.append((mask, cv2.dilate(mask, self.kernel)))
        if not self.refs:
            raise RuntimeError(
                "No bright menu text in saved calibration templates. "
                "Re-run --mode calibrate with a tight ROI covering FIND RANKED MATCH; "
                "inspect ai_data/start_roi_preview.png."
            )

    @staticmethod
    def _foreground(image):
        rgb = np.asarray(image.convert('RGB'), dtype=np.int16)
        r, g, b = rgb[:, :, 0], rgb[:, :, 1], rgb[:, :, 2]
        # Cream-white/yellow menu text: exclude dark outlines and most scenery.
        mask = (r >= 190) & (g >= 180) & (b >= 110) & (r - g < 80) & (r >= b)
        mask = (mask.astype(np.uint8) * 255)
        return cv2.medianBlur(mask, 3)

    def score(self, canvas_image):
        sample = self._foreground(ctrl.crop_image(canvas_image, self.roi))
        count = cv2.countNonZero(sample)
        if count < 100:
            return 0.0
        sample_dilated = cv2.dilate(sample, self.kernel)
        best = 0.0
        for ref, ref_dilated in self.refs:
            # F1-type foreground overlap, with a 1-pixel tolerance on each side.
            # Unlike Canny correlation, this ignores moving dark/background objects.
            correct_ref = cv2.countNonZero(cv2.bitwise_and(ref, sample_dilated))
            correct_sample = cv2.countNonZero(cv2.bitwise_and(sample, ref_dilated))
            recall = correct_ref / max(1, cv2.countNonZero(ref))
            precision = correct_sample / max(1, count)
            score = 2 * precision * recall / max(1e-8, precision + recall)
            best = max(best, score)
        return float(best)


class TinyCNN(BaseFeaturesExtractor):
    """Small CNN so a 4 GB GTX 1650 can run PPO on 4 stacked grayscale frames."""
    def __init__(self, observation_space: spaces.Box, features_dim: int = 128):
        super().__init__(observation_space, features_dim)
        n = observation_space.shape[0]
        self.net = nn.Sequential(
            nn.Conv2d(n, 16, 5, stride=2, padding=2), nn.ReLU(),
            nn.Conv2d(16, 32, 3, stride=2, padding=1), nn.ReLU(),
            nn.Conv2d(32, 48, 3, stride=2, padding=1), nn.ReLU(),
            nn.AdaptiveAvgPool2d((3, 6)), nn.Flatten(),
            nn.Linear(48 * 3 * 6, features_dim), nn.ReLU(),
        )

    def forward(self, observations: torch.Tensor):
        return self.net(observations)


class LocalPracticeEnv(gym.Env):
    """Gym API driving an async browser in a worker thread.

    The Unity WebGL tab has to stay on screen because MSS records the desktop.
    One action = one key TAP (or NOOP); the policy acts at --fps Hz.
    """
    metadata = {"render_modes": []}

    def __init__(self, args, output_dir):
        super().__init__()
        self.args = args
        self.output_dir = Path(output_dir)
        self.observation_space = spaces.Box(0, 255, (4, IMAGE_H, IMAGE_W), dtype=np.uint8)
        self.action_space = spaces.Discrete(len(ACTIONS))
        self.frames = deque(maxlen=4)
        self.episode_number = 0
        self.episode_start = 0.0
        self.menu_scores = deque(maxlen=8)
        self.suspected_since = None
        self.clear_streak = 0
        self.last_debug = time.monotonic()
        self.last_step = time.monotonic()
        self.closed = False
        self.page = None
        self.browser = None
        self.context = None
        self.playwright = None
        self.grabber = None
        self.canvas = None
        self.region = None
        self._capture_anchor_box = None
        self._capture_anchor_region = None
        self._last_alignment_check = 0.0
        self.worker_loop = asyncio.new_event_loop()
        self.worker_thread = threading.Thread(target=self._run_async_loop, daemon=True)
        self.worker_thread.start()

        if not ctrl.TEMPLATE_INFO.is_file():
            self.close()
            raise RuntimeError("Missing death-screen calibration. Run the controller --mode calibrate first.")
        self.roi = json.loads(ctrl.TEMPLATE_INFO.read_text(encoding="utf8"))["roi"]
        references = sorted(ctrl.TEMPLATES.glob("*.png")) if ctrl.TEMPLATES.is_dir() else []
        if not references:
            references = [ctrl.TEMPLATE]
        if not all(path.exists() for path in references):
            self.close()
            raise RuntimeError("Missing start-screen template images.")
        self.menu_detector = MenuTextDetector(references, self.roi)
        self.last_raw = None
        print(f"Text detector: {len(self.menu_detector.refs)} saved menu samples; "
              f"ROI={self.roi}; threshold={args.threshold:.2f}", flush=True)

        try:
            self._call(self._open_browser(), timeout=120)
        except Exception:
            self.close()
            raise

    def _run_async_loop(self):
        asyncio.set_event_loop(self.worker_loop)
        self.worker_loop.run_forever()

    def _call(self, coro, timeout=60):
        task = asyncio.run_coroutine_threadsafe(coro, self.worker_loop)
        return task.result(timeout=timeout)

    async def _open_browser(self):
        self.playwright = await ctrl.async_playwright().start()
        launch_args = ["--start-maximized"]
        try:
            self.browser = await self.playwright.chromium.launch(headless=False, args=launch_args)
        except Exception:
            executable = shutil.which("chromium") or shutil.which("chromium-browser")
            if not executable:
                raise
            self.browser = await self.playwright.chromium.launch(
                headless=False, executable_path=executable, args=launch_args)
        self.context = await self.browser.new_context(no_viewport=True)
        self.page = await self.context.new_page()
        try:
            cdp = await self.context.new_cdp_session(self.page)
            window = await cdp.send("Browser.getWindowForTarget")
            await cdp.send("Browser.setWindowBounds", {
                "windowId": window["windowId"],
                "bounds": {"windowState": "maximized"},
            })
        except Exception as exc:
            print("Window maximize warning:", str(exc)[:160])
        await self.page.goto(self.args.url, wait_until="domcontentloaded", timeout=45000)
        await self.page.bring_to_front()
        self.grabber = mss.MSS()  # Only accessed from this worker thread.
        self.canvas = await ctrl.find_canvas(self.page, timeout=60)
        await self._anchor_capture_region()
        print("Browser ready. IMPORTANT: keep Chromium visible and unobstructed.", flush=True)

    async def _anchor_capture_region(self):
        """Record the exact canvas geometry whenever the page is initialized or restarted."""
        self.region = await ctrl.mss_region_for_canvas(
            self.page, self.canvas, self.grabber, self.args)
        self._capture_anchor_box = await self.canvas.bounding_box()
        self._capture_anchor_region = dict(self.region)
        self._last_alignment_check = 0.0

    async def _sync_capture_region(self):
        """Track scrolling during play without scrolling the page ourselves.

        MSS takes desktop coordinates, but Unity's canvas is inside a scrolling
        page. If the canvas moves, the old crop becomes invalid; moving the
        browser page manually used to accidentally 'fix' missed menu detections.
        """
        now = time.monotonic()
        if now - self._last_alignment_check < 0.25:
            return
        self._last_alignment_check = now
        if not self._capture_anchor_box or not self._capture_anchor_region:
            return
        box = await self.canvas.bounding_box()
        if not box or box["width"] < 40 or box["height"] < 40:
            return
        anchor = self._capture_anchor_box
        base = self._capture_anchor_region
        scale = base["width"] / anchor["width"]
        proposed = dict(base)
        proposed["left"] += round((box["x"] - anchor["x"]) * scale)
        proposed["top"] += round((box["y"] - anchor["y"]) * scale)
        proposed["width"] = round(box["width"] * scale)
        proposed["height"] = round(box["height"] * scale)
        screen = self.grabber.monitors[0]
        if not (screen["left"] <= proposed["left"] and
                screen["top"] <= proposed["top"] and
                proposed["left"] + proposed["width"] <= screen["left"] + screen["width"] and
                proposed["top"] + proposed["height"] <= screen["top"] + screen["height"]):
            # Never sample outside the physical desktop (e.g. during resizing).
            return
        moved_x = proposed["left"] - self.region["left"]
        moved_y = proposed["top"] - self.region["top"]
        if abs(moved_x) >= 2 or abs(moved_y) >= 2:
            print(f"[Capture] Canvas moved on screen (dx={moved_x}, dy={moved_y}); "
                  "MSS crop realigned automatically", flush=True)
        self.region = proposed

    async def _capture(self):
        # Recheck canvas position before grabbing the desktop. No disk writes.
        await self._sync_capture_region()
        raw = ctrl.screenshot_mss(self.grabber, self.region, image_type="jpeg", quality=75)
        self.last_raw = raw
        with Image.open(io.BytesIO(raw)) as stream:
            im = stream.convert('RGB')
        score = self.menu_detector.score(im)
        gray = cv2.resize(cv2.cvtColor(np.asarray(im), cv2.COLOR_RGB2GRAY),
                          (IMAGE_W, IMAGE_H), interpolation=cv2.INTER_AREA)
        return gray.astype(np.uint8), float(score), raw

    async def _start_episode(self):
        """Start Practice, including recovery from deaths during the first second.

        The previous implementation slept 1.2s before sampling *anything*.
        If a coin killed the player during this gap, the screenshot could
        already show the menu. This version observes the transition promptly,
        retries if Practice never reaches confirmed gameplay, and has a bound.
        """
        max_tries = 4
        for attempt in range(1, max_tries + 1):
            for key in ctrl.KEYS:
                try:
                    await self.page.keyboard.up(key)
                except Exception:
                    pass
            await ctrl.click_practice(self.page, timeout=45)
            self.canvas = await ctrl.find_canvas(self.page, timeout=25)
            await ctrl.focus_canvas(self.canvas)
            await self._anchor_capture_region()

            clear_streak = 0
            menu_streak = 0
            first_seen = time.monotonic()
            deadline = first_seen + min(self.args.load_timeout, 8.0)
            while time.monotonic() < deadline:
                gray, score, raw = await self._capture()
                # A gameplay frame is characteristically different from the
                # stationary-text menu. Reject near-blank loading frames.
                plausible_gameplay = (score < self.args.threshold * 0.4 and
                                      float(np.std(gray)) >= 10.0)
                clear_streak = clear_streak + 1 if plausible_gameplay else 0
                if clear_streak >= 2:
                    return gray, raw

                menu_streak = (menu_streak + 1
                               if score >= self.args.threshold * 0.85 else 0)
                # The menu still being present ~1.4s after clicking Practice
                # can mean an instant death or failed game startup. Retry it.
                if (menu_streak >= 5 and
                        time.monotonic() - first_seen >= 1.4):
                    print(f"[RL] Practice returned to menu before gameplay "
                          f"(attempt {attempt}/{max_tries}); retrying", flush=True)
                    break
                await asyncio.sleep(0.055)
            else:
                print(f"[RL] No confirmed gameplay after Practice click "
                      f"(attempt {attempt}/{max_tries}); retrying", flush=True)
            await asyncio.sleep(0.2)
        raise RuntimeError(
            "Practice never reached a stable gameplay frame after 4 tries. "
            "Keep the browser visible and review the menu score/canvas position.")

    def reset(self, *, seed=None, options=None):
        super().reset(seed=seed)
        self.episode_number += 1
        gray, raw = self._call(self._start_episode(), timeout=self.args.load_timeout + 90)
        self.frames.clear()
        for _ in range(4):
            self.frames.append(gray.copy())
        self.menu_scores.clear()
        self.suspected_since = None
        self.clear_streak = 0
        self.episode_start = time.monotonic()
        self.last_debug = self.episode_start
        self.last_step = self.episode_start
        if self.episode_number == 1:
            self.output_dir.mkdir(parents=True, exist_ok=True)
            (self.output_dir / "capture_preview.jpg").write_bytes(raw)
            print("Capture preview saved:", self.output_dir / "capture_preview.jpg")
            print("Open it AFTER stopping the game to verify the correct crop.")
        print(f"[RL] Episode {self.episode_number} started", flush=True)
        return np.stack(self.frames), {}

    async def _step_browser(self, action):
        # Act on the currently observed gameplay, not a suspected death menu.
        if self.suspected_since is None and ACTIONS[action] is not None:
            await self.page.keyboard.press(ACTIONS[action], delay=25)
        # Include keypress time in the decision interval (rather than adding it).
        await asyncio.sleep(max(0, 1.0 / self.args.fps - 0.03))
        return await self._capture()

    def _menu_votes(self):
        """Consider the start-screen match over recent observations.

        Calibrated screenshots have typically scored ~0.18-0.40 on the menu
        and near zero in live gameplay. One weak frame alone is insufficient.
        """
        scores = list(self.menu_scores)[-6:]
        strong = sum(x >= self.args.threshold for x in scores)
        weak = sum(x >= self.args.threshold * 0.85 for x in scores)
        return strong, weak

    async def _wait_menu_confirmation(self):
        """Stop keyboard actions while gathering enough menu evidence.

        Don't abandon detection after one low score: the menu is animated,
        and previously this could cause the trainer to remain on the menu.
        """
        gray = None
        score = 0.0
        for _ in range(12):
            await asyncio.sleep(0.095)
            gray, score, _ = await self._capture()
            self.menu_scores.append(score)
            strong, weak = self._menu_votes()
            if strong >= 2 or weak >= 4:
                return gray, score, True
            if score < self.args.threshold * 0.45:
                self.clear_streak += 1
            else:
                self.clear_streak = 0
            if self.clear_streak >= 5 and strong == 0 and weak == 0:
                # Brief resemblance during real gameplay, not a death.
                return gray, score, False
        return gray, score, False

    def step(self, action):
        assert self.action_space.contains(action), action
        gray, score, _ = self._call(self._step_browser(int(action)), timeout=25)
        self.menu_scores.append(score)
        now = time.monotonic()
        elapsed = min(now - self.last_step, 0.3)
        self.last_step = now
        dead = False

        strong, weak = self._menu_votes()
        if (score >= self.args.threshold * 0.85 or
                self.suspected_since is not None):
            if self.suspected_since is None:
                self.suspected_since = now
                self.clear_streak = 0
                print(f"[RL] Possible menu/death (score={score:.2f}); pausing keys", flush=True)
            if strong >= 2 or weak >= 4:
                dead = True
            else:
                gray, score, dead = self._call(self._wait_menu_confirmation(), timeout=10)
                now = time.monotonic()
                self.last_step = now

        if not dead and self.suspected_since is not None:
            # A confirmed low-scoring *gameplay* interval clears suspicion;
            # don't clear it after just one mismatched menu-animation frame.
            strong, weak = self._menu_votes()
            if self.clear_streak >= 5 and strong == 0 and weak == 0:
                self.menu_scores.clear()
                self.suspected_since = None
                self.clear_streak = 0
                print('[RL] Menu suspicion cleared; resuming keys', flush=True)
            elif now - self.suspected_since > 2.0 and (strong >= 1 or weak >= 2):
                dead = True
                print('[RL] Recovery: persistent menu indicators', flush=True)
            elif now - self.suspected_since > 3.0:
                # If the match was spurious, restore controls instead of
                # waiting silently for the rest of the episode.
                self.menu_scores.clear()
                self.suspected_since = None
                self.clear_streak = 0
                print('[RL] Ambiguous menu cleared after 3s', flush=True)

        self.frames.append(gray)
        reward = 0.02 * elapsed * self.args.fps
        if dead:
            reward -= 1.0
        duration = time.monotonic() - self.episode_start
        truncated = not dead and duration >= self.args.episode_max_seconds
        if time.monotonic() - self.last_debug >= self.args.debug_every_seconds:
            print(f'[RL] heartbeat: episode={self.episode_number} '
                  f'time={duration:.1f}s menu_score={score:.3f} '
                  f'suspected={self.suspected_since is not None}', flush=True)
            self.last_debug = time.monotonic()
        info = {'duration': round(duration, 2), 'menu_similarity': round(score, 3)}
        if dead or truncated:
            if truncated and self.last_raw:
                debug_path = self.output_dir / f'watchdog_episode_{self.episode_number:04d}.jpg'
                debug_path.write_bytes(self.last_raw)
                print(f'[RL] Detector missed a possible death? Review {debug_path}', flush=True)
            reason = 'death' if dead else 'TIME-LIMIT WATCHDOG (possible missed death)' 
            print(f'[RL] Episode {self.episode_number} done | {duration:.1f}s '
                  f'| {reason} | last menu score {score:.2f}', flush=True)
        return np.stack(self.frames), float(reward), bool(dead), bool(truncated), info

    async def _close_browser(self):
        if self.grabber is not None:
            try:
                self.grabber.close()
            except Exception:
                pass
        if self.context is not None:
            try:
                await self.context.close()
            except Exception:
                pass
        if self.browser is not None:
            try:
                await self.browser.close()
            except Exception:
                pass
        if self.playwright is not None:
            try:
                await self.playwright.stop()
            except Exception:
                pass

    def close(self):
        if self.closed:
            return
        self.closed = True
        if self.worker_loop.is_running():
            try:
                self._call(self._close_browser(), timeout=20)
            except Exception as exc:
                print("Browser cleanup warning:", str(exc)[:140])
            self.worker_loop.call_soon_threadsafe(self.worker_loop.stop)
        self.worker_thread.join(timeout=4)
        if not self.worker_loop.is_closed() and not self.worker_loop.is_running():
            self.worker_loop.close()


class TimeAndCheckpointCallback(BaseCallback):
    def __init__(self, output_dir, max_minutes, save_every_minutes=10):
        super().__init__()
        self.output_dir = Path(output_dir)
        self.max_minutes = max_minutes
        self.save_every = save_every_minutes * 60
        self.started = time.monotonic()
        self.last_save = self.started

    def _on_step(self):
        now = time.monotonic()
        if now - self.last_save >= self.save_every:
            self.model.save(self.output_dir / "latest")
            self.last_save = now
            print(f"[RL] Checkpoint saved after {self.model.num_timesteps} steps", flush=True)
        return (now - self.started) < self.max_minutes * 60


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--url", default="http://localhost:8000")
    ap.add_argument("--fps", type=int, default=10, help="Decision rate, not Unity rendering FPS")
    ap.add_argument("--threshold", type=float, default=0.68, help="Foreground text-shape similarity voting threshold (NOT the old edge score)")
    ap.add_argument("--test-detector", action="store_true",
                    help="Only inspect live menu scores; click Practice manually, no PPO and no keyboard control")
    ap.add_argument("--debug-every-seconds", type=float, default=5, help="Print a menu detector heartbeat every N seconds")
    ap.add_argument("--steps", type=int, default=20000, help="Maximum RL decisions; also bounded by --max-minutes")
    ap.add_argument("--max-minutes", type=float, default=10, help="Wall-clock training limit, excluding initial browser load")
    ap.add_argument("--save-every-minutes", type=float, default=10)
    ap.add_argument("--episode-max-seconds", type=float, default=180)
    ap.add_argument("--load-timeout", type=float, default=25)
    ap.add_argument("--screen-x-offset", type=int, default=0)
    ap.add_argument("--screen-y-offset", type=int, default=0)
    ap.add_argument("--resume", type=Path, default=None, help="Resume from a saved PPO .zip checkpoint, not the imitation-learning .pt")
    ap.add_argument("--seed", type=int, default=123)
    args = ap.parse_args()

    if not args.url.startswith(("http://localhost:", "http://127.0.0.1:")):
        ap.error("Only a local practice game is supported (localhost), not the public ranked website")
    if not (2 <= args.fps <= 30 and 0.35 <= args.threshold <= 0.98
            and 10 <= args.steps <= 10_000_000 and args.max_minutes > 0
            and args.save_every_minutes > 0 and args.episode_max_seconds >= 5
            and args.load_timeout >= 5 and args.debug_every_seconds >= 1):
        ap.error("FPS, threshold, time limits, or step counts out of range")

    root = ctrl.DATA / "rl_runs"
    root.mkdir(parents=True, exist_ok=True)
    output = root / datetime.now().strftime("%Y%m%d_%H%M%S")
    output.mkdir(parents=True, exist_ok=True)
    (output / "settings.json").write_text(json.dumps(vars(args), indent=2, default=str))
    if args.test_detector:
        print("DETECTOR TEST: no PPO, no keypresses; you can click Practice manually.")
        print("Wait for the game to load, then let a practice attempt end.")
        env = None
        was_menu = False
        recorded = False
        try:
            env = LocalPracticeEnv(args, output)
            start = time.monotonic()
            while time.monotonic() - start < args.max_minutes * 60:
                _, score, raw = env._call(env._capture(), timeout=12)
                menu = score >= args.threshold
                print(f"Menu text similarity: {score:.3f} | "
                      f"{'MENU' if menu else 'gameplay/other'}", flush=True)
                if menu and not recorded:
                    img = Image.open(io.BytesIO(raw)).convert('RGB')
                    mask = env.menu_detector._foreground(ctrl.crop_image(img, env.roi))
                    filename = output / 'recognized_menu_text_mask.png'
                    Image.fromarray(mask).save(filename)
                    print(f"Captured bright-letter mask: {filename}", flush=True)
                    recorded = True
                was_menu = menu
                time.sleep(0.4)
        except KeyboardInterrupt:
            print('Detector test interrupted.')
        finally:
            if env is not None:
                env.close()
        return

    device = "cuda" if torch.cuda.is_available() else "cpu"
    print("PPO device:", device)
    if device == "cuda":
        print("GPU:", torch.cuda.get_device_name(0))
    print("Local practice only. A visible Chromium desktop is REQUIRED for MSS.")
    print("Using bright-text recognition, not the legacy animated-background edge score.")
    print("Run folder:", output)
    env = None
    model = None
    try:
        env = Monitor(LocalPracticeEnv(args, output), filename=str(output / "episodes.monitor.csv"))
        if args.resume:
            model = PPO.load(str(args.resume), env=env, device=device)
            print("Resumed checkpoint:", args.resume)
        else:
            model = PPO(
                "CnnPolicy", env, device=device, verbose=1, seed=args.seed,
                n_steps=128, batch_size=32, n_epochs=4,
                learning_rate=3e-4, gamma=0.99, gae_lambda=0.95,
                ent_coef=0.02, clip_range=0.2,
                policy_kwargs={
                    "features_extractor_class": TinyCNN,
                    "features_extractor_kwargs": {"features_dim": 128},
                    "net_arch": dict(pi=[64], vf=[64]),
                },
                tensorboard_log=None,
            )
            # A uniformly random 10Hz action stream is very self-destructive.
            # Initially bias toward 'NONE' without fixing the action forever.
            with torch.no_grad():
                model.policy.action_net.bias[0] += 1.6
        callback = TimeAndCheckpointCallback(output, args.max_minutes,
                                             save_every_minutes=args.save_every_minutes)
        print("Starting PPO. This can take hours to show improvement, or fail to learn.")
        model.learn(total_timesteps=args.steps, callback=callback,
                    reset_num_timesteps=not bool(args.resume))
    except KeyboardInterrupt:
        print('Interrupted; saving a checkpoint if a model is available.')
    except Exception as exc:
        # If the user closes Chromium, Playwright raises TargetClosedError.
        # Save a checkpoint without dumping an alarming traceback in that case.
        if ('Target page, context or browser has been closed' in str(exc) or
                'TargetClosedError' in type(exc).__name__):
            print('Chromium was closed; stopping and saving latest checkpoint.')
        else:
            raise
    finally:
        if model is not None:
            model.save(output / "latest")
            print("Saved PPO checkpoint:", output / "latest.zip")
        if env is not None:
            env.close()
    print("Done. Inspect", output / "episodes.monitor.csv")


if __name__ == "__main__":
    main()
