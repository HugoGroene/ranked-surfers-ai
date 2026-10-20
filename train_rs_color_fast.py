#!/usr/bin/env python3
"""Faster local RGB PPO training; resumes train_rs_color_human_named.py checkpoints.

Keep this next to train_rs_color_human_named.py, train_rs_rl_early_death.py,
and rs_practice_controller.py. Uses the SAME RGB observations, action space,
CNN, PPO hyperparameters, rewards, and human-demo store as the named trainer.

Improvements:
 - No full-frame JPEG encoding at every game step. Encode only the first
   capture_preview.jpg and on a watchdog timeout.
 - Resize BGRA desktop captures directly into 160x90 RGB, rather than convert
   the whole 1160x644 screenshot to RGB first.
 - Extract and score the small menu-text ROI with OpenCV, avoiding full-size PIL
   conversion; an optional exact detector provides a safe A/B comparison.
 - Skip Chromium's F8/human keyboard polling on every PPO step by default.
   Continue to collect demonstrations with the ORIGINAL script separately.
 - Check scrolling/canvas alignment periodically, and at EVERY episode reset.
 - Print wall-time profiling periodically (frame grab, ROI, resizing, action,
   episode reset) for realistic throughput measurements.

First test:
  python train_rs_color_fast.py --run-name model_fast \\
    --resume ai_data/rl_color_runs/model/latest.zip --max-minutes 5 \\
    --fps 20 --demo-max 5000 --bc-updates 1 --save-every-minutes 2

Optional flags:
  --legacy-detector       Use the original exact PIL menu score for comparison.
  --enable-f8             Re-enable the original browser F8/human polling.
  --alignment-seconds 0.5 Recheck canvas position at least twice/second.
  --profile-seconds 30    Print profiler measurements every 30 seconds.

Only local practice (localhost). Chromium must stay visible and unobstructed.
"""
import argparse
from collections import defaultdict
from pathlib import Path
import sys
import time

import cv2
import numpy as np
from PIL import Image

import train_rs_color_human_named as original
ORIGINAL_COLOR_ENV = original.ColorPracticeEnv
from train_rs_rl_early_death import LocalPracticeEnv, IMAGE_W, IMAGE_H


class FastColorPracticeEnv(original.ColorPracticeEnv):
    """Drop-in replacement; same observation/action/reward semantics."""

    def __init__(self, args, output_dir, demos):
        self._perf_seconds = defaultdict(float)
        self._perf_calls = defaultdict(int)
        self._perf_last = time.monotonic()
        self._perf_steps = 0
        self._perf_resets = 0
        self._last_bgra = None
        self._fast_alignment_last = float('-inf')
        self._fast_detector_refs = None
        super().__init__(args, output_dir, demos)

    def _metric(self, name, elapsed):
        self._perf_seconds[name] += elapsed
        self._perf_calls[name] += 1

    async def _open_browser(self):
        if self.args.enable_f8:
            await super()._open_browser()
        else:
            # No keyboard listeners for normal PPO (manual demonstrations are
            # collected with train_rs_color_human_named.py --collect-only).
            await LocalPracticeEnv._open_browser(self)
            print('[FAST] F8 polling disabled; collect demos separately.', flush=True)

    async def _start_episode(self):
        t0 = time.perf_counter()
        # Keep the fully tested instant-death retry logic from the parent.
        frame, _ = await LocalPracticeEnv._start_episode(self)
        if self.args.enable_f8:
            await self._install_key_listeners()
        # The parent reset saves a full-size preview ONLY for episode one.
        preview = self._encode_last_frame() if self.episode_number == 1 else None
        self._metric('browser_start', time.perf_counter() - t0)
        return frame, preview

    def _encode_last_frame(self):
        if self._last_bgra is None:
            raise RuntimeError('No desktop capture available for screenshot')
        bgr = cv2.cvtColor(self._last_bgra, cv2.COLOR_BGRA2BGR)
        success, data = cv2.imencode('.jpg', bgr, [cv2.IMWRITE_JPEG_QUALITY, 85])
        if not success:
            raise RuntimeError('Could not encode diagnostic screenshot')
        return data.tobytes()

    async def _sync_capture_region(self):
        now = time.monotonic()
        if now - self._fast_alignment_last < self.args.alignment_seconds:
            return
        self._fast_alignment_last = now
        t0 = time.perf_counter()
        await LocalPracticeEnv._sync_capture_region(self)
        self._metric('alignment', time.perf_counter() - t0)

    async def _anchor_capture_region(self):
        await LocalPracticeEnv._anchor_capture_region(self)
        self._fast_alignment_last = float('-inf')

    def _fast_menu_score(self, shot):
        if self._fast_detector_refs is None:
            kernel = self.menu_detector.kernel
            self._fast_detector_refs = [
                (ref, dilated, cv2.countNonZero(ref))
                for ref, dilated in self.menu_detector.refs
            ]
            self._fast_kernel = kernel
        h, w = shot.shape[:2]
        x, y, rw, rh = self.roi
        x0, y0 = round(x*w), round(y*h)
        x1, y1 = round((x+rw)*w), round((y+rh)*h)
        if x1 <= x0 or y1 <= y0:
            return 0.0
        region = shot[y0:y1, x0:x1, :3]   # BGRA -> BGR channels only
        shrunk = cv2.resize(region, (256, 160), interpolation=cv2.INTER_LINEAR)
        # Equivalent bright/cream color selection to MenuTextDetector._foreground.
        b = shrunk[:, :, 0].astype(np.int16)
        g = shrunk[:, :, 1].astype(np.int16)
        r = shrunk[:, :, 2].astype(np.int16)
        mask = (r >= 190) & (g >= 180) & (b >= 110) & (r-g < 80) & (r >= b)
        sample = cv2.medianBlur(mask.astype(np.uint8) * 255, 3)
        count = cv2.countNonZero(sample)
        if count < 100:
            return 0.0
        sample_dilated = cv2.dilate(sample, self._fast_kernel)
        best = 0.0
        for ref, ref_dilated, ref_count in self._fast_detector_refs:
            recall = cv2.countNonZero(cv2.bitwise_and(ref, sample_dilated)) / max(1, ref_count)
            precision = cv2.countNonZero(cv2.bitwise_and(sample, ref_dilated)) / max(1, count)
            score = 2 * precision * recall / max(1e-8, precision + recall)
            best = max(best, score)
        return float(best)

    async def _capture(self):
        await self._sync_capture_region()
        t0 = time.perf_counter()
        shot = np.asarray(self.grabber.grab(self.region))
        self._metric('mss_grab', time.perf_counter() - t0)
        self._last_bgra = shot

        t0 = time.perf_counter()
        if self.args.legacy_detector:
            rgb = cv2.cvtColor(shot, cv2.COLOR_BGRA2RGB)
            score = self.menu_detector.score(Image.fromarray(rgb))
        else:
            score = self._fast_menu_score(shot)
        self._metric('menu_detect', time.perf_counter() - t0)

        t0 = time.perf_counter()
        small_bgra = cv2.resize(shot, (IMAGE_W, IMAGE_H), interpolation=cv2.INTER_AREA)
        small_rgb = cv2.cvtColor(small_bgra, cv2.COLOR_BGRA2RGB)
        chw = np.ascontiguousarray(small_rgb.transpose(2, 0, 1))
        self._metric('image_resize', time.perf_counter() - t0)
        # Return raw=None; full-screen JPEG is unnecessary for PPO.
        return chw, float(score), None

    def reset(self, *, seed=None, options=None):
        t0 = time.perf_counter()
        observation, info = ORIGINAL_COLOR_ENV.reset(self, seed=seed, options=options)
        self._metric('reset_total', time.perf_counter() - t0)
        self._perf_resets += 1
        # The preview is already saved by parent.reset(). Don't retain it as
        # the watchdog frame; watchdogs are encoded on demand in step().
        self.last_raw = None
        return observation, info

    def step(self, action):
        t0 = time.perf_counter()
        if self.args.enable_f8:
            result = ORIGINAL_COLOR_ENV.step(self, action)
        else:
            # Skip per-step frame.evaluate(DRAIN_JS) calls (normally 2+ browser
            # round trips) while preserving the parent's robust death handling.
            stacked, reward, terminated, truncated, info = LocalPracticeEnv.step(self, action)
            obs = ORIGINAL_COLOR_ENV._rgb_obs(stacked)
            death_penalty = 1.0 if terminated else 0.0
            reward = (reward + death_penalty) * (20.0 / self.args.fps) - death_penalty
            result = (obs, float(reward), terminated, truncated, info)
        if result[3]:
            # Parent cannot save JPEG on watchdog (self.last_raw=None), so
            # produce its diagnostic from the actual MOST RECENT screenshot.
            filename = self.output_dir / f'watchdog_episode_{self.episode_number:04d}.jpg'
            filename.write_bytes(self._encode_last_frame())
            print(f'[FAST] Watchdog capture saved: {filename}', flush=True)
        self._metric('env_step', time.perf_counter() - t0)
        self._perf_steps += 1
        self._report_if_due()
        return result

    def _report_if_due(self):
        now = time.monotonic()
        elapsed = now - self._perf_last
        if elapsed < self.args.profile_seconds:
            return
        chunks = []
        for name in ('mss_grab','menu_detect','image_resize','alignment',
                     'env_step','reset_total','browser_start'):
            calls = self._perf_calls[name]
            if calls:
                ms = self._perf_seconds[name] / calls * 1000
                chunks.append(f'{name}={ms:.1f}ms({calls})')
        print(f'[FAST] {self._perf_steps / elapsed:.1f} steps/s wall-clock; '
              f'{self._perf_steps} decisions, {self._perf_resets} resets / {elapsed:.0f}s | '
              + ' '.join(chunks), flush=True)
        self._perf_last = now
        self._perf_seconds.clear()
        self._perf_calls.clear()
        self._perf_steps = 0
        self._perf_resets = 0


def main():
    # Fast-specific arguments are peeled off; all existing trainer args are
    # forwarded, so checkpoint/model/BC settings retain their original behavior.
    fast_parser = argparse.ArgumentParser(add_help=False)
    fast_parser.add_argument('--profile-seconds', type=float, default=30)
    fast_parser.add_argument('--alignment-seconds', type=float, default=0.5)
    fast_parser.add_argument('--legacy-detector', action='store_true')
    fast_parser.add_argument('--enable-f8', action='store_true')
    flags, rest = fast_parser.parse_known_args()
    if flags.profile_seconds < 1 or flags.alignment_seconds < 0.1:
        fast_parser.error('--profile-seconds must be >= 1 and --alignment-seconds >= 0.1')
    sys.argv = [sys.argv[0]] + rest
    build_args_original = original.build_args

    def build_args_fast():
        parser, args = build_args_original()
        args.profile_seconds = flags.profile_seconds
        args.alignment_seconds = flags.alignment_seconds
        args.legacy_detector = flags.legacy_detector
        args.enable_f8 = flags.enable_f8 or args.collect_only
        return parser, args

    original.build_args = build_args_fast
    original.ColorPracticeEnv = FastColorPracticeEnv
    print('[FAST] RGB PPO optimization enabled; model architecture unchanged.', flush=True)
    print('[FAST] Resume from your existing RGB latest.zip; use a NEW run-name '
          'to retain the original checkpoint.', flush=True)
    original.main()


if __name__ == '__main__':
    main()
