#!/usr/bin/env python3
"""Speed up LOCAL Practice restarts while preserving existing RGB PPO checkpoints.

Requires train_rs_color_fast.py, train_rs_color_human_named.py,
train_rs_rl_early_death.py and rs_practice_controller.py beside this file.

Retains existing death detection, retries after instant deaths, PPO architecture,
RGB observations, rewards, and BC updates. On restart it reuses the Practice
button locator and Unity canvas locator, avoiding repeated DOM searches. If
locators become stale, it discovers fresh ones or uses the original helper.

Use --alignment-seconds 10 if the Chromium window stays in the same place.
Use --legacy-restart to compare against the original restart behavior.
"""
import argparse
import asyncio
import sys
import time

import numpy as np

import rs_practice_controller as ctrl
import train_rs_color_fast as fast
from train_rs_rl_early_death import LocalPracticeEnv


class QuickRestartEnv(fast.FastColorPracticeEnv):
    def __init__(self, args, output_dir, demos):
        self._cached_practice = None
        self._quick_restarts = 0
        self._fallback_restarts = 0
        super().__init__(args, output_dir, demos)

    async def _press_practice(self):
        """Use a cached Playwright Locator, which re-resolves DOM nodes after rerender."""
        cached = self._cached_practice
        if cached is not None:
            try:
                await cached.click(timeout=1100)
                self._quick_restarts += 1
                return
            except Exception as exc:
                self._cached_practice = None
                print(f'[RESTART] Cached Practice locator no longer usable ({str(exc)[:100]}); refreshing', flush=True)

        # Discover the control once (or after a frame/navigation change).
        for frame in self.page.frames:
            try:
                async for target in ctrl.practice_targets(frame):
                    try:
                        await target.click(timeout=2500)
                        self._cached_practice = target
                        self._fallback_restarts += 1
                        print('[RESTART] Practice locator discovered and cached.', flush=True)
                        return
                    except Exception:
                        continue
            except Exception:
                continue

        # Keep the original known-working full discovery and debug fallback.
        self._fallback_restarts += 1
        print('[RESTART] Falling back to original Practice button search.', flush=True)
        await ctrl.click_practice(self.page, timeout=45)

    async def _ensure_canvas(self):
        """Reuse the locator when the same iframe remains alive."""
        if self.canvas is not None:
            try:
                if await self.canvas.is_visible():
                    return
            except Exception:
                pass
        self.canvas = await ctrl.find_canvas(self.page, timeout=25)
        print('[RESTART] Unity canvas locator refreshed.', flush=True)

    async def _start_episode(self):
        if self.args.legacy_restart:
            return await super()._start_episode()
        start = time.perf_counter()
        max_tries = 4
        for attempt in range(1, max_tries + 1):
            # Playwright keyboard.press() sends both keydown and keyup; there
            # are no persistent pressed keys in normal autonomous PPO mode.
            # Keep the defensive release if manual takeover was enabled.
            if self.args.enable_f8:
                t = time.perf_counter()
                for key in ctrl.KEYS:
                    try:
                        await self.page.keyboard.up(key)
                    except Exception:
                        pass
                self._metric('restart_key_release', time.perf_counter() - t)

            t = time.perf_counter()
            await self._press_practice()
            self._metric('restart_click', time.perf_counter() - t)

            t = time.perf_counter()
            await self._ensure_canvas()
            self._metric('restart_canvas', time.perf_counter() - t)

            t = time.perf_counter()
            await ctrl.focus_canvas(self.canvas)
            self._metric('restart_focus', time.perf_counter() - t)

            # Always calculate fresh screen geometry after clicking Practice.
            # This is intentionally kept: stale geometry caused missed deaths.
            t = time.perf_counter()
            await self._anchor_capture_region()
            self._metric('restart_anchor', time.perf_counter() - t)

            t = time.perf_counter()
            clear_streak = 0
            menu_streak = 0
            first_seen = time.monotonic()
            deadline = first_seen + min(self.args.load_timeout, 8.0)
            while time.monotonic() < deadline:
                image, score, _ = await self._capture()
                plausible_gameplay = (
                    score < self.args.threshold * 0.4 and
                    float(np.std(image)) >= 10.0
                )
                clear_streak = clear_streak + 1 if plausible_gameplay else 0
                if clear_streak >= 2:
                    self._metric('restart_game_ready', time.perf_counter() - t)
                    if self.args.enable_f8:
                        await self._install_key_listeners()
                    preview = self._encode_last_frame() if self.episode_number == 1 else None
                    self._metric('browser_start', time.perf_counter() - start)
                    return image, preview

                menu_streak = menu_streak + 1 if score >= self.args.threshold * 0.85 else 0
                if menu_streak >= 5 and time.monotonic() - first_seen >= 1.4:
                    print(f'[RL] Practice returned to menu before gameplay '
                          f'(attempt {attempt}/{max_tries}); retrying', flush=True)
                    break
                await asyncio.sleep(0.055)
            else:
                print(f'[RL] No confirmed gameplay after Practice click '
                      f'(attempt {attempt}/{max_tries}); retrying', flush=True)
            self._metric('restart_game_ready', time.perf_counter() - t)
            await asyncio.sleep(0.2)
        raise RuntimeError(
            'Practice never reached a stable gameplay frame after 4 tries. '
            'Keep Chromium visible and inspect calibration / menu scores.')

    def _report_if_due(self):
        now = time.monotonic()
        if now - self._perf_last >= self.args.profile_seconds:
            detail = []
            for name in ('restart_click', 'restart_canvas', 'restart_focus',
                         'restart_anchor', 'restart_game_ready', 'restart_key_release'):
                calls = self._perf_calls[name]
                if calls:
                    avg = 1000 * self._perf_seconds[name] / calls
                    detail.append(f'{name}={avg:.0f}ms({calls})')
            if detail:
                print('[RESTART] ' + ' '.join(detail) +
                      f' | cached={self._quick_restarts} rediscovered={self._fallback_restarts}', flush=True)
        super()._report_if_due()


def main():
    ap = argparse.ArgumentParser(add_help=False)
    ap.add_argument('--legacy-restart', action='store_true', help='Use the original restart path (for A/B testing)')
    flags, remaining = ap.parse_known_args()
    sys.argv = [sys.argv[0]] + remaining
    old_build = fast.original.build_args

    def build_args_quick():
        parser, args = old_build()
        args.legacy_restart = flags.legacy_restart
        return parser, args

    fast.original.build_args = build_args_quick
    # fast.main() installs fast.FastColorPracticeEnv into the trainer module;
    # substitute our subclass at that exact installation point.
    fast.FastColorPracticeEnv = QuickRestartEnv
    print('[RESTART] Cached Practice/canvas fast path enabled. '
          'Use --legacy-restart for comparison.', flush=True)
    fast.main()


if __name__ == '__main__':
    main()
