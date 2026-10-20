#!/usr/bin/env python3
"""Reduce costly full MSS crop re-anchoring between LOCAL Practice runs.

Requires these modules in the same project folder:
  train_rs_color_quick_restart.py
  train_rs_color_fast.py
  train_rs_color_human_named.py
  train_rs_rl_early_death.py
  rs_practice_controller.py

This is a conservative wrapper around the working quick-restart trainer.
Its CNN, observations, action mapping, rewards, PPO settings, demonstration
handling, save formats and death-detection logic are unchanged.

When the browser remains stationary, a full geometry re-anchor on EVERY restart
is unnecessary. This version reuses the existing MSS region on most restarts,
forces a full anchor once per N start attempts, and immediately re-anchors if
Unity's canvas locator has been recreated or the capture region is missing.

The existing periodic --alignment-seconds check remains active, so canvas
scrolling/movement within the browser continues to be detected. However, MOVING
THE CHROME WINDOW (not merely scrolling the web page) may require a fresh full
anchor. Keep Chromium maximized and stationary during training.

Usage example:
  python -u train_rs_color_anchor_cache.py \\
    --run-name model_anchor --resume ai_data/rl_color_runs/model_quick/latest.zip \\
    --max-minutes 5 --fps 20 --alignment-seconds 10 \\
    --full-anchor-every 8 --bc-updates 1 --save-every-minutes 2

A/B baseline: --full-anchor-every 1 (full anchor on every start attempt).
"""
import argparse
import sys
import time

import train_rs_color_quick_restart as quick


class CachedAnchorEnv(quick.QuickRestartEnv):
    def __init__(self, args, output_dir, demos):
        self._anchor_start_attempts = 0
        self._anchor_full_count = 0
        self._anchor_skipped_count = 0
        self._anchor_forced = True
        super().__init__(args, output_dir, demos)

    async def _ensure_canvas(self):
        old_canvas = self.canvas
        await super()._ensure_canvas()
        if self.canvas is not old_canvas:
            self._anchor_forced = True
            print('[ANCHOR] Canvas locator changed: forcing full geometry refresh.', flush=True)

    async def _press_practice(self):
        # The original cached locator succeeded on nearly all of your recent
        # attempts. If this needed rediscovery, refresh geometry defensively.
        cached = self._cached_practice
        await super()._press_practice()
        if cached is None or self._cached_practice is not cached:
            self._anchor_forced = True

    async def _start_episode(self):
        self._anchor_start_attempts += 1
        return await super()._start_episode()

    async def _anchor_capture_region(self):
        interval = self.args.full_anchor_every
        full = (
            self._anchor_forced
            or self.region is None
            or self._capture_anchor_box is None
            or self._capture_anchor_region is None
            or interval == 1
            or (self._anchor_start_attempts > 0 and
                self._anchor_start_attempts % interval == 0)
        )
        if full:
            await super()._anchor_capture_region()
            self._anchor_forced = False
            self._anchor_full_count += 1
        else:
            # Keep both the existing anchor and the current screenshot region.
            # _capture() still invokes the periodic movement check, and this
            # method is followed by the SAME game-ready verification as before.
            self._anchor_skipped_count += 1

    def _report_if_due(self):
        if time.monotonic() - self._perf_last >= self.args.profile_seconds:
            print(f'[ANCHOR] full={self._anchor_full_count} '
                  f'skipped={self._anchor_skipped_count} '
                  f'interval={self.args.full_anchor_every} start attempts', flush=True)
        return super()._report_if_due()


def main():
    parser = argparse.ArgumentParser(add_help=False)
    parser.add_argument('--full-anchor-every', type=int, default=8)
    flags, remaining = parser.parse_known_args()
    if flags.full_anchor_every < 1:
        parser.error('--full-anchor-every must be >= 1')
    sys.argv = [sys.argv[0]] + remaining
    original_builder = quick.fast.original.build_args

    def build_args_cached():
        parser, args = original_builder()
        args.full_anchor_every = flags.full_anchor_every
        return parser, args

    quick.fast.original.build_args = build_args_cached
    quick.QuickRestartEnv = CachedAnchorEnv
    print(f'[ANCHOR] Full canvas re-anchor every {flags.full_anchor_every} '
          'start attempts, and when locator changes; normal alignment checks stay on.',
          flush=True)
    quick.main()


if __name__ == '__main__':
    main()
