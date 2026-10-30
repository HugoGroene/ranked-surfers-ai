#!/usr/bin/env python3
"""Observation-only pause/resume probe for LOCAL Ranked Surfers Practice.

Run as a module from the separate Claude development clone:
  python -B -m tools.probe_unity_pause --state-model ... --detector ...

No PPO learning, no movement actions, no checkpoints loaded or saved. Opens a
normal V6.1.2 local Practice environment, waits for a verified gameplay
reset, clicks the site's own Pause/Resume controls, and checks the real MSS
captured track for changes during the pause vs before and after.

This does NOT install pause-during-training; its purpose is to establish
whether the game's pause UI truly freezes the Unity scene on this setup.
"""
from __future__ import annotations

import argparse
import asyncio
from datetime import datetime
import json
from pathlib import Path
import sys
import time

import numpy as np


def frame_motion(before, after):
    """Raw visual motion diagnostics; do not treat low motion as death proof."""
    a, b = np.asarray(before), np.asarray(after)
    if a.shape != b.shape or a.ndim != 2 or a.size == 0:
        raise ValueError(f'Incompatible track frames: {a.shape} vs {b.shape}')
    delta = np.abs(a.astype(np.int16) - b.astype(np.int16))
    return {
        'mean_absolute_pixel_change': round(float(delta.mean()), 4),
        'pixel_fraction_changed_gt_8': round(float(np.mean(delta > 8)), 5),
        'max_pixel_change': int(delta.max()),
    }


def pause_freeze_support(baseline, paused):
    """Conservative heuristic; corroboration still requires live game observation."""
    b = float(baseline['mean_absolute_pixel_change'])
    p = float(paused['mean_absolute_pixel_change'])
    # A static scene before the pause cannot prove the pause worked.
    return b >= 1.0 and p <= 0.5 and p < b * 0.15


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--state-model', type=Path, required=True)
    parser.add_argument('--detector', type=Path, required=True)
    parser.add_argument('--hold-seconds', type=float, default=1.2)
    opts = parser.parse_args()
    if not 0.9 <= opts.hold_seconds <= 3.0:
        parser.error('--hold-seconds must be between 0.9 and 3.0')
    if not opts.state_model.is_file() or not opts.detector.is_file():
        parser.error('Expected both local screen-state and YOLO model files')

    # The source trainer provides exactly the browser, fullscreen, capture,
    # startup and screenshot mechanics used in actual PPO runs. Use eval+none
    # ONLY to validate configuration; do not call evaluate() or model.predict().
    import train_rs_v612_ppo as trainer
    old_argv = sys.argv
    try:
        sys.argv = ['train_rs_v612_ppo.py', '--mode', 'eval', '--eval-policy', 'none',
                    '--max-minutes', '2', '--save-overlays', '0',
                    '--fps', '30', '--rate-mode', 'uncapped',
                    '--press-delay-ms', '5', '--imgsz', '960', '--detect-every', '3',
                    '--max-stale-ms', '300',
                    '--state-model', str(opts.state_model),
                    '--detector', str(opts.detector)]
        a = trainer.parse_args()
    finally:
        sys.argv = old_argv

    from train_rs_v2_verify import RUN_ROOT
    output = RUN_ROOT / ('v612_pause_probe_' + datetime.now().strftime('%Y%m%d_%H%M%S'))
    output.mkdir(parents=True, exist_ok=False)
    report_path = output / 'pause_probe_report.json'
    report = {'status': 'NOT_RUN', 'game_url': 'http://localhost:8000',
              'local_only': True, 'hold_seconds': opts.hold_seconds,
              'output_dir': str(output), 'no_policy_training': True,
              'controls': 'site Pause game/Resume game buttons; no Escape injected'}
    raw = monitored = None

    try:
        print('[PPO PAUSE PROBE] Opening local Practice; no PPO learning/actions.', flush=True)
        monitored, raw = trainer.create_live_env(a, output)
        obs, info = monitored.reset()
        report['reset_ok'] = set(obs) == {'image', 'aux'}

        # All Playwright/MSS operations occur on the environment's OWN worker
        # event loop. Do not access the browser/grabber from this thread.
        async def inspect_pause():
            import cv2
            page = raw.page
            overlay = page.locator('.game-pause-panel[aria-label="Game paused"]')
            pause_button = page.locator('button[aria-label="Pause game"]')
            # The page-level Resume game button is behind the active pause layer.
            # Click the visible Resume action *inside* that layer instead;
            # never use force=True to bypass Playwright's pointer safety checks.
            resume_button = overlay.get_by_role('button', name='Resume', exact=True)
            still_paused = False

            async def capture(name):
                await raw._capture()
                if raw._v2_last_bgra is None:
                    raise RuntimeError('Missing live BGRA capture')
                shot = raw._v2_last_bgra.copy()
                cv2.imwrite(str(output / (name + '.jpg')),
                            cv2.cvtColor(shot, cv2.COLOR_BGRA2BGR),
                            [cv2.IMWRITE_JPEG_QUALITY, 90])
                return raw._motion_view(shot, raw.crop)

            try:
                focused = await page.evaluate(
                    'document.hasFocus() && document.visibilityState === "visible"')
                if not focused:
                    raise RuntimeError('Game window is not focused/visible')
                await raw._capture()
                state, probs = raw._read_screen_state()
                if state != 'GAMEPLAY':
                    raise RuntimeError(f'Gameplay not confirmed before pause: {state}; {probs}')
                if await overlay.is_visible():
                    raise RuntimeError('Game already paused before probe; refusing toggle')
                if await pause_button.count() != 1 or not await pause_button.is_visible():
                    raise RuntimeError('Cannot locate ONE visible website Pause game button')

                baseline_a = await capture('before_pause_a')
                await asyncio.sleep(.18)
                baseline_b = await capture('before_pause_b')
                baseline = frame_motion(baseline_a, baseline_b)
                print('[PPO PAUSE PROBE] Unpaused scene change:', baseline, flush=True)

                await pause_button.click(timeout=1500)
                await overlay.wait_for(state='visible', timeout=2500)
                still_paused = True
                await asyncio.sleep(.30)  # UI/pause animation must settle
                paused_a = await capture('during_pause_a')
                await asyncio.sleep(opts.hold_seconds)
                paused_b = await capture('during_pause_b')
                paused = frame_motion(paused_a, paused_b)
                print('[PPO PAUSE PROBE] Paused scene change:', paused, flush=True)

                if await resume_button.count() != 1 or not await resume_button.is_visible():
                    raise RuntimeError('Cannot locate ONE visible website Resume game button')
                await resume_button.click(timeout=1500)
                await overlay.wait_for(state='hidden', timeout=2500)
                still_paused = False

                resumed_a = await capture('after_resume_a')
                await asyncio.sleep(.18)
                resumed_b = await capture('after_resume_b')
                resumed = frame_motion(resumed_a, resumed_b)
                print('[PPO PAUSE PROBE] Resumed scene change:', resumed, flush=True)
                return {
                    'pause_overlay_confirmed': True,
                    'resume_overlay_hidden': True,
                    'baseline_scene': baseline,
                    'paused_scene': paused,
                    'resumed_scene': resumed,
                    'freeze_supported_by_pixels': pause_freeze_support(baseline, paused),
                    'resume_motion_seen': resumed['mean_absolute_pixel_change'] >= 1.0,
                    'warning': 'Visual freeze is not by itself proof the Unity gameplay clock stopped; check saved screenshots and gameplay HUD before changing PPO.',
                }
            finally:
                # Never intentionally leave Practice paused. Close the game if
                # a resumed state cannot be confirmed; no movement keys sent.
                if still_paused:
                    try:
                        if await overlay.is_visible() and await resume_button.is_visible():
                            await resume_button.click(timeout=1500)
                            await overlay.wait_for(state='hidden', timeout=2000)
                            print('[PPO PAUSE PROBE] Cleanup: resumed UI.', flush=True)
                    except Exception as cleanup_exc:
                        print(f'[PPO PAUSE PROBE] Cleanup could not unpause: {cleanup_exc}; browser will close.', flush=True)

        report.update(raw._call(inspect_pause(), timeout=40))
        report['status'] = ('PASS_CANDIDATE' if report['freeze_supported_by_pixels']
                            and report['resume_motion_seen'] else 'INCONCLUSIVE')
    except BaseException as exc:
        report['status'] = 'STOPPED_WITHOUT_TRAINING'
        report['error'] = f'{type(exc).__name__}: {exc}'
        print('[PPO PAUSE PROBE] Safe stop:', report['error'], flush=True)
    finally:
        if monitored is not None:
            monitored.close()
        elif raw is not None:
            raw.close()
        report_path.write_text(json.dumps(report, indent=2) + '\n', encoding='utf-8')
        print('[PPO PAUSE PROBE] Report:', report_path, flush=True)
        print('[PPO PAUSE PROBE] Result:', report['status'], flush=True)
        print('[PPO PAUSE PROBE] No PPO model was loaded, trained or saved.', flush=True)
    if report['status'] not in ('PASS_CANDIDATE',):
        raise SystemExit(2)


if __name__ == '__main__':
    main()
