#!/usr/bin/env python3
"""Observation-only pause/distance/reward-clock probe for LOCAL Ranked Surfers Practice.

Run as a module from the separate Claude development clone:
  python -B -m tools.probe_unity_pause_clock --state-model ... --detector ...

No PPO learning, no movement actions, no checkpoints loaded or saved. Opens a
normal V6.1.2 local Practice environment, waits for a verified gameplay
reset, clicks the site's own Pause/Resume controls, and checks the real MSS
captured track for changes during the pause vs before and after.

This does NOT install pause-during-training; its purpose is to establish
whether the game's pause UI freezes the track and the visible distance HUD
while the trainer's monotonic survival clock continues advancing. This tool
NEVER mutates or trains the PPO model and never adjusts reward clocks.
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


def meter_activity(before, after):
    """Compare the existing vision pipeline's yellow-distance-HUD masks.

    Returns zero activity when the ROI contains no identifiable distance
    pixels; we explicitly treat that as inconclusive, never as a freeze.
    """
    a, b = np.asarray(before), np.asarray(after)
    if a.shape != b.shape or a.ndim != 2 or not a.size:
        raise ValueError(f'Incompatible meter masks: {a.shape} vs {b.shape}')
    return {
        'yellow_pixels_first': int(np.count_nonzero(a)),
        'yellow_pixels_second': int(np.count_nonzero(b)),
        'changed_pixels': int(np.count_nonzero(a != b)),
    }


def hud_freeze_support(running, paused, resumed, *, minimum_yellow=6):
    """Candidate HUD freeze only with visible digits AND changing distance.

    This is evidence of a static HUD, not direct access to Unity's simulation
    clock. Reject an unreadable/static HUD rather than overclaiming success.
    """
    readings = (running, paused, resumed)
    return (all(min(r['yellow_pixels_first'], r['yellow_pixels_second'])
                >= minimum_yellow for r in readings)
            and running['changed_pixels'] > 0
            and paused['changed_pixels'] == 0
            and resumed['changed_pixels'] > 0)


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
    parser.add_argument('--hold-seconds', type=float, default=1.8)
    opts = parser.parse_args()
    if not 1.2 <= opts.hold_seconds <= 4.0:
        parser.error('--hold-seconds must be between 1.2 and 4.0')
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
    output = RUN_ROOT / ('v612_pause_clock_probe_' + datetime.now().strftime('%Y%m%d_%H%M%S'))
    output.mkdir(parents=True, exist_ok=False)
    report_path = output / 'pause_clock_probe_report.json'
    report = {'status': 'NOT_RUN', 'game_url': 'http://localhost:8000',
              'local_only': True, 'hold_seconds': opts.hold_seconds,
              'output_dir': str(output), 'no_policy_training': True,
              'controls': 'site Pause game/visible-overlay Resume buttons; no Escape injected',
              'read_only': True, 'touches_trainer_code': False}
    raw = monitored = None

    try:
        print('[PPO PAUSE CLOCK] Opening local Practice; no PPO learning/actions.', flush=True)
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
                return raw._motion_view(shot, raw.crop), raw._meter_mask(shot)

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

                baseline_a, meter_before_a = await capture('before_pause_a')
                await asyncio.sleep(.18)
                baseline_b, meter_before_b = await capture('before_pause_b')
                baseline = frame_motion(baseline_a, baseline_b)
                print('[PPO PAUSE CLOCK] Running scene change:', baseline, flush=True)
                meter_running = meter_activity(meter_before_a, meter_before_b)
                print('[PPO PAUSE CLOCK] Running HUD change:', meter_running, flush=True)
                episode_t0 = float(raw.episode_start)
                trainer_duration_before = time.monotonic() - episode_t0

                await pause_button.click(timeout=1500)
                await overlay.wait_for(state='visible', timeout=2500)
                still_paused = True
                pause_started = time.monotonic()
                await asyncio.sleep(.30)  # UI/pause animation must settle
                paused_a, meter_paused_a = await capture('during_pause_a')
                await asyncio.sleep(opts.hold_seconds)
                paused_b, meter_paused_b = await capture('during_pause_b')
                paused = frame_motion(paused_a, paused_b)
                print('[PPO PAUSE CLOCK] Paused scene change:', paused, flush=True)
                meter_stopped = meter_activity(meter_paused_a, meter_paused_b)
                print('[PPO PAUSE CLOCK] Paused HUD change:', meter_stopped, flush=True)

                if await resume_button.count() != 1 or not await resume_button.is_visible():
                    raise RuntimeError('Cannot locate ONE visible website Resume game button')
                await resume_button.click(timeout=1500)
                await overlay.wait_for(state='hidden', timeout=2500)
                still_paused = False
                pause_ended = time.monotonic()
                trainer_duration_after = pause_ended - episode_t0
                wall_pause_seconds = pause_ended - pause_started
                print('[PPO PAUSE CLOCK] Trainer raw episode duration advanced while paused:',
                      {'pause_wall_seconds': round(wall_pause_seconds, 3),
                       'episode_duration_growth_seconds': round(trainer_duration_after - trainer_duration_before, 3)},
                      flush=True)

                resumed_a, meter_resumed_a = await capture('after_resume_a')
                await asyncio.sleep(.18)
                resumed_b, meter_resumed_b = await capture('after_resume_b')
                resumed = frame_motion(resumed_a, resumed_b)
                print('[PPO PAUSE CLOCK] Resumed scene change:', resumed, flush=True)
                meter_resumed = meter_activity(meter_resumed_a, meter_resumed_b)
                print('[PPO PAUSE CLOCK] Resumed HUD change:', meter_resumed, flush=True)
                hud_supported = hud_freeze_support(meter_running, meter_stopped, meter_resumed)
                print('[PPO PAUSE CLOCK] HUD freeze candidate:', hud_supported, flush=True)
                return {
                    'pause_overlay_confirmed': True,
                    'resume_overlay_hidden': True,
                    'baseline_scene': baseline,
                    'hud_running': meter_running,
                    'hud_paused': meter_stopped,
                    'hud_resumed': meter_resumed,
                    'hud_freeze_supported': hud_supported,
                    'pause_wall_seconds': round(wall_pause_seconds, 3),
                    'trainer_duration_growth_while_paused_s': round(trainer_duration_after - trainer_duration_before, 3),
                    'reward_clock_warning': 'episode_start is unchanged during the pause. Unmodified PPO computes survival from wall-clock and would include paused seconds.',
                    'paused_scene': paused,
                    'resumed_scene': resumed,
                    'freeze_supported_by_pixels': pause_freeze_support(baseline, paused),
                    'resume_motion_seen': resumed['mean_absolute_pixel_change'] >= 1.0,
                    'warning': 'Still-image/HUD evidence cannot prove the internal Unity clock was frozen. No PPO step/reward was taken; the raw episode duration was inspected without mutating it.',
                }
            finally:
                # Never intentionally leave Practice paused. Close the game if
                # a resumed state cannot be confirmed; no movement keys sent.
                if still_paused:
                    try:
                        if await overlay.is_visible() and await resume_button.is_visible():
                            await resume_button.click(timeout=1500)
                            await overlay.wait_for(state='hidden', timeout=2000)
                            print('[PPO PAUSE CLOCK] Cleanup: resumed UI.', flush=True)
                    except Exception as cleanup_exc:
                        print(f'[PPO PAUSE CLOCK] Cleanup could not unpause: {cleanup_exc}; browser will close.', flush=True)

        report.update(raw._call(inspect_pause(), timeout=40))
        hud_paused = report['hud_paused']
        hud_visibly_present = min(hud_paused['yellow_pixels_first'],
                                  hud_paused['yellow_pixels_second']) >= 6
        if hud_visibly_present and hud_paused['changed_pixels'] > 0:
            report['status'] = 'REJECT_HUD_MOVED_DURING_PAUSE'
        elif (report['freeze_supported_by_pixels'] and report['resume_motion_seen']
              and report['hud_freeze_supported']):
            report['status'] = 'PASS_HUD_CANDIDATE'
        elif report['freeze_supported_by_pixels'] and report['resume_motion_seen']:
            report['status'] = 'PASS_SCENE_HUD_INCONCLUSIVE'
        else:
            report['status'] = 'INCONCLUSIVE'
    except BaseException as exc:
        report['status'] = 'STOPPED_WITHOUT_TRAINING'
        report['error'] = f'{type(exc).__name__}: {exc}'
        print('[PPO PAUSE CLOCK] Safe stop:', report['error'], flush=True)
    finally:
        if monitored is not None:
            monitored.close()
        elif raw is not None:
            raw.close()
        report_path.write_text(json.dumps(report, indent=2) + '\n', encoding='utf-8')
        print('[PPO PAUSE CLOCK] Report:', report_path, flush=True)
        print('[PPO PAUSE CLOCK] Result:', report['status'], flush=True)
        print('[PPO PAUSE CLOCK] No PPO model was loaded, trained or saved.', flush=True)
    if report['status'] not in ('PASS_HUD_CANDIDATE', 'PASS_SCENE_HUD_INCONCLUSIVE'):
        raise SystemExit(2)


if __name__ == '__main__':
    main()
