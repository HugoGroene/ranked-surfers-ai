#!/usr/bin/env python3
"""Ranked Surfers V2 A/B input timing probe. NO MODEL TRAINING.

Uses the same browser, fullscreen, R restart, death detection and RGB capture
as train_rs_v2_verify.py. Measures Playwright key-tap round trips, capture,
frame-to-input delay, decision intervals and (optionally) YOLO inference.

The Playwright key-tap round-trip includes its intentional 25 ms hold.
This is NOT a measurement of when Unity applies the input/rendered movement.

Examples:
  python -u rs_v2_timing_probe.py --fullscreen --seconds 120 --fps 20 \\
    2>&1 | tee ai_data/v2_timing.log
  python -u rs_v2_timing_probe.py --fullscreen --seconds 120 --fps 20 \\
    --detector ai_data/v2_yolo_dataset/runs/hazards/weights/best.pt
"""
import argparse
import asyncio
import json
from pathlib import Path
import sys
import time
from collections import Counter

import numpy as np

import train_rs_v2_verify as v2
from train_rs_rl_early_death import ACTIONS


class TimedEnv(v2.V2PracticeEnv):
    def __init__(self, args, output, demos):
        self.step_timings = []
        self.last_capture_end = None
        self.last_press_start = None
        super().__init__(args, output, demos)

    async def _capture(self):
        result = await super()._capture()
        self.last_capture_end = time.perf_counter()
        return result

    async def _step_browser(self, action):
        # Replicate precisely the current action system while timing its parts.
        began = time.perf_counter()
        previous_capture = self.last_capture_end
        if self.suspected_since is None and ACTIONS[action] is not None:
            press_started = time.perf_counter()
            await self.page.keyboard.press(ACTIONS[action], delay=self.probe_press_delay_ms)
            press_ended = time.perf_counter()
            pressed = True
        else:
            press_started = began
            press_ended = began
            pressed = False
        after_press = press_ended
        if self.probe_wait_mode == 'legacy':
            pause = max(0.0, 1.0 / self.args.fps - 0.03)
        elif self.probe_wait_mode == 'paced':
            # If the keypress already used the allotted time, do not add delay.
            pause = max(0.0, 1.0 / self.args.fps - (time.perf_counter() - began))
        else:
            pause = 0.0
        if pause:
            await asyncio.sleep(pause)
        after_sleep = time.perf_counter()
        data = await self._capture()
        ended = time.perf_counter()
        self.step_timings.append({
            'press': bool(pressed),
            'action': int(action),
            'press_roundtrip_ms': (press_ended - press_started) * 1000 if pressed else None,
            'scheduled_wait_ms': (after_sleep - after_press) * 1000,
            'capture_ms': (ended - after_sleep) * 1000,
            'async_step_ms': (ended - began) * 1000,
            # Approximate frame age when Playwright starts receiving command.
            'frame_age_at_command_ms': (press_started - previous_capture) * 1000 if (pressed and previous_capture is not None) else None,
            'press_timestamp': press_started if pressed else None,
        })
        return data


def summarize(values):
    arr = np.asarray([float(x) for x in values if x is not None], dtype=float)
    if arr.size == 0:
        return {'count': 0}
    return {'count': int(arr.size), 'mean_ms': round(float(np.mean(arr)), 2),
            'p50_ms': round(float(np.percentile(arr, 50)), 2),
            'p95_ms': round(float(np.percentile(arr, 95)), 2),
            'max_ms': round(float(np.max(arr)), 2)}


def parse_args():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--seconds', type=float, default=120)
    p.add_argument('--fps', type=int, default=20)
    p.add_argument('--fullscreen', action='store_true')
    p.add_argument('--run-name', default='v2_timing_probe')
    p.add_argument('--detector', type=Path, default=None,
                   help='Optional trained YOLO best.pt; benchmark inference before each input.')
    p.add_argument('--detector-imgsz', type=int, default=640)
    p.add_argument('--detect-every', type=int, default=1,
                   help='Run YOLO once every N observations (the cached result is not used to choose actions).')
    p.add_argument('--press-delay-ms', type=int, default=25,
                   help='Playwright key hold duration. Start with 25; test 10 only after checking inputs register.')
    p.add_argument('--wait-mode', choices=('legacy','paced','none'), default='legacy',
                   help='legacy adds the old fixed wait, paced accounts for keypress time, none has no intentional wait.')

    p.add_argument('--policy', choices=('random', 'noop', 'cycle'), default='random')
    p.add_argument('--seed', type=int, default=123)
    a = p.parse_args()
    if not (10 <= a.seconds <= 1200 and 2 <= a.fps <= 30):
        p.error('--seconds must be 10..1200 and --fps 2..30')
    if not (a.detect_every >= 1 and 1 <= a.press_delay_ms <= 100):
        p.error('--detect-every must be >=1, --press-delay-ms must be 1..100')
    if a.detector and not a.detector.is_file():
        p.error(f'No detector checkpoint: {a.detector}')
    if Path(a.run_name).name != a.run_name or a.run_name in ('.', '..'):
        p.error('--run-name must be a simple folder name')
    return a


def main():
    opts = parse_args()
    # Get the *same* environment defaults as the verified v2 training script.
    original_argv = sys.argv[:]
    try:
        sys.argv = [str(Path(v2.__file__)), '--mode', 'smoke', '--fps', str(opts.fps),
                    '--max-minutes', str(opts.seconds / 60 + 1), '--run-name', opts.run_name,
                    '--episode-max-seconds', '15']
        if opts.fullscreen:
            sys.argv.append('--fullscreen')
        defaults = v2.parser_args()
    finally:
        sys.argv = original_argv
    output = v2.RUN_ROOT / opts.run_name
    output.mkdir(parents=True, exist_ok=True)
    demos = v2.DemoStore(v2.DEMO_ROOT, max_samples=100, load_existing=False)
    detector = None
    if opts.detector:
        from ultralytics import YOLO
        detector = YOLO(str(opts.detector))
        print(f'[TIMING] Detector loaded: {opts.detector}', flush=True)
    env = None
    measurements = []
    detector_ms = []
    action_count = Counter()
    completed = 0
    start = time.monotonic()
    rng = np.random.default_rng(opts.seed)
    try:
        env = TimedEnv(defaults, output, demos)
        env.probe_press_delay_ms = opts.press_delay_ms
        env.probe_wait_mode = opts.wait_mode
        step_number = 0
        last_obs = None
        while time.monotonic() - start < opts.seconds:
            reset_began = time.perf_counter()
            obs, _ = env.reset()
            reset_ms = (time.perf_counter()-reset_began)*1000
            completed += 1
            last_obs = obs
            print(f'[TIMING] Attempt {completed} started; reset={reset_ms:.0f}ms', flush=True)
            for _ in range(500):
                if time.monotonic() - start >= opts.seconds:
                    break
                predict_ms = None
                if detector is not None and step_number % opts.detect_every == 0:
                    rgb = v2.vision.latest_rgb(last_obs)
                    before = time.perf_counter()
                    detector.predict(source=rgb, imgsz=opts.detector_imgsz,
                                     verbose=False, conf=.25)
                    predict_ms = (time.perf_counter()-before)*1000
                    detector_ms.append(predict_ms)
                if opts.policy == 'noop':
                    action = 0
                elif opts.policy == 'cycle':
                    action = len(measurements) % 5
                else:
                    action = int(rng.choice(5, p=[.60, .10, .10, .10, .10]))
                before = time.perf_counter()
                last_obs, reward, dead, timeout, info = env.step(action)
                step_number += 1
                call_ms = (time.perf_counter()-before)*1000
                action_count[v2.vision.ACTIONS[action]] += 1
                measurements.append({'env_step_ms': call_ms, 'detector_ms': predict_ms,
                                     'dead': dead, 'timeout': timeout})
                if dead or timeout:
                    break
    except KeyboardInterrupt:
        print('[TIMING] Stopped by user; saving report.', flush=True)
    finally:
        if env is not None:
            env.close()
        elapsed = time.monotonic()-start
        phases = env.step_timings if env is not None else []
        press_times = [p['press_timestamp'] for p in phases if p['press_timestamp'] is not None]
        press_intervals = [(b-a)*1000 for a,b in zip(press_times, press_times[1:])]
        report = {
            'note': ('Timing measures Playwright command roundtrip and screenshot capture, '
                     'not the exact frame when Unity registers the key.'),
            'seconds_elapsed': round(elapsed, 2), 'requested_fps': opts.fps,
            'gameplay_step_rate_including_resets': round(len(measurements)/max(elapsed, .001), 2),
            'steps': len(measurements), 'attempts': completed,
            'action_counts': dict(action_count),
            'press_roundtrip': summarize(p['press_roundtrip_ms'] for p in phases),
            'frame_age_at_command': summarize(p['frame_age_at_command_ms'] for p in phases),
            'scheduled_wait': summarize(p['scheduled_wait_ms'] for p in phases),
            'capture': summarize(p['capture_ms'] for p in phases),
            'async_step': summarize(p['async_step_ms'] for p in phases),
            'env_step': summarize(m['env_step_ms'] for m in measurements),
            'keypress_intervals': summarize(press_intervals),
            'detector_inference': summarize(detector_ms),
            'policy': opts.policy, 'fullscreen': opts.fullscreen,
            'press_delay_ms': opts.press_delay_ms, 'wait_mode': opts.wait_mode,
            'detector_every': opts.detect_every,
            'detector_calls': len(detector_ms),
        }
        out = output/'timing_report.json'
        out.write_text(json.dumps(report, indent=2))
        print(f'[TIMING] CONFIG: press_delay={opts.press_delay_ms}ms wait={opts.wait_mode} '
              f'detector_every={opts.detect_every} detector_calls={len(detector_ms)}', flush=True)
        print('\n[TIMING] RESULTS:')
        for k in ('gameplay_step_rate_including_resets','frame_age_at_command','press_roundtrip',
                  'scheduled_wait','capture','env_step','keypress_intervals','detector_inference'):
            print(f'  {k}: {report[k]}', flush=True)
        print(f'[TIMING] Report: {out}', flush=True)


if __name__ == '__main__':
    main()
