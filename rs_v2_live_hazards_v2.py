#!/usr/bin/env python3
"""Live LOCAL Practice: non-blocking YOLO -> timestamped hazard vector + overlays.

Reuses rs_v2_timing_probe_ab.py for its verified fullscreen/R restart and
20 ms/no-wait Playwright control. This is an integration/audit experiment,
NOT an autonomous avoidance policy or a PPO training run.

Usage:
  python -u rs_v2_live_hazards.py --fullscreen --seconds 90 \
    --detector ai_data/v2_yolo_dataset/runs/hazards/weights/best.pt \
    2>&1 | tee ai_data/live_hazards.log
"""
import argparse
from collections import Counter
from concurrent.futures import ThreadPoolExecutor
from datetime import datetime
import json
from pathlib import Path
import sys
import time

import cv2
import numpy as np

CLASSES = ('coin', 'train', 'barrier')
PER_CLASS = 4
FEATURES_PER_BOX = 6  # cx, cy, width, height, conf, age_seconds
FEATURE_COUNT = 3 * PER_CLASS * FEATURES_PER_BOX + 1  # + fresh_flag


def detect(model, frame_rgb, captured_at, episode, request_id, imgsz, conf):
    """Run YOLO in its only inference thread; results refer to THIS exact frame."""
    began = time.perf_counter()
    result = model.predict(source=frame_rgb, imgsz=imgsz, conf=conf,
                           verbose=False)[0]
    finished = time.perf_counter()
    h, w = frame_rgb.shape[:2]
    detections = []
    if result.boxes is not None and len(result.boxes):
        xyxy = result.boxes.xyxy.cpu().numpy()
        scores = result.boxes.conf.cpu().numpy()
        classids = result.boxes.cls.cpu().numpy()
        for xy, score, class_id in zip(xyxy, scores, classids):
            name = result.names.get(int(class_id), str(int(class_id)))
            if name not in CLASSES:
                continue
            x1, y1, x2, y2 = [float(n) for n in xy]
            detections.append({
                'class': name,
                'conf': round(float(score), 4),
                'box': [round(x1 / w, 5), round(y1 / h, 5),
                        round(x2 / w, 5), round(y2 / h, 5)]
            })
    return {
        'episode': episode, 'request_id': request_id,
        'captured_at': captured_at, 'finished_at': finished,
        'inference_ms': round((finished - began) * 1000, 2),
        'detections': detections, 'source_frame': frame_rgb,
    }


def hazard_features(snapshot, now=None, max_age_s=0.6):
    """Fixed-shape float32 feature vector. No stale/previous-episode boxes.

    Zero-padded slots; final fresh flag 1 only for a recent result.
    Crucially this is NOT a lane or collision-risk oracle.
    """
    if now is None:
        now = time.perf_counter()
    vector = np.zeros(FEATURE_COUNT, dtype=np.float32)
    if snapshot is None:
        return vector, None, False
    age = max(0.0, now - snapshot['captured_at'])
    if age > max_age_s:
        return vector, age * 1000.0, False
    for idx, name in enumerate(CLASSES):
        objects = [d for d in snapshot['detections'] if d['class'] == name]
        # Favor nearer/larger objects; visual proximity is not exact time-to-impact.
        objects.sort(key=lambda o: (o['box'][3], o['conf']), reverse=True)
        for slot, item in enumerate(objects[:PER_CLASS]):
            x1, y1, x2, y2 = item['box']
            start = (idx * PER_CLASS + slot) * FEATURES_PER_BOX
            vector[start:start + FEATURES_PER_BOX] = [
                (x1 + x2) * .5, (y1 + y2) * .5,
                x2 - x1, y2 - y1, item['conf'], age]
    vector[-1] = 1.0
    return vector, age * 1000.0, True


def overlay(snapshot, dest):
    rgb = snapshot['source_frame']
    bgr = cv2.cvtColor(rgb, cv2.COLOR_RGB2BGR)
    h, w = bgr.shape[:2]
    colors = {'coin': (0, 240, 255), 'train': (230, 150, 20),
              'barrier': (55, 60, 245)}
    for obj in snapshot['detections']:
        x1, y1, x2, y2 = obj['box']
        x1, x2 = round(x1 * w), round(x2 * w)
        y1, y2 = round(y1 * h), round(y2 * h)
        color = colors[obj['class']]
        cv2.rectangle(bgr, (x1, y1), (x2, y2), color, 1)
        cv2.putText(bgr, f"{obj['class']} {obj['conf']:.2f}",
                    (x1, max(8, y1 - 2)), cv2.FONT_HERSHEY_SIMPLEX,
                    .31, color, 1, cv2.LINE_AA)
    image = cv2.resize(bgr, (w * 3, h * 3), interpolation=cv2.INTER_NEAREST)
    cv2.imwrite(str(dest), image)


def stats(values):
    if not values:
        return {'count': 0}
    a = np.asarray(values, dtype=np.float64)
    return {'count': len(values), 'p50_ms': round(float(np.median(a)), 2),
            'p95_ms': round(float(np.percentile(a, 95)), 2),
            'mean_ms': round(float(np.mean(a)), 2),
            'max_ms': round(float(np.max(a)), 2)}


def options():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--detector', type=Path, required=True)
    p.add_argument('--seconds', type=float, default=90)
    p.add_argument('--fps', type=int, default=20)
    p.add_argument('--fullscreen', action='store_true')
    p.add_argument('--imgsz', type=int, default=640)
    p.add_argument('--conf', type=float, default=.25)
    p.add_argument('--detect-every', type=int, default=3)
    p.add_argument('--max-stale-ms', type=int, default=600)
    p.add_argument('--press-delay-ms', type=int, default=20)
    p.add_argument('--max-overlays', type=int, default=20)
    p.add_argument('--overlay-delay', type=float, default=0.9,
                   help='Do not save screenshots from the first N seconds of an episode.')
    p.add_argument('--overlay-spacing', type=float, default=3.0,
                   help='Minimum wall-clock seconds between saved overlays.')
    p.add_argument('--overlay-min-objects', type=int, default=1,
                   help='Only save overlays with at least this many detections.')
    p.add_argument('--policy', choices=('random', 'noop'), default='random')
    p.add_argument('--seed', type=int, default=123)
    p.add_argument('--run-name', default='live_hazards_' + datetime.now().strftime('%Y%m%d_%H%M%S'))
    a = p.parse_args()
    if not a.detector.is_file():
        p.error(f'Checkpoint not found: {a.detector}')
    if not (10 <= a.seconds <= 1200 and 2 <= a.fps <= 30 and a.detect_every >= 1):
        p.error('seconds=10..1200, fps=2..30, detect-every>=1')
    if not (1 <= a.press_delay_ms <= 100 and 0 < a.conf < 1 and a.imgsz >= 160):
        p.error('Invalid press-delay-ms/conf/imgsz')
    if a.overlay_delay < 0 or a.overlay_spacing < 0 or a.overlay_min_objects < 0:
        p.error('overlay-delay, overlay-spacing and overlay-min-objects must be nonnegative')
    if Path(a.run_name).name != a.run_name or a.run_name in ('.', '..'):
        p.error('--run-name must be a single folder name')
    return a


def main():
    a = options()
    # Importing the browser environment may load torch/SB3; defer until after args.
    from ultralytics import YOLO
    import train_rs_v2_verify as v2
    from rs_v2_timing_probe_ab import TimedEnv

    argv = sys.argv[:]
    try:
        sys.argv = [str(Path(v2.__file__)), '--mode', 'smoke', '--fps', str(a.fps),
                    '--max-minutes', str(a.seconds / 60 + 1),
                    '--run-name', a.run_name, '--episode-max-seconds', '15']
        if a.fullscreen:
            sys.argv.append('--fullscreen')
        args = v2.parser_args()
    finally:
        sys.argv = argv

    folder = v2.RUN_ROOT / a.run_name
    if (folder / 'live_hazards_report.json').exists():
        raise SystemExit(f'Output already exists: {folder}. Use a new --run-name.')
    folder.mkdir(parents=True, exist_ok=True)
    picdir = folder / 'overlays'
    picdir.mkdir(exist_ok=True)
    rawdir = folder / 'overlay_source_frames'
    rawdir.mkdir(exist_ok=True)

    print('[LIVE] Loading trained detector:', a.detector, flush=True)
    model = YOLO(str(a.detector))
    print('[LIVE] Warmup inference (not included in live timings)...', flush=True)
    detect(model, np.zeros((144, 256, 3), np.uint8), time.perf_counter(), 0, 0,
           a.imgsz, a.conf)
    print('[LIVE] Detector warmed up. Starting local Practice.', flush=True)

    demos = v2.DemoStore(v2.DEMO_ROOT, max_samples=100, load_existing=False)
    pool = ThreadPoolExecutor(max_workers=1, thread_name_prefix='rs-yolo')
    env = None
    running = None
    latest = None
    inferences = []
    ages = []
    valid = 0
    sampled = 0
    skipped_busy = 0
    resets = 0
    completions = 0
    num_pics = 0
    overlay_episodes = set()
    overlay_last_at = -1e20
    overlay_skips = Counter()
    current_episode_started = None
    counts = Counter()
    rng = np.random.default_rng(a.seed)
    started = time.monotonic()

    def receive():
        nonlocal running, latest, num_pics, completions, overlay_last_at
        if running is not None and running.done():
            finished = running.result()  # Raises real model error, don't conceal it.
            running = None
            completions += 1
            inferences.append(finished['inference_ms'])
            if finished['episode'] != resets:
                return  # Discard results from a previous/expired episode.
            latest = finished
            counts.update(x['class'] for x in finished['detections'])
            # Audit screenshots MUST represent real active gameplay, not merely
            # the first N detector completions after the browser starts.
            # Spread previews over the full run and over different episodes.
            age_in_episode = (finished['captured_at'] - current_episode_started
                              if current_episode_started is not None else -1)
            if num_pics >= a.max_overlays:
                overlay_skips['max_overlays_reached'] += 1
            elif age_in_episode < a.overlay_delay:
                overlay_skips['early_episode_transition'] += 1
            elif resets in overlay_episodes:
                overlay_skips['already_saved_this_episode'] += 1
            elif finished['captured_at'] - overlay_last_at < a.overlay_spacing:
                overlay_skips['not_enough_spacing'] += 1
            elif len(finished['detections']) < a.overlay_min_objects:
                overlay_skips['no_detected_hazards'] += 1
            else:
                filename = f"episode{resets:03d}_request{finished['request_id']:05d}.jpg"
                output = picdir / filename
                overlay(finished, output)
                # Save the exact same unannotated source frame, so we can
                # tell a bad capture apart from a bad detection.
                cv2.imwrite(str(rawdir / filename),
                            cv2.cvtColor(finished['source_frame'], cv2.COLOR_RGB2BGR))
                overlay_episodes.add(resets)
                overlay_last_at = finished['captured_at']
                num_pics += 1
                print(f'[OVERLAY] Saved episode={resets}, '
                      f'seconds_into_episode={age_in_episode:.2f}, '
                      f'objects={len(finished["detections"])}: {output.name}',
                      flush=True)

    try:
        env = TimedEnv(args, folder, demos)
        env.probe_press_delay_ms = a.press_delay_ms
        env.probe_wait_mode = 'none'
        with (folder / 'hazard_states.jsonl').open('w') as log:
            while time.monotonic() - started < a.seconds:
                rgb_obs, _ = env.reset()
                resets += 1
                current_episode_started = time.perf_counter()
                latest = None  # Never feed previous-episode detections into a new run.
                print(f'[LIVE] Episode {resets} started', flush=True)
                for step in range(500):
                    if time.monotonic() - started >= a.seconds:
                        break
                    receive()
                    # Always use the exact image/timestamp submitted to YOLO.
                    if step % a.detect_every == 0:
                        if running is None:
                            frame = v2.vision.latest_rgb(rgb_obs)
                            cap_at = env.last_capture_end or time.perf_counter()
                            running = pool.submit(detect, model, frame, cap_at,
                                                  resets, sampled, a.imgsz, a.conf)
                            sampled += 1
                        else:
                            skipped_busy += 1  # One-job queue: no stale backlog.
                    features, age_ms, fresh = hazard_features(
                        latest, max_age_s=a.max_stale_ms / 1000)
                    if fresh:
                        valid += 1
                        ages.append(age_ms)
                    if a.policy == 'noop':
                        action = 0
                    else:
                        action = int(rng.choice(5, p=[.60, .10, .10, .10, .10]))
                    log.write(json.dumps({
                        'episode': resets, 'step': step,
                        'action': v2.vision.ACTIONS[action],
                        'fresh': fresh,
                        'hazard_age_ms': round(age_ms, 2) if age_ms is not None else None,
                        'feature_count': int(features.size),
                        'features': features.tolist(),
                        'objects': latest['detections'] if fresh else [],
                    }) + '\n')
                    rgb_obs, _, dead, timeout, _ = env.step(action)
                    if (step + 1) % 40 == 0:
                        print(f'[LIVE] steps={step + 1} fresh={fresh} '
                              f'age_ms={round(age_ms) if age_ms is not None else "none"} '
                              f'boxes={len(latest["detections"]) if fresh else 0}',
                              flush=True)
                    if dead or timeout:
                        break
            receive()
    except KeyboardInterrupt:
        print('[LIVE] Stopped; saving reports.', flush=True)
    finally:
        if env is not None:
            env.close()
        pool.shutdown(wait=True, cancel_futures=True)
        elapsed = time.monotonic() - started
        report = {
            'seconds': round(elapsed, 2), 'episodes_started': resets,
            'steps': sum(1 for _ in (folder / 'hazard_states.jsonl').open()) if (folder / 'hazard_states.jsonl').exists() else 0,
            'steps_per_sec_including_resets': None,
            'detector_requests': sampled, 'detector_completed_before_shutdown': completions,
            'detector_skipped_busy': skipped_busy,
            'fresh_state_steps': valid,
            'fresh_state_fraction': None,
            'detector_inference_ms': stats(inferences),
            'age_of_fresh_hazards_at_decision_ms': stats(ages),
            'objects_in_completed_current_episode_results': dict(counts),
            'overlays': str(picdir),
            'overlay_source_frames': str(rawdir),
            'overlays_saved': num_pics,
            'overlay_skip_reasons': dict(overlay_skips),
            'log': str(folder / 'hazard_states.jsonl'),
            'feature_count': FEATURE_COUNT,
            'note': ('Boxes refer to a previous timestamped frame. Not yet motion-compensated. '
                     'This is a live perception integration test with random/noop input, '
                     'NOT an avoidance policy; controller speed includes game resets.'),
        }
        report['steps_per_sec_including_resets'] = round(report['steps'] / max(elapsed, .01), 2)
        report['fresh_state_fraction'] = round(valid / max(report['steps'], 1), 3)
        outfile = folder / 'live_hazards_report.json'
        outfile.write_text(json.dumps(report, indent=2))
        print('\n[LIVE] RESULTS:', flush=True)
        for key in ('steps_per_sec_including_resets','fresh_state_fraction',
                    'detector_requests','detector_completed_before_shutdown',
                    'detector_skipped_busy','detector_inference_ms',
                    'age_of_fresh_hazards_at_decision_ms',
                    'objects_in_completed_current_episode_results',
                    'overlays_saved', 'overlay_skip_reasons'):
            print(f'  {key}: {report[key]}', flush=True)
        print('[LIVE] Overlays:', picdir, flush=True)
        print('[LIVE] Matching unannotated frames:', rawdir, flush=True)
        print('[LIVE] Report:', outfile, flush=True)


if __name__ == '__main__':
    main()
