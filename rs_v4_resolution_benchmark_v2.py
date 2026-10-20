#!/usr/bin/env python3
"""Ranked Surfers V4: benchmark screenshot sizes and YOLO resolution (gameplay-filtered V2).

LOCAL Practice only. Does not train or overwrite a PPO model.

Run from your existing Windows project folder with venv active.

Step 1 (Unity server running):
 python -u rs_v4_resolution_benchmark.py capture --fullscreen --fps 30 --seconds 75
Step 2 (offline, no game needed):
 python -u rs_v4_resolution_benchmark.py analyze --detector checkpoints/yolo_hazards_best.pt

The CAPTURE stage collects original screen pixels from the existing trusted
V2 environment, not reconstructed/upsampled 256x144 PPO observations.
The ANALYZE stage compares the SAME frames at several input resolutions.
Quality must be judged by reviewing saved overlays: counts are not accuracy.
"""
from __future__ import annotations

import argparse
from collections import Counter, deque
import csv
import json
from pathlib import Path
import sys
import time

import cv2
import numpy as np

DEFAULT_RES = [(256, 144), (384, 216), (512, 288), (768, 432)]


def shape(text):
    try:
        a, b = text.lower().split('x')
        w, h = int(a), int(b)
        if w < 128 or h < 72 or w > 2560 or h > 1440:
            raise ValueError()
        return w, h
    except (ValueError, AttributeError):
        raise argparse.ArgumentTypeError('Use WxH, e.g. 384x216')


def timing(vals):
    if not vals:
        return {'count': 0}
    x = np.asarray(vals, dtype=np.float64)
    return {'count': len(vals), 'p50_ms': round(float(np.median(x)), 2),
            'p95_ms': round(float(np.percentile(x, 95)), 2),
            'mean_ms': round(float(x.mean()), 2)}


def save_json(path, x):
    path.write_text(json.dumps(x, indent=2), encoding='utf-8')


def _signature(bgr):
    """Compact grayscale signature, used only to reject known loading images."""
    return cv2.resize(cv2.cvtColor(bgr, cv2.COLOR_BGR2GRAY), (160, 90),
                      interpolation=cv2.INTER_AREA).astype(np.uint8)


def _loading_references(folder, limit=24):
    """Previous capture set contains screenshots the user identified as loading."""
    if not folder or not folder.is_dir():
        return []
    paths = sorted(folder.glob('*.jpg')) + sorted(folder.glob('*.png'))
    if len(paths) > limit:
        # Spread references across the entire old capture, not just its first seconds.
        idx = np.linspace(0, len(paths) - 1, limit, dtype=int)
        paths = [paths[int(i)] for i in idx]
    refs = []
    for path in paths:
        img = cv2.imread(str(path), cv2.IMREAD_COLOR)
        if img is not None:
            refs.append((path.name, _signature(img)))
    return refs


def _contact_sheet(paths, dest):
    """Immediate visual audit of what the capture test actually accepted."""
    tiles = []
    for path in paths[:30]:
        img = cv2.imread(str(path), cv2.IMREAD_COLOR)
        if img is None:
            continue
        img = cv2.resize(img, (360, 203), interpolation=cv2.INTER_AREA)
        tile = np.full((230, 360, 3), 25, np.uint8)
        tile[27:] = img
        cv2.putText(tile, path.name[:42], (6, 19), cv2.FONT_HERSHEY_SIMPLEX,
                    .39, (235, 235, 235), 1, cv2.LINE_AA)
        tiles.append(tile)
    if tiles:
        rows = []
        for i in range(0, len(tiles), 3):
            row = tiles[i:i+3]
            while len(row) < 3:
                row.append(np.full_like(tiles[0], 25))
            rows.append(np.hstack(row))
        cv2.imwrite(str(dest), np.vstack(rows), [cv2.IMWRITE_JPEG_QUALITY, 88])


def capture(a):
    import train_rs_v2_verify as v2
    from rs_v2_timing_probe_ab import TimedEnv

    dest = a.output / 'native_frames'
    dest.mkdir(parents=True, exist_ok=True)
    preexisting = list(dest.glob('*.jpg')) + list(dest.glob('*.png'))
    if preexisting:
        raise SystemExit(f'{dest} already has {len(preexisting)} screenshots. '
                         'Choose a NEW --output directory to avoid mixing runs.')

    refs = _loading_references(a.loading_reference_dir)
    print(f'[CAPTURE] No arrow inputs (policy={a.policy})' if a.policy == 'noop'
          else '[CAPTURE] Optional random-input policy enabled', flush=True)
    if refs:
        print(f'[CAPTURE] Loaded {len(refs)} known LOADING reference frames from '
              f'{a.loading_reference_dir}; rejecting similar captures.', flush=True)
    else:
        print('[CAPTURE] No loading references found. '
              'Only warmup and motion filters will be used.', flush=True)

    argv = sys.argv[:]
    try:
        sys.argv = [str(Path(v2.__file__)), '--mode', 'smoke',
                    '--fps', str(a.fps), '--run-name', a.run_name,
                    '--max-minutes', str(a.seconds / 60 + 2),
                    '--episode-max-seconds', '15']
        if a.fullscreen:
            sys.argv.append('--fullscreen')
        args = v2.parser_args()
    finally:
        sys.argv = argv

    folder = v2.RUN_ROOT / a.run_name
    folder.mkdir(parents=True, exist_ok=True)
    demos = v2.DemoStore(v2.DEMO_ROOT, max_samples=100, load_existing=False)
    rng = np.random.default_rng(7321)
    deadline = time.monotonic() + a.seconds
    env = None
    saved, episode_no, steps = 0, 0, 0
    last_at = -1e9
    skipped = Counter()
    saved_files = []
    first_logged = False
    try:
        env = TimedEnv(args, folder, demos)
        env.probe_press_delay_ms = a.press_delay_ms
        env.probe_wait_mode = 'none'
        print('[CAPTURE] SITE fullscreen; restarting episodes with R. '
              'Saving only post-loading native screenshots.', flush=True)
        while time.monotonic() < deadline and saved < a.max_frames:
            env.reset()  # R starts Practice. No arrow inputs are necessary.
            episode_no += 1
            episode_start = time.monotonic()
            prior_motion = None
            motion_time = -1e9
            for _ in range(700):
                if time.monotonic() >= deadline or saved >= a.max_frames:
                    break
                action = (0 if a.policy == 'noop' else
                          int(rng.choice(5, p=[.60, .10, .10, .10, .10])))
                _, _, dead, truncated, info = env.step(action)
                steps += 1
                now = time.monotonic()
                if dead or truncated:
                    skipped['episode_ended'] += 1
                    break
                if now - episode_start < a.warmup_seconds:
                    skipped['post_r_warmup'] += 1
                    continue
                if now - last_at < a.spacing_seconds:
                    skipped['spacing'] += 1
                    continue
                bgra = env._v2_last_bgra
                if bgra is None or bgra.ndim != 3 or bgra.shape[2] != 4:
                    skipped['missing_native_frame'] += 1
                    continue
                # The screenshot is native resolution. We only shrink a copy
                # temporarily for fast, conservative filtering.
                bgr = cv2.cvtColor(bgra, cv2.COLOR_BGRA2BGR)
                gray = _signature(bgr)
                menu_score = float(info.get('menu_similarity', 0.0))
                if menu_score >= args.threshold * .70:
                    skipped['menu_screen'] += 1
                    continue

                nearest_loading_diff = None
                if refs:
                    nearest_loading_diff = min(float(cv2.absdiff(gray, r).mean())
                                               for _, r in refs)
                    if nearest_loading_diff < a.loading_diff_threshold:
                        skipped['matched_loading_reference'] += 1
                        continue

                # Screen-motion gate: prevents saving a static loading or frozen
                # frame. It does not prescribe/execute any player action.
                if prior_motion is None or now - motion_time >= .18:
                    if prior_motion is None:
                        prior_motion, motion_time = gray, now
                        skipped['establish_motion_baseline'] += 1
                        continue
                    motion = float(cv2.absdiff(gray, prior_motion).mean())
                    prior_motion, motion_time = gray, now
                    if motion < a.motion_threshold:
                        skipped['low_visual_motion'] += 1
                        continue
                else:
                    skipped['motion_sample_wait'] += 1
                    continue

                filename = dest / f'ep{episode_no:03d}_frame{steps:05d}.jpg'
                if cv2.imwrite(str(filename), bgr,
                               [cv2.IMWRITE_JPEG_QUALITY, 97]):
                    saved += 1
                    saved_files.append(filename)
                    last_at = now
                    if not first_logged or saved % 10 == 0:
                        first_logged = True
                        score_str = (f'{nearest_loading_diff:.1f}' if
                                     nearest_loading_diff is not None else 'n/a')
                        print(f'[CAPTURE] Accepted {saved} frame(s), episode={episode_no}, '
                              f'native={bgr.shape[1]}x{bgr.shape[0]}, '
                              f'loading_diff={score_str}, '
                              f'motion={motion:.1f}, menu_score={menu_score:.2f}', flush=True)
    except KeyboardInterrupt:
        print('[CAPTURE] Interrupted; existing screenshots are preserved.', flush=True)
    finally:
        if env is not None:
            env.close()

    sheet = a.output / 'captured_contact_sheet.jpg'
    _contact_sheet(saved_files, sheet)
    summary = {
        'native_frames': saved, 'episodes_started': episode_no, 'steps': steps,
        'fps_flag': a.fps, 'policy': a.policy,
        'warmup_seconds': a.warmup_seconds, 'loading_refs': len(refs),
        'loading_diff_threshold': a.loading_diff_threshold,
        'motion_threshold': a.motion_threshold,
        'skipped': dict(skipped),
        'screenshots_folder': str(dest), 'contact_sheet': str(sheet)
    }
    save_json(a.output / 'capture_report.json', summary)
    print('[CAPTURE] ' + json.dumps(summary), flush=True)
    if saved == 0:
        print('[CAPTURE] NO ACCEPTED FRAMES: review skip counts; try '
              '--warmup-seconds 0.9 or --motion-threshold 0.5. '
              'Do NOT analyze the old loading screenshots.', flush=True)
    else:
        print('[CAPTURE] FIRST check contact sheet: ' + str(sheet), flush=True)
        print('[CAPTURE] Analyze with the SAME --output folder.', flush=True)


def make_cnn_encoder(device, w, h, batch, iters):
    """Measures the EXACT TinyCNN conv/linear architecture used by V3.

    Not a complete PPO rollout/update, so this is a hardware comparison only.
    """
    import torch
    from torch import nn
    model = nn.Sequential(
        nn.Conv2d(12, 16, 5, stride=2, padding=2), nn.ReLU(),
        nn.Conv2d(16, 32, 3, stride=2, padding=1), nn.ReLU(),
        nn.Conv2d(32, 48, 3, stride=2, padding=1), nn.ReLU(),
        nn.AdaptiveAvgPool2d((3, 6)), nn.Flatten(),
        nn.Linear(48 * 3 * 6, 128), nn.ReLU(),
    ).to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=0.0003)
    # Preallocate input so this isolates network cost, not frame transfer/capture.
    x = torch.rand((batch, 12, h, w), dtype=torch.float32, device=device)

    def sync():
        if device.type == 'cuda':
            torch.cuda.synchronize()

    def forward():
        with torch.inference_mode():
            _ = model(x)

    def update():
        optimizer.zero_grad(set_to_none=True)
        z = model(x)
        loss = z.square().mean()
        loss.backward()
        optimizer.step()

    for _ in range(4):
        forward()
        update()
    sync()
    forward_ms, update_ms = [], []
    for _ in range(iters):
        sync(); t0 = time.perf_counter(); forward(); sync()
        forward_ms.append((time.perf_counter() - t0) * 1000)
        sync(); t0 = time.perf_counter(); update(); sync()
        update_ms.append((time.perf_counter() - t0) * 1000)
    del x, optimizer, model
    if device.type == 'cuda':
        torch.cuda.empty_cache()
    return {'forward': timing(forward_ms), 'encoder_update': timing(update_ms)}


def draw_frame(img_bgr, result, header, box_w=576, box_h=324):
    h, w = img_bgr.shape[:2]
    vis = img_bgr.copy()
    found = Counter()
    for box in result.boxes:
        cid = int(box.cls[0].item())
        name = result.names.get(cid, str(cid))
        score = float(box.conf[0].item())
        found[name] += 1
        xy = box.xyxy[0].cpu().numpy().tolist()
        x1, y1, x2, y2 = [int(v) for v in xy]
        color = {'coin': (0, 210, 255), 'train': (255, 160, 20),
                 'barrier': (30, 40, 250)}.get(name, (150, 255, 150))
        cv2.rectangle(vis, (x1, y1), (x2, y2), color, max(1, round(w / 450)))
        cv2.putText(vis, f'{name}:{score:.2f}', (x1, max(11, y1-4)),
                    cv2.FONT_HERSHEY_SIMPLEX, max(.36, w / 1300), color, 1, cv2.LINE_AA)
    vis = cv2.resize(vis, (box_w, box_h), interpolation=cv2.INTER_LINEAR)
    canvas = np.full((box_h+35, box_w, 3), 25, dtype=np.uint8)
    canvas[35:, :] = vis
    cv2.putText(canvas, header[:85], (8, 23), cv2.FONT_HERSHEY_SIMPLEX,
                .46, (250, 250, 250), 1, cv2.LINE_AA)
    return canvas, found


def analyze(a):
    import torch
    frames_dir = a.output / 'native_frames'
    paths = sorted(frames_dir.glob('*.jpg')) + sorted(frames_dir.glob('*.png'))
    if not paths:
        raise SystemExit(f'No game screenshots in {frames_dir}. Run capture first.')
    paths = paths[:a.max_analyze_frames]
    frames = []
    for path in paths:
        im = cv2.imread(str(path), cv2.IMREAD_COLOR)
        if im is not None:
            frames.append((path, im))
    if not frames:
        raise SystemExit('Screenshots could not be read.')
    native_shape = frames[0][1].shape
    print(f'[ANALYZE] {len(frames)} scenes, native size={native_shape[1]}x{native_shape[0]}', flush=True)

    dest = a.output / 'comparison_overlays'
    dest.mkdir(parents=True, exist_ok=True)
    sizes = a.resolutions
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f'[ANALYZE] Torch: {torch.__version__}; device={device}', flush=True)
    preprocess = {}
    for w, h in sizes:
        duration = []
        for _, src in frames:
            t0 = time.perf_counter()
            small = cv2.resize(src, (w, h), interpolation=cv2.INTER_AREA)
            _ = np.ascontiguousarray(cv2.cvtColor(small, cv2.COLOR_BGR2RGB).transpose(2, 0, 1))
            duration.append((time.perf_counter() - t0)*1000)
        preprocess[f'{w}x{h}'] = timing(duration)
        print(f'[RESIZE] {w}x{h}: median={preprocess[f"{w}x{h}"]["p50_ms"]}ms', flush=True)

    ppo = {}
    for w, h in sizes:
        try:
            bench = make_cnn_encoder(device, w, h, a.batch, a.iters)
            mem = int(128*12*w*h*4)  # SB3 DictRolloutBuffer uses float32 image arrays.
            ppo[f'{w}x{h}'] = {
                'cnn_forward_batch': bench['forward'],
                'cnn_optimizer_step_batch': bench['encoder_update'],
                'batch_size': a.batch,
                'rgb_stack_uint8_kib': round(12*w*h/1024, 1),
                'rollout128_float32_image_mib_estimate': round(mem/1024**2, 1),
            }
            print(f'[PPO ENCODER] {w}x{h}: forward p50={bench["forward"]["p50_ms"]}ms; '
                  f'update p50={bench["encoder_update"]["p50_ms"]}ms; '
                  f'rollout ~{mem/1024**2:.0f}MiB image buffer', flush=True)
        except RuntimeError as exc:
            if 'out of memory' not in str(exc).lower():
                raise
            ppo[f'{w}x{h}'] = {'error': 'CUDA out of memory'}
            if device.type == 'cuda':
                torch.cuda.empty_cache()
            print(f'[PPO ENCODER] {w}x{h}: OUT OF MEMORY', flush=True)

    from ultralytics import YOLO
    if not a.detector.is_file():
        raise SystemExit(f'Detector weights missing: {a.detector}')
    model = YOLO(str(a.detector))
    yolo_device = 0 if device.type == 'cuda' else 'cpu'
    all_rows = []
    yolo_summaries = {}
    for w, h in sizes:
        key = f'{w}x{h}'
        for yolo_size in a.yolo_sizes:
            spec = f'{key}_yolo{yolo_size}'
            # Warm each input shape before measuring latency; GPU preprocessing
            # + inference + box extraction are included in timed region.
            warm_img = cv2.cvtColor(cv2.resize(frames[0][1], (w,h),
                                    interpolation=cv2.INTER_AREA), cv2.COLOR_BGR2RGB)
            for _ in range(2):
                _ = model.predict(source=warm_img, imgsz=yolo_size,
                                  conf=a.conf, device=yolo_device, verbose=False)[0]
            durations = []
            classes = Counter()
            scenes_any = 0
            for i, (path, bgr) in enumerate(frames):
                resized = cv2.resize(bgr, (w, h), interpolation=cv2.INTER_AREA)
                rgb = cv2.cvtColor(resized, cv2.COLOR_BGR2RGB)
                if device.type == 'cuda':
                    torch.cuda.synchronize()
                t0 = time.perf_counter()
                prediction = model.predict(source=rgb, imgsz=yolo_size,
                                           conf=a.conf, device=yolo_device, verbose=False)[0]
                # Explicit CPU box access ensures GPU jobs finished before timing.
                confidences = prediction.boxes.conf.cpu().numpy().tolist()
                class_ids = prediction.boxes.cls.cpu().numpy().astype(int).tolist()
                if device.type == 'cuda':
                    torch.cuda.synchronize()
                durations.append((time.perf_counter()-t0)*1000)
                found = Counter(prediction.names.get(cid, str(cid)) for cid in class_ids)
                classes.update(found)
                scenes_any += bool(class_ids)
                all_rows.append({'frame': path.name, 'input_resolution': key,
                                 'yolo_imgsz': yolo_size, 'inference_ms': round(durations[-1],3),
                                 'coin': found['coin'], 'train': found['train'],
                                 'barrier': found['barrier'],
                                 'max_conf': round(float(max(confidences)), 3) if confidences else 0})
                if i < a.preview_frames:
                    panel, _ = draw_frame(resized, prediction,
                                          f'{key} / YOLO {yolo_size} / c={found["coin"]} t={found["train"]} b={found["barrier"]}')
                    cv2.imwrite(str(dest / f'frame{i:02d}_{key}_yolo{yolo_size}.jpg'), panel)
            yolo_summaries[spec] = {'inference_ms': timing(durations),
                                    'detections': dict(classes),
                                    'scenes_with_any_detection': scenes_any}
            print(f'[YOLO] {spec}: p50={yolo_summaries[spec]["inference_ms"]["p50_ms"]}ms '
                  f'p95={yolo_summaries[spec]["inference_ms"]["p95_ms"]}ms '
                  f'boxes={dict(classes)}', flush=True)

    # One side-by-side sheet per sampled frame, aligned to identical scene.
    for i in range(min(a.preview_frames, len(frames))):
        rows = []
        for w, h in sizes:
            panels = []
            for yolo_size in a.yolo_sizes:
                p = dest / f'frame{i:02d}_{w}x{h}_yolo{yolo_size}.jpg'
                panel = cv2.imread(str(p))
                if panel is not None:
                    panels.append(panel)
            if panels:
                rows.append(np.hstack(panels))
        if rows and all(row.shape[1] == rows[0].shape[1] for row in rows):
            cv2.imwrite(str(dest / f'COMPARISON_frame{i:02d}.jpg'), np.vstack(rows),
                        [cv2.IMWRITE_JPEG_QUALITY, 94])

    with (a.output / 'detections_by_frame.csv').open('w', newline='', encoding='utf-8') as f:
        writer = csv.DictWriter(f, fieldnames=list(all_rows[0].keys()))
        writer.writeheader()
        writer.writerows(all_rows)
    report = {
        'note': ('All sizes tested on identical ORIGINAL game frames. Model was '
                 'trained on low-resolution data, so higher resolutions are an '
                 'untrained distribution shift. Box counts are NOT precision/recall; '
                 'visually audit COMPARISON images and later label unseen frames.'),
        'native_source_resolution': [native_shape[1], native_shape[0]],
        'frames_tested': len(frames),
        'preprocess_bgr_to_rgb_chw': preprocess,
        'ppo_cnn_encoder_only': ppo,
        'yolo_old_model': yolo_summaries,
        'comparison_images_folder': str(dest),
    }
    save_json(a.output / 'resolution_benchmark_report.json', report)
    print(f'[ANALYZE] Report: {a.output / "resolution_benchmark_report.json"}', flush=True)
    print(f'[ANALYZE] Side-by-side images: {dest / "COMPARISON_frame00.jpg"}', flush=True)
    print('[ANALYZE] Box COUNTS ARE NOT accuracy. Compare missed coins/trains and false positives by eye.', flush=True)


def main():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = p.add_subparsers(dest='mode', required=True)
    cap = sub.add_parser('capture', help='Capture ORIGINAL high-res screenshots from local Unity gameplay')
    cap.add_argument('--fullscreen', action='store_true')
    cap.add_argument('--seconds', type=float, default=75)
    cap.add_argument('--fps', type=int, default=30)
    cap.add_argument('--press-delay-ms', type=int, default=20)
    cap.add_argument('--warmup-seconds', type=float, default=1.15)
    cap.add_argument('--spacing-seconds', type=float, default=.18)
    cap.add_argument('--policy', choices=('noop', 'random'), default='noop',
                     help='noop sends ZERO arrow keys; only R restarts each Practice attempt.')
    cap.add_argument('--loading-reference-dir', type=Path,
                     default=Path('ai_data/v4_resolution_benchmark/native_frames'),
                     help='Prior known loading-screen images to reject (optional).')
    cap.add_argument('--loading-diff-threshold', type=float, default=15.0,
                     help='Reject screenshots this visually similar to an old loading frame.')
    cap.add_argument('--motion-threshold', type=float, default=1.1,
                     help='Require scene motion in successive images (smaller accepts more).')
    cap.add_argument('--max-frames', type=int, default=50)
    cap.add_argument('--run-name', default='v4_resolution_capture')
    cap.add_argument('--output', type=Path, default=Path('ai_data/v4_resolution_gameplay_v2'))

    an = sub.add_parser('analyze', help='Offline YOLO quality and PPO encoder speed sweep')
    an.add_argument('--output', type=Path, default=Path('ai_data/v4_resolution_gameplay_v2'))
    an.add_argument('--detector', type=Path, default=Path('checkpoints/yolo_hazards_best.pt'))
    an.add_argument('--resolutions', type=shape, nargs='+', default=DEFAULT_RES)
    an.add_argument('--yolo-sizes', type=int, nargs='+', default=[640, 960])
    an.add_argument('--conf', type=float, default=.10)
    an.add_argument('--max-analyze-frames', type=int, default=20)
    an.add_argument('--preview-frames', type=int, default=6)
    an.add_argument('--batch', type=int, default=8)
    an.add_argument('--iters', type=int, default=25)

    args = p.parse_args()
    if args.mode == 'capture':
        if not (10 <= args.seconds <= 1200 and 2 <= args.fps <= 30
                and 1 <= args.press_delay_ms <= 100 and args.max_frames >= 1
                and args.spacing_seconds >= 0 and args.warmup_seconds >= 0):
            p.error('Invalid capture parameters')
        capture(args)
    else:
        if not (args.max_analyze_frames >= 1 and args.preview_frames >= 0 and
                args.batch >= 1 and args.iters >= 4 and 0 < args.conf < 1):
            p.error('Invalid analysis parameters')
        if any(sz < 160 or sz % 32 for sz in args.yolo_sizes):
            p.error('YOLO imgsz must be >=160 and divisible by 32')
        analyze(args)


if __name__ == '__main__':
    main()
