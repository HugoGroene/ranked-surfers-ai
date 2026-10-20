#!/usr/bin/env python3
"""Ranked Surfers V4: benchmark screenshot sizes and YOLO resolution.

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
from collections import Counter
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


def capture(a):
    import train_rs_v2_verify as v2
    from rs_v2_timing_probe_ab import TimedEnv

    dest = a.output / 'native_frames'
    dest.mkdir(parents=True, exist_ok=True)
    preexisting = list(dest.glob('*.jpg')) + list(dest.glob('*.png'))
    if preexisting:
        raise SystemExit(f'{dest} already has {len(preexisting)} screenshots. '
                         'Choose a NEW --output directory to avoid mixing runs.')
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
    try:
        env = TimedEnv(args, folder, demos)
        env.probe_press_delay_ms = a.press_delay_ms
        env.probe_wait_mode = 'none'
        print('[CAPTURE] Keep Chromium visible; capturing NATIVE pixels from the fullscreen game.', flush=True)
        while time.monotonic() < deadline and saved < a.max_frames:
            env.reset()
            episode_no += 1
            episode_start = time.monotonic()
            for _ in range(450):
                if time.monotonic() >= deadline or saved >= a.max_frames:
                    break
                action = int(rng.choice(5, p=[.60, .10, .10, .10, .10]))
                _, _, dead, truncated, _ = env.step(action)
                steps += 1
                now = time.monotonic()
                if (not dead and not truncated
                        and now - episode_start >= a.warmup_seconds
                        and now - last_at >= a.spacing_seconds):
                    # This is the ORIGINAL MSS screenshot captured by the environment.
                    # It has never passed through the 256x144 PPO preprocessor.
                    bgra = env._v2_last_bgra
                    if bgra is not None and bgra.ndim == 3 and bgra.shape[2] == 4:
                        bgr = cv2.cvtColor(bgra, cv2.COLOR_BGRA2BGR)
                        filename = dest / f'ep{episode_no:03d}_frame{steps:05d}.jpg'
                        if cv2.imwrite(str(filename), bgr,
                                       [cv2.IMWRITE_JPEG_QUALITY, 97]):
                            saved += 1
                            last_at = now
                            if saved % 10 == 0 or saved == 1:
                                print(f'[CAPTURE] {saved} native frames, {episode_no} episodes, '
                                      f'{bgr.shape[1]}x{bgr.shape[0]} pixels', flush=True)
                if dead or truncated:
                    break
    except KeyboardInterrupt:
        print('[CAPTURE] Interrupted; existing screenshots are preserved.', flush=True)
    finally:
        if env is not None:
            env.close()
    summary = {'native_frames': saved, 'episodes_started': episode_no, 'steps': steps,
               'fps_flag': a.fps, 'press_delay_ms': a.press_delay_ms,
               'screenshots_folder': str(dest)}
    save_json(a.output / 'capture_report.json', summary)
    print('[CAPTURE] ' + json.dumps(summary), flush=True)
    print('[CAPTURE] Next: analyze --detector checkpoints/yolo_hazards_best.pt', flush=True)


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
    cap.add_argument('--warmup-seconds', type=float, default=.75)
    cap.add_argument('--spacing-seconds', type=float, default=.30)
    cap.add_argument('--max-frames', type=int, default=50)
    cap.add_argument('--run-name', default='v4_resolution_capture')
    cap.add_argument('--output', type=Path, default=Path('ai_data/v4_resolution_benchmark'))

    an = sub.add_parser('analyze', help='Offline YOLO quality and PPO encoder speed sweep')
    an.add_argument('--output', type=Path, default=Path('ai_data/v4_resolution_benchmark'))
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
