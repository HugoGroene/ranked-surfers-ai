#!/usr/bin/env python3
"""Capture native-resolution local Ranked Surfers GAMEPLAY for V5 polygon labeling.

Records the Windows monitor used by the existing V4 controller. It sends NO
keyboard/mouse inputs and never launches/controls the game. Run the LOCAL game
in the same Chromium window/layout used to benchmark V4, and PLAY manually.

Examples:
  python rs_v5_record_native.py --session v5_a --seconds 120 --fps 8
  python rs_v5_record_native.py --session v5_b --seconds 180 --fps 10 --monitor 1
"""
from __future__ import annotations
import argparse
from collections import Counter
from datetime import datetime
import json
from pathlib import Path
import time

import cv2
import numpy as np


def valid_crop(v):
    if len(v) != 4 or not (0 <= v[0] < v[2] <= 1 and 0 <= v[1] < v[3] <= 1):
        raise ValueError('--crop: 4 normalized coordinates: left top right bottom')
    return tuple(v)


def rect_from_crop(shape, crop):
    h, w = shape[:2]
    l, t, r, b = crop
    return (round(l*w), round(t*h), round(r*w), round(b*h))


def capture(a):
    import mss
    from rs_v4_screen_states import ScreenStateClassifier

    a.crop = valid_crop(a.crop)
    if not a.session or Path(a.session).name != a.session or a.session in ('.', '..'):
        raise SystemExit('Session must be a single folder name (e.g. v5_run_01)')
    dest = a.root / a.session
    if dest.exists() and any(dest.iterdir()):
        raise SystemExit(f'Output exists; refusing to overwrite: {dest}')
    if not a.state_model.is_file():
        raise SystemExit(f'Missing V4 state model: {a.state_model}')
    dest.mkdir(parents=True, exist_ok=True)
    classifier = ScreenStateClassifier(a.state_model, threads=1)
    counts = Counter()
    saved = 0
    last_saved_at = -1e8
    last_gray = None
    capture_wh = None
    start = None
    try:
        with mss.mss() as sct:
            if not 1 <= a.monitor < len(sct.monitors):
                raise SystemExit(f'Invalid --monitor {a.monitor}; available 1..{len(sct.monitors)-1}')
            mon = sct.monitors[a.monitor]
            print(f'[V5] Monitor {a.monitor} {mon}; game crop {a.crop}', flush=True)
            print('[V5] Open your LOCAL game in the same Chromium framing as V4.', flush=True)
            print('[V5] Play manually (including R); this script sends NO controls.', flush=True)
            for i in range(a.delay, 0, -1):
                print(f'[V5] Starting in {i}...', flush=True)
                time.sleep(1)
            start = time.perf_counter()
            next_time = start
            while time.perf_counter() - start < a.seconds and saved < a.max_images:
                now = time.perf_counter()
                if now < next_time:
                    time.sleep(min(next_time - now, 0.025))
                    continue
                next_time = max(next_time + 1 / a.fps, now)
                shot = np.asarray(sct.grab(mon))
                if capture_wh is None:
                    capture_wh = [int(shot.shape[1]), int(shot.shape[0])]
                state, probs = classifier.state(shot, confidence=a.state_confidence)
                counts[state] += 1
                if state != 'GAMEPLAY':
                    continue
                if now - last_saved_at < a.save_every:
                    counts['SKIP_SPACING'] += 1
                    continue
                x1, y1, x2, y2 = rect_from_crop(shot.shape, a.crop)
                crop = shot[y1:y2, x1:x2]
                if not crop.size:
                    raise RuntimeError('Empty cropped frame; check --crop')
                gray = cv2.resize(cv2.cvtColor(crop, cv2.COLOR_BGRA2GRAY), (192,108), interpolation=cv2.INTER_AREA)
                if last_gray is not None and a.min_change > 0:
                    delta = float(np.abs(gray.astype(np.int16)-last_gray.astype(np.int16)).mean())
                    if delta < a.min_change:
                        counts['SKIP_SIMILAR'] += 1
                        continue
                name = f'frame_{saved:05d}_{round((now-start)*1000):08d}ms.{a.format}'
                destfile = dest / name
                params = [cv2.IMWRITE_JPEG_QUALITY, a.quality] if a.format == 'jpg' else [cv2.IMWRITE_PNG_COMPRESSION, 2]
                if not cv2.imwrite(str(destfile), cv2.cvtColor(crop,cv2.COLOR_BGRA2BGR), params):
                    raise OSError(f'Failed to write {destfile}')
                if saved == 0:
                    print(f'[V5] FIRST IMAGE {destfile}; cropped resolution {crop.shape[1]}x{crop.shape[0]}.', flush=True)
                last_saved_at = now
                last_gray = gray
                saved += 1
                if saved % 25 == 0:
                    print(f'[V5] Saved {saved}; states={dict(counts)}', flush=True)
    except KeyboardInterrupt:
        print('[V5] Interrupted; keeping saved images.', flush=True)
    finally:
        report = {
            'session':a.session, 'start_time': datetime.now().isoformat(timespec='seconds'),
            'monitor':a.monitor, 'monitor_wh':capture_wh,
            'crop':list(a.crop), 'seconds_requested':a.seconds,'fps_requested':a.fps,
            'save_every':a.save_every,'min_change':a.min_change,
            'state_confidence':a.state_confidence, 'saved_images':saved,
            'state_samples':dict(counts),
            'note':'Raw native-resolution cropped game images; human must annotate object polygons with LabelMe.',
        }
        (dest / 'session_manifest.json').write_text(json.dumps(report,indent=2), encoding='utf8')
    print(f'[V5] Finished: {saved} images in {dest}',flush=True)
    print('[V5] Inspect these frames for correct game crop before labeling.',flush=True)


def main():
    p=argparse.ArgumentParser(description=__doc__,formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    p.add_argument('--session',default=None,help='Unique run/session name')
    p.add_argument('--root',type=Path,default=Path('ai_data/v5_hazards/raw'))
    p.add_argument('--seconds',type=int,default=120)
    p.add_argument('--fps',type=float,default=8.0,help='Monitor checks per second')
    p.add_argument('--save-every',type=float,default=.30,help='Minimum seconds between saved gameplay frames')
    p.add_argument('--max-images',type=int,default=400)
    p.add_argument('--min-change',type=float,default=1.25,help='Mean grayscale difference threshold for near-duplicates, 0 disables')
    p.add_argument('--delay',type=int,default=5)
    p.add_argument('--monitor',type=int,default=1)
    p.add_argument('--state-model',type=Path,default=Path('ai_data/v4_screen_states/screen_state_best.pt'))
    p.add_argument('--state-confidence',type=float,default=.90)
    p.add_argument('--crop',nargs=4,type=float,default=[0, .055, 1, .965],metavar=('LEFT','TOP','RIGHT','BOTTOM'))
    p.add_argument('--format',choices=('jpg','png'),default='jpg')
    p.add_argument('--quality',type=int,default=96)
    a=p.parse_args()
    if not (0<a.seconds<=7200 and 0<a.fps<=60 and a.save_every>=0 and a.max_images>0 and a.delay>=0
            and 0.5<=a.state_confidence<1.0 and 1<=a.quality<=100 and a.min_change>=0):
        p.error('Invalid numeric argument')
    if a.session is None:
        a.session=datetime.now().strftime('v5_%Y%m%d_%H%M%S')
    capture(a)

if __name__=='__main__':
    main()
