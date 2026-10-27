#!/usr/bin/env python3
"""Local-Practice human demo recorder/reviewer. Never sends gameplay keys.

Collect exact V6.1 RGB and auxiliary observations with human arrow-key labels.
Each episode is frozen at MENU, then manually approved/truncated from terminal.
No existing PPO files/checkpoints are modified.
"""
from __future__ import annotations

import argparse
import ctypes
import asyncio
from collections import Counter, deque
from concurrent.futures import ThreadPoolExecutor
from datetime import datetime
import json
import os
from pathlib import Path
import queue
import threading
import time
from urllib.parse import urlsplit

import cv2
import numpy as np

from rs_v61_features import ACTION_NAMES, AUX_DIM, FEATURE_VERSION, ActionContext, encode_aux
from rs_v6_ppo_features import encode_hazards
from rs_v4_screen_states import ScreenStateClassifier
from rs_v5_shadow_segmentation import check_model_names, detect_v5
from rs_v5_segmentation_live_benchmark import DEFAULT_CROP, crop_game, fit_rgb_to_target

ARROWS = {'ArrowLeft': 1, 'ArrowRight': 2, 'ArrowUp': 3, 'ArrowDown': 4}
ROOT = Path('ai_data/demonstrations_v1')
# Capture arrows AND optional WASD/Space. The labels remain PPO's five actions.
WINDOWS_KEYS = {
    'ArrowLeft': 0x25, 'ArrowUp': 0x26, 'ArrowRight': 0x27, 'ArrowDown': 0x28,
    'KeyA': 0x41, 'KeyW': 0x57, 'KeyD': 0x44, 'KeyS': 0x53, 'Space': 0x20,
}
ALIASES = {'KeyA': 'ArrowLeft', 'KeyD': 'ArrowRight',
           'KeyW': 'ArrowUp', 'KeyS': 'ArrowDown', 'Space': 'ArrowUp'}


def canonical_key(code):
    return ALIASES.get(code, code)


def trusted_keydowns(events, *, duplicate_ms=45):
    """Merge the Win32 and browser event streams, avoiding double labels.

    Uses the earliest arrival for any matching real keydown. It never invents
    a keypress from camera/player motion.
    """
    raw = [e for e in events
           if e.get('type') == 'keydown'
           and canonical_key(e.get('code')) in ARROWS
           and not e.get('repeat', False)]
    # Win32 is timestamped at the physical press. Playwright bindings can arrive
    # 100-300ms late if Chromium is busy. A delayed DOM event must NEVER create
    # a second action or shift the human label into the future. Use the whole
    # run's Win32 stream when present; otherwise allow browser-only fallback.
    if any(e.get('source') == 'win32_key_state' for e in raw):
        raw = [e for e in raw if e.get('source') == 'win32_key_state']
    raw.sort(key=lambda e: float(e['at']))
    kept = []
    last = {}
    for e in raw:
        canonical = canonical_key(e['code'])
        at = float(e['at'])
        if canonical not in last or at - last[canonical] > duplicate_ms / 1000.:
            kept.append({**e, 'code': canonical})
        last[canonical] = at
    return kept


def monitor_windows_arrows(events, stop, *, interval=.004):
    """Read *physical* arrow states from Windows, avoiding Unity iframe JS hooks.

    Keys are edge-triggered: holding LEFT produces one keydown, not repeated
    labels. time.perf_counter is the same clock used by MSS capture. Events
    after focus loss do not enter a buffered run (timestamp cutoff at capture).
    """
    if os.name != 'nt':
        raise RuntimeError('This recorder requires Windows keyboard polling')
    get_state = ctypes.windll.user32.GetAsyncKeyState
    get_state.argtypes = (ctypes.c_int,)
    get_state.restype = ctypes.c_short
    previous = {code: False for code in WINDOWS_KEYS}
    while not stop.is_set():
        at = time.perf_counter()
        for code, vk in WINDOWS_KEYS.items():
            down = bool(get_state(vk) & 0x8000)
            if down != previous[code]:
                events.append({'at': at, 'code': code,
                               'type': 'keydown' if down else 'keyup',
                               'repeat': False, 'source': 'win32_key_state'})
                if down:
                    print(f'[HUMAN INPUT] {code} captured', flush=True)
                previous[code] = down
        stop.wait(interval)


def locate_canvas_in_desktop(reference_bgr, desktop_bgra, *, left=0, top=0,
                             min_score=.70, downscale=.23):
    """Locate Chromium's canvas PNG in a physical desktop frame using pixels.

    Unlike the old rs_practice_controller coordinate helper, this does not
    assume Linux Chrome borders, window position, screen scale, or fullscreen.
    A static menu / stopped game is recommended for the initial calibration.
    The return value is an MSS screen rectangle (physical pixels).

    Fail closed if the matched pixels are not similar enough: we prefer refusing
    to record over confidently labeling the wrong screen region as gameplay.
    """
    if (reference_bgr is None or reference_bgr.ndim != 3
            or desktop_bgra is None or desktop_bgra.ndim != 3):
        raise ValueError('Reference and desktop must be color images')
    ref_h, ref_w = reference_bgr.shape[:2]
    screen_h, screen_w = desktop_bgra.shape[:2]
    if min(ref_h, ref_w, screen_h, screen_w) < 40:
        raise ValueError('Unexpectedly small screenshot')
    if ref_w > screen_w or ref_h > screen_h:
        raise ValueError('Canvas screenshot is larger than captured desktop; '
                         'verify Windows display scaling and the correct monitor')
    screen_gray = cv2.cvtColor(desktop_bgra, cv2.COLOR_BGRA2GRAY)
    ref_gray = cv2.cvtColor(reference_bgr, cv2.COLOR_BGR2GRAY)
    size = lambda w,h: (max(1, int(round(w * downscale))),
                       max(1, int(round(h * downscale))))
    small_screen = cv2.resize(screen_gray, size(screen_w,screen_h),
                              interpolation=cv2.INTER_AREA)
    small_ref = cv2.resize(ref_gray, size(ref_w,ref_h),
                           interpolation=cv2.INTER_AREA)
    if small_ref.shape[0] > small_screen.shape[0] or small_ref.shape[1] > small_screen.shape[1]:
        raise ValueError('Canvas exceeds physical desktop after scaling')
    coarse = cv2.matchTemplate(small_screen,small_ref,cv2.TM_CCOEFF_NORMED)
    _,coarse_score,_,pos = cv2.minMaxLoc(coarse)
    x0,y0 = int(round(pos[0]/downscale)), int(round(pos[1]/downscale))
    # Match once in a small full-resolution search window near the coarse
    # location. Calling matchTemplate dozens of times on 1920px frames would
    # needlessly delay the initial calibration.
    radius = 8
    x_left = max(0, x0-radius)
    y_top = max(0, y0-radius)
    x_end = min(screen_w, x0+ref_w+radius+1)
    y_end = min(screen_h, y0+ref_h+radius+1)
    search = screen_gray[y_top:y_end,x_left:x_end]
    if search.shape[0] < ref_h or search.shape[1] < ref_w:
        raise ValueError('Initial canvas match touches invalid desktop coordinates')
    fine = cv2.matchTemplate(search, ref_gray, cv2.TM_CCOEFF_NORMED)
    _, score, _, fine_pos = cv2.minMaxLoc(fine)
    x,y = x_left+fine_pos[0], y_top+fine_pos[1]
    if score < min_score:
        raise ValueError(f'Cannot match Chromium canvas to the physical display '
                         f'(score {score:.3f}; coarse={coarse_score:.3f}); '
                         'keep game visible, use a static menu and retry')
    return {'left':int(left+x),'top':int(top+y),
            'width':int(ref_w),'height':int(ref_h)}, score


async def calibrate_canvas_mss(page,canvas,grabber, *, attempts=4):
    """One-time Playwright screenshot only, then MSS on every gameplay frame."""
    import rs_practice_controller as controller
    best_error = None
    for attempt in range(attempts):
        try:
            png = await controller.screenshot_canvas(page,canvas,image_type='png')
            reference = cv2.imdecode(np.frombuffer(png,dtype=np.uint8),
                                     cv2.IMREAD_COLOR)
            if reference is None:
                raise ValueError('Could not decode browser calibration screenshot')
            desktop = grabber.monitors[0]
            shot = np.asarray(grabber.grab(desktop))
            region,score = locate_canvas_in_desktop(
                reference,shot,left=desktop['left'],top=desktop['top'])
            print(f'[CALIBRATION] MSS canvas {region} pixel similarity={score:.3f}',
                  flush=True)
            return region
        except Exception as exc:
            best_error = exc
            print(f'[CALIBRATION] Attempt {attempt+1}/{attempts}: {exc}',flush=True)
            await asyncio.sleep(.20)
    raise RuntimeError('Cannot safely calibrate fast canvas capture. '
                       'Do not collect training runs with incorrect pixels. '
                       'Open the Practice menu, make game fully visible, then '
                       'restart the recorder; last error: '+str(best_error))


def nonzero_approved_actions(actions, valid, seconds, segments):
    """Count actionable non-NONE labels restricted to approved time ranges."""
    chosen = np.zeros(len(actions), dtype=bool)
    for start, end in segments:
        chosen |= (seconds >= start) & (seconds < end)
    return int(np.count_nonzero((actions != 0) & valid & chosen))


def parse_segments(spec, length):
    """Non-destructive selection in gameplay-relative seconds, strictly positive intervals."""
    result = []
    for chunk in spec.split(','):
        pair = chunk.strip().split('-')
        if len(pair) != 2:
            raise ValueError('Use start-end,start-end, e.g. 0.2-4.8,5.2-8.5')
        a, b = map(float, pair)
        if not (np.isfinite(a) and np.isfinite(b) and 0 <= a < b <= length + 1e-5):
            raise ValueError(f'Invalid interval: {chunk!r}; run length {length:.2f}s')
        result.append([round(a, 3), round(min(b, length), 3)])
    ordered = sorted(result)
    if not ordered or any(ordered[i][0] < ordered[i-1][1] for i in range(1, len(ordered))):
        raise ValueError('Intervals must be nonempty and non-overlapping')
    return ordered


def action_labels(times, events, *, max_lag=.13):
    """Label observation i with first real keydown before next observation.

    Samples with conflicting simultaneous arrows are excluded, not guessed.
    Returns actions and valid masks. No action is assigned from a future image.
    """
    times = np.asarray(times, dtype=float)
    actions = np.zeros(len(times), dtype=np.int64)
    valid = np.zeros(len(times), dtype=bool)
    ordered = trusted_keydowns(events)
    p = 0
    for i in range(len(times)-1):
        t0, t1 = times[i], times[i+1]
        while p < len(ordered) and ordered[p]['at'] <= t0:
            p += 1
        found = []
        q = p
        while q < len(ordered) and ordered[q]['at'] < t1:
            if ordered[q]['at'] - t0 <= max_lag:
                found.append(ARROWS[ordered[q]['code']])
            q += 1
        # No keypress means NONE; multiple distinct actions means label uncertain.
        if len(found) == 0:
            valid[i] = True
        elif len(found) == 1:
            actions[i] = found[0]
            valid[i] = True
        # Also reject intervals with a long capture gap (alt-tab/focus/sleep).
        if t1-t0 > max_lag:
            valid[i] = False
        p = q
    return actions, valid


def replay_features(samples, events, *, episode=1, max_stale_ms=300.):
    """Reconstruct the V6.1.2 observation from causal past labels, no label leakage."""
    if len(samples) < 4:
        raise ValueError('Not enough recorded frames to reconstruct replay')
    ts = np.asarray([s['at'] for s in samples], dtype=np.float64)
    actions, valid = action_labels(ts, events)
    # Live detector requests use the real episode counter, not always 1.
    # Preserve that tag so approved demos after the first episode don't lose
    # every YOLO hazard feature as 'other episode'.
    detector_episode = next((int(s['detection']['episode']) for s in samples
                             if s.get('detection') is not None), episode)
    history = deque(maxlen=4)
    intervals = deque(maxlen=3)
    context = ActionContext(assume_center=False)
    aux = np.zeros((len(samples), AUX_DIM), dtype=np.float32)
    for i, s in enumerate(samples):
        now = float(s['at'])
        if i > 0:
            intervals.append(max(0, now-float(samples[i-1]['at'])))
        snapshot = s['detection']
        hazards, age_ms, _ = encode_hazards(snapshot, now=now,
                                             episode_tag=detector_episode, max_stale_ms=max_stale_ms)
        aux[i] = encode_aux(hazards, history, now-ts[0], intervals, age_ms,
                            snapshot=snapshot, episode_tag=detector_episode, now=now,
                            action_context=context, max_stale_ms=max_stale_ms)
        # Process the action only AFTER forming this frame's observation.
        # This is the same order as observation -> PPO action -> next observation.
        if valid[i]:
            history.append(int(actions[i]))
            context.note_action(int(actions[i]), now)
        else:
            history.append(0)
    frames = np.stack([s['frame'] for s in samples])
    return frames, aux, actions, ts-ts[0], valid


def make_contact_sheet(frames, actions, secs, path, count=18):
    total = len(frames)
    picks = np.unique(np.linspace(0, total-1, min(total, count), dtype=int))
    tiles = []
    for k in picks:
        rgb = frames[k].transpose(1, 2, 0)
        tile = cv2.cvtColor(cv2.resize(rgb, (256, 144)), cv2.COLOR_RGB2BGR)
        tile = cv2.copyMakeBorder(tile, 0, 24, 0, 0, cv2.BORDER_CONSTANT, value=(20, 20, 20))
        cv2.putText(tile, f'{secs[k]:.2f}s {ACTION_NAMES[actions[k]]}', (7, 160),
                    cv2.FONT_HERSHEY_SIMPLEX, .48, (255, 255, 255), 1, cv2.LINE_AA)
        tiles.append(tile)
    while len(tiles) % 3:
        tiles.append(np.zeros_like(tiles[0]))
    sheet = np.vstack([np.hstack(tiles[i:i+3]) for i in range(0, len(tiles), 3)])
    cv2.imwrite(str(path), sheet, [cv2.IMWRITE_JPEG_QUALITY, 92])


def make_preview_video(frames, actions, seconds, folder, canvas_jpegs=None):
    """Human preview uses the entire original canvas, preserving its aspect ratio.

    PPO frames.npz remains the same 3x288x512 game-only image as before.
    """
    fps = int(np.clip(1. / max(1e-3, float(np.median(np.diff(seconds)))), 5, 60))
    has_canvas = bool(canvas_jpegs and canvas_jpegs[0] is not None)
    source = (cv2.imdecode(np.frombuffer(canvas_jpegs[0], dtype=np.uint8),
                           cv2.IMREAD_COLOR)
              if has_canvas else cv2.cvtColor(frames[0].transpose(1, 2, 0),
                                              cv2.COLOR_RGB2BGR))
    if source is None or source.ndim != 3:
        raise ValueError('Invalid video source frame')
    source_h, source_w = source.shape[:2]
    scale = min(1., 960. / source_w)
    width = max(2, 2 * round(source_w * scale / 2))
    height = max(2, 2 * round(source_h * scale / 2))
    for name, codec in (('preview.mp4', 'mp4v'), ('preview.avi', 'MJPG')):
        target = folder / name
        writer = cv2.VideoWriter(str(target), cv2.VideoWriter_fourcc(*codec),
                                 fps, (width, height))
        if not writer.isOpened():
            writer.release()
            continue
        try:
            for idx, frame in enumerate(frames):
                if has_canvas:
                    if canvas_jpegs[idx] is None:
                        raise ValueError('Missing full-canvas frame in replay')
                    bgr = cv2.imdecode(np.frombuffer(canvas_jpegs[idx],
                                                     dtype=np.uint8), cv2.IMREAD_COLOR)
                else:
                    bgr = cv2.cvtColor(frame.transpose(1, 2, 0), cv2.COLOR_RGB2BGR)
                if bgr is None or bgr.shape[:2] != (source_h, source_w):
                    raise ValueError('Inconsistent full-canvas dimensions during replay')
                if bgr.shape[1] != width or bgr.shape[0] != height:
                    bgr = cv2.resize(bgr, (width, height), interpolation=cv2.INTER_AREA)
                labeled = bgr.copy()
                y0 = max(0, height - 32)
                cv2.rectangle(labeled, (0, y0), (min(width, 265), height),
                              (17, 17, 17), -1)
                cv2.putText(labeled,
                            f'{seconds[idx]:.2f}s {ACTION_NAMES[actions[idx]]}',
                            (10, height-10), cv2.FONT_HERSHEY_SIMPLEX,
                            .55, (255, 255, 255), 1, cv2.LINE_AA)
                writer.write(labeled)
        finally:
            writer.release()
        if target.exists() and target.stat().st_size > 1024:
            return name
        target.unlink(missing_ok=True)
    return None


def save_replay(samples, events, output, *, status, segments=None, reason=''):
    """Write immutable sampled data and a separately editable curation manifest."""
    if len(samples) < 4:
        raise ValueError('Run is too short to save')
    frames, aux, actions, secs, valid = replay_features(samples, events)
    duration = float(secs[-1])
    default_end = max(0., duration-.7)
    if segments is None:
        segments = ([[0., round(default_end, 3)]] if default_end >= .3 else [])
    if status == 'approved' and not segments:
        raise ValueError('Cannot approve a run without a positive-length safe segment')
    if status == 'approved' and not nonzero_approved_actions(actions, valid, secs, segments):
        raise ValueError('No correctly labeled physical arrow actions in approved ranges! '
                         'Use S (candidate), inspect keyboard capture, and record again.')
    output.mkdir(parents=True, exist_ok=False)
    np.savez_compressed(output/'frames.npz', frames=frames)
    np.save(output/'aux.npy', aux)
    np.save(output/'actions.npy', actions)
    np.save(output/'timestamps.npy', secs)
    np.save(output/'valid.npy', valid)
    (output/'events.jsonl').write_text(''.join(json.dumps(e)+'\n' for e in events), encoding='utf-8')
    make_contact_sheet(frames, actions, secs, output/'contact_sheet.jpg')
    video = make_preview_video(frames, actions, secs, output,
                               [s.get('canvas_jpeg') for s in samples])
    metadata = {
        'schema': 1, 'status': status, 'feature_version': FEATURE_VERSION, 'preview_video': video,
        'image_shape': list(frames.shape[1:]), 'aux_dim': AUX_DIM,
        'action_names': list(ACTION_NAMES), 'frames': int(len(frames)),
        'duration_seconds': round(duration, 3), 'valid_segments_seconds': segments,
        'valid_action_samples': int(valid.sum()),
        'action_counts_valid': dict(Counter(ACTION_NAMES[int(a)] for a, good in zip(actions, valid) if good)),
        'reason': reason, 'last_0_7s_excluded_by_default': True,
        'frame_storage': 'single RGB CHW frame; reconstruct 4-frame stack with oldest-first order',
        'preview_source': 'full_canvas_before_model_crop_when_available',
        'recorded_keydown_count': len(trusted_keydowns(events)),
        'keyboard_sources': dict(Counter(e.get('source','unknown') for e in events)),
        'preview_canvas_original_shape': (
            list(cv2.imdecode(np.frombuffer(samples[0]['canvas_jpeg'],
                 dtype=np.uint8), cv2.IMREAD_COLOR).shape[:2])
            if samples[0].get('canvas_jpeg') is not None else None),
        'review_note': 'Approve only genuinely good intervals; long survival alone does not prove optimal actions.',
    }
    (output/'manifest.json').write_text(json.dumps(metadata, indent=2)+'\n', encoding='utf-8')
    return metadata


def curate(folder, status, segments=None):
    file = folder/'manifest.json'
    data = json.loads(file.read_text(encoding='utf-8'))
    if data['schema'] != 1 or data['feature_version'] != FEATURE_VERSION:
        raise ValueError('Incompatible demo schema/version')
    if segments is not None:
        data['valid_segments_seconds'] = parse_segments(segments, float(data['duration_seconds']))
    if status == 'approved' and not data['valid_segments_seconds']:
        raise ValueError('Approved demos require at least one valid interval')
    if status == 'approved':
        acts = np.load(folder/'actions.npy', allow_pickle=False)
        valid = np.load(folder/'valid.npy', allow_pickle=False)
        secs = np.load(folder/'timestamps.npy', allow_pickle=False)
        if not nonzero_approved_actions(acts, valid, secs, data['valid_segments_seconds']):
            raise ValueError('Replay has no valid physical arrow actions in approved intervals; '
                             'leave it as a candidate. The old recorder may have lost all key events.')
    if status not in {'candidate', 'approved', 'rejected'}:
        raise ValueError(status)
    data['status'] = status
    file.write_text(json.dumps(data, indent=2)+'\n', encoding='utf-8')
    return data


def console_reader(commands, stop):
    print('Commands after Alt+Tab: G=approve, S=save candidate, T=trim+approve, P=preview, R=discard, Q=quit', flush=True)
    while not stop.is_set():
        try:
            cmd = input('demo> ').strip().lower()
        except (EOFError, KeyboardInterrupt):
            cmd = 'q'
        if cmd == 't':
            try:
                spec = input('Good ranges, e.g. 0.2-4.8,5.1-7.4: ').strip()
            except (EOFError, KeyboardInterrupt):
                spec = ''
            commands.put(('t', spec))
        elif cmd in {'g','s','p','r','q'}:
            commands.put((cmd, ''))
        else:
            print('Use G / S / T / P / R / Q', flush=True)


async def capture_loop(a):
    url_parts = urlsplit(a.url)
    if (url_parts.scheme != 'http' or url_parts.hostname not in {'127.0.0.1','localhost'}
            or url_parts.username or url_parts.password):
        raise ValueError('Only local HTTP Practice URLs on localhost are permitted')
    if not a.detector.is_file() or not a.state_model.is_file():
        raise FileNotFoundError('Missing YOLO detector or screen-state checkpoint')
    import mss
    from playwright.async_api import async_playwright
    from ultralytics import YOLO
    import rs_practice_controller as controller
    detector = YOLO(str(a.detector))
    check_model_names(detector)
    states = ScreenStateClassifier(a.state_model)
    commands = queue.Queue()
    stop = threading.Event()
    thread = threading.Thread(target=console_reader, args=(commands, stop), daemon=True)
    thread.start()
    events = []
    if os.name != 'nt':
        raise RuntimeError('Windows is required for reliable physical keyboard capture')
    keyboard_thread = threading.Thread(target=monitor_windows_arrows,
                                       args=(events,stop), daemon=True)
    keyboard_thread.start()
    print('[DEMO] Windows physical-key monitor active. Press an arrow key during '
          'gameplay; check for [HUMAN INPUT] in the terminal before approving.', flush=True)

    a.root.mkdir(parents=True, exist_ok=True)
    with ThreadPoolExecutor(max_workers=1, thread_name_prefix='demo-yolo') as pool:
      async with async_playwright() as pw:
        browser = await pw.chromium.launch(headless=False, args=['--start-maximized'])
        ctx = await browser.new_context(no_viewport=True)

        def browser_keys(_source, payload):
            # The browser callback also works for child Unity iframes. Playwright
            # may deliver it a few ms after the event; Win32 is preferred if
            # both sources captured the same physical keypress.
            if (isinstance(payload, dict)
                    and payload.get('code') in WINDOWS_KEYS
                    and payload.get('event') in ('keydown', 'keyup')):
                events.append({'at': time.perf_counter(),
                               'code': payload['code'], 'type': payload['event'],
                               'repeat': bool(payload.get('repeat', False)),
                               'source': 'browser_dom'})
                if payload['event'] == 'keydown' and not payload.get('repeat'):
                    print(f"[BROWSER INPUT] {payload['code']} captured", flush=True)

        await ctx.expose_binding('__rs_demo_key_event', browser_keys)
        await ctx.add_init_script("""
          (() => {
            const accepted = new Set([
              'ArrowLeft','ArrowRight','ArrowUp','ArrowDown',
              'KeyA','KeyD','KeyW','KeyS','Space'
            ]);
            for (const type of ['keydown','keyup']) {
              window.addEventListener(type, e => {
                if (!e.isTrusted || !accepted.has(e.code)) return;
                window.__rs_demo_key_event({
                  event: type, code: e.code, repeat: e.repeat
                }).catch(() => {});
              }, true);
            }
          })();
        """)
        page = await ctx.new_page()
        try:
            cdp = await ctx.new_cdp_session(page)
            wid = (await cdp.send('Browser.getWindowForTarget'))['windowId']
            await cdp.send('Browser.setWindowBounds', {'windowId':wid, 'bounds':{'windowState':'maximized'}})
        except Exception as e:
            print('Could not maximize Chromium:', e)
        await page.goto(a.url, wait_until='domcontentloaded')
        landed = urlsplit(page.url)
        if landed.hostname not in {'127.0.0.1','localhost'}:
            raise RuntimeError('Local Practice page redirected outside localhost; refusing recording')
        await page.bring_to_front()
        print('Click local Practice and use keyboard normally. Manually press R to restart after death.', flush=True)
        canvas = await controller.find_canvas(page, timeout=60)
        with mss.mss() as grabber:
            print('[CAPTURE] Calibrating canvas against physical desktop once. '
                  'There are NO Chromium screenshots during normal gameplay.',
                  flush=True)
            region = await calibrate_canvas_mss(page,canvas,grabber)
            async def canvas_view_signature():
                info = await page.evaluate("""() => [
                    window.innerWidth,window.innerHeight,
                    window.devicePixelRatio,window.screenX,window.screenY
                ]""")
                bounds = await canvas.bounding_box()
                if not bounds:
                    raise RuntimeError('Game canvas temporarily not visible')
                # Canvas CSS geometry can change even without a browser
                # viewport change (e.g. the game enters its own fullscreen).
                return tuple(info) + tuple(round(bounds[k],1)
                       for k in ('x','y','width','height'))
            canvas_view = await canvas_view_signature()
            last_view_check = time.perf_counter()
            mode, streak, samples, start_event_index = 'idle', 0, [], 0
            recent = None
            menu_first = None
            future, last_detection, request_id = None, None, 0
            episode = 0
            interval = 1./a.fps
            first_capture_written = False
            capture_intervals = deque(maxlen=80)
            last_captured = None
            last_rate_log = time.perf_counter()
            next_tick = time.perf_counter()
            try:
                while True:
                    # Commands can be entered after Alt+Tab without capturing terminal pixels.
                    try:
                        cmd, spec = commands.get_nowait()
                    except queue.Empty:
                        cmd = None
                    if cmd == 'q':
                        break
                    if cmd and cmd != 'q':
                        if cmd == 'r':
                            recent = None
                            print('Discarded the buffered completed run.', flush=True)
                        elif recent is None:
                            print('No completed run in memory. Finish a run / wait for MENU first.', flush=True)
                        else:
                            run_samples, run_events, duration = recent
                            if cmd in {'g','s','p','t'}:
                                try:
                                    segments = parse_segments(spec, duration) if cmd == 't' else None
                                    status = 'approved' if cmd in {'g','t'} else 'candidate'
                                    name = datetime.now().strftime('demo_%Y%m%d_%H%M%S_%f')
                                    folder = a.root/'runs'/name
                                    summary = save_replay(run_samples, run_events, folder, status=status, segments=segments)
                                    print(f'Saved {status}: {folder} | {summary["action_counts_valid"]}', flush=True)
                                    if cmd == 'p' and os.name == 'nt':
                                        os.startfile(str(folder/summary['preview_video']) if summary.get('preview_video') else str(folder/'contact_sheet.jpg'))
                                    recent = None
                                except Exception as e:
                                    print('Could not save:', e, flush=True)
                    # Never sample the terminal once browser has lost focus. Menu is already frozen.
                    try:
                        focused = await page.evaluate('document.hasFocus() && document.visibilityState === "visible"')
                    except Exception:
                        focused = False
                    if not focused:
                        if mode == 'recording':
                            if len(samples) >= 4:
                                recent = (samples, [e for e in events[start_event_index:] if e['at'] <= samples[-1]['at']],
                                          samples[-1]['at']-samples[0]['at'])
                                print(f'[RUN {episode}] Alt+Tab: paused and buffered {recent[2]:.2f}s. Use G/S/T/P/R.', flush=True)
                            mode, samples = 'idle', []
                        await asyncio.sleep(.08)
                        next_tick = time.perf_counter()+interval
                        continue
                    # Only check for window move/fullscreen transitions about
                    # once a second; the heavy screenshot happens only on change.
                    if time.perf_counter()-last_view_check >= 1.0:
                        current_view = await canvas_view_signature()
                        last_view_check = time.perf_counter()
                        if current_view != canvas_view:
                            print('[CAPTURE] Browser moved/resized/fullscreen changed. '
                                  'Discarding partial run and recalibrating!', flush=True)
                            mode,streak,samples,recent = 'idle',0,[],None
                            last_detection,future,request_id = None,None,0
                            region = await calibrate_canvas_mss(page,canvas,grabber)
                            canvas_view=current_view
                            first_capture_written=False
                            last_captured=None
                            capture_intervals.clear()
                            next_tick=time.perf_counter()+interval
                            continue
                    shot = np.asarray(grabber.grab(region))
                    captured_at = time.perf_counter()
                    if last_captured is not None:
                        capture_intervals.append(captured_at-last_captured)
                    last_captured=captured_at
                    if captured_at-last_rate_log>=5.0:
                        effective_fps = (1. / float(np.median(capture_intervals))
                                         if capture_intervals else 0.0)
                        print(f'[CAPTURE] Measured frame interval '
                              f'{1000./effective_fps:.0f} ms ('
                              f'{effective_fps:.1f} FPS)'
                              if effective_fps else '[CAPTURE] FPS not yet available',
                              flush=True)
                        if effective_fps < 8:
                            print('[WARNING] Capture below 8 FPS; BC action '
                                  'samples may be invalid. Save only candidate.',
                                  flush=True)
                        last_rate_log=captured_at
                    if not first_capture_written:
                        preview_file = a.root/'capture_preview_full_canvas.png'
                        if not cv2.imwrite(str(preview_file), shot):
                            raise RuntimeError('Could not save initial canvas preview')
                        first_capture_written = True
                        print(f'[CAPTURE] Raw canvas {shot.shape[1]}x{shot.shape[0]}; '
                              f'preview saved: {preview_file}', flush=True)
                    state, probs = states.state(shot, confidence=a.state_confidence)
                    # Uncertain screenshots never get recorded as gameplay.
                    if state == 'GAMEPLAY':
                        if mode != 'recording':
                            streak += 1
                            if streak >= 3:
                                episode += 1
                                mode = 'recording'
                                samples = []
                                start_event_index = len(events)
                                last_detection, future = None, None
                                request_id = 0
                                menu_first = None
                                print(f'[RUN {episode}] Gameplay confirmed; playing (no injected keys).', flush=True)
                        else:
                            streak = 0
                            menu_first = None  # menu suspicion was a transient false alarm
                    elif state == 'MENU':
                        if mode == 'recording':
                            if menu_first is None:
                                menu_first = captured_at
                            streak += 1
                            if streak >= 3:
                                if len(samples) >= 4:
                                    recent = (samples, [e for e in events[start_event_index:] if e['at'] <= menu_first],
                                              samples[-1]['at']-samples[0]['at'])
                                    print(f'[RUN {episode}] Completed {recent[2]:.2f}s, {len(samples)} frames. Alt+Tab -> G/S/T/P/R.', flush=True)
                                mode, samples = 'idle', []
                                last_detection = None
                                streak = 0
                        else:
                            streak = 0
                    else:
                        streak = 0
                    if mode == 'recording' and state == 'GAMEPLAY':
                        if future is not None and future.done():
                            try:
                                result = future.result()
                                if result['episode'] == episode:
                                    last_detection = {k:v for k,v in result.items() if k != 'source_frame'}
                            except Exception as e:
                                print('YOLO failed; stopping recorder rather than saving mismatched features:', e, flush=True)
                                raise
                            future = None
                        native = cv2.cvtColor(crop_game(shot, DEFAULT_CROP), cv2.COLOR_BGRA2RGB)
                        rgb = fit_rgb_to_target(native, (512,288), interpolation=cv2.INTER_LINEAR)
                        frame = np.ascontiguousarray(rgb.transpose(2,0,1))
                        # Human review shows the ENTIRE canvas, while V6.1 PPO still sees
                        # the exact crop+letterbox input it was trained on.
                        # Keep the entire source canvas at its true original
                        # width/height for human review, not another 512x288 crop.
                        full_canvas_bgr = cv2.cvtColor(shot, cv2.COLOR_BGRA2BGR)
                        ok, encoded = cv2.imencode(
                            '.jpg', full_canvas_bgr,
                            [cv2.IMWRITE_JPEG_QUALITY, 89])
                        if not ok:
                            raise RuntimeError('Could not encode human-review canvas frame')
                        samples.append({'at':captured_at, 'frame':frame,
                                        'canvas_jpeg':encoded.tobytes(), 'detection':last_detection})
                        if len(samples)%a.detect_every == 1 and future is None:
                            future = pool.submit(detect_v5, detector, native.copy(), captured_at,
                                                 episode, request_id, a.imgsz, a.conf)
                            request_id += 1
                        if len(samples) >= a.fps*a.max_run_seconds:
                            # Max-length run: store as complete but only pre-stop frames.
                            recent = (samples, [e for e in events[start_event_index:] if e['at'] <= captured_at],
                                      samples[-1]['at']-samples[0]['at'])
                            print('Max run length reached; buffered. Stop/restart game manually.', flush=True)
                            mode, samples = 'idle', []
                    # Yield for browser focus/visibility monitoring.
                    next_tick += interval
                    if next_tick < time.perf_counter()-.5:
                        next_tick = time.perf_counter()
                    await asyncio.sleep(max(.002, next_tick-time.perf_counter()))
            finally:
                stop.set()
                await browser.close()
    print('Recorder exited. PPO checkpoints were not touched.', flush=True)


def main():
    p = argparse.ArgumentParser(description=__doc__)
    sub = p.add_subparsers(dest='command', required=True)
    rec = sub.add_parser('record', help='Local manual game + RAM ring-buffer recorder')
    rec.add_argument('--url', default='http://localhost:8000')
    rec.add_argument('--root', type=Path, default=ROOT)
    rec.add_argument('--detector', type=Path, default=Path('ai_data/v5_hazards/runs/segment_round04/weights/best.pt'))
    rec.add_argument('--state-model', type=Path, default=Path('ai_data/v4_screen_states_rebuild_01/screen_state_candidate_v2.pt'))
    rec.add_argument('--fps', type=int, default=20)
    rec.add_argument('--detect-every', type=int, default=3)
    rec.add_argument('--imgsz', type=int, default=960)
    rec.add_argument('--conf', type=float, default=.20)
    rec.add_argument('--state-confidence', type=float, default=.90)
    rec.add_argument('--max-run-seconds', type=int, default=45)
    rec.add_argument('--screen-x-offset', type=int, default=0)
    rec.add_argument('--screen-y-offset', type=int, default=0)
    review = sub.add_parser('review', help='Change approval intervals without touching captured data')
    review.add_argument('folder', type=Path)
    review.add_argument('--status', choices=['candidate','approved','rejected'], required=True)
    review.add_argument('--segments', default=None, help='e.g. 0.2-4.5,5.0-7.2')
    a = p.parse_args()
    if a.command == 'record':
        if not 5 <= a.fps <= 40 or a.detect_every < 1 or a.max_run_seconds < 3:
            p.error('FPS must be 5..40, detect-every >=1, max run >=3 seconds')
        asyncio.run(capture_loop(a))
    else:
        print(json.dumps(curate(a.folder, a.status, a.segments), indent=2))


if __name__ == '__main__':
    main()

