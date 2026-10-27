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
import platform
import queue
import subprocess
import sys
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
# V1.3 capture/labeling behavior plus read-only diagnostics (no behavior change).
RECORDER_VERSION = '1.3.1-diagnostics'
DIAGNOSTICS_SCHEMA = 1
# Per-frame capture-loop timestamps (perf_counter seconds), in loop order.
TIMING_STAGES = ('loop_start', 'focus_done', 'grab_start', 'grab_end',
                 'classify_done', 'preprocess_done', 'jpeg_done',
                 'iteration_end', 'sleep_end')


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


def monitor_windows_arrows(events, stop, *, interval=.004, stats=None, get_state=None):
    """Read *physical* arrow states from Windows, avoiding Unity iframe JS hooks.

    Keys are edge-triggered: holding LEFT produces one keydown, not repeated
    labels. time.perf_counter is the same clock used by MSS capture. Events
    after focus loss do not enter a buffered run (timestamp cutoff at capture).

    Optional ``stats`` (a dict) receives diagnostics only: the measured poll
    period (``poll_intervals``, seconds) and any fatal ``error``. Requested
    ``interval`` may be rounded up by the Windows timer; this measures it.
    ``get_state`` is injectable for offline tests.
    """
    if get_state is None:
        if os.name != 'nt':
            raise RuntimeError('This recorder requires Windows keyboard polling')
        get_state = ctypes.windll.user32.GetAsyncKeyState
        get_state.argtypes = (ctypes.c_int,)
        get_state.restype = ctypes.c_short
    if stats is not None:
        stats.setdefault('poll_intervals', deque(maxlen=4000))
        stats['requested_interval_ms'] = interval * 1000.
        stats['error'] = None
    previous = {code: False for code in WINDOWS_KEYS}
    last_poll = None
    try:
        while not stop.is_set():
            at = time.perf_counter()
            if stats is not None and last_poll is not None:
                stats['poll_intervals'].append(at - last_poll)
            last_poll = at
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
    except BaseException as exc:
        if stats is not None:
            stats['error'] = repr(exc)
        raise


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


async def calibrate_canvas_mss(page,canvas,grabber, *, attempts=4, with_score=False):
    """One-time Playwright screenshot only, then MSS on every gameplay frame.

    ``with_score=True`` returns ``(region, pixel_similarity)`` for diagnostics.
    """
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
            return (region, float(score)) if with_score else region
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


# --- Read-only diagnostics -------------------------------------------------
# Nothing below changes labels, validity, observations or approval rules.

def _ms_summary(values_ms):
    a = np.asarray([v for v in values_ms if v is not None and np.isfinite(v)], dtype=float)
    if a.size == 0:
        return {'count': 0}
    return {'count': int(a.size), 'median': round(float(np.median(a)), 2),
            'p95': round(float(np.percentile(a, 95)), 2),
            'mean': round(float(np.mean(a)), 2),
            'min': round(float(np.min(a)), 2), 'max': round(float(np.max(a)), 2)}


def frame_interval_stats(times, *, max_lag=.13):
    """Capture-interval distribution relative to the label-validity bound."""
    times = np.asarray(times, dtype=float)
    gaps = np.diff(times) * 1000.
    out = {'frames': int(len(times)), 'max_lag_ms': round(max_lag * 1000., 1),
           'intervals_ms': _ms_summary(gaps)}
    if gaps.size:
        over = int(np.count_nonzero(gaps > max_lag * 1000.))
        out.update({'intervals_over_max_lag': over,
                    'fraction_intervals_within_max_lag': round(1. - over / gaps.size, 4),
                    'effective_fps_from_median': round(1000. / float(np.median(gaps)), 2)
                    if np.median(gaps) > 0 else None,
                    # Gaps this long usually mean a pause, focus loss, or two
                    # episodes joined into one buffered run.
                    'intervals_over_1s': int(np.count_nonzero(gaps > 1000.))})
    return out


def validity_stats(actions, valid, seconds, *, max_lag=.13, segments=None):
    """Label/validity counts. ``valid_with_stack_gap`` is informational: valid
    samples whose 4-frame history spans an interval longer than ``max_lag``."""
    actions = np.asarray(actions)
    valid = np.asarray(valid, dtype=bool)
    seconds = np.asarray(seconds, dtype=float)
    gaps = np.diff(seconds)
    stack_gap = 0
    for i in np.flatnonzero(valid):
        if np.any(gaps[max(0, i-3):i] > max_lag):
            stack_gap += 1
    out = {'samples': int(len(actions)), 'valid_samples': int(valid.sum()),
           'valid_fraction': round(float(valid.mean()), 4) if len(valid) else None,
           'valid_by_action': {name: int(np.count_nonzero(valid & (actions == k)))
                               for k, name in enumerate(ACTION_NAMES)},
           'valid_nonnone_samples': int(np.count_nonzero(valid & (actions != 0))),
           'invalid_labeled_nonnone': int(np.count_nonzero(~valid & (actions != 0))),
           'valid_with_stack_gap': int(stack_gap)}
    if segments:
        chosen = np.zeros(len(actions), dtype=bool)
        for start, end in segments:
            chosen |= (seconds >= start) & (seconds < end)
        out['segments_seconds'] = [list(map(float, s)) for s in segments]
        out['valid_in_segments_by_action'] = {
            name: int(np.count_nonzero(chosen & valid & (actions == k)))
            for k, name in enumerate(ACTION_NAMES)}
    return out


def label_diagnostics(times, events, *, max_lag=.13):
    """Explain what happened to every trusted keydown under ``action_labels``.

    ``times`` must be on the same absolute clock as event ``at`` values.
    Returns ``(summary, outcomes)``. Eligible keydowns are those pressed while
    gameplay frames were being recorded (strictly between the first and last
    frame); terminal commands and menu inputs fall outside that window.
    Reasons: assigned, long_interval, conflicting_keys, duplicate_key,
    frame_boundary, before_first_frame, after_last_frame.
    """
    times = np.asarray(times, dtype=float)
    actions, valid = action_labels(times, events, max_lag=max_lag)
    keydowns = trusted_keydowns(events)
    keys_at = np.asarray([float(k['at']) for k in keydowns], dtype=float)
    outcomes = []
    for k in keydowns:
        at = float(k['at'])
        action = ARROWS[k['code']]
        rec = {'at': at, 'code': k['code'], 'action': ACTION_NAMES[action],
               'source': k.get('source', 'unknown'), 'frame': None,
               'seconds': round(at - float(times[0]), 4) if len(times) else None,
               'offset_after_frame_ms': None, 'interval_ms': None}
        if len(times) < 2 or at <= times[0]:
            reason = 'before_first_frame'
        elif at >= times[-1]:
            reason = 'after_last_frame'
        else:
            j = int(np.searchsorted(times, at, side='left'))
            if times[j] == at:
                reason = 'frame_boundary'  # action_labels assigns it to no interval
            else:
                i = j - 1
                t0, t1 = float(times[i]), float(times[j])
                rec.update({'frame': int(i), 'offset_after_frame_ms': round((at-t0)*1000., 2),
                            'interval_ms': round((t1-t0)*1000., 2)})
                # Same candidate set action_labels uses for interval i.
                found = np.flatnonzero((keys_at > t0) & (keys_at < t1) & (keys_at - t0 <= max_lag))
                labels = {ARROWS[keydowns[n]['code']] for n in found}
                if t1 - t0 > max_lag:
                    reason = 'long_interval'
                elif len(found) > 1:
                    reason = 'conflicting_keys' if len(labels) > 1 else 'duplicate_key'
                elif valid[i] and int(actions[i]) == action:
                    reason = 'assigned'
                else:  # Defensive: should be unreachable if rules stay in sync.
                    reason = 'unexplained'
        rec['outcome'] = reason
        outcomes.append(rec)
    eligible = [o for o in outcomes if o['outcome'] not in ('before_first_frame', 'after_last_frame')]
    assigned = [o for o in eligible if o['outcome'] == 'assigned']
    sources = {k.get('source', 'unknown') for k in keydowns}
    # Raw keydowns from the stream trusted_keydowns used, before 45 ms dedup.
    raw = [e for e in events if e.get('type') == 'keydown' and not e.get('repeat', False)
           and canonical_key(e.get('code')) in ARROWS
           and (not sources or e.get('source', 'unknown') in sources)]
    summary = {
        'max_lag_ms': round(max_lag * 1000., 1),
        'trusted_source': ('win32_key_state' if 'win32_key_state' in sources else
                           'browser_dom' if 'browser_dom' in sources else
                           (sorted(sources)[0] if sources else None)),
        'keydowns_total': len(outcomes),
        'merged_by_dedup': len(raw) - len(outcomes),
        'keydowns_eligible': len(eligible),
        'keydowns_assigned': len(assigned),
        'keydown_yield': round(len(assigned) / len(eligible), 4) if eligible else None,
        'outcomes': dict(Counter(o['outcome'] for o in outcomes)),
        'eligible_by_action': dict(Counter(o['action'] for o in eligible)),
        'assigned_by_action': dict(Counter(o['action'] for o in assigned)),
        'assigned_offset_after_frame_ms': _ms_summary([o['offset_after_frame_ms'] for o in assigned]),
    }
    return summary, outcomes


def keyboard_delivery_stats(events, *, window=.5):
    """Pair Win32 physical keydowns with browser keydowns of the same action.

    ``browser_minus_win32_ms`` > 0 means the browser callback arrived later.
    Negative values indicate Win32 poll timestamps lagging the true press.
    """
    def downs(source):
        return sorted((float(e['at']), canonical_key(e.get('code')))
                      for e in events if e.get('source') == source
                      and e.get('type') == 'keydown' and not e.get('repeat', False)
                      and canonical_key(e.get('code')) in ARROWS)
    win, browser = downs('win32_key_state'), downs('browser_dom')
    used = set()
    deltas = []
    for at, code in win:
        best = None
        for n, (bat, bcode) in enumerate(browser):
            if n in used or bcode != code or abs(bat - at) > window:
                continue
            if best is None or abs(bat - at) < abs(browser[best][0] - at):
                best = n
        if best is not None:
            used.add(best)
            deltas.append((browser[best][0] - at) * 1000.)
    return {'win32_keydowns': len(win), 'browser_keydowns': len(browser),
            'matched_pairs': len(deltas),
            'win32_without_browser': len(win) - len(deltas),
            'browser_without_win32': len(browser) - len(deltas),
            'browser_before_win32': int(sum(d < 0 for d in deltas)),
            'browser_minus_win32_ms': _ms_summary(deltas)}


def summarize_capture_timing(timings):
    """Per-stage capture-loop durations (ms) from per-frame timestamp dicts."""
    spans = {
        'focus_check': ('loop_start', 'focus_done'),
        'view_check_and_misc': ('focus_done', 'grab_start'),
        'grab': ('grab_start', 'grab_end'),
        'classify': ('grab_end', 'classify_done'),
        'detector_poll_and_preprocess': ('classify_done', 'preprocess_done'),
        'jpeg_encode': ('preprocess_done', 'jpeg_done'),
        'post_frame': ('jpeg_done', 'iteration_end'),
        'sleep_actual': ('iteration_end', 'sleep_end'),
        'loop_total': ('loop_start', 'sleep_end'),
    }
    out = {}
    for name, (a, b) in spans.items():
        out[name] = _ms_summary([(t[b] - t[a]) * 1000. for t in timings
                                 if t.get(a) is not None and t.get(b) is not None])
    out['sleep_requested'] = _ms_summary([t['sleep_requested'] * 1000. for t in timings
                                          if t.get('sleep_requested') is not None])
    out['sleep_overshoot'] = _ms_summary([
        (t['sleep_end'] - t['iteration_end'] - t['sleep_requested']) * 1000.
        for t in timings if None not in (t.get('sleep_end'), t.get('iteration_end'),
                                         t.get('sleep_requested'))])
    return out


def canvas_monitor_match(region, monitors, *, tolerance=2):
    """Index of the physical monitor the capture region fills, else None.

    PPO crops the full fullscreen monitor; a demo is only pixel-compatible
    when the calibrated canvas covers one entire monitor. Diagnostic only.
    """
    if not region:
        return None
    for index, mon in enumerate(monitors):
        if index == 0 and len(monitors) > 1:
            continue  # MSS monitor 0 is the union of all screens
        if all(abs(int(region[k]) - int(mon[k])) <= tolerance
               for k in ('left', 'top', 'width', 'height')):
            return index
    return None


def capture_stage_line(timings, keyboard_stats=None):
    """One-line median stage breakdown for the periodic [CAPTURE] log."""
    s = summarize_capture_timing(timings)
    med = lambda name: s.get(name, {}).get('median', float('nan'))
    line = (f'[CAPTURE] Stage medians ms: focus {med("focus_check"):.1f} | '
            f'grab {med("grab"):.1f} | classify {med("classify"):.1f} | '
            f'preprocess {med("detector_poll_and_preprocess"):.1f} | '
            f'jpeg {med("jpeg_encode"):.1f} | sleep req {med("sleep_requested"):.1f} '
            f'actual {med("sleep_actual"):.1f} | loop {med("loop_total"):.1f}')
    polls = (keyboard_stats or {}).get('poll_intervals')
    if polls:
        line += f' | key poll {float(np.median(polls.copy()))*1000.:.1f}'
    return line


def print_save_diagnostics(manifest):
    """Short console summary of a saved replay's diagnostics. Never raises:
    it runs after the replay is already on disk."""
    try:
        _print_save_diagnostics(manifest)
    except Exception as exc:
        print(f'[DIAGNOSTICS] summary unavailable: {exc!r}', flush=True)


def _print_save_diagnostics(manifest):
    d = manifest.get('diagnostics') or {}
    if 'error' in d:
        print(f'[DIAGNOSTICS] unavailable: {d["error"]}', flush=True)
        return
    fi, va, kd = d.get('frame_intervals', {}), d.get('validity', {}), d.get('keydowns', {})
    iv = fi.get('intervals_ms', {})
    print(f'[DIAGNOSTICS] frames={fi.get("frames")} interval median={iv.get("median")}ms '
          f'p95={iv.get("p95")}ms within130={fi.get("fraction_intervals_within_max_lag")} | '
          f'valid={va.get("valid_samples")}/{va.get("samples")} '
          f'non-NONE valid={va.get("valid_by_action")} | '
          f'keydowns assigned {kd.get("keydowns_assigned")}/{kd.get("keydowns_eligible")} eligible '
          f'outcomes={kd.get("outcomes")}', flush=True)


def git_state():
    root = Path(__file__).resolve().parent
    try:
        commit = subprocess.run(['git', 'rev-parse', 'HEAD'], cwd=root, capture_output=True,
                                text=True, timeout=5).stdout.strip() or None
        dirty = subprocess.run(['git', 'status', '--porcelain', '--untracked-files=no'],
                               cwd=root, capture_output=True, text=True, timeout=5).stdout.strip()
        return {'commit': commit, 'tracked_changes': bool(dirty)}
    except (OSError, subprocess.SubprocessError):
        return {'commit': None, 'tracked_changes': None}


def _json_default(value):
    if isinstance(value, np.generic):
        return value.item()
    if isinstance(value, (np.ndarray, deque, tuple, set)):
        return list(value)
    if isinstance(value, Path):
        return str(value)
    raise TypeError(f'Not JSON serializable: {type(value).__name__}')


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


def replay_diagnostics(samples, events, actions, valid, secs, segments):
    """Manifest diagnostics + per-keydown outcomes + per-frame timing arrays.

    Never raises: a diagnostics failure is recorded instead of losing a run.
    """
    diagnostics, outcomes, timing_arrays, timing_summary = {}, [], None, None
    try:
        ts = np.asarray([s['at'] for s in samples], dtype=np.float64)
        label_summary, outcomes = label_diagnostics(ts, events)
        diagnostics = {
            'schema': DIAGNOSTICS_SCHEMA,
            'frame_intervals': frame_interval_stats(ts),
            'validity': validity_stats(actions, valid, secs, segments=segments),
            'keydowns': label_summary,
            'keyboard_delivery': keyboard_delivery_stats(events),
        }
        timings = [s.get('timing') or {} for s in samples]
        if any(timings):
            timing_summary = summarize_capture_timing(timings)
            origin = float(ts[0])
            timing_arrays = {
                stage: np.asarray([t[stage] - origin if t.get(stage) is not None else np.nan
                                   for t in timings], dtype=np.float64)
                for stage in TIMING_STAGES}
            timing_arrays['sleep_requested'] = np.asarray(
                [t.get('sleep_requested', np.nan) for t in timings], dtype=np.float64)
    except Exception as exc:  # Never lose a recorded run to a diagnostics bug.
        diagnostics = {'schema': DIAGNOSTICS_SCHEMA, 'error': repr(exc)}
    return diagnostics, outcomes, timing_arrays, timing_summary


def save_replay(samples, events, output, *, status, segments=None, reason='',
                capture_info=None):
    """Write immutable sampled data and a separately editable curation manifest.

    ``capture_info`` (optional) adds recorder/session metadata to the manifest.
    Diagnostic fields are additive; existing manifest keys keep their meaning.
    """
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
    diagnostics, outcomes, timing_arrays, timing_summary = replay_diagnostics(
        samples, events, actions, valid, secs, segments)
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
        # --- Additive diagnostics (recorder 1.3.1); ignored by curate/BC. ---
        'recorder_version': RECORDER_VERSION,
        # timestamps.npy is relative to this perf_counter value; events.jsonl
        # uses absolute perf_counter times. Needed to re-check labels offline.
        'timestamp_origin_perf_counter': float(samples[0]['at']),
        'diagnostics': diagnostics,
    }
    if timing_summary is not None:
        metadata['capture_timing_ms'] = timing_summary
    if capture_info is not None:
        metadata['capture'] = capture_info
    try:
        (output/'label_diagnostics.json').write_text(
            json.dumps({'origin_perf_counter': float(samples[0]['at']), 'keydowns': outcomes},
                       indent=2, default=_json_default)+'\n', encoding='utf-8')
        if timing_arrays is not None:
            np.savez_compressed(output/'capture_timing.npz', **timing_arrays)
    except Exception as exc:  # Diagnostics must never cost a recorded run.
        metadata['diagnostics']['sidecar_error'] = repr(exc)
    (output/'manifest.json').write_text(json.dumps(metadata, indent=2, default=_json_default)+'\n',
                                        encoding='utf-8')
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


# --- Run preservation (autosave), calibration recovery, session logs ------

CALIBRATION_RETRY_SECONDS = 2.0
VIEW_KEYS = ('inner_width', 'inner_height', 'device_pixel_ratio', 'screen_x', 'screen_y',
             'canvas_x', 'canvas_y', 'canvas_width', 'canvas_height')


def unique_run_folder(root, *, now=None):
    """New, unused run folder name under root/runs. Never returns an existing path
    (save_replay's mkdir(exist_ok=False) is the final guard)."""
    runs = Path(root)/'runs'
    base = (now or datetime.now()).strftime('demo_%Y%m%d_%H%M%S_%f')
    folder, n = runs/base, 1
    while folder.exists():
        folder, n = runs/f'{base}_{n}', n + 1
    return folder


class RunRecord:
    """One finished (or force-buffered) gameplay run. Raw samples/events stay in
    memory until the run is safely on disk."""
    def __init__(self, number, samples, events, *, reason, capture_info=None):
        self.number = number
        self.samples = list(samples)
        self.events = list(events)
        self.duration = float(self.samples[-1]['at'] - self.samples[0]['at'])
        self.frames = len(self.samples)
        self.reason = reason
        self.capture_info = capture_info
        self.state = 'queued'  # queued -> saved | failed
        self.folder = None
        self.error = None


class RunStore:
    """Automatically saves every buffered run as a *candidate*; never approves.

    All disk work (autosaves and manual G/S/T/P/R review commands) runs on ONE
    worker thread in submission order, so a review command always applies after
    that run's save has finished: no races, no partially written runs. A failed
    save keeps the run in memory (S retries into a NEW folder). shutdown() waits
    for every queued job and makes a final save attempt for unsaved runs.
    """
    def __init__(self, root, *, save=None, curate_fn=None, opener=None, log=None):
        self.root = Path(root)
        self._save_fn = save or save_replay
        self._curate = curate_fn or curate
        self._opener = opener or (os.startfile if hasattr(os, 'startfile') else None)
        self._log = log or (lambda msg: print(msg, flush=True))
        self._executor = ThreadPoolExecutor(max_workers=1, thread_name_prefix='demo-save')
        self._futures = []
        self.records = []
        self.closed = False

    def complete(self, samples, events, *, reason, capture_info=None):
        if len(samples) < 4:
            self._log(f'[AUTOSAVE] Run has only {len(samples)} frames; too short to save.')
            return None
        if self.closed:
            raise RuntimeError('RunStore already shut down')
        rec = RunRecord(len(self.records) + 1, samples, events, reason=reason,
                        capture_info=capture_info)
        self.records.append(rec)
        self._log(f'[AUTOSAVE] Run {rec.number} ({rec.duration:.2f}s, {rec.frames} frames, '
                  f'{reason}) queued; saving as candidate.')
        self._futures.append(self._executor.submit(self._save_job, rec))
        return rec

    def _save_job(self, rec):
        folder = unique_run_folder(self.root)
        try:
            meta = self._save_fn(rec.samples, rec.events, folder, status='candidate',
                                 reason=rec.reason, capture_info=rec.capture_info)
        except Exception as exc:
            rec.state, rec.error = 'failed', repr(exc)
            left = f' Partial folder left untouched: {folder}' if folder.exists() else ''
            self._log(f'[AUTOSAVE] ERROR: run {rec.number} was NOT saved ({exc!r}). It is still '
                      f'in memory; type S + Enter to retry.{left}')
            return None
        rec.state, rec.folder, rec.error = 'saved', folder, None
        rec.samples = rec.events = None  # safely on disk; release memory
        self._log(f'[AUTOSAVE] Saved candidate run {rec.number}: {folder} | '
                  f'valid labels {meta.get("action_counts_valid")}')
        print_save_diagnostics(meta)
        return folder

    def command(self, cmd, spec=''):
        """Apply G/T (approve), S (save/retry), P (preview), R (reject) to the latest run."""
        rec = self.records[-1] if self.records else None
        if rec is None:
            self._log('No completed run yet. Runs are saved automatically when they finish.')
            return None
        if self.closed:
            self._log('Recorder is shutting down; command ignored.')
            return None
        future = self._executor.submit(self._command_job, rec, cmd, spec)
        self._futures.append(future)
        return future

    def _command_job(self, rec, cmd, spec):
        try:
            if rec.state != 'saved':
                if cmd == 'r':
                    self._log(f'Run {rec.number} is not on disk; it stays in memory. Use S to retry saving.')
                    return None
                if self._save_job(rec) is None:
                    return None
            if cmd == 's':
                self._log(f'Run {rec.number} is saved as candidate: {rec.folder}')
            elif cmd in ('g', 't'):
                m = self._curate(rec.folder, 'approved', spec if cmd == 't' else None)
                self._log(f'Approved run {rec.number}: {rec.folder} | segments {m["valid_segments_seconds"]}')
            elif cmd == 'r':
                self._curate(rec.folder, 'rejected')
                self._log(f'Marked run {rec.number} rejected (files kept): {rec.folder}')
            elif cmd == 'p':
                m = json.loads((rec.folder/'manifest.json').read_text(encoding='utf-8'))
                target = rec.folder/(m.get('preview_video') or 'contact_sheet.jpg')
                if self._opener is None:
                    self._log(f'Preview: {target}')
                else:
                    self._opener(str(target))
            return rec.folder
        except Exception as exc:
            self._log(f'Could not apply {cmd.upper()} to run {rec.number}: {exc}')
            return None

    def pending(self):
        return sum(not f.done() for f in self._futures)

    def shutdown(self):
        """Wait for all queued saves/commands; retry unsaved runs once. Idempotent."""
        if self.closed:
            return
        if self.pending():
            self._log(f'[AUTOSAVE] Waiting for {self.pending()} queued save/review job(s)...')
        self._executor.shutdown(wait=True)
        self.closed = True
        for rec in self.records:
            if rec.state != 'saved' and rec.samples is not None:
                self._log(f'[AUTOSAVE] Final save attempt for run {rec.number}...')
                self._save_job(rec)
        saved = [r for r in self.records if r.state == 'saved']
        unsaved = [r.number for r in self.records if r.state != 'saved']
        self._log(f'[AUTOSAVE] Session summary: {len(saved)} run(s) saved as candidates'
                  + (f'; UNSAVED runs: {unsaved}' if unsaved else '') + '.')


async def calibrate_or_pause(page, canvas, grabber, *, attempts=4, log=None):
    """Calibration that never raises: returns (region, score) or (None, None).

    Pixel matching and its threshold are unchanged; failure only pauses capture.
    """
    log = log or (lambda msg: print(msg, flush=True))
    try:
        return await calibrate_canvas_mss(page, canvas, grabber, attempts=attempts, with_score=True)
    except Exception as exc:
        log(f'[CALIBRATION] PAUSED: {exc}. Capture is paused (completed runs are safe and '
            f'commands still work). Make the game visible; retrying every '
            f'{CALIBRATION_RETRY_SECONDS:.0f}s.')
        return None, None


def view_changes(old, new, keys=VIEW_KEYS):
    """Human-readable differences between two canvas view signatures."""
    if new is None:
        return ['game canvas not visible']
    if old is None:
        return ['no previous signature']
    return [f'{k}: {a} -> {b}' for k, a, b in zip(keys, old, new) if a != b]


def build_capture_snapshot(static_info, *, region, calibration_score, monitors, canvas_view,
                           keyboard_stats, keyboard_alive):
    """Session metadata for a run's manifest. Tolerates paused (uncalibrated) capture."""
    polls = keyboard_stats.get('poll_intervals')
    polls = polls.copy() if polls is not None else ()  # C-level copy; thread-safe
    return {**static_info, 'region': dict(region) if region else None,
            'calibration_score': calibration_score,
            'monitors': [dict(m) for m in monitors],
            'canvas_monitor_index': canvas_monitor_match(region, monitors),
            'view_signature': dict(zip(VIEW_KEYS, canvas_view)) if canvas_view else None,
            'keyboard_poll_ms': _ms_summary([x*1000. for x in polls]),
            'keyboard_requested_interval_ms': keyboard_stats.get('requested_interval_ms'),
            'keyboard_monitor_alive': keyboard_alive,
            'keyboard_monitor_error': keyboard_stats.get('error')}


class TeeStream:
    """Write-through to the console stream and a session log file."""
    def __init__(self, stream, handle, lock):
        self._stream, self._handle, self._lock = stream, handle, lock

    def write(self, text):
        n = self._stream.write(text)
        with self._lock:
            if not self._handle.closed:
                self._handle.write(text)
                self._handle.flush()
        return n

    def flush(self):
        self._stream.flush()

    def fileno(self):
        return self._stream.fileno()

    def isatty(self):
        return self._stream.isatty()

    def __getattr__(self, name):
        return getattr(self._stream, name)


def open_session_log(root, *, now=None):
    """Create a new, never-reused session log under root/logs (exclusive create)."""
    logs = Path(root)/'logs'
    logs.mkdir(parents=True, exist_ok=True)
    base = (now or datetime.now()).strftime('recorder_%Y%m%d_%H%M%S_%f')
    for n in range(1000):
        path = logs/(f'{base}.log' if n == 0 else f'{base}_{n}.log')
        try:
            return path, open(path, 'x', encoding='utf-8', newline='')
        except FileExistsError:
            continue
    raise RuntimeError('Could not create a unique session log')


def run_with_session_log(root, body):
    """Run body() with stdout/stderr also written to a unique session log."""
    import traceback
    path, handle = open_session_log(root)
    lock = threading.Lock()
    old_out, old_err = sys.stdout, sys.stderr
    sys.stdout, sys.stderr = TeeStream(old_out, handle, lock), TeeStream(old_err, handle, lock)
    print(f'[LOG] Session log: {path}', flush=True)
    try:
        return body()
    except BaseException:
        with lock:  # the interpreter prints the traceback to the console afterwards
            handle.write(traceback.format_exc())
        raise
    finally:
        sys.stdout, sys.stderr = old_out, old_err
        with lock:
            handle.close()
        print(f'[LOG] Session log saved: {path}', flush=True)


def console_reader(commands, stop):
    print('Runs are AUTO-SAVED as candidates when they finish. Commands (type letter + Enter, '
          'applies to the latest run): G=approve, T=trim+approve, S=save/retry, '
          'P=preview, R=mark rejected (files kept), Q=quit', flush=True)
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
        elif cmd == 'q':
            commands.put((cmd, ''))
            return  # stop reading stdin so interpreter shutdown is clean
        elif cmd in {'g','s','p','r'}:
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
    keyboard_stats = {}  # diagnostics only: measured poll period / thread error
    keyboard_thread = threading.Thread(target=monitor_windows_arrows,
                                       args=(events,stop), kwargs={'stats': keyboard_stats},
                                       daemon=True)
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
        store = RunStore(a.root)
        with mss.mss() as grabber:
            print('[CAPTURE] Calibrating canvas against physical desktop once. '
                  'There are NO Chromium screenshots during normal gameplay.',
                  flush=True)
            region, calibration_score = await calibrate_or_pause(page, canvas, grabber)
            next_calibration_at = time.perf_counter() + CALIBRATION_RETRY_SECONDS
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

            async def read_view():
                """Signature, or None while the canvas is not visible (never raises)."""
                try:
                    return await canvas_view_signature()
                except Exception:
                    return None

            canvas_view = await read_view()
            static_info = {'recorder_version': RECORDER_VERSION, 'git': git_state(),
                           'cli_args': {k: str(v) for k, v in vars(a).items()},
                           'platform': platform.platform(),
                           'python': sys.version.split()[0], 'opencv': cv2.__version__}

            def capture_snapshot():
                """Session metadata at save time (diagnostics only)."""
                return build_capture_snapshot(
                    static_info, region=region, calibration_score=calibration_score,
                    monitors=grabber.monitors, canvas_view=canvas_view,
                    keyboard_stats=keyboard_stats, keyboard_alive=keyboard_thread.is_alive())

            def buffer_run(reason, cutoff):
                """Hand the current run to the autosaving store (never discarded)."""
                if mode == 'recording' and len(samples) >= 4:
                    run_events = [e for e in events[start_event_index:] if e['at'] <= cutoff]
                    store.complete(samples, run_events, reason=reason,
                                   capture_info=capture_snapshot())

            recent_timings = deque(maxlen=80)
            last_view_check = time.perf_counter()
            mode, streak, samples, start_event_index = 'idle', 0, [], 0
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
                    # Diagnostics only: per-iteration stage timestamps (TIMING_STAGES).
                    timing = {'loop_start': time.perf_counter()}
                    # Commands can be entered after Alt+Tab without capturing terminal pixels.
                    try:
                        cmd, spec = commands.get_nowait()
                    except queue.Empty:
                        cmd = None
                    if cmd == 'q':
                        break
                    if cmd:
                        # Serialized after the latest run's autosave on the store's worker.
                        store.command(cmd, spec)
                    # Paused (uncalibrated) capture: keep serving commands, retry calibration.
                    if region is None:
                        if time.perf_counter() >= next_calibration_at:
                            region, calibration_score = await calibrate_or_pause(
                                page, canvas, grabber, attempts=1, log=lambda _m: None)
                            next_calibration_at = time.perf_counter() + CALIBRATION_RETRY_SECONDS
                            if region is not None:
                                canvas_view = await read_view()
                                last_view_check = time.perf_counter()
                                first_capture_written, last_captured = False, None
                                capture_intervals.clear()
                                print('[CALIBRATION] Capture resumed.', flush=True)
                        if region is None:
                            await asyncio.sleep(.1)
                            next_tick = time.perf_counter()+interval
                            continue
                    # Never sample the terminal once browser has lost focus. Menu is already frozen.
                    try:
                        focused = await page.evaluate('document.hasFocus() && document.visibilityState === "visible"')
                    except Exception:
                        focused = False
                    timing['focus_done'] = time.perf_counter()
                    if not focused:
                        if mode == 'recording':
                            print(f'[RUN {episode}] Browser lost focus: run buffered for autosave.', flush=True)
                            buffer_run('focus_lost_partial', samples[-1]['at'] if samples else 0.)
                            mode, samples = 'idle', []
                        await asyncio.sleep(.08)
                        next_tick = time.perf_counter()+interval
                        continue
                    # Only check for window move/fullscreen transitions about
                    # once a second; the heavy screenshot happens only on change.
                    if time.perf_counter()-last_view_check >= 1.0:
                        current_view = await read_view()
                        last_view_check = time.perf_counter()
                        if current_view != canvas_view:
                            print('[CAPTURE] Browser moved/resized/fullscreen changed: '
                                  + '; '.join(view_changes(canvas_view, current_view))
                                  + '. Recalibrating.', flush=True)
                            # Frames before the change share one geometry: keep them as a
                            # candidate. Earlier completed runs are already with the store.
                            if mode == 'recording':
                                buffer_run('geometry_changed_partial',
                                           samples[-1]['at'] if samples else 0.)
                            mode,streak,samples = 'idle',0,[]
                            last_detection,future,request_id = None,None,0
                            region, calibration_score = await calibrate_or_pause(page, canvas, grabber)
                            next_calibration_at = time.perf_counter() + CALIBRATION_RETRY_SECONDS
                            canvas_view = await read_view() if region is not None else current_view
                            first_capture_written=False
                            last_captured=None
                            capture_intervals.clear()
                            next_tick=time.perf_counter()+interval
                            continue
                    timing['grab_start'] = time.perf_counter()
                    shot = np.asarray(grabber.grab(region))
                    captured_at = time.perf_counter()
                    timing['grab_end'] = captured_at
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
                        if recent_timings:
                            print(capture_stage_line(list(recent_timings), keyboard_stats), flush=True)
                        last_rate_log=captured_at
                    if not first_capture_written:
                        preview_file = a.root/'capture_preview_full_canvas.png'
                        if not cv2.imwrite(str(preview_file), shot):
                            raise RuntimeError('Could not save initial canvas preview')
                        first_capture_written = True
                        print(f'[CAPTURE] Raw canvas {shot.shape[1]}x{shot.shape[0]}; '
                              f'preview saved: {preview_file}', flush=True)
                    state, probs = states.state(shot, confidence=a.state_confidence)
                    timing['classify_done'] = time.perf_counter()
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
                                    print(f'[RUN {episode}] Completed '
                                          f'{samples[-1]["at"]-samples[0]["at"]:.2f}s, {len(samples)} frames.',
                                          flush=True)
                                buffer_run('completed', menu_first)
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
                        timing['preprocess_done'] = time.perf_counter()
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
                        timing['jpeg_done'] = time.perf_counter()
                        samples.append({'at':captured_at, 'frame':frame,
                                        'canvas_jpeg':encoded.tobytes(), 'detection':last_detection,
                                        'timing':timing})
                        recent_timings.append(timing)
                        if len(samples)%a.detect_every == 1 and future is None:
                            future = pool.submit(detect_v5, detector, native.copy(), captured_at,
                                                 episode, request_id, a.imgsz, a.conf)
                            request_id += 1
                        if len(samples) >= a.fps*a.max_run_seconds:
                            # Max-length run: store as complete but only pre-stop frames.
                            print('Max run length reached; saving. Stop/restart game manually.', flush=True)
                            buffer_run('max_run_length', captured_at)
                            mode, samples = 'idle', []
                    # Yield for browser focus/visibility monitoring.
                    next_tick += interval
                    if next_tick < time.perf_counter()-.5:
                        next_tick = time.perf_counter()
                    delay = max(.002, next_tick-time.perf_counter())
                    timing['iteration_end'] = time.perf_counter()
                    timing['sleep_requested'] = delay
                    await asyncio.sleep(delay)
                    # The same dict is referenced by this iteration's sample, if any.
                    timing['sleep_end'] = time.perf_counter()
            finally:
                stop.set()
                # Queued saves must finish before exit, even after an error.
                store.shutdown()
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
        run_with_session_log(a.root, lambda: asyncio.run(capture_loop(a)))
    else:
        print(json.dumps(curate(a.folder, a.status, a.segments), indent=2))


if __name__ == '__main__':
    main()

