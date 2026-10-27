#!/usr/bin/env python3
"""Read-only analysis of saved human-demonstration replays (V1.3 and later).

Never writes into a run folder. Reads only manifest.json, timestamps.npy,
actions.npy, valid.npy, events.jsonl and, when present, capture_timing.npz and
the preview video (frame count/fps only). Labels are RECOMPUTED from raw
timestamps + keyboard events with the recorder's own rules, then compared with
the saved actions/valid arrays.

Older V1.3 replays did not store the absolute time origin of timestamps.npy;
it is recovered by searching for offsets that reproduce the saved labels.

Usage (from the repository root):
  python tools/analyze_demo_run.py RUN_DIR [RUN_DIR ...]
  python tools/analyze_demo_run.py --latest ai_data/demonstrations_v13 --count 3
  python tools/analyze_demo_run.py RUN_DIR --json report.json
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from rs_v612_demo import (ACTION_NAMES, ARROWS, _json_default, action_labels,  # noqa: E402
                          frame_interval_stats, keyboard_delivery_stats, label_diagnostics,
                          summarize_capture_timing, trusted_keydowns, validity_stats,
                          TIMING_STAGES)

MAX_LAG = .13
# Pass/fail thresholds for a trustworthy recording (see CLAUDE.md / plan).
CRITERIA = {
    'median_interval_ms_max': 55.,
    'p95_interval_ms_max': 100.,
    'fraction_within_130ms_min': .95,
    'valid_fraction_min': .90,
    'keydown_yield_min': .90,
    'preview_duration_tolerance': .03,
}


def load_run(folder):
    folder = Path(folder)
    manifest = json.loads((folder/'manifest.json').read_text(encoding='utf-8'))
    events_path = folder/'events.jsonl'
    events = ([json.loads(line) for line in events_path.read_text(encoding='utf-8').splitlines()
               if line.strip()] if events_path.exists() else [])
    return {
        'folder': folder, 'manifest': manifest, 'events': events,
        'seconds': np.load(folder/'timestamps.npy', allow_pickle=False).astype(np.float64),
        'actions': np.load(folder/'actions.npy', allow_pickle=False),
        'valid': np.load(folder/'valid.npy', allow_pickle=False).astype(bool),
    }


def recover_origin(seconds, events, actions, *, max_lag=MAX_LAG, step=.001):
    """Absolute perf_counter origin of relative timestamps, from saved labels.

    Returns (origin, info). origin is None if no label constrains it or the
    feasible offsets are ambiguous. Every candidate is verified by fully
    re-running action_labels and requiring an exact match with saved labels.
    """
    seconds = np.asarray(seconds, dtype=float)
    actions = np.asarray(actions)
    labeled = np.flatnonzero(actions != 0)
    keys = trusted_keydowns(events)
    if not len(labeled):
        return None, {'method': 'unrecoverable', 'reason': 'no saved non-NONE labels to anchor the origin'}
    i0 = int(labeled[0])
    target = int(actions[i0])
    if i0 + 1 >= len(seconds):
        return None, {'method': 'unrecoverable', 'reason': 'anchor label on final frame'}
    span = min(float(seconds[i0+1] - seconds[i0]), max_lag)
    feasible = []
    for k in keys:
        if ARROWS[k['code']] != target:
            continue
        at = float(k['at'])
        # Key must fall in (t0, t0+span] of the anchor frame: origin in [at-ts-span, at-ts).
        for origin in np.arange(at - seconds[i0] - span, at - seconds[i0], step):
            acts, _ = action_labels(seconds + origin, events, max_lag=max_lag)
            if np.array_equal(acts, actions):
                feasible.append(float(origin))
    if not feasible:
        return None, {'method': 'unrecoverable', 'reason': 'no offset reproduces saved labels'}
    feasible.sort()
    clusters = int(np.count_nonzero(np.diff(feasible) > 2*step)) + 1
    if clusters > 1:
        return None, {'method': 'ambiguous', 'clusters': clusters,
                      'range': [feasible[0], feasible[-1]]}
    return (feasible[0] + feasible[-1]) / 2, {
        'method': 'recovered_from_labels', 'candidates': len(feasible),
        'uncertainty_ms': round((feasible[-1] - feasible[0]) * 1000. / 2, 1)}


def preview_check(folder, manifest, seconds):
    name = manifest.get('preview_video')
    if not name or not (folder/name).exists():
        return {'available': False}
    import cv2
    cap = cv2.VideoCapture(str(folder/name))
    try:
        frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        fps = float(cap.get(cv2.CAP_PROP_FPS))
        width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    finally:
        cap.release()
    out = {'available': True, 'file': name, 'frames': frames, 'fps': fps,
           'width': width, 'height': height}
    n = len(seconds)
    if frames > 0 and fps > 0 and n > 1:
        real = float(seconds[-1] - seconds[0]) * n / (n - 1)  # include one mean interval
        out['playback_seconds'] = round(frames / fps, 3)
        out['real_seconds'] = round(real, 3)
        out['speed_ratio'] = round((frames / fps) / real, 4) if real > 0 else None
    shape = manifest.get('preview_canvas_original_shape')
    if shape and width and height:
        out['aspect_error'] = round(abs((width/height) - (shape[1]/shape[0])) / (shape[1]/shape[0]), 4)
    return out


def analyze_run(folder, *, max_lag=MAX_LAG, lateral=20, skip_preview=False):
    run = load_run(folder)
    m, secs, actions, valid, events = (run['manifest'], run['seconds'], run['actions'],
                                       run['valid'], run['events'])
    report = {
        'run': str(run['folder']), 'recorder_version': m.get('recorder_version', '1.3 (pre-diagnostics)'),
        'status': m.get('status'), 'duration_seconds': m.get('duration_seconds'),
        'frames': int(len(secs)),
        'frame_intervals': frame_interval_stats(secs, max_lag=max_lag),
        'validity': validity_stats(actions, valid, secs, max_lag=max_lag,
                                   segments=m.get('valid_segments_seconds')),
        'keyboard_delivery': keyboard_delivery_stats(events),
        'keyboard_sources': dict(m.get('keyboard_sources', {})),
    }
    origin = m.get('timestamp_origin_perf_counter')
    if origin is not None:
        report['origin'] = {'method': 'manifest', 'value': float(origin)}
    else:
        origin, info = recover_origin(secs, events, actions, max_lag=max_lag)
        report['origin'] = {**info, 'value': origin}
    if origin is not None:
        times = secs + float(origin)
        summary, outcomes = label_diagnostics(times, events, max_lag=max_lag)
        report['keydowns'] = summary
        acts, val = action_labels(times, events, max_lag=max_lag)
        report['labels_reproducible'] = bool(np.array_equal(acts, actions) and np.array_equal(val, valid))
        lat = [o for o in outcomes if o['action'] in ('LEFT', 'RIGHT')]
        report['lateral_review'] = {
            'assigned': [{k: o[k] for k in ('seconds', 'frame', 'action', 'offset_after_frame_ms')}
                         for o in lat if o['outcome'] == 'assigned'][:lateral],
            'not_assigned': [{k: o[k] for k in ('seconds', 'frame', 'action', 'outcome', 'interval_ms')}
                             for o in lat if o['outcome'] != 'assigned'][:lateral],
        }
    capture = m.get('capture') or {}
    if capture:
        report['capture'] = {k: capture.get(k) for k in (
            'region', 'calibration_score', 'canvas_monitor_index', 'view_signature',
            'keyboard_poll_ms', 'keyboard_requested_interval_ms',
            'keyboard_monitor_alive', 'keyboard_monitor_error', 'git')}
    timing_file = run['folder']/'capture_timing.npz'
    if timing_file.exists():
        with np.load(timing_file, allow_pickle=False) as z:
            stages = {k: z[k] for k in z.files}
        rows = []
        for i in range(len(secs)):
            row = {s: (float(stages[s][i]) if s in stages and np.isfinite(stages[s][i]) else None)
                   for s in TIMING_STAGES}
            req = stages.get('sleep_requested')
            row['sleep_requested'] = float(req[i]) if req is not None and np.isfinite(req[i]) else None
            rows.append(row)
        report['capture_timing_ms'] = summarize_capture_timing(rows)
    elif m.get('capture_timing_ms'):
        report['capture_timing_ms'] = m['capture_timing_ms']
    report['preview'] = ({'available': False, 'skipped': True} if skip_preview
                         else preview_check(run['folder'], m, secs))
    report['criteria'] = evaluate(report)
    return report


def _crit(name, value, ok, threshold):
    status = 'N/A' if value is None else ('PASS' if ok else 'FAIL')
    return {'criterion': name, 'status': status, 'value': value, 'threshold': threshold}


def evaluate(report):
    c = CRITERIA
    fi = report['frame_intervals']
    iv = fi.get('intervals_ms', {})
    med, p95 = iv.get('median'), iv.get('p95')
    within = fi.get('fraction_intervals_within_max_lag')
    vf = report['validity'].get('valid_fraction')
    kd = report.get('keydowns') or {}
    yld = kd.get('keydown_yield')
    out = [
        _crit('median frame interval (ms)', med, med is not None and med <= c['median_interval_ms_max'],
              f"<= {c['median_interval_ms_max']:.0f}"),
        _crit('p95 frame interval (ms)', p95, p95 is not None and p95 <= c['p95_interval_ms_max'],
              f"<= {c['p95_interval_ms_max']:.0f}"),
        _crit('intervals within 130 ms', within,
              within is not None and within >= c['fraction_within_130ms_min'],
              f">= {c['fraction_within_130ms_min']:.0%}"),
        _crit('valid sample fraction', vf, vf is not None and vf >= c['valid_fraction_min'],
              f">= {c['valid_fraction_min']:.0%}"),
        _crit('keydown yield (eligible gameplay keydowns)', yld,
              yld is not None and yld >= c['keydown_yield_min'], f">= {c['keydown_yield_min']:.0%}"),
        _crit('saved labels reproducible from raw events', report.get('labels_reproducible'),
              report.get('labels_reproducible') is True, 'exact match'),
    ]
    cap = report.get('capture') or {}
    alive = cap.get('keyboard_monitor_alive') if cap else None
    out.append(_crit('keyboard monitor alive, no error', alive,
                     alive is True and not cap.get('keyboard_monitor_error'), 'true'))
    # Informational only: PPO's capture geometry varies with how fullscreen was
    # detected (full monitor incl. browser/taskbar vs canvas only), so canvas
    # size alone cannot establish PPO compatibility. Never pass/fail on it.
    region = cap.get('region') if cap else None
    out.append({'criterion': 'capture geometry (recorder captures canvas only)',
                'status': 'INFO' if cap else 'N/A',
                'value': ({'region': region, 'canvas_fills_monitor': cap.get('canvas_monitor_index') is not None}
                          if cap else None),
                'threshold': 'compare with PPO capture before BC'})
    ratio = report['preview'].get('speed_ratio')
    out.append(_crit('preview playback speed ratio', ratio,
                     ratio is not None and abs(ratio - 1) <= c['preview_duration_tolerance'],
                     f"1 +/- {c['preview_duration_tolerance']:.0%}"))
    for manual in ('label alignment: lateral keys in lateral_review start moving at/after labeled frame',
                   'preview shows the full, unobstructed canvas at plausible speed',
                   'approved segments exclude mistakes and the death animation'):
        out.append({'criterion': manual, 'status': 'MANUAL', 'value': None, 'threshold': 'human review'})
    return out


def format_report(r):
    lines = [f"=== {r['run']}", f"recorder {r['recorder_version']} | status {r['status']} | "
             f"{r['frames']} frames | {r['duration_seconds']} s"]
    iv = r['frame_intervals'].get('intervals_ms', {})
    lines.append(f"intervals ms: median {iv.get('median')} p95 {iv.get('p95')} max {iv.get('max')} | "
                 f"within 130 ms {r['frame_intervals'].get('fraction_intervals_within_max_lag')} | "
                 f">1 s gaps {r['frame_intervals'].get('intervals_over_1s')}")
    v = r['validity']
    lines.append(f"valid {v['valid_samples']}/{v['samples']} ({v['valid_fraction']}) | valid by action "
                 f"{v['valid_by_action']} | stack-gap valid {v['valid_with_stack_gap']}")
    if 'valid_in_segments_by_action' in v:
        lines.append(f"valid in approved segments {v['segments_seconds']}: {v['valid_in_segments_by_action']}")
    o = r['origin']
    lines.append(f"time origin: {o.get('method')} {('+/-%s ms' % o['uncertainty_ms']) if 'uncertainty_ms' in o else ''}"
                 f"{(' - ' + o['reason']) if 'reason' in o else ''}")
    if 'keydowns' in r:
        k = r['keydowns']
        lines.append(f"keydowns: {k['keydowns_assigned']}/{k['keydowns_eligible']} eligible assigned "
                     f"(yield {k['keydown_yield']}) | total {k['keydowns_total']} | source {k['trusted_source']}")
        lines.append(f"  outcomes {k['outcomes']} | eligible by action {k['eligible_by_action']}")
        lines.append(f"  labels reproducible: {r['labels_reproducible']}")
    d = r['keyboard_delivery']
    lines.append(f"keyboard delivery: win32 {d['win32_keydowns']} browser {d['browser_keydowns']} "
                 f"matched {d['matched_pairs']} | browser-win32 ms {d['browser_minus_win32_ms']}")
    if 'capture' in r:
        c = r['capture']
        lines.append(f"capture: region {c.get('region')} monitor {c.get('canvas_monitor_index')} "
                     f"calibration {c.get('calibration_score')} | key poll ms {c.get('keyboard_poll_ms')}")
    if 'capture_timing_ms' in r:
        t = r['capture_timing_ms']
        lines.append('stage medians ms: ' + ', '.join(
            f"{name} {t[name].get('median')}" for name in t if t[name].get('count')))
    p = r['preview']
    if p.get('available'):
        lines.append(f"preview: {p['file']} {p['width']}x{p['height']} {p['frames']} frames @ {p['fps']} fps | "
                     f"speed ratio {p.get('speed_ratio')} | aspect error {p.get('aspect_error')}")
    lr = r.get('lateral_review')
    if lr:
        lines.append('lateral keys to check in preview (seconds, frame):')
        for x in lr['assigned']:
            lines.append(f"  {x['action']:5s} at {x['seconds']:.2f}s -> frame {x['frame']} "
                         f"(+{x['offset_after_frame_ms']} ms)")
        for x in lr['not_assigned']:
            lines.append(f"  {x['action']:5s} at {x['seconds']}s NOT assigned: {x['outcome']}")
    lines.append('criteria:')
    for c in r['criteria']:
        lines.append(f"  [{c['status']:6s}] {c['criterion']}: {c['value']} (target {c['threshold']})")
    return '\n'.join(lines)


def main(argv=None):
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument('runs', nargs='*', type=Path, help='Replay folders containing manifest.json')
    p.add_argument('--latest', type=Path, default=None, help='Demonstration root; analyze newest runs')
    p.add_argument('--count', type=int, default=1, help='Number of newest runs with --latest')
    p.add_argument('--lateral', type=int, default=20, help='Max LEFT/RIGHT keys listed for review')
    p.add_argument('--json', type=Path, default=None, help='Optional report path (never inside a run folder)')
    p.add_argument('--skip-preview', action='store_true', help='Do not open the preview video')
    a = p.parse_args(argv)
    runs = list(a.runs)
    if a.latest is not None:
        found = sorted((d for d in (a.latest/'runs').glob('demo_*') if (d/'manifest.json').exists()),
                       key=lambda d: d.name)
        runs += found[-a.count:]
        if not found:
            p.error(f'No saved replays found in {a.latest/"runs"} (nothing was saved with S/G/T/P yet)')
    if not runs:
        p.error('Give run folders or --latest ROOT')
    if a.json is not None:
        target = a.json.resolve()
        if any(target.is_relative_to(Path(r).resolve()) for r in runs):
            p.error('Refusing to write a report inside a replay folder; runs stay immutable')
    reports = [analyze_run(r, lateral=a.lateral, skip_preview=a.skip_preview) for r in runs]
    for r in reports:
        print(format_report(r))
        print()
    if a.json is not None:
        a.json.write_text(json.dumps(reports, indent=2, default=_json_default)+'\n', encoding='utf-8')
        print(f'Report written: {a.json}')
    return reports


if __name__ == '__main__':
    main()
