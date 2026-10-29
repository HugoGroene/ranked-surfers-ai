"""Build a local, read-only video review queue from one recorder session log.

Writes only to a new report directory. Suggestions never approve or edit replays.
"""
import argparse
import hashlib
import json
import os
from pathlib import Path
import re
from urllib.parse import quote

import numpy as np

from tools.analyze_demo_run import analyze_run

NAMES = ['NONE', 'LEFT', 'RIGHT', 'JUMP', 'ROLL']


def frame_window(times, valid, actions, start, end):
    """Snap half-open bounds and validate targets AND all four input frames."""
    times = np.asarray(times, dtype=float)
    if (len(times) < 4 or not np.isfinite(times).all() or
            np.any(np.diff(times) <= 0) or not 0 <= start < end <= times[-1]):
        return None
    lo = int(np.searchsorted(times, start, side='left'))
    hi = int(np.searchsorted(times, end, side='right')) - 1
    if lo < 3 or hi <= lo:
        return None
    ids = np.arange(lo, hi)
    ids = ids[np.asarray(valid, dtype=bool)[ids]]
    if not len(ids):
        return None
    for i in ids:
        if np.any(np.diff(times[i-3:i+1]) > .130):
            return None
    counts = np.bincount(np.asarray(actions)[ids], minlength=5)
    if not counts[1:].sum():
        return None
    return {'start': float(times[lo]), 'end': float(times[hi]),
            'valid_samples': len(ids),
            'action_counts': dict(zip(NAMES, map(int, counts)))}


def suggested_window(times, valid, actions, *, opening=2.15, tail=2.5):
    """Conservative review suggestion, stopping before the first interior gap."""
    times = np.asarray(times)
    if not len(times):
        return None
    end = float(times[-1]) - tail
    for i in np.flatnonzero(np.diff(times) > .130) + 1:
        if times[i] >= opening and times[i-1] < end:
            end = float(times[i-1])
            break
    if end <= opening:
        return None
    return frame_window(times, valid, actions, opening, end)


def session_runs(log):
    matches = re.findall(
        r'\[AUTOSAVE\] Saved candidate run (\d+): .*?(demo_\d{8}_\d{6}_\d{6})', log)
    if not matches or len({name for _, name in matches}) != len(matches):
        raise ValueError('Session has no unique saved-run list')
    return [(int(number), name) for number, name in matches]


def build(log_path, root, output):
    if output.resolve().is_relative_to(root.resolve()):
        raise ValueError('Review output must be outside replay root')
    output.mkdir(parents=True, exist_ok=False)
    raw_log = log_path.read_bytes()
    entries = session_runs(raw_log.decode('utf8', errors='replace'))
    records, audits = [], []

    def relative(path):
        return quote(os.path.relpath(path.resolve(), output.resolve()).replace('\\', '/'), safe='/.:')

    for number, name in entries:
        folder = root/'runs'/name
        manifest_bytes = (folder/'manifest.json').read_bytes()
        manifest = json.loads(manifest_bytes)
        audit = analyze_run(folder, lateral=10000, skip_preview=False)
        audits.append(audit)
        times = np.load(folder/'timestamps.npy', allow_pickle=False)
        actions = np.load(folder/'actions.npy', allow_pickle=False)
        valid = np.load(folder/'valid.npy', allow_pickle=False)
        if not len(times) == len(actions) == len(valid):
            raise ValueError(f'Array lengths disagree: {name}')
        if not np.isfinite(times).all() or np.any(np.diff(times) <= 0):
            raise ValueError(f'Invalid timestamps: {name}')
        gaps = [{'frame': int(i), 'seconds': float(times[i]),
                 'ms': round(float(times[i]-times[i-1])*1000, 2)}
                for i in np.flatnonzero(np.diff(times) > .130)+1]
        flags = []
        blockers = []
        if audit.get('labels_reproducible') is not True:
            blockers.append('Saved labels do not reproduce')
        if not audit['preview'].get('available'):
            blockers.append('Missing preview video')
        elif audit['preview'].get('frames') != len(times):
            blockers.append('Preview/frame count mismatch')
        if gaps:
            flags.append(f'{len(gaps)} capture gap(s) >130 ms')
        if audit['validity']['valid_fraction'] < .95:
            flags.append('Less than 95% valid labels')
        if audit['keyboard_delivery']['win32_without_browser']:
            flags.append('Some physical keydowns lack browser matches')
        if abs(audit['preview'].get('speed_ratio', 1)-1) > .03:
            flags.append('Video time differs from recorded time; use mapped timestamps')
        suggestion = suggested_window(times, valid, actions)
        if suggestion is None:
            flags.append('No useful automatic prefix; inspect/edit or skip')
        video = folder/manifest.get('preview_video', 'preview.mp4')
        records.append({
            'number': number, 'name': name, 'duration': float(times[-1]),
            'status': manifest['status'],
            'manifest_sha256': hashlib.sha256(manifest_bytes).hexdigest(),
            'video': relative(video), 'sheet': relative(folder/'contact_sheet.jpg'),
            'video_fps': audit['preview'].get('fps', 20),
            'times': times.tolist(), 'actions': actions.tolist(),
            'valid': np.asarray(valid, dtype=int).tolist(),
            'gaps': gaps, 'flags': flags, 'blockers': blockers,
            'suggestion': suggestion,
            'events': [{'frame': int(i), 'seconds': float(times[i]),
                        'action': NAMES[int(actions[i])]} for i in np.flatnonzero(actions)],
            'raw_counts': audit['validity']['valid_by_action'],
            'valid_fraction': audit['validity']['valid_fraction']})
        if number % 25 == 0:
            print(f'[REVIEW] Audited {number}/{len(entries)} runs', flush=True)
    identity = hashlib.sha256(raw_log + json.dumps(
        [r['manifest_sha256'] for r in records]).encode()).hexdigest()
    payload = {'version': 1, 'dataset_id': identity, 'session_log': str(log_path),
               'source_root': str(root), 'runs': records,
               'suggestion_note': '2.15s leading trim, 2.5s trailing trim, stop before first interior gap. NOT visually verified.'}
    (output/'audit.json').write_text(json.dumps(audits, indent=2, default=str), encoding='utf8')
    (output/'queue.json').write_text(json.dumps(payload), encoding='utf8')
    template = (Path(__file__).with_name('demo_review_template.html')).read_text(encoding='utf8')
    encoded = json.dumps(payload).replace('<', '\\u003c')
    (output/'review.html').write_text(template.replace('__REVIEW_DATA__', encoded), encoding='utf8')
    summary = {'runs': len(records), 'recorded_minutes': sum(r['duration'] for r in records)/60,
               'short_under_10s': sum(r['duration'] < 10 for r in records),
               'medium_10_to_20s': sum(10 <= r['duration'] < 20 for r in records),
               'long_20s_plus': sum(r['duration'] >= 20 for r in records),
               'label_mismatches': sum('Saved labels do not reproduce' in r['blockers'] for r in records),
               'runs_with_gaps': sum(bool(r['gaps']) for r in records),
               'suggested_useful_prefixes': sum(r['suggestion'] is not None for r in records),
               'suggested_actions': {key: sum((r['suggestion'] or {}).get('action_counts', {}).get(key, 0)
                                               for r in records) for key in NAMES}}
    (output/'summary.json').write_text(json.dumps(summary, indent=2), encoding='utf8')
    print(json.dumps(summary, indent=2), flush=True)
    print('REVIEW_PAGE:', output/'review.html', flush=True)
    return payload


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--session-log', type=Path, required=True)
    parser.add_argument('--root', type=Path, default=Path('ai_data/demonstrations_v13'))
    parser.add_argument('--output', type=Path, required=True)
    args = parser.parse_args()
    build(args.session_log, args.root, args.output)
