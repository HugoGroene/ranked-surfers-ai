"""Offline curation and adjustable pixel export. Does not train or start a browser."""
import argparse
from datetime import datetime, timezone
import hashlib
import html
import json
from pathlib import Path
import shutil
import sys

import cv2
import numpy as np

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[2]
sys.path.insert(0, str(ROOT))
sys.path.append(str(ROOT/'development/experiments/road_vision'))
CROPS = {'full': (0., 0., 1., 1.), 'road_wide': (.15, 0., .85, 1.),
         'road_center': (.25, 0., .75, 1.)}


def read(path):
    return json.loads(Path(path).read_text(encoding='utf-8'))


def digest(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def write_new(path, data):
    with Path(path).open('x', encoding='utf-8') as f:
        json.dump(data, f, indent=2, allow_nan=False)


def arrays(folder):
    times = np.load(folder/'timestamps.npy', allow_pickle=False)
    labels = np.load(folder/'actions.npy', allow_pickle=False)
    valid = np.load(folder/'valid.npy', allow_pickle=False)
    if (times.ndim != 1 or len(times) < 4 or not np.isfinite(times).all()
            or not np.all(np.diff(times) > 0) or labels.shape != times.shape
            or valid.shape != times.shape or not np.isin(labels, range(5)).all()
            or not np.isin(valid, [0, 1]).all()):
        raise ValueError('Invalid timestamps, labels or validity mask')
    return times, labels, valid.astype(bool)


def segments(text, duration):
    result = []
    for part in text.split(','):
        start, end = map(float, part.split(':'))
        if not (np.isfinite([start, end]).all() and 0 <= start < end <= duration
                and (not result or start > result[-1][1])):
            raise ValueError('Use ordered disjoint start:end intervals within the run')
        result.append([start, end])
    return result


def segment_ids(times, intervals):
    result = np.full(len(times), -1, dtype=np.int32)
    for i, (start, end) in enumerate(intervals):
        result[(times >= start) & (times <= end)] = i
    return result


def review(folder, *, status, intervals=None, reviewed=False, note=''):
    folder = Path(folder)
    times, labels, valid = arrays(folder)
    chosen = segments(intervals, float(times[-1])) if intervals else []
    if status == 'approved':
        if not reviewed or not chosen:
            raise ValueError('Approval requires --reviewed and explicit --segments after visual inspection')
        if not np.any(valid & (segment_ids(times, chosen) >= 0) & (labels != 0)):
            raise ValueError('Approved intervals contain no valid physical action labels')
    revisions = folder/'reviews'
    revisions.mkdir(exist_ok=True)
    name = datetime.now(timezone.utc).strftime('%Y%m%dT%H%M%S_%fZ.json')
    data = {'schema': 1, 'status': status, 'visually_reviewed': reviewed,
            'segments_seconds': chosen, 'note': note, 'manifest_sha256': digest(folder/'manifest.json'),
            'label_sha256': {key: digest(folder/key) for key in
                             ('timestamps.npy', 'actions.npy', 'valid.npy', 'aux.npy', 'events.jsonl')},
            'created_utc': name[:-5]}
    write_new(revisions/name, data)
    return revisions/name


def latest_review(folder):
    paths = sorted((Path(folder)/'reviews').glob('*.json'))
    return (read(paths[-1]), paths[-1]) if paths else ({'status': 'candidate'}, None)


def source_index(folder):
    metadata = read(folder/'manifest.json')
    archive = metadata.get('source_archive', {})
    if archive.get('version') != 1 or archive.get('index') != 'source_frames/index.json':
        raise ValueError('Requires a staged native recording; legacy JPEG previews are not masters')
    path = folder/'source_frames/index.json'
    if digest(path) != archive['index_sha256']:
        raise ValueError('Source index changed')
    index = read(path)
    times, _, _ = arrays(folder)
    if len(index) != len(times) or archive['count'] != len(times):
        raise ValueError('Source/label frame count mismatch')
    stamps = np.asarray([item['captured_at_perf_counter'] for item in index])
    if not np.allclose(stamps-stamps[0], times, rtol=0, atol=1e-6):
        raise ValueError('Source/label timestamp mismatch')
    return metadata, index


def source_frame(folder, index, i):
    item = index[i]
    if item['file'] != f'{i:06d}.png':
        raise ValueError('Unexpected source filename')
    path = folder/'source_frames'/item['file']
    if digest(path) != item['sha256']:
        raise ValueError(f'Source frame checksum mismatch: {path}')
    bgr = cv2.imread(str(path), cv2.IMREAD_COLOR)
    if bgr is None:
        raise ValueError('Cannot decode source')
    return cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB)


def policy_frame(rgb, crop, size):
    if crop not in CROPS or not 64 <= size <= 1024:
        raise ValueError('Unknown crop or invalid size (64..1024)')
    # Use the exact active road experiment crop/letterbox implementation, read-only.
    from development.experiments.road_vision.road_crop import crop_rgb
    return crop_rgb(rgb, {'roi': list(CROPS[crop]), 'size': [size, size]})


def contact_sheet(folder, output, crop, size):
    _, index = source_index(folder)
    times, labels, valid = arrays(folder)
    chosen = np.unique(np.linspace(0, len(index)-1, min(18, len(index))).astype(int))
    tiles = []
    for i in chosen:
        frame = policy_frame(source_frame(folder, index, i), crop, size).transpose(1, 2, 0)
        tile = np.zeros((size+24, size, 3), np.uint8)
        tile[:size] = cv2.cvtColor(frame, cv2.COLOR_RGB2BGR)
        cv2.putText(tile, f'{times[i]:.2f}s a={labels[i]} v={int(valid[i])}', (3, size+17),
                    cv2.FONT_HERSHEY_SIMPLEX, .4, (255,255,255), 1)
        tiles.append(tile)
    while len(tiles) % 3:
        tiles.append(np.zeros_like(tiles[0]))
    sheet = np.concatenate([np.concatenate(tiles[i:i+3], axis=1)
                            for i in range(0, len(tiles), 3)], axis=0)
    output = Path(output)
    if output.exists():
        raise FileExistsError(output)
    output.parent.mkdir(parents=True, exist_ok=True)
    if not cv2.imwrite(str(output), sheet):
        raise IOError('Could not save contact sheet')


def export(folder, output, crop, size):
    folder, output = Path(folder), Path(output)
    metadata, index = source_index(folder)
    capture = metadata.get('capture', {})
    if capture.get('framing_profile') != 'canvas_v1' or capture.get('detector_mode') != 'off':
        raise ValueError('Export requires recorded canvas_v1 framing and detector off')
    decision, review_path = latest_review(folder)
    if decision['status'] != 'approved' or not decision.get('visually_reviewed'):
        raise ValueError('Only explicitly reviewed approved recordings may be exported')
    if decision['manifest_sha256'] != digest(folder/'manifest.json') or any(
            digest(folder/key) != value for key, value in decision['label_sha256'].items()):
        raise ValueError('Recording changed after approval; review it again')
    times, labels, valid = arrays(folder)
    ids = segment_ids(times, decision['segments_seconds'])
    if not np.any(valid & (ids >= 0) & (labels != 0)):
        raise ValueError('No valid approved actions')
    output.mkdir(parents=True, exist_ok=False)
    frames = np.lib.format.open_memmap(output/'frames.npy', mode='w+', dtype=np.uint8,
                                      shape=(len(index), 3, size, size))
    for i in range(len(index)):
        rgb = source_frame(folder, index, i)
        if list(rgb.shape) != metadata['source_archive']['shape_hwc']:
            raise ValueError('Source geometry changed')
        frames[i] = policy_frame(rgb, crop, size)
    frames.flush()
    del frames
    for name in ('timestamps.npy', 'actions.npy', 'valid.npy', 'events.jsonl'):
        shutil.copyfile(folder/name, output/name)
    from tools.ppo_detector_off_features import detector_off_aux
    aux = np.load(folder/'aux.npy', allow_pickle=False)
    if aux.shape != (len(index), 557) or not np.isfinite(aux).all():
        raise ValueError('Invalid auxiliary observation shape/values')
    np.save(output/'aux.npy', detector_off_aux(aux))
    np.save(output/'segment_ids.npy', ids)
    np.save(output/'training_mask.npy', valid & (ids >= 0))
    # Completion marker written LAST. A partial export must never be a dataset.
    write_new(output/'export.json', {
        'schema': 'native-bc-export-v1', 'status': 'COMPLETE', 'crop': crop, 'roi': CROPS[crop],
        'size': [size,size], 'frames': len(index), 'image_shape': [3,size,size],
        'source_run': str(folder.resolve()), 'source_manifest_sha256': digest(folder/'manifest.json'),
        'review': decision, 'review_sha256': digest(review_path),
        'training_examples': int(np.sum(valid & (ids >= 0))),
        'history': 'single frames, chronological; reconstruct inside each segment only',
        'label_bound_ms': 130, 'detector_mode': 'off',
        'split_unit': 'whole recording/session, never random neighboring frames',
        'compatibility': 'Requires native-bc-export-v1 loader; not the legacy 512x288 BC dataset'})
    return output/'export.json'


def catalogue(root, output, trim_end=.7):
    if not np.isfinite(trim_end) or trim_end < 0:
        raise ValueError('Trim must be finite and nonnegative')
    items = []
    for path in Path(root).rglob('manifest.json'):
        m = read(path)
        if 'source_archive' not in m:
            continue
        review_data, _ = latest_review(path.parent)
        items.append((float(m['duration_seconds']), path.parent, m, review_data))
    rows = []
    for duration, folder, m, r in sorted(items, key=lambda item: item[0], reverse=True):
        video = folder/(m.get('preview_video') or 'contact_sheet.jpg')
        rows.append(f'<tr><td>{html.escape(folder.name)}</td><td>{duration:.2f}</td>'
                    f'<td>{html.escape(r["status"])}</td><td>{m.get("valid_action_samples",0)}</td>'
                    f'<td>0:{max(0,duration-trim_end):.2f}</td>'
                    f'<td><a href="{html.escape(video.resolve().as_uri(), quote=True)}">Preview</a></td></tr>')
    output = Path(output)
    with output.open('x', encoding='utf-8') as f:
        f.write('<!doctype html><meta charset="utf-8"><title>BC review</title>'
                '<h1>Recordings, longest first</h1><p>Trim suggestions are not approval. '
                'Inspect startup, all retained gameplay, and the ending. Check the no-coin objective.</p>'
                '<table border="1"><tr><th>Run</th><th>Seconds</th><th>Status</th>'
                '<th>Valid labels</th><th>Suggested interval</th><th>Review</th></tr>'
                + ''.join(rows) + '</table>')
    return output


def main():
    p = argparse.ArgumentParser(description=__doc__)
    sub = p.add_subparsers(dest='command', required=True)
    c = sub.add_parser('catalogue'); c.add_argument('root', type=Path)
    c.add_argument('--output', type=Path, required=True); c.add_argument('--trim-end', type=float, default=.7)
    r = sub.add_parser('review'); r.add_argument('folder', type=Path)
    r.add_argument('--status', choices=['candidate','approved','rejected'], required=True)
    r.add_argument('--segments', help='Explicit seconds: 0.4:6.2,7.0:9.1')
    r.add_argument('--reviewed', action='store_true'); r.add_argument('--note', default='')
    for name in ('preview', 'export'):
        e = sub.add_parser(name); e.add_argument('folder', type=Path)
        e.add_argument('--output', type=Path, required=True)
        e.add_argument('--crop', choices=list(CROPS), default='road_wide')
        e.add_argument('--size', type=int, default=192)
    a = p.parse_args()
    if a.command == 'catalogue': print(catalogue(a.root, a.output, a.trim_end))
    elif a.command == 'review': print(review(a.folder, status=a.status, intervals=a.segments,
                                            reviewed=a.reviewed, note=a.note))
    elif a.command == 'preview': contact_sheet(a.folder, a.output, a.crop, a.size); print(a.output)
    else: print(export(a.folder, a.output, a.crop, a.size))


if __name__ == '__main__': main()
