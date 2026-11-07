"""Read-only label/frame causality audit; no full-resolution image loading."""
import argparse
from pathlib import Path
import numpy as np
from common import KEYS, read, write_new


def audit(folder):
    folder = Path(folder)
    meta = read(folder/'dataset.json')
    result = dict(samples=0, movement_samples=0, runs=0, failures=[], movement_delay_ms=[],
                  decision_intervals_ms=[], history_counts=[],
                  note='Checks exported targets against recorded key timestamps and frame indices. '
                       'Does not measure Unity input acceptance or human reaction time.')
    for run in meta['runs']:
        source = Path(run['source'])
        rows = read(source/'frames.json')
        events = read(source/'events.json')
        samples = read(folder/run['path']/'samples.json')
        arrays = {k: np.load(folder/run['path']/(k+'.npy'), mmap_mode='r', allow_pickle=False)
                  for k in ('indices', 'actions', 'aux')}
        try:
            times = np.asarray([r['at'] for r in rows])
            ends = np.asarray([r['end'] for r in rows])
            chosen = []
            for i, stamp in enumerate(times):
                if not chosen or stamp-times[chosen[-1]] >= 1/meta['decision_hz']-1e-6:
                    chosen.append(i)
            next_index = dict(zip(chosen[:-1], chosen[1:]))
            downs = sorted((e for e in events if e['type'] == 'keydown' and e['code'] in KEYS),
                           key=lambda e: e['at'])
            stamps = np.asarray([e['at'] for e in downs])
            used = np.asarray(run['source_frame_indices'])
            if len(samples) != len(arrays['actions']):
                raise ValueError('Sample/label count mismatch')
            for k, s in enumerate(samples):
                i = s['index']
                j = next_index.get(i)
                if j is None:
                    raise ValueError('Sample not in exported decision grid')
                lo, hi = np.searchsorted(stamps, [times[i], times[j]], side='left')
                keys = downs[lo:hi]
                label = int(arrays['actions'][k])
                if label != s['action'] or len(keys) > 1 or label != (KEYS[keys[0]['code']] if keys else 0):
                    raise ValueError('Key/action mismatch at sample '+str(k))
                if times[j]-times[i] > .130000001 or ends[i] < times[i]:
                    raise ValueError('Invalid decision interval')
                result['decision_intervals_ms'].append(float((times[j]-times[i])*1000))
                result['history_counts'].append(int(arrays['aux'][k,565:].reshape(32,7)[:,6].sum()))
                if keys:
                    delay = keys[0]['at']-ends[i]
                    if delay <= 0:
                        raise ValueError('Action occurs before capture completes')
                    result['movement_delay_ms'].append(delay*1000)
                    result['movement_samples'] += 1
                indices = arrays['indices'][k]
                if np.any(indices < -1) or np.any(indices >= len(used)):
                    raise ValueError('Invalid image index')
                selected = used[indices[indices >= 0]]
                if indices[-1] < 0 or used[indices[-1]] != i or np.any(times[selected] > times[i]):
                    raise ValueError('Future/wrong latest frame')
                result['samples'] += 1
            result['runs'] += 1
        except ValueError as exc:
            result['failures'].append(dict(run=run['path'], reason=str(exc)))
        finally:
            for array in arrays.values():
                array._mmap.close()
    delays = result.pop('movement_delay_ms')
    result['movement_after_capture_ms'] = dict(count=len(delays),
        median=float(np.median(delays)) if delays else None,
        p95=float(np.percentile(delays, 95)) if delays else None,
        max=max(delays) if delays else None)
    result['passed'] = not result['failures']
    for key in ('decision_intervals_ms', 'history_counts'):
        values=result.pop(key)
        result[key]=dict(count=len(values),median=float(np.median(values)) if values else None,
                        p95=float(np.percentile(values,95)) if values else None)
    return result


if __name__ == '__main__':
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--dataset', type=Path, required=True)
    p.add_argument('--output', type=Path, required=True)
    a = p.parse_args()
    report = audit(a.dataset)
    a.output.parent.mkdir(parents=True, exist_ok=True)
    write_new(a.output, report)
    print(report)
    if not report['passed']:
        raise SystemExit(1)
