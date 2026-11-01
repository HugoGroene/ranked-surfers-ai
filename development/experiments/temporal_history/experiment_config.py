"""Pure configuration/provenance helpers; no browser or ML imports."""
import hashlib
import math
from pathlib import Path

ROOT = Path(__file__).resolve().parents[3]
HERE = Path(__file__).resolve().parent


def configuration(frames=4, span_ms=500, capture_hz=60, decision_hz=30):
    if type(frames) is not int or frames not in (4,6,7,8,9,10,11,12):
        raise ValueError('frames must be 4 or 6..12')
    for name,value,limit in [('span_ms',span_ms,2000),('capture_hz',capture_hz,60),('decision_hz',decision_hz,60)]:
        if not math.isfinite(value) or not 0 < value <= limit:
            raise ValueError(f'Invalid {name}')
        if name.endswith('_hz') and value < 1:
            raise ValueError(f'{name} must be at least 1 Hz')
    if span_ms/1000*capture_hz < frames-1:
        raise ValueError('Requested cadence cannot fill the distinct-frame span')
    return dict(frames=frames,span_ms=float(span_ms),capture_hz=float(capture_hz),decision_hz=float(decision_hz))


# Equal decision limits. One capture-rate control arm; no BC/checkpoint migration.
PRESETS = {
    'short4_60': configuration(4,100,60),
    'wide4_60': configuration(4,500,60),
    'wide6_60': configuration(6,500,60),
    'wide8_60': configuration(8,500,60),
    'wide12_60': configuration(12,500,60),
    'wide8_30': configuration(8,500,30),
}


def source_hashes():
    files = list(ROOT.glob('*.py')) + list((ROOT/'tools').rglob('*.py'))
    files += list((ROOT/'tools').rglob('*.cjs')) + list(HERE.glob('*.py'))
    return {p.relative_to(ROOT).as_posix():hashlib.sha256(p.read_bytes()).hexdigest()
            for p in sorted(set(files)) if 'node_modules' not in p.parts}


def file_hash(path):
    digest=hashlib.sha256()
    with Path(path).open('rb') as f:
        for block in iter(lambda:f.read(1024*1024),b''):digest.update(block)
    return digest.hexdigest()
