"""Low-cost availability checks before loading Torch, datasets or browsers."""
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[3]

def source_fingerprints():
    import hashlib
    paths=[*Path(__file__).resolve().parent.glob('*.py'),
        *(ROOT/'development/experiments/latency_tail_20261107').glob('*.py'),
        ROOT/'development/experiments/command_history/command_adapter.py']
    return {f.relative_to(ROOT).as_posix():hashlib.sha256(f.read_bytes()).hexdigest() for f in paths}


def require_available():
    import psutil
    # Reuse the recorder's tested Windows venv-launcher/self exclusion rules.
    sys.path.insert(0, str(ROOT/'development/staged/multimap_bc_20261106'))
    from common import require_idle
    require_idle()
    current = psutil.Process()
    own = {current.pid}
    command = current.cmdline()[1:]
    parent = current.parent()
    while parent is not None:
        if not parent.name().lower().startswith('python') or parent.cmdline()[1:] != command:
            break
        own.add(parent.pid)
        parent = parent.parent()
    entries = {str((Path(__file__).parent/name).resolve()).lower()
               for name in ('probe.py', 'case.py', 'evaluate.py', 'eval_case.py', 'check.py')}
    active = []
    for process in psutil.process_iter(['pid', 'name', 'cmdline']):
        if process.pid in own or 'python' not in (process.info['name'] or '').lower():
            continue
        if any(str(Path(arg).resolve()).lower() in entries
               for arg in (process.info['cmdline'] or [])[1:] if arg.endswith('.py')):
            active.append(process.pid)
    if active:
        raise RuntimeError(f'Another staged check/probe/evaluator is active: {active}')
    available = psutil.virtual_memory().available
    if available < 2*1024**3:
        raise RuntimeError(f'Only {available/1024**3:.2f} GiB available; need 2 GiB before checks/benchmarks')


def require_correctness(features):
    if not features:
        return
    import hashlib
    import json
    here = Path(__file__).resolve().parent
    cuda = bool(set(features) & {'pinned', 'gpu_cache'})
    path = ROOT/'reports/pipeline_optimization_checks'/('cuda.json' if cuda else 'cpu.json')
    if not path.exists():
        raise RuntimeError('Run staged check.py'+(' --cuda' if cuda else '')+' before live testing')
    report = json.loads(path.read_text())
    current = source_fingerprints()
    if not report.get('passed') or report.get('sources') != current or (cuda and not report.get('cuda')):
        raise RuntimeError('Correctness result missing/stale/failed; rerun staged check.py')
