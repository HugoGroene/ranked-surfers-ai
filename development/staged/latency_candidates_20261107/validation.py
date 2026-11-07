"""Lightweight gates run before any model/browser imports."""
import hashlib
import json
from pathlib import Path
import sys

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[2]
RESULT = ROOT/'reports/latency_candidate_checks/result.json'
MODEL_RESULT = ROOT/'reports/latency_candidate_checks/models.json'


def digest(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def fingerprints():
    paths = list(HERE.glob('*.py')) + [
        ROOT/'rs_v4_screen_states.py',
        ROOT/'train_rs_v612_ppo.py',
        ROOT/'development/experiments/temporal_history/live_adapter.py',
        ROOT/'development/experiments/latency_tail_20261107/pacing.py',
        ROOT/'development/experiments/latency_tail_20261107/adam_steps.py',
        ROOT/'development/experiments/latency_tail_20261107/runtime.py',
        ROOT/'development/experiments/latency_tail_20261107/resync_spacing.py',
        ROOT/'tools/ppo_settled_learning.py',
        ROOT/'tools/ppo_parallel_learning.py',
        ROOT/'development/experiments/road_vision/road_models.py',
        ROOT/'development/staged/multimap_bc_20261106/normalized_actor.py',
    ]
    return {p.relative_to(ROOT).as_posix(): digest(p) for p in paths}


def require_idle():
    sys.path.insert(0, str(ROOT))
    from development.staged.pipeline_optimizations_20261106.preflight import require_available
    require_available()
    import psutil
    current = psutil.Process(); own = {current.pid}; args = current.cmdline()[1:]
    parent = current.parent()
    while parent is not None and parent.name().lower().startswith('python') and parent.cmdline()[1:] == args:
        own.add(parent.pid); parent = parent.parent()
    active = []
    entries = {str((HERE/name).resolve()).lower() for name in ('probe.py', 'case.py', 'check.py', 'check_models.py')}
    for proc in psutil.process_iter(['pid', 'name', 'cmdline']):
        if proc.pid in own or 'python' not in (proc.info['name'] or '').lower(): continue
        if any(str(Path(arg).resolve()).lower() in entries for arg in (proc.info['cmdline'] or [])[1:] if arg.endswith('.py')):
            active.append(proc.pid)
    if active: raise RuntimeError(f'Another latency candidate is active: {active}')


def require_checked(classifier, arm='both', checkpoint=None):
    if arm == 'baseline': return
    if arm in ('policy_layout','fused_adam','policy_fused','adam_cpu_steps','pump_cpu_steps'):
        if not MODEL_RESULT.exists(): raise RuntimeError('Run check_models.py --cuda while idle first')
        report = json.loads(MODEL_RESULT.read_text())
        if not report.get('passed') or not report.get('cuda') or report.get('sources') != fingerprints():
            raise RuntimeError('CUDA model correctness result missing, failed or stale')
        if checkpoint is None or report.get('checkpoint_sha256') != digest(checkpoint):
            raise RuntimeError('Policy checkpoint changed since model correctness checks')
        if arm != 'pump_cpu_steps': return
    if not RESULT.exists(): raise RuntimeError('Run this directory\'s check.py while idle first')
    report = json.loads(RESULT.read_text())
    if not report.get('passed') or report.get('sources') != fingerprints():
        raise RuntimeError('Candidate correctness result missing, failed or stale')
    if report.get('classifier_sha256') != digest(classifier):
        raise RuntimeError('Classifier changed since parity check')
