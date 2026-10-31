"""Compare saved speed diagnostics, keeping reaction latency and throughput separate."""
import argparse
import json
from pathlib import Path
import numpy as np


def distribution(values):
    values=np.asarray(values,dtype=float)
    if not len(values):return None
    if not np.isfinite(values).all() or (values<0).any():raise ValueError('Invalid latency samples')
    return dict(count=len(values),median_ms=float(np.median(values)),p95_ms=float(np.percentile(values,95)),
        p99_ms=float(np.percentile(values,99)),max_ms=float(values.max()),
        above_100ms=int((values>100).sum()),above_150ms=int((values>150).sum()))


def summarize(folder,warmup_commands=20):
    folder=Path(folder)
    config=json.loads((folder/'config.json').read_text())
    report=json.loads((folder/'parallel_training.json').read_text())
    expected='COMPLETE_FROZEN_POLICY_DIAGNOSTIC' if config['benchmark_only'] else 'COMPLETE'
    failures=[]
    if report['status']!=expected:failures.append('incomplete run')
    if not report.get('source_unchanged'):failures.append('source changed or unverified')
    if config['benchmark_only'] and (report.get('model_updated') or report.get('optimized_transitions') or not report.get('policy_unchanged')):
        failures.append('frozen policy changed or unverified')
    if not config['benchmark_only']:
        audits=report.get('audits',[])
        if len(audits)!=config['workers'] or any(a.get('status')!='PASS_SETTLED_LEARNING' or a.get('failures')!=[] for a in audits):
            failures.append('learning audit missing/failed')
    workers=[];all_samples=[];steady_samples=[]
    guard_samples=[]
    for index in range(config['workers']):
        worker=folder/f'worker_{index}'
        result_path=worker/'result.json'
        if not result_path.exists():failures.append(f'worker {index} result missing');continue
        result=json.loads(result_path.read_text())
        if result.get('error') or result.get('guard_hard_stopped') or result['learning_transitions']['discarded_partial_transitions']:
            failures.append(f'worker {index} stopped/discarded')
        records=[]
        ledger=worker/'learning_transitions.jsonl'
        if ledger.exists():
            for line in ledger.read_text().splitlines():
                row=json.loads(line)
                if row.get('command') and row.get('reaction_latency',{}):records.append(row)
        samples=[r['reaction_latency']['policy_request_to_command_ms'] for r in records]
        steady=samples[warmup_commands:]
        guard_samples.extend(r['reaction_latency']['guard_capture_complete_to_command_ms'] for r in records)
        all_samples.extend(samples);steady_samples.extend(steady)
        stats=distribution(steady)
        if stats is None:failures.append(f'worker {index} has insufficient commands')
        workers.append(dict(worker=index,all=distribution(samples),steady=stats))
    throughput=report.get('processed_transitions_per_second') if config['benchmark_only'] else report.get('optimized_transitions_per_second')
    if throughput is None or not np.isfinite(throughput) or throughput<=0:
        failures.append('throughput missing/invalid')
    return dict(report=str(folder),workers=config['workers'],detector=config['detector_mode'],
        compact=config['compact_observations'],shared_inference=config['shared_inference'],electron_host=config['electron_host'],
        benchmark_only=config['benchmark_only'],source_sha256=config['source_sha256'],
        valid=not failures,failures=failures,throughput=throughput,
        throughput_metric='processed/s without optimization' if config['benchmark_only'] else 'optimized/s',
        all=distribution(all_samples),steady=distribution(steady_samples),
        worst_worker_p95_ms=max((w['steady']['p95_ms'] for w in workers if w['steady']),default=None),
        guard_age=distribution(guard_samples),per_worker=workers)


def compare(rows):
    valid=[r for r in rows if r['valid']]
    if not valid:return dict(lowest_latency=None,highest_throughput=None,pareto_candidates=[])
    if len({(r['source_sha256'],r['benchmark_only']) for r in valid})!=1:
        raise ValueError('Compare the same source and collection mode; do not mix training/frozen throughput')
    latency=min(valid,key=lambda r:(r['worst_worker_p95_ms'],r['steady']['median_ms']))
    throughput=max(valid,key=lambda r:r['throughput'])
    frontier=[r['report'] for r in valid if not any(
        other['worst_worker_p95_ms']<=r['worst_worker_p95_ms'] and other['throughput']>=r['throughput'] and
        (other['worst_worker_p95_ms']<r['worst_worker_p95_ms'] or other['throughput']>r['throughput'])
        for other in valid)]
    return dict(lowest_latency=latency['report'],highest_throughput=throughput['report'],pareto_candidates=frontier)


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('runs',nargs='+',type=Path)
    parser.add_argument('--warmup-commands',type=int,default=20)
    parser.add_argument('--report',type=Path,required=True)
    args=parser.parse_args()
    if args.warmup_commands<0:parser.error('Warmup count must be nonnegative')
    if args.report.exists():parser.error('Choose a new report directory')
    rows=[summarize(path,args.warmup_commands) for path in args.runs]
    result=dict(runs=rows,comparison=compare(rows),excluded_initial_commands_per_worker=args.warmup_commands,
        note='Latency is screenshot-request to command issue, not Unity response. Warmup exclusion is descriptive. Short stochastic diagnostics can vary; memory is not used to rank configurations.')
    args.report.mkdir(parents=True)
    (args.report/'speed_comparison.json').write_text(json.dumps(result,indent=2),encoding='utf8')
    for row in rows:
        print(json.dumps({k:row[k] for k in ['report','workers','detector','compact','valid','throughput','steady','worst_worker_p95_ms']}))
    print(json.dumps(result['comparison'],indent=2))


if __name__=='__main__':main()
