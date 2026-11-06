"""Post-run summaries; no sampling or I/O in the live decision loop."""
import json
from pathlib import Path
import numpy as np


def read(path):return json.loads(Path(path).read_text(encoding='utf8'))


def stats(values):
    values=[float(v) for v in values if v is not None and np.isfinite(v)]
    return dict(count=len(values),median=float(np.median(values)) if values else None,
        p95=float(np.percentile(values,95)) if values else None,max=max(values) if values else None)


def lines(path):
    if not Path(path).exists():return
    with Path(path).open(encoding='utf8') as handle:
        for line in handle:
            if line.strip():yield json.loads(line)


def clean(folder,workers):
    try:
        if read(folder/'parallel_training.json').get('finalization_errors')!=[]:return False
        return all(read(folder/f'worker_{i}'/'hybrid_close.json').get('status')=='COMPLETE' for i in range(workers))
    except (OSError,ValueError):return False


def validate(folder,workers):
    import zipfile
    r=read(folder/'parallel_training.json')
    if r.get('status')!='COMPLETE' or not r.get('model_updated') or not r.get('source_unchanged'):
        raise ValueError('Incomplete or changed-source training probe')
    if r.get('optimized_transitions',0)<=0 or len(r.get('audits',[]))!=workers:
        raise ValueError('Missing optimized transitions/audits')
    if any(a['status']!='PASS_SETTLED_LEARNING' for a in r['audits']):raise ValueError('Learning audit failed')
    if not clean(folder,workers):raise ValueError('Cleanup not confirmed')
    if not read(folder/'hybrid_benchmark.json')['source_unchanged']:raise ValueError('Probe source changed')
    for i in range(workers):
        row=read(folder/f'worker_{i}'/'result.json')
        if row.get('error') or row.get('guard_hard_stopped') or row.get('profile_error'):
            raise ValueError(f'Worker {i} failure')
        if read(folder/f'worker_{i}'/'hybrid_history.json')['recent_frames']!=3:
            raise ValueError('Wrong history')
    with zipfile.ZipFile(folder/'latest.zip') as checkpoint:
        if checkpoint.testzip() is not None:raise ValueError('Corrupt final checkpoint')
    return r


def summarize_case(job):
    folder=Path(job['output']);r=read(folder/'parallel_training.json')
    row=dict(name=job['name'],workers=job['workers'],phase=job['phase'],status=r.get('status'),
        retained=bool(job.get('retained')),optimized_s=r.get('optimized_transitions_per_second'),
        optimized_transitions=r.get('optimized_transitions'),rounds=r.get('rounds'),
        active_wall_seconds=r.get('active_wall_seconds'),error=r.get('error'),
        finalization_errors=r.get('finalization_errors'),audits=r.get('audits'),worker_details=[])
    if row['status']=='COMPLETE':
        try:validate(folder,job['workers']);row['validated']=True
        except (OSError,ValueError,KeyError) as exc:
            row.update(status='INVALID_RESULT',validated=False,validation_error=repr(exc))
    medians=[];tails=[]
    for i in range(job['workers']):
        worker=folder/f'worker_{i}'
        if not (worker/'result.json').exists():continue
        result=read(worker/'result.json');latency=result.get('reaction_latency',{})
        metric=latency.get('metrics',{}).get('policy_request_to_command_ms',{})
        if metric.get('median_ms') is not None:medians.append(metric['median_ms'])
        if metric.get('p95_ms') is not None:tails.append(metric['p95_ms'])
        detail=dict(worker=i,reaction_latency=latency,guard_hard_stopped=result.get('guard_hard_stopped'),
            frame_state_cache=result.get('frame_state_cache'),error=result.get('error'))
        for name,file in [('history','hybrid_history.json'),('capture','temporal_capture.json'),('close','hybrid_close.json')]:
            detail[name]=read(worker/file) if (worker/file).exists() else None
        detail['rollout_seconds']=stats([x['seconds'] for x in read(worker/'rollout_timings.json')]) if (worker/'rollout_timings.json').exists() else None
        kinds={};gpu=[]
        for event in lines(worker/'pipeline_profile.jsonl'):
            if event['kind']=='policy' and event.get('passive'):continue
            kinds.setdefault(event['kind'],[]).append((event['end']-event['start'])*1000)
            if event.get('gpu_stream_ms') is not None:gpu.append(event['gpu_stream_ms'])
        detail['pipeline_host_ms']={k:stats(v) for k,v in kinds.items()}
        detail['policy_gpu_stream_ms']=stats(gpu)
        detail['profile_metadata']=read(worker/'pipeline_profile_metadata.json') if (worker/'pipeline_profile_metadata.json').exists() else None
        row['worker_details'].append(detail)
    row['worker_median_range_ms']=[min(medians),max(medians)] if medians else None
    row['worst_worker_p95_ms']=max(tails) if tails else None
    rounds=list(lines(folder/'round_timings.jsonl'))
    row['collection_seconds']=stats([x.get('collection_seconds') for x in rounds])
    row['optimizer_seconds']=stats([x.get('update_seconds') for x in rounds])
    resources=list(lines(folder/'system_usage.jsonl'))
    row['system_cpu_mean_percent']=stats([x.get('cpu_mean_percent') for x in resources])
    row['system_busiest_cpu_percent']=stats([x.get('cpu_busiest_core_percent') for x in resources])
    available=[x['memory_available_gib'] for x in resources if 'memory_available_gib' in x]
    row['min_system_available_gib']=min(available) if available else None
    gpu_load=[];gpu_memory=[]
    for sample in resources:
        for line in sample.get('nvidia_csv','').splitlines():
            try:
                columns=line.split(',')
                gpu_load.append(float(columns[0].strip().split()[0]))
                gpu_memory.append(float(columns[2].strip().split()[0]))
            except (IndexError,ValueError):pass
    row['system_gpu_utilization_percent']=stats(gpu_load)
    row['system_gpu_memory_mib']=stats(gpu_memory)
    row['resource_samples']=str(folder/'system_usage.jsonl')
    row['note']='Latency p95 is the worst worker p95, not a pooled percentile. Profiles include instrumentation. Resources describe the whole system.'
    return row


def write_summary(output,plan):
    rows=[]
    for job in plan['jobs']:
        if (Path(job['output'])/'parallel_training.json').exists():
            try:rows.append(summarize_case(job))
            except (OSError,ValueError,KeyError) as exc:
                rows.append(dict(name=job['name'],workers=job['workers'],phase=job['phase'],status='REPORT_INCOMPLETE',error=repr(exc)))
    (output/'scaling_summary.json').write_text(json.dumps(dict(rows=rows,
        note='Same initial checkpoint per case; each case performs independent PPO updates. Larger worker counts increase cohort size, so this is not a learning-quality comparison.'),indent=2))
    def fmt(x):return f'{x:.2f}' if isinstance(x,(float,int)) else 'n/a'
    text=['# Five-frame worker scaling','',
        '| Stage | Workers | Status | Optimized/s | Worker median latency range ms | Worst worker p95 ms | Median optimizer seconds |',
        '|---|---:|---|---:|---|---:|---:|']
    for r in rows:
        med=r.get('worker_median_range_ms')
        text.append(f'| {r["name"]} | {r["workers"]} | {r["status"]} | {fmt(r.get("optimized_s"))} | '
            f'{" – ".join(fmt(v) for v in med) if med else "n/a"} | {fmt(r.get("worst_worker_p95_ms"))} | '
            f'{fmt(r.get("optimizer_seconds",{}).get("median"))} |')
    text+=['','Latency ends at command issue, not Unity response. Review per-worker capture retries, '
        'mature history availability and audits in the JSON. Throughput includes resets, pauses and updates.',
        'A short pass does not establish overnight stability. No configuration is adopted automatically.']
    (output/'scaling_summary.md').write_text('\n'.join(text),encoding='utf8')
