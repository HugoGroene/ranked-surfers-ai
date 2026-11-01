"""Read-only artifact aggregation; writes one new analysis JSON, no model work."""
import hashlib
import json
from collections import Counter
from pathlib import Path
import numpy as np

ROOT=Path(__file__).resolve().parent
batch=ROOT/'temporal_batch_20261101_180312_752232'
read=lambda p:json.loads(p.read_text())
rows=[]
for old in read(batch/'comparison.json')['rows']:
    folder=batch/old['name'];train=read(folder/'train/trial_report.json')
    row={k:old.get(k) for k in ('name','training_status','evaluation_status','mean_seconds','median_seconds',
        'max_seconds','optimized_transitions','optimized_per_second','actions','mean_bootstrap_95pct_seconds','capture')}
    row['audit']=train.get('audit',{}).get('status')
    row['error']=train.get('error')
    row['finalization_errors']=train.get('finalization_errors')
    row['config']=read(folder/'train/trial_config.json')['config']
    row['latency_last1000']=old.get('reaction_latency',{}).get('metrics',{}).get('policy_request_to_command_ms')
    count=sum((row['actions'] or {}).values())
    row['dominant_action_fraction']=max((row['actions'] or {}).values(),default=0)/count if count else None
    evaluation=folder/'eval/v61_eval_report.json'
    if evaluation.exists():
        ev=read(evaluation)
        row['mean_action_probabilities']=ev.get('mean_action_probabilities')
        row['mean_policy_max_probability']=ev.get('mean_policy_max_probability')
    latency=[];durations=[];episode_seconds={};command_counts=Counter()
    with (folder/'train/learning_transitions.jsonl').open() as f:
        for line in f:
            item=json.loads(line)
            if item.get('event')!='collected':continue
            value=(item.get('reaction_latency') or {}).get('policy_request_to_command_ms')
            if value is not None:latency.append(value)
            ep=item['episode'];episode_seconds[ep]=episode_seconds.get(ep,0)+item.get('active_seconds',0)
            if item.get('command'):command_counts[item['command']]+=1
            if item.get('done'):durations.append(episode_seconds.pop(ep))
    row['training_dispatched_commands']=dict(command_counts)
    if latency:row['latency_all_collected']={
        'count':len(latency),'median_ms':float(np.median(latency)),
        'p95_ms':float(np.percentile(latency,95)),'p99_ms':float(np.percentile(latency,99)),
        'over100ms':int(np.sum(np.asarray(latency)>100))}
    if durations:
        row['ledger_completed_episode_active_time_quartiles']=[float(np.mean(x)) for x in np.array_split(durations,4)]
        row['ledger_duration_note']='Sum of recorded active_seconds for completed ledger episodes; episode-index quartiles, not independent evaluations or Unity-clock measurements.'
    rows.append(row)

captures=[]
for name in ('capture_matrix_20261101_154254_066769','capture_matrix_20261101_161307_636905'):
    data=read(ROOT/name/'comparison.json')
    for index,s in enumerate(data['stages']):
        lat=[r['metrics']['policy_request_to_command_ms']['p95_ms'] for r in s.get('reaction_latency',[])
             if 'policy_request_to_command_ms' in r['metrics']]
        captures.append(dict(batch=name,stage=index,arm=s['arm'],workers=s['workers'],
            status=s['result']['status'],collected_per_second=s['result'].get('processed_transitions_per_second'),
            per_worker_p95_ms=lat,model_updated=s['result']['model_updated']))
inventory=[]
for path in sorted(ROOT.glob('temporal_batch_*/plan.json')):
    plan=read(path);counts=Counter()
    outputs=set()
    for job in plan.get('jobs',[]):
        out=Path(job['output']).resolve()
        if str(out) in outputs:continue
        outputs.add(str(out))
        f=out/('trial_report.json' if job['mode']=='train' else 'v61_eval_report.json')
        if f.exists():counts[job['mode']+'_'+read(f).get('status','UNKNOWN')]+=1
    inventory.append(dict(batch=path.parent.name,minutes=plan.get('train_minutes'),
        episodes=plan.get('eval_episodes'),counts=dict(counts),
        source_digest=hashlib.sha256((path.parent/'sources.json').read_bytes()).hexdigest()))
result=dict(temporal=rows,capture_stages=captures,historical_inventory=inventory,
    note='Historical smoke/partial/source-version results are retained separately, never pooled as independent completed learning trials. Latency measures command issue, not Unity response.')
output=ROOT/'TEMPORAL_FINAL_ACCUMULATED_20261101.json'
output.write_text(json.dumps(result,indent=2))
for row in rows:
    print(row['name'],row['training_status'],row['mean_seconds'],row.get('latency_all_collected'),
          row.get('ledger_completed_episode_active_time_quartiles'))
print('REPORT:',output)
