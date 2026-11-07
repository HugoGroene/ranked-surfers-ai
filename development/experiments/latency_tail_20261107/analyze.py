"""Read-only phase-separated latency and optimizer summaries."""
import argparse
import json
from pathlib import Path
import numpy as np


def read(p):return json.loads(p.read_text())
def lines(p):return [json.loads(s) for s in p.read_text().splitlines() if s.strip()] if p.exists() else []
def stats(values):
    v=np.asarray([x for x in values if x is not None and np.isfinite(x)],float)
    return dict(n=len(v),median=float(np.median(v)),p95=float(np.percentile(v,95)),max=float(v.max())) if len(v) else dict(n=0)


def training(folder):
    root=folder/'case/train';report=read(root/'parallel_training.json')
    records=[];profiles=[];breakdown={};phases={};episodes=set()
    for worker in sorted(root.glob('worker_*')):
        previous=None
        table={}
        for row in lines(worker/'learning_transitions.jsonl'):
            if row.get('event')!='collected':continue
            phase=('passive' if row['kind']!='policy' else 'startup' if previous is None else
                   'reset' if previous['episode']!=row['episode'] else
                   'resume' if previous['kind']=='passive_resync' else 'ordinary')
            row=dict(row,phase=phase,worker=worker.name)
            table[row['transition']]=row;records.append(row);previous=row
            if row.get('done'):episodes.add((worker.name,row['episode']))
        for row in lines(worker/'pipeline_profile.jsonl'):
            if row['kind']=='policy' and not row['passive']:
                t=table.get(row['transition'])
                if not t or t['phase']!='ordinary':continue
                source=row.get('source') or {}
                item=dict(inference_ms=(row['end']-row['start'])*1000)
                if source.get('image_received_at') is not None:
                    item['image_reply_to_inference_ms']=(row['start']-source['image_received_at'])*1000
                if t.get('dispatch_at') is not None:item['inference_ready_to_command_ms']=(t['dispatch_at']-row['end'])*1000
                profiles.append(item)
            elif row['kind'] in ('_pack','_read_screen_state','_native_frames','rollout_buffer_add'):
                breakdown.setdefault(row['kind'],[]).append((row['end']-row['start'])*1000)
    for phase in ('startup','reset','resume','ordinary'):
        group=[r['reaction_latency'] for r in records if r['phase']==phase and r.get('reaction_latency')]
        phases[phase]={k:stats([r.get(k) for r in group]) for k in ('policy_request_to_command_ms','guard_request_to_command_ms')}
    rounds=lines(root/'round_timings.jsonl');warm=rounds[1:]
    transfers=lines(root/'pipeline_experiment.jsonl')+lines(root/'pipeline_runtime.jsonl')
    return dict(folder=str(folder),status=read(folder/'result.json')['status'],rounds=rounds,
        optimized=report.get('optimized_transitions'),active_wall_s=report.get('active_wall_seconds'),
        optimized_s=report.get('optimized_transitions',0)/max(.001,report.get('active_wall_seconds',0)),
        warm_update_s=stats([r['update_seconds'] for r in warm]),
        warm_collection_s=stats([r['collection_seconds'] for r in warm]),
        phase_latency=phases,ordinary_breakdown={k:stats([r.get(k) for r in profiles]) for k in
            ('inference_ms','image_reply_to_inference_ms','inference_ready_to_command_ms')},
        component_ms={k:stats(v) for k,v in breakdown.items()},completed_episodes=len(episodes),
        transfer_modes=transfers,cleanup=read(folder/'result.json').get('remaining_owned_pids'),
        audit_statuses=[v.get('status') for v in report.get('audits',[])])


if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('folders',type=Path,nargs='+');p.add_argument('--output',type=Path)
    a=p.parse_args();value=[training(f) for f in a.folders]
    if a.output:
        with a.output.open('x') as f:json.dump(value,f,indent=2)
    print(json.dumps(value,indent=2))
