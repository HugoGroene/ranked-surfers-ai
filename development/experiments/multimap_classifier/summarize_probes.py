"""Separate startup/reset/resume command latency from ordinary gameplay."""
import argparse
from datetime import datetime
import json
from pathlib import Path
import numpy as np


def phase_latencies(rows):
    phases={name:[] for name in ('initial_decision','post_reset','post_resume','ordinary')}
    seen=set();resume=False
    for row in rows:
        if row.get('kind')=='passive_resync':
            resume=True
            continue
        if row.get('kind')!='policy':continue
        episode=row['episode']
        first=episode not in seen
        phase=('initial_decision' if not seen else 'post_reset') if first else ('post_resume' if resume else 'ordinary')
        seen.add(episode);resume=False
        latency=(row.get('reaction_latency') or {}).get('policy_request_to_command_ms')
        # NONE decisions have no issued-key latency. Do not mislabel the later
        # first movement as the first decision of an episode.
        if latency is not None and np.isfinite(latency):phases[phase].append(float(latency))
    return phases


def metrics(values):
    if not values:return dict(count=0)
    return dict(count=len(values),median_ms=float(np.median(values)),
                p95_ms=float(np.percentile(values,95)),max_ms=max(values))


def summarize(folder):
    probe=json.loads((folder/'result.json').read_text())
    train=folder/'case/train'
    report=json.loads((train/'parallel_training.json').read_text())
    rounds=[json.loads(s) for s in (train/'round_timings.jsonl').read_text().splitlines()]
    later=rounds[1:]
    cohort=report['processed_transitions']//max(1,report['rounds'])
    pooled={name:[] for name in ('initial_decision','post_reset','post_resume','ordinary')}
    workers={}
    for worker in sorted(train.glob('worker_*')):
        rows=[json.loads(s) for s in (worker/'learning_transitions.jsonl').read_text().splitlines() if s.strip()]
        phases=phase_latencies(rows)
        for name,values in phases.items():pooled[name].extend(values)
        saved=json.loads((worker/'crop_reuse.json').read_text())
        resets=saved.get('reset_wall_ms')
        workers[worker.name]=dict(latency={name:metrics(values) for name,values in phases.items()},
            completed_episodes=len((worker/'survival_episodes.jsonl').read_text().splitlines()),
            reset_wall_ms=resets,
            initial_reset_ms=resets[0] if resets else None,
            later_reset_total_ms=sum(resets[1:]) if resets else None,
            reset_note='Unavailable for older probes; not inferred from terminal/restart gaps.' if resets is None else saved.get('reset_note'))
    return dict(report=str(folder),workers=probe['workers'],status=report['status'],
        optimized_per_second=report['optimized_transitions_per_second'],rounds=len(rounds),
        subsequent_rounds_collection_plus_optimizer_per_second=(
            cohort*len(later)/sum(r['collection_seconds']+r['update_seconds'] for r in later) if later else None),
        subsequent_round_note='Excludes first round and saving/coordination outside measured collection+update; includes later resets and pause/resume. Not pure gameplay throughput.',
        latency={name:metrics(values) for name,values in pooled.items()},worker_details=workers)


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('reports',type=Path,nargs='+')
    args=parser.parse_args()
    result=dict(note='Command issue latency, not Unity response. All phases retained. Short runs cannot establish sustained throughput or learning quality.',
                runs=[summarize(folder) for folder in args.reports])
    root=Path(__file__).resolve().parents[3]
    output=root/'reports'/datetime.now().strftime('latency_phases_%Y%m%d_%H%M%S_%f.json')
    with output.open('x') as handle:json.dump(result,handle,indent=2)
    print(output)
    for run in result['runs']:
        print(json.dumps({key:run[key] for key in ('workers','optimized_per_second','subsequent_rounds_collection_plus_optimizer_per_second','latency')},indent=2))


if __name__=='__main__':main()
