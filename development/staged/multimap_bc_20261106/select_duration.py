"""Explicit user-directed pilot selection; never claims a complete visual review."""
import argparse
from collections import Counter
from datetime import datetime
from pathlib import Path
from common import ROOT,DATA,read,write_new
from review import approve,latest
from dataset import reconstruct


def select(sessions,minimum=12.,trim=3.,apply=False):
    if not 0<trim<minimum:raise ValueError('Require 0 < trim < minimum')
    report=dict(rule=dict(minimum_seconds=minimum,trim_end_seconds=trim),applied=apply,
        note='User-directed duration selection, not full visual approval. No original recording modified.',
        selected=[],excluded=[],totals={})
    for session in sessions:
        for manifest in sorted(Path(session).glob('run_*/manifest.json')):
            run=manifest.parent;m=read(manifest)
            if m['duration']<minimum:
                report['excluded'].append(dict(run=str(run.resolve()),reason='shorter_than_minimum'));continue
            intervals=[[0.,m['duration']-trim]]
            rows=read(run/'frames.json');events=read(run/'events.json')
            samples=reconstruct(rows,events,intervals,30)
            counts=Counter(s['action'] for s in samples)
            if not counts:
                report['excluded'].append(dict(run=str(run.resolve()),reason='no_valid_samples'));continue
            revision=None
            if apply:
                previous,path=latest(run)
                if (previous.get('status')=='approved' and previous.get('approval_basis')=='user_directed_duration_trim'
                        and previous.get('duration_rule')==dict(minimum_seconds=minimum,trim_end_seconds=trim)
                        and previous.get('intervals')==intervals):revision=path
                else:revision=approve(run,f'0:{m["duration"]-trim}',False,duration_rule=(minimum,trim))
            entry=dict(run=str(run.resolve()),map=m['map'],split=m['split'],original_seconds=m['duration'],
                intervals=intervals,retained_seconds=intervals[0][1],valid_samples=len(samples),
                action_counts=[counts[i] for i in range(5)],
                review=str(revision.resolve()) if revision else None)
            report['selected'].append(entry)
            total=report['totals'].setdefault(m['map'],dict(runs=0,retained_seconds=0.,actions=[0]*5))
            total['runs']+=1;total['retained_seconds']+=entry['retained_seconds']
            total['actions']=[x+y for x,y in zip(total['actions'],entry['action_counts'])]
    return report


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--sessions',type=Path,nargs='+',required=True)
    p.add_argument('--minimum-seconds',type=float,default=12)
    p.add_argument('--trim-end-seconds',type=float,default=3)
    p.add_argument('--apply-user-selection',action='store_true')
    a=p.parse_args()
    for session in a.sessions:
        if not (session/'session.json').is_file():p.error('Not a recording session: '+str(session))
    report=select(a.sessions,a.minimum_seconds,a.trim_end_seconds,a.apply_user_selection)
    output=ROOT/'reports'/f'bc_selection_{datetime.now():%Y%m%d_%H%M%S_%f}.json'
    write_new(output,report)
    print('SELECTION:',output)
    for name,total in report['totals'].items():print(name,total)


if __name__=='__main__':main()
