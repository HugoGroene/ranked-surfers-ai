"""Focused crop/encoder and spacing comparison; offline + all smoke gates first."""
import argparse
from datetime import datetime
import json
from pathlib import Path
import random
import subprocess
import sys
import math
from road_config import ROOT,configuration,source_hashes,write_json
from road_batch import jobs,run

def cases():
    result=[(arm,'uniform') for arm in ('full128_tiny','road128_tiny','road192_tiny',
            'road128_spatial','road192_spatial','road256_spatial','road384_spatial')]
    return result+ [('road192_spatial',p) for p in ('consecutive','golden','logarithmic')]

def phase_plan(folder,seeds,minutes,episodes,eval_minutes):
    configs={};all_jobs=[]
    for arm,spacing in cases():
        label=arm+'_'+spacing
        configs[label]=configuration(arm,spacing=spacing)
        generated=jobs([arm],seeds,minutes,episodes,eval_minutes,folder,
                       dict(frames=4,span_ms=500,capture_hz=60,decision_hz=30,spacing=spacing))
        for job in generated:
            old=job['name'];new=old.replace(arm,label,1)
            job['name']=new
            job['output']=job['output'].replace(old,new)
            job['command']=[part.replace(old,new) for part in job['command']]
            all_jobs.append(job)
    training=[j for j in all_jobs if j['mode']=='train']
    evaluation=[j for j in all_jobs if j['mode']=='eval']
    random.Random(712).shuffle(training)
    evaluation.sort(key=lambda j:next(i for i,t in enumerate(training) if t['name']==j['name']))
    return dict(configs=configs,seeds=seeds,train_minutes=minutes,eval_episodes=episodes,
                jobs=training+evaluation)

def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--execute',action='store_true')
    p.add_argument('--smoke-only',action='store_true')
    p.add_argument('--train-minutes',type=float,default=30)
    p.add_argument('--eval-episodes',type=int,default=300)
    p.add_argument('--eval-max-minutes',type=float,default=90)
    p.add_argument('--max-hours',type=float,default=0,help='Optional stage scheduling budget; zero disables it')
    p.add_argument('--seeds',type=int,nargs='+',default=[123,456])
    p.add_argument('--output',type=Path)
    p.add_argument('--resume',type=Path)
    a=p.parse_args()
    if not 0<a.train_minutes<=120 or not 0<a.eval_max_minutes<=2880 or a.eval_episodes<1:p.error('Invalid budgets')
    if not math.isfinite(a.max_hours) or not 0<=a.max_hours<=168:p.error('Invalid batch hours')
    if not a.seeds or len(set(a.seeds))!=len(a.seeds) or any(not 0<=s<2**32 for s in a.seeds):p.error('Invalid seeds')
    folder=(a.output or ROOT/'reports'/('road_overnight_'+datetime.now().strftime('%Y%m%d_%H%M%S_%f'))).resolve()
    if not folder.is_relative_to(ROOT/'reports') or folder.exists():p.error('Use a new reports folder')
    hashes=source_hashes()
    parent=a.resume.resolve() if a.resume else None
    if parent and json.loads((parent/'sources.json').read_text())!=hashes:p.error('Source changed; preserve older results separately')
    plan=(json.loads((parent/'plan.json').read_text()) if parent else dict(
        suite='road_and_spacing_v1',cases=cases(),seeds=a.seeds,train_minutes=a.train_minutes,
        eval_episodes=a.eval_episodes,eval_max_minutes=a.eval_max_minutes,max_hours=a.max_hours,
        initialization='scratch',capture_backend='gpu_100',workers=1,
        note='Consecutive is a legacy-style pack-history control, not exact production replay. '
             'Other spacing arms share 500ms. No BC. No proven winner.'))
    if plan.get('suite')!='road_and_spacing_v1':p.error('Resume requires this suite')
    folder.mkdir(parents=True);write_json(folder/'sources.json',hashes)
    import batch as engine
    engine.source_hashes=source_hashes
    for phase in ('smoke','trials'):
        dest=folder/phase;dest.mkdir();write_json(dest/'sources.json',hashes)
        if parent:
            stage=engine.continue_smoke(parent/phase/'plan.json',dest,same_sources=True)
        else:
            stage=phase_plan(dest,[a.seeds[0]] if phase=='smoke' else a.seeds,
                1 if phase=='smoke' else a.train_minutes,3 if phase=='smoke' else a.eval_episodes,
                3 if phase=='smoke' else a.eval_max_minutes)
        write_json(dest/'plan.json',stage)
    if parent:plan['parent_batch']=str(parent)
    write_json(folder/'plan.json',plan)
    print('OVERNIGHT PLAN:',folder/'plan.json',flush=True)
    print('All offline checks, then all 10 smoke training/evaluation pairs, then all full training, then evaluations.',flush=True)
    count=len(plan['cases'])*len(plan['seeds'])
    print(f"{count} fresh models x {plan['train_minutes']:g} minutes; {plan['eval_episodes']} evaluation episodes/model. "
          f"Batch scheduling budget: {str(plan['max_hours'])+' hours' if plan['max_hours'] else 'disabled'}. Completed stages are resumable.",flush=True)
    if a.execute:
        for suite in ('temporal_history','capture_pipeline'):
            result=subprocess.run([sys.executable,'-B','-s','-m','unittest','discover','-s',
                str(Path(__file__).resolve().parent.parent/suite),'-p','test_*.py','-q'],cwd=ROOT)
            if result.returncode:raise RuntimeError(f'{suite} offline gate failed; no browser launched')
        run(plan,folder,a.smoke_only)
    else:print('Plan only; add --execute to launch. Use --execute --smoke-only for short checks only.')

if __name__=='__main__':main()
