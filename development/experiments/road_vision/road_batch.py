"""Plan-only by default: smoke gate, all full training, then matched evaluation."""
import argparse
import json
from datetime import datetime
from pathlib import Path
import random
import subprocess
import sys
import time
from road_config import ROOT,HERE,PRESETS,DEFAULT_ARMS,configuration,source_hashes,write_json

def jobs(configs,seeds,minutes,episodes,eval_minutes,output,temporal,bc_root=None,bc_epochs=5,capture_backend='gpu_100'):
    arms=[(arm,seed,False) for arm in configs for seed in seeds]
    if bc_root:
        arms += [(arm,seed,True) for arm in configs if PRESETS[arm][2]=='spatial' for seed in seeds]
    random.Random(42).shuffle(arms)
    training=[];evaluation=[]
    for arm,seed,bc in arms:
        name=f'{arm}_{"bc" if bc else "scratch"}_seed{seed}'
        base=[sys.executable,'-B','-s','-u',str(HERE/'run_road_trial.py'),'--arm',arm,'--seed',str(seed),
              '--capture-backend',capture_backend]
        for key,value in temporal.items():base+=['--'+key.replace('_','-'),str(value)]
        base+=['--expected-sources',str(output/'sources.json')]
        train_dir=output/name/'train';eval_dir=output/name/'eval'
        command=base+['--mode','train','--minutes',str(minutes),'--output',str(train_dir)]
        if minutes<=1:command+=['--crop-previews']
        if bc:command+=['--bc-root',str(bc_root),'--allow-downsampled-replays','--bc-epochs',str(bc_epochs)]
        training.append(dict(name=name,mode='train',output=str(train_dir),command=command))
        evaluation.append(dict(name=name,mode='eval',output=str(eval_dir),command=base+[
            '--mode','eval','--minutes',str(eval_minutes),'--episodes',str(episodes),
            '--checkpoint',str(train_dir/'latest.zip'),'--output',str(eval_dir)]))
    return training+evaluation

def summarize_phase(engine,plan):
    result=engine.summarize(plan)
    for row in result['rows']:
        job=next(j for j in plan['jobs'] if j['name']==row['name'] and j['mode']=='train')
        path=Path(job['output'])/'trial_report.json'
        if path.exists():
            report=json.loads(path.read_text())
            for key in ('initialization','bc_seconds','total_stage_seconds','road_vision','capture_backend'):
                row[key]=report.get(key)
    return result

class BatchBudgetReached(RuntimeError):
    pass

def check_stage_budget(job,deadline):
    if deadline is None:return
    command=job['command']
    # Reserve the complete configured stage plus startup/finalization allowance.
    reserve=float(command[command.index('--minutes')+1])*60+180
    if time.monotonic()+reserve>deadline:
        raise BatchBudgetReached('Batch wall budget reached; no new stage launched. Resume to finish remaining stages.')

def run(plan,output,smoke_only=False):
    import batch as engine
    engine.source_hashes=source_hashes  # Process-local reuse; no active source edits.
    started=time.monotonic()
    deadline=started+plan['max_hours']*3600 if plan.get('max_hours') else None
    test=subprocess.run([sys.executable,'-B','-s','-m','unittest','discover','-s',str(HERE),'-p','test_*.py','-q'],cwd=ROOT)
    if test.returncode:raise RuntimeError('Offline gate failed; no live stage launched')
    if source_hashes()!=json.loads((output/'sources.json').read_text()):raise RuntimeError('Source changed since plan')
    if plan.get('bc_root'):
        from road_bc import inspect_demos
        inspect_demos(plan['bc_root'],True)
    status='RUNNING'
    try:
        for phase in ('smoke','trials'):
            if smoke_only and phase=='trials':break
            folder=output/phase;phase_plan=json.loads((folder/'plan.json').read_text())
            engine.execute(phase_plan,folder,keep_going=(phase=='trials'),
                           before_job=lambda job:check_stage_budget(job,deadline))
        status='SMOKE_COMPLETE' if smoke_only else 'COMPLETE'
    except BatchBudgetReached:
        status='BUDGET_LIMIT'
        raise
    except BaseException:
        status='STOPPED'
        raise
    finally:
        result={phase:summarize_phase(engine,json.loads((output/phase/'plan.json').read_text())) for phase in ('smoke','trials')}
        result['note']='Smoke is functional validation only. BC cost is separate from PPO wall budget. No automatic winner promotion.'
        write_json(output/'comparison.json',result)
        write_json(output/'batch_status.json',dict(status=status,elapsed_seconds=time.monotonic()-started,
                   max_hours=plan.get('max_hours'),note='Budget checked before stages; running stages finalize normally. Not a process-kill watchdog.'))

def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--arms',nargs='+',choices=list(PRESETS),default=DEFAULT_ARMS)
    p.add_argument('--seeds',nargs='+',type=int,default=[123])
    p.add_argument('--frames',type=int,default=4)
    p.add_argument('--span-ms',type=float,default=500)
    p.add_argument('--capture-hz',type=float,default=60)
    p.add_argument('--decision-hz',type=float,default=30)
    p.add_argument('--train-minutes',type=float,default=30)
    p.add_argument('--eval-episodes',type=int,default=100)
    p.add_argument('--eval-max-minutes',type=float,default=60)
    p.add_argument('--bc-root',type=Path,help='Add matched BC-initialized spatial arms; requires compatible approved recordings')
    p.add_argument('--allow-downsampled-replays',action='store_true')
    p.add_argument('--bc-epochs',type=int,default=5)
    p.add_argument('--output',type=Path)
    p.add_argument('--resume',type=Path,help='Previous road batch folder; only identical source versions can be reused')
    p.add_argument('--plan',type=Path,help='Execute an already written plan')
    p.add_argument('--execute',action='store_true')
    p.add_argument('--smoke-only',action='store_true',help='Run offline gates and smoke only; resume later to reuse passed stages')
    p.add_argument('--capture-backend',choices=['native_bitmap','bitmap_100','gpu_100'],default='gpu_100')
    a=p.parse_args()
    if a.plan and a.resume:p.error('Choose plan or resume')
    if a.plan:
        output=a.plan.resolve().parent;plan=json.loads(a.plan.read_text())
    else:
        if not 0<a.train_minutes<=2880 or not 0<a.eval_max_minutes<=2880 or a.eval_episodes<1:p.error('Invalid budget')
        if not 1<=a.bc_epochs<=30:p.error('BC epochs must be 1..30')
        if len(set(a.arms))!=len(a.arms) or len(set(a.seeds))!=len(a.seeds) or any(not 0<=s<2**32 for s in a.seeds):p.error('Invalid/duplicate arms or seeds')
        t=dict(frames=a.frames,span_ms=a.span_ms,capture_hz=a.capture_hz,decision_hz=a.decision_hz)
        configs={arm:configuration(arm,**t) for arm in a.arms}
        if a.bc_root:
            from road_bc import inspect_demos
            inspect_demos(a.bc_root,a.allow_downsampled_replays)
        output=(a.output or ROOT/'reports'/('road_batch_'+datetime.now().strftime('%Y%m%d_%H%M%S_%f'))).resolve()
        if not output.is_relative_to(ROOT/'reports') or output.exists():p.error('Use a new reports directory')
        hashes=source_hashes()
        if a.resume:
            old=a.resume.resolve()
            if json.loads((old/'sources.json').read_text())!=hashes:p.error('Changed sources; old batch remains separate evidence')
            plan=json.loads((old/'plan.json').read_text());plan['parent_batch']=str(old)
        else:plan=dict(configs=configs,seeds=a.seeds,temporal=t,bc_root=str(a.bc_root.resolve()) if a.bc_root else None,
                       train_minutes=a.train_minutes,eval_episodes=a.eval_episodes,eval_max_minutes=a.eval_max_minutes,
                       bc_epochs=a.bc_epochs,capture_backend=a.capture_backend)
        output.mkdir(parents=True);write_json(output/'sources.json',hashes)
        import batch as engine
        engine.source_hashes=source_hashes
        for phase in ('smoke','trials'):
            folder=output/phase;folder.mkdir();write_json(folder/'sources.json',hashes)
            if a.resume:
                phase_plan=engine.continue_smoke(old/phase/'plan.json',folder,same_sources=True)
            else:
                minutes,episodes,eval_minutes=(1,3,3) if phase=='smoke' else (a.train_minutes,a.eval_episodes,a.eval_max_minutes)
                phase_plan=dict(configs=configs,seeds=a.seeds,train_minutes=minutes,eval_episodes=episodes,
                    jobs=jobs(a.arms,a.seeds,minutes,episodes,eval_minutes,folder,t,
                              a.bc_root.resolve() if a.bc_root and phase=='trials' else None,a.bc_epochs,a.capture_backend))
            write_json(folder/'plan.json',phase_plan)
        write_json(output/'plan.json',plan)
    print('ROAD PLAN:',output/'plan.json',flush=True)
    print('Read crop previews from smoke before accepting a crop for long training. No automated lane-visibility claim.',flush=True)
    if a.execute:run(plan,output,a.smoke_only)
    else:print('Plan only. No browser/model launched. Execute later with --plan <path> --execute.')

if __name__=='__main__':main()
