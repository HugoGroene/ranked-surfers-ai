"""Shorter orchestration of unchanged hybrid probes; user launches live checks."""
import argparse
import copy
from datetime import datetime
import hashlib
import json
from pathlib import Path
import random
import subprocess
import sys
import zipfile

HERE=Path(__file__).resolve().parent
HYBRID=HERE.parent/'hybrid_history_20261102'
sys.path.insert(0,str(HYBRID))
import suite as full
from hybrid import LAYOUTS

REPRESENTATIVES=('recent4','recent3_context2','recent3_context3','recent4_context3','recent4_context4')


def value(command,flag):return command[command.index(flag)+1]


def signature(command):
    """Ignore artifact paths and budgets, never policy/configuration switches."""
    ignore={'--output','--expected-sources','--checkpoint','--minutes','--episodes'}
    result=[];i=0
    while i<len(command):
        if command[i] in ignore:i+=2
        else:result.append(command[i]);i+=1
    return result


def reuse_completed(wanted,old_plan):
    old_jobs={(j['name'],j['mode']):j for j in old_plan['jobs']}
    result=copy.deepcopy(wanted);retained=[];train_paths={}
    for index,job in enumerate(result['jobs']):
        if job['mode']=='eval':
            cmd=job['command'];cmd[cmd.index('--checkpoint')+1]=str(train_paths[job['name']]/'latest.zip')
        old=old_jobs.get((job['name'],job['mode']))
        if old is not None:
            try:
                if signature(old['command'])!=signature(job['command']):raise ValueError('Different configuration')
                if job['mode']=='train':
                    if float(value(old['command'],'--minutes'))<float(value(job['command'],'--minutes')):
                        raise ValueError('Insufficient old training budget')
                else:
                    if int(value(old['command'],'--episodes'))<int(value(job['command'],'--episodes')):
                        raise ValueError('Insufficient completed episodes')
                    if Path(value(old['command'],'--checkpoint')).resolve()!=Path(value(job['command'],'--checkpoint')).resolve():
                        raise ValueError('Evaluation belongs to another checkpoint')
                report=full.engine.checked_result(old)
                if report.get('finalization_errors')!=[]:raise ValueError('Cleanup not confirmed')
                if report.get('guard_hard_stopped'):raise ValueError('Guard stopped')
                folder=Path(old['output'])
                files=['trial_config.json','hybrid_history.json','temporal_capture.json']
                if job['mode']=='train':
                    with zipfile.ZipFile(folder/'latest.zip') as archive:
                        if archive.testzip() is not None:raise ValueError('Corrupt checkpoint')
                    files+=['latest.zip','trial_report.json','policy_inputs/index.json']
                else:files+=['v61_eval_report.json']
                retained_job=copy.deepcopy(old);retained_job['reused']=True
                result['jobs'][index]=job=retained_job
                for name in files:
                    file=folder/name
                    if file.exists():retained.append(dict(path=str(file),sha256=full.engine.file_hash(file)))
            except (OSError,ValueError,RuntimeError,KeyError,zipfile.BadZipFile):
                pass  # Keep partial outputs; rerun this stage in the new folder.
        if job['mode']=='train':train_paths[job['name']]=Path(job['output'])
    result['retained_artifacts']=retained
    return result


def phase_jobs(phase,folder):
    layouts=list(LAYOUTS) if phase=='smoke' else list(REPRESENTATIVES)
    cases=[(size,layout) for size in (256,384) for layout in layouts]
    random.Random(218).shuffle(cases)
    jobs=full.jobs(cases,folder,.5 if phase=='smoke' else 3,phase=='smoke')
    if phase=='smoke':
        for job in jobs:
            if job['mode']=='eval':
                cmd=job['command'];cmd[cmd.index('--episodes')+1]='1';cmd[cmd.index('--minutes')+1]='1'
    return {'jobs':jobs}


def busy_processes():
    import psutil
    current=psutil.Process()
    own={current.pid}
    # Windows venv python.exe redirects to the base interpreter and stays alive
    # with the same argv. Exclude only this exact invocation's Python ancestors,
    # never a sibling instance or an unrelated parent batch.
    command=current.cmdline()[1:]
    parent=current.parent()
    while parent is not None:
        try:
            if not parent.name().lower().startswith('python') or parent.cmdline()[1:]!=command:break
            own.add(parent.pid);parent=parent.parent()
        except psutil.Error:
            break
    names={'suite.py','capacity_suite.py','run_probe.py','run_road_trial.py','run_trial.py',
           'train_rs_v612_ppo.py','tools.train_ppo_parallel'}
    busy=[]
    for process in psutil.process_iter(['pid','name','cmdline']):
        if process.pid in own or not (process.info['name'] or '').lower().startswith('python'):continue
        if any(Path(arg).name in names for arg in (process.info['cmdline'] or [])):
            busy.append(process.pid)
    return busy


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--execute',action='store_true')
    p.add_argument('--check-processes',action='store_true',help='Read-only launcher/competition diagnosis; never starts a game')
    p.add_argument('--reuse',type=Path,help='Original hybrid or previous capacity folder; completed compatible stages only')
    a=p.parse_args()
    if a.check_processes:
        import psutil
        current=psutil.Process()
        chain=[current,*current.parents()]
        print(json.dumps(dict(process_chain=[dict(pid=item.pid,name=item.name(),command=item.cmdline())
            for item in chain if item.name().lower().startswith('python')],busy=busy_processes()),indent=2))
        return
    if a.execute:
        busy=busy_processes()
        if busy:p.error(f'Another benchmark/trainer is still running (PIDs {busy}). Stop it with Ctrl+C and wait for its terminal prompt first.')
    hashes=full.sources();full.engine.source_hashes=full.sources
    if a.reuse:
        a.reuse=a.reuse.resolve()
        if full.read(a.reuse/'sources.json')!=hashes:p.error('Probe sources changed; retained timings cannot be mixed')
    output=full.ROOT/'reports'/datetime.now().strftime('hybrid_capacity_%Y%m%d_%H%M%S_%f')
    output.mkdir();full.write(output/'sources.json',hashes)
    runner_sources={str(f):hashlib.sha256(f.read_bytes()).hexdigest() for f in HERE.glob('*.py')}
    full.write(output/'orchestrator_sources.json',runner_sources)
    full.write(output/'plan.json',dict(kind='capacity',parent=str(a.reuse) if a.reuse else None,
        smoke_layouts=list(LAYOUTS),performance_layouts=list(REPRESENTATIVES),sizes=[256,384],workers=1,
        smoke_minutes=.5,performance_minutes=3,smoke_eval_episodes=1))
    plans={}
    for phase in ('smoke','performance'):
        folder=output/phase;folder.mkdir();full.write(folder/'sources.json',hashes)
        plan=phase_jobs(phase,folder)
        if a.reuse:plan=reuse_completed(plan,full.read(a.reuse/phase/'plan.json'))
        plans[phase]=plan;full.write(folder/'plan.json',plan)
    jobs=[j for plan in plans.values() for j in plan['jobs']]
    retained=sum(bool(j.get('reused')) for j in jobs)
    minutes=sum(float(value(j['command'],'--minutes')) for j in jobs if j['mode']=='train' and not j.get('reused'))
    print('CAPACITY SUITE:',output,flush=True)
    print(f'Stages: {len(jobs)}; retained: {retained}; remaining scheduled training: {minutes:g} minutes',flush=True)
    if not a.execute:
        full.summary(output);print('Plan only. No live tests launched.');return
    for directory in (HERE,HYBRID,full.ROAD):
        subprocess.run([sys.executable,'-B','-s','-m','unittest','discover','-s',str(directory),'-p','test_*.py','-q'],
                       cwd=full.ROOT,check=True)
    for layout in ('recent4_context4','sparse5','sparse4'):
        subprocess.run([sys.executable,'-B','-s',str(HYBRID/'offline_integration.py'),'--layout',layout],cwd=full.ROOT,check=True)
    progress=full.BatchProgress(len(jobs))
    try:
        for phase in ('smoke','performance'):
            if any(hashlib.sha256(Path(path).read_bytes()).hexdigest()!=digest for path,digest in runner_sources.items()):
                raise RuntimeError('Orchestrator changed during suite')
            progress.phase=phase
            full.engine.execute(plans[phase],output/phase,keep_going=phase=='performance',progress=progress)
    finally:
        full.summary(output)
        print('SUMMARY:',output/'performance_summary.md',flush=True)
        print('INPUT GALLERY:',output/'policy_input_gallery.html',flush=True)


if __name__=='__main__':main()
