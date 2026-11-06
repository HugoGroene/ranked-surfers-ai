"""Evaluate completed overnight models without launching any training stage."""
import argparse
import copy
from datetime import datetime
import hashlib
import json
from pathlib import Path
import subprocess
import sys
import zipfile

HERE=Path(__file__).resolve().parent
ROOT=HERE.parents[2]
ROAD=ROOT/'development/experiments/road_vision'
sys.path.insert(0,str(ROOT))
sys.path.insert(0,str(ROAD))
from road_config import source_hashes
import batch as engine
from batch_progress import BatchProgress

DEFAULT_BATCH=ROOT/'reports/road_overnight_20261102_001333_968036'

def read(path):return json.loads(Path(path).read_text(encoding='utf-8'))
def write(path,data):
    with Path(path).open('x',encoding='utf-8') as f:json.dump(data,f,indent=2)
def digest(path):
    h=hashlib.sha256()
    with Path(path).open('rb') as f:
        for chunk in iter(lambda:f.read(1024*1024),b''):h.update(chunk)
    return h.hexdigest()
def option(command,key):return command[command.index(key)+1]
def replace(command,key,value):command[command.index(key)+1]=str(value)

def select_completed(plan,scope='vision',seed=None):
    """Only equal-budget audited final models; never promote a power-cut checkpoint."""
    trains=[];evals=[];omitted=[]
    evaluations={j['name']:j for j in plan['jobs'] if j['mode']=='eval'}
    for job in plan['jobs']:
        if job['mode']!='train':continue
        name=job['name'];cmd=job['command']
        if scope=='vision' and option(cmd,'--spacing')!='uniform':continue
        if seed is not None and int(option(cmd,'--seed'))!=seed:continue
        folder=Path(job['output']);file=folder/'trial_report.json'
        r=read(file) if file.exists() else {}
        budget=float(option(cmd,'--minutes'))*60
        safe=(r.get('status')=='COMPLETE' and r.get('audit',{}).get('status')=='PASS_SETTLED_LEARNING'
              and r.get('finalization_errors')==[] and r.get('training_seconds',0)>=budget
              and (folder/'latest.zip').is_file() and name in evaluations)
        if not safe:
            omitted.append({'name':name,'reason':'Missing/incomplete training, audit, budget, checkpoint or cleanup'})
            continue
        with zipfile.ZipFile(folder/'latest.zip') as z:
            if 'policy.pth' not in z.namelist():raise RuntimeError(f'Missing policy: {name}')
            if 'data' in z.namelist():
                saved=json.loads(z.read('data')).get('num_timesteps')
                if saved!=r['optimized_transitions']:raise RuntimeError(f'Checkpoint/report step mismatch: {name}')
        kept=copy.deepcopy(job);kept['reused']=True
        trains.append(kept);evals.append(copy.deepcopy(evaluations[name]))
    return trains,evals,omitted

def prepare(parent,output,scope,seed,episodes,resume=None):
    original=read(parent/'trials/plan.json')
    hashes=source_hashes()
    if read(parent/'sources.json')!=hashes:raise RuntimeError('Active sources differ from training; refusing a silent mismatch')
    trains,evals,omitted=select_completed(original,scope,seed)
    if not evals:raise ValueError('No eligible completed models in requested selection')
    previous={}
    if resume:
        old=read(resume/'plan.json')
        if read(resume/'sources.json')!=hashes or old['episodes']!=episodes:
            raise ValueError('Resume requires identical sources and episode budget')
        previous={j['name']:j for j in old['jobs'] if j['mode']=='eval'}
    checkpoint_hashes={}
    for job in evals:
        command=job['command'];checkpoint=Path(option(command,'--checkpoint'))
        checkpoint_hashes[str(checkpoint)]=digest(checkpoint)
        destination=output/job['name']/'eval'
        job['output']=str(destination)
        replace(command,'--output',destination)
        replace(command,'--episodes',episodes)
        replace(command,'--expected-sources',output/'sources.json')
        old=previous.get(job['name'])
        if old:
            try:
                r=engine.checked_result(old)
                old_checkpoint=Path(option(old['command'],'--checkpoint'))
                recorded_hash=read(resume/'plan.json')['checkpoint_hashes'].get(str(old_checkpoint))
                if (recorded_hash==checkpoint_hashes[str(checkpoint)] and
                        int(option(old['command'],'--episodes'))==episodes and
                        r.get('episodes')==episodes):
                    job.update(copy.deepcopy(old));job['reused']=True
            except (OSError,KeyError,ValueError,RuntimeError):pass
    output.mkdir(parents=True,exist_ok=False)
    result={'parent_batch':str(parent),'scope':scope,'seed':seed,'episodes':episodes,
            'jobs':trains+evals,'omitted':omitted,'checkpoint_hashes':checkpoint_hashes,
            'note':'Only final completed equal-budget models. Partial training excluded. Reused training stages never execute.'}
    write(output/'sources.json',hashes);write(output/'plan.json',result)
    return result

def execute(plan,output):
    if any(j['mode']=='train' and not j.get('reused') for j in plan['jobs']):
        raise RuntimeError('Evaluation-only runner refuses live training jobs')
    engine.source_hashes=source_hashes
    # All offline checks precede live evaluation. Existing completed smoke evidence
    # remains in the parent batch; models and observation contracts are checked by each child.
    for folder,pattern in ((HERE,'test_*.py'),(ROAD,'test_*.py')):
        subprocess.run([sys.executable,'-B','-s','-m','unittest','discover','-s',str(folder),
                        '-p',pattern,'-q'],cwd=ROOT,check=True)
    def before(job):
        checkpoint=Path(option(job['command'],'--checkpoint'))
        if digest(checkpoint)!=plan['checkpoint_hashes'][str(checkpoint)]:
            raise RuntimeError('Checkpoint changed after planning')
    progress=BatchProgress(len(plan['jobs']));progress.phase='recovered'
    engine.execute(plan,output,keep_going=True,before_job=before,progress=progress)

def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--parent',type=Path,default=DEFAULT_BATCH)
    p.add_argument('--scope',choices=['vision','all'],default='vision')
    p.add_argument('--seed',type=int,help='Optional filter; 123 alone omits the completed 384 model')
    p.add_argument('--episodes',type=int,default=200)
    p.add_argument('--resume',type=Path,help='Retain complete evaluations from an earlier recovery folder')
    p.add_argument('--plan',type=Path,help='Execute an existing recovery plan')
    p.add_argument('--execute',action='store_true')
    a=p.parse_args()
    if a.episodes<1:p.error('Episodes must be positive')
    if a.plan:
        plan=read(a.plan);output=a.plan.resolve().parent
    else:
        output=ROOT/'reports'/datetime.now().strftime('recovered_evaluations_%Y%m%d_%H%M%S_%f')
        plan=prepare(a.parent.resolve(),output,a.scope,a.seed,a.episodes,a.resume.resolve() if a.resume else None)
    print('RECOVERY PLAN:',output/'plan.json',flush=True)
    print('Evaluations:',sum(j['mode']=='eval' for j in plan['jobs']),
          '| training runs reused:',sum(j['mode']=='train' for j in plan['jobs']),flush=True)
    for job in plan['jobs']:
        if job['mode']=='eval':print('  ',job['name'], '(retained)' if job.get('reused') else '')
    if a.execute:execute(plan,output)
    else:print('Plan only; no browser, game, training or evaluation started.')

if __name__=='__main__':main()
