"""User-run 1–5 worker scaling suite, same source checkpoint, no dashboard."""
import argparse
from datetime import datetime
import hashlib
import json
from pathlib import Path
import subprocess
import sys
import time

from parallel_adapter import HERE,ROOT,DEFAULT_CHECKPOINT,sources
from results import read,validate,clean,write_summary


def digest(file):return hashlib.sha256(Path(file).read_bytes()).hexdigest()


def busy_processes():
    import psutil
    current=psutil.Process();own={current.pid};signature=current.cmdline()[1:];parent=current.parent()
    while parent is not None:
        try:
            if not parent.name().lower().startswith('python') or parent.cmdline()[1:]!=signature:break
            own.add(parent.pid);parent=parent.parent()
        except psutil.Error:break
    names={'benchmark_workers.py','run_workers.py','suite.py','capacity_suite.py','run_probe.py',
           'run_road_trial.py','run_trial.py','train_rs_v612_ppo.py','tools.train_ppo_parallel'}
    return [p.pid for p in psutil.process_iter(['name','cmdline']) if p.pid not in own
        and (p.info['name'] or '').lower().startswith('python')
        and any(Path(arg).name in names for arg in (p.info['cmdline'] or []))]


def jobs(output,checkpoint,minutes):
    cases=[('smoke_w1',1,1,'smoke')]+[(f'workers_{n}',n,minutes,'performance') for n in range(1,6)]
    cases.append(('repeat_w1',1,minutes,'repeat'))
    return [dict(name=name,workers=count,minutes=budget,phase=phase,output=str(output/name),
        command=[sys.executable,'-B','-s','-u',str(HERE/'run_workers.py'),'--workers',str(count),
            '--minutes',str(budget),'--checkpoint',str(checkpoint),'--output',str(output/name),
            '--expected-sources',str(output/'sources.json')]) for name,count,budget,phase in cases]


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--execute',action='store_true')
    p.add_argument('--checkpoint',type=Path,default=DEFAULT_CHECKPOINT)
    p.add_argument('--minutes',type=float,default=5)
    p.add_argument('--resume',type=Path)
    p.add_argument('--check-processes',action='store_true')
    a=p.parse_args()
    if a.check_processes:print(json.dumps({'busy':busy_processes()}));return
    if not 3<=a.minutes<=10:p.error('Performance budget must be 3..10 minutes')
    if a.execute and busy_processes():p.error('Another training/benchmark process is active; stop it and wait for the prompt')
    a.checkpoint=a.checkpoint.resolve()
    if not a.checkpoint.is_file():p.error('Matching five-frame checkpoint not found')
    manifest=sources();checkpoint_hash=digest(a.checkpoint)
    old=None
    if a.resume:
        old=read(a.resume/'plan.json')
        if read(a.resume/'sources.json')!=manifest or old['checkpoint_sha256']!=checkpoint_hash:
            p.error('Resume requires identical source code and checkpoint')
        if old['minutes']!=a.minutes:p.error('Resume requires the same performance budget')
    output=ROOT/'reports'/datetime.now().strftime('hybrid_workers_%Y%m%d_%H%M%S_%f')
    output.mkdir();(output/'sources.json').write_text(json.dumps(manifest,indent=2))
    plan=dict(checkpoint=str(a.checkpoint),checkpoint_sha256=checkpoint_hash,minutes=a.minutes,
              jobs=jobs(output,a.checkpoint,a.minutes))
    if old:
        lookup={job['name']:job for job in old['jobs']}
        for i,job in enumerate(plan['jobs']):
            previous=lookup.get(job['name'])
            if previous:
                try:validate(Path(previous['output']),job['workers'])
                except (OSError,ValueError,KeyError):continue
                plan['jobs'][i]=dict(previous,retained=True)
    (output/'plan.json').write_text(json.dumps(plan,indent=2))
    print('WORKER BENCHMARK:',output,flush=True)
    print('One-worker smoke; 1,2,3,4,5 workers; one-worker repeat. No dashboard.',flush=True)
    print('Remaining scheduled active minutes:',sum(j['minutes'] for j in plan['jobs'] if not j.get('retained')),flush=True)
    if not a.execute:print('Plan only; no game launched.');return
    # Local server check only; never contact the public game site.
    from urllib.request import urlopen
    try:
        with urlopen('http://127.0.0.1:8000/',timeout=5) as response:
            if response.status!=200:raise RuntimeError('Unexpected local server response')
    except Exception as exc:raise RuntimeError('Start the local server with python -B -s -u server.py first') from exc
    for directory in (HERE,HERE.parent/'hybrid_history_20261102'):
        subprocess.run([sys.executable,'-B','-s','-m','unittest','discover','-s',str(directory),'-p','test_*.py','-q'],cwd=ROOT,check=True)
    subprocess.run([sys.executable,'-B','-s',str(HERE/'offline_checks.py'),'--checkpoint',str(a.checkpoint)],cwd=ROOT,check=True)
    completed=[];failed=[];start=time.perf_counter()
    try:
        for index,job in enumerate(plan['jobs'],1):
            if sources()!=manifest or digest(a.checkpoint)!=checkpoint_hash:
                raise RuntimeError('Sources/checkpoint changed during benchmark')
            if job.get('retained'):
                validate(Path(job['output']),job['workers'])
                print(f'[SCALING {index}/7] RETAINED {job["name"]}',flush=True)
                completed.append(job['name']);continue
            print(f'[SCALING {index}/7] {job["name"]}: {job["workers"]} workers, {job["minutes"]} active minutes; '
                  f'elapsed={(time.perf_counter()-start)/60:.1f}m',flush=True)
            log=output/(job['name']+'.log')
            with log.open('x',encoding='utf8') as handle:
                process=subprocess.Popen(job['command'],cwd=ROOT,stdout=subprocess.PIPE,stderr=subprocess.STDOUT,
                    text=True,encoding='utf8',errors='replace')
                try:
                    for line in process.stdout:
                        print(line,end='',flush=True);handle.write(line);handle.flush()
                    code=process.wait()
                except BaseException:
                    print('Interrupted; waiting for child checkpoint/cleanup.',flush=True)
                    process.wait();raise
            error=None
            try:
                if code:raise ValueError(f'Process exited {code}')
                validate(Path(job['output']),job['workers'])
            except (OSError,ValueError,KeyError) as exc:error=repr(exc)
            if error:
                failed.append(dict(name=job['name'],error=error))
                r=read(Path(job['output'])/'parallel_training.json') if (Path(job['output'])/'parallel_training.json').exists() else {}
                if (job['phase']=='smoke' or not clean(Path(job['output']),job['workers']) or
                        'KeyboardInterrupt' in str(r.get('error')) or code in (130,-1073741510,3221225786)):
                    raise RuntimeError(f'{job["name"]} failed; batch stopped: {error}')
                print(f'[SCALING] {job["name"]} FAILED with confirmed cleanup; continuing independent cases',flush=True)
            else:completed.append(job['name'])
            (output/'stage_status.json').write_text(json.dumps(dict(completed=completed,failed=failed),indent=2))
            write_summary(output,plan)
    finally:
        (output/'stage_status.json').write_text(json.dumps(dict(completed=completed,failed=failed),indent=2))
        write_summary(output,plan)
        print('SUMMARY:',output/'scaling_summary.md',flush=True)
    if failed:raise RuntimeError('Some worker-count cases failed; inspect the retained reports')


if __name__=='__main__':main()
