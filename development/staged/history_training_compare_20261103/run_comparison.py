"""Two fresh 260-minute training arms; both smokes first, evaluation deferred."""
import argparse
from datetime import datetime
import hashlib
import html
import json
import os
from pathlib import Path
import subprocess
import sys
import time
import zipfile

from parallel_adapter import HERE,ROOT,LAYOUTS,sources
from results import read,validate,clean,write_summary
from resume_state import continuation


def digest(path):return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def busy_processes():
    import psutil
    current=psutil.Process();own={current.pid};signature=current.cmdline()[1:];parent=current.parent()
    while parent is not None:
        try:
            if not parent.name().lower().startswith('python') or parent.cmdline()[1:]!=signature:break
            own.add(parent.pid);parent=parent.parent()
        except psutil.Error:break
    names={'run_comparison.py','benchmark_workers.py','run_workers.py','suite.py','capacity_suite.py','run_probe.py',
        'run_road_trial.py','run_trial.py','train_rs_v612_ppo.py','tools.train_ppo_parallel',
        'batch.py','run_short_checks.py','short_checks.py'}
    return [p.pid for p in psutil.process_iter(['name','cmdline']) if p.pid not in own
        and (p.info['name'] or '').lower().startswith('python')
        and any(Path(arg).name in names for arg in (p.info['cmdline'] or []))]


def jobs(output,checkpoints,minutes):
    cases=[(layout,spec['workers'],1,'smoke') for layout,spec in LAYOUTS.items()]
    cases.extend((layout,spec['workers'],minutes,'train') for layout,spec in LAYOUTS.items())
    result=[]
    for layout,count,budget,phase in cases:
        name=f'{layout}_{phase}_w{count}'
        result.append(dict(name=name,layout=layout,workers=count,minutes=budget,phase=phase,
            total_budget_minutes=budget,carried_seconds=0.,source_timesteps=0,prior_attempts=[],
            output=str(output/name),command=[sys.executable,'-B','-s','-u',str(HERE/'run_workers.py'),
                '--workers',str(count),'--minutes',str(budget),'--checkpoint',str(checkpoints[layout]),
                '--output',str(output/name),'--expected-sources',str(output/'sources.json')]))
    return result


def comparison_summary(output,plan):
    rows=[]
    for job in plan['jobs']:
        if job['phase']!='train':continue
        folder=Path(job['output']);path=folder/'parallel_training.json'
        r=read(path) if path.exists() else {}
        row=dict(layout=job['layout'],workers=job['workers'],status=r.get('status','NOT_STARTED'),
            active_seconds=job['carried_seconds']+r.get('active_wall_seconds',0.),
            optimized_transitions=job['source_timesteps']+r.get('optimized_transitions',0),
            checkpoint=str(folder/'latest.zip') if (folder/'latest.zip').exists() else None,
            report=str(path),prior_attempts=job['prior_attempts'],evaluation='PENDING')
        rows.append(row)
    (output/'training_comparison.json').write_text(json.dumps(dict(arms=rows,
        note='Equal active wall-time budgets; different worker counts/cohort sizes. Fresh seed123 models, no BC. Evaluate matching contracts before judging skill.'),indent=2))
    text=['# Training comparison (evaluation pending)','',
        '| Arm | Workers | Status | Active minutes | Optimized transitions |',
        '|---|---:|---|---:|---:|']
    for row in rows:
        text.append(f'| {row["layout"]} | {row["workers"]} | {row["status"]} | {row["active_seconds"]/60:.2f} | {row["optimized_transitions"]} |')
    text+=['','Fresh models with seed 123. Equal time compares the complete configurations, not just frame count.',
        'No automatic gameplay evaluation. Checkpoint paths are in training_comparison.json.',
        'Training episode means are not matched evaluation scores. Resume segments have separate logs/curves.']
    (output/'training_comparison.md').write_text('\n'.join(text),encoding='utf8')


def gallery(output,plan):
    parts=['<!doctype html><meta charset="utf-8"><title>Fixed spacing inputs</title>',
        '<style>body{background:#181818;color:#eee;font-family:sans-serif}img{max-width:100%}a{color:#8cf}</style>',
        '<h1>Exact smoke-test policy inputs</h1><p>Oldest to newest. Labels show measured ages, not requested ages. '
        'Black invalid slots are missing history. These are raw actor observations, exported after gameplay.</p>']
    for job in plan['jobs']:
        if job['phase']!='smoke':continue
        folder=Path(job['output'])/'worker_0'/'policy_inputs'
        for sheet in sorted(folder.glob('*_sheet.png')):
            relative=os.path.relpath(sheet,output).replace('\\','/')
            metadata=relative.replace('_sheet.png','.json')
            parts.append(f'<h2>{html.escape(job["name"])} / {sheet.stem}</h2>'
                f'<a href="{html.escape(metadata,quote=True)}">Metadata</a>'
                f'<p><img loading="lazy" src="{html.escape(relative,quote=True)}"></p>')
    (output/'policy_input_gallery.html').write_text('\n'.join(parts),encoding='utf8')


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--execute',action='store_true')
    p.add_argument('--minutes',type=float,default=260,help='Active training minutes per arm, default 260 = 4h20m')
    p.add_argument('--resume',type=Path)
    p.add_argument('--check-processes',action='store_true')
    a=p.parse_args()
    if a.check_processes:print(json.dumps({'busy':busy_processes()}));return
    if not 1<=a.minutes<=2880:p.error('Per-arm budget must be 1..2880 minutes')
    if a.execute and busy_processes():p.error('Another benchmark/trainer is active; stop it and wait for the prompt')
    manifest=sources();old=None
    if a.resume:
        a.resume=a.resume.resolve();old=read(a.resume/'plan.json')
        if read(a.resume/'sources.json')!=manifest or old['minutes']!=a.minutes:
            p.error('Resume requires identical source and budget')
    output=ROOT/'reports'/datetime.now().strftime('history_training_compare_%Y%m%d_%H%M%S_%f')
    output.mkdir();(output/'sources.json').write_text(json.dumps(manifest,indent=2))
    checkpoints=({key:Path(value) for key,value in old['checkpoints'].items()} if old else
        {key:output/'initial'/f'{key}.zip' for key in LAYOUTS})
    plan=dict(minutes=a.minutes,checkpoints={k:str(v) for k,v in checkpoints.items()},
        initialization='scratch_seed123_per_layout',layouts=LAYOUTS,jobs=jobs(output,checkpoints,a.minutes))
    if old:
        if {k:digest(v) for k,v in checkpoints.items()}!=old['checkpoint_sha256']:
            p.error('Initial checkpoints changed')
        lookup={job['name']:job for job in old['jobs']}
        for index,job in enumerate(plan['jobs']):
            previous=lookup.get(job['name'])
            if previous:
                try:validate(Path(previous['output']),job['workers'])
                except (OSError,ValueError,KeyError,zipfile.BadZipFile):
                    if job['phase']=='train':plan['jobs'][index]=continuation(previous,job)
                    continue
                plan['jobs'][index]=dict(previous,retained=True)
        plan['checkpoint_sha256']=old['checkpoint_sha256']
    plan_file=output/'plan.json';plan_file.write_text(json.dumps(plan,indent=2))
    print('TRAINING COMPARISON:',output,flush=True)
    print('Both smokes first; fixed4 x 4 workers then fixed5 x 3 workers. Fresh initialization; no dashboard or detector.',flush=True)
    print('Remaining scheduled active minutes:',sum(j['minutes'] for j in plan['jobs'] if not j.get('retained')),flush=True)
    if not a.execute:print('Plan only; no game or checkpoint creation.');return
    from urllib.request import urlopen
    try:
        with urlopen('http://127.0.0.1:8000/',timeout=5) as response:
            if response.status!=200:raise RuntimeError('Unexpected local server response')
    except Exception as exc:raise RuntimeError('Start python -B -s -u server.py first') from exc
    for directory in (HERE,HERE.parent/'hybrid_history_20261102'):
        subprocess.run([sys.executable,'-B','-s','-m','unittest','discover','-s',str(directory),
            '-p','test_*.py','-q'],cwd=ROOT,check=True)
    if not old:
        for layout,path in checkpoints.items():
            subprocess.run([sys.executable,'-B','-s',str(HERE/'prepare_checkpoint.py'),
                '--layout',layout,'--output',str(path)],cwd=ROOT,check=True)
        plan['checkpoint_sha256']={k:digest(v) for k,v in checkpoints.items()}
        plan_file.write_text(json.dumps(plan,indent=2))
    for layout,path in checkpoints.items():
        env=dict(os.environ,RS_COMPARISON_LAYOUT=layout)
        subprocess.run([sys.executable,'-B','-s',str(HERE/'offline_checks.py'),'--checkpoint',str(path)],
            env=env,cwd=ROOT,check=True)
    completed=[];failed=[];start=time.perf_counter()
    try:
        for index,job in enumerate(plan['jobs'],1):
            if sources()!=manifest or {k:digest(v) for k,v in checkpoints.items()}!=plan['checkpoint_sha256']:
                raise RuntimeError('Sources/initial checkpoints changed')
            if job.get('resume_sha256') and digest(job['resumed_from'])!=job['resume_sha256']:
                raise RuntimeError('Resume source checkpoint changed')
            if job.get('retained'):
                validate(Path(job['output']),job['workers']);completed.append(job['name'])
                print(f'[SCALING {index}/{len(plan["jobs"])}] RETAINED {job["name"]}',flush=True);continue
            print(f'[COMPARISON {index}/{len(plan["jobs"])}] {job["name"]}; {job["minutes"]:.2f} active minutes; '
                f'elapsed={(time.perf_counter()-start)/60:.1f}m',flush=True)
            env=dict(os.environ,RS_COMPARISON_LAYOUT=job['layout'],RS_COMPARISON_INPUTS='1' if job['phase']=='smoke' else '0')
            with (output/(job['name']+'.log')).open('x',encoding='utf8') as handle:
                process=subprocess.Popen(job['command'],cwd=ROOT,env=env,stdout=subprocess.PIPE,
                    stderr=subprocess.STDOUT,text=True,encoding='utf8',errors='replace')
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
            except (OSError,ValueError,KeyError,zipfile.BadZipFile) as exc:error=repr(exc)
            if error:
                failed.append(dict(name=job['name'],error=error))
                report=Path(job['output'])/'parallel_training.json';r=read(report) if report.exists() else {}
                if (job['phase']=='smoke' or not clean(Path(job['output']),job['workers']) or
                        'KeyboardInterrupt' in str(r.get('error')) or code in (130,-1073741510,3221225786)):
                    raise RuntimeError(f'{job["name"]} failed; stopping: {error}')
                print('[SCALING] Failed with confirmed cleanup; continuing independent cases',flush=True)
            else:completed.append(job['name'])
            (output/'stage_status.json').write_text(json.dumps(dict(completed=completed,failed=failed),indent=2))
            write_summary(output,plan);gallery(output,plan);comparison_summary(output,plan)
    finally:
        (output/'stage_status.json').write_text(json.dumps(dict(completed=completed,failed=failed),indent=2))
        write_summary(output,plan);gallery(output,plan);comparison_summary(output,plan)
        print('SUMMARY:',output/'training_comparison.md',flush=True)
        print('CHECKPOINTS:',output/'training_comparison.json',flush=True)
        print('INPUT GALLERY:',output/'policy_input_gallery.html',flush=True)
    if failed:raise RuntimeError('Some cases failed; completed results preserved')


if __name__=='__main__':main()
