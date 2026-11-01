"""All calibration tests, then frozen one/three-worker capture comparisons."""
import argparse
from datetime import datetime
import hashlib
import json
from pathlib import Path
import subprocess
import sys
import zipfile
import statistics
from adapter import ROOT,HERE,ARMS

DEFAULT_CHECKPOINT=ROOT/'reports/v612_3workers_menu_recovery_24h_20261101_111341/latest.zip'

def checkpoint_info(path):
    path=Path(path)
    with zipfile.ZipFile(path) as archive:
        if archive.testzip() is not None:raise ValueError('Checkpoint CRC failure')
        data=json.loads(archive.read('data'))
    return dict(path=str(path.resolve()),sha256=hashlib.sha256(path.read_bytes()).hexdigest(),
                num_timesteps=data['num_timesteps'])

def command(arm,workers,minutes,checkpoint,output):
    return [sys.executable,'-B','-s','-u',str(HERE/'run_frozen.py'),'--arm',arm,
        '--benchmark-only','--headless','--headless-capture','electron',
        '--electron-transport','shared_memory','--rollout-transport','shared_memory',
        '--displays',*[str(i+1) for i in range(workers)],'--checkpoint',str(checkpoint),
        '--framing-profile','canvas_v1','--canvas-policy-input','--detector-mode','off',
        '--reuse-frame-state','--opencv-threads','1','--profile-reaction','--telemetry',
        '--worker-seed-base','123','--steps','768000','--max-minutes',str(minutes),
        '--worker-timeout','160','--end-plot','off','--report',str(output)]

def run(command,log):
    with log.open('x',encoding='utf8') as handle:
        process=subprocess.Popen(command,cwd=ROOT,stdout=handle,stderr=subprocess.STDOUT)
        try:code=process.wait()
        except BaseException:
            # Let the collector receive console Ctrl+C and finalize; do not kill unrelated processes.
            try:process.wait(timeout=30)
            except subprocess.TimeoutExpired:process.terminate();process.wait(timeout=10)
            raise
    if code:raise RuntimeError(f'Stage failed ({code}); inspect {log}')

def capture_summary(folder):
    """Summarize saved timings after workers stop; never in the decision path."""
    values={}
    for ledger in sorted(Path(folder).glob('worker_*/learning_transitions.jsonl')):
        for line in ledger.open(encoding='utf8'):
            row=json.loads(line)
            capture=row.get('next_capture') or {}
            for name,value in (capture.get('capture_profile') or {}).items():
                if isinstance(value,(int,float)) and name.endswith('_ms'):
                    values.setdefault(name,[]).append(value)
    def summary(items):
        items=sorted(items)
        return dict(count=len(items),median=statistics.median(items),p95=items[min(len(items)-1,int(.95*len(items)))])
    return {name:summary(items) for name,items in values.items()}

def main(argv=None):
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--checkpoint',type=Path,default=DEFAULT_CHECKPOINT)
    p.add_argument('--minutes',type=float,default=1.)
    p.add_argument('--workers',type=int,nargs='+',default=[1,3])
    p.add_argument('--arms',nargs='+',choices=list(ARMS),default=list(ARMS))
    p.add_argument('--repeats',type=int,default=1)
    p.add_argument('--output',type=Path)
    p.add_argument('--execute',action='store_true')
    a=p.parse_args(argv)
    if not .25<=a.minutes<=5 or any(not 1<=w<=6 for w in a.workers):p.error('Use .25..5 minutes and 1..6 workers')
    if a.repeats not in (1,2,3) or len(set(a.arms))!=len(a.arms):p.error('Use 1..3 repeats and distinct arms')
    a.output=a.output or ROOT/'reports'/('capture_matrix_'+datetime.now().strftime('%Y%m%d_%H%M%S_%f'))
    source=checkpoint_info(a.checkpoint)
    a.output.mkdir(parents=True,exist_ok=False)
    files=list(ROOT.glob('*.py'))+[f for d in [HERE,ROOT/'tools'] for f in d.rglob('*')
        if f.suffix in ('.py','.cjs') and 'node_modules' not in f.parts]
    files.append(ROOT/'rankedsurfers-downloaded/game/releases/ae44e7782a4c8bbe5b10/arena.html')
    sources={str(f.relative_to(ROOT)):hashlib.sha256(f.read_bytes()).hexdigest() for f in files}
    (a.output/'sources.json').write_text(json.dumps(sources,indent=2))
    plan=[]
    for arm in a.arms:
        plan.append(dict(kind='calibration',arm=arm,command=[sys.executable,'-B','-s','-u',str(HERE/'probe.py'),
            '--arm',arm,'--output',str(a.output/f'calibration_{arm}')]))
    for repeat in range(a.repeats):
        for workers in (a.workers if repeat%2==0 else list(reversed(a.workers))):
            # Repeat bitmap control and reverse arm order on the next repetition.
            arms=a.arms if repeat%2==0 else list(reversed(a.arms))
            for i,arm in enumerate([*arms,'bitmap_100']):
                output=a.output/f'r{repeat}_w{workers}_{i:02d}_{arm}'
                plan.append(dict(kind='frozen',arm=arm,workers=workers,repeat=repeat,output=str(output),
                    command=command(arm,workers,a.minutes,a.checkpoint,output)))
    report=dict(status='PLANNED',checkpoint=source,model_updated=False,plan=plan,stages=[],
        note='Frozen stochastic collection, not optimized throughput or a policy-quality evaluation. '
             'GPU half readback changes effective detail and restores guard geometry. No automatic backend promotion.')
    target=a.output/'comparison.json'
    target.write_text(json.dumps(report,indent=2))
    print('CAPTURE MATRIX: '+str(a.output),flush=True)
    if not a.execute:return
    try:
        report['status']='RUNNING'
        for i,job in enumerate(plan):
            if checkpoint_info(a.checkpoint)!=source:raise RuntimeError('Source checkpoint changed')
            for relative,digest in sources.items():
                if hashlib.sha256((ROOT/relative).read_bytes()).hexdigest()!=digest:raise RuntimeError('Source changed: '+relative)
            print(f'[{i+1}/{len(plan)}] {job["kind"]} {job["arm"]} workers={job.get("workers",0)}',flush=True)
            run(job['command'],a.output/f'{i:02d}_{job["kind"]}_{job["arm"]}.log')
            if job['kind']=='frozen':
                folder=Path(job['output']);result=json.loads((folder/'parallel_training.json').read_text())
                if (result['status']!='COMPLETE_FROZEN_POLICY_DIAGNOSTIC' or result['optimized_transitions']!=0
                    or result.get('model_updated') is not False or not result.get('policy_unchanged')
                    or not result.get('source_unchanged')):raise RuntimeError('Frozen-policy audit failed')
                workers=[json.loads(f.read_text()) for f in sorted(folder.glob('worker_*/result.json'))]
                if len(workers)!=job['workers'] or any(w.get('guard_hard_stopped') for w in workers):raise RuntimeError('Worker guard failed')
                render_configs=[json.loads(f.read_text()) for f in sorted(folder.glob('worker_*/electron/render_config.json'))]
                if len(render_configs)!=job['workers']:raise RuntimeError('Missing loaded Unity scale/renderer verification')
                report['stages'].append(dict(arm=job['arm'],workers=job['workers'],result=result,
                    reaction_latency=[w.get('reaction_latency') for w in workers],capture_timing=capture_summary(folder),
                    render_config=render_configs,
                    report=str(folder)))
            else:
                result=json.loads((a.output/f'calibration_{job["arm"]}'/'probe.json').read_text())
                if result['status']!='PASS':raise RuntimeError('Calibration did not pass')
            target.write_text(json.dumps(report,indent=2))
        report['status']='COMPLETE'
    except BaseException as exc:
        report.update(status='STOPPED',error=repr(exc));raise
    finally:
        report['source_unchanged']=checkpoint_info(a.checkpoint)==source
        target.write_text(json.dumps(report,indent=2))
    print('Capture matrix COMPLETE. Reports require review; production defaults remain unchanged.',flush=True)

if __name__=='__main__':main()
