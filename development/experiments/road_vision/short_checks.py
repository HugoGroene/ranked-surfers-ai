"""Resumable diagnostics only: crop smokes, fixed-action controls and capture scaling."""
import argparse
from datetime import datetime
import hashlib
import importlib.util
import json
from pathlib import Path
import subprocess
import sys
from road_config import ROOT,HERE,PRESETS,source_hashes,digest,write_json

CAPTURE=HERE.parent/'capture_pipeline'
TEMPORAL=HERE.parent/'temporal_history'

def sources():
    result=source_hashes()
    for p in [*CAPTURE.glob('*.py'),*CAPTURE.glob('*.cjs'),
              ROOT/'rankedsurfers-downloaded/game/releases/ae44e7782a4c8bbe5b10/arena.html']:
        result[p.relative_to(ROOT).as_posix()]=digest(p)
    return result

def capture_module():
    sys.path.insert(0,str(CAPTURE))
    try:
        spec=importlib.util.spec_from_file_location('short_capture_matrix',CAPTURE/'batch.py')
        module=importlib.util.module_from_spec(spec);spec.loader.exec_module(module)
        return module
    finally:sys.path.remove(str(CAPTURE))

def plan_jobs(output,episodes):
    py=[sys.executable,'-B','-s','-u'];jobs=[]
    def add(name,kind,cmd,report=None,depends=None):
        jobs.append(dict(name=name,kind=kind,command=cmd,report=str(report) if report else None,depends=depends))
    for label,folder in [('road',HERE),('temporal',TEMPORAL),('capture',CAPTURE)]:
        add('offline_'+label,'offline',py+['-m','unittest','discover','-s',str(folder),'-p','test_*.py','-q'])
    for mode in ('train','eval'):
        for arm,backend in [('road128_spatial','bitmap_100')]+[(arm,'gpu_100') for arm in PRESETS]:
            name=arm+('_bitmap' if backend=='bitmap_100' else '')+'_'+mode
            folder=output/name
            cmd=py+[str(HERE/'run_road_trial.py'),'--mode',mode,'--arm',arm,'--output',str(folder),
                '--frames','4','--span-ms','500','--capture-hz','60','--decision-hz','30','--seed','123',
                '--expected-sources',str(output/'road_sources.json'),'--capture-backend',backend]
            dependency=None
            if mode=='train':cmd+=['--minutes','1','--crop-previews']
            else:
                dependency=arm+('_bitmap' if backend=='bitmap_100' else '')+'_train'
                cmd+=['--minutes','3','--episodes','3','--checkpoint',str(output/dependency/'latest.zip')]
            add(name,'road_'+mode,cmd,folder/('trial_report.json' if mode=='train' else 'v61_eval_report.json'),dependency)
    for name,control,evidence in [('crop_evidence','learned',True),('control_none','none',False),
            ('control_learned','learned',False),('control_roll','roll',False),('control_jump','jump',False)]:
        folder=output/name
        cmd=py+[str(HERE/'control_eval.py'),'--control',control,'--output',str(folder),
            '--episodes',str(6 if evidence else episodes),'--minutes',str(2 if evidence else 8)]
        if evidence:cmd+=['--crop-evidence']
        add(name,'control',cmd,folder/'v61_eval_report.json')
    cap=capture_module()
    for arm in ('bitmap_100','gpu_100'):
        name='calibrate_'+arm;folder=output/name
        add(name,'calibration',py+[str(CAPTURE/'probe.py'),'--arm',arm,'--output',str(folder)],folder/'probe.json')
    for repeat in range(2):
        for workers in ([4,6] if repeat==0 else [6,4]):
            arms=['bitmap_100','gpu_100','bitmap_100'] if repeat==0 else ['gpu_100','bitmap_100','bitmap_100']
            for index,arm in enumerate(arms):
                name=f'capture_r{repeat}_w{workers}_{index}_{arm}';folder=output/name
                add(name,'capture',cap.command(arm,workers,1,cap.DEFAULT_CHECKPOINT,folder),
                    folder/'parallel_training.json','calibrate_'+arm)
    return jobs

def read_result(job,code):
    if job['kind']=='offline':return dict(ok=code==0,clean=code==0,status='PASS' if code==0 else 'FAIL')
    path=Path(job['report']);r=json.loads(path.read_text()) if path.exists() else {}
    expected={'road_train':'COMPLETE','road_eval':'COMPLETE','control':'COMPLETE',
              'calibration':'PASS','capture':'COMPLETE_FROZEN_POLICY_DIAGNOSTIC'}[job['kind']]
    ok=code==0 and r.get('status')==expected
    if job['kind']=='road_train':ok=ok and r.get('audit',{}).get('status')=='PASS_SETTLED_LEARNING' and (path.parent/'latest.zip').is_file()
    if job['kind'] in ('road_eval','control'):ok=ok and r.get('completed_requested_episodes',False)
    if job['kind']=='capture':ok=ok and r.get('policy_unchanged') and r.get('source_unchanged') and not r.get('model_updated') and r.get('optimized_transitions')==0
    clean=bool(ok or r.get('finalization_errors')==[])
    if r.get('stop_reason')=='interrupted' or 'KeyboardInterrupt' in str(r.get('error','')):
        clean=False  # Never continue independent stages after a user cancellation.
    return dict(ok=bool(ok),clean=clean,status=r.get('status','MISSING_REPORT'),
        mean_seconds=r.get('mean_survival_seconds'),actions=r.get('action_counts'),
        optimized_per_second=r.get('optimized_per_second'),collected_per_second=r.get('processed_transitions_per_second'),
        error=r.get('error'))

def run_job(job,log):
    print('[RUN]',job['name'],'LOG:',log,flush=True)
    with log.open('x',encoding='utf8') as handle:
        process=subprocess.Popen(job['command'],cwd=ROOT,stdout=handle,stderr=subprocess.STDOUT)
        try:return process.wait()
        except BaseException:
            # Same console: Ctrl+C is delivered to the child too. Let it save.
            print('Interrupted; waiting for child finalization. Outputs are retained.',flush=True)
            process.wait();raise

def evidence_hashes(job):
    if not job['report']:return {}
    paths=[Path(job['report'])]
    if job['kind']=='road_train':paths.append(paths[0].parent/'latest.zip')
    return {str(p):digest(p) for p in paths}

def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--output',type=Path)
    p.add_argument('--resume',type=Path)
    p.add_argument('--episodes',type=int,default=50)
    p.add_argument('--execute',action='store_true')
    a=p.parse_args()
    if not 3<=a.episodes<=100:p.error('Control episodes must be 3..100')
    output=(a.output or ROOT/'reports'/('short_checks_'+datetime.now().strftime('%Y%m%d_%H%M%S_%f'))).resolve()
    if not output.is_relative_to(ROOT/'reports') or output.exists():p.error('Use a new reports directory')
    hashes=sources();parent=None
    if a.resume:
        parent=json.loads((a.resume.resolve()/'short_checks.json').read_text())
        if parent['sources']!=hashes:p.error('Source changed; retain prior results separately and make a fresh run')
        a.episodes=parent['control_episodes']
    output.mkdir(parents=True);write_json(output/'road_sources.json',source_hashes())
    jobs=plan_jobs(output,a.episodes);completed={}
    if parent:
        old={s['job']['name']:s for s in parent['stages'] if s['result']['ok']}
        for job in jobs:
            prior=old.get(job['name'])
            if not prior or job['kind']=='offline':continue  # Offline checks are cheap and rerun.
            if not all(Path(f).is_file() and digest(f)==h for f,h in prior['artifacts'].items()):
                raise RuntimeError('A retained artifact changed: '+job['name'])
            if not read_result(prior['job'],0)['ok']:raise RuntimeError('Retained stage no longer passes')
            job.update(prior['job']);job['retained']=True
        train_paths={j['name']:Path(j['report']).parent for j in jobs if j['kind']=='road_train'}
        for job in jobs:
            if job['kind']=='road_eval' and not job.get('retained'):
                cmd=job['command'];cmd[cmd.index('--checkpoint')+1]=str(train_paths[job['depends']]/'latest.zip')
    report=dict(status='PLANNED',control_episodes=a.episodes,sources=hashes,jobs=jobs,stages=[],
        parent=str(a.resume) if a.resume else None,
        note='Short diagnostics only; no BC, no long training, no optimal-learning claim. Crop images require review. Each smoke uses four frames over 500 ms.')
    dest=output/'short_checks.json';write_json(dest,report)
    print('SHORT CHECKS:',dest,flush=True)
    if not a.execute:return
    try:
        import urllib.request
        with urllib.request.urlopen('http://127.0.0.1:8000/',timeout=5) as response:
            if response.status!=200:raise RuntimeError('Local game server not ready')
        report['status']='RUNNING'
        for job in jobs:
            if sources()!=hashes:raise RuntimeError('Source changed during suite')
            if job['depends'] and not completed.get(job['depends'],False):
                result=dict(ok=False,clean=True,status='SKIPPED_DEPENDENCY')
            elif job.get('retained'):
                result=read_result(job,0);print('[RETAINED]',job['name'],flush=True)
            else:
                code=run_job(job,output/(job['name']+'.log'));result=read_result(job,code)
                if code in (130,-1073741510,3221225786):result.update(clean=False,error='User interrupted')
            completed[job['name']]=result['ok']
            report['stages'].append(dict(job=job,result=result,artifacts=evidence_hashes(job) if result['ok'] else {}))
            write_json(dest,report)
            print('[RESULT]',job['name'],result['status'],flush=True)
            if not result['clean']:raise RuntimeError('Stage failed without verified cleanup; stop to avoid contaminating next test')
        report['status']='COMPLETE' if all(completed.values()) else 'COMPLETED_WITH_FAILURES'
    except BaseException as exc:
        report.update(status='STOPPED',error=repr(exc));raise
    finally:write_json(dest,report)
    print('REPORT:',dest,flush=True)
    if not all(completed.values()):raise SystemExit(1)

if __name__=='__main__':main()
