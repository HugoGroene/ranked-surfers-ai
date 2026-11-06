"""Read-only headless checkpoint/guard comparison on both local maps.

All preflights precede gameplay. Every case persists separately; --reuse retains
only completed evaluations with the same checkpoint hashes and settings.
"""
import argparse
from datetime import datetime
from functools import partial
import hashlib
import json
import os
from pathlib import Path
import statistics
import subprocess
import sys
import threading
from record import ROOT, MAPS
from classifier_config import classifier_path

CHECKPOINTS = {
    'before_multimap': 'reports/command_history_resume_20261104_093033/train/latest.zip',
    'before_overnight': 'reports/multimap_stability_20261105_104943_304736/stability/train/latest.zip',
    'latest': 'reports/multimap_stability_20261106_125542_791418/stability/train/latest.zip',
}


def save(path, value):
    temp=path.with_suffix('.tmp')
    temp.write_text(json.dumps(value,indent=2),encoding='utf8')
    temp.replace(path)


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--episodes',type=int,default=100)
    p.add_argument('--execute',action='store_true')
    p.add_argument('--reuse',type=Path)
    p.add_argument('--state-model',type=Path)
    a=p.parse_args()
    if not 1<=a.episodes<=1000:p.error('episodes must be 1..1000')
    os.chdir(ROOT)
    classifier=classifier_path(a.state_model)
    runner=ROOT/'development/experiments/command_history/evaluate.py'
    paths={name:(ROOT/path).resolve() for name,path in CHECKPOINTS.items()}
    hashes={name:hashlib.sha256(path.read_bytes()).hexdigest() for name,path in paths.items()}
    source_paths=[ROOT/'train_rs_v612_ppo.py',ROOT/'tools/ppo_stateful_continuity.py',
                  ROOT/'tools/ppo_resync_transient_v172.py',Path(__file__)]
    plan=dict(episodes_per_case=a.episodes,checkpoints={k:str(v) for k,v in paths.items()},
              checkpoint_sha256=hashes,classifier_sha256=hashlib.sha256(classifier.read_bytes()).hexdigest(),
              sources={str(f.relative_to(ROOT)):hashlib.sha256(f.read_bytes()).hexdigest() for f in source_paths},
              capture='gpu_100',workers=1,deterministic=True,dashboard=False,
              note='Same evaluator; Unity obstacle sequences are not guaranteed identical. No training.')
    output=(a.reuse.resolve() if a.reuse else ROOT/'reports'/datetime.now().strftime('checkpoint_compare_%Y%m%d_%H%M%S_%f'))
    if not output.is_relative_to(ROOT/'reports'):p.error('Output must be under reports')
    cases=[(name,map_name,True) for map_name in MAPS for name in paths]
    # Additional pair isolates evaluation-time guard behaviour on identical weights.
    cases += [('latest',map_name,False) for map_name in reversed(MAPS)]
    def command(name,destination,check=False):
        cmd=[sys.executable,'-B','-s','-u',str(runner),'--layout','fixed4',
             '--checkpoint',str(paths[name]),'--state-model',str(classifier),
             '--output',str(destination),'--episodes',str(a.episodes),'--minutes','120']
        return cmd+(['--check-only'] if check else [])
    env=dict(os.environ,RS_STATEFUL_GUARD='1',RS_MULTIMAP_DASHBOARD='0')
    # No browsers or servers in preflight/default mode.
    for name in paths:
        subprocess.run(command(name,output/('preflight_'+name),True),cwd=ROOT,env=env,check=True)
    print('PLAN:',len(cases),'cases;',a.episodes,'episodes each; no checkpoint updates.',flush=True)
    if not a.execute:return
    if a.reuse:
        if json.loads((output/'plan.json').read_text())!=plan:p.error('Reuse plan differs; start a new comparison')
    else:
        output.mkdir(parents=True,exist_ok=False);save(output/'plan.json',plan)
    sys.path.insert(0,str(ROOT/'development/staged/map_import_20261102'))
    from record_validation import Handler,ThreadingHTTPServer
    class Server(ThreadingHTTPServer):
        request_queue_size=128
        daemon_threads=True
    results=[]
    print('REPORT:',output,flush=True)
    for index,(name,map_name,stateful) in enumerate(cases,1):
        key=f'{name}_{map_name}_{"stateful" if stateful else "strict"}'
        marker=output/(key+'.json')
        if marker.exists() and (old:=json.loads(marker.read_text())).get('status')=='COMPLETE':
            results.append(old);print('[RETAINED]',key,flush=True);continue
        destination=output/(key+'_'+datetime.now().strftime('%H%M%S_%f'))
        log=destination.with_suffix('.log')
        print(f'[{index}/{len(cases)}] {key}; LOG: {log}',flush=True)
        server=Server(('127.0.0.1',8000),partial(Handler,directory=str(MAPS[map_name])))
        threading.Thread(target=server.serve_forever,daemon=True).start()
        try:
            with log.open('w',encoding='utf8') as handle:
                code=subprocess.call(command(name,destination),cwd=ROOT,
                    env=dict(env,RS_STATEFUL_GUARD='1' if stateful else '0'),stdout=handle,stderr=subprocess.STDOUT)
        finally:
            server.shutdown();server.server_close()
        report=destination/'v61_eval_report.json'
        data=json.loads(report.read_text()) if report.exists() else {}
        durations=data.get('durations_seconds',[])
        unchanged=hashlib.sha256(paths[name].read_bytes()).hexdigest()==hashes[name]
        complete=code==0 and data.get('status')=='COMPLETE' and len(durations)==a.episodes and unchanged
        result=dict(case=key,status='COMPLETE' if complete else 'INCOMPLETE',exit_code=code,
                    checkpoint_unchanged=unchanged,report=str(report),episodes=len(durations),
                    mean=statistics.mean(durations) if durations else None,
                    median=statistics.median(durations) if durations else None,
                    max=max(durations) if durations else None,
                    at_least_10s=sum(x>=10 for x in durations),at_least_20s=sum(x>=20 for x in durations),
                    actions=data.get('action_counts',data.get('actions')),
                    terminal_evidence_counts=data.get('terminal_evidence_counts'),
                    gameplay_only_intervals_ms=data.get('gameplay_only_intervals_ms'),
                    steps_per_second_including_resets=data.get('avg_steps_per_sec_including_resets'),
                    error=data.get('error'))
        save(marker,result);results.append(result);save(output/'summary.json',results)
        print('[RESULT]',key,result['status'],'mean=',result['mean'],'n=',len(durations),flush=True)
        if not unchanged:raise RuntimeError('Source checkpoint changed; comparison stopped')
    save(output/'summary.json',results)
    print('SUMMARY:',output/'summary.json',flush=True)
    if any(x['status']!='COMPLETE' for x in results):raise SystemExit(1)


if __name__=='__main__':main()
