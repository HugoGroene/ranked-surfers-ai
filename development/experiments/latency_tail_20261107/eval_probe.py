"""Single-map evaluation with a hard wall bound and owned-process cleanup."""
import argparse
from datetime import datetime
import json
import os
from pathlib import Path
import subprocess
import sys
import time
import psutil

ROOT=Path(__file__).resolve().parents[3]
sys.path.insert(0,str(ROOT/'development/staged/multimap_bc_20261106'))
from common import require_idle,digest


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--checkpoint',type=Path,required=True)
    p.add_argument('--map',choices=('halloween','tokyo'),default='halloween')
    p.add_argument('--episodes',type=int,default=8);p.add_argument('--headless',action='store_true')
    p.add_argument('--pack',action='store_true');p.add_argument('--wall-seconds',type=float,default=120)
    p.add_argument('--rate-wait',choices=('sleep','precise'),default='sleep')
    p.add_argument('--production',action='store_true')
    a=p.parse_args();require_idle()
    if not 1<=a.episodes<=15 or not 30<=a.wall_seconds<=120:p.error('Invalid probe limits')
    runner=ROOT/('development/staged/pipeline_optimizations_20261106/evaluate.py' if a.pack else
                 'development/experiments/multimap_classifier/evaluate_both.py')
    output=ROOT/'reports'/datetime.now().strftime('eval_tail_probe_%Y%m%d_%H%M%S_%f');output.mkdir()
    before={f.name for f in (ROOT/'reports').glob('*visible_*')}|{f.name for f in (ROOT/'reports').glob('*headless_*')}
    command=[sys.executable,'-B','-s','-u',str(runner),'--checkpoint',str(a.checkpoint.resolve()),
             '--maps',a.map,'--episodes',str(a.episodes),*(['--headless'] if a.headless else [])]
    source=digest(a.checkpoint);started=time.monotonic();owned={};child=None
    result=dict(command=command,checkpoint_sha256=source,status='RUNNING',map=a.map,
                headless=a.headless,pack=a.pack or a.production,production=a.production,
                rate_wait='precise' if a.production else a.rate_wait)
    print('EVAL PROBE:',output,flush=True)
    def remember():
        if child is None:return
        try:parent=psutil.Process(child.pid)
        except psutil.NoSuchProcess:return
        for proc in [parent,*parent.children(recursive=True)]:
            try:owned[proc.pid,proc.create_time()]=proc
            except psutil.NoSuchProcess:pass
    try:
        with (output/'console.log').open('w',encoding='utf8') as log:
            env=dict(os.environ,
                RS_PIPELINE_FAST_PATH='1' if a.production else '0',
                RS_RATE_WAIT='precise' if a.production else a.rate_wait)
            env.pop('RS_PIPELINE_EXPERIMENT',None)
            child=subprocess.Popen(command,cwd=ROOT,env=env,stdout=log,stderr=subprocess.STDOUT)
            while child.poll() is None and time.monotonic()-started<a.wall_seconds-7:
                remember();time.sleep(.1)
            remember();result['exit_code']=child.poll()
            result['status']='INCOMPLETE_DEADLINE' if child.poll() is None else 'COMPLETE' if child.returncode==0 else 'FAILED'
    finally:
        killed=[]
        for proc in reversed(list(owned.values())):
            try:
                if proc.is_running():proc.kill();killed.append(proc)
            except psutil.NoSuchProcess:pass
        _,alive=psutil.wait_procs(killed,timeout=1)
        if killed and result['status']=='COMPLETE':result['status']='FAILED_PROCESS_LEAK'
        if child is not None:
            try:child.wait(timeout=1)
            except subprocess.TimeoutExpired:pass
        result.update(wall_seconds=time.monotonic()-started,forced_cleanup_pids=[p.pid for p in killed],
                      remaining_owned_pids=[p.pid for p in alive],source_unchanged=digest(a.checkpoint)==source)
        result['evaluation_reports']=[str(f) for f in (ROOT/'reports').iterdir()
            if f.name not in before and f.name.startswith(('multimap_visible_','multimap_headless_','pipeline_visible_','pipeline_headless_'))]
        (output/'result.json').write_text(json.dumps(result,indent=2));print(json.dumps(result,indent=2),flush=True)
    if result['status']!='COMPLETE':raise SystemExit(1)


if __name__=='__main__':main()
