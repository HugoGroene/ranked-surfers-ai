"""Two-map, two-episode headless evaluation check; hard 120s incl. cleanup."""
import argparse
from datetime import datetime
import hashlib
import json
from pathlib import Path
import subprocess
import sys
import time
import psutil

ROOT=Path(__file__).resolve().parents[3]


def main():
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('--checkpoint',type=Path,required=True)
    a=p.parse_args();checkpoint=a.checkpoint.resolve();digest=hashlib.sha256(checkpoint.read_bytes()).hexdigest()
    output=ROOT/'reports'/f'eval_latency_probe_{datetime.now():%Y%m%d_%H%M%S_%f}';output.mkdir()
    command=[sys.executable,'-B','-s','-u',str(ROOT/'development/experiments/multimap_classifier/evaluate_both.py'),
             '--headless','--episodes','2','--checkpoint',str(checkpoint)]
    owned={};child=None;started=time.monotonic();result=dict(command=command,status='FAILED')
    def remember():
        try:parent=psutil.Process(child.pid)
        except psutil.NoSuchProcess:return
        for process in [parent,*parent.children(recursive=True)]:
            try:owned[(process.pid,process.create_time())]=process
            except psutil.NoSuchProcess:pass
    print('EVAL PROBE:',output,flush=True)
    try:
        with (output/'console.log').open('w',encoding='utf8') as log:
            child=subprocess.Popen(command,cwd=ROOT,stdout=log,stderr=subprocess.STDOUT)
            while child.poll() is None and time.monotonic()-started<113:
                remember();time.sleep(.1)
            remember();result['exit_code']=child.poll()
            result['status']='INCOMPLETE_DEADLINE' if child.poll() is None else ('COMPLETE' if child.returncode==0 else 'FAILED')
    finally:
        killed=[]
        for process in reversed(list(owned.values())):
            try:
                if process.is_running():process.kill();killed.append(process)
            except psutil.NoSuchProcess:pass
        _,alive=psutil.wait_procs(killed,timeout=1)
        if child is not None:
            try:child.wait(timeout=1)
            except subprocess.TimeoutExpired:pass
        if killed and result['status']=='COMPLETE':result['status']='FAILED_PROCESS_LEAK'
        result.update(forced_cleanup_pids=[x.pid for x in killed],remaining_pids=[x.pid for x in alive],
                      wall_seconds=time.monotonic()-started,source_unchanged=hashlib.sha256(checkpoint.read_bytes()).hexdigest()==digest)
        (output/'result.json').write_text(json.dumps(result,indent=2));print(json.dumps(result,indent=2),flush=True)
    if result['status']!='COMPLETE':raise SystemExit(1)


if __name__=='__main__':main()
