"""Bounded, isolated multimap probe; never a continuation candidate.

Thirty-six active seconds by default, with a hard wall budget including owned-process
cleanup. A deadline exit is INCOMPLETE, never a passing stability result.
"""
import argparse
from datetime import datetime
from functools import partial
import hashlib
from http.server import ThreadingHTTPServer
import json
import os
from pathlib import Path
import subprocess
import sys
import threading
import time
import psutil
sys.path[:0]=[str(Path(__file__).resolve().parents[3]),str(Path(__file__).resolve().parents[3]/'development/experiments/multimap_classifier')]
from record import ROOT,MAPS
from classifier_config import classifier_path


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--arm',choices=('baseline','pump','classifier','both','policy_layout','fused_adam','policy_fused','adam_cpu_steps','pump_cpu_steps','resync_spacing'),required=True)
    p.add_argument('--checkpoint',type=Path,required=True)
    p.add_argument('--state-model',type=Path)
    p.add_argument('--workers',type=int,choices=(4,5),default=4)
    p.add_argument('--reuse-pump-crop',action='store_true')
    p.add_argument('--threaded-preprocess',action='store_true')
    p.add_argument('--defer-startup-previews',action='store_true')
    p.add_argument('--bulk-readback',action='store_true',help='One native copy for tightly packed GPU readback rows')
    p.add_argument('--direct-readback',action='store_true',help='Write mapped GPU pixels directly into unpublished shared memory')
    p.add_argument('--exclude-first-episode',action=argparse.BooleanOptionalAction,default=True)
    p.add_argument('--active-minutes',type=float,default=.6)
    p.add_argument('--wall-seconds',type=float,default=120)
    p.add_argument('--features',default='pack,metrics',help='Comma-separated pack,metrics,pinned OR gpu_cache; none for baseline')
    p.add_argument('--rate-wait',choices=('sleep','precise'),help='Instrument the existing cap, optionally using a precise Windows timer')
    p.add_argument('--production',action='store_true',help='Validate the normal measured runtime defaults')
    p.set_defaults(production=True,features='none',reuse_pump_crop=True,threaded_preprocess=True,
                   defer_startup_previews=True,bulk_readback=True,direct_readback=True)
    a=p.parse_args()
    if a.features!='none' or a.rate_wait:p.error('Candidates require the unchanged production baseline')
    from validation import require_idle, require_checked, fingerprints
    require_idle()
    os.environ['RS_LATENCY_CANDIDATE']=a.arm
    os.environ['RS_PIPELINE_EXPERIMENT']='' if a.features=='none' else a.features
    if a.production:
        if a.features!='none' or a.rate_wait:p.error('--production requires --features none and no --rate-wait')
        os.environ.pop('RS_PIPELINE_EXPERIMENT',None)
        os.environ['RS_PIPELINE_FAST_PATH']='1'
    if a.rate_wait:os.environ['RS_RATE_WAIT']=a.rate_wait
    else:os.environ.pop('RS_RATE_WAIT',None)
    from development.staged.pipeline_optimizations_20261106.adapter import configuration
    experiment=configuration()
    from development.staged.pipeline_optimizations_20261106.preflight import require_correctness
    require_correctness(['pack','pinned'] if a.production else experiment['features'])
    if not .25<=a.active_minutes<=3:p.error('Active minutes must be .25..3')
    if not 30<=a.wall_seconds<=300:p.error('Wall budget must be 30..300 seconds')
    checkpoint=a.checkpoint.resolve()
    model=classifier_path(a.state_model)
    require_checked(model,a.arm,checkpoint)
    staged_sources=fingerprints()
    if not checkpoint.is_file():p.error('Missing checkpoint')
    output=ROOT/'reports'/datetime.now().strftime('latency_candidate_%Y%m%d_%H%M%S_%f')
    output.mkdir()
    started=time.monotonic()
    report=dict(candidate=a.arm,candidate_sources=staged_sources,pipeline_experiment=experiment,production=a.production,diagnostic_only=True,checkpoint=str(checkpoint),classifier=str(model),
                classifier_sha256=hashlib.sha256(model.read_bytes()).hexdigest(),reuse_pump_crop=a.reuse_pump_crop,
                threaded_preprocess=a.threaded_preprocess,
                defer_startup_previews=a.defer_startup_previews,
                bulk_readback=a.bulk_readback,
                direct_readback=a.direct_readback,
                exclude_first_episode=a.exclude_first_episode,
                source_sha256=hashlib.sha256(checkpoint.read_bytes()).hexdigest(),
                workers=a.workers,maps=['halloween' if i%2==0 else 'tokyo' for i in range(a.workers)],active_minutes=a.active_minutes,
                wall_budget_seconds=a.wall_seconds,
                status='RUNNING',dashboard=False)
    servers=[];owned={};child=None
    sys.path.insert(0,str(ROOT/'development/staged/map_import_20261102'))
    from record_validation import Handler
    class Server(ThreadingHTTPServer):
        request_queue_size=128
        daemon_threads=True
    def remember():
        if child is None:return
        try:
            process=psutil.Process(child.pid)
            descendants=process.children(recursive=True)
        except psutil.NoSuchProcess:return
        for item in [process,*descendants]:
            try:owned[(item.pid,item.create_time())]=item
            except psutil.NoSuchProcess:pass
    print('PROBE:',output,flush=True)
    try:
        for port,name in [(8000,'halloween'),(8010,'tokyo')]:
            server=Server(('127.0.0.1',port),partial(Handler,directory=str(MAPS[name])))
            servers.append(server)
            threading.Thread(target=lambda s=server:s.serve_forever(poll_interval=.05),daemon=True).start()
        env=dict(os.environ,RS_MULTIMAP_CLASSIFIER=str(model),
                 RS_MULTIMAP_DASHBOARD='0',RS_STATEFUL_GUARD='1',RS_MULTIMAP_MAP_MODE='multimap',
                 RS_MULTIMAP_VISIBLE_WORKER='',RS_REUSE_PUMP_CROP='1' if a.reuse_pump_crop else '0',
                 RS_PUMP_PREPROCESS_THREAD='1' if a.threaded_preprocess else '0',
                 RS_DEFER_STARTUP_PREVIEWS='1' if a.defer_startup_previews else '0',
                 RS_GPU_BULK_COPY='1' if a.bulk_readback else '0',
                 RS_GPU_DIRECT_SHARED='1' if a.direct_readback else '0',
                 RS_EXCLUDE_FIRST_EPISODE='1' if a.exclude_first_episode else '0')
        command=[sys.executable,'-B','-s','-u',str(Path(__file__).with_name('case.py')),
                 '--checkpoint',str(checkpoint),'--workers',str(a.workers),'--minutes',str(a.active_minutes),'--output',str(output/'case')]
        report['command']=command
        (output/'plan.json').write_text(json.dumps(report,indent=2))
        with (output/'console.log').open('w',encoding='utf8') as log:
            child=subprocess.Popen(command,cwd=ROOT,env=env,stdout=log,stderr=subprocess.STDOUT)
            while child.poll() is None and time.monotonic()-started<a.wall_seconds-7:
                remember();time.sleep(.1)
            remember()
            report['exit_code']=child.poll()
            report['status']=('INCOMPLETE_DEADLINE' if child.poll() is None else
                              'COMPLETE' if child.returncode==0 else 'FAILED')
    except BaseException as exc:
        report.update(status='FAILED',error=repr(exc))
        raise
    finally:
        # Only objects discovered as descendants of this probe. psutil's
        # identity-aware is_running/kill protects against PID reuse.
        survivors=[]
        for proc in reversed(list(owned.values())):
            try:
                if proc.is_running():survivors.append(proc);proc.kill()
            except psutil.NoSuchProcess:pass
        _,alive=psutil.wait_procs(survivors,timeout=1)
        report['forced_cleanup_pids']=[proc.pid for proc in survivors]
        report['remaining_owned_pids']=[proc.pid for proc in alive]
        if survivors and report['status']=='COMPLETE':report['status']='FAILED_PROCESS_LEAK'
        if child is not None:
            try:child.wait(timeout=1)
            except subprocess.TimeoutExpired:pass
        for server in servers:server.shutdown();server.server_close()
        report['source_unchanged']=hashlib.sha256(checkpoint.read_bytes()).hexdigest()==report['source_sha256']
        report['candidate_sources_unchanged']=fingerprints()==staged_sources
        if not report['candidate_sources_unchanged']:report['status']='FAILED_SOURCE_CHANGE'
        report['wall_seconds']=time.monotonic()-started
        (output/'result.json').write_text(json.dumps(report,indent=2))
        print(json.dumps(report,indent=2),flush=True)
    if report['status']!='COMPLETE':raise SystemExit(1)


if __name__=='__main__':main()
