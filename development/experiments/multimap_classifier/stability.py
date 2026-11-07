"""Two-minute smoke then twenty-minute shared PPO stability test on two local maps."""
import argparse
from datetime import datetime
from functools import partial
import json
import os
from pathlib import Path
import subprocess
import sys
import threading
from record import ROOT,MAPS,REGISTRY
from classifier_config import classifier_path

def episode_summary(rows):
    values=[float(r['duration']) for r in sorted(rows,key=lambda r:r['at'])]
    n=len(values)
    quarters=[values[i*n//4:(i+1)*n//4] for i in range(4)]
    return dict(episodes=n,overall_mean=sum(values)/n if n else None,
                max_run=max(values) if n else None,
                chronological_quarter_means=[sum(q)/len(q) if q else None for q in quarters],
                quarter_counts=[len(q) for q in quarters],
                quarter_definition='Four consecutive approximately equal episode-count groups, ordered by completion; not duration percentiles')

def plots(folder,workers=2,show=True,map_mode='multimap',attempts=None):
    if not show:
        import matplotlib
        matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    assignments=([('halloween',tuple(range(workers)))] if map_mode=='halloween' else
                 [('halloween',tuple(range(0,workers,2))),('tokyo',tuple(range(1,workers,2)))])
    fig,axes=plt.subplots(len(assignments),1,figsize=(11,4*len(assignments)),layout='constrained',squeeze=False)
    summaries={}
    for ax,(name,workers) in zip(axes[:,0],assignments):
        rows=[]
        for attempt in attempts or [folder]:
            for worker in workers:
                path=attempt/f'train/worker_{worker}/survival_episodes.jsonl'
                if path.exists():rows += [json.loads(s) for s in path.read_text().splitlines() if s.strip()]
        rows.sort(key=lambda r:r['at'])
        stats=episode_summary(rows);summaries[name]=stats
        caption='No completed episodes'
        if rows:
            quarters=' / '.join(f'{x:.2f}' if x is not None else '-' for x in stats['chronological_quarter_means'])
            caption=f'Mean {stats["overall_mean"]:.2f}s | Max {stats["max_run"]:.2f}s | Q1–Q4 means: {quarters}s | n={stats["episodes"]}'
        print(f'[MAP STATS] {name}: {caption}',flush=True)
        if rows:
            values=[r['duration'] for r in rows];x=list(range(1,len(rows)+1))
            ax.plot(x,[sum(values[max(0,i-99):i+1])/len(values[max(0,i-99):i+1]) for i in range(len(values))],label='Recent mean (up to 100 episodes; small sample at start)')
            ax.plot(x,[sum(values[:i+1])/(i+1) for i in range(len(values))],label='Overall mean')
            ax.legend()
        ax.set(title=name+'\n'+caption,xlabel='Completed episodes on this map (quarters by episode count)',ylabel='Survival seconds')
    (folder/'per_map_summary.json').write_text(json.dumps(summaries,indent=2))
    fig.savefig(folder/'per_map_survival.png')
    if show:plt.show()
    else:plt.close(fig)

def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--minutes',type=float,default=20)
    p.add_argument('--workers',type=int,choices=(2,4),default=2)
    p.add_argument('--dashboard',action='store_true')
    p.add_argument('--diagnostic-only',action='store_true',
                   help='Exclude this run from automatic checkpoint selection in resume.ps1')
    p.add_argument('--stateful-guard',action='store_true')
    p.add_argument('--max-restarts',type=int,default=3,
                   help='Automatic recovery attempts for known guard stops after smoke passes; 0 disables')
    p.add_argument('--map-mode',choices=('multimap','halloween'),default='multimap')
    p.add_argument('--reuse-pump-crop',action=argparse.BooleanOptionalAction,default=True,
                   help='Reuse the exact delivered crop; disable to reproduce the previous pipeline')
    p.add_argument('--threaded-preprocess',action=argparse.BooleanOptionalAction,default=True,
                   help='Keep image preprocessing off the browser command event loop')
    p.add_argument('--defer-startup-previews',action=argparse.BooleanOptionalAction,default=True,
                   help='Save owned startup preview images in a bounded background writer')
    p.add_argument('--exclude-first-episode',action=argparse.BooleanOptionalAction,default=True,
                   help='Exclude one complete startup episode per worker from PPO and learning statistics')
    p.add_argument('--visible-worker',type=int,choices=(0,),
                   help='Diagnostic: visible Halloween worker 0 plus three offscreen workers')
    p.add_argument('--checkpoint',type=Path,default=ROOT/REGISTRY['continuation_checkpoint'])
    p.add_argument('--check-only',action='store_true')
    p.add_argument('--state-model',type=Path)
    a=p.parse_args()
    if not 1<=a.minutes<=2880:p.error('minutes must be 1..2880')
    if not 0<=a.max_restarts<=20:p.error('max-restarts must be 0..20')
    if a.visible_worker is not None and (a.workers!=4 or a.dashboard):
        p.error('Mixed visible diagnostic requires --workers 4 and dashboard off')
    checkpoint=a.checkpoint.resolve()
    model=classifier_path(a.state_model)
    for f in (checkpoint,model):
        if not f.is_file():p.error('Missing '+str(f))
    sys.path.insert(0,str(ROOT/'development/staged/map_import_20261102'))
    from record_validation import Handler,ThreadingHTTPServer
    class MapServer(ThreadingHTTPServer):
        # Simultaneous worker launches can burst well beyond HTTPServer's
        # default five pending connections, particularly through route.fetch.
        request_queue_size=128
        daemon_threads=True
    if a.check_only:
        print('CHECKPOINT:',checkpoint)
        print('Paths OK; map mode',a.map_mode,'workers',a.workers,'dashboard',a.dashboard,'visible worker',a.visible_worker,'smoke 2m then',a.minutes,'m');return
    prefix='halloween_stability' if a.map_mode=='halloween' else 'multimap_stability'
    output=ROOT/'reports'/datetime.now().strftime(prefix+'_%Y%m%d_%H%M%S_%f')
    output.mkdir()
    (output/'plan.json').write_text(json.dumps(dict(checkpoint=str(checkpoint),classifier=str(model),
        workers={str(i):('halloween' if a.map_mode=='halloween' or i%2==0 else 'tokyo') for i in range(a.workers)},minutes=a.minutes,map_mode=a.map_mode,
        dashboard=a.dashboard,stateful_guard=a.stateful_guard,visible_worker=a.visible_worker,
        diagnostic_only=a.diagnostic_only or a.visible_worker is not None,reuse_pump_crop=a.reuse_pump_crop,
        threaded_preprocess=a.threaded_preprocess,defer_startup_previews=a.defer_startup_previews,
        exclude_first_episode=a.exclude_first_episode,
        max_restarts=a.max_restarts,renderer_fps=60,automatic_promotion=False),indent=2))
    from runtime_diagnostics import append_snapshot
    memory_log=output/'memory_lifecycle.jsonl'
    append_snapshot(memory_log,'before_servers',ROOT)
    servers=[]
    try:
        map_servers=[(8000,'halloween')]+([(8010,'tokyo')] if a.map_mode=='multimap' else [])
        for port,name in map_servers:
            server=MapServer(('127.0.0.1',port),partial(Handler,directory=str(MAPS[name])))
            servers.append(server)
            threading.Thread(target=server.serve_forever,daemon=True).start()
        env=dict(os.environ,RS_MULTIMAP_CLASSIFIER=str(model),RS_MULTIMAP_DASHBOARD='1' if a.dashboard else '0',
                 RS_STATEFUL_GUARD='1' if a.stateful_guard else '0',
                 RS_MULTIMAP_MAP_MODE=a.map_mode,
                 RS_REUSE_PUMP_CROP='1' if a.reuse_pump_crop else '0',
                 RS_PUMP_PREPROCESS_THREAD='1' if a.threaded_preprocess else '0',
                 RS_DEFER_STARTUP_PREVIEWS='1' if a.defer_startup_previews else '0',
                 RS_EXCLUDE_FIRST_EPISODE='1' if a.exclude_first_episode else '0',
                 RS_MULTIMAP_VISIBLE_WORKER='' if a.visible_worker is None else str(a.visible_worker))
        env.pop('RS_TRAIN_WALL_DEADLINE',None)
        from recovery import supervise,run_child
        training_attempts=[]
        def launch(dest,source,minutes,deadline=None):
            child_env=dict(env)
            if deadline is not None:child_env['RS_TRAIN_WALL_DEADLINE']=str(deadline)
            return run_child([sys.executable,'-B','-s','-u',str(Path(__file__).with_name('stability_case.py')),
                '--checkpoint',str(source),'--workers',str(a.workers),'--minutes',str(minutes),
                '--output',str(dest)],child_env,dest,cwd=ROOT)
        for stage,minutes in [('smoke',2),('stability',a.minutes)]:
            from urllib.request import urlopen
            for port,_name in map_servers:
                with urlopen(f'http://127.0.0.1:{port}/index.html',timeout=10) as response:
                    if response.status!=200:raise RuntimeError(f'Map server {port} unhealthy')
            dest=output/stage
            print('STAGE:',stage,'OUTPUT:',dest,'DASHBOARD: '+('http://127.0.0.1:8001/' if a.dashboard else 'OFF'),flush=True)
            if stage=='smoke':
                code=launch(dest,checkpoint,minutes)
            else:
                code,dest,training_attempts=supervise(output,checkpoint,minutes,a.max_restarts,launch)
            append_snapshot(memory_log,'after_'+stage+'_workers_exit',ROOT)
            if code==0:
                for worker in range(a.workers):
                    wd=dest/f'train/worker_{worker}'
                    deaths=wd/'survival_episodes.jsonl'
                    if not deaths.exists() or not any(json.loads(s).get('reason')=='death' for s in deaths.read_text().splitlines() if s.strip()):
                        print('INCOMPLETE COVERAGE: no death observed for worker',worker);code=1
                    if a.map_mode=='multimap' and worker%2==1 and not (wd/'map_route.json').exists():
                        print('FAILED: Tokyo routing unverified for worker',worker);code=1
                metadata=dest/'train/hybrid_benchmark.json'
                if metadata.exists():
                    value=json.loads(metadata.read_text());value.update(dashboard=a.dashboard,
                        visible_worker=a.visible_worker,diagnostic_only=a.diagnostic_only or a.visible_worker is not None,
                        reuse_pump_crop=a.reuse_pump_crop,threaded_preprocess=a.threaded_preprocess,
                        defer_startup_previews=a.defer_startup_previews,
                        exclude_first_episode=a.exclude_first_episode,
                        worker_maps=['halloween' if a.map_mode=='halloween' or i%2==0 else 'tokyo' for i in range(a.workers)])
                    metadata.write_text(json.dumps(value,indent=2))
            if code:
                if stage=='stability':
                    recovery_path=output/'recovery.json'
                    recovery=json.loads(recovery_path.read_text())
                    if recovery['status']=='COMPLETE':
                        recovery['status']='STOPPED_COVERAGE_CHECK'
                        recovery_path.write_text(json.dumps(recovery,indent=2))
                print('STOPPED: preserved',dest,flush=True)
                plots(dest,a.workers,map_mode=a.map_mode,attempts=training_attempts or None)
                raise SystemExit(code)
        print('REPORT:',output,flush=True)
        recovery=json.loads((output/'recovery.json').read_text())
        print('CHECKPOINT:',recovery['latest_checkpoint'],flush=True)
        plots(output/'stability',a.workers,map_mode=a.map_mode,attempts=training_attempts)
    finally:
        for server in servers:server.shutdown();server.server_close()
        append_snapshot(memory_log,'after_servers_close',ROOT)

if __name__=='__main__':main()
