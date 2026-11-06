"""Opt-in spawn-safe two-map training adapter; runtime defaults unchanged."""
import json
import os
from pathlib import Path
import sys
import asyncio
from urllib.parse import urlsplit, urlunsplit

ROOT=Path(__file__).resolve().parents[3]
sys.path[:0]=[str(ROOT),str(ROOT/'development/experiments/command_history'),
               str(ROOT/'development/experiments/capture_pipeline')]
os.environ['RS_COMMAND_HISTORY_ACTIVE']='1'
import run_training
import run_workers
from tools import train_ppo_parallel as parallel
original_worker=parallel.worker_main

async def fetch_local(route,target):
    # Retry only failed local connection establishment for read-only assets.
    # Never replay game API writes or hide persistent server failure.
    for attempt in range(3):
        try:return await route.fetch(url=target,timeout=15000)
        except Exception as exc:
            if (route.request.method not in ('GET','HEAD') or
                    'ECONNREFUSED 127.0.0.1:8010' not in str(exc) or attempt==2):raise
            print('[MAP SERVER] Local connection refused; bounded asset retry',flush=True)
            await asyncio.sleep(.2*(attempt+1))

# This reply is fully consumed by receive(); retrying cannot desynchronise the
# stream. Never retry socket timeouts, invalid clocks or arbitrary guard failures.
from tools.ppo_electron_capture import ElectronCapture
from rs_v612_gameplay_guard import GuardUncertainError
original_capture_once=ElectronCapture._capture_once
async def fresh_capture(self):
    for attempt in range(3):
        try:return await original_capture_once(self)
        except GuardUncertainError as exc:
            if str(exc)!='No fresh offscreen paint within 500ms' or attempt==2:raise
            self.multimap_paint_retries=getattr(self,'multimap_paint_retries',0)+1
            print('[MULTIMAP CAPTURE] Fresh-paint timeout; requesting a new frame',flush=True)
            await asyncio.sleep(.01)
ElectronCapture._capture_once=fresh_capture

from tools import ppo_worker_preview as preview
old_control=preview.preview_control
def preview_control(folder,default_fps=1):
    value=old_control(folder,default_fps)
    value['fps']=min(value['fps'],1.)
    return value
preview.preview_control=preview_control
old_start=preview.start_preview
def start_preview(folder,*args,**kwargs):
    folder=Path(folder);folder.mkdir(parents=True,exist_ok=True)
    (folder/'preview_control.json').write_text(json.dumps(dict(enabled=False,fps=1)))
    server=old_start(folder,*args,**kwargs)
    page=folder/'index.html'
    html=page.read_text(encoding='utf8')
    html=html.replace("let imagesEnabled=localStorage.getItem('ppoImagesEnabled')!=='false';",'let imagesEnabled=false;')
    html=html.replace("let previewFps=Number(localStorage.getItem('ppoPreviewFps')||1);",'let previewFps=1;')
    for fps in (5,10,15,30):html=html.replace(f'<option>{fps}</option>','')
    page.write_text(html,encoding='utf8')
    return server
preview.start_preview=start_preview

def worker_main(index,display,options,connection,abort,resources=None):
    if os.environ.get('RS_DEFER_STARTUP_PREVIEWS')=='1':
        from deferred_previews import install as install_previews
        import train_rs_v612_ppo as trainer
        install_previews(trainer)
    if os.environ.get('RS_REUSE_PUMP_CROP')=='1':
        from crop_reuse import install
        import train_rs_v612_ppo as trainer
        install(trainer)
    if os.environ.get('RS_MULTIMAP_VISIBLE_WORKER') == str(index):
        if index != 0:raise ValueError('Mixed diagnostic currently shows Halloween worker 0 only')
        # Spawn-local patch: the other three processes retain offscreen GPU capture.
        from visible_native import install as install_visible
        from visible_startup import install as install_ready
        import train_rs_v612_ppo as trainer
        os.environ['RS_VISIBLE_CAPTURE']='desktop'
        install_visible()
        install_ready(trainer)
        print('[MIXED WORKERS] Worker 0 is visible; keep its window foreground and unobscured.',flush=True)
    if index%2==1 and os.environ.get('RS_MULTIMAP_MAP_MODE','multimap')=='multimap':
        import adapter
        async def map_scale(context,scale):
            errors=[]
            async def route_local(route):
                u=urlsplit(route.request.url)
                if u.hostname not in ('localhost','127.0.0.1') or u.port!=8000:
                    await route.abort();return
                try:
                    target=urlunsplit(('http','127.0.0.1:8010',u.path,u.query,''))
                    response=await fetch_local(route,target)
                    if u.path.endswith('/arena.html'):
                        if '/c4df9560f722c362f2c4/' not in u.path:
                            raise RuntimeError('Wrong Tokyo release: '+u.path)
                        body=adapter.scaled_html(await response.text(),scale)
                        (Path(options['report'])/f'worker_{index}'/'map_route.json').write_text(
                            json.dumps(dict(map='tokyo',arena=u.path,source=target)))
                        await route.fulfill(response=response,body=body)
                    else:await route.fulfill(response=response)
                except Exception as exc:
                    errors.append(repr(exc));await route.abort()
            await context.route('**/*',route_local)
            return errors
        adapter.install_scale=map_scale
    original_worker(index,display,options,connection,abort,resources)

parallel.worker_main=worker_main
original_command=run_workers.command
def command(*args):
    result=original_command(*args)+['--state-model',os.environ['RS_MULTIMAP_CLASSIFIER']]
    if os.environ.get('RS_EXCLUDE_FIRST_EPISODE','1')=='1':
        result+=['--exclude-first-episode']
    if os.environ.get('RS_TRAIN_WALL_DEADLINE'):
        result+=['--wall-deadline',os.environ['RS_TRAIN_WALL_DEADLINE']]
    if os.environ.get('RS_MULTIMAP_DASHBOARD')=='1':
        result+=['--preview-port','8001','--preview-fps','1']
    return result
run_workers.command=command

if __name__=='__main__':run_training.main()
