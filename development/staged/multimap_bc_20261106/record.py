"""Human-only Electron recorder: native canvas masters, automatic candidate runs."""
import argparse
import asyncio
from concurrent.futures import ThreadPoolExecutor
from datetime import datetime
from functools import partial
import json
import os
import threading
import time
import numpy as np
import cv2
from common import ROOT,DATA,MAPS,CLASSIFIER,KEYS,write_new,require_idle,digest
from guard_monitor import GuardMonitor


class FrameWriter:
    """Four pending full frames maximum; overload drops frames, never grows RAM."""
    def __init__(self):
        self.pool=ThreadPoolExecutor(max_workers=2,thread_name_prefix='bc-save')
        self.pending=[];self.dropped=0

    def check(self):
        pending=[]
        for future in self.pending:
            if future.done():future.result()
            else:pending.append(future)
        self.pending=pending

    def submit(self,path,bgra):
        self.check()
        if len(self.pending)>=4:self.dropped+=1;return False
        image=bgra.copy()
        def save():
            if not cv2.imwrite(str(path),image[:,:,:3],[cv2.IMWRITE_PNG_COMPRESSION,1,
                    cv2.IMWRITE_PNG_STRATEGY,cv2.IMWRITE_PNG_STRATEGY_RLE]):
                raise OSError('Native frame save failed')
        self.pending.append(self.pool.submit(save));return True

    def flush(self):
        for f in self.pending:f.result()
        self.pending=[]

    def close(self):
        try:self.flush()
        finally:self.pool.shutdown(wait=True)


def keyboard(events,stop,hwnd):
    import ctypes
    user=ctypes.windll.user32
    user.GetForegroundWindow.restype=ctypes.c_void_p
    user.GetAsyncKeyState.argtypes=[ctypes.c_int];user.GetAsyncKeyState.restype=ctypes.c_short
    codes={'ArrowLeft':0x25,'ArrowRight':0x27,'ArrowUp':0x26,'ArrowDown':0x28,
           'KeyA':0x41,'KeyD':0x44,'KeyW':0x57,'KeyS':0x53,'Escape':0x1b}
    previous={key:False for key in codes}
    while not stop.wait(.002):
        at=time.perf_counter();focused=user.GetForegroundWindow()==hwnd
        for key,vk in codes.items():
            down=bool(user.GetAsyncKeyState(vk)&0x8000)
            if down!=previous[key] and focused:
                events.append(dict(at=at,code=key,type='keydown' if down else 'keyup',source='win32_key_state'))
            previous[key]=down


async def record(a,session):
    from playwright.async_api import async_playwright
    from visible_native import VisibleNativeCapture
    from tools.ppo_background_display import apply_viewport_layout
    from tools.ppo_headless_env import compose_guard_frame
    from rs_v4_screen_states import ScreenStateClassifier
    from record_validation import Handler,ThreadingHTTPServer
    from urllib.parse import urlsplit
    from rs_v612_gameplay_guard import GuardUncertainError
    classifier=ScreenStateClassifier(CLASSIFIER)
    import torch
    torch.set_num_threads(1);cv2.setNumThreads(1)
    server=ThreadingHTTPServer(('127.0.0.1',8012),partial(Handler,directory=str(ROOT/MAPS[a.map]['root'])))
    threading.Thread(target=server.serve_forever,daemon=True).start()
    writer=FrameWriter();source=None;events=[];stop=threading.Event();thread=None
    monitor=GuardMonitor(session)
    evidence=session/'state_evidence';evidence.mkdir()
    evidence_count=0;last_evidence=float('-inf')
    current=None;rows=[];run_number=0;armed=False;votes=0;menu_votes=0;event_start=0
    def finish(reason):
        nonlocal current,rows
        if current is None:return
        writer.flush()
        for row in rows:row['sha256']=digest(current/'frames'/row['file'])
        write_new(current/'frames.json',rows)
        write_new(current/'events.json',list(events[event_start:]))
        write_new(current/'manifest.json',dict(schema='multimap-bc-native-v1',status='candidate',
            map=a.map,split=a.split,session=session.name,reason=reason,frames=len(rows),
            duration=rows[-1]['at']-rows[0]['at'] if rows else 0,
            frame_shape=[945,1920,3],clock='perf_counter_capture_request',
            index_sha256=digest(current/'frames.json'),events_sha256=digest(current/'events.json'),
            note='Physical arrows/WASD; timestamps are host capture clocks, not Unity presentation time.'))
        print('CANDIDATE:',current,'seconds:',round(rows[-1]['at']-rows[0]['at'],2) if rows else 0,flush=True)
        current=None;rows=[]
    try:
        async with async_playwright() as pw:
            os.environ['RS_VISIBLE_CAPTURE']='desktop'
            source=await VisibleNativeCapture.launch(pw,session/'electron',1920,945,local_port=8012)
            page=source.page
            async def local_only(route):
                u=urlsplit(route.request.url)
                if u.scheme in ('data','blob') or (u.hostname in ('127.0.0.1','localhost') and u.port==8012):
                    await route.continue_()
                else:await route.abort()
            await source.context.route('**/*',local_only)
            await source.context.add_init_script("""addEventListener('keydown',e=>{
                if(e.code==='F8'){e.preventDefault();window.top.postMessage({bcLayout:true},location.origin);}
            },true);addEventListener('message',e=>{
                if(e.origin===location.origin && e.source===parent &&
                    e.data?.source==='ranked-site' && e.data.type==='practice')
                    window.top.postMessage({bcLayout:true},location.origin);
                if(e.origin===location.origin && e.data?.bcLayout)window.__bcLayout=true;
            });document.addEventListener('click',e=>{
                const b=e.target.closest?.('button');
                if(b && (b.getAttribute('aria-label')==='Practice' ||
                    /^Practice(?:\\s|$)/.test(b.textContent.trim())))
                    window.top.postMessage({bcLayout:true},location.origin);
            },true);""")
            await page.goto('http://127.0.0.1:8012/',wait_until='domcontentloaded')
            thread=threading.Thread(target=keyboard,args=(events,stop,int(source.ready['hwnd'])),daemon=True)
            thread.start()
            print('Click Practice, then F8 to fill the window. Use ARROW keys or WASD; tap rather than hold. '
                  'Restart with R yourself. Ctrl+C ends and saves candidates. No inputs are injected.',flush=True)
            layout=False;began=None;startup_started=time.perf_counter()
            last_layout_check=0;last_wait_message=0;seen_events=0
            while began is None or time.perf_counter()-began<a.seconds:
                tick=time.perf_counter()
                if not layout and tick-startup_started>=120:
                    raise RuntimeError('No ready recording layout within 120 seconds. Click Practice and press F8; no gameplay was recorded.')
                if not layout and tick-last_wait_message>=5:
                    print('[BC WAITING] Click Practice; press F8 if needed. Recording timer has not started.',flush=True)
                    last_wait_message=tick
                if tick-last_layout_check>.5:
                    last_layout_check=tick
                    if not layout and await page.evaluate('window.__bcLayout===true'):
                        arena=next((f for f in page.frames if '/arena.html' in f.url),None)
                        if arena is None or not await arena.locator('canvas#game').count():
                            await asyncio.sleep(.1);continue
                        await apply_viewport_layout(page)
                        box=await arena.locator('canvas#game').bounding_box() if arena else None
                        if not box or any(abs(box[k]-v)>1 for k,v in dict(x=0,y=0,width=1920,height=945).items()):
                            raise ValueError('F8 must produce an exact 1920x945 canvas; refusing mismatched crop')
                        layout=True;began=time.perf_counter()
                        print(f'[BC RECORDING] Layout ready; {a.seconds} seconds of capture starts now. '
                              'After a menu is observed, restart with R to record candidates.',flush=True)
                if not layout:await asyncio.sleep(.1);continue
                try:image=await source.capture()
                except GuardUncertainError:
                    finish('capture_or_focus_interruption');raise
                timing=source.last_timing;stamp=timing['request_started_at'];end=timing['image_received_at']
                classify_started=time.perf_counter();classification_error=None
                try:state,probs=classifier.state(compose_guard_frame(image,0,87),confidence=.9)
                except Exception as exc:
                    # Preserve human footage; uncertain frames remain excluded by export.
                    state,probs='UNCERTAIN',{}
                    classification_error=repr(exc)
                classifier_ms=(time.perf_counter()-classify_started)*1000
                diagnostic_run=current
                diagnostic_age=stamp-rows[0]['at'] if current and rows else None
                diagnostic_frame=None
                recent=events[seen_events:];seen_events=len(events)
                if any(e['code']=='Escape' and e['type']=='keydown' for e in recent):
                    finish('manual_pause');armed=False;votes=0
                if state=='MENU':
                    menu_votes+=1;votes=0
                    if menu_votes>=2:finish('menu');armed=True
                else:
                    menu_votes=0;votes=votes+1 if state=='GAMEPLAY' else 0
                if current is None and armed and votes>=2:
                    run_number+=1;current=session/f'run_{run_number:04d}';current.mkdir()
                    (current/'frames').mkdir();event_start=0
                    # Keep only events from this episode; earlier input never becomes history.
                    event_start=next((i for i,e in enumerate(events) if e['at']>=stamp-2),len(events))
                    rows=[];armed=False
                if current is not None:
                    name=f'{len(rows):06d}.png';path=current/'frames'/name
                    if writer.submit(path,image):
                        rows.append(dict(file=name,at=stamp,end=end,state=state,probabilities=probs))
                        diagnostic_frame=str(path.relative_to(session))
                        diagnostic_run=current
                        diagnostic_age=stamp-rows[0]['at']
                    if rows and stamp-rows[0]['at']>=a.max_run_seconds:
                        finish('max_length');armed=False;votes=0
                # Terminal/transition frames may fall outside a candidate. Keep a bounded
                # sample of these too, using the existing capture and bounded writer.
                if (diagnostic_frame is None and (classification_error or
                        (diagnostic_run is not None and state!='GAMEPLAY'))
                        and stamp-last_evidence>=1 and evidence_count<120):
                    path=evidence/f'{evidence_count:04d}.png'
                    if writer.submit(path,image):
                        diagnostic_frame=str(path.relative_to(session))
                        evidence_count+=1;last_evidence=stamp
                monitor.observe(at=stamp,state=state,probabilities=probs,
                    run=diagnostic_run.name if diagnostic_run else None,run_age=diagnostic_age,
                    frame=diagnostic_frame,capture_ms=(end-stamp)*1000,
                    classifier_ms=classifier_ms,error=classification_error)
                writer.check()
                await asyncio.sleep(max(.001,1/a.fps-(time.perf_counter()-tick)))
            finish('session_end')
            if run_number==0:
                print('[BC EMPTY] No candidate runs saved. Capture diagnostics were retained. '
                      'Allow a death/menu, then restart with R; do not begin a long session until candidates appear.',flush=True)
    finally:
        stop.set()
        if thread:thread.join(timeout=2)
        try:
            try:
                try:finish('interrupted_partial')
                finally:writer.close()
            finally:
                try:
                    if source:await source.close()
                finally:server.shutdown();server.server_close()
        finally:
            try:
                write_new(session/'capture_summary.json',dict(runs=run_number,dropped_frames=writer.dropped,
                          labels='candidate only; review required',automatic_inputs=False))
            finally:monitor.close()


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--map',choices=list(MAPS),required=True)
    p.add_argument('--split',choices=['train','validation'],required=True)
    p.add_argument('--seconds',type=int,default=600)
    p.add_argument('--fps',type=int,choices=[30,60],default=30)
    p.add_argument('--max-run-seconds',type=int,default=900)
    p.add_argument('--root',type=type(DATA),default=DATA)
    p.add_argument('--check-processes',action='store_true',help='Check for competing processes only; no browser/server/recording')
    a=p.parse_args()
    if not 10<=a.seconds<=3600 or not 5<=a.max_run_seconds<=3600:p.error('Invalid duration')
    require_idle()
    if a.check_processes:
        print('IDLE CHECK PASS: no competing recorder/trainer/evaluator.');return
    session=a.root/'sessions'/f'{a.map}_{a.split}_{datetime.now():%Y%m%d_%H%M%S_%f}'
    session.mkdir(parents=True,exist_ok=False)
    write_new(session/'session.json',dict(schema=1,map=a.map,split=a.split,map_release=MAPS[a.map]['release'],
        classifier=str(CLASSIFIER),classifier_sha256=digest(CLASSIFIER),capture_fps_target=a.fps,
        backend='visible Electron + desktop MSS',native_shape=[945,1920,3]))
    print('SESSION:',session,flush=True)
    try:asyncio.run(record(a,session))
    except KeyboardInterrupt:print('Stopped; completed candidate runs retained.',flush=True)


if __name__=='__main__':main()
