"""Human-operated LOCAL new-map evidence capture; no PPO, training or automatic inputs."""
import argparse
import asyncio
from datetime import datetime
from functools import partial
from http.server import SimpleHTTPRequestHandler, ThreadingHTTPServer
import json
from pathlib import Path
import sys
import threading
import time
from urllib.parse import urlsplit

ROOT=Path(__file__).resolve().parents[3]
sys.path.insert(0,str(ROOT))
MAP=ROOT/'rankedsurfers-maps/current_20261102'

class Handler(SimpleHTTPRequestHandler):
    def log_message(self,*args):pass
    def handle(self):
        try:super().handle()
        except (ConnectionResetError,BrokenPipeError):
            pass # Browser canceled an asset download during navigation/close.
    def guess_type(self,path):
        if path.endswith('.wasm.unityweb'):return 'application/wasm'
        if path.endswith('.js.unityweb') or path.endswith('.mjs'):return 'application/javascript'
        if path.endswith('.data.unityweb'):return 'application/octet-stream'
        return super().guess_type(path)
    def end_headers(self):
        if urlsplit(self.path).path.endswith('.unityweb'):
            self.send_header('Content-Encoding','gzip')
        self.send_header('Cross-Origin-Opener-Policy','same-origin')
        self.send_header('Cross-Origin-Embedder-Policy','require-corp')
        super().end_headers()

async def record(seconds, startup_only=False, classifier_session=None, map_id=None,
                 model_path=None, candidate_check=False):
    import cv2
    import numpy as np
    from playwright.async_api import async_playwright
    from rs_v4_screen_states import ScreenStateClassifier
    from tools.ppo_headless_env import compose_guard_frame
    from rs_practice_controller import screenshot_canvas
    from playwright.async_api import Error as BrowserError
    from desktop_capture import crop_client
    sys.path.insert(0,str(ROOT/'development/experiments/capture_pipeline'))
    from visible_native import VisibleNativeCapture
    from tools.ppo_background_display import apply_viewport_layout
    source=None
    output=ROOT/'reports'/('new_map_validation_'+datetime.now().strftime('%Y%m%d_%H%M%S_%f'))
    output.mkdir(parents=True)
    (output/'frames').mkdir()
    model_path=Path(model_path or ROOT/'checkpoints/v612_snapshot_20261021/screen_state_candidate_v2.pt')
    model=ScreenStateClassifier(model_path)
    detector=None
    if candidate_check:
        from train_rs_rl_early_death import MenuTextDetector
        import rs_practice_controller as ctrl
        from PIL import Image
        roi=json.loads(ctrl.TEMPLATE_INFO.read_text(encoding='utf8'))['roi']
        refs=sorted(ctrl.TEMPLATES.glob('*.png')) if ctrl.TEMPLATES.is_dir() else []
        detector=MenuTextDetector(refs or [ctrl.TEMPLATE],roi)
    server=ThreadingHTTPServer(('127.0.0.1',8010),partial(Handler,directory=str(MAP)))
    threading.Thread(target=server.serve_forever,daemon=True).start()
    errors=[];missing=set();blocked=set();samples=[]
    entry_file=MAP/'actual_entry.json'
    report=dict(status='COLLECTING',map_entry=json.loads(entry_file.read_text()) if entry_file.exists() else {'root':str(MAP)},
                training=False,automatic_inputs=False)
    report.update(map_id=map_id,classifier=str(model_path),candidate_check=candidate_check)
    if classifier_session is not None:
        classifier_session=Path(classifier_session)
        classifier_session.mkdir(parents=True,exist_ok=False)
        (classifier_session/'session.json').write_text(json.dumps(dict(
            map_id=map_id,map_root=str(MAP),evidence=str(output),
            format='canvas_v1_guard_1920x1080_offset_0_87',
            labels='human review required',created=datetime.now().isoformat()),indent=2))
    print('OUTPUT:',output,flush=True)
    print('In the browser: click Practice, test LEFT/RIGHT/JUMP/ROLL, allow deaths and restart. '
          'Also pause with Escape, leave it paused briefly, then resume. Keep the window open. '
          'All states require image review; no labels are auto-approved.',flush=True)
    try:
        async with async_playwright() as pw:
            if startup_only:
                browser=await pw.chromium.launch(headless=True,args=['--force-device-scale-factor=1'])
            else:
                source=await VisibleNativeCapture.launch(pw,output/'electron',1920,945,local_port=8010)
                browser=source.browser
            try:
                context=(await browser.new_context(viewport=dict(width=1920,height=945),device_scale_factor=1)
                         if startup_only else source.context)
                async def route(request):
                    u=urlsplit(request.request.url)
                    if u.scheme in ('data','blob') or (u.scheme=='http' and u.hostname=='127.0.0.1' and u.port==8010):
                        await request.continue_()
                    else:
                        blocked.add(request.request.url);await request.abort()
                await context.route('**/*',route)
                page=await context.new_page() if startup_only else source.page
                await context.add_init_script('''document.addEventListener('click',e=>{
                    const b=e.target.closest?.('button');
                    if(b && (b.getAttribute('aria-label')==='Practice' ||
                        /^Practice(?:\\s|$)/.test(b.textContent.trim())))
                        window.top.postMessage({localRecorderLayout:true},location.origin);
                },true);
                // Listen to the game's actual Practice transition as well as
                // its HTML button. Icons/text inside the button vary by map.
                addEventListener('message',e=>{
                    if(e.origin!==location.origin)return;
                    if(e.source===parent && e.data?.source==='ranked-site' && e.data.type==='practice')
                        window.top.postMessage({localRecorderLayout:true},location.origin);
                    if(window===top && e.data?.localRecorderLayout===true)
                        window.__localPracticeClicked=true;
                });
                addEventListener('keydown',e=>{
                    if(e.code==='F8'){
                        e.preventDefault();e.stopImmediatePropagation();
                        window.top.postMessage({localRecorderLayout:true},location.origin);
                    }
                },true);''')
                page.on('pageerror',lambda e:errors.append(str(e)))
                page.on('console',lambda m:errors.append('console: '+m.text) if m.type=='error' else None)
                page.on('response',lambda r:missing.add(r.url) if r.status>=400 else None)
                await page.goto('http://127.0.0.1:8010/',wait_until='domcontentloaded')
                started=time.perf_counter()
                theater=False
                with (output/'samples.jsonl').open('w',encoding='utf-8') as log:
                    while time.perf_counter()-started<seconds and not page.is_closed():
                        began=time.perf_counter()
                        if not theater and await page.evaluate('window.__localPracticeClicked === true'):
                            await apply_viewport_layout(page)
                            theater=True
                            report['window_filling_layout_applied']=True
                            print('[MAP CHECK] Game fills the fixed window; Escape remains available for pause.',flush=True)
                        arena=next((f for f in page.frames if '/arena.html' in f.url),None)
                        canvas=arena.locator('canvas#game') if arena else None
                        if canvas is not None and await canvas.count() and await canvas.is_visible():
                            if (classifier_session is not None or candidate_check) and not theater:
                                await apply_viewport_layout(page)
                                theater=True
                                report['window_filling_layout_applied']=True
                            # Element screenshots may scroll/resize the viewport
                            # to expose the element. Reuse the established clipped
                            # page capture, which never scrolls the game into view.
                            if source is not None:
                                box=await canvas.bounding_box()
                                if not box:raise RuntimeError('Canvas bounds unavailable')
                                raw=await source.capture()
                                image=crop_client(raw,box,dict(width=1920,height=945))
                                ok,encoded=cv2.imencode('.png',image)
                                if not ok:raise RuntimeError('Failed to encode desktop frame')
                                png=encoded.tobytes()
                            else:
                                png=await screenshot_canvas(page,canvas,image_type='png')
                                image=cv2.imdecode(np.frombuffer(png,np.uint8),cv2.IMREAD_COLOR)
                            capture_ms=(time.perf_counter()-began)*1000
                            # Classifier preprocessing approximates the training guard
                            # canvas geometry; raw screenshots retain original pixels.
                            fitted=cv2.resize(image,(1920,945),interpolation=cv2.INTER_AREA)
                            guard=compose_guard_frame(cv2.cvtColor(fitted,cv2.COLOR_BGR2BGRA),0,87)
                            if classifier_session is not None:
                                if image.shape[:2] != (945,1920):
                                    raise RuntimeError('Classifier recording requires exact 1920x945 canvas geometry')
                                target=classifier_session/f'{len(samples):05d}.png'
                                if not cv2.imwrite(str(target),guard[:,:,:3]):
                                    raise RuntimeError('Failed to save classifier input')
                            state,probs=model.state(guard,confidence=.9)
                            name=f'{len(samples):05d}_{state}.png'
                            (output/'frames'/name).write_bytes(png)
                            row=dict(seconds=time.perf_counter()-started,image=name,state=state,
                                     probabilities=probs,canvas_shape=list(image.shape),capture_ms=capture_ms)
                            if detector is not None:
                                row['menu_template_score']=float(detector.score(Image.fromarray(cv2.cvtColor(guard,cv2.COLOR_BGRA2RGB))))
                                row['template_input']='canvas_v1_guard_1920x1080_offset_0_87'
                                print(f'[CANDIDATE] {row["seconds"]:.1f}s {state} '
                                      f'menu={probs.get("MENU",0):.3f} gameplay={probs.get("GAMEPLAY",0):.3f} '
                                      f'other={probs.get("OTHER",0):.3f} template={row["menu_template_score"]:.3f}',flush=True)
                            samples.append(row);log.write(json.dumps(row)+'\n');log.flush()
                            if startup_only:
                                report['startup_canvas_seen']=True
                            if len(samples)%20==0:print(f'[MAP CHECK] {row["seconds"]:.0f}s; state={state}; frames={len(samples)}',flush=True)
                        await asyncio.sleep(max(0,.5-(time.perf_counter()-began)))
                if startup_only:await page.screenshot(path=str(output/'final_page.png'))
                report['frame_urls']=[f.url for f in page.frames]
                report['status']=('STARTUP_CAPTURED_NOT_GAMEPLAY_VALIDATED' if startup_only else 'COLLECTED_REVIEW_REQUIRED') if samples else 'NO_CANVAS_CAPTURED'
            finally:
                if source is not None:
                    await source.close()
                    source=None
                else:await browser.close()
    except BrowserError as exc:
        report.update(status='INCOMPLETE',error=repr(exc),
                      stop_reason='browser_closed' if 'closed' in str(exc).lower() else 'browser_error')
    except BaseException as exc:
        report.update(status='INCOMPLETE',error=repr(exc));raise
    finally:
        server.shutdown();server.server_close()
        report.update(frames=len(samples),page_errors=errors,failed_requests=sorted(missing),
                      blocked_external_requests=sorted(blocked),
                      pending=['Manual review of startup/gameplay/death/pause labels',
                               'Actual input effectiveness and crop inspection',
                               'Matched pipeline timing and pause/resume verification'],
                      capture_backend='headless_page' if startup_only else 'electron_visible_desktop_mss',
                      timing_note='Diagnostic capture with PNG disk writes; not a training latency benchmark.')
        (output/'validation_report.json').write_text(json.dumps(report,indent=2),encoding='utf-8')
        print('REPORT:',output/'validation_report.json',flush=True)

if __name__=='__main__':
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--seconds',type=int,default=180)
    p.add_argument('--startup-only',action='store_true',help='Headless loading probe; no inputs')
    a=p.parse_args()
    if not 10<=a.seconds<=600:p.error('seconds must be 10..600')
    asyncio.run(record(a.seconds,a.startup_only))
