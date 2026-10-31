"""One Electron worker: pixel/HTML parity and local Practice raw-frame timing."""
import argparse
import asyncio
import json
from pathlib import Path
import time

import cv2
import numpy as np
from rs_practice_controller import async_playwright,find_canvas,click_practice
from tools.ppo_background_display import apply_viewport_layout
from tools.ppo_electron_capture import ElectronCapture


async def main(args):
    args.report.mkdir(parents=True,exist_ok=False)
    result=dict(status='STARTING',model_updated=False)
    source=None
    try:
        async with async_playwright() as pw:
            source=await ElectronCapture.launch(pw,args.report/'electron',1920,945,transport=args.transport)
            page=source.page
            session=await source.context.new_cdp_session(page)
            await session.send('Emulation.setFocusEmulationEnabled',dict(enabled=True))
            await page.set_content('''<style>body{margin:0;background:#123456}canvas{display:block}
            #pause{position:absolute;left:650px;top:300px;width:500px;height:200px;
            background:#ffdd00;color:black;font:60px Arial}</style>
            <canvas width="1920" height="945"></canvas><div id="pause">Paused<br>Resume</div>''')
            await page.locator('canvas').evaluate('''el=>{const c=el.getContext('2d');
            c.fillStyle='#123456';c.fillRect(0,0,1920,945);
            c.fillStyle='#ff0000';c.fillRect(0,0,100,100);
            c.fillStyle='#0000ff';c.fillRect(1820,845,100,100)}''')
            await page.evaluate('''()=>new Promise(resolve=>requestAnimationFrame(()=>requestAnimationFrame(resolve)))''')
            frame=await source.capture()
            reference=cv2.imdecode(np.frombuffer(await page.screenshot(type='png'),np.uint8),cv2.IMREAD_COLOR)
            difference=np.abs(frame[:,:,:3].astype(np.int16)-reference.astype(np.int16))
            assert frame.shape==(945,1920,4)
            assert frame[20,20,:3].tolist()==[0,0,255],'Red/BGRA mismatch'
            assert frame[920,1900,:3].tolist()==[255,0,0],'Blue/orientation mismatch'
            assert frame[310,660,:3].tolist()==[0,221,255],'HTML overlay missing'
            assert difference.max()<=1,'Raw frame differs from rendered-page reference'
            result['synthetic']=dict(mean_absolute_difference=float(difference.mean()),
                maximum_difference=int(difference.max()),html_overlay=True,channels='BGRA',shape=list(frame.shape))
            cv2.imwrite(str(args.report/'synthetic.png'),frame)
            # Static content must produce a new independently painted frame.
            serial=source.serial
            await source.capture()
            assert source.serial>serial
            result['static_repaint']=True
            await page.evaluate("window.__keys=[];addEventListener('keydown',e=>window.__keys.push({key:e.key,trusted:e.isTrusted}))")
            await page.keyboard.press('ArrowLeft')
            events=await page.evaluate('window.__keys')
            assert events==[dict(key='ArrowLeft',trusted=True)],events
            result['synthetic_input']=events
            await page.goto('http://localhost:8000',wait_until='domcontentloaded')
            canvas=await find_canvas(page,timeout=60)
            await asyncio.sleep(3)
            await click_practice(page,timeout=60)
            await apply_viewport_layout(page)
            await canvas.focus()
            box=await canvas.bounding_box()
            assert box and all(abs(box[k]-v)<.5 for k,v in
                [('x',0),('y',0),('width',1920),('height',945)]),box
            result['webgl']=await canvas.evaluate('''el=>{const gl=el.getContext('webgl2')||el.getContext('webgl');
            if(!gl)return null;const e=gl.getExtension('WEBGL_debug_renderer_info');
            return {version:gl.getParameter(gl.VERSION),renderer:e?gl.getParameter(e.UNMASKED_RENDERER_WEBGL):'unknown'}}''')
            times=[];end=time.perf_counter()+args.seconds
            while time.perf_counter()<end:
                began=time.perf_counter();frame=await source.capture()
                times.append((time.perf_counter()-began)*1000)
                if len(times)==10:cv2.imwrite(str(args.report/'practice.png'),frame)
            result.update(status='COMPLETE',runtime=source.ready,raw_capture_ms=dict(count=len(times),
                median_ms=float(np.median(times)),p95_ms=float(np.percentile(times,95)),max_ms=float(max(times))),
                note='Paint callback and capture timings, not Unity physics/input response or policy throughput.')
    except BaseException as exc:
        result.update(status='STOPPED',error=repr(exc))
        raise
    finally:
        if source is not None:
            await source.close()
            result['renderer_exited']=source.process.poll() is not None
            result['renderer_exit_code']=source.process.returncode
        (args.report/'result.json').write_text(json.dumps(result,indent=2),encoding='utf8')
        print(json.dumps(result,indent=2))


if __name__=='__main__':
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--report',type=Path,required=True)
    p.add_argument('--seconds',type=int,choices=range(1,31),default=5)
    p.add_argument('--transport',choices=['socket','shared_memory'],default='socket')
    asyncio.run(main(p.parse_args()))
