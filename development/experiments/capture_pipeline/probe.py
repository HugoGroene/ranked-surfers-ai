"""Synthetic compositor calibration only. No game, model or movement inputs."""
import argparse
import asyncio
import json
from pathlib import Path
import sys
import time
ROOT=Path(__file__).resolve().parents[3]
sys.path.insert(0,str(ROOT))
from adapter import ARMS,install_capture

HTML='''<style>body{margin:0;background:#123456}canvas{display:block}
#pause{position:absolute;left:650px;top:300px;width:500px;height:200px;
background:#ffdd00;color:black;font:60px Arial}</style>
<canvas width="1920" height="945"></canvas><div id="pause">Paused<br>Resume</div>
<script>const c=document.querySelector('canvas').getContext('2d');
c.fillStyle='#123456';c.fillRect(0,0,1920,945);
c.fillStyle='#ff0000';c.fillRect(0,0,100,100);
c.fillStyle='#0000ff';c.fillRect(1820,845,100,100);
let n=0;function animate(){c.fillStyle=(++n%2)?'#ffffff':'#000000';
c.fillRect(1100,700,50,50);requestAnimationFrame(animate);}animate();</script>'''

async def probe(arm,output,seconds):
    import cv2
    import numpy as np
    from playwright.async_api import async_playwright
    Source=install_capture(arm)
    output.mkdir(parents=True,exist_ok=False)
    result=dict(status='STARTING',arm=arm,model_updated=False)
    source=None
    try:
        async with async_playwright() as pw:
            source=await Source.launch(pw,output/'electron',1920,945,transport='shared_memory',profile=True)
            await source.page.set_content(HTML)
            await asyncio.sleep(.3)
            previous=0;samples=[]
            end=time.perf_counter()+seconds
            while time.perf_counter()<end:
                frame=await source.capture()
                assert frame.shape==(945,1920,4)
                assert np.max(np.abs(frame[20,20,:3].astype(int)-[0,0,255]))<=2,'BGRA red mismatch'
                assert np.max(np.abs(frame[920,1900,:3].astype(int)-[255,0,0]))<=2,'Orientation/blue mismatch'
                assert np.max(np.abs(frame[310,660,:3].astype(int)-[0,221,255]))<=2,'HTML overlay missing'
                assert source.serial>previous,'Reused frame serial'
                previous=source.serial;samples.append(dict(source.last_timing['capture_profile']))
            cv2.imwrite(str(output/'synthetic.png'),frame)
            reference=cv2.imdecode(np.frombuffer(await source.page.screenshot(type='png'),np.uint8),cv2.IMREAD_COLOR)
            if source.width!=1920:
                reference=cv2.resize(cv2.resize(reference,(source.width,source.height),interpolation=cv2.INTER_LINEAR),
                                     (1920,945),interpolation=cv2.INTER_LINEAR)
            diff=np.abs(frame[:,:,:3].astype(np.int16)-reference.astype(np.int16))
            diff[690:760,1090:1160]=0 # Exclude the deliberately animated paint trigger.
            assert float(diff.mean())<1.5 and float(np.percentile(diff,99))<=4,'Compositor pixel parity failed'
            # The source must respond to a changed composed HTML overlay, not cached pixels.
            await source.page.locator('#pause').evaluate("e=>e.style.background='#00ff00'")
            for update_captures in range(1,7):
                changed=await source.capture()
                if np.max(np.abs(changed[310,660,:3].astype(int)-[0,255,0]))<=2:break
            else:raise AssertionError('Overlay update not captured within six fresh captures')
            cv2.imwrite(str(output/'changed_overlay.png'),changed)
            result.update(status='PASS',frames=len(samples),samples=samples,stale_retries=source.stale_frame_retries,
                overlay_update_captures=update_captures,pixel_mae=float(diff.mean()),pixel_p99=float(np.percentile(diff,99)),
                geometry=list(frame.shape),runtime=source.ready,
                note='Synthetic geometry/color/overlay/fresh serial check; does not establish Unity clock or learning quality.')
    except BaseException as exc:
        result.update(status='STOPPED',error=repr(exc));raise
    finally:
        try:
            if source:await source.close()
        finally:(output/'probe.json').write_text(json.dumps(result,indent=2))
    print(f'Capture calibration {arm}: PASS',flush=True)

if __name__=='__main__':
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--arm',choices=ARMS,required=True);p.add_argument('--output',type=Path,required=True)
    p.add_argument('--seconds',type=float,default=3)
    a=p.parse_args();asyncio.run(probe(a.arm,a.output,a.seconds))
