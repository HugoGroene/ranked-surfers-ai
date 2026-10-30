"""Paired local headless capture benchmark; no policy or movement inputs."""
import argparse
import asyncio
import json
from pathlib import Path
import time
import cv2
import numpy as np
from rs_practice_controller import async_playwright,find_canvas,click_practice,screenshot_canvas
from tools.ppo_background_display import BACKGROUND_SWITCHES,apply_viewport_layout
from tools.ppo_cdp_capture import CdpCanvasCapture


async def main(args):
    folder=Path(args.report)
    folder.mkdir(parents=True,exist_ok=False)
    async with async_playwright() as pw:
        browser=await pw.chromium.launch(headless=True,args=['--enable-gpu',*BACKGROUND_SWITCHES])
        try:
            context=await browser.new_context(viewport=dict(width=1920,height=945),device_scale_factor=1)
            page=await context.new_page()
            session=await context.new_cdp_session(page)
            await session.send('Emulation.setFocusEmulationEnabled',dict(enabled=True))
            await page.set_content('<style>body{margin:0}canvas{display:block}</style><canvas width="1920" height="945"></canvas>')
            canvas=page.locator('canvas')
            await canvas.evaluate("el=>{const c=el.getContext('2d');c.fillStyle='#35a080';c.fillRect(0,0,1920,945);c.fillStyle='red';c.fillRect(0,0,200,200);c.fillStyle='blue';c.fillRect(1720,745,200,200)}")
            source=await CdpCanvasCapture.create(session,canvas,1920,945)
            a=cv2.imdecode(np.frombuffer(await screenshot_canvas(page,canvas,image_type='jpeg',quality=90),np.uint8),1)
            b=cv2.imdecode(np.frombuffer(await source.capture(),np.uint8),1)
            assert a.shape==b.shape==(945,1920,3)
            difference=float(np.abs(a.astype(float)-b).mean())
            assert difference<1,difference
            await page.goto('http://localhost:8000',wait_until='domcontentloaded')
            canvas=await find_canvas(page,timeout=60)
            await asyncio.sleep(3)
            await click_practice(page,timeout=60)
            await apply_viewport_layout(page)
            source=await CdpCanvasCapture.create(session,canvas,1920,945)
            methods={'playwright':lambda:screenshot_canvas(page,canvas,image_type='jpeg',quality=90),
                     'cdp':source.capture}
            rows={name:[] for name in methods}
            end=time.perf_counter()+args.seconds
            i=0
            while time.perf_counter()<end:
                for name in (('playwright','cdp') if i%2==0 else ('cdp','playwright')):
                    began=time.perf_counter()
                    data=await methods[name]()
                    frame=cv2.imdecode(np.frombuffer(data,np.uint8),1)
                    rows[name].append((time.perf_counter()-began)*1000)
                    assert frame.shape==(945,1920,3),frame.shape
                    if i==5:
                        (folder/f'{name}_game.jpg').write_bytes(data)
                i+=1
            report=dict(synthetic_same_pixels_mean_absolute_difference=difference,
                paired_capture_ms={name:dict(count=len(v),median_ms=float(np.median(v)),
                    p95_ms=float(np.percentile(v,95)),max_ms=float(max(v))) for name,v in rows.items()})
            previous=None
            duplicates=count=0
            intervals=[]
            end=time.perf_counter()+5
            while time.perf_counter()<end:
                began=time.perf_counter()
                data=await source.capture()
                frame=cv2.imdecode(np.frombuffer(data,np.uint8),1)
                intervals.append((time.perf_counter()-began)*1000)
                if previous is not None and np.array_equal(frame,previous):
                    duplicates+=1
                previous=frame
                count+=1
            report['cdp_only']=dict(captures=count,seconds=5,identical_consecutive_images=duplicates,
                median_ms=float(np.median(intervals)),p95_ms=float(np.percentile(intervals,95)))
            streamed=[]
            async def on_frame(event):
                streamed.append(dict(received=time.perf_counter(),wall=time.time(),metadata=event['metadata']))
                await session.send('Page.screencastFrameAck',dict(sessionId=event['sessionId']))
            session.on('Page.screencastFrame',on_frame)
            await session.send('Page.startScreencast',dict(format='jpeg',quality=90,maxWidth=1920,maxHeight=945,everyNthFrame=1))
            await asyncio.sleep(3)
            await session.send('Page.stopScreencast')
            session.remove_listener('Page.screencastFrame',on_frame)
            report['stream_probe']=dict(frames=len(streamed),first=streamed[:2],last=streamed[-1:] if streamed else [])
            from tools.ppo_cdp_stream import LatestFrameStream
            stream=LatestFrameStream(session,source,1920,945)
            await stream.start()
            durations=[]
            end=time.perf_counter()+5
            while time.perf_counter()<end:
                began=time.perf_counter()
                data=await stream.capture()
                image=cv2.imdecode(np.frombuffer(data,np.uint8),1)
                assert image.shape==(945,1920,3),image.shape
                durations.append((time.perf_counter()-began)*1000)
                if len(durations)==5:
                    (folder/'stream_game.jpg').write_bytes(data)
            report['demand_stream']=dict(captures=len(durations),median_ms=float(np.median(durations)),
                p95_ms=float(np.percentile(durations,95)),max_ms=float(max(durations)),
                stream_reads=stream.stream_reads,surface_fallbacks=stream.fallback_reads)
            await stream.close()
            report['note']='Capture calls, not unique Unity ticks or policy throughput. Image motion does not establish gameplay/input response.'
            (folder/'report.json').write_text(json.dumps(report,indent=2),encoding='utf8')
            print(json.dumps(report,indent=2))
        finally:
            await browser.close()


if __name__ == '__main__':
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--report',required=True)
    parser.add_argument('--seconds',type=int,default=10,choices=range(5,21))
    asyncio.run(main(parser.parse_args()))
