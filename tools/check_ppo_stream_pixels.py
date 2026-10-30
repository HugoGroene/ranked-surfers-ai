"""Synthetic composited-image parity check; no game, policy or training."""
import argparse
import asyncio
import json
from pathlib import Path
import cv2
import numpy as np
from playwright.async_api import async_playwright
from tools.ppo_cdp_capture import CdpCanvasCapture
from tools.ppo_cdp_stream import LatestFrameStream


async def main(folder):
    folder.mkdir(parents=True,exist_ok=False)
    async with async_playwright() as pw:
        browser=await pw.chromium.launch(headless=True,args=['--enable-gpu'])
        try:
            page=await browser.new_page(viewport=dict(width=1920,height=945),device_scale_factor=1)
            await page.set_content('''<style>body{margin:0}canvas{display:block}
            #overlay{position:absolute;left:800px;top:400px;width:300px;height:100px;
            background:white;color:black;font:30px sans-serif}</style>
            <canvas width="1920" height="945"></canvas><div id="overlay">Paused <button>Resume</button></div>''')
            canvas=page.locator('canvas')
            await canvas.evaluate('''el=>{const c=el.getContext('2d');let i=0;
              function paint(){c.fillStyle='#35a080';c.fillRect(0,0,1920,945);
                c.fillStyle='red';c.fillRect(0,0,200,200);c.fillStyle='blue';c.fillRect(1720,745,200,200);
                c.fillStyle=i++%2?'#101010':'#202020';c.fillRect(0,0,2,2);requestAnimationFrame(paint)}paint()}''')
            session=await page.context.new_cdp_session(page)
            direct=await CdpCanvasCapture.create(session,canvas,1920,945)
            stream=LatestFrameStream(session,direct,1920,945)
            await stream.start()
            a=cv2.imdecode(np.frombuffer(await direct.capture(),np.uint8),1)
            b=cv2.imdecode(np.frombuffer(await stream.capture(),np.uint8),1)
            await stream.close()
            assert stream.stream_reads==1 and stream.fallback_reads==0
            assert a.shape==b.shape==(945,1920,3)
            # The 2x2 animated timing marker can affect its JPEG MCU; exclude it.
            aa,bb=a.copy(),b.copy()
            aa[:32,:32]=bb[:32,:32]=0
            difference=float(np.abs(aa.astype(float)-bb).mean())
            assert difference<1,difference
            overlay_delta=float(np.abs(a[400:500,800:1100].astype(float)-b[400:500,800:1100]).mean())
            assert overlay_delta<1,overlay_delta
            assert float(b[470:490,1050:1090].mean())>240  # White HTML overlay, not raw canvas.
            cv2.imwrite(str(folder/'direct.png'),a)
            cv2.imwrite(str(folder/'stream.png'),b)
            result=dict(synthetic=True,shape=list(b.shape),mean_absolute_difference=difference,
                        html_overlay_difference=overlay_delta,stream_reads=stream.stream_reads,
                        fallback_reads=stream.fallback_reads)
            (folder/'result.json').write_text(json.dumps(result,indent=2))
            print(json.dumps(result))
        finally:
            await browser.close()


if __name__=='__main__':
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--report',type=Path,required=True)
    asyncio.run(main(parser.parse_args().report))
