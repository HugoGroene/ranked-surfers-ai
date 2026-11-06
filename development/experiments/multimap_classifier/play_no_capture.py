"""Visible Electron comparison: matching host/scale, no frame reads or AI."""
import argparse
import asyncio
from datetime import datetime
from functools import partial
from pathlib import Path
import sys
import threading
from record import ROOT,MAPS

async def play(name,seconds):
    sys.path[:0]=[str(ROOT),str(ROOT/'development/staged/map_import_20261102'),
                  str(ROOT/'development/staged/history_training_compare_20261103'),
                  str(ROOT/'development/experiments/capture_pipeline')]
    from record_validation import Handler,ThreadingHTTPServer
    from playwright.async_api import async_playwright
    from visible_native import VisibleNativeCapture
    from tools.ppo_background_display import apply_viewport_layout
    output=ROOT/'reports'/datetime.now().strftime('electron_no_capture_%Y%m%d_%H%M%S_%f')
    output.mkdir()
    server=ThreadingHTTPServer(('127.0.0.1',8000),partial(Handler,directory=str(MAPS[name])))
    threading.Thread(target=server.serve_forever,daemon=True).start()
    try:
        async with async_playwright() as pw:
            source=await VisibleNativeCapture.launch(pw,output/'electron',1920,945)
            try:
                await source.page.goto('http://localhost:8000/',wait_until='domcontentloaded')
                print('Click Practice and play manually. F8 fills the game window. No screenshots, policy or classifier running.',flush=True)
                await source.context.add_init_script('')
                await source.page.evaluate("addEventListener('keydown',e=>{if(e.code==='F8')window.__fillGame=true;});")
                end=asyncio.get_running_loop().time()+seconds
                filled=False
                while not source.page.is_closed() and asyncio.get_running_loop().time()<end:
                    if not filled and await source.page.evaluate('window.__fillGame===true'):
                        await apply_viewport_layout(source.page);filled=True
                    await asyncio.sleep(.5)
            finally:await source.close()
    finally:server.shutdown();server.server_close()
    print('REPORT:',output)

if __name__=='__main__':
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--map',choices=MAPS,default='halloween')
    p.add_argument('--seconds',type=int,default=120)
    a=p.parse_args()
    if not 10<=a.seconds<=600:p.error('seconds must be 10..600')
    asyncio.run(play(a.map,a.seconds))
