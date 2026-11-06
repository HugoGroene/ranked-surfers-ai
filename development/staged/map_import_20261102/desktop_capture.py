"""Manual recording: capture visible Chrome client through MSS, never CDP screenshots."""
import asyncio
from concurrent.futures import ThreadPoolExecutor
import ctypes
from ctypes import wintypes
from pathlib import Path
import sys
import time
import uuid
import numpy as np

BASE=Path(__file__).resolve().parents[1]/'history_training_compare_20261103'
sys.path.insert(0,str(BASE))
from visible_desktop import DesktopReader

def crop_client(image, box, viewport):
    h,w=image.shape[:2]
    if viewport['width']!=w or viewport['height']>h or viewport['height']<=0:
        raise RuntimeError('Browser viewport/client geometry mismatch; keep browser maximized')
    # Win32 client includes Chrome's toolbar; JS innerHeight excludes it.
    x=round(box['x']);y=round(box['y'])+h-viewport['height']
    right=x+round(box['width']);bottom=y+round(box['height'])
    if x<0 or y<h-viewport['height'] or right>w or bottom>h:
        raise RuntimeError('Canvas is outside visible viewport; scroll it fully into view')
    return image[y:bottom,x:right,:3].copy()

class Capture:
    def __init__(self):
        self.pool=ThreadPoolExecutor(max_workers=1)
        self.reader=None
        self.title='LOCAL MAP RECORDING '+uuid.uuid4().hex[:10]

    def open(self,title):
        u=ctypes.WinDLL('user32')
        u.GetForegroundWindow.restype=wintypes.HWND
        u.GetWindowTextW.argtypes=[wintypes.HWND,wintypes.LPWSTR,ctypes.c_int]
        u.GetClientRect.argtypes=[wintypes.HWND,ctypes.POINTER(wintypes.RECT)]
        hwnd=u.GetForegroundWindow();text=ctypes.create_unicode_buffer(1024)
        u.GetWindowTextW(hwnd,text,1024)
        if title not in text.value:return False
        rect=wintypes.RECT()
        if not u.GetClientRect(hwnd,ctypes.byref(rect)):raise RuntimeError('Cannot read browser geometry')
        self.reader=DesktopReader(hwnd,rect.right,rect.bottom)
        return True

    async def ready(self,page):
        await page.bring_to_front()
        deadline=time.monotonic()+60
        print('[MAP CAPTURE] Waiting for the recording browser to be foreground. '
              'Click its title bar if needed; no game keys are sent.',flush=True)
        while time.monotonic()<deadline:
            # Page scripts and Windows title propagation can race initial load.
            # Retry identity verification rather than capturing an arbitrary HWND.
            await page.evaluate('(title)=>{document.title=title}',self.title)
            await asyncio.sleep(.25)
            if await asyncio.get_running_loop().run_in_executor(self.pool,self.open,self.title):
                print('[MAP CAPTURE] Foreground window verified; desktop capture ready.',flush=True)
                return
        raise RuntimeError('Recording window did not become foreground within 60 seconds; no desktop pixels captured')

    async def read(self,page,canvas):
        if self.reader is None:
            await self.ready(page)
        box=await canvas.bounding_box()
        viewport=await page.evaluate('({width:innerWidth,height:innerHeight})')
        if not box:raise RuntimeError('Canvas has no visible bounds')
        data=await asyncio.get_running_loop().run_in_executor(self.pool,self.reader.read)
        image=np.frombuffer(data,np.uint8).reshape(self.reader.height,self.reader.width,4)
        return crop_client(image,box,viewport)

    async def close(self):
        try:
            if self.reader:
                await asyncio.get_running_loop().run_in_executor(self.pool,self.reader.close)
        finally:self.pool.shutdown(wait=True)
