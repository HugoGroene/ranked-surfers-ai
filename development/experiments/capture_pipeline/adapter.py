"""Process-local capture experiments. Production imports never opt into this."""
import asyncio
import json
import math
import os
from pathlib import Path
import secrets
import socket
import subprocess
import time

ROOT=Path(__file__).resolve().parents[3]
HERE=Path(__file__).resolve().parent
ARMS={
    'bitmap_100':dict(backend='bitmap',render_scale=1.,readback_width=1920),
    'bitmap_75':dict(backend='bitmap',render_scale=.75,readback_width=1920),
    'bitmap_50':dict(backend='bitmap',render_scale=.5,readback_width=1920),
    'gpu_100':dict(backend='gpu',render_scale=1.,readback_width=1920),
    'gpu_half_readback':dict(backend='gpu',render_scale=1.,readback_width=960),
}


def scaled_html(text,scale):
    """Force only render resolution; no preference SendMessage/gameplay changes."""
    if scale not in (.5,.75,1.):raise ValueError('Unsupported scale')
    replacements={
        'let renderScale = 1;':f'let renderScale = {scale};',
        'renderBudget.configure(d.preferences.autoResolution===true,ceiling);':
            f'renderBudget.configure(false,{scale});',
    }
    for old,new in replacements.items():
        if text.count(old)!=1:raise ValueError('Local game wrapper changed: '+old)
        text=text.replace(old,new)
    return text


async def install_scale(context,scale):
    errors=[]
    async def route_arena(route):
        try:
            response=await route.fetch()
            body=scaled_html(await response.text(),scale)
            await route.fulfill(response=response,body=body)
        except Exception as exc:
            errors.append(repr(exc));await route.abort()
    await context.route('**/arena.html*',route_arena)
    return errors


def install_capture(arm):
    """Replace launch in this child process only; reused by spawned workers."""
    from tools.ppo_electron_capture import ElectronCapture,RUNTIME
    from tools.ppo_shared_frame import open_frame_mapping
    config=ARMS[arm]
    class ExperimentalCapture(ElectronCapture):
        @classmethod
        async def launch(cls,playwright,folder,width,height,*,profile=False,transport='shared_memory'):
            if transport!='shared_memory':raise ValueError('Use shared_memory for all arms')
            self=cls();self.width=config['readback_width'];self.height=math.floor(height*self.width/width+.5)
            self.full_size=(width,height)
            self.serial=self.reads=self.stale_frame_retries=0;self.last_timing={};self.profile_enabled=True
            self.reader=self.writer=self.process=self.browser=self.mapping=self.log=None
            self.scale_errors=[];self.checked_canvas=False
            self.review_frames=[]
            folder=Path(folder).resolve();folder.mkdir(exist_ok=False)
            self.folder=folder
            self.log=(folder/'electron.log').open('wb');self.token=secrets.token_hex(24)
            with socket.socket() as reservation:
                reservation.bind(('127.0.0.1',0));port=reservation.getsockname()[1]
            ready=folder/'ready.json'
            env=dict(os.environ);env.pop('ELECTRON_RUN_AS_NODE',None)
            command=[str(RUNTIME),str(HERE/'host.cjs'),f'--width={width}',f'--height={height}',
                f'--output-width={self.width}',f'--output-height={self.height}',f'--backend={config["backend"]}',
                f'--cdp-port={port}',f'--profile={folder/"profile"}',f'--ready={ready}',f'--token={self.token}',
                '--profile-timings=1','--transport=shared_memory']
            try:
                self.process=subprocess.Popen(command,stdout=self.log,stderr=self.log,env=env,
                    creationflags=getattr(subprocess,'CREATE_NO_WINDOW',0))
                deadline=time.perf_counter()+30
                while not ready.exists():
                    if self.process.poll() is not None or time.perf_counter()>deadline:
                        raise RuntimeError('Experimental Electron startup failed: '+str(folder/'electron.log'))
                    await asyncio.sleep(.1)
                self.ready=json.loads(ready.read_text())
                if (self.ready['outputWidth'],self.ready['outputHeight'])!=(self.width,self.height):raise ValueError('Host output mismatch')
                self.mapping=open_frame_mapping(self.ready['shared_name'],self.width*self.height*4+64)
                self.browser=await playwright.chromium.connect_over_cdp(f'http://127.0.0.1:{port}')
                self.context=self.browser.contexts[0];self.page=self.context.pages[0]
                self.page.on('framenavigated',lambda _frame:setattr(self,'checked_canvas',False))
                self.scale_errors=await install_scale(self.context,config['render_scale'])
                self.reader,self.writer=await asyncio.open_connection('127.0.0.1',self.ready['port'],limit=self.width*self.height*4+4096)
                return self
            except BaseException:
                await self.close();raise

        async def capture(self):
            import cv2
            if self.scale_errors:raise RuntimeError(self.scale_errors[-1])
            # Startup-only metadata; never add DOM queries to ordinary decisions.
            if not self.checked_canvas and 'localhost:8000' in self.page.url:
                from rs_practice_controller import find_canvas
                canvas=await find_canvas(self.page,timeout=5)
                state=await canvas.evaluate('''el=>{const r=el.getBoundingClientRect();
                    const gl=el.getContext('webgl2')||el.getContext('webgl');
                    const ext=gl&&gl.getExtension('WEBGL_debug_renderer_info');
                    return {width:el.width,height:el.height,cssWidth:r.width,cssHeight:r.height,
                      renderer:ext?gl.getParameter(ext.UNMASKED_RENDERER_WEBGL):null}}''')
                scale=config['render_scale']
                if any(abs(state[k]-round(state[c]*scale))>1 for k,c in [('width','cssWidth'),('height','cssHeight')]):
                    raise RuntimeError('Unity internal render scale not applied: '+str(state))
                (self.folder/'render_config.json').write_text(json.dumps(dict(arm=arm,**config,actual=state),indent=2))
                self.checked_canvas=True
            frame=await super().capture()
            if self.width!=self.full_size[0]:
                began=time.perf_counter()
                # Experimental loss of detail is explicit. Preserve guard ROI geometry;
                # acceptance thresholds and freshness limits remain unchanged.
                frame=cv2.resize(frame,self.full_size,interpolation=cv2.INTER_LINEAR)
                self.last_timing['capture_profile']['cpu_restore_geometry_ms']=(time.perf_counter()-began)*1000
            if self.reads in (128,256,384):
                # Each delivered array owns a stable copy already. Keep only three;
                # defer encoding and file I/O until the worker has stopped.
                self.review_frames.append((self.serial,frame))
            return frame

        async def close(self):
            try:await super().close()
            finally:
                import cv2
                for serial,frame in getattr(self,'review_frames',[]):
                    cv2.imwrite(str(self.folder/f'review_capture_{serial:06d}.png'),frame)
                self.review_frames=[]
    ElectronCapture.launch=ExperimentalCapture.launch
    return ExperimentalCapture
