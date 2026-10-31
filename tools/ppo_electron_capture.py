"""Per-worker raw Electron capture. No JPEG, cached frames or global install."""
import asyncio
import json
import math
import os
from pathlib import Path
import secrets
import socket
import struct
import subprocess
import time

import numpy as np
from rs_v612_gameplay_guard import GuardUncertainError

ROOT=Path(__file__).resolve().parents[1]
RUNTIME=ROOT/'development/local/electron_worker/node_modules/electron/dist/electron.exe'


class StaleElectronFrame(GuardUncertainError):
    pass


def decode_frame(metadata,pixels,width,height,previous_serial,received,wall):
    expected=width*height*4
    if (metadata.get('width')!=width or metadata.get('height')!=height or
        metadata.get('channels')!=4 or metadata.get('format')!='BGRA' or
        metadata.get('byte_length')!=expected or len(pixels)!=expected):
        raise GuardUncertainError('Electron frame geometry/format changed')
    serial=metadata.get('serial')
    if not isinstance(serial,int) or isinstance(serial,bool) or serial<=previous_serial:
        raise GuardUncertainError('Electron reused or regressed frame serial')
    stamp=metadata.get('paint_wall')
    if not isinstance(stamp,(int,float)) or not math.isfinite(stamp) or wall-stamp<-.003:
        raise GuardUncertainError('Electron frame clock invalid')
    if wall-stamp>.05:
        raise StaleElectronFrame(f'Electron frame age {(wall-stamp)*1000:.1f}ms exceeds 50ms')
    return np.frombuffer(pixels,np.uint8).reshape(height,width,4),dict(
        paint_callback_at=received-max(0.,wall-stamp),electron_serial=serial,
        capture_kind='electron_raw_bitmap')


class ElectronCapture:
    @classmethod
    async def attach(cls,playwright,ready,width,height,*,transport='shared_memory'):
        self=cls()
        self.width,self.height=width,height
        self.serial=self.reads=self.stale_frame_retries=0
        self.last_timing={}
        self.profile_enabled=False
        self.process=self.log=self.mapping=None
        self.reader=self.writer=self.browser=None
        self.shared_host=True
        self.ready=ready
        self.token=ready['token']
        if ready['width']!=width or ready['height']!=height or ready['transport']!=transport:
            raise GuardUncertainError('Shared Electron window schema mismatch')
        try:
            if transport=='shared_memory':
                from tools.ppo_shared_frame import open_frame_mapping
                self.mapping=open_frame_mapping(ready['shared_name'],width*height*4+64)
            self.browser=await playwright.chromium.connect_over_cdp(f'http://127.0.0.1:{ready["debug_port"]}')
            matches=[(context,page) for context in self.browser.contexts for page in context.pages
                     if page.url==ready['initial_url']]
            if len(matches)!=1:raise GuardUncertainError('Shared Electron worker target missing or ambiguous')
            self.context,self.page=matches[0]
            self.reader,self.writer=await asyncio.open_connection('127.0.0.1',ready['port'],limit=width*height*4+4096)
            return self
        except BaseException:
            await self.close()
            raise

    @classmethod
    async def launch(cls,playwright,folder,width,height,*,profile=False,transport='socket'):
        if not RUNTIME.is_file():
            raise RuntimeError('Install isolated Electron runtime: see README_ELECTRON_CAPTURE.md')
        self=cls()
        self.width,self.height=width,height
        self.serial=0
        self.last_timing={}
        self.reads=0
        self.stale_frame_retries=0
        self.profile_enabled=profile
        self.reader=self.writer=self.process=self.browser=None
        self.mapping=None
        folder=Path(folder).resolve()
        folder.mkdir(exist_ok=False)
        profile_dir=folder/'profile';profile_dir.mkdir()
        ready=folder/'ready.json'
        self.log=(folder/'electron.log').open('wb')
        with socket.socket() as reservation:
            reservation.bind(('127.0.0.1',0));debug=reservation.getsockname()[1]
        self.token=secrets.token_hex(24)
        env=dict(os.environ);env.pop('ELECTRON_RUN_AS_NODE',None)
        command=[str(RUNTIME),str(ROOT/'tools/electron_worker/main.cjs'),
            f'--width={width}',f'--height={height}',f'--cdp-port={debug}',
            f'--profile={profile_dir}',f'--ready={ready}',f'--token={self.token}']
        if self.profile_enabled:command.append('--profile-timings=1')
        command.append(f'--transport={transport}')
        try:
            self.process=subprocess.Popen(command,stdout=self.log,stderr=self.log,env=env,
                creationflags=getattr(subprocess,'CREATE_NO_WINDOW',0))
            deadline=time.perf_counter()+30
            while not ready.exists():
                if self.process.poll() is not None or time.perf_counter()>deadline:
                    raise RuntimeError(f'Electron failed to start; inspect {folder / "electron.log"}')
                await asyncio.sleep(.1)
            self.ready=json.loads(ready.read_text())
            if self.ready.get('transport')!=transport:
                raise GuardUncertainError('Electron transport handshake mismatch')
            if transport=='shared_memory':
                from tools.ppo_shared_frame import open_frame_mapping
                self.mapping=open_frame_mapping(self.ready['shared_name'],width*height*4+64)
            self.browser=await playwright.chromium.connect_over_cdp(f'http://127.0.0.1:{debug}')
            self.context=self.browser.contexts[0]
            self.page=self.context.pages[0]
            # Exactly one bounded frame is requested at a time. The default
            # 64KiB reader limit repeatedly pauses/resumes a ~7.3MB readexactly.
            reader_limit=width*height*4+4096
            self.reader,self.writer=await asyncio.open_connection(
                '127.0.0.1',self.ready['port'],limit=reader_limit)
            self.ready['reader_limit_bytes']=reader_limit
            return self
        except BaseException:
            await self.close()
            raise

    async def capture(self):
        began=time.perf_counter()
        for attempt in range(3):
            try:
                image=await self._capture_once()
                self.last_timing['request_started_at']=began
                if 'capture_profile' in self.last_timing:
                    values=self.last_timing['capture_profile']
                    values['capture_total_wall_ms']=(time.perf_counter()-began)*1000
                    values['retry_discard_wall_ms']=max(0.,values['capture_total_wall_ms']-values['capture_attempt_wall_ms'])
                    values['stale_attempts_before_return']=attempt
                return image
            except StaleElectronFrame:
                if attempt==2:raise
                self.stale_frame_retries+=1
                # Discard all pixels. The next request demands a future paint;
                # retries never authorize movement or relax the 50ms bound.

    async def _capture_once(self):
        began=time.perf_counter()
        try:
            self.writer.write((json.dumps(dict(op='frame',token=self.token))+'\n').encode())
            await self.writer.drain()
            async def receive():
                size=struct.unpack('<I',await self.reader.readexactly(4))[0]
                if not 0<size<=4096:
                    raise GuardUncertainError('Invalid Electron frame header')
                metadata=json.loads(await self.reader.readexactly(size))
                if 'error' in metadata:
                    raise GuardUncertainError(metadata['error'])
                if metadata.get('byte_length')!=self.width*self.height*4:
                    raise GuardUncertainError('Invalid Electron frame payload length')
                header_at=time.perf_counter()
                if metadata.get('transport')=='shared_memory':
                    from tools.ppo_shared_frame import read_shared_frame
                    if getattr(self,'mapping',None) is None:
                        raise GuardUncertainError('Unexpected shared-memory reply')
                    pixels=read_shared_frame(self.mapping,metadata,self.width,self.height)
                else:
                    if getattr(self,'mapping',None) is not None:
                        raise GuardUncertainError('Unexpected socket pixel reply')
                    pixels=await self.reader.readexactly(metadata['byte_length'])
                return metadata,pixels,header_at,time.perf_counter()
            metadata,pixels,header_at,payload_at=await asyncio.wait_for(receive(),timeout=2.)
            received=time.perf_counter()
            try:
                image,timing=decode_frame(metadata,pixels,self.width,self.height,self.serial,received,time.time())
            except StaleElectronFrame:
                self.serial=metadata['serial']
                raise
            self.serial=metadata['serial'];self.reads+=1
            self.last_timing=dict(request_started_at=began,image_received_at=received,**timing)
            if getattr(self,'profile_enabled',False) and isinstance(metadata.get('profile'),dict):
                values=dict(metadata['profile'])
                if metadata.get('shared_copy_ms') is not None:values['shared_copy_ms']=metadata['shared_copy_ms']
                values.update(capture_attempt_wall_ms=(received-began)*1000,
                    request_to_header_ms=(header_at-began)*1000,
                    payload_read_ms=(payload_at-header_at)*1000,
                    response_resume_ms=(received-payload_at)*1000)
                values['transport_and_scheduling_residual_ms']=max(0.,
                    values['capture_attempt_wall_ms']-values['node_request_to_send_ms'])
                self.last_timing['capture_profile']=values
            return image
        except (OSError,ValueError,KeyError,asyncio.IncompleteReadError,asyncio.TimeoutError) as exc:
            raise GuardUncertainError(f'Electron capture failed: {exc}') from exc

    async def close(self):
        if self.writer is not None:
            try:
                self.writer.write((json.dumps(dict(op='close',token=self.token))+'\n').encode())
                await self.writer.drain()
            except (OSError,ConnectionError):
                pass
            self.writer.close()
        if self.process is not None:
            for _ in range(20):
                if self.process.poll() is not None:break
                await asyncio.sleep(.1)
            if self.process.poll() is None:
                self.process.terminate()
                await asyncio.to_thread(self.process.wait,timeout=5)
        if self.log is not None:self.log.close()
        if getattr(self,'mapping',None) is not None:self.mapping.close()


class ElectronGroup:
    """Coordinator-owned host; worker shutdown closes only its own window."""
    def __init__(self,folder,count,width,height,transport,timeout=60):
        folder=Path(folder);folder.mkdir(exist_ok=False)
        self.log=(folder/'electron.log').open('wb')
        ready=folder/'ready.json'
        with socket.socket() as reservation:
            reservation.bind(('127.0.0.1',0));debug=reservation.getsockname()[1]
        token=secrets.token_hex(24)
        env=dict(os.environ);env.pop('ELECTRON_RUN_AS_NODE',None)
        self.process=subprocess.Popen([str(RUNTIME),str(ROOT/'tools/electron_worker/multi.cjs'),
            f'--width={width}',f'--height={height}',f'--count={count}',f'--transport={transport}',
            f'--cdp-port={debug}',f'--profile={folder.resolve()/"profile"}',f'--ready={ready.resolve()}',
            f'--token={token}'],stdout=self.log,stderr=self.log,env=env,
            creationflags=getattr(subprocess,'CREATE_NO_WINDOW',0))
        try:
            deadline=time.perf_counter()+timeout
            while not ready.exists():
                if self.process.poll() is not None or time.perf_counter()>deadline:
                    raise RuntimeError('Shared Electron host startup failed; inspect its log')
                time.sleep(.05)
            self.windows=json.loads(ready.read_text())['windows']
            if len(self.windows)!=count:raise ValueError('Missing shared Electron windows')
        except BaseException:
            self.close()
            raise

    def close(self):
        if self.process.poll() is None:
            self.process.terminate()
            try:self.process.wait(timeout=5)
            except subprocess.TimeoutExpired:self.process.kill();self.process.wait(timeout=5)
        self.log.close()
