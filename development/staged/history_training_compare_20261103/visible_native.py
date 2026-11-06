"""Visible Chromium with desktop capture to avoid repeated capturePage readbacks."""
import asyncio
from concurrent.futures import ThreadPoolExecutor
import json
import os
from pathlib import Path
import secrets
import socket
import subprocess
import time

from rs_v612_gameplay_guard import GuardUncertainError
from tools.ppo_electron_capture import ElectronCapture, RUNTIME, decode_frame
from tools.ppo_shared_frame import open_frame_mapping
from visible_desktop import DesktopReader, RAF_SCRIPT, collect_render


def validate_ready(ready, width, height):
    if (ready.get('width'), ready.get('height')) != (width, height) or (
            ready.get('visible') is not True or ready.get('offscreen') is not False or
            ready.get('transport') != 'shared_memory' or not ready.get('shared_name')):
        raise GuardUncertainError('Visible native window/transport handshake mismatch')


class VisibleNativeCapture(ElectronCapture):
    @classmethod
    async def launch(cls, playwright, folder, width, height, **kwargs):
        from adapter import install_scale
        from visible_browser import install_practice_start
        self = cls()
        self.backend = os.environ.get('RS_VISIBLE_CAPTURE', 'desktop')
        if self.backend not in ('desktop', 'page'):
            raise ValueError('RS_VISIBLE_CAPTURE must be desktop or page')
        self.executor = self.desktop = None
        self.width, self.height = width, height
        self.serial = self.reads = self.stale_frame_retries = 0
        self.last_timing = {}
        self.profile_enabled = True
        self.reader = self.writer = self.process = self.browser = self.mapping = self.log = None
        folder = Path(folder).resolve()
        self.folder = folder
        folder.mkdir(parents=True, exist_ok=False)
        self.log = (folder/'electron.log').open('wb')
        ready = folder/'ready.json'
        self.token = secrets.token_hex(24)
        with socket.socket() as reservation:
            reservation.bind(('127.0.0.1', 0))
            debug = reservation.getsockname()[1]
        env = dict(os.environ)
        env.pop('ELECTRON_RUN_AS_NODE', None)
        command = [str(RUNTIME), str(Path(__file__).with_name('visible_host.cjs')),
            f'--local-port={kwargs.get("local_port",8000)}',
            f'--width={width}', f'--height={height}', f'--cdp-port={debug}',
            f'--profile={folder/"profile"}', f'--ready={ready}', f'--token={self.token}']
        try:
            self.process = subprocess.Popen(command, stdout=self.log, stderr=self.log, env=env,
                creationflags=getattr(subprocess, 'CREATE_NO_WINDOW', 0))
            deadline = time.perf_counter()+30
            while not ready.exists():
                if self.process.poll() is not None or time.perf_counter()>deadline:
                    raise RuntimeError('Visible Chromium startup failed: '+str(folder/'electron.log'))
                await asyncio.sleep(.1)
            self.ready = json.loads(ready.read_text())
            validate_ready(self.ready, width, height)
            self.mapping = open_frame_mapping(self.ready['shared_name'], width*height*4+64)
            self.browser = await playwright.chromium.connect_over_cdp(f'http://127.0.0.1:{debug}')
            self.context = self.browser.contexts[0]
            self.page = self.context.pages[0]
            await self.context.add_init_script(RAF_SCRIPT)
            self.scale_errors = await install_scale(self.context, 1.)
            install_practice_start(folder, defer_first=True)
            self.reader, self.writer = await asyncio.open_connection(
                '127.0.0.1', self.ready['port'], limit=width*height*4+4096)
            if self.backend == 'desktop':
                hwnd = int(self.ready.get('hwnd', '0'))
                if hwnd <= 0:
                    raise GuardUncertainError('Missing visible window handle')
                self.executor = ThreadPoolExecutor(max_workers=1, thread_name_prefix='visible-capture')
                self.desktop = await asyncio.get_running_loop().run_in_executor(
                    self.executor, DesktopReader, hwnd, width, height)
            print(f'[VISIBLE EVAL] Actual on-screen game; capture={self.backend}. '
                  'Keep the game foreground and unobscured. Render timing will be saved.', flush=True)
            return self
        except BaseException:
            await self.close()
            raise

    async def capture(self):
        if self.scale_errors:
            raise GuardUncertainError(self.scale_errors[-1])
        # Existing bounded retries discard slow replies and retain the 50 ms
        # per-request bound. The temporal pump separately checks total delivery age.
        frame = await super().capture()
        self.last_timing.update(capture_kind=('visible_desktop_mss' if self.backend == 'desktop'
                                             else 'visible_native_bitmap_shared_memory'),
            frame_clock_kind='capture request duration; GPU presentation age unverified')
        return frame

    async def _capture_once(self):
        if self.backend == 'page':
            return await super()._capture_once()
        began = time.perf_counter()
        pixels = await asyncio.get_running_loop().run_in_executor(self.executor, self.desktop.read)
        received = time.perf_counter()
        metadata = dict(width=self.width, height=self.height, channels=4, format='BGRA',
                        serial=self.serial+1, byte_length=len(pixels))
        previous = self.serial
        self.serial += 1
        image, timing = decode_frame(metadata, pixels, self.width, self.height, previous,
                                     received, time.time(), request_started=began)
        self.reads += 1
        self.last_timing = dict(request_started_at=began, image_received_at=received, **timing,
            capture_profile=dict(capture_attempt_wall_ms=(received-began)*1000,
                                 desktop_readback_wall_ms=(received-began)*1000))
        return image

    async def close(self):
        try:
            if getattr(self, 'page', None) is not None:
                try:
                    timing = await asyncio.wait_for(collect_render(self.page), 3)
                    report = dict(backend=self.backend, reads=self.reads,
                                  stale_retries=self.stale_frame_retries, **timing)
                except Exception as exc:
                    report = dict(backend=self.backend, status='UNAVAILABLE', error=str(exc))
                (self.folder/'visible_render_timing.json').write_text(json.dumps(report, indent=2))
        finally:
            try:
                if getattr(self, 'desktop', None) is not None:
                    await asyncio.get_running_loop().run_in_executor(self.executor, self.desktop.close)
                    self.desktop = None
            finally:
                if getattr(self, 'executor', None) is not None:
                    self.executor.shutdown(wait=True)
                    self.executor = None
                await super().close()


def install():
    ElectronCapture.launch = VisibleNativeCapture.launch
