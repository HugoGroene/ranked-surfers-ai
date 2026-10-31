"""Experimental browser-image adapter for the existing guarded PPO environment."""
import asyncio
import time
import cv2
import numpy as np
from rs_v612_gameplay_guard import GuardUncertainError


class BrowserImage:
    def grab(self, region):
        if self.image is None:
            raise GuardUncertainError('No headless browser image')
        return self.image

    def close(self):
        pass


def compose_guard_frame(image,left,top,width=1920,height=1080):
    """Same BGRA pixels as the historical padded guard, using contiguous copies."""
    h,w=image.shape[:2]
    if image.ndim!=3 or image.shape[2] not in (3,4) or min(left,top,width-left-w,height-top-h)<0:
        raise GuardUncertainError('Canvas does not fit guard image')
    bgra=cv2.cvtColor(image,cv2.COLOR_BGR2BGRA) if image.shape[2]==3 else image
    return cv2.copyMakeBorder(bgra,top,height-top-h,left,width-left-w,
                             cv2.BORDER_CONSTANT,value=(0,0,0,255))


def install(trainer,capture='playwright',*,profile=False,electron_transport='socket',electron_window=None):
    original = trainer.build_env_class

    def build():
        base = original()

        class HeadlessEnv(base):
            async def _open_browser(self):
                import rs_practice_controller as ctrl
                from tools.ppo_background_display import BACKGROUND_SWITCHES, apply_viewport_layout
                self.grabber = BrowserImage()
                self.grabber.image = None
                self.playwright = await ctrl.async_playwright().start()
                clean = getattr(self,'_framing_profile','legacy') == 'canvas_v1'
                self._headless_wh = (1920,945) if clean else (1904,937)
                width,height = self._headless_wh
                if capture=='electron':
                    from tools.ppo_electron_capture import ElectronCapture
                    if electron_window is None:
                        self._electron_host=await ElectronCapture.launch(
                            self.playwright,self.output_dir/'electron',width,height,profile=profile,
                            transport=electron_transport)
                    else:
                        self._electron_host=await ElectronCapture.attach(
                            self.playwright,electron_window,width,height,transport=electron_transport)
                    self.browser=self._electron_host.browser
                    self.context=self._electron_host.context
                    self.page=self._electron_host.page
                    self._headless_source=self._electron_host
                else:
                    self.browser = await self.playwright.chromium.launch(headless=True,
                        args=['--enable-gpu', *BACKGROUND_SWITCHES])
                    self.context = await self.browser.new_context(
                        viewport={'width':width,'height':height}, device_scale_factor=1)
                    self.page = await self.context.new_page()
                self._headless_errors = []
                self.page.on('pageerror', lambda error: self._headless_errors.append(str(error)))
                cdp = await self.context.new_cdp_session(self.page)
                self._headless_cdp=cdp
                await cdp.send('Emulation.setFocusEmulationEnabled', {'enabled':True})
                await self.page.goto('http://localhost:8000', wait_until='domcontentloaded')
                try:
                    self.canvas = await ctrl.find_canvas(self.page, timeout=60)
                except RuntimeError:
                    import json
                    (self.output_dir/'headless_startup_error.json').write_text(json.dumps(dict(
                        errors=self._headless_errors, frames=[f.url for f in self.page.frames])), encoding='utf8')
                    await self.page.screenshot(path=str(self.output_dir/'headless_startup_error.png'))
                    # One bounded reload before any policy action or transition exists.
                    await self.page.reload(wait_until='domcontentloaded')
                    self.canvas = await ctrl.find_canvas(self.page, timeout=60)
                await asyncio.sleep(3)
                await ctrl.click_practice(self.page, timeout=60)
                await apply_viewport_layout(self.page)
                await ctrl.focus_canvas(self.canvas)
                self.region = dict(left=0,top=0,width=1920,height=1080)
                self._background_canvas = dict(left=0 if clean else 8,top=87,width=width,height=height)
                self._background_canvas_policy = True
                self._background_previews_saved = False
                self._background_geometry_invalid = None
                self._background_view = await self._background_geometry()
                await self._background_install_geometry_watch()
                self._headless_backend=capture
                if capture in ('cdp','stream'):
                    from tools.ppo_cdp_capture import CdpCanvasCapture
                    self._headless_source=await CdpCanvasCapture.create(cdp,self.canvas,width,height)
                    if capture=='stream':
                        from tools.ppo_cdp_stream import LatestFrameStream
                        self._headless_source=LatestFrameStream(cdp,self._headless_source,width,height)
                        await self._headless_source.start()
                await self._headless_image()

            async def _headless_image(self):
                from rs_practice_controller import screenshot_canvas
                began = time.perf_counter()
                if self._background_geometry_invalid:
                    raise GuardUncertainError(self._background_geometry_invalid)
                data = (await self._headless_source.capture() if capture in ('cdp','stream','electron') else
                        await screenshot_canvas(self.page,self.canvas,image_type='jpeg',quality=90))
                received=time.perf_counter()
                image = data if capture=='electron' else cv2.imdecode(np.frombuffer(data,np.uint8),cv2.IMREAD_COLOR)
                decoded_at=time.perf_counter() if profile else None
                width,height = self._headless_wh
                if image is None or image.shape != (height,width,4 if capture=='electron' else 3):
                    raise GuardUncertainError('Headless canvas shape changed')
                left = self._background_canvas['left']
                raw=compose_guard_frame(image,left,87)
                self.grabber.image = raw
                self._headless_capture_ms = (time.perf_counter()-began)*1000
                self._headless_capture_timing=dict(request_started_at=began,image_received_at=received)
                if capture in ('stream','electron'):
                    self._headless_capture_timing.update(self._headless_source.last_timing)
                if profile:
                    values=self._headless_capture_timing.setdefault('capture_profile',{})
                    values.update(decode_ms=(decoded_at-received)*1000,
                                  guard_compose_ms=(time.perf_counter()-decoded_at)*1000)

            async def _capture(self):
                await self._headless_image()
                return await super()._capture()

            async def _sync_capture_region(self):
                if self._background_geometry_invalid:
                    self._guard_hard_stopped = True
                    raise GuardUncertainError(self._background_geometry_invalid)

            async def _headless_reload_after_time_limit(self):
                import rs_practice_controller as ctrl
                from tools.ppo_background_display import apply_viewport_layout
                if capture=='stream':
                    await self._headless_source.close()
                await self.page.goto('http://localhost:8000',wait_until='domcontentloaded',timeout=45000)
                self.canvas = await ctrl.find_canvas(self.page,timeout=60)
                await apply_viewport_layout(self.page)
                await ctrl.focus_canvas(self.canvas)
                self._background_geometry_invalid = None
                self._background_view = await self._background_geometry()
                await self._background_install_geometry_watch()
                if capture in ('cdp','stream'):
                    from tools.ppo_cdp_capture import CdpCanvasCapture
                    session=self._headless_cdp
                    self._headless_source=await CdpCanvasCapture.create(session,self.canvas,*self._headless_wh)
                    if capture=='stream':
                        from tools.ppo_cdp_stream import LatestFrameStream
                        self._headless_source=LatestFrameStream(session,self._headless_source,*self._headless_wh)
                        await self._headless_source.start()
                await self._headless_image()
                self._game_fullscreen = False

            async def _background_verify_window(self):
                if await self._background_geometry() != self._background_view:
                    raise GuardUncertainError('Headless geometry changed')

            async def _close_browser(self):
                if getattr(getattr(self,'_electron_host',None),'shared_host',False):
                    try:await self._electron_host.close()
                    finally:
                        # Do not close a browser/context owned by sibling workers.
                        if self.playwright is not None:await self.playwright.stop()
                    return
                try:
                    if getattr(self,'_electron_host',None) is not None:
                        await self._electron_host.close()
                finally:
                    await super()._close_browser()

        return HeadlessEnv
    trainer.build_env_class = build
