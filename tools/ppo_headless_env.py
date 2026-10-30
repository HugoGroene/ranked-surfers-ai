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


def install(trainer):
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
                self.browser = await self.playwright.chromium.launch(headless=True,
                    args=['--enable-gpu', *BACKGROUND_SWITCHES])
                clean = getattr(self,'_framing_profile','legacy') == 'canvas_v1'
                self._headless_wh = (1920,945) if clean else (1904,937)
                width,height = self._headless_wh
                self.context = await self.browser.new_context(
                    viewport={'width':width,'height':height}, device_scale_factor=1)
                self.page = await self.context.new_page()
                self._headless_errors = []
                self.page.on('pageerror', lambda error: self._headless_errors.append(str(error)))
                cdp = await self.context.new_cdp_session(self.page)
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
                await self._headless_image()

            async def _headless_image(self):
                from rs_practice_controller import screenshot_canvas
                began = time.perf_counter()
                data = await screenshot_canvas(self.page,self.canvas,image_type='jpeg',quality=90)
                received=time.perf_counter()
                image = cv2.imdecode(np.frombuffer(data,np.uint8),cv2.IMREAD_COLOR)
                width,height = self._headless_wh
                if image is None or image.shape != (height,width,3):
                    raise GuardUncertainError('Headless canvas shape changed')
                raw = np.zeros((1080,1920,4),dtype=np.uint8)
                raw[:,:,3] = 255
                left = self._background_canvas['left']
                raw[87:87+height,left:left+width,:3] = image
                self.grabber.image = raw
                self._headless_capture_ms = (time.perf_counter()-began)*1000
                self._headless_capture_timing=dict(request_started_at=began,image_received_at=received)

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
                await self.page.goto('http://localhost:8000',wait_until='domcontentloaded',timeout=45000)
                self.canvas = await ctrl.find_canvas(self.page,timeout=60)
                await apply_viewport_layout(self.page)
                await ctrl.focus_canvas(self.canvas)
                self._background_geometry_invalid = None
                self._background_view = await self._background_geometry()
                await self._background_install_geometry_watch()
                await self._headless_image()
                self._game_fullscreen = False

            async def _background_verify_window(self):
                if await self._background_geometry() != self._background_view:
                    raise GuardUncertainError('Headless geometry changed')

        return HeadlessEnv
    trainer.build_env_class = build
