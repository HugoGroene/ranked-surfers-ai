"""Visible Chromium capture for qualitative evaluation; no desktop screenshots."""
import asyncio
import base64
import json
import time

import cv2
import numpy as np

from rs_v612_gameplay_guard import GuardUncertainError
from tools.ppo_electron_capture import StaleElectronFrame


async def maximize_window(session):
    window = await session.send('Browser.getWindowForTarget')
    window_id = window['windowId']
    await session.send('Browser.setWindowBounds', dict(windowId=window_id,
                                                     bounds={'windowState': 'normal'}))
    await session.send('Browser.setWindowBounds', dict(windowId=window_id,
                                                     bounds={'windowState': 'maximized'}))
    return await session.send('Browser.getWindowBounds', {'windowId': window_id})


def install_practice_start(folder, *, defer_first=False):
    """Select only the real local HTML Practice button, with actionable failures."""
    import rs_practice_controller as ctrl
    async def click_practice(page, timeout=60, debug_dir=None):
        from urllib.parse import urlparse
        url = urlparse(page.url)
        if url.hostname not in ('localhost', '127.0.0.1') or url.port != 8000:
            raise GuardUncertainError('Visible evaluation requires local Practice')
        deadline = time.monotonic() + timeout
        last_error = None
        while time.monotonic() < deadline:
            for frame in page.frames:
                buttons = frame.get_by_role('button', name='Practice', exact=True)
                for i in range(await buttons.count()):
                    button = buttons.nth(i)
                    if not await button.is_visible() or not await button.is_enabled():
                        continue
                    try:
                        # HTML activation avoids headed viewport/emulated-viewport
                        # coordinate differences. No Unity canvas input is sent.
                        # Resolve now, while the control is visible. A role
                        # locator re-resolves later and excludes controls hidden
                        # by apply_viewport_layout, even for Locator.evaluate().
                        handle = await button.element_handle(timeout=3000)
                        if handle is None:
                            raise GuardUncertainError('Practice control disappeared before setup')
                        async def activate(handle=handle):
                            try:
                                current = urlparse(page.url)
                                if current.hostname not in ('localhost','127.0.0.1') or current.port != 8000:
                                    raise GuardUncertainError('Deferred Practice requires local page')
                                # Use textContent: innerText depends on visibility.
                                # A replaced/detached node must fail, never redirect
                                # activation to some other control.
                                await asyncio.wait_for(handle.evaluate('''el => {
                                    if(el.tagName!=='BUTTON' || !el.isConnected || el.disabled ||
                                       el.textContent.trim()!=='Practice')
                                        throw new Error('Practice control changed');
                                    el.click();
                                }'''), timeout=3.)
                            finally:
                                await handle.dispose()
                        if defer_first and not getattr(page, '_ppo_initial_practice_released', False):
                            page._ppo_pending_practice = activate
                            print('[VISIBLE START] Practice held until policy and capture setup are ready.',flush=True)
                            return
                        await activate()
                        print('[VISIBLE EVAL] Activated local HTML Practice button.', flush=True)
                        return
                    except Exception as exc:
                        last_error = repr(exc)
            await asyncio.sleep(.25)
        await ctrl.debug_practice(page, folder)
        raise RuntimeError(f'Local HTML Practice activation failed: {last_error}; diagnostics: {folder}')
    ctrl.click_practice = click_practice


class VisibleBrowserCapture:
    @classmethod
    async def launch(cls, playwright, folder, width, height, **kwargs):
        from tools.ppo_background_display import BACKGROUND_SWITCHES
        # Apply the same full render scale as the comparison's gpu_100 backend.
        from adapter import install_scale
        self = cls()
        self.width, self.height = width, height
        self.serial = self.reads = self.stale_frame_retries = 0
        self.last_timing = {}
        self.browser = None
        self.closed = False
        folder.mkdir(parents=True, exist_ok=False)
        self.folder = folder
        try:
            self.browser = await playwright.chromium.launch(headless=False,
                args=['--enable-gpu', '--force-device-scale-factor=1',
                      '--start-maximized', *BACKGROUND_SWITCHES])
            self.context = await self.browser.new_context(
                viewport={'width': width, 'height': height}, device_scale_factor=1)
            self.scale_errors = await install_scale(self.context, 1.)
            self.page = await self.context.new_page()
            self.session = await self.context.new_cdp_session(self.page)
            bounds = await maximize_window(self.session)
            await self.page.bring_to_front()
            install_practice_start(self.folder)
            self.folder.joinpath('visible_capture.json').write_text(json.dumps(dict(
                backend='visible_chromium_lossless_png', headless=False,
                width=width, height=height, freshness_limit_ms=50,
                window_bounds=bounds,
                note='Screenshot request duration is not verified GPU presentation age.'), indent=2))
            print('[VISIBLE EVAL] Live Chromium window opened. Keep it visible; '
                  'do not resize or press game controls.', flush=True)
            return self
        except BaseException:
            await self.close()
            raise

    async def capture(self):
        if self.scale_errors:
            raise GuardUncertainError(self.scale_errors[-1])
        began = time.perf_counter()
        response = await asyncio.wait_for(self.session.send('Page.captureScreenshot', dict(
            format='png', fromSurface=True, captureBeyondViewport=False,
            optimizeForSpeed=True,
            clip=dict(x=0, y=0, width=self.width, height=self.height, scale=1))), timeout=5)
        data = base64.b64decode(response['data'], validate=True)
        image = cv2.imdecode(np.frombuffer(data, np.uint8), cv2.IMREAD_UNCHANGED)
        if image is None or image.dtype != np.uint8 or image.shape not in (
                (self.height, self.width, 3), (self.height, self.width, 4)):
            raise GuardUncertainError('Visible browser capture geometry/format changed')
        if image.shape[2] == 3:
            image = cv2.cvtColor(image, cv2.COLOR_BGR2BGRA)
        received = time.perf_counter()
        age = received - began
        if not 0 <= age <= .05:
            self.stale_frame_retries += 1
            raise StaleElectronFrame(f'Visible screenshot delivery {age*1000:.1f}ms exceeds 50ms')
        self.serial += 1
        self.reads += 1
        self.last_timing = dict(request_started_at=began, image_received_at=received,
            # Legacy pump field name: a delivery counter, NOT an Electron paint ID.
            electron_serial=self.serial, capture_kind='visible_chromium_lossless_png',
            frame_clock_kind='screenshot request duration; GPU presentation age unverified',
            capture_profile=dict(screenshot_and_decode_ms=age*1000))
        return image

    async def close(self):
        if self.closed:
            return
        self.closed = True
        if self.browser is not None:
            await self.browser.close()


def install():
    # Reuse the existing browser-image/temporal environment through its factory
    # interface. Despite that interface's legacy name, this opens headed Chromium.
    from tools.ppo_electron_capture import ElectronCapture
    ElectronCapture.launch = VisibleBrowserCapture.launch
