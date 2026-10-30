"""Opt-in isolated-display capture for LOCAL Practice; no Windows input hooks.

The raw guard image remains the selected display. The policy/detector source
can explicitly use a pixel-calibrated canvas instead. Legacy defaults are intact.
"""
import asyncio
import json
import time
from urllib.parse import urlsplit

import cv2
import numpy as np

from rs_v612_gameplay_guard import GuardUncertainError


RECT_KEYS = ('left', 'top', 'width', 'height')
BACKGROUND_SWITCHES = (
    '--disable-background-timer-throttling',
    '--disable-backgrounding-occluded-windows',
    '--disable-renderer-backgrounding',
)


def window_work_area(display):
    """Avoid the taskbar: an ordinary window must fit the monitor work area."""
    import sys
    if sys.platform != 'win32':
        return rectangle(display)
    import ctypes as c
    from ctypes import wintypes as w
    class MonitorInfo(c.Structure):
        _fields_ = [('size', w.DWORD), ('monitor', w.RECT), ('work', w.RECT), ('flags', w.DWORD)]
    api = c.WinDLL('user32', use_last_error=True)
    api.MonitorFromPoint.argtypes = [w.POINT, w.DWORD]
    api.MonitorFromPoint.restype = w.HMONITOR
    api.GetMonitorInfoW.argtypes = [w.HMONITOR, c.POINTER(MonitorInfo)]
    point = w.POINT(display['left']+display['width']//2, display['top']+display['height']//2)
    handle = api.MonitorFromPoint(point, 0)
    info = MonitorInfo(size=c.sizeof(MonitorInfo))
    if not handle or not api.GetMonitorInfoW(handle, c.byref(info)):
        raise RuntimeError('Cannot obtain selected display work area')
    work = dict(left=info.work.left, top=info.work.top,
                width=info.work.right-info.work.left, height=info.work.bottom-info.work.top)
    if not contains(display, work):
        raise RuntimeError('Selected display work area does not match its physical rectangle')
    return work


def rectangle(value):
    return {key: int(value[key]) for key in RECT_KEYS}


def contains(outer, inner):
    return (inner['width'] > 0 and inner['height'] > 0
            and inner['left'] >= outer['left'] and inner['top'] >= outer['top']
            and inner['left']+inner['width'] <= outer['left']+outer['width']
            and inner['top']+inner['height'] <= outer['top']+outer['height'])


def select_display(monitors, index):
    if index < 1 or index >= len(monitors):
        raise ValueError(f'Display {index} unavailable; individual displays: 1..{len(monitors)-1}')
    selected = dict(monitors[index])
    if selected.get('is_primary', index == 1):
        raise ValueError('Background mode requires a non-primary display; extend the desktop first')
    if selected['width'] < 960 or selected['height'] < 540:
        raise ValueError('Background display must be at least 960x540')
    return selected


def display_identity(display):
    return (tuple(rectangle(display).values()), display.get('unique_id'), display.get('name'))


def canvas_from_display(shot, display, canvas):
    if not contains(display, canvas):
        raise GuardUncertainError('[BACKGROUND] Canvas left its selected display; NO input allowed')
    if tuple(shot.shape[:2]) != (display['height'], display['width']):
        raise GuardUncertainError('[BACKGROUND] Capture size changed; NO input allowed')
    x, y = canvas['left']-display['left'], canvas['top']-display['top']
    return shot[y:y+canvas['height'], x:x+canvas['width']]


def validate_window(display, bounds, metrics):
    if bounds.get('windowState') == 'minimized':
        raise GuardUncertainError('[BACKGROUND] Chromium minimized; capture is not trustworthy')
    if abs(metrics['dpr']-1.) > .001:
        raise GuardUncertainError('[BACKGROUND] Prototype requires 100% display scaling (DPR 1)')
    if metrics['visibility'] != 'visible':
        raise GuardUncertainError('[BACKGROUND] Game page is hidden; NO input allowed')
    if not contains(display, rectangle(bounds)):
        raise GuardUncertainError('[BACKGROUND] Chromium moved outside its selected display')


async def apply_viewport_layout(page):
    stages = page.locator('.game-stage')
    if await stages.count() != 1:
        raise GuardUncertainError('[BACKGROUND] Expected one local game-stage for viewport layout')
    await stages.evaluate('''el => {
        for (const [key,value] of Object.entries({position:'fixed',inset:'0',
            width:'100vw',height:'100vh',maxHeight:'none',border:'0',
            borderRadius:'0',zIndex:'2147483647'})) el.style.setProperty(key,value,'important');
        // Keep the entire ancestor chain rendered. A visible child below a
        // hidden ancestor can still be treated as an unimportant frame.
        // Never override visibility on descendants: pause/menu controls own it.
        for (let node=el;node;node=node.parentElement) {
            node.style.setProperty('visibility','visible','important');
            if(node===document.body)break;
            if(node.parentElement)for(const sibling of node.parentElement.children){
                if(sibling!==node)sibling.style.setProperty('visibility','hidden','important');
            }
        }
        document.documentElement.style.overflow='hidden';
        document.body.style.overflow='hidden';
    }''')


class BackgroundDisplayMixin:
    def _background_enabled(self):
        return getattr(self, '_background_display_index', None) is not None

    async def _open_browser(self):
        if not self._background_enabled():
            return await super()._open_browser()
        import mss
        import rs_practice_controller as ctrl
        from rs_v612_demo import locate_canvas_in_desktop

        parsed = urlsplit(self.args.url)
        if parsed.scheme != 'http' or parsed.hostname not in ('localhost', '127.0.0.1') or parsed.port != 8000:
            raise ValueError('Background displays support only LOCAL Practice on port 8000')
        session = getattr(self, '_persistent_browser_session', None)
        reused = session is not None and session.restore(self)
        if self.grabber is None:
            if getattr(self, '_background_capture_mode', 'mss') == 'dxgi':
                from tools.ppo_dxgi_source import DxgiDisplayGrabber
                self.grabber = DxgiDisplayGrabber(self._background_display_index)
            else:
                self.grabber = mss.MSS()
        self._background_monitor = select_display(self.grabber.monitors, self._background_display_index)
        self.region = rectangle(self._background_monitor)
        work_area = window_work_area(self.region)
        if not reused:
            self.playwright = await ctrl.async_playwright().start()
            r = work_area
            self.browser = await self.playwright.chromium.launch(headless=False, args=[
                f'--window-position={r["left"]},{r["top"]}',
                f'--window-size={r["width"]},{r["height"]}', *BACKGROUND_SWITCHES])
            self.context = await self.browser.new_context(no_viewport=True)
            self.page = await self.context.new_page()
        self._background_cdp = await self.context.new_cdp_session(self.page)
        window = await self._background_cdp.send('Browser.getWindowForTarget')
        self._background_window_id = window['windowId']
        await self._background_cdp.send('Browser.setWindowBounds', {
            'windowId': self._background_window_id, 'bounds': {'windowState': 'normal'}})
        await self._background_cdp.send('Browser.setWindowBounds', {
            'windowId': self._background_window_id, 'bounds': work_area})
        await self.page.goto(self.args.url, wait_until='domcontentloaded', timeout=45000)
        await self._background_cdp.send('Emulation.setFocusEmulationEnabled', {'enabled': True})
        self.canvas = await ctrl.find_canvas(self.page, timeout=60)
        await ctrl.focus_canvas(self.canvas)
        # Use the local site's existing game-stage as a viewport-sized panel.
        # Native fullscreen can minimize on OS focus loss; background workers
        # must not depend on that global foreground state. Keep all stage UI.
        await apply_viewport_layout(self.page)
        await asyncio.sleep(.3)
        self._background_view = await self._background_geometry()
        await self._background_verify_window()
        error = None
        for attempt in range(8):
            try:
                before = await self._background_geometry()
                png = await ctrl.screenshot_canvas(self.page, self.canvas, image_type='png')
                reference = cv2.imdecode(np.frombuffer(png, np.uint8), cv2.IMREAD_COLOR)
                shot = np.asarray(self.grabber.grab(self.region))
                cv2.imwrite(str(self.output_dir/f'calibration_{attempt}_display.png'), shot)
                cv2.imwrite(str(self.output_dir/f'calibration_{attempt}_reference.png'), reference)
                after = await self._background_geometry()
                if before != after:
                    raise RuntimeError('Canvas moved during calibration; waiting for a stable page')
                self._background_canvas, score = locate_canvas_in_desktop(
                    reference, shot, left=self.region['left'], top=self.region['top'], min_score=.85)
                canvas_from_display(shot, self.region, self._background_canvas)
                self._background_view = after
                await self._background_verify_window()
                cv2.imwrite(str(self.output_dir/'background_calibration_display.png'), shot)
                cv2.imwrite(str(self.output_dir/'background_calibration_reference.png'), reference)
                break
            except Exception as exc:
                error = exc
                await asyncio.sleep(.2)
        else:
            raise GuardUncertainError(f'[BACKGROUND] Canvas pixel calibration failed: {error}')
        self._game_fullscreen = False
        self._background_checked_at = 0.
        self._background_previews_saved = False
        self._background_geometry_invalid = None
        self._background_native_window = None
        import sys
        if sys.platform == 'win32':
            from tools.ppo_native_window import identify_native_window
            self._background_native_window = await identify_native_window(self.page, self.region)
        await self._background_install_geometry_watch()
        self._background_write_metadata(score)
        if session is not None:
            session.remember(self)
        print('[BACKGROUND] Display and canvas calibrated. No Windows foreground activation; '
              'background rendering/input response still require live validation.', flush=True)

    async def _background_geometry(self):
        metrics = await self.page.evaluate('''() => ({dpr: devicePixelRatio,
            x: screenX, y: screenY, width: innerWidth, height: innerHeight,
            visibility: document.visibilityState})''')
        box = await self.canvas.bounding_box()
        if box is None:
            raise GuardUncertainError('[BACKGROUND] Game canvas disappeared')
        return {'metrics': metrics, 'canvas': box}

    async def _background_verify_window(self):
        window = await self._background_cdp.send('Browser.getWindowBounds', {
            'windowId': self._background_window_id})
        view = await self._background_geometry()
        validate_window(self.region, window['bounds'], view['metrics'])
        if view != self._background_view:
            raise GuardUncertainError('[BACKGROUND] Window/canvas geometry changed; restart calibration')

    async def _background_install_geometry_watch(self):
        import uuid
        name = '_rsGeometry_' + uuid.uuid4().hex
        def changed(_source, reason):
            self._background_geometry_invalid = str(reason)
        await self.page.expose_binding(name, changed)
        # Observe the canvas in its own frame, and the fixed stage in the root.
        # Layout reads occur only on actual resize/removal/attribute changes.
        script = '''(el,name) => {
            const w=el.ownerDocument.defaultView;
            const baseline=el.getBoundingClientRect();
            let reported=false;
            const report=reason=>{if(!reported){reported=true;w[name](reason);}};
            const check=()=>{const r=el.getBoundingClientRect();
                if(!el.isConnected)return report('calibrated element removed');
                if(['x','y','width','height'].some(k=>Math.abs(r[k]-baseline[k])>1))report('calibrated element moved/resized');
            };
            new w.ResizeObserver(check).observe(el);
            new w.MutationObserver(check).observe(el,{attributes:true});
            new w.MutationObserver(()=>{if(!el.isConnected)report('calibrated element replaced');})
                .observe(el.ownerDocument,{childList:true,subtree:true});
            w.addEventListener('scroll',()=>report('calibrated document scrolled'),true);
            w.addEventListener('pagehide',()=>report('calibrated document hidden'));
        }'''
        await self.canvas.evaluate(script, name)
        await self.page.locator('.game-stage').evaluate(script, name)

    def _background_write_metadata(self, score):
        data = {'experimental': True, 'display_index': self._background_display_index,
                'display': self._background_monitor, 'canvas': self._background_canvas,
                'pixel_similarity': score, 'view': self._background_view,
                'policy_source': 'calibrated_canvas' if self._background_canvas_policy else 'legacy_display_crop',
                'raw_guard_source': 'selected_display', 'page_focus_emulation': True,
                'layout': 'viewport_game_stage_not_native_fullscreen',
                'capture_backend': getattr(self, '_background_capture_mode', 'mss'),
                'input_response_verified': False}
        (self.output_dir/'background_display.json').write_text(json.dumps(data, indent=2), encoding='utf8')

    async def _sync_capture_region(self):
        if not self._background_enabled():
            return await super()._sync_capture_region()
        now = time.perf_counter()
        if now-getattr(self, '_background_checked_at', 0.) < .5:
            return
        try:
            if getattr(self, '_background_geometry_invalid', None):
                raise GuardUncertainError('[BACKGROUND] '+self._background_geometry_invalid)
            native = getattr(self, '_background_native_window', None)
            if native is not None:
                native.verify()
            else:
                import mss
                with mss.MSS() as inventory:
                    selected = select_display(inventory.monitors, self._background_display_index)
                if display_identity(selected) != display_identity(self._background_monitor):
                    raise GuardUncertainError('[BACKGROUND] Display topology changed; NO input allowed')
                await self._background_verify_window()
        except (ValueError, GuardUncertainError) as exc:
            self._guard_hard_stopped = True
            raise GuardUncertainError(f'[BACKGROUND] Capture verification stopped: {exc}') from exc
        self._background_checked_at = now

    async def _anchor_capture_region(self):
        if not self._background_enabled():
            return await super()._anchor_capture_region()
        await self._background_verify_window()

    def _background_present_at(self):
        sample = getattr(getattr(self, 'grabber', None), 'last_sample', None)
        return sample.present_at if sample is not None else None

    def _background_action_frame_current(self):
        sample = getattr(getattr(self, 'grabber', None), 'last_sample', None)
        return sample is None or (0 <= time.perf_counter()-sample.received_at <= .020
                                  and 0 <= sample.received_at-sample.present_at <= .050)

    async def _background_retry_action_frame(self, action):
        retries = getattr(self, '_background_frame_retries', 0)
        if retries >= 6:
            self._guard_hard_stopped = True
            raise GuardUncertainError('[BACKGROUND] No recent acquired GPU frame; NO movement sent')
        self._background_frame_retries = retries+1
        try:
            await asyncio.sleep(.010)
            await self._capture()
            # Re-run the existing full guard. A timestamp alone never permits input.
            return await self._step_browser(action)
        finally:
            self._background_frame_retries = retries

    def _background_resync_present(self, previous):
        sample = getattr(getattr(self, 'grabber', None), 'last_sample', None)
        if sample is None:
            return True, previous
        valid = sample.fresh and sample.serial != previous and time.perf_counter()-sample.present_at <= .050
        return valid, sample.serial

    async def _calibrate_policy_canvas(self):
        import rs_practice_controller as ctrl
        from rs_v612_demo import locate_canvas_in_desktop
        reference = cv2.imdecode(np.frombuffer(await ctrl.screenshot_canvas(
            self.page,self.canvas,image_type='png'),np.uint8),cv2.IMREAD_COLOR)
        shot = np.asarray(self.grabber.grab(self.region))
        self._background_canvas, score = locate_canvas_in_desktop(reference,shot,
            left=self.region['left'],top=self.region['top'],min_score=.85)
        self._background_geometry_invalid = None
        await self._background_install_geometry_watch()
        (self.output_dir/'framing_calibration.json').write_text(json.dumps(dict(
            profile='canvas_v1',canvas=self._background_canvas,similarity=score)),encoding='utf8')

    def _native_frames(self):
        if getattr(self,'_framing_profile','legacy') == 'canvas_v1':
            from tools.ppo_framing import canvas_observation
            if getattr(self,'_background_geometry_invalid',None):
                self._guard_hard_stopped = True
                raise GuardUncertainError(self._background_geometry_invalid)
            source = canvas_from_display(self._v2_last_bgra,self.region,self._background_canvas)
            small,native = canvas_observation(source,self.target_wh,
                cv2.INTER_AREA if self.resize_method=='area' else cv2.INTER_LINEAR)
            self.native_image_wh = (native.shape[1],native.shape[0])
            self.last_source_native = native
            if not getattr(self,'_canvas_profile_preview_saved',False):
                for name,bgr in [('canvas_v1_native.png',cv2.cvtColor(native,cv2.COLOR_RGB2BGR)),
                                 ('canvas_v1_policy.png',cv2.cvtColor(small.transpose(1,2,0),cv2.COLOR_RGB2BGR))]:
                    if not cv2.imwrite(str(self.output_dir/name),bgr):
                        raise RuntimeError('Could not save framing evidence')
                (self.output_dir/'framing_profile.json').write_text(json.dumps(dict(
                    profile='canvas_v1',native_wh=self.native_image_wh,target_wh=self.target_wh,
                    canvas=self._background_canvas,full_canvas=True)),encoding='utf8')
                self._canvas_profile_preview_saved = True
            return small,native
        if not self._background_enabled() or not self._background_canvas_policy:
            return super()._native_frames()
        import rs_v5_segmentation_live_benchmark as bench
        began = time.perf_counter()
        source = canvas_from_display(self._v2_last_bgra, self.region, self._background_canvas)
        native = cv2.cvtColor(bench.crop_game(source, self.crop), cv2.COLOR_BGRA2RGB)
        interp = cv2.INTER_AREA if self.resize_method == 'area' else cv2.INTER_LINEAR
        small = bench.fit_rgb_to_target(native, self.target_wh, interpolation=interp)
        self.native_image_wh = (native.shape[1], native.shape[0])
        self.last_source_native = native
        self.game_crop_ms.append((time.perf_counter()-began)*1000)
        if not self._background_previews_saved:
            for name, bgr in (('background_display.png', self._v2_last_bgra[:, :, :3]),
                              ('background_canvas.png', source[:, :, :3]),
                              ('background_policy.png', cv2.cvtColor(small, cv2.COLOR_RGB2BGR))):
                if not cv2.imwrite(str(self.output_dir/name), bgr):
                    raise RuntimeError(f'Could not save {name}')
            self._background_previews_saved = True
        return np.ascontiguousarray(small.transpose(2, 0, 1)), native


def main():
    """List physical/virtual displays without launching a browser."""
    import mss
    with mss.MSS() as capture:
        print(json.dumps([dict(index=i, **m) for i, m in enumerate(capture.monitors) if i], indent=2))


if __name__ == '__main__':
    main()
