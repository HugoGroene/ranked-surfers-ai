"""Read-only Windows window/display checks; no activation, clicks or key hooks."""
import asyncio
import ctypes as c
from ctypes import wintypes as w
import uuid

from rs_v612_gameplay_guard import GuardUncertainError


class MonitorInfo(c.Structure):
    _fields_ = [('size', w.DWORD), ('monitor', w.RECT), ('work', w.RECT),
                ('flags', w.DWORD), ('device', w.WCHAR*32)]


def rect_tuple(rect):
    return (rect.left, rect.top, rect.right-rect.left, rect.bottom-rect.top)


class NativeWindowCheck:
    def __init__(self, foreground_api, handle, display):
        self.foreground_api, self.handle = foreground_api, handle
        self.api = foreground_api.api
        self.api.GetWindowRect.argtypes = [w.HWND, c.POINTER(w.RECT)]
        self.api.MonitorFromPoint.argtypes = [w.POINT, w.DWORD]
        self.api.MonitorFromPoint.restype = w.HMONITOR
        self.api.GetMonitorInfoW.argtypes = [w.HMONITOR, c.POINTER(MonitorInfo)]
        self.display = tuple(display[k] for k in ('left','top','width','height'))
        self.point = w.POINT(display['left']+display['width']//2, display['top']+display['height']//2)
        self.expected_window = self.window_rectangle()
        self.expected_monitor = self.monitor_snapshot()
        self.verify()

    def window_rectangle(self):
        rect = w.RECT()
        if not self.api.GetWindowRect(self.handle, c.byref(rect)):
            raise GuardUncertainError('[BACKGROUND] Owned window disappeared')
        return rect_tuple(rect)

    def monitor_snapshot(self):
        handle = self.api.MonitorFromPoint(self.point, 0)
        info = MonitorInfo(size=c.sizeof(MonitorInfo))
        if not handle or not self.api.GetMonitorInfoW(handle, c.byref(info)):
            raise GuardUncertainError('[BACKGROUND] Selected display disappeared')
        if rect_tuple(info.monitor) != self.display:
            raise GuardUncertainError('[BACKGROUND] Selected display rectangle changed')
        return str(info.device), rect_tuple(info.monitor), rect_tuple(info.work)

    def verify(self):
        if self.api.IsIconic(self.handle) or not self.api.IsWindowVisible(self.handle):
            raise GuardUncertainError('[BACKGROUND] Owned window minimized or hidden')
        if self.window_rectangle() != self.expected_window:
            raise GuardUncertainError('[BACKGROUND] Owned window moved or resized')
        if self.monitor_snapshot() != self.expected_monitor:
            raise GuardUncertainError('[BACKGROUND] Display/work area changed')


async def identify_native_window(page, display):
    from tools.ppo_browser_focus import WindowsForeground
    native = WindowsForeground()
    marker = 'RS-background-' + uuid.uuid4().hex
    original = await page.title()
    try:
        await page.evaluate('(value)=>document.title=value', marker)
        for attempt in range(8):
            await asyncio.sleep(.1)
            try:
                return NativeWindowCheck(native, native.find(marker), display)
            except RuntimeError:
                if attempt == 7:
                    raise
    finally:
        await page.evaluate('(value)=>document.title=value', original)
