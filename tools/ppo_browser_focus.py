"""Bounded startup focus for the owned Practice browser; never sends inputs."""
import asyncio
import sys
import uuid


class WindowsForeground:
    def __init__(self):
        import ctypes
        from ctypes import wintypes
        self.ctypes = ctypes
        self.w = wintypes
        self.api = ctypes.WinDLL('user32', use_last_error=True)
        self.callback = ctypes.WINFUNCTYPE(wintypes.BOOL, wintypes.HWND, wintypes.LPARAM)
        self.api.EnumWindows.argtypes = [self.callback, wintypes.LPARAM]
        self.api.GetWindowTextLengthW.argtypes = [wintypes.HWND]
        self.api.GetWindowTextW.argtypes = [wintypes.HWND, wintypes.LPWSTR, ctypes.c_int]
        self.api.GetClassNameW.argtypes = [wintypes.HWND, wintypes.LPWSTR, ctypes.c_int]
        for name in ('IsWindowVisible', 'IsIconic', 'SetForegroundWindow'):
            getattr(self.api, name).argtypes = [wintypes.HWND]
        self.api.ShowWindow.argtypes = [wintypes.HWND, ctypes.c_int]
        self.api.GetForegroundWindow.restype = wintypes.HWND
        self.api.GetWindowThreadProcessId.argtypes = [wintypes.HWND, ctypes.POINTER(wintypes.DWORD)]
        self.api.GetWindowThreadProcessId.restype = wintypes.DWORD
        self.api.AttachThreadInput.argtypes = [wintypes.DWORD, wintypes.DWORD, wintypes.BOOL]
        self.api.BringWindowToTop.argtypes = [wintypes.HWND]
        self.api.SetActiveWindow.argtypes = [wintypes.HWND]
        self.api.SetActiveWindow.restype = wintypes.HWND
        self.kernel = ctypes.WinDLL('kernel32', use_last_error=True)
        self.kernel.GetCurrentThreadId.restype = wintypes.DWORD

    def find(self, marker):
        matches = []

        @self.callback
        def visit(hwnd, _):
            if self.api.IsWindowVisible(hwnd):
                title = self.ctypes.create_unicode_buffer(self.api.GetWindowTextLengthW(hwnd) + 1)
                cls = self.ctypes.create_unicode_buffer(256)
                self.api.GetWindowTextW(hwnd, title, len(title))
                self.api.GetClassNameW(hwnd, cls, len(cls))
                if marker in title.value and cls.value.startswith('Chrome_WidgetWin_'):
                    matches.append(hwnd)
            return True

        if not self.api.EnumWindows(visit, 0):
            raise RuntimeError('Could not enumerate browser windows')
        if len(matches) != 1:
            raise RuntimeError(f'Expected one owned Chromium window, found {len(matches)}')
        return matches[0]

    def activate(self, hwnd):
        if self.api.IsIconic(hwnd):
            self.api.ShowWindow(hwnd, 9)  # SW_RESTORE; do not resize a normal window.
        self.api.SetForegroundWindow(hwnd)

    def is_foreground(self, hwnd):
        return self.api.GetForegroundWindow() == hwnd

    def activate_attached(self, hwnd):
        """One temporary input-queue handoff; always detach, never synthesize keys."""
        current = self.kernel.GetCurrentThreadId()
        foreground = self.api.GetForegroundWindow()
        threads = {self.api.GetWindowThreadProcessId(hwnd, None)}
        if foreground:
            threads.add(self.api.GetWindowThreadProcessId(foreground, None))
        attached = []
        try:
            for other in sorted(threads):
                if other and other != current and self.api.AttachThreadInput(current, other, True):
                    attached.append(other)
            self.api.BringWindowToTop(hwnd)
            self.api.SetForegroundWindow(hwnd)
            self.api.SetActiveWindow(hwnd)
        finally:
            for other in reversed(attached):
                self.api.AttachThreadInput(current, other, False)

    def shell_activate(self, marker):
        # One supported Windows Script Host activation attempt, scoped to the
        # unique title already matched to this Chromium HWND. No SendKeys.
        import base64
        import subprocess
        if not marker.startswith('RS-focus-') or not marker[9:].isalnum():
            raise ValueError('Invalid owned-window marker')
        command = f"$app = New-Object -ComObject WScript.Shell; $null = $app.AppActivate('{marker}')"
        encoded = base64.b64encode(command.encode('utf-16-le')).decode('ascii')
        try:
            subprocess.run(['powershell.exe', '-NoProfile', '-NonInteractive',
                            '-WindowStyle', 'Hidden', '-EncodedCommand', encoded],
                           creationflags=subprocess.CREATE_NO_WINDOW,
                           capture_output=True, timeout=5, check=True)
        except (OSError, subprocess.SubprocessError):
            return


async def ensure_startup_focus(page, *, native=None, platform=None):
    """Identify this page by a temporary unique title, restore it, then verify.

    Windows may refuse activation. That is a stop, never permission to send a
    key, click, or relax a gameplay classifier. No focus retries during play.
    """
    platform = sys.platform if platform is None else platform
    await page.bring_to_front()
    if platform == 'win32':
        native = WindowsForeground() if native is None else native
        marker = 'RS-focus-' + uuid.uuid4().hex
        title = await page.title()
        try:
            await page.evaluate('(title) => { document.title = title; }', marker)
            hwnd = None
            for _ in range(3):
                await asyncio.sleep(.15)
                try:
                    hwnd = native.find(marker)
                    break
                except RuntimeError:
                    pass
            if hwnd is None:
                raise RuntimeError('Could not uniquely identify the owned Chromium window')
            for _ in range(3):
                native.activate(hwnd)
                await asyncio.sleep(.15)
                focused = await page.evaluate('document.hasFocus() && document.visibilityState === "visible"')
                if native.is_foreground(hwnd) and focused:
                    print('[PPO FOCUS] Owned Chromium is foreground; browser focus verified.', flush=True)
                    return
            native.shell_activate(marker)
            await asyncio.sleep(.15)
            focused = await page.evaluate('document.hasFocus() && document.visibilityState === "visible"')
            if native.is_foreground(hwnd) and focused:
                print('[PPO FOCUS] Owned Chromium activated by Windows shell; focus verified.', flush=True)
                return
            attached_activate = getattr(native, 'activate_attached', None)
            if attached_activate is not None:
                attached_activate(hwnd)
                await asyncio.sleep(.15)
                focused = await page.evaluate('document.hasFocus() && document.visibilityState === "visible"')
                if native.is_foreground(hwnd) and focused:
                    print('[PPO FOCUS] Owned Chromium activated by bounded queue handoff; focus verified.', flush=True)
                    return
            print('[PPO FOCUS] Windows denied automatic activation. Waiting up to 8s for '
                  'you to bring Chromium forward; no gameplay input will be sent.', flush=True)
            for _ in range(40):
                await asyncio.sleep(.2)
                focused = await page.evaluate('document.hasFocus() && document.visibilityState === "visible"')
                if native.is_foreground(hwnd) and focused:
                    print('[PPO FOCUS] Chromium foreground and browser focus verified.', flush=True)
                    return
            raise RuntimeError('Windows denied Chromium focus; no gameplay input sent. Bring the game forward and retry.')
        finally:
            await page.evaluate('(title) => { document.title = title; }', title)
    else:
        if not await page.evaluate('document.hasFocus() && document.visibilityState === "visible"'):
            raise RuntimeError('Browser focus could not be verified; no gameplay input sent')
