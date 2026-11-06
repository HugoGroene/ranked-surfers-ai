"""Visible-only desktop readback. All Windows/MSS calls use one owning thread."""
import ctypes
from ctypes import wintypes

from rs_v612_gameplay_guard import GuardUncertainError


class DesktopReader:
    def __init__(self, hwnd, width, height):
        import mss
        self.hwnd, self.width, self.height = hwnd, width, height
        self.user = ctypes.WinDLL('user32', use_last_error=True)
        signatures = {
            'GetForegroundWindow': ([], wintypes.HWND),
            'IsWindowVisible': ([wintypes.HWND], wintypes.BOOL),
            'IsIconic': ([wintypes.HWND], wintypes.BOOL),
            'GetClientRect': ([wintypes.HWND, ctypes.POINTER(wintypes.RECT)], wintypes.BOOL),
            'ClientToScreen': ([wintypes.HWND, ctypes.POINTER(wintypes.POINT)], wintypes.BOOL),
            'SetThreadDpiAwarenessContext': ([ctypes.c_void_p], ctypes.c_void_p),
        }
        for name, (args, result) in signatures.items():
            fn = getattr(self.user, name)
            fn.argtypes, fn.restype = args, result
        self.old_dpi = self.user.SetThreadDpiAwarenessContext(ctypes.c_void_p(-4))
        self.grabber = mss.MSS()
        try:
            self.initial_region = self.region()
        except BaseException:
            self.close()
            raise

    def region(self):
        u, hwnd = self.user, self.hwnd
        if not u.IsWindowVisible(hwnd) or u.IsIconic(hwnd) or u.GetForegroundWindow() != hwnd:
            raise GuardUncertainError('Keep the live game foreground and unobscured; desktop capture stopped')
        rect, origin = wintypes.RECT(), wintypes.POINT(0, 0)
        if not u.GetClientRect(hwnd, ctypes.byref(rect)) or not u.ClientToScreen(hwnd, ctypes.byref(origin)):
            raise GuardUncertainError('Cannot verify visible game client rectangle')
        if (rect.right-rect.left, rect.bottom-rect.top) != (self.width, self.height):
            raise GuardUncertainError('Visible game client size changed')
        return dict(left=origin.x, top=origin.y, width=self.width, height=self.height)

    def read(self):
        region = self.region()
        if region != self.initial_region:
            raise GuardUncertainError('Visible game moved; restart evaluation at its fixed geometry')
        shot = self.grabber.grab(region)
        # Check again so a focus/geometry change during readback rejects pixels.
        if self.region() != region:
            raise GuardUncertainError('Visible game geometry changed during capture')
        if (shot.width, shot.height) != (self.width, self.height):
            raise GuardUncertainError('Desktop capture dimensions changed')
        return bytes(shot.raw)

    def close(self):
        self.grabber.close()
        if self.old_dpi:
            self.user.SetThreadDpiAwarenessContext(self.old_dpi)


# Bounded histogram, no per-frame Python/CDP messages. rAF is a rendering
# scheduler proxy, NOT a Unity simulation or monitor presentation timestamp.
RAF_SCRIPT = """(() => {
  if (window.__ppoRenderTiming) return;
  if (window !== window.top && !location.pathname.endsWith('/arena.html')) return;
  const s = {count:0, total_ms:0, max_ms:0, over25:0, over50:0,
             histogram_ms:Array(1001).fill(0)};
  window.__ppoRenderTiming = s;
  let previous;
  function tick(t) {
    if (previous !== undefined && document.visibilityState === 'visible') {
      const dt=t-previous;
      s.count++; s.total_ms+=dt; s.max_ms=Math.max(s.max_ms,dt);
      s.over25+=Number(dt>25); s.over50+=Number(dt>50);
      s.histogram_ms[Math.min(1000,Math.max(0,Math.round(dt)))]++;
    }
    previous=t; requestAnimationFrame(tick);
  }
  document.addEventListener('visibilitychange',()=>{previous=undefined;});
  requestAnimationFrame(tick);
})();"""


def summarize_render(s):
    if not s or not s.get('count'):
        return dict(status='NO_SAMPLES')
    def percentile(q):
        seen = 0
        for value, count in enumerate(s['histogram_ms']):
            seen += count
            if seen >= s['count']*q:
                return value
    return dict(status='COLLECTED', samples=s['count'],
                mean_interval_ms=s['total_ms']/s['count'],
                p50_interval_ms=percentile(.5), p95_interval_ms=percentile(.95),
                p99_interval_ms=percentile(.99), max_interval_ms=s['max_ms'],
                intervals_over_25ms=s['over25'], intervals_over_50ms=s['over50'],
                note='Visible-page requestAnimationFrame intervals across startup/gameplay/menus; '
                     'histogram bins cap at 1000 ms. Not verified Unity ticks or monitor FPS.')


async def collect_render(page):
    contexts = []
    for frame in page.frames:
        try:
            sample = await frame.evaluate('''() => ({timing:window.__ppoRenderTiming || null,
                game_canvas:!!document.querySelector('canvas#game'), url:location.href})''')
            if sample['timing']:
                contexts.append(dict(url=sample['url'], game_canvas=sample['game_canvas'],
                                     **summarize_render(sample['timing'])))
        except Exception as exc:
            contexts.append(dict(url=frame.url,status='UNAVAILABLE',error=str(exc)))
    game = next((s for s in contexts if s.get('game_canvas') and s.get('status')=='COLLECTED'), None)
    selected = game or next((s for s in contexts if s.get('status')=='COLLECTED'), {})
    return dict(selected, measured_context='unity_iframe' if game else 'outer_page_only',
                contexts=contexts)
