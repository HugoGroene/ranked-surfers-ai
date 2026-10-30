"""Reset a legitimately time-limited local episode without movement keys."""
from urllib.parse import urlsplit

from rs_v612_gameplay_guard import GuardUncertainError


async def reload_after_time_limit(env):
    import rs_practice_controller as ctrl

    url = urlsplit(env.args.url)
    if url.scheme != 'http' or url.hostname not in ('localhost', '127.0.0.1') or url.port != 8000:
        raise GuardUncertainError('Time-limit reload is restricted to local Practice')
    headless_reload = getattr(env,'_headless_reload_after_time_limit',None)
    if headless_reload is not None:
        await headless_reload()
        print('[PPO TIME LIMIT] Reloaded headless local canvas; normal startup guard follows.',flush=True)
        return
    focused = await env.page.evaluate('document.hasFocus() && document.visibilityState === "visible"')
    if not focused:
        raise GuardUncertainError('Focus lost before time-limit reset; no keys sent')
    await env.page.goto(env.args.url, wait_until='domcontentloaded', timeout=45000)
    env._game_fullscreen = False
    env.canvas = await ctrl.find_canvas(env.page, timeout=60)
    await env._anchor_capture_region()
    if env.args.fullscreen:
        await env._enter_game_fullscreen()
    if getattr(env,'_framing_profile','legacy') == 'canvas_v1' and not env._background_enabled():
        await env._calibrate_policy_canvas()
    if not await env.page.evaluate('document.hasFocus() && document.visibilityState === "visible"'):
        raise GuardUncertainError('Focus lost during time-limit reset; no keys sent')
    print('[PPO TIME LIMIT] Reloaded local game without movement; normal MENU/GAMEPLAY startup guard follows.', flush=True)
