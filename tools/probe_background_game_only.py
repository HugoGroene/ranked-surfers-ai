"""Short local game-only browsing check: no captures, policy, or training."""
import asyncio
from pathlib import Path
from rs_practice_controller import async_playwright, find_canvas, click_practice, focus_canvas
from tools.ppo_background_display import BACKGROUND_SWITCHES, apply_viewport_layout


async def main():
    async with async_playwright() as pw:
        browser = await pw.chromium.launch(headless=False, args=[
            '--window-position=1920,0', '--window-size=1920,1032', *BACKGROUND_SWITCHES])
        try:
            context = await browser.new_context(no_viewport=True)
            page = await context.new_page()
            cdp = await context.new_cdp_session(page)
            window = await cdp.send('Browser.getWindowForTarget')
            await cdp.send('Browser.setWindowBounds', {'windowId': window['windowId'],
                'bounds': {'left':1920, 'top':0, 'width':1920, 'height':1032}})
            await cdp.send('Emulation.setFocusEmulationEnabled', {'enabled':True})
            await page.goto('http://localhost:8000', wait_until='domcontentloaded')
            canvas = await find_canvas(page, timeout=60)
            await click_practice(page, timeout=60)
            await apply_viewport_layout(page)
            await focus_canvas(canvas)
            await asyncio.sleep(.3)
            output = Path('reports/game_only_two_virtual_60hz_20261030_01')
            output.mkdir(exist_ok=False)
            await page.screenshot(path=str(output/'startup.png'))
            print('Saved one startup screenshot for gameplay verification; no capture loop.', flush=True)
            print('GAME-ONLY CHECK READY: browse normally for 45 seconds. No capture or AI.', flush=True)
            await asyncio.sleep(45)
        finally:
            await browser.close()


if __name__ == '__main__':
    asyncio.run(main())
