"""Local headless rendering/capture prototype. No PPO, movement or training."""
import argparse
import asyncio
import json
from pathlib import Path
import time

import cv2
import numpy as np
from rs_practice_controller import async_playwright, find_canvas, click_practice, focus_canvas, screenshot_canvas
from tools.ppo_background_display import BACKGROUND_SWITCHES, apply_viewport_layout


async def worker(pw, root, index, seconds):
    output = root / f'worker_{index}'
    output.mkdir()
    browser = await pw.chromium.launch(headless=True, args=['--enable-gpu', *BACKGROUND_SWITCHES])
    result = {'worker': index, 'training': False, 'movement_inputs': False,
              'gameplay_verified': False, 'error': None}
    try:
        context = await browser.new_context(viewport={'width':1904, 'height':937}, device_scale_factor=1)
        page = await context.new_page()
        errors = []
        page.on('pageerror', lambda error: errors.append(str(error)[:500]))
        cdp = await context.new_cdp_session(page)
        await cdp.send('Emulation.setFocusEmulationEnabled', {'enabled':True})
        await page.goto('http://localhost:8000', wait_until='domcontentloaded', timeout=45000)
        canvas = await find_canvas(page, timeout=60)
        result['renderer'] = await canvas.evaluate('''el => {
          const gl=el.getContext('webgl2')||el.getContext('webgl');
          if(!gl)return 'NO_WEBGL';
          const ext=gl.getExtension('WEBGL_debug_renderer_info');
          return ext?gl.getParameter(ext.UNMASKED_RENDERER_WEBGL):gl.getParameter(gl.RENDERER);
        }''')
        # Let Unity finish initial loading before requesting local Practice.
        await asyncio.sleep(3)
        await click_practice(page, timeout=60)
        await apply_viewport_layout(page)
        await focus_canvas(canvas)
        print(f'Worker {index}: renderer={result["renderer"]}; capturing for {seconds}s', flush=True)
        timings, changes = [], []
        previous = None
        deadline = time.perf_counter()+seconds
        count = 0
        while time.perf_counter() < deadline:
            start = time.perf_counter()
            data = await screenshot_canvas(page, canvas, image_type='jpeg', quality=90)
            frame = cv2.imdecode(np.frombuffer(data, np.uint8), cv2.IMREAD_COLOR)
            if frame is None:
                raise RuntimeError('Browser returned an undecodable frame')
            timings.append((time.perf_counter()-start)*1000)
            small = cv2.resize(frame, (160,90))
            if previous is not None:
                changes.append(float(np.abs(small.astype(float)-previous).mean()))
            previous = small.astype(float)
            count += 1
            if count in (1, 5, 20):
                (output/f'frame_{count:04}.jpg').write_bytes(data)
            last = data
        (output/'last.jpg').write_bytes(last)
        result.update(captures=count, capture_ms=dict(zip(('median','p95','max'),
                      map(float, np.percentile(timings,[50,95,100])))),
                      capture_rate=count/seconds, changed_frames=sum(x>1 for x in changes),
                      frame_shape=list(frame.shape), page_errors=errors,
                      note='Image motion does not prove gameplay. Review saved images; no input-response or PPO validation.')
    except Exception as exc:
        result['error'] = repr(exc)
    finally:
        await browser.close()
        (output/'result.json').write_text(json.dumps(result, indent=2), encoding='utf-8')
    return result


async def main(args):
    root = Path(args.report)
    root.mkdir(parents=True, exist_ok=False)
    async with async_playwright() as pw:
        results = await asyncio.gather(*(worker(pw,root,i,args.seconds) for i in range(args.workers)))
    report = {'prototype_only':True, 'workers':results, 'virtual_displays_required':False}
    (root/'report.json').write_text(json.dumps(report, indent=2), encoding='utf-8')
    print(json.dumps(report, indent=2))
    if any(r['error'] for r in results):
        raise SystemExit(1)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--workers', type=int, choices=[1,2], default=1)
    parser.add_argument('--seconds', type=int, choices=range(5,31), default=15)
    parser.add_argument('--report', required=True)
    asyncio.run(main(parser.parse_args()))
