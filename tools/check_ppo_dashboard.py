"""Browser UI check using clearly labeled synthetic metrics, no game or training."""
import asyncio
import json
from pathlib import Path
import shutil
import tempfile
from tools.ppo_worker_preview import start_preview
from tools.ppo_survival_metrics import survival_curve


async def main():
    from playwright.async_api import async_playwright
    output=Path('reports/dashboard_ui_check_20261030_03')
    output.mkdir(exist_ok=False)
    with tempfile.TemporaryDirectory() as directory:
        folder=Path(directory)
        for i in range(6):
            worker=folder/f'worker_{i}'
            worker.mkdir()
            source=Path('reports/headless_canvas_v1_six_3min_20261030_01')/f'worker_{i}'/'live.jpg'
            if source.exists():
                shutil.copyfile(source,worker/'live.jpg')
            else:
                import cv2
                import numpy as np
                image=np.full((288,512,3),30,np.uint8)
                cv2.putText(image,f'Synthetic worker {i+1}',(30,140),cv2.FONT_HERSHEY_SIMPLEX,1,(160,220,180),2)
                cv2.imwrite(str(worker/'live.jpg'),image)
            (worker/'live.json').write_text(json.dumps(dict(phase='UI DEMO',steps=640,updated_at=0)))
        events=[dict(at=100+i*10,duration=3+i/100+(i%5)/10) for i in range(300)]
        metrics=survival_curve(events,100,3600)
        (folder/'survival.json').write_text(json.dumps(metrics))
        (folder/'runtime.json').write_text(json.dumps(dict(rounds=50,optimized_transitions=38400,
            elapsed_seconds=3000,requested_seconds=3600,finished=True,status='UI DEMO',optimized_per_second=18.4,
            workers=6,worker_rollout=128,aggregate_rollout=768,learning_rate=.0003,framing_profile='canvas_v1',
            collection_seconds=38,update_seconds=1.5,last_checkpoint_transitions=38400,
            training_stats={'train/approx_kl':.02,'train/clip_fraction':.15,'train/entropy_loss':-.5,
                            'train/explained_variance':.8,'train/value_loss':.3})))
        (folder/'telemetry.json').write_text(json.dumps(dict(cpu_mean_percent=40,cpu_busiest_core_percent=80,
            memory_percent=80,memory_available_gib=6,nvidia_csv='40 %, 15 %, 6500 MiB, 50 W',wall_time=0)))
        server=start_preview(folder,6,0)
        path=folder/'index.html'
        path.write_text(path.read_text(encoding='utf8').replace('Practice training</h1>',
            'UI DEMO — synthetic graph, saved policy images</h1>'),encoding='utf8')
        errors=[]
        try:
            async with async_playwright() as pw:
                browser=await pw.chromium.launch(headless=True)
                try:
                    page=await browser.new_page(viewport=dict(width=1920,height=1080))
                    page.on('pageerror',lambda e:errors.append(str(e)))
                    await page.goto(f'http://127.0.0.1:{server.server_port}/')
                    await page.wait_for_selector('#chart polyline')
                    await page.wait_for_function('Array.from(document.images).every(i=>i.complete && i.naturalWidth>0)')
                    layout=await page.evaluate('''() => ({columns:getComputedStyle(document.querySelector('main')).gridTemplateColumns.split(' ').length,
                        figures:document.querySelectorAll('figure').length,points:document.querySelectorAll('#chart circle').length,
                        horizontalOverflow:document.documentElement.scrollWidth>innerWidth})''')
                    assert layout['columns']==3 and layout['figures']==6 and not layout['horizontalOverflow'],layout
                    assert layout['points']==len(metrics['points']),layout
                    assert await page.locator('#chart polyline').get_attribute('stroke-width')=='3.5'
                    assert await page.locator('#chart circle').first.get_attribute('fill')=='#ff6868'
                    assert '0.0200' in await page.locator('#learning').inner_text()
                    assert '6500 MiB' in await page.locator('#system').inner_text()
                    after=survival_curve(events+[dict(at=3210,duration=7)],100,3600)
                    assert after['points'][:len(metrics['points'])]==metrics['points']
                    (folder/'survival.json').write_text(json.dumps(after))
                    await page.wait_for_function('n => document.querySelectorAll("#chart circle").length === n',
                                                 arg=len(after['points']))
                    await page.screenshot(path=str(output/'desktop_demo.png'),full_page=True)
                    await page.set_viewport_size(dict(width=800,height=1000))
                    columns=await page.evaluate("getComputedStyle(document.querySelector('main')).gridTemplateColumns.split(' ').length")
                    assert columns==2,columns
                    await page.screenshot(path=str(output/'tablet_demo.png'),full_page=True)
                    assert not errors,errors
                    (output/'result.json').write_text(json.dumps(dict(synthetic_data=True,layout=layout,
                        tablet_columns=columns,page_errors=errors),indent=2))
                    print(json.dumps(layout))
                finally:
                    await browser.close()
        finally:
            server.shutdown()
            server.server_close()


if __name__ == '__main__':
    asyncio.run(main())
