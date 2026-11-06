"""Offline saved-image parity and cost check for exact delivered crop reuse."""
import argparse
from datetime import datetime
import json
from pathlib import Path
import sys
import time
from types import SimpleNamespace
import cv2
import numpy as np
from record import ROOT
sys.path[:0]=[str(ROOT),str(ROOT/'development/experiments/road_vision')]
from crop_reuse import install
from road_crop import observation
from tools.ppo_background_display import canvas_from_display
from tools.ppo_headless_env import compose_guard_frame


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--report',type=Path,required=True,help='Training folder containing worker_N canvas previews')
    args=parser.parse_args()
    cv2.setNumThreads(1)
    results=[]
    for folder in sorted(args.report.glob('worker_*')):
        file=folder/'canvas_v1_native.png'
        if not file.is_file():continue
        config=json.loads((folder/'framing_profile.json').read_text())['road_vision']
        canvas=cv2.cvtColor(cv2.imread(str(file)),cv2.COLOR_BGR2BGRA)
        prepared=observation(canvas,config)
        raw=SimpleNamespace(target_wh=tuple(config['size']),_framing_profile='canvas_v1',
            _canvas_profile_preview_saved=True,_road_preview_saved=True,
            _headless_capture_timing=dict(request_started_at=10.,electron_serial=5),
            pump=SimpleNamespace(delivered_prepared=(10.,5,prepared)),
            region=dict(left=0,top=0,width=1920,height=1080),
            _background_canvas=dict(left=0,top=87,width=1920,height=945))
        raw._v2_last_bgra=compose_guard_frame(canvas,0,87)
        raw.grabber=SimpleNamespace(image=raw._v2_last_bgra)
        def baseline():
            return observation(canvas_from_display(raw._v2_last_bgra,raw.region,raw._background_canvas),config)
        raw._native_frames=baseline
        async def close():pass
        raw._close_browser=close
        trainer=SimpleNamespace(create_live_env=lambda:(None,raw))
        install(trainer);trainer.create_live_env()
        expected=baseline();actual=raw._native_frames()
        equal=all(np.array_equal(x,y) for x,y in zip(expected,actual))
        if not equal:raise AssertionError('Crop pixels changed')
        timings={}
        # Alternate order to reduce clock/order bias. No live game is running.
        samples=dict(baseline=[],reuse=[])
        methods=[('baseline',baseline),('reuse',raw._native_frames)]
        for index in range(100):
            for name,fn in methods[::1 if index%2==0 else -1]:
                began=time.perf_counter();fn();samples[name].append((time.perf_counter()-began)*1000)
        for name,values in samples.items():
            timings[name]=dict(median_ms=float(np.median(values)),p95_ms=float(np.percentile(values,95)))
        results.append(dict(worker=folder.name,pixels_equal=equal,**timings))
    if not results:raise RuntimeError('No saved canvas images found')
    output=ROOT/'reports'/datetime.now().strftime('crop_replay_%Y%m%d_%H%M%S.json')
    output.write_text(json.dumps(dict(source=str(args.report),results=results,
        note='Isolated repeated preprocessing cost, not end-to-end gameplay latency.'),indent=2))
    print(output)
    print(json.dumps(results,indent=2))


if __name__=='__main__':main()
