#!/usr/bin/env python3
"""Save unlabeled 256x144 V2 screenshots while a HUMAN plays local Practice.

Does not record human key/action labels and does not train a model.
Uses the verified fullscreen + R controller; env.step(0) sends no arrow key.

 python -u rs_v2_capture_hazards_manual.py --fullscreen --minutes 10
"""
import argparse
import re
import sys
import time
from pathlib import Path

import cv2
import numpy as np

import rs_v2_vision as vision
import train_rs_v2_verify as v2


def next_episode(root):
    n=0
    for p in root.glob('ep*_frame*.png'):
        m=re.match(r'ep(\d+)_frame',p.name)
        if m:n=max(n,int(m.group(1)))
    return n+1


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--fullscreen',action='store_true')
    p.add_argument('--minutes',type=float,default=10)
    p.add_argument('--interval',type=float,default=.45,help='Min seconds between saved frames')
    p.add_argument('--warmup-seconds',type=float,default=1.1,help='Ignore first seconds of each run to reduce camera-pan frames')
    p.add_argument('--max-images',type=int,default=350)
    p.add_argument('--output',type=Path,default=Path('ai_data/v2_hazard_images'))
    a=p.parse_args()
    if a.minutes<=0 or a.interval<.15 or a.warmup_seconds<0 or a.max_images<=0:p.error('Invalid timing')
    argv=sys.argv
    sys.argv=['train_rs_v2_verify.py','--mode','smoke',
              '--run-name','v2_human_hazard_capture',
              '--max-minutes',str(a.minutes),'--episode-max-seconds','60',
              '--fps','20']
    if a.fullscreen:sys.argv.append('--fullscreen')
    try:args=v2.parser_args()
    finally:sys.argv=argv
    a.output.mkdir(parents=True,exist_ok=True)
    session_dir=v2.RUN_ROOT/'v2_human_hazard_capture'
    session_dir.mkdir(parents=True,exist_ok=True)
    env=None
    number=next_episode(a.output)
    saved=0
    deadline=time.monotonic()+60*a.minutes
    try:
        env=v2.V2PracticeEnv(args,session_dir,v2.DemoStore(v2.DEMO_ROOT,load_existing=False))
        print('[HUMAN CAPTURE] Click the game if needed and PLAY using arrow keys.',flush=True)
        print('[HUMAN CAPTURE] The script sends only R on death and no movement actions.',flush=True)
        print('[HUMAN CAPTURE] Saves screenshots ONLY; no demonstration labels, no PPO.',flush=True)
        while time.monotonic()<deadline and saved<a.max_images:
            obs,_=env.reset()
            t0=time.monotonic()
            last_save=-1e10
            tick=0
            while time.monotonic()<deadline and saved<a.max_images:
                obs,reward,done,trunc,info=env.step(0)  # no arrow key is sent
                tick+=1
                now=time.monotonic()
                if (now-t0>=a.warmup_seconds and now-last_save>=a.interval
                        and not done and not trunc):
                    rgb=vision.latest_rgb(obs)
                    path=a.output / f'ep{number:05d}_frame{tick:05d}.png'
                    ok=cv2.imwrite(str(path),cv2.cvtColor(rgb,cv2.COLOR_RGB2BGR))
                    if ok:
                        saved+=1
                        last_save=now
                        if saved%25==0:
                            print(f'[HUMAN CAPTURE] {saved} frames saved across episodes through ep{number}',flush=True)
                if done or trunc:break
            number+=1
    except KeyboardInterrupt:
        print('[HUMAN CAPTURE] Stopped by user; images already saved.',flush=True)
    finally:
        if env is not None:env.close()
    print(f'[HUMAN CAPTURE] Saved {saved} unlabeled frames to {a.output}',flush=True)

if __name__=='__main__':main()
