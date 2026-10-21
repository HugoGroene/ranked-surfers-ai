#!/usr/bin/env python3
"""Preview V5.4 segmentation on a few diverse saved gameplay screenshots.

Examples (run from project root):
  python rs_v5_preview_predictions.py --source ai_data/v5_hazards/raw/v5_capture_04 --sample 12
  python rs_v5_preview_predictions.py --source ai_data/v5_hazards/export_round04_fixed/segment/images/val --sample 12

Never edits input screenshots or annotations. Does not estimate detection recall;
only visually compares model suggestions on selected saved frames.
"""
from __future__ import annotations
import argparse
from collections import Counter
import json
from pathlib import Path
import random
import time

import cv2
import numpy as np

from rs_v5_shadow_segmentation import (check_model_names, detect_v5,
                                       draw_mask_overlay, V5_CLASSES)

EXT={'.jpg','.jpeg','.png','.webp'}


def choose_images(source:Path, n:int, seed:int):
    if source.is_file():
        if source.suffix.lower() not in EXT:
            raise ValueError(f'Unsupported image extension: {source.suffix}')
        return [source]
    if not source.is_dir():
        raise FileNotFoundError(f'Image source missing: {source}')
    paths=sorted(p for p in source.iterdir() if p.is_file() and p.suffix.lower() in EXT)
    if not paths:
        raise ValueError(f'No image files in {source}')
    # Pick one from each ordered group for broad coverage, not 12 neighbors.
    rng=random.Random(seed)
    count=min(n,len(paths))
    return [paths[rng.randrange(i*len(paths)//count, (i+1)*len(paths)//count)]
            for i in range(count)]


def main():
    parser=argparse.ArgumentParser(description=__doc__,formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument('--model',type=Path,default=Path('ai_data/v5_hazards/runs/segment_round04/weights/best.pt'))
    parser.add_argument('--source',type=Path,required=True)
    parser.add_argument('--output',type=Path,default=Path('ai_data/v5_hazards/preview_round04'))
    parser.add_argument('--sample',type=int,default=12)
    parser.add_argument('--seed',type=int,default=42)
    parser.add_argument('--imgsz',type=int,default=960)
    parser.add_argument('--conf',type=float,default=.20)
    parser.add_argument('--device',default='0')
    args=parser.parse_args()
    if not args.model.is_file():parser.error(f'No checkpoint at {args.model}')
    if not 1 <= args.sample <= 200:parser.error('--sample must be between 1 and 200')
    if args.imgsz%32 != 0 or args.imgsz < 160:parser.error('--imgsz must be a multiple of 32')
    if not 0<args.conf<1:parser.error('--conf must be between 0 and 1')
    if args.output.resolve()==args.source.resolve():parser.error('--output must differ from --source')
    photos=choose_images(args.source,args.sample,args.seed)
    args.output.mkdir(parents=True,exist_ok=True)
    from ultralytics import YOLO
    model=YOLO(str(args.model))
    check_model_names(model)
    model.to('cuda:0' if args.device in ('0','cuda','cuda:0') else args.device)
    summary=[]
    totals=Counter()
    for i,source in enumerate(photos):
        bgr=cv2.imread(str(source),cv2.IMREAD_COLOR)
        if bgr is None:raise RuntimeError(f'Could not read {source}')
        rgb=cv2.cvtColor(bgr,cv2.COLOR_BGR2RGB)
        snapshot=detect_v5(model,rgb,time.perf_counter(),0,i,args.imgsz,args.conf)
        annotated=draw_mask_overlay(snapshot)
        filename=f'{i:02d}_{source.stem}'
        src_path=args.output/f'{filename}_original.jpg'
        out_path=args.output/f'{filename}_masks.jpg'
        cv2.imwrite(str(src_path),bgr,[cv2.IMWRITE_JPEG_QUALITY,90])
        cv2.imwrite(str(out_path),annotated,[cv2.IMWRITE_JPEG_QUALITY,90])
        counts=Counter(d['class'] for d in snapshot['detections'])
        totals.update(counts)
        summary.append({'original':str(source),'preview':str(out_path),
                        'objects':dict(counts),
                        'inference_ms':snapshot['inference_ms'],
                        'polygons':snapshot['mask_count']})
        print(f'[{i+1}/{len(photos)}] {source.name}: {dict(counts)}',flush=True)
    report={'model':str(args.model),'source':str(args.source),'imgsz':args.imgsz,
            'conf':args.conf,'class_names':list(V5_CLASSES),
            'total_detections':dict(totals),'images':summary}
    (args.output/'preview_report.json').write_text(json.dumps(report,indent=2),encoding='utf8')
    print(f'[DONE] {len(summary)} image pairs saved to: {args.output}',flush=True)
    print('[NOTE] Predictions are not ground truth. Inspect small/missed coins and barrier types.',flush=True)

if __name__=='__main__':main()
