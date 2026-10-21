#!/usr/bin/env python3
"""Measure approximate V5 bbox recall by vertical image region on labeled VAL images.

This is NOT physical distance, avoidance quality, or mask IoU. Uses same-class
one-to-one box IoU matching against polygon-derived ground truth boxes. Useful
for checking whether a missing distant coin becomes detectable closer to camera.

Default --split val. Avoid inspecting held-out test while selecting checkpoints.
"""
from __future__ import annotations
import argparse
from collections import Counter, defaultdict
import json
from pathlib import Path
import time
import cv2

from rs_v5_shadow_segmentation import check_model_names, detect_v5, distance_band, V5_CLASSES


def iou_box(a,b):
    x1=max(a[0],b[0]);y1=max(a[1],b[1]);x2=min(a[2],b[2]);y2=min(a[3],b[3])
    intersection=max(0,x2-x1)*max(0,y2-y1)
    aa=max(0,a[2]-a[0])*max(0,a[3]-a[1])
    bb=max(0,b[2]-b[0])*max(0,b[3]-b[1])
    return intersection/(aa+bb-intersection) if aa+bb-intersection>0 else 0.0


def parse_ground_truth(label):
    objects=[]
    for row in label.read_text(encoding='utf8').splitlines():
        if not row.strip():continue
        fields=row.split()
        idx=int(fields[0])
        pts=[float(v) for v in fields[1:]]
        if idx < 0 or idx >= len(V5_CLASSES) or len(pts)<6 or len(pts)%2:
            raise ValueError(f'Invalid polygon label in {label}: {row[:150]}')
        xs=pts[::2];ys=pts[1::2]
        box=[min(xs),min(ys),max(xs),max(ys)]
        objects.append({'class':V5_CLASSES[idx],'box':box,'image_region':distance_band(box[3])})
    return objects


def match_counts(ground_truth, predictions, threshold=.50):
    """Match each same-class GT box at most once to highest-confidence predictions."""
    matched=set();hits=[]
    for pred in sorted(predictions,key=lambda d:d['conf'],reverse=True):
        best=None;score=threshold
        for i,gt in enumerate(ground_truth):
            if i in matched or gt['class']!=pred['class']:continue
            value=iou_box(gt['box'],pred['box'])
            if value >= score:best=i;score=value
        if best is not None:
            matched.add(best)
    for i,gt in enumerate(ground_truth):
        hits.append((gt['class'],gt['image_region'],i in matched))
    return hits


def main():
    p=argparse.ArgumentParser(description=__doc__,formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument('--dataset',type=Path,default=Path('ai_data/v5_hazards/export_round04_fixed/segment'))
    p.add_argument('--model',type=Path,default=Path('ai_data/v5_hazards/runs/segment_round04/weights/best.pt'))
    p.add_argument('--split',choices=['val','test'],default='val')
    p.add_argument('--iou',type=float,default=.50)
    p.add_argument('--imgsz',type=int,default=960)
    p.add_argument('--conf',type=float,default=.20)
    p.add_argument('--device',default='0')
    p.add_argument('--report',type=Path,default=Path('ai_data/v5_hazards/v54_val_by_region.json'))
    args=p.parse_args()
    if not args.model.is_file():p.error(f'Model not found: {args.model}')
    if not 0<args.iou<=1:p.error('IoU threshold must be (0,1]')
    images=sorted(p for p in (args.dataset/'images'/args.split).iterdir()
                  if p.suffix.lower() in {'.jpg','.jpeg','.png','.webp'})
    if not images:p.error('No labeled dataset images found')
    from ultralytics import YOLO
    model=YOLO(str(args.model))
    check_model_names(model)
    model.to('cuda:0' if args.device in ('0','cuda','cuda:0') else args.device)
    counts=defaultdict(lambda:Counter())
    totals=Counter()
    for n,image_path in enumerate(images,1):
        label_path=args.dataset/'labels'/args.split/(image_path.stem+'.txt')
        if not label_path.exists():raise FileNotFoundError(f'Missing label {label_path}')
        expected=parse_ground_truth(label_path)
        bgr=cv2.imread(str(image_path))
        if bgr is None:raise ValueError(f'Cannot read {image_path}')
        rgb=cv2.cvtColor(bgr,cv2.COLOR_BGR2RGB)
        predicted=detect_v5(model,rgb,time.perf_counter(),0,n,args.imgsz,args.conf)
        for name,region,hit in match_counts(expected,predicted['detections'],args.iou):
            for key in (f'{region}/{name}',f'ALL/{name}',f'{region}/ALL','ALL/ALL'):
                counts[key]['total']+=1
                counts[key]['matched']+=int(hit)
            totals[name]+=1
        print(f'[{n}/{len(images)}] {image_path.name}  GT={len(expected)}  predicted={len(predicted["detections"])}',flush=True)
    results={}
    for key,c in sorted(counts.items()):
        results[key]={'n':c['total'],'matched':c['matched'],
                      'recall_at_iou':round(c['matched']/c['total'],3)}
    report={'model':str(args.model),'dataset':str(args.dataset),'split':args.split,
            'imgsz':args.imgsz,'conf':args.conf,'iou':args.iou,
            'warning':'Vertical image regions are not calibrated 3D distance. '
                      'Box IoU of polygon bounds, not segmentation mAP; small-class samples are noisy.',
            'images':len(images),'results':results}
    args.report.parent.mkdir(parents=True,exist_ok=True)
    args.report.write_text(json.dumps(report,indent=2),encoding='utf8')
    for key in ['upper_frame/coin','middle_frame/coin','lower_frame/coin',
                'lower_frame/train_front','lower_frame/barrier_jump',
                'lower_frame/barrier_roll','ALL/ALL']:
        v=results.get(key)
        if v:print(f'{key:32s} {v["matched"]:3d}/{v["n"]:3d} recall={v["recall_at_iou"]:.3f}',flush=True)
    print(f'[DONE] Saved {args.report}',flush=True)

if __name__=='__main__':main()
