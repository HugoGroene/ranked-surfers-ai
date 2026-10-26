#!/usr/bin/env python3
"""Export manually labeled LabelMe V5 sessions into BOTH Ultralytics segmentation and detection datasets.

Input folder convention:
  ai_data/v5_hazards/raw/v5_a/frame_00000.jpg
  ai_data/v5_hazards/raw/v5_a/frame_00000.json  # saved by LabelMe

The whole capture session stays entirely in train or val to avoid leakage.
Accepts LabelMe polygon, rectangle, and circle shapes.
"""
from __future__ import annotations
import argparse
from collections import Counter,defaultdict
import itertools
import json
import math
from pathlib import Path
import random
import shutil

import cv2

CLASSES = (
    'coin',              # fatal in no-coin mode; tiny round objects
    'train_front',       # oncoming or parked train bumper/front face
    'train_side',        # visible train side/body region
    'train_roof',        # roof running / landing surface
    'ramp',              # climbable ramp leading to roof
    'barrier_jump',      # visibly identified jump-only obstacle
    'barrier_roll',      # visibly identified roll-only overhead obstacle
    'barrier_either',    # either jump or roll works; confirm in local game
    'tunnel_entrance',   # entrance frame/arch requiring roof drop
)
IMG_EXTS = {'.jpg','.jpeg','.png','.webp'}


def points_for(shape):
    xy = shape.get('points',[])
    kind = shape.get('shape_type','polygon')
    if kind == 'polygon' and len(xy)>=3:
        return [(float(x),float(y)) for x,y in xy]
    if kind == 'rectangle' and len(xy)==2:
        (x1,y1),(x2,y2)=xy
        return [(x1,y1),(x2,y1),(x2,y2),(x1,y2)]
    if kind == 'circle' and len(xy)==2:
        (cx,cy),(ex,ey)=xy
        radius=math.hypot(ex-cx,ey-cy)
        return [(cx+radius*math.cos(2*math.pi*i/24),cy+radius*math.sin(2*math.pi*i/24)) for i in range(24)]
    raise ValueError(f'Unsupported/invalid shape type {kind!r}; use polygon, rectangle, or circle with valid points')


def norm_points(points, width, height):
    result=[]
    for x,y in points:
        nx=max(0.,min(1.,float(x)/width))
        ny=max(0.,min(1.,float(y)/height))
        if not result or (abs(nx-result[-1][0])+abs(ny-result[-1][1]))>1e-7:
            result.append((nx,ny))
    if len(result)>1 and result[0] == result[-1]: result.pop()
    if len(result)<3:
        raise ValueError('Degenerate polygon with fewer than 3 distinct points')
    return result


def load_annotation(image):
    ann=image.with_suffix('.json')
    if not ann.exists():return None
    data=json.loads(ann.read_text(encoding='utf8'))
    # image bytes (imageData) need not be present; inspect actual image for dimensions
    bgr=cv2.imread(str(image), cv2.IMREAD_UNCHANGED)
    if bgr is None:raise ValueError(f'Unreadable image: {image}')
    h,w=bgr.shape[:2]
    if data.get('imageWidth') and int(data['imageWidth'])!=w:
        raise ValueError(f'Annotation width mismatch for {image}: {data["imageWidth"]} != {w}')
    if data.get('imageHeight') and int(data['imageHeight'])!=h:
        raise ValueError(f'Annotation height mismatch for {image}: {data["imageHeight"]} != {h}')
    shapes=[]
    for k,s in enumerate(data.get('shapes', [])):
        label=s.get('label','').strip()
        if label not in CLASSES:
            raise ValueError(f'{ann}: shape {k} label {label!r} unknown. Choose exactly from: {", ".join(CLASSES)}')
        pts=norm_points(points_for(s),w,h)
        xs=[p[0] for p in pts];ys=[p[1] for p in pts]
        if max(xs)-min(xs)<1/w or max(ys)-min(ys)<1/h:
            raise ValueError(f'{ann}: shape {k} extremely small/degenerate')
        shapes.append((label,pts))
    return shapes


def choose_validation(sessions, counter, seed, ratio):
    names=sorted(sessions)
    if len(names)<2:
        raise ValueError('Need at least TWO separately labeled capture sessions; validation is session-held-out')
    totals=Counter()
    for n in names:totals.update(counter[n])
    active={cls for cls in CLASSES if totals[cls]>0}
    rng=random.Random(seed)
    choices=[]
    if len(names)<=12:
        for k in range(1,len(names)):
            for subset in itertools.combinations(names,k):
                choices.append(set(subset))
    else:
        for _ in range(6000):
            k=max(1,min(len(names)-1,round(len(names)*ratio)))
            choices.append(set(rng.sample(names,k)))
    def assess(val):
        train=[n for n in names if n not in val]
        val_counter=Counter();train_counter=Counter()
        for n in val:val_counter.update(counter[n])
        for n in train:train_counter.update(counter[n])
        missing_train=sum(cls in active and train_counter[cls]==0 for cls in active)
        missing_val=sum(cls in active and val_counter[cls]==0 for cls in active)
        images_val=sum(len(sessions[n]) for n in val)
        images_total=sum(len(v) for v in sessions.values())
        ratio_error=abs(images_val / images_total-ratio)
        return (missing_train*10000+missing_val*1000 + 100*ratio_error + .01*len(val),val_counter,train_counter)
    best=min(choices,key=lambda val:assess(val)[0])
    score,vc,tc=assess(best)
    if any(tc[c]==0 for c in active):
        raise ValueError('Cannot create training set covering every labeled class. Record additional sessions.')
    return best,vc,tc


def build(a):
    if not a.raw.is_dir():raise SystemExit(f'Not found: {a.raw}')
    sessions={};class_counts={};unlabeled=Counter()
    for session_dir in sorted(p for p in a.raw.iterdir() if p.is_dir()):
        images=[];count=Counter()
        for im in sorted(p for p in session_dir.iterdir() if p.suffix.lower() in IMG_EXTS and p.name.startswith('frame_')):
            ann=load_annotation(im)
            if ann is None:
                unlabeled[session_dir.name]+=1
                continue
            images.append((im,ann))
            count.update(s[0] for s in ann)
        if images:
            sessions[session_dir.name]=images
            class_counts[session_dir.name]=count
    if not sessions:
        raise SystemExit('No labeled image/JSON pairs. Save LabelMe annotations next to frames.')
    print('[DATA] Labeled frames per session:',{k:len(v) for k,v in sessions.items()},flush=True)
    print('[DATA] Unlabeled frames ignored:',dict(unlabeled),flush=True)
    # Explicit split controls prevent newly labeled held-out captures from accidentally
    # becoming training data as the number of sessions changes.
    if a.val_sessions is None and a.test_sessions is None:
        val,vc,tc=choose_validation(sessions,class_counts,a.seed,a.val_fraction)
        test=set()
    else:
        if not a.val_sessions:
            raise SystemExit('When specifying session roles, pass --val-sessions SESSION [SESSION ...]')
        val=set(a.val_sessions)
        test=set(a.test_sessions or [])
        known=set(sessions)
        unknown=(val|test)-known
        if unknown:
            raise SystemExit(f'Unknown/unlabeled session(s): {sorted(unknown)}. Known: {sorted(known)}')
        if val & test:
            raise SystemExit(f'Session(s) cannot be both validation and test: {sorted(val & test)}')
        train=set(sessions)-val-test
        if not train:
            raise SystemExit('No training sessions remain after val/test exclusions')
        vc=Counter();tc=Counter()
        for sn in val:vc.update(class_counts[sn])
        for sn in train:tc.update(class_counts[sn])
    test_counts=Counter()
    for sn in test:test_counts.update(class_counts[sn])
    print('[SPLIT] Train sessions:',[sn for sn in sessions if sn not in val and sn not in test],flush=True)
    print('[SPLIT] Validation sessions:',sorted(val),flush=True)
    print('[SPLIT] Test sessions (HELD OUT):',sorted(test),flush=True)
    print('[SPLIT] Train object counts:',dict(tc),flush=True)
    print('[SPLIT] Val object counts:',dict(vc),flush=True)
    if test:print('[SPLIT] Test object counts:',dict(test_counts),flush=True)
    missing_from_train=[cls for cls in CLASSES if (vc[cls] or test_counts[cls]) and not tc[cls]]
    if missing_from_train:
        raise SystemExit(f'Cannot train these classes: no training examples for {missing_from_train}. Add/train-label examples first.')
    missing=[c for c in CLASSES if tc[c] and not vc[c]]
    if missing:print('[WARN] These classes are absent from independent validation:',missing,flush=True)
    if a.output.exists() and any(a.output.iterdir()):
        raise SystemExit(f'Output already exists: {a.output}. Choose --output new folder; do not silently overwrite.')
    splits=('train','val','test') if test else ('train','val')
    for mode in ('segment','detect'):
        for split in splits:
            (a.output / mode / 'images' / split).mkdir(parents=True,exist_ok=True)
            (a.output / mode / 'labels' / split).mkdir(parents=True,exist_ok=True)
    counts=Counter();shape_counts=Counter()
    for session,images in sessions.items():
        split='test' if session in test else ('val' if session in val else 'train')
        for src,anns in images:
            stem=f'{session}__{src.stem}'
            seg=[];det=[]
            for cls,pts in anns:
                idx=CLASSES.index(cls)
                xs=[x for x,y in pts];ys=[y for x,y in pts]
                x1,x2=min(xs),max(xs);y1,y2=min(ys),max(ys)
                seg.append(f'{idx} '+ ' '.join(f'{value:.6f}' for xy in pts for value in xy))
                det.append(f'{idx} {(x1+x2)/2:.6f} {(y1+y2)/2:.6f} {x2-x1:.6f} {y2-y1:.6f}')
                shape_counts[cls]+=1
            for mode,lines in (('segment',seg),('detect',det)):
                target=a.output / mode
                shutil.copy2(src, target / 'images' / split / (stem+src.suffix.lower()))
                (target / 'labels' / split / (stem+'.txt')).write_text('\n'.join(lines)+'\n' if lines else '',encoding='utf8')
            counts[split]+=1
    for mode in ('segment','detect'):
        target=(a.output / mode).resolve()
        yaml='path: '+json.dumps(target.as_posix())+'\ntrain: images/train\nval: images/val\n'
        if test:yaml+='test: images/test\n'
        yaml+='names:\n'
        for i,name in enumerate(CLASSES):yaml+=f'  {i}: {name}\n'
        (target / 'data.yaml').write_text(yaml,encoding='utf8')
    meta={'classes':CLASSES,'train_sessions':[s for s in sessions if s not in val and s not in test],
          'val_sessions':sorted(val),'test_sessions':sorted(test),'images_per_split':dict(counts),
          'classes_per_train':dict(tc),'classes_per_val':dict(vc),
          'classes_per_test':dict(test_counts),'unlabeled_frames_ignored':dict(unlabeled)}
    (a.output / 'export_manifest.json').write_text(json.dumps(meta,indent=2),encoding='utf8')
    print('[DONE] Exported',dict(counts),'images,',dict(shape_counts),'labeled objects to',a.output,flush=True)
    print('[NEXT] Train separate YOLO segment/detect models using their corresponding data.yaml files.',flush=True)


def main():
    p=argparse.ArgumentParser(description=__doc__,formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    p.add_argument('--raw',type=Path,default=Path('ai_data/v5_hazards/raw'))
    p.add_argument('--output',type=Path,default=Path('ai_data/v5_hazards/export_01'))
    p.add_argument('--val-fraction',type=float,default=.2)
    p.add_argument('--val-sessions',nargs='+',default=None,metavar='NAME',
                   help='Explicit validation sessions; all other non-test sessions become training')
    p.add_argument('--test-sessions',nargs='+',default=None,metavar='NAME',
                   help='Explicit held-out test sessions (never used for fitting or validation)')
    p.add_argument('--seed',type=int,default=42)
    a=p.parse_args()
    if not 0<a.val_fraction<.5:p.error('--val-fraction must be >0 and <.5')
    build(a)

if __name__=='__main__':main()
