#!/usr/bin/env python3
"""YOLO-seg assisted LabelMe annotation for Ranked Surfers V5.

Commands:
  propose : create a separate LabelMe review batch from unlabeled native frames.
  accept  : import only an explicitly selected and HUMAN-REVIEWED subset of a batch.

A batch can be accepted in pieces using --start and --count (or --images-file).
Unreviewed frames remain in the review folder and NEVER become training data.
Use --dry-run to inspect exact filenames before accepting. No code can verify
human review automatically; --confirm-reviewed is your explicit attestation.
"""
from __future__ import annotations
import argparse
from collections import Counter
import json
import os
from pathlib import Path
import shutil
import sys

import cv2

CLASSES = (
    'coin','train_front','train_side','train_roof','ramp',
    'barrier_jump','barrier_roll','barrier_either','tunnel_entrance',
)
EXTS = {'.jpg','.jpeg','.png','.webp'}
RAW_DEFAULT = Path('ai_data/v5_hazards/raw')
BATCH_DEFAULT = Path('ai_data/v5_hazards/review_batches')


def find_images(folder):
    return sorted(p for p in folder.iterdir() if p.is_file()
                  and p.suffix.lower() in EXTS and p.stem.startswith('frame_'))


def copy_or_link(src, dst):
    try:
        os.link(src, dst)
    except OSError:
        shutil.copy2(src, dst)


def points_from_mask(mask_pts, box, width, height, epsilon):
    import numpy as np
    pts = np.asarray(mask_pts, dtype=np.float32)
    if pts.shape[0] >= 3:
        contour = pts.reshape((-1,1,2))
        perimeter = cv2.arcLength(contour, True)
        if epsilon > 0:
            contour = cv2.approxPolyDP(contour, max(.25, epsilon * perimeter), True)
        pts = contour.reshape((-1,2))
    # Mask can be degenerate for small distant coins. Use a rectangle instead.
    if pts.shape[0] < 3:
        x1,y1,x2,y2 = map(float,box)
        return 'rectangle', [[x1,y1],[x2,y2]]
    pts[:,0] = np.clip(pts[:,0],0,width-1)
    pts[:,1] = np.clip(pts[:,1],0,height-1)
    return 'polygon', [[round(float(x),2),round(float(y),2)] for x,y in pts]


def reserved_by_other_batches(review_root, session):
    """Avoid suggesting the same unaccepted screenshots in multiple review batches."""
    reserved=set()
    if not review_root.exists():
        return reserved
    for manifest in review_root.glob('*/_review_manifest.json'):
        try:
            data=json.loads(manifest.read_text(encoding='utf8'))
            if data.get('session')==session:
                reserved.update(Path(item['image']).name for item in data.get('images', []))
        except (OSError, ValueError, KeyError, TypeError):
            print(f'[WARN] Ignoring unreadable existing manifest: {manifest}', file=sys.stderr)
    return reserved


def select_candidates(candidates, n, selection, seed):
    """Return at most n source frames; never reassign an already labeled frame."""
    if selection=='sequential':
        return candidates[:n]
    import random
    rng=random.Random(seed)
    if len(candidates)<=n:
        return candidates
    if selection=='random':
        # Sampling without replacement, reproducible from --seed.
        return sorted(rng.sample(candidates,n))
    if selection=='spread':
        # One random image from each approximately equal span of the timeline.
        # This avoids a batch of near-identical consecutive screenshots.
        chosen=[]
        for k in range(n):
            lo=(k*len(candidates))//n
            hi=((k+1)*len(candidates))//n
            chosen.append(candidates[rng.randrange(lo,hi)])
        return chosen
    raise ValueError(f'Invalid selection: {selection}')


def propose(args):
    source_dir = args.raw / args.session
    if not source_dir.is_dir():
        raise SystemExit(f'Cannot find source session: {source_dir}')
    out = args.review_root / args.batch
    if out.exists():
        raise SystemExit(f'Review batch already exists: {out}\nChoose a NEW --batch name.')
    reserved=reserved_by_other_batches(args.review_root,args.session)
    candidates = [im for im in find_images(source_dir)
                  if not im.with_suffix('.json').exists() and im.name not in reserved]
    if args.start_index:
        candidates = candidates[args.start_index:]
    candidates = select_candidates(candidates,args.max_images,args.selection,args.seed)
    if not candidates:
        raise SystemExit(f'No unlabeled frames in {source_dir} within requested range.')
    from ultralytics import YOLO
    model = YOLO(str(args.model))
    if getattr(model,'task',None) != 'segment':
        raise SystemExit(f'Expected a YOLO *segmentation* checkpoint; detected task={model.task!r}')
    names = model.names
    unknown = sorted({str(n) for n in names.values() if str(n) not in CLASSES})
    if unknown:
        raise SystemExit(f'Model class names do not match V5 taxonomy: {unknown}.\n'
                         f'Expected names among: {list(CLASSES)}')
    out.mkdir(parents=True,exist_ok=False)
    counts = Counter()
    entries = []
    try:
        for idx,im in enumerate(candidates,1):
            dest_image = out / im.name
            copy_or_link(im, dest_image)
            frame = cv2.imread(str(im))
            if frame is None:
                raise RuntimeError(f'Failed reading {im}')
            h,w = frame.shape[:2]
            result = model.predict(source=str(im),conf=args.conf,iou=args.iou,
                                   imgsz=args.imgsz,device=args.device,
                                   verbose=False)[0]
            shapes=[]
            boxes = result.boxes
            masks = result.masks
            if boxes is not None and len(boxes):
                xyxy = boxes.xyxy.cpu().numpy()
                classes = boxes.cls.cpu().numpy().astype(int)
                confs = boxes.conf.cpu().numpy()
                mask_xy = masks.xy if masks is not None else []
                for j,(box,ci,confidence) in enumerate(zip(xyxy,classes,confs)):
                    label = str(names[int(ci)])
                    mask = mask_xy[j] if j < len(mask_xy) else []
                    typ, pts = points_from_mask(mask,box,w,h,args.epsilon)
                    shapes.append({'label':label,'points':pts,'group_id':None,
                                   'description':f'AUTO proposal confidence={float(confidence):.3f}; review me',
                                   'shape_type':typ,'flags':{},'mask':None})
                    counts[label] += 1
            annotation = {
                'version':'5.0.1', 'flags':{},'shapes':shapes,
                'imagePath':im.name,'imageData':None,
                'imageHeight':h,'imageWidth':w,
                'description':'AUTO-GENERATED PROPOSAL. Review ALL objects and missing objects in LabelMe before importing.'
            }
            (out / (im.stem + '.json')).write_text(json.dumps(annotation,indent=2),encoding='utf8')
            entries.append({'image':im.name,'source':str(im),'proposals':len(shapes)})
            print(f'[{idx}/{len(candidates)}] {im.name}: {len(shapes)} proposed objects',flush=True)
        manifest = {
            'session':args.session,'batch':args.batch,'model':str(args.model),
            'confidence':args.conf,'imgsz':args.imgsz,
            'selection':args.selection,'seed':args.seed,'images':entries,
            'instructions':'Open folder in LabelMe; correct every proposal and missing object; accept only after reviewing every image.'
        }
        (out / '_review_manifest.json').write_text(json.dumps(manifest,indent=2),encoding='utf8')
    except Exception:
        print(f'[ERROR] Incomplete batch at {out}; inspect/remove it before retrying',file=sys.stderr)
        raise
    print(f'\n[DONE] {len(entries)} suggestions in: {out}')
    print(f'[OBJECTS] {dict(counts)}')
    print(f'[SELECTION] {args.selection}, seed={args.seed}; skipped frames already labeled or in review batches')
    print('[NEXT] Open the review folder with LabelMe. Add missing objects, correct class and polygons.')
    print('[SAFETY] Original raw data is unchanged; nothing auto-generated is used for training yet.')


def selected_manifest_items(info,args):
    items=info['images']
    if args.images_file is not None:
        if args.start!=0 or args.count is not None:
            raise SystemExit('--images-file cannot be combined with --start or --count')
        names=[Path(line.strip()).name for line in args.images_file.read_text(encoding='utf8').splitlines()
               if line.strip() and not line.lstrip().startswith('#')]
        if not names or len(names)!=len(set(names)):
            raise SystemExit('--images-file must contain unique, nonempty image names')
        by_name={i['image']:i for i in items}
        unknown=[name for name in names if name not in by_name]
        if unknown:
            raise SystemExit(f'Unknown image names in --images-file: {unknown}')
        return [by_name[name] for name in names]
    if args.start<0 or (args.count is not None and args.count<1):
        raise SystemExit('--start must be >=0; --count must be >=1')
    return items[args.start:args.start+args.count if args.count is not None else None]


def accept(args):
    batch = args.review_root / args.batch
    manifest_file = batch / '_review_manifest.json'
    if not manifest_file.exists():
        raise SystemExit(f'Review manifest not found: {manifest_file}')
    info = json.loads(manifest_file.read_text(encoding='utf8'))
    source_dir = args.raw / info['session']
    if not source_dir.is_dir():
        raise SystemExit(f'Source session not found: {source_dir}')
    chosen=selected_manifest_items(info,args)
    if not chosen:
        raise SystemExit('No frames selected. Check --start and --count or --images-file.')
    print(f'[SELECTED] {len(chosen)} of {len(info["images"])} frames from {args.batch}:',flush=True)
    for item in chosen:
        print(f'  {item["image"]}',flush=True)
    if not args.confirm_reviewed and not args.dry_run:
        raise SystemExit('Refusing to import unchecked model guesses. First use --dry-run '
                         'to check these filenames; then add --confirm-reviewed ONLY '
                         'if every selected frame has been manually verified.')
    prepared=[]
    for item in chosen:
        image = item['image']
        review_json = batch / Path(image).with_suffix('.json').name
        target_image = source_dir / image
        target_json = target_image.with_suffix('.json')
        if not target_image.exists():
            raise SystemExit(f'Missing original image: {target_image}')
        if not review_json.exists():
            raise SystemExit(f'Missing proposed/reviewed JSON: {review_json}')
        if target_json.exists() and not args.overwrite:
            raise SystemExit(f'Original annotation already exists: {target_json}. '
                             'Refusing overwrite; choose a different batch or --overwrite explicitly.')
        obj = json.loads(review_json.read_text(encoding='utf8'))
        frame = cv2.imread(str(target_image))
        if frame is None:raise SystemExit(f'Cannot read {target_image}')
        h,w=frame.shape[:2]
        if int(obj.get('imageWidth',-1))!=w or int(obj.get('imageHeight',-1))!=h:
            raise SystemExit(f'Image dimensions mismatch: {review_json}')
        for shape in obj.get('shapes',[]):
            if shape.get('label') not in CLASSES:
                raise SystemExit(f'Bad label {shape.get("label")!r}: {review_json}')
        # Strip AUTO proposal explanatory confidence text, but preserve corrected shapes.
        for shape in obj.get('shapes',[]):
            desc=shape.get('description') or ''  # LabelMe may save null for edited shapes
            if desc.startswith('AUTO proposal confidence='):
                shape['description']=''
        obj['description']='Reviewed LabelMe annotation imported from V5 proposal batch.'
        obj['imagePath']=image
        obj['imageData']=None
        prepared.append((target_json,obj))
    if args.dry_run:
        print(f'[DRY RUN] {len(prepared)} valid selected annotations; no files written.')
        print('[NEXT] Check the filenames above; rerun without --dry-run and with --confirm-reviewed.')
        return
    for target,obj in prepared:
        target.write_text(json.dumps(obj,indent=2),encoding='utf8')
    print(f'[IMPORTED] {len(prepared)} explicitly confirmed annotations into {source_dir}')
    print(f'[LEFT UNIMPORTED] {len(info["images"])-len(chosen)} original suggestions still only in the review batch')
    print('[NEXT] Preview corrected annotations, then make a NEW YOLO export and retrain.')


def main():
    p=argparse.ArgumentParser(description=__doc__,formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    subs=p.add_subparsers(dest='command',required=True)
    a=subs.add_parser('propose',help='Generate YOLO segmentation polygons in isolated LabelMe review batch')
    a.add_argument('--session',required=True,help='Original V5 capture session name')
    a.add_argument('--batch',required=True,help='NEW unique review batch name')
    a.add_argument('--model',required=True,type=Path,help='Trained V5 YOLO segmentation best.pt')
    a.add_argument('--raw',default=RAW_DEFAULT,type=Path)
    a.add_argument('--review-root',default=BATCH_DEFAULT,type=Path)
    a.add_argument('--max-images',type=int,default=40)
    a.add_argument('--start-index',type=int,default=0)
    a.add_argument('--selection',choices=('sequential','random','spread'),default='sequential',
                   help='spread: one random frame from each time span; random: uniform random frames')
    a.add_argument('--seed',type=int,default=42,help='Seed for random/spread selection')
    a.add_argument('--conf',type=float,default=.20)
    a.add_argument('--iou',type=float,default=.5)
    a.add_argument('--imgsz',type=int,default=960)
    a.add_argument('--device',default='0',help='0=CUDA GPU, cpu=CPU')
    a.add_argument('--epsilon',type=float,default=.002,help='Contour simplification as fraction of perimeter')
    b=subs.add_parser('accept',help='Explicitly import fully hand-verified suggestions')
    b.add_argument('--batch',required=True)
    b.add_argument('--raw',default=RAW_DEFAULT,type=Path)
    b.add_argument('--review-root',default=BATCH_DEFAULT,type=Path)
    b.add_argument('--start',type=int,default=0,help='Zero-based manifest position of first frame to accept')
    b.add_argument('--count',type=int,help='Number of consecutive manifest frames to accept')
    b.add_argument('--images-file',type=Path,help='Text file of exact image basenames to accept instead')
    b.add_argument('--dry-run',action='store_true',help='Validate and show selected frames without writing')
    b.add_argument('--confirm-reviewed',action='store_true')
    b.add_argument('--overwrite',action='store_true')
    args=p.parse_args()
    if args.command=='propose':
        if args.max_images<1 or args.start_index<0 or not 0<args.conf<1 or args.imgsz<64:
            p.error('Invalid propose parameter')
        propose(args)
    else:accept(args)

if __name__=='__main__':main()
