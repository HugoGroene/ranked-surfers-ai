#!/usr/bin/env python3
"""Ranked Surfers V2: box labels -> YOLO detector -> held-out visual audit.

Commands (from ~/Documents/rankedsurflocal):
    python rs_v2_object_detector.py prepare
    python rs_v2_object_detector.py train --epochs 40
    python rs_v2_object_detector.py predict --split val --limit 16

Uses existing ai_data/v2_hazard_images/labels.json, no relabeling needed.
IMPORTANT: an accurate detector needs all visible objects annotated in each included
frame. 'empty' boxes are ignored; they are not a YOLO detection class. This is a
local-Practice research diagnostic, not a game-playing policy.
"""
import argparse
import itertools
import json
import math
import random
import shutil
from collections import Counter, defaultdict
from pathlib import Path

import cv2
import numpy as np

CLASSES = ('coin', 'train', 'barrier')
CLSID = {label: i for i, label in enumerate(CLASSES)}
COLOR = {'coin': (40, 240, 240), 'train': (240, 155, 65), 'barrier': (70, 90, 245)}


def episode(entry):
    value = entry.get('episode')
    if value is not None:
        return str(value)
    name = entry['image']
    if name.startswith('ep') and '_frame' in name:
        return name[2:].split('_frame', 1)[0]
    return name


def load_source(root):
    p = root / 'labels.json'
    if not p.exists():
        raise SystemExit(f'Missing {p}. Label screenshots with rs_v2_hazard_probe_curated.py first.')
    data = json.loads(p.read_text())
    if not isinstance(data, list):
        raise SystemExit('labels.json should be a list of bounding-box objects.')
    reviewed = root / 'reviewed.json'
    reviewed_set = set(json.loads(reviewed.read_text())) if reviewed.exists() else set()
    excluded_file = root / 'excluded.json'
    exclude_data = json.loads(excluded_file.read_text()) if excluded_file.exists() else {}
    excluded = set(exclude_data if isinstance(exclude_data, (dict, list)) else [])
    groups = defaultdict(list)
    for record in data:
        name = Path(str(record.get('image', ''))).name
        if not name or name in excluded:
            continue
        if record.get('class') not in (*CLASSES, 'empty'):
            continue
        groups[name].append(record)
    included = {}
    missing = []
    unreviewed = []
    for name, boxes in sorted(groups.items()):
        path = root / name
        if not path.is_file():
            missing.append(name)
            continue
        if reviewed_set and name not in reviewed_set:
            unreviewed.append(name)
            continue
        img = cv2.imread(str(path))
        if img is None:
            missing.append(name)
            continue
        h, w = img.shape[:2]
        objects = []
        for record in boxes:
            if record['class'] not in CLSID:
                continue
            try:
                x0, y0, x1, y1 = map(float, record['box'])
                x0, x1 = sorted((max(0., min(w, x0)), max(0., min(w, x1))))
                y0, y1 = sorted((max(0., min(h, y0)), max(0., min(h, y1))))
            except (ValueError, TypeError, KeyError):
                continue
            if x1 - x0 < 2 or y1 - y0 < 2:
                continue
            objects.append((record['class'], (x0, y0, x1, y1)))
        # Background-only labeled frames are allowed, as long as reviewed.
        included[name] = {'source': path, 'episode': episode(boxes[0]),
                          'objects': objects, 'image_shape': (h, w)}
    return included, missing, unreviewed


def choose_split(data, fraction, seed, min_val_barriers):
    eps = sorted({v['episode'] for v in data.values()})
    if len(eps) < 4:
        raise SystemExit(f'Need at least 4 independent episodes; found {len(eps)}.')
    totals = Counter()
    by_ep = defaultdict(Counter)
    for ent in data.values():
        for name, _ in ent['objects']:
            totals[name] += 1
            by_ep[ent['episode']][name] += 1
    required = [name for name in CLASSES if totals[name] > 0]
    if any(totals[c] < 6 for c in CLASSES):
        print('[WARNING] Low count for a detection class:', dict(totals))
    k = max(1, min(len(eps)-1, round(len(eps) * fraction)))
    rng = random.Random(seed)
    valid = []
    combos = list(itertools.combinations(eps, k)) if math.comb(len(eps), k) <= 50000 else []
    if not combos:
        combos = [tuple(sorted(rng.sample(eps, k))) for _ in range(10000)]
    for heldout in combos:
        hset = set(heldout)
        va = Counter()
        for ep in heldout:
            va.update(by_ep[ep])
        tr = totals - va
        if not all(va[c] >= (min_val_barriers if c == 'barrier' else 2) and tr[c] >= 3 for c in required):
            continue
        # Prefer a validation set near the desired fraction of labeled objects.
        val_n = sum(va.values())
        share = val_n / max(1, sum(totals.values()))
        skew = sum(abs(va[c] / totals[c] - fraction) for c in required) / max(1, len(required))
        priority = abs(share-fraction) + .3*skew
        valid.append((priority, rng.random(), hset, tr, va))
    if not valid:
        desc = {str(ep): dict(by_ep[ep]) for ep in eps}
        raise SystemExit('Cannot split independent episodes with every class on both sides.\n'
                         'Object counts per episode:\n'+json.dumps(desc, indent=2)+'\n'
                         'Collect and label more barriers in independent episodes.')
    _, _, val_eps, tr_counts, val_counts = min(valid, key=lambda x: (x[0], x[1]))
    return val_eps, tr_counts, val_counts


def draw_gt_preview(output, data, rows, seed):
    rng = random.Random(seed)
    selected = sorted(data)
    rng.shuffle(selected)
    selected = selected[:min(rows, len(selected))]
    tiles = []
    for name in selected:
        ent = data[name]
        img = cv2.imread(str(ent['source']))
        if img is None:
            continue
        h, w = img.shape[:2]
        # Render at 2x for readability.
        canvas = cv2.resize(img, (w*2, h*2), interpolation=cv2.INTER_NEAREST)
        for typ, (x0, y0, x1, y1) in ent['objects']:
            c = COLOR[typ]
            cv2.rectangle(canvas, (round(x0*2), round(y0*2)), (round(x1*2), round(y1*2)), c, 2)
            cv2.putText(canvas, typ, (round(x0*2), max(14, round(y0*2)-3)),
                        cv2.FONT_HERSHEY_SIMPLEX, .45, c, 1, cv2.LINE_AA)
        caption = np.zeros((24, canvas.shape[1], 3), np.uint8)
        cv2.putText(caption, f'{name} ep={ent["episode"]}', (4, 17),
                    cv2.FONT_HERSHEY_SIMPLEX, .43, (255,255,255), 1, cv2.LINE_AA)
        tiles.append(np.concatenate([caption,canvas], axis=0))
    if tiles:
        max_h = max(tile.shape[0] for tile in tiles)
        max_w = max(tile.shape[1] for tile in tiles)
        columns = min(3,len(tiles))
        rows = math.ceil(len(tiles)/columns)
        sheet = np.zeros((rows*max_h, columns*max_w,3),np.uint8)
        for i,tile in enumerate(tiles):
            yy=(i//columns)*max_h;xx=(i%columns)*max_w
            sheet[yy:yy+tile.shape[0],xx:xx+tile.shape[1]]=tile
        cv2.imwrite(str(output),sheet)


def prepare(args):
    root = args.images
    data, missing, unreviewed = load_source(root)
    if not data:
        raise SystemExit('No usable labeled screenshots found in '+str(root))
    val_eps, train_counts, val_counts = choose_split(data, args.val_fraction, args.seed, args.min_val_barriers)
    dest = args.output.resolve()
    if dest.exists() and any(dest.iterdir()) and not args.overwrite:
        raise SystemExit(f'{dest} exists and is nonempty. Use --overwrite to replace the PREPARED dataset only.')
    if dest.exists() and args.overwrite:
        shutil.rmtree(dest)
    for split in ('train','val'):
        (dest / 'images' / split).mkdir(parents=True, exist_ok=True)
        (dest / 'labels' / split).mkdir(parents=True, exist_ok=True)
    counts = Counter()
    examples = {'train': [], 'val': []}
    for filename, ent in sorted(data.items()):
        split = 'val' if ent['episode'] in val_eps else 'train'
        output_image = dest / 'images' / split / filename
        shutil.copy2(ent['source'], output_image)
        h,w = ent['image_shape']
        lines=[]
        for typ, (x0,y0,x1,y1) in ent['objects']:
            # YOLO cx, cy, width, height / image dimension
            cx=((x0+x1)/2)/w; cy=((y0+y1)/2)/h
            bw=(x1-x0)/w; bh=(y1-y0)/h
            lines.append(f'{CLSID[typ]} {cx:.8f} {cy:.8f} {bw:.8f} {bh:.8f}')
            counts[f'{split}/{typ}'] += 1
        label_path = dest / 'labels' / split / (Path(filename).stem+'.txt')
        label_path.write_text('\n'.join(lines)+('\n' if lines else ''))
        examples[split].append(filename)
    yaml = f'''path: {dest.as_posix()}\ntrain: images/train\nval: images/val\nnames:\n  0: coin\n  1: train\n  2: barrier\n'''
    (dest / 'data.yaml').write_text(yaml)
    report={'images':len(data),'train_images':len(examples['train']),
            'val_images':len(examples['val']),'train_episodes':sorted(set(ent['episode'] for ent in data.values())-val_eps),
            'val_episodes':sorted(val_eps),'train_boxes':dict(train_counts),'val_boxes':dict(val_counts),
            'unreviewed_ignored':len(unreviewed),'missing_ignored':len(missing),
            'classes':list(CLASSES),'empty_boxes_ignored':True,
            'warning':'Train YOLO only if ALL coins/trains/barriers in each used image are annotated. '
                      'Unlabeled visible objects are incorrectly treated as background.'}
    (dest/'split_report.json').write_text(json.dumps(report,indent=2))
    draw_gt_preview(dest/'ground_truth_preview.jpg',data,args.preview_count,args.seed)
    print('[DATASET] YOLO dataset:', dest)
    print('[DATASET] Images:', report['train_images'],'train /',report['val_images'],'validation')
    print('[DATASET] Episodes train:',report['train_episodes'],'holdout:',report['val_episodes'])
    print('[DATASET] Boxes train:',dict(train_counts),'val:',dict(val_counts))
    print('[DATASET] Excluded:',len(unreviewed),'unreviewed,',len(missing),'missing images')
    print('[DATASET] GT preview:',dest/'ground_truth_preview.jpg')
    print('[WARNING] EMPTY track is not a detected object; it is background.')
    print('[IMPORTANT] Check the ground_truth_preview.jpg for missing annotations BEFORE training.')


def require_yolo():
    try:
        from ultralytics import YOLO
    except ImportError as exc:
        raise SystemExit('Install Ultralytics in the activated venv:\n'
                         '  python -m pip install "ultralytics>=8.3,<9"\n'
                         'Then repeat this command.\n') from exc
    return YOLO


def train(args):
    data_yaml=args.output.resolve()/'data.yaml'
    if not data_yaml.exists():
        raise SystemExit(f'{data_yaml} missing; run `prepare` first.')
    YOLO=require_yolo()
    import torch
    device=args.device if args.device is not None else ('0' if torch.cuda.is_available() else 'cpu')
    print('[TRAIN] Loading pretrained detector; weights download on first run.',flush=True)
    model=YOLO(args.weights)
    print('[TRAIN] device=',device,' epochs=',args.epochs,' imgsz=',args.imgsz,' batch=',args.batch,flush=True)
    model.train(data=str(data_yaml),epochs=args.epochs,imgsz=args.imgsz,batch=args.batch,
                device=device,workers=0,project=str(args.output.resolve()/'runs'),name='hazards',
                exist_ok=True,seed=args.seed,patience=12,mosaic=0.0,fliplr=0.0,
                verbose=True,plots=True)
    best=args.output.resolve()/'runs'/'hazards'/'weights'/'best.pt'
    print('[TRAIN] Best checkpoint:',best)
    print('[TRAIN] Inspect:',args.output.resolve()/'runs'/'hazards'/'results.png')
    if best.is_file():
        check=YOLO(str(best))
        print('[EVAL] Independent held-out episodes:')
        check.val(data=str(data_yaml),imgsz=args.imgsz,batch=args.batch,device=device,workers=0,plots=True)


def predict(args):
    YOLO=require_yolo()
    best=args.weights if args.weights!='yolo11n.pt' else str(args.output.resolve()/'runs'/'hazards'/'weights'/'best.pt')
    if not Path(best).exists():
        raise SystemExit(f'Detector checkpoint missing: {best}. Run `train` first.')
    imgdir=args.source if args.source is not None else args.output.resolve()/'images'/args.split
    files=sorted(imgdir.glob('*.png')) if imgdir.is_dir() else ([imgdir] if imgdir.is_file() else [])
    if not files:
        raise SystemExit(f'No images in {imgdir}')
    rng=random.Random(args.seed)
    rng.shuffle(files)
    files=files[:args.limit]
    model=YOLO(best)
    dest=args.output.resolve()/'predictions'/('external' if args.source is not None else args.split)
    dest.mkdir(parents=True,exist_ok=True)
    object_counts=Counter()
    for path in files:
        results=model.predict(source=str(path),imgsz=args.imgsz,conf=args.conf,verbose=False)
        result=results[0]
        if result.boxes is not None:
            for cls in result.boxes.cls.cpu().numpy():
                object_counts[CLASSES[int(cls)]]+=1
        annotated=result.plot()
        cv2.imwrite(str(dest/path.name),annotated)
    print('[PREDICT] Saved',len(files),'annotated',args.split,'images to:',dest)
    print('[PREDICT] Counts:',dict(object_counts))
    print('[PREDICT] Open them and CHECK missing coins, fake coins, barriers, and distant hazards.')
    print('[WARNING] Visual validation is required; mAP from a 49-frame dataset can be misleading.')


def main():
    parser=argparse.ArgumentParser(description=__doc__,formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument('command',choices=('prepare','train','predict'))
    parser.add_argument('--images',type=Path,default=Path('ai_data/v2_hazard_images'))
    parser.add_argument('--output',type=Path,default=Path('ai_data/v2_yolo_dataset'))
    parser.add_argument('--seed',type=int,default=124)
    parser.add_argument('--val-fraction',type=float,default=.25)
    parser.add_argument('--min-val-barriers',type=int,default=2)
    parser.add_argument('--overwrite',action='store_true',help='Replace only prepared YOLO dataset output')
    parser.add_argument('--preview-count',type=int,default=15)
    parser.add_argument('--epochs',type=int,default=40)
    parser.add_argument('--batch',type=int,default=4)
    parser.add_argument('--imgsz',type=int,default=640)
    parser.add_argument('--weights',type=str,default='yolo11n.pt',help='Pretrained model for train, optional checkpoint for predict')
    parser.add_argument('--device',type=str,default=None)
    parser.add_argument('--split',choices=('train','val'),default='val')
    parser.add_argument('--limit',type=int,default=16)
    parser.add_argument('--source',type=Path,default=None,help='For predict: optional directory of genuinely NEW gameplay images')
    parser.add_argument('--conf',type=float,default=.25)
    args=parser.parse_args()
    if not (0 < args.val_fraction < .5):
        parser.error('--val-fraction must be in (0,0.5)')
    if args.command=='prepare':prepare(args)
    elif args.command=='train':train(args)
    else:predict(args)


if __name__=='__main__':
    main()
