#!/usr/bin/env python3
"""Shortlist and label varied Ranked Surfers V2 hazard frames.

Non-destructive: labels.json and reviewed.json are compatible with
rs_v2_hazard_probe.py; excluded.json records ignored frames without deleting them.

Examples:
 python rs_v2_hazard_probe_curated.py label --max-images 80
 python rs_v2_hazard_probe_curated.py label --reshuffle --max-images 80
 python rs_v2_hazard_probe_curated.py stats
 python rs_v2_hazard_probe_curated.py train --epochs 30
"""
import argparse
import json
import re
from collections import Counter, defaultdict
from pathlib import Path

import cv2
import numpy as np

import rs_v2_hazard_probe as base


def read_json(path, fallback):
    if not path.exists():
        return fallback
    return json.loads(path.read_text())


def write_json(path, obj):
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix('.tmp')
    tmp.write_text(json.dumps(obj, indent=2))
    tmp.replace(path)


def perceptual_hash(bgr):
    """Compact whole-frame pHash for SHORTLISTING ONLY, never auto-deletes data."""
    gray = cv2.cvtColor(bgr, cv2.COLOR_BGR2GRAY)
    small = cv2.resize(gray, (32, 32), interpolation=cv2.INTER_AREA).astype(np.float32)
    dct = cv2.dct(small)[:8, :8]
    values = dct.ravel()[1:]
    return (dct > np.median(values)).ravel()


def distinct(a, b, minimum):
    return int(np.count_nonzero(a != b)) >= minimum


def make_shortlist(images, max_images, per_episode, min_frame, hash_distance, seed):
    """Interleave independent episodes to avoid 80 copies of the same opening."""
    rng = np.random.default_rng(seed)
    episodes = defaultdict(list)
    for image in images:
        m = re.search(r'_frame(\d+)', image.stem)
        if m and int(m.group(1)) < min_frame:
            continue
        episodes[base.episode(image.name)].append(image)
    # Iterate episodes in a reproducible random order; within each episode
    # prioritize dispersed moments over consecutive screenshots.
    order = list(episodes)
    rng.shuffle(order)
    spread = {}
    for ep in order:
        files = sorted(episodes[ep])
        # Select at most per_episode * 4 candidates, roughly evenly spaced.
        if len(files) > per_episode * 4:
            ix = np.linspace(0, len(files)-1, per_episode*4).round().astype(int)
            files = [files[i] for i in sorted(set(ix))]
        rng.shuffle(files)
        spread[ep] = files
    chosen = []
    hashes = []
    per_counts = Counter()
    while len(chosen) < max_images:
        moved = False
        for ep in order:
            if per_counts[ep] >= per_episode:
                continue
            while spread[ep]:
                path = spread[ep].pop()
                bgr = cv2.imread(str(path))
                if bgr is None:
                    continue
                phash = perceptual_hash(bgr)
                if hash_distance and any(not distinct(phash, old, hash_distance) for old in hashes):
                    continue
                hashes.append(phash)
                chosen.append(path.name)
                per_counts[ep] += 1
                moved = True
                break
            if len(chosen) >= max_images:
                break
        if not moved:
            break
    return chosen


def label(args):
    root = args.images
    root.mkdir(parents=True, exist_ok=True)
    images = sorted(root.glob('ep*_frame*.png'))
    if not images:
        raise SystemExit(f'No frames found in {root}. Capture some gameplay first.')
    labels_path = root/'labels.json'
    reviewed_path = root/'reviewed.json'
    excluded_path = root/'excluded.json'
    selected_path = root/'curated_selection.json'
    labels = base.load_labels(labels_path)
    reviewed = set(read_json(reviewed_path, []))
    excluded = read_json(excluded_path, {})
    if isinstance(excluded, list):
        excluded = {name:'excluded' for name in excluded}
    if args.all_images:
        selection = [p.name for p in images]
    elif selected_path.exists() and not args.reshuffle:
        selection = read_json(selected_path, [])
    else:
        selection = make_shortlist(images, args.max_images, args.per_episode,
                                   args.min_frame, args.hash_distance, args.seed)
        write_json(selected_path, selection)
    by_name = {p.name:p for p in images}
    pending = [by_name[n] for n in selection
               if n in by_name and n not in excluded
               and (args.review_all or n not in reviewed)]
    print(f'[CURATE] Total frames={len(images)}, shortlist={len(selection)}, '
          f'pending={len(pending)}, reviewed={len(reviewed)}, excluded={len(excluded)}', flush=True)
    print('[KEYS] C/T/B/E choose box class; drag mouse to label. '
          'N=reviewed/next, S=skip for later, X=startup/irrelevant, '
          'D=duplicate, U=undo, Q=quit.', flush=True)
    if not pending:
        print('No pending frames. Use --reshuffle for a new shortlist or '
              '--review-all to open the current shortlist again.')
        return
    window = 'Ranked Surfers V2 - curated hazard labeling'
    cv2.namedWindow(window,cv2.WINDOW_NORMAL)
    cv2.resizeWindow(window, 1050, 700)
    chosen_class = 'coin'
    scale = 4
    try:
        for i,path in enumerate(pending, 1):
            bgr = cv2.imread(str(path))
            if bgr is None:
                continue
            h,w = bgr.shape[:2]
            frame_entries = [e for e in labels if e.get('image')==path.name]
            state = {'start':None,'preview':None,'to_add':None}
            def on_mouse(event,xx,yy,flags,param):
                if yy<64:
                    return
                x = int(np.clip(xx/scale,0,w-1))
                y = int(np.clip((yy-64)/scale,0,h-1))
                if event == cv2.EVENT_LBUTTONDOWN:
                    state['start']=(x,y)
                elif event == cv2.EVENT_MOUSEMOVE and state['start'] is not None:
                    state['preview']=(*state['start'],x,y)
                elif event == cv2.EVENT_LBUTTONUP and state['start'] is not None:
                    x0,y0=state['start']
                    xx0,xx1=sorted((x0,x))
                    yy0,yy1=sorted((y0,y))
                    state['start']=None
                    state['preview']=None
                    if xx1-xx0>=3 and yy1-yy0>=3:
                        state['to_add']={'image':path.name,'episode':base.episode(path.name),
                                          'class':chosen_class,'box':[xx0,yy0,xx1,yy1]}
            cv2.setMouseCallback(window,on_mouse)
            while True:
                if state['to_add'] is not None:
                    ent=state['to_add']
                    labels.append(ent)
                    frame_entries.append(ent)
                    state['to_add']=None
                    base.save_labels(labels_path,labels)
                canvas = cv2.resize(bgr, (w*scale,h*scale), interpolation=cv2.INTER_NEAREST)
                for e in frame_entries:
                    x0,y0,x1,y1=e['box']
                    col=base.COLORS.get(e['class'],(255,255,255))
                    cv2.rectangle(canvas,(x0*scale,y0*scale),(x1*scale,y1*scale),col,2)
                    cv2.putText(canvas,e['class'],(x0*scale,max(16,y0*scale-3)),
                                cv2.FONT_HERSHEY_SIMPLEX,.5,col,1)
                if state['preview'] is not None:
                    x0,y0,x1,y1=state['preview']
                    cv2.rectangle(canvas,(x0*scale,y0*scale),(x1*scale,y1*scale),
                                  base.COLORS[chosen_class],1)
                top = np.zeros((64,canvas.shape[1],3),dtype=np.uint8)
                cv2.putText(top,f'{i}/{len(pending)}  {path.name}  Class:{chosen_class.upper()}  Boxes:{len(frame_entries)}',
                            (6,22),cv2.FONT_HERSHEY_SIMPLEX,.49,(255,255,255),1,cv2.LINE_AA)
                cv2.putText(top,'C/T/B/E class | drag box | N next | S later | X startup | D duplicate | U undo | Q quit',
                            (6,48),cv2.FONT_HERSHEY_SIMPLEX,.42,(200,230,200),1,cv2.LINE_AA)
                cv2.imshow(window,np.vstack([top,canvas]))
                k=cv2.waitKey(40)&255
                if k in base.KEYMAP:
                    chosen_class=base.KEYMAP[k]
                elif k == ord('u') and frame_entries:
                    entry=frame_entries.pop()
                    for j in range(len(labels)-1,-1,-1):
                        if labels[j] is entry:
                            labels.pop(j)
                            break
                    base.save_labels(labels_path,labels)
                elif k in (ord('n'),32):
                    reviewed.add(path.name)
                    write_json(reviewed_path,sorted(reviewed))
                    break
                elif k == ord('s'):
                    # Deliberately leave unreviewed for future sessions.
                    break
                elif k in (ord('x'),ord('d')):
                    reason='startup_or_irrelevant' if k==ord('x') else 'duplicate'
                    excluded[path.name]=reason
                    reviewed.add(path.name)
                    # Remove any accidental labels on this rejected image.
                    labels=[e for e in labels if e.get('image')!=path.name]
                    base.save_labels(labels_path,labels)
                    write_json(excluded_path,excluded)
                    write_json(reviewed_path,sorted(reviewed))
                    break
                elif k in (ord('q'),27):
                    print(f'[CURATE] Saved {len(labels)} labels, {len(reviewed)} reviewed, '
                          f'{len(excluded)} excluded. Files are NOT deleted.')
                    return
    finally:
        cv2.destroyAllWindows()
    print(f'[CURATE] Finished: {len(labels)} labeled boxes, '
          f'{len(reviewed)} reviewed frames, {len(excluded)} excluded. '
          f'Images remain untouched.')


def stats(args):
    root=args.images
    images=list(root.glob('ep*_frame*.png'))
    labels=base.load_labels(root/'labels.json')
    reviewed=read_json(root/'reviewed.json',[])
    excluded=read_json(root/'excluded.json',{})
    selection=read_json(root/'curated_selection.json',[])
    print(f'Images: {len(images)} in {len(set(base.episode(x.name) for x in images))} episodes')
    print(f'Shortlist: {len(selection)} | reviewed: {len(reviewed)} | excluded: {len(excluded)}')
    print(f'Labeled boxes: {len(labels)} | class counts: {dict(Counter(x["class"] for x in labels))}')
    print(f'Labeled episodes: {len(set(x.get("episode",base.episode(x["image"])) for x in labels))}')
    print('Original images are not deleted by this tool.')


def main():
    p=argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    sub=p.add_subparsers(dest='mode',required=True)
    lp=sub.add_parser('label',help='Label a diverse shortlist; skip/mark irrelevant frames.')
    lp.add_argument('--images',type=Path,default=Path('ai_data/v2_hazard_images'))
    lp.add_argument('--max-images',type=int,default=80)
    lp.add_argument('--per-episode',type=int,default=3)
    lp.add_argument('--min-frame',type=int,default=2,help='Automatically ignore first N captured frames per episode in shortlist')
    lp.add_argument('--hash-distance',type=int,default=5,help='Near-duplicate screening threshold, 0 disables')
    lp.add_argument('--seed',type=int,default=123)
    lp.add_argument('--reshuffle',action='store_true',help='Replace shortlist; keeps existing labels/reviews/exclusions')
    lp.add_argument('--review-all',action='store_true')
    lp.add_argument('--all-images',action='store_true',help='Turn off shortlisting')
    sp=sub.add_parser('stats',help='Print current curation and label totals')
    sp.add_argument('--images',type=Path,default=Path('ai_data/v2_hazard_images'))
    tp=sub.add_parser('train',help='Use existing held-out patch-classifier trainer')
    tp.add_argument('--images',type=Path,default=Path('ai_data/v2_hazard_images'))
    tp.add_argument('--epochs',type=int,default=30)
    tp.add_argument('--batch-size',type=int,default=32)
    tp.add_argument('--seed',type=int,default=123)
    a=p.parse_args()
    if a.mode=='label':
        if a.max_images<1 or a.per_episode<1 or not 0<=a.hash_distance<=64 or a.min_frame<0:
            p.error('Invalid shortlist settings')
        label(a)
    elif a.mode=='stats':stats(a)
    else:
        # Existing train implementation uses only labels.json; unreviewed and
        # excluded images have no labels and therefore cannot enter training.
        base.train(a)

if __name__=='__main__':main()
