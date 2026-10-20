#!/usr/bin/env python3
"""Ranked Surfers V4 screen-state model (separate from obstacle YOLO / PPO).

Pipeline (Windows PowerShell):
  record   Collect real screenshots while a human operates the LOCAL game.
  label    Label MENU / LOADING / GAMEPLAY / OTHER screenshots with keyboard.
  ingest   Copy an existing folder of manually VERIFIED images into a class.
  train    Train a small FULL-SCREEN classifier; validate by recording session.
  inspect  Print probabilities for an image or folder; no game control.
  watch    Read the display and print predictions WITHOUT sending keys.

Do not let this component control ranked matching before validation of each UI
state. Human operates game during data collection. No network access required.
"""
from __future__ import annotations

import argparse
import collections
import csv
import datetime
import json
import math
from pathlib import Path
import shutil
import time

import cv2
import numpy as np

CLASSES = ('MENU', 'LOADING', 'GAMEPLAY', 'OTHER')
IMG_WH = (192, 112)
CROP = (0.0, 0.055, 1.0, 0.965)
ROOT = Path('ai_data/v4_screen_states')


def normalize_crop(raw):
    x1, y1, x2, y2 = map(float, raw)
    if not (0 <= x1 < x2 <= 1 and 0 <= y1 < y2 <= 1):
        raise ValueError('Crop: normalized left top right bottom in 0..1')
    return (x1, y1, x2, y2)


def preprocess(bgr, crop=CROP):
    h, w = bgr.shape[:2]
    x1, y1, x2, y2 = crop
    sub = bgr[round(y1*h):round(y2*h), round(x1*w):round(x2*w)]
    if not sub.size:
        raise ValueError('empty crop')
    resized = cv2.resize(sub, IMG_WH, interpolation=cv2.INTER_AREA)
    return np.ascontiguousarray(cv2.cvtColor(resized, cv2.COLOR_BGR2RGB).transpose(2, 0, 1))


def get_torch():
    import torch
    from torch import nn
    return torch, nn


def create_net(n):
    torch, nn = get_torch()
    return nn.Sequential(
        nn.Conv2d(3, 16, 5, 2, 2), nn.ReLU(),
        nn.Conv2d(16, 32, 3, 2, 1), nn.ReLU(),
        nn.Conv2d(32, 64, 3, 2, 1), nn.ReLU(),
        nn.Conv2d(64, 96, 3, 2, 1), nn.ReLU(),
        nn.AdaptiveAvgPool2d((3, 5)), nn.Flatten(),
        nn.Linear(96*3*5, 128), nn.ReLU(), nn.Linear(128, n),
    )


class ScreenStateClassifier:
    """Low-cost classification of native BGRA frames; independent of PPO/YOLO."""
    def __init__(self, checkpoint, threads=1):
        torch, _ = get_torch()
        if threads is not None:
            torch.set_num_threads(int(threads))
        try:
            saved = torch.load(str(checkpoint), map_location='cpu', weights_only=False)
        except TypeError:  # older torch
            saved = torch.load(str(checkpoint), map_location='cpu')
        self.classes = tuple(saved['classes'])
        self.crop = normalize_crop(saved['crop'])
        self.net = create_net(len(self.classes)).cpu().eval()
        self.net.load_state_dict(saved['model'])
        self.torch = torch

    def predict(self, frame, *, bgr=True):
        if frame.shape[-1] == 4:
            frame = cv2.cvtColor(frame, cv2.COLOR_BGRA2BGR)
        elif not bgr:
            frame = cv2.cvtColor(frame, cv2.COLOR_RGB2BGR)
        chw = preprocess(frame, self.crop)
        x = self.torch.from_numpy(chw).unsqueeze(0).float().div_(255.)
        with self.torch.inference_mode():
            probabilities = self.torch.softmax(self.net(x), -1)[0].cpu().numpy()
        return {name: float(probabilities[i]) for i, name in enumerate(self.classes)}

    def state(self, frame, *, confidence=0.85):
        probs = self.predict(frame)
        label = max(probs, key=probs.get)
        return (label if probs[label] >= confidence else 'UNCERTAIN'), probs


def cmd_record(a):
    import mss
    sess = a.session or datetime.datetime.now().strftime('session_%Y%m%d_%H%M%S')
    dest = a.root / 'raw' / sess
    if dest.exists() and any(dest.iterdir()):
        raise SystemExit(f'Raw capture exists; refusing overwrite: {dest}')
    dest.mkdir(parents=True, exist_ok=True)
    with mss.mss() as sct:
        if not (1 <= a.monitor < len(sct.monitors)):
            raise SystemExit(f'Invalid monitor index {a.monitor}; available: 1..{len(sct.monitors)-1}')
        mon = sct.monitors[a.monitor]
        print(f'[RECORD] Monitor {a.monitor}: {mon}', flush=True)
        print(f'[RECORD] Switch to fullscreen LOCAL game during the {a.delay}s countdown.', flush=True)
        print('[RECORD] Press R manually, let a few runs die, press R again. No automatic inputs.', flush=True)
        for remaining in range(a.delay, 0, -1):
            print(f'  starting in {remaining}...', flush=True)
            time.sleep(1)
        begin = time.perf_counter()
        next_at = begin
        count = 0
        while time.perf_counter() - begin < a.seconds:
            now = time.perf_counter()
            if now < next_at:
                time.sleep(min(next_at-now, .02))
                continue
            shot = np.asarray(sct.grab(mon))
            bgr = cv2.cvtColor(shot, cv2.COLOR_BGRA2BGR)
            # Store 960px source: visual inspection and classification, not YOLO data.
            scale = min(1, a.width / bgr.shape[1])
            if scale < 1:
                bgr = cv2.resize(bgr, (round(scale*bgr.shape[1]), round(scale*bgr.shape[0])), interpolation=cv2.INTER_AREA)
            elapsed_ms = round((now-begin)*1000)
            filename = f'frame_{count:05d}_{elapsed_ms:07d}ms.jpg'
            cv2.imwrite(str(dest / filename), bgr, [cv2.IMWRITE_JPEG_QUALITY, 87])
            count += 1
            next_at += 1/a.fps
            if now-next_at > 1/a.fps:
                next_at = now + 1/a.fps
        (dest / 'capture.json').write_text(json.dumps({'session': sess, 'fps': a.fps, 'seconds': a.seconds,
                                                        'monitor': a.monitor, 'frames': count,
                                                        'note': 'Label human-verified states; source frames are NOT detector training images.'}, indent=2))
    print(f'[RECORD] Saved {count} frames to {dest}', flush=True)
    print(f'[NEXT] python rs_v4_screen_states.py label --session {dest}', flush=True)


def cmd_ingest(a):
    if a.class_name not in CLASSES:
        raise SystemExit('Use one of ' + ', '.join(CLASSES))
    paths = sorted(p for p in a.source.iterdir() if p.suffix.lower() in ('.jpg','.jpeg','.png'))
    if not paths:
        raise SystemExit(f'No images: {a.source}')
    dest = a.root / 'dataset' / a.class_name / a.session
    dest.mkdir(parents=True, exist_ok=True)
    for p in paths:
        shutil.copy2(p, dest / p.name)
    print(f'[INGEST] {len(paths)} manually verified {a.class_name} frames -> {dest}')
    print('WARNING: do not ingest unreviewed transition/loading frames under GAMEPLAY.')


def cmd_label(a):
    frames = sorted(p for p in a.session.iterdir() if p.suffix.lower() in ('.jpg', '.jpeg', '.png'))
    if not frames:
        raise SystemExit(f'No images: {a.session}')
    session = a.session.name
    decision_file = a.session / 'labels.csv'
    labels = {}
    if decision_file.exists():
        with decision_file.open('r', newline='', encoding='utf-8') as f:
            labels.update({row['filename']: row['label'] for row in csv.DictReader(f)})
    print('Label UI: M=MENU  L=LOADING  G=GAMEPLAY  O=OTHER  K=skip  B=back  Q=save/quit')
    print('After choosing a label, 3/5/0 apply that SAME label to the next 3/5/10 frames (review carefully).')
    print('Only label frames you are sure about. Transition frames can be OTHER.')
    keys = {ord('m'):'MENU', ord('l'):'LOADING', ord('g'):'GAMEPLAY', ord('o'):'OTHER'}
    i = 0
    last_label = None
    while i < len(frames):
        f = frames[i]
        bgr = cv2.imread(str(f))
        if bgr is None:
            i += 1
            continue
        canvas = cv2.resize(bgr, (960, 540), interpolation=cv2.INTER_AREA)
        cv2.rectangle(canvas, (0,0), (960,78), (20,20,20),-1)
        cv2.putText(canvas, f'{i+1}/{len(frames)}  {f.name}  label={labels.get(f.name,"UNLABELED")}',
                    (10,26), cv2.FONT_HERSHEY_SIMPLEX,.60,(255,255,255),1,cv2.LINE_AA)
        cv2.putText(canvas, 'M MENU / L LOADING / G GAMEPLAY / O OTHER / 3,5,0 repeat label / K skip / B back / Q quit',
                    (10,58),cv2.FONT_HERSHEY_SIMPLEX,.48,(180,220,220),1,cv2.LINE_AA)
        cv2.imshow('V4 Screen State Labeler', canvas)
        k = cv2.waitKey(0) & 0xff
        if k in keys:
            last_label = keys[k]
            labels[f.name] = last_label
            i += 1
        elif k in (ord('3'), ord('5'), ord('0')) and last_label is not None:
            how_many = 10 if k==ord('0') else int(chr(k))
            for file in frames[i:i+how_many]:
                labels[file.name] = last_label
            i += how_many
        elif k == ord('k'):
            labels.pop(f.name, None)
            i += 1
        elif k == ord('b'):
            i = max(0, i-1)
        elif k in (ord('q'), 27):
            break
    cv2.destroyAllWindows()
    with decision_file.open('w', newline='', encoding='utf-8') as f:
        writer = csv.DictWriter(f, fieldnames=['filename','label'])
        writer.writeheader()
        for p in frames:
            if p.name in labels:
                writer.writerow({'filename': p.name, 'label':labels[p.name]})
    # Clear stale labels for this exact source session if the user relabeled
    # a frame; otherwise one image can end up in multiple classes.
    for source in frames:
        for class_name in CLASSES:
            old = a.root / 'dataset' / class_name / session / source.name
            if old.exists():
                old.unlink()
    copied = collections.Counter()
    for p in frames:
        label = labels.get(p.name)
        if label not in CLASSES:
            continue
        dest = a.root / 'dataset' / label / session
        dest.mkdir(parents=True, exist_ok=True)
        shutil.copy2(p, dest / p.name)
        copied[label] += 1
    print(f'[LABEL] Copied {dict(copied)} from session {session}. Decisions: {decision_file}')
    print('[NOTE] Need at least TWO separate recordings with all main states before training.')


def load_dataset(root):
    examples=[]
    for label in CLASSES:
        sub = root / 'dataset' / label
        if not sub.is_dir():
            continue
        for session in sorted(sub.iterdir()):
            if not session.is_dir():
                continue
            for p in session.iterdir():
                if p.suffix.lower() in ('.jpg','.jpeg','.png'):
                    examples.append((p,label,session.name))
    return examples


def cmd_train(a):
    torch, nn = get_torch()
    torch.set_num_threads(max(1,a.threads))
    dataset = load_dataset(a.root)
    by_session=collections.defaultdict(set)
    counts=collections.Counter()
    for _,label,session in dataset:
        by_session[session].add(label)
        counts[label]+=1
    present = [x for x in CLASSES if counts[x] >= a.min_per_class]
    if len(present) < 3 or not {'MENU','LOADING','GAMEPLAY'}.issubset(present):
        raise SystemExit(f'Need at least {a.min_per_class} screenshots of MENU, LOADING, and GAMEPLAY each. Have {dict(counts)}')
    # Require at least 2 sessions for EACH class, so near-duplicate video frames
    # cannot leak into the validation dataset.
    if not all(sum(label in labs for labs in by_session.values()) >= 2 for label in present):
        raise SystemExit('Need >=2 distinct capture sessions for every included class, including GAMEPLAY. '
                         f'Coverage: { {k:sorted(v) for k,v in by_session.items()} }')
    sessions=sorted(by_session)
    val_session=a.val_session or sessions[-1]
    if val_session not in sessions:
        raise SystemExit(f'Validation session not found: {val_session}; choices: {sessions}')
    if not all(label in by_session[val_session] for label in present):
        raise SystemExit(f'Validation session {val_session} is missing classes {set(present)-by_session[val_session]}; '
                         'record another full menu -> loading -> gameplay session or specify --val-session')
    tr=[ex for ex in dataset if ex[2]!=val_session and ex[1] in present]
    va=[ex for ex in dataset if ex[2]==val_session and ex[1] in present]
    if not set(present).issubset({e[1] for e in tr}):
        raise SystemExit('Not all classes exist in training sessions')
    net=create_net(len(present))
    dev='cuda' if torch.cuda.is_available() else 'cpu'
    net=net.to(dev)
    index={label:i for i,label in enumerate(present)}
    def batches(items, randomize=False):
        inds = np.random.permutation(len(items)) if randomize else np.arange(len(items))
        for at in range(0,len(items),a.batch_size):
            subset=[items[int(j)] for j in inds[at:at+a.batch_size]]
            xs=[]; ys=[]
            for p,label,_ in subset:
                im=cv2.imread(str(p))
                if im is None:
                    raise RuntimeError(f'Unreadable image: {p}')
                x=preprocess(im,a.crop)
                if randomize:
                    x=np.clip(x.astype(np.float32)*np.random.uniform(.88,1.12) + np.random.uniform(-5,5),0,255).astype(np.uint8)
                xs.append(x)
                ys.append(index[label])
            yield torch.from_numpy(np.stack(xs)).float().to(dev).div_(255.), torch.tensor(ys,dtype=torch.long,device=dev)
    c=collections.Counter(label for _,label,_ in tr)
    weights=torch.tensor([len(tr)/(len(present)*c[l]) for l in present],dtype=torch.float32,device=dev)
    criterion=nn.CrossEntropyLoss(weight=weights)
    optimizer=torch.optim.Adam(net.parameters(),lr=a.lr)
    best_loss=float('inf'); best=None
    print(f'[TRAIN] Classes: {present}; train={len(tr)}, heldout={len(va)} (session={val_session}), device={dev}',flush=True)
    for epoch in range(1,a.epochs+1):
        net.train()
        loss_sum=0; total=0
        for x,y in batches(tr,True):
            logits=net(x); loss=criterion(logits,y)
            optimizer.zero_grad(); loss.backward(); optimizer.step()
            loss_sum+=float(loss.item())*len(y); total+=len(y)
        net.eval(); correct=0; confusion=np.zeros((len(present),len(present)),dtype=int)
        vloss=0; vtotal=0
        with torch.no_grad():
            for x,y in batches(va):
                pred=net(x); vloss+=float(criterion(pred,y).item())*len(y); vtotal+=len(y)
                yp=pred.argmax(1)
                correct+=int((yp==y).sum().item())
                for truth,guess in zip(y.tolist(),yp.tolist()):
                    confusion[truth,guess]+=1
        val_loss=vloss/max(vtotal,1)
        print(f'[TRAIN] epoch={epoch:02d} loss={loss_sum/max(total,1):.4f} '
              f'heldout_loss={val_loss:.4f} heldout_acc={correct/max(vtotal,1):.3f}',flush=True)
        if val_loss<best_loss:
            best_loss=val_loss
            best={'model':{k:v.cpu().clone() for k,v in net.state_dict().items()},
                  'classes':present,'crop':list(a.crop),'input_wh':IMG_WH,
                  'validation_session':val_session}
            best_confusion=confusion.copy()
            best_acc=correct/max(vtotal,1)
    a.output.parent.mkdir(parents=True,exist_ok=True)
    torch.save(best,a.output)
    report={'model':str(a.output),'classes':present,'train_count':len(tr),'validation_count':len(va),
            'validation_session':val_session,'best_validation_accuracy':round(best_acc,4),
            'confusion_rows_truth_cols_pred':best_confusion.tolist(), 'class_order':present,
            'warning':'A separate manual live watch test is required before allowing automated keyboard actions.'}
    a.output.with_suffix('.json').write_text(json.dumps(report,indent=2))
    print('[TRAIN] Saved model:',a.output)
    print('[TRAIN] Confusion matrix (rows=true, cols=pred):',present,best_confusion.tolist())
    if best_acc < .98 or any(best_confusion[i,i] < best_confusion[i].sum() for i in range(len(present))):
        print('[TRAIN] HOLD: misclassifications on held-out session. Add examples and retrain.', flush=True)
    else:
        print('[TRAIN] Heldout session all correct; STILL validate with watch on new episodes.',flush=True)


def cmd_inspect(a):
    cls=ScreenStateClassifier(a.model)
    paths=[a.source] if a.source.is_file() else sorted(p for p in a.source.iterdir() if p.suffix.lower() in ('.jpg','.jpeg','.png'))
    if not paths: raise SystemExit('No images found')
    for p in paths[:a.limit]:
        image=cv2.imread(str(p))
        if image is None: continue
        label, probs=cls.state(image,confidence=a.confidence)
        print(f'[INSPECT] {p.name}: {label} '+', '.join(f'{k}={v:.3f}' for k,v in probs.items()))


def cmd_watch(a):
    import mss
    cls=ScreenStateClassifier(a.model)
    with mss.mss() as sct:
        if not 1 <= a.monitor < len(sct.monitors):
            raise SystemExit('Invalid monitor')
        monitor=sct.monitors[a.monitor]
        print(f'[WATCH] Capturing {monitor}. No keyboard inputs will be sent.', flush=True)
        print('[WATCH] Switch to game and press R manually. Watch for wrong states. Ctrl+C to stop.',flush=True)
        time.sleep(a.delay)
        started=time.perf_counter(); next_at=started
        last=None; history=collections.deque(maxlen=3)
        try:
            while time.perf_counter()-started<a.seconds:
                if time.perf_counter()<next_at:
                    time.sleep(.005); continue
                shot=np.asarray(sct.grab(monitor))
                t=time.perf_counter(); pred,probs=cls.state(shot,confidence=a.confidence)
                history.append(pred)
                stable=pred if len(history)==3 and len(set(history))==1 else 'TRANSITION'
                if stable!=last or a.verbose:
                    print(f'[WATCH] {t-started:6.2f}s stable={stable:10s} raw={pred:10s} '
                          +' '.join(f'{k}={v:.2f}' for k,v in probs.items()),flush=True)
                    last=stable
                next_at+=1/a.fps
        except KeyboardInterrupt:
            pass
    print('[WATCH] Ended. Verify zero GAMEPLAY predictions on loading/menu before integration.')


def parser():
    p=argparse.ArgumentParser(description=__doc__,formatter_class=argparse.RawDescriptionHelpFormatter)
    sub=p.add_subparsers(dest='command',required=True)
    def common(q):
        q.add_argument('--root',type=Path,default=ROOT)
    r=sub.add_parser('record'); common(r)
    r.add_argument('--session',default=None)
    r.add_argument('--seconds',type=float,default=35)
    r.add_argument('--fps',type=float,default=5)
    r.add_argument('--delay',type=int,default=5)
    r.add_argument('--width',type=int,default=960)
    r.add_argument('--monitor',type=int,default=1)
    l=sub.add_parser('label'); common(l)
    l.add_argument('--session',type=Path,required=True)
    ing=sub.add_parser('ingest'); common(ing)
    ing.add_argument('--source',type=Path,required=True)
    ing.add_argument('--class-name',choices=CLASSES,required=True)
    ing.add_argument('--session',required=True)
    tr=sub.add_parser('train'); common(tr)
    tr.add_argument('--output',type=Path,default=ROOT/'screen_state_best.pt')
    tr.add_argument('--crop',type=float,nargs=4,default=CROP)
    tr.add_argument('--epochs',type=int,default=20)
    tr.add_argument('--batch-size',type=int,default=32)
    tr.add_argument('--lr',type=float,default=0.001)
    tr.add_argument('--min-per-class',type=int,default=12)
    tr.add_argument('--threads',type=int,default=2)
    tr.add_argument('--val-session',default=None)
    insp=sub.add_parser('inspect')
    insp.add_argument('--model',type=Path,default=ROOT/'screen_state_best.pt')
    insp.add_argument('--source',type=Path,required=True)
    insp.add_argument('--limit',type=int,default=30)
    insp.add_argument('--confidence',type=float,default=.85)
    w=sub.add_parser('watch'); w.add_argument('--model',type=Path,default=ROOT/'screen_state_best.pt')
    w.add_argument('--monitor',type=int,default=1); w.add_argument('--seconds',type=float,default=35)
    w.add_argument('--fps',type=float,default=8); w.add_argument('--delay',type=float,default=5)
    w.add_argument('--confidence',type=float,default=.85); w.add_argument('--verbose',action='store_true')
    return p


def main():
    p=parser(); a=p.parse_args()
    if hasattr(a,'crop'): a.crop=normalize_crop(a.crop)
    if a.command=='record': cmd_record(a)
    elif a.command=='label': cmd_label(a)
    elif a.command=='ingest': cmd_ingest(a)
    elif a.command=='train': cmd_train(a)
    elif a.command=='inspect': cmd_inspect(a)
    elif a.command=='watch': cmd_watch(a)

if __name__=='__main__': main()
