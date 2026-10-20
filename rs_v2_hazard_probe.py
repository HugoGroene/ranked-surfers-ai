#!/usr/bin/env python3
"""Label & test visual object recognition from REAL 256x144 V2 observations.

This is a DIAGNOSTIC PATCH CLASSIFIER, not a hazard-detector reward or policy.
No claims of lane safety, motion prediction, or maneuver correctness are made.
Draw rectangles on enlarged real frames, selecting c/t/b/e for
coin/train/barrier/empty. Train/holdout split is by independent Practice episode
so nearly identical consecutive video frames don't leak into the test set.

  python rs_v2_hazard_probe.py label --images ai_data/v2_hazard_images
  python rs_v2_hazard_probe.py train --images ai_data/v2_hazard_images --epochs 30
"""
import argparse
import json
import random
import re
from collections import Counter, defaultdict
from pathlib import Path
import cv2
import numpy as np

CLASSES=('coin','train','barrier','empty')
KEYMAP={ord('c'):'coin',ord('t'):'train',ord('b'):'barrier',ord('e'):'empty'}
COLORS={'coin':(50,230,230),'train':(220,150,70),'barrier':(60,100,240),'empty':(140,230,100)}


def episode(name):
    match=re.match(r'ep(\d+)_',name)
    return match.group(1) if match else name


def load_labels(path):
    if not path.is_file(): return []
    data=json.loads(path.read_text())
    if not isinstance(data,list): raise ValueError('Invalid labels.json (expected list)')
    return data


def save_labels(path, labels):
    tmp=path.with_suffix('.tmp')
    tmp.write_text(json.dumps(labels,indent=2))
    tmp.replace(path)


def draw_header(im, selected, count, current, total, msg=''):
    h,w=im.shape[:2]
    canvas=np.zeros((h+58,w,3),dtype=np.uint8)
    canvas[58:]=im
    cv2.putText(canvas,f'Image {current}/{total}   class={selected.upper()}   boxes={count}',
                (8,22),cv2.FONT_HERSHEY_SIMPLEX,.62,(255,255,255),1,cv2.LINE_AA)
    cv2.putText(canvas,'C coin  T train  B barrier  E empty   DRAG box   U undo   N next   Q quit',
                (8,42),cv2.FONT_HERSHEY_SIMPLEX,.47,(185,225,255),1,cv2.LINE_AA)
    if msg:
        cv2.putText(canvas,msg[:65],(8,h+49),cv2.FONT_HERSHEY_SIMPLEX,.5,(0,0,255),1)
    return canvas


def label(args):
    root=args.images
    images=sorted(root.glob('ep*_frame*.png'))
    if not images: raise SystemExit(f'No V2 frames in {root}. Run rs_v2_capture_hazards.py first.')
    labels_path=root/'labels.json'
    labels=load_labels(labels_path)
    reviewed_path=root/'reviewed.json'
    reviewed=set(json.loads(reviewed_path.read_text())) if reviewed_path.exists() else set()
    pending=[p for p in images if args.review_all or p.name not in reviewed]
    if not pending:
        print(f'All {len(images)} images reviewed. Use --review-all to inspect again.')
        return
    window='Ranked Surfers - real frame hazard labels'
    cv2.namedWindow(window,cv2.WINDOW_NORMAL)
    cv2.resizeWindow(window,1040,650)
    selected='coin'
    scale=4
    try:
        for index,p in enumerate(pending,1):
            bgr=cv2.imread(str(p))
            if bgr is None: continue
            h,w=bgr.shape[:2]
            frame_entries=[entry for entry in labels if entry['image']==p.name]
            state={'start':None,'temporary':None,'add':None}
            def on_mouse(event,xx,yy,flags,param):
                x=int(np.clip(xx/scale,0,w-1)); y=int(np.clip((yy-58)/scale,0,h-1))
                if yy<58: return
                if event==cv2.EVENT_LBUTTONDOWN:
                    state['start']=(x,y)
                elif event==cv2.EVENT_MOUSEMOVE and state['start'] is not None:
                    state['temporary']=(*state['start'],x,y)
                elif event==cv2.EVENT_LBUTTONUP and state['start'] is not None:
                    x0,y0=state['start']
                    x1,x2=sorted((x0,x)); y1,y2=sorted((y0,y))
                    state['start']=None
                    state['temporary']=None
                    if x2-x1>=3 and y2-y1>=3:
                        state['add']={'image':p.name,'episode':episode(p.name),'class':selected,
                                      'box':[x1,y1,x2,y2]}
            cv2.setMouseCallback(window,on_mouse)
            while True:
                if state['add'] is not None:
                    box=state['add']; labels.append(box);frame_entries.append(box)
                    state['add']=None
                    save_labels(labels_path,labels)
                show=cv2.resize(bgr,(w*scale,h*scale),interpolation=cv2.INTER_NEAREST)
                for ent in frame_entries:
                    x0,y0,x1,y1=ent['box']
                    color=COLORS.get(ent['class'],(255,255,255))
                    cv2.rectangle(show,(x0*scale,y0*scale),(x1*scale,y1*scale),color,2)
                    cv2.putText(show,ent['class'],(x0*scale,max(15,y0*scale-3)),
                                cv2.FONT_HERSHEY_SIMPLEX,.49,color,1)
                if state['temporary'] is not None:
                    x0,y0,x1,y1=state['temporary']
                    cv2.rectangle(show,(x0*scale,y0*scale),(x1*scale,y1*scale),COLORS[selected],1)
                cv2.imshow(window,draw_header(show,selected,len(frame_entries),index,len(pending)))
                k=cv2.waitKey(50)&255
                if k in KEYMAP: selected=KEYMAP[k]
                elif k==ord('u') and frame_entries:
                    last=frame_entries.pop()
                    for i in range(len(labels)-1,-1,-1):
                        if labels[i] is last:
                            labels.pop(i);break
                    save_labels(labels_path,labels)
                elif k in (ord('n'),32):
                    reviewed.add(p.name)
                    reviewed_path.write_text(json.dumps(sorted(reviewed),indent=2))
                    break
                elif k in (ord('q'),27):
                    save_labels(labels_path,labels)
                    print(f'[{len(labels)} BOXES] Labels saved to {labels_path}')
                    return
    finally: cv2.destroyAllWindows()
    print(f'Finished. {len(labels)} boxes across {len(reviewed)} reviewed frames. Labels: {labels_path}')
    print('Use --review-all to re-open already reviewed images and add/undo boxes.')


def patch(img,box,margin=1.4):
    h,w=img.shape[:2]
    x0,y0,x1,y1=box
    cx=(x0+x1)/2;cy=(y0+y1)/2
    width=max(8, (x1-x0)*margin); height=max(8, (y1-y0)*margin)
    x0=max(0,round(cx-width/2));x1=min(w,round(cx+width/2))
    y0=max(0,round(cy-height/2));y1=min(h,round(cy+height/2))
    src=img[y0:y1,x0:x1]
    if src.size==0: raise ValueError(f'Invalid box {box}')
    return cv2.cvtColor(cv2.resize(src,(64,64),interpolation=cv2.INTER_LINEAR), cv2.COLOR_BGR2RGB)


def get_arrays(root):
    labels=load_labels(root/'labels.json')
    results=[]
    cache={}
    for ent in labels:
        if ent.get('class') not in CLASSES: continue
        p=root/ent['image']
        if not p.is_file():continue
        if p.name not in cache: cache[p.name]=cv2.imread(str(p))
        try:
            img=patch(cache[p.name],ent['box'])
            results.append((img,CLASSES.index(ent['class']),ent.get('episode') or episode(p.name)))
        except Exception as exc: print(f'Skipping invalid {p.name}: {exc}')
    return results


def f1_score(conf):
    f=[]
    for i in range(len(CLASSES)):
        tp=conf[i,i]
        fp=conf[:,i].sum()-tp
        fn=conf[i,:].sum()-tp
        p=tp/max(1,tp+fp)
        r=tp/max(1,tp+fn)
        f.append(2*p*r/max(1e-8,p+r))
    return f


def train(args):
    import torch
    from torch import nn
    torch.manual_seed(args.seed)
    rng=np.random.default_rng(args.seed)
    examples=get_arrays(args.images)
    if not examples: raise SystemExit('No labeled patches. Run: python rs_v2_hazard_probe.py label')
    groups=sorted(set(e[2] for e in examples))
    counts=Counter(CLASSES[e[1]] for e in examples)
    print('Labeled boxes:',dict(counts),'episodes:',len(groups))
    if len(groups)<5: raise SystemExit('Need labels from >=5 DIFFERENT Practice episodes for trustworthy held-out evaluation.')
    if any(counts.get(c,0)<12 for c in CLASSES):
        raise SystemExit('Please label at least 12 examples of EACH: coin/train/barrier/empty. More is better.')
    rng.shuffle(groups)
    holdouts=set(groups[:max(1,round(.25*len(groups)))])
    train=[e for e in examples if e[2] not in holdouts]
    val=[e for e in examples if e[2] in holdouts]
    ct=Counter(CLASSES[e[1]] for e in train)
    cv=Counter(CLASSES[e[1]] for e in val)
    if any(ct.get(c,0)<6 or cv.get(c,0)<3 for c in CLASSES):
        raise SystemExit(f'Need all 4 classes represented in BOTH train and holdout episodes. Train={dict(ct)} holdout={dict(cv)}. Label more episodes and retry with --seed 124 if needed.')
    print('Training split by episode, NOT frame. Train=',dict(ct),'Holdout=',dict(cv))
    class SmallCNN(nn.Module):
        def __init__(self):
            super().__init__()
            self.net=nn.Sequential(nn.Conv2d(3,16,5,padding=2),nn.ReLU(),nn.MaxPool2d(2),
                nn.Conv2d(16,32,3,padding=1),nn.ReLU(),nn.MaxPool2d(2),
                nn.Conv2d(32,64,3,padding=1),nn.ReLU(),nn.MaxPool2d(2),
                nn.AdaptiveAvgPool2d(1),nn.Flatten(),nn.Linear(64,len(CLASSES)))
        def forward(self,x):return self.net(x)
    device='cuda' if torch.cuda.is_available() else 'cpu'
    model=SmallCNN().to(device)
    optimizer=torch.optim.Adam(model.parameters(),lr=.001)
    loss_fn=nn.CrossEntropyLoss(weight=torch.tensor([len(train)/(len(CLASSES)*ct[c]) for c in CLASSES],device=device,dtype=torch.float32))
    def tensors(data):
        X=torch.tensor(np.stack([e[0] for e in data]).transpose(0,3,1,2).copy(),dtype=torch.float32,device=device)/255
        Y=torch.tensor([e[1] for e in data],dtype=torch.int64,device=device)
        return X,Y
    X,Y=tensors(train)
    XV,YV=tensors(val)
    best=(-1.,None,0)
    for epoch in range(1,args.epochs+1):
        model.train()
        order=torch.randperm(len(train),device=device)
        losses=[]
        for ids in order.split(args.batch_size):
            output=model(X[ids]);loss=loss_fn(output,Y[ids])
            optimizer.zero_grad(set_to_none=True);loss.backward();optimizer.step()
            losses.append(float(loss.item()))
        model.eval()
        with torch.no_grad():
            pred=model(XV).argmax(1)
            conf=np.zeros((len(CLASSES),len(CLASSES)),dtype=np.int64)
            for y,predy in zip(YV.cpu().numpy(),pred.cpu().numpy()): conf[y,predy]+=1
            acc=float(np.trace(conf)/max(1,conf.sum()))
            mf=float(np.mean(f1_score(conf)))
        if mf>best[0]:
            best=(mf,{k:v.detach().cpu().clone() for k,v in model.state_dict().items()},epoch)
        if epoch==1 or epoch%5==0 or epoch==args.epochs:
            print(f'[VISION] epoch {epoch:3d} loss={np.mean(losses):.3f} holdout_accuracy={acc:.1%} holdout_macroF1={mf:.2f}',flush=True)
    model.load_state_dict(best[1]); model.eval()
    with torch.no_grad():
        pred=model(XV).argmax(1).cpu().numpy()
    conf=np.zeros((4,4),dtype=np.int64)
    for yy,pp in zip(YV.cpu().numpy(),pred): conf[yy,pp]+=1
    accuracy=float(np.trace(conf)/conf.sum())
    perclass=f1_score(conf)
    majority=max(cv.values())/len(val)
    print('\nBEST held-out checkpoint, epoch',best[2])
    print('Classes:',list(CLASSES))
    print('Confusion matrix: ROW=truth COLUMN=prediction')
    for name,row in zip(CLASSES,conf):print(f' {name:8s}',list(map(int,row)))
    print(f'holdout accuracy={accuracy:.1%}; macro-F1={np.mean(perclass):.3f}; majority baseline={majority:.1%}')
    report={'classes':list(CLASSES),'training_samples':len(train),'holdout_samples':len(val),
        'training_episodes':len(groups)-len(holdouts),'heldout_episodes':len(holdouts),
        'holdout_accuracy':accuracy,'macro_f1':float(np.mean(perclass)),
        'majority_baseline':majority,'per_class_f1':dict(zip(CLASSES,map(float,perclass))),
        'confusion_matrix':conf.tolist(),'best_epoch':best[2],
        'warning':'Diagnostic on MANUALLY CROPPED still images, not proof of autonomous obstacle recognition or safe maneuvers.'}
    dest=args.images/'hazard_probe_report.json'
    dest.write_text(json.dumps(report,indent=2))
    torch.save({'state_dict':best[1],'classes':CLASSES},args.images/'hazard_probe_weights.pt')
    print(f'Report: {dest}')
    if accuracy<=majority:
        print('NOT VALIDATED: no improvement over majority guess. Do not use predicted hazards for rewards.')
    elif accuracy>=.80 and np.mean(perclass)>=.75:
        print('PROMISING PATCH RECOGNITION: test on more unseen episodes before integration into the policy.')
    else:
        print('INCONCLUSIVE: real-patch recognition not reliable enough for reward shaping yet.')


def main():
    p=argparse.ArgumentParser(description=__doc__,formatter_class=argparse.RawDescriptionHelpFormatter)
    sub=p.add_subparsers(dest='mode',required=True)
    lp=sub.add_parser('label',help='Draw class-labeled boxes on actual V2 RGB screenshots')
    lp.add_argument('--images',type=Path,default=Path('ai_data/v2_hazard_images'))
    lp.add_argument('--review-all',action='store_true')
    tp=sub.add_parser('train',help='Train/evaluate a held-out object-patch classifier')
    tp.add_argument('--images',type=Path,default=Path('ai_data/v2_hazard_images'))
    tp.add_argument('--epochs',type=int,default=30)
    tp.add_argument('--batch-size',type=int,default=32)
    tp.add_argument('--seed',type=int,default=123)
    args=p.parse_args()
    if args.mode=='label':label(args)
    else:train(args)
if __name__=='__main__': main()
