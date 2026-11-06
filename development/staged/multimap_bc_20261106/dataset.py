"""Causal fixed4/native BC export with the live 789-value auxiliary contract."""
import argparse
from collections import deque
from datetime import datetime
import json
import hashlib
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
import numpy as np
import cv2
from common import ROOT,DATA,KEYS,read,write_new,digest,contract,require_idle
from review import latest,runs,authorized
from development.staged.hybrid_history_20261102.hybrid import HybridHistory
from development.experiments.command_history.history import CommandHistory
from rs_v61_features import ActionContext,encode_aux
from rs_v6_ppo_features import HAZARD_DIM
from development.experiments.road_vision.road_crop import crop_rgb
from export_support import reviewed_states,apply_reviewed,compact_indices


def labels(rows,events,decision_hz=30):
    """No future pixels, held/repeated action ambiguity, or >130ms labels."""
    times=np.array([r['at'] for r in rows]);ends=np.array([r['end'] for r in rows])
    if not np.isfinite(decision_hz) or not 1<=decision_hz<=30:raise ValueError('Invalid decision rate')
    if any(not np.isfinite(e['at']) or e['type'] not in ('keydown','keyup') for e in events):
        raise ValueError('Invalid physical event')
    if (len(times)<4 or not np.isfinite(times).all() or not np.isfinite(ends).all()
            or np.any(np.diff(times)<=0) or np.any(ends<times) or np.any(ends[:-1]>times[1:])):
        raise ValueError('Invalid capture clock or overlapping captures')
    chosen=[]
    for i,t in enumerate(times):
        if not chosen or t-times[chosen[-1]]>=1/decision_hz-1e-6:chosen.append(i)
    commands=[];samples=[];held={};holds=[]
    ordered=sorted((e for e in events if e['code'] in KEYS),key=lambda e:e['at'])
    for e in ordered:
        if e['type']=='keydown':held.setdefault(e['code'],e['at'])
        elif e['code'] in held:
            start=held.pop(e['code'])
            if e['at']-start>.13:holds.append((start,e['at']))
    holds += [(start,float('inf')) for start in held.values()]
    downs=[e for e in ordered if e['type']=='keydown']
    commands.extend((e['at'],KEYS[e['code']]) for e in downs)
    for i,j in zip(chosen,chosen[1:]):
        start,end=times[i],times[j]
        keys=[e for e in downs if start<=e['at']<end]
        good=(end-start<=.13 and rows[i]['state']=='GAMEPLAY' and rows[j]['state']=='GAMEPLAY'
              and not any(lo<end and hi>start for lo,hi in holds)
              and not any(start<=e['at']<=ends[i] for e in keys) and len(keys)<=1)
        action=KEYS[keys[0]['code']] if len(keys)==1 else 0
        if not keys:
            # A completed empty decision interval is the only synthetic NONE.
            # It is added to later observations, never its own target observation.
            commands.append((float(ends[i]),0))
        samples.append(dict(index=i,next_index=j,action=action,valid=good))
    commands.sort()
    return samples,commands


def reconstruct(rows,events,intervals,decision_hz=30):
    """Use the same temporal/history encoders as live; each approved segment resets."""
    targets,commands=labels(rows,events,decision_hz)
    target_by_index={s['index']:s for s in targets};times=np.array([r['at'] for r in rows])
    origin=times[0];results=[]
    for segment,(lo,hi) in enumerate(intervals):
        temporal=HybridHistory(1,(.05,.25,.75),clock=lambda:origin+lo)
        timeline=CommandHistory();context=ActionContext();history=deque(maxlen=4);dts=deque(maxlen=3)
        previous=None;pending=deque((t,a) for t,a in commands if origin+lo<=t<origin+hi)
        for i in np.flatnonzero((times>=origin+lo)&(times<origin+hi)):
            at=times[i]
            while pending and pending[0][0]<at:
                t,action=pending.popleft();timeline.add(action,t);context.note_action(action,t);history.append(action)
            # Never pass a menu/pause/uncertain image as a valid memory frame.
            if rows[i]['state']!='GAMEPLAY':continue
            # Tiny integer tags exercise the exact live selection algorithm;
            # native pixels stay in a disk-backed array, not a giant stack list.
            frame=np.full((3,1,1),int(i)+1,np.int64)
            if temporal.token() is None:temporal.observation(frame,at)
            temporal.add(frame,at,temporal.token())
            sample=target_by_index.get(int(i))
            if sample is None:continue
            if previous is not None:dts.append(at-previous)
            previous=at
            image,timing=temporal.observation(frame,at)
            aux=encode_aux(np.zeros(HAZARD_DIM,np.float32),history,at-origin,dts,None,
                snapshot=None,episode_tag=1,now=at,action_context=context)
            aux=np.concatenate((aux,timing,timeline.vector(at)))
            if sample['valid'] and times[sample['next_index']]<origin+hi:
                results.append(dict(index=int(i),segment=segment,action=sample['action'],
                                    indices=image[::3,0,0]-1,aux=aux))
    return results


def stacked(frames,indices):
    image=np.zeros((4,*frames.shape[1:]),np.uint8)
    good=indices>=0;image[good]=frames[indices[good]]
    return image.reshape(12,*frames.shape[-2:])


def export(root,checkpoint,output,decision_hz=30,*,reviewed_root=None,workers=2,resume=False):
    c=contract(checkpoint);cfg=c['road_vision'];output=Path(output)
    checkpoint_sha=digest(checkpoint)
    reviewed,provenance=reviewed_states(reviewed_root)
    selected=[]
    for file in runs(root):
        review,review_file=latest(file.parent)
        if review['status']!='approved':continue
        if not authorized(review,read(file)):raise ValueError('Missing visual approval or explicit duration selection')
        for name,key in [('manifest.json','manifest_sha256'),('frames.json','index_sha256'),('events.json','events_sha256')]:
            if digest(file.parent/name)!=review[key]:raise ValueError('Recording changed since approval')
        selected.append((file.parent,review,review_file))
    if not selected:raise ValueError('No approved recordings')
    plan=dict(contract=c,checkpoint_sha256=checkpoint_sha,decision_hz=decision_hz,
              reviewed_states=provenance,runs=[dict(source=str(r.resolve()),review_sha256=digest(f)) for r,_,f in selected])
    if resume:
        if read(output/'export_plan.json')!=plan:raise ValueError('Export plan changed; use a new output folder')
    else:
        output.mkdir(parents=True,exist_ok=False);write_new(output/'export_plan.json',plan)
    records=[];cv2.setNumThreads(1)
    print('EXPORT:',output.resolve(),'; approved runs:',len(selected),flush=True)
    for number,(run,review,review_file) in enumerate(selected):
        m=read(run/'manifest.json');rows=read(run/'frames.json');events=read(run/'events.json')
        completed=sorted(output.glob(f'run_{number:04d}*/run_record.json'))
        if completed:
            record=read(completed[-1]);dest=completed[-1].parent
            if any(digest(dest/name)!=sha for name,sha in record['hashes'].items()):raise ValueError('Completed export checksum mismatch')
            records.append(record)
            print(f'[{number+1}/{len(selected)}] RETAINED {run.name}',flush=True)
            continue
        effective,corrections=apply_reviewed(rows,run,m['split'],reviewed)
        samples=reconstruct(effective,events,review['intervals'],decision_hz)
        if not samples:raise ValueError('Approved run has no valid training samples')
        used,indices=compact_indices(samples)
        dest=output/f'run_{number:04d}'
        if dest.exists():dest=output/f'run_{number:04d}_retry_{datetime.now():%Y%m%d_%H%M%S_%f}'
        dest.mkdir()
        images=np.lib.format.open_memmap(dest/'frames.npy',mode='w+',dtype=np.uint8,
                                        shape=(len(used),3,*c['image_shape'][-2:]))
        def decode(i):
            r=rows[i]
            if r['file']!=f'{i:06d}.png':raise ValueError('Unexpected frame path')
            path=run/'frames'/r['file']
            raw=path.read_bytes()
            if hashlib.sha256(raw).hexdigest()!=r['sha256']:raise ValueError('Native frame checksum mismatch')
            native=cv2.imdecode(np.frombuffer(raw,np.uint8),cv2.IMREAD_COLOR)
            if native is None or native.shape!=(945,1920,3):raise ValueError('Native canvas mismatch')
            return crop_rgb(cv2.cvtColor(native,cv2.COLOR_BGR2RGB),cfg)
        # Bound both decoder concurrency and queued pixels while the classifier trains.
        with ThreadPoolExecutor(max_workers=workers) as pool:
            for start in range(0,len(used),8):
                for offset,image in enumerate(pool.map(decode,used[start:start+8])):
                    images[start+offset]=image
        images.flush()
        np.save(dest/'indices.npy',indices)
        np.save(dest/'aux.npy',np.stack([s['aux'] for s in samples]))
        np.save(dest/'actions.npy',np.array([s['action'] for s in samples],np.int64))
        write_new(dest/'samples.json',[{k:v for k,v in s.items() if k not in ('indices','aux')} for s in samples])
        record=dict(path=dest.name,map=m['map'],split=m['split'],session=m['session'],
            source_frame_indices=used.tolist(),source_frames=len(rows),reviewed_state_changes=corrections,
            approval_basis=review.get('approval_basis','visual_review'),visually_reviewed=review.get('reviewed',False),
            source=str(run.resolve()),review_sha256=digest(review_file),count=len(samples),
            action_counts=np.bincount([s['action'] for s in samples],minlength=5).tolist(),
            hashes={name:digest(dest/name) for name in ('frames.npy','indices.npy','aux.npy','actions.npy','samples.json')})
        # A few exact model inputs: channels ordered oldest to newest, no labels drawn on pixels.
        for k in sorted(set([0,len(samples)//2,len(samples)-1])):
            image=stacked(images,indices[k]);tiles=image.reshape(4,3,*image.shape[-2:])
            sheet=np.concatenate([f.transpose(1,2,0) for f in tiles],axis=1)
            cv2.imwrite(str(dest/f'input_{k:06d}.png'),cv2.cvtColor(sheet,cv2.COLOR_RGB2BGR))
        images._mmap.close();del samples,images
        write_new(dest/'run_record.json',record);records.append(record)
        print(f'[{number+1}/{len(selected)}] {m["map"]} {m["split"]} {run.name}: {record["count"]} samples; {len(used)}/{len(rows)} frames',flush=True)
    if digest(checkpoint)!=checkpoint_sha:raise ValueError('Source checkpoint changed during export')
    manifest=dict(schema='multimap-bc-fixed4-v1',status='COMPLETE',contract=c,
        checkpoint=str(Path(checkpoint).resolve()),checkpoint_sha256=checkpoint_sha,decision_hz=decision_hz,
        reviewed_states=provenance,runs=records,label_bound_ms=130,
        validation='whole recording sessions; no neighboring-frame random split')
    if (output/'dataset.json').exists():
        if read(output/'dataset.json')!=manifest:raise ValueError('Final manifest mismatch')
    else:write_new(output/'dataset.json',manifest)
    return output/'dataset.json'


def main():
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('--root',type=Path,default=DATA)
    p.add_argument('--checkpoint',type=Path,required=True);p.add_argument('--output',type=Path)
    p.add_argument('--decision-hz',type=int,choices=[15,20,30],default=30)
    p.add_argument('--reviewed-states-root',type=Path)
    p.add_argument('--workers',type=int,choices=[1,2],default=2)
    p.add_argument('--resume',action='store_true')
    p.add_argument('--allow-classifier-training',action='store_true',help='CPU export may run alongside the separate classifier trainer')
    a=p.parse_args();require_idle(allow_classifier_training=a.allow_classifier_training)
    output=a.output or DATA/'exports'/f'{datetime.now():%Y%m%d_%H%M%S_%f}'
    print(export(a.root,a.checkpoint,output,a.decision_hz,reviewed_root=a.reviewed_states_root,workers=a.workers,resume=a.resume))


if __name__=='__main__':main()
