"""Timestamped recent-decision + older-capture history; no live dependencies."""
from collections import deque
import math
import threading
import time
import numpy as np

LAYOUTS={
    'recent4':(4,()),
    'sparse4':(1,(.1,.5,1.)),
    'sparse5':(1,(.05,.1,.5,1.)),
    'recent3_context2':(3,(.5,1.)),
    'recent4_context1':(4,(1.,)),
    'recent3_context3':(3,(.2,.6,1.)),
    'recent4_context3':(4,(.2,.6,1.)),
    'recent3_context4':(3,(.2,.4,.7,1.)),
    'recent4_context4':(4,(.2,.4,.7,1.)),
}

class HybridHistory:
    def __init__(self,recent,offsets,*,tolerance=.06,clock=time.perf_counter):
        if recent not in (1,3,4) or len(offsets)>4 or any(not math.isfinite(t) or t<=0 or t>2 for t in offsets):
            raise ValueError('Invalid recent count or context offsets')
        if list(offsets)!=sorted(set(offsets)):raise ValueError('Offsets must be unique and increasing')
        self.frames=recent+len(offsets);self.recent_count=recent
        self.offsets=tuple(sorted(offsets,reverse=True));self.span=max(offsets,default=1.)
        self.tolerance=tolerance;self.clock=clock;self.lock=threading.Lock()
        self.items=deque(maxlen=256);self.recent=deque(maxlen=recent)
        self.enabled=False;self.generation=0;self.after=float('-inf')
        self.packs=0;self.slot_valid=np.zeros(self.frames,dtype=np.int64)
        self.age_samples=deque(maxlen=1000);self.mature_packs=0
        self.mature_slot_valid=np.zeros(self.frames,dtype=np.int64)
        self.first_observation=None

    def clear(self):
        with self.lock:
            self.items.clear();self.recent.clear();self.enabled=False
            self.generation+=1;self.after=self.clock();self.first_observation=None

    def token(self):
        with self.lock:return self.generation if self.enabled else None

    def add(self,frame,stamp,generation):
        with self.lock:
            if generation is None or generation!=self.generation or not self.enabled or stamp<self.after:return
            if not math.isfinite(stamp):raise ValueError('Nonfinite frame time')
            if self.items and stamp<=self.items[-1][0]:raise ValueError('Nonmonotonic capture time')
            if self.offsets:
                self.items.append((stamp,frame.copy()))
                while self.items and self.items[0][0]<stamp-self.span-self.tolerance:self.items.popleft()

    def observation(self,latest,stamp,*,enable=True):
        if not math.isfinite(stamp):raise ValueError('Nonfinite observation time')
        with self.lock:
            # All slots are anchored to the guarded latest observation. Pump frames
            # newer than that observation never leak into the decision.
            recent=([(t,f) for t,f in self.recent if t<stamp][-(self.recent_count-1):]
                    if self.recent_count>1 else [])
            context=[(t,f) for t,f in self.items if t<stamp]
            self.enabled=enable
            if enable and stamp>=self.after:
                if not self.recent or stamp>self.recent[-1][0]:self.recent.append((stamp,latest.copy()))
                if self.first_observation is None:self.first_observation=stamp
            first=self.first_observation
        recent.append((stamp,latest))
        recent=[None]*(self.recent_count-len(recent))+recent
        used={entry[0] for entry in recent if entry is not None}
        selected=[]
        # Older context must precede even the oldest recent slot. Under heavy load
        # overlap becomes missing history instead of duplicated/misordered evidence.
        oldest_recent=min(used)
        times=np.array([t for t,_ in context])
        for age in self.offsets:
            target=stamp-age
            i=int(np.searchsorted(times,target,side='right'))-1
            entry=context[i] if i>=0 else None
            if (entry is None or entry[0] in used or entry[0]>=oldest_recent
                    or target-entry[0]>self.tolerance+1e-9):entry=None
            if entry is not None:used.add(entry[0])
            selected.append(entry)
        selected.extend(recent)
        images=[];ages=[];valid=[]
        for entry in selected:
            good=entry is not None
            images.append(entry[1] if good else np.zeros_like(latest))
            ages.append(min(1.,max(0.,(stamp-entry[0])/(self.span+self.tolerance))) if good else 1.)
            valid.append(float(good))
        self.packs+=1;self.slot_valid+=np.array(valid,dtype=np.int64)
        self.age_samples.append([None if entry is None else (stamp-entry[0])*1000 for entry in selected])
        if first is not None and stamp-first>=self.span+self.tolerance:
            self.mature_packs+=1;self.mature_slot_valid+=np.array(valid,dtype=np.int64)
        return np.concatenate(images),np.asarray(ages+valid,np.float32)

    def report(self):
        slot_ages=[]
        for i in range(self.frames):
            values=[row[i] for row in self.age_samples if row[i] is not None]
            slot_ages.append({'count':len(values),'median_ms':float(np.median(values)) if values else None,
                              'p95_ms':float(np.percentile(values,95)) if values else None})
        return {'recent_frames':self.recent_count,'context_offsets_seconds':list(self.offsets),
                'ordering':'oldest context to newest, followed by recent decision observations oldest to newest',
                'packs':self.packs,'valid_fraction_by_slot':(self.slot_valid/max(1,self.packs)).tolist(),
                'mature_packs':self.mature_packs,
                'mature_valid_fraction_by_slot':(self.mature_slot_valid/max(1,self.mature_packs)).tolist(),
                'last_1000_slot_ages':slot_ages,'startup':'zero image + invalid flag; no wait',
                'clock':'capture request time, not Unity simulation time'}
