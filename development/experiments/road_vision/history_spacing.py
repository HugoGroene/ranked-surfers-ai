"""Isolated four-frame schedules; no change to production capture or guards."""
import math
import numpy as np
from live_adapter import ContextHistory

PATTERNS=('consecutive','uniform','golden','logarithmic')

def offsets(pattern,frames,span):
    if pattern not in PATTERNS or frames!=4 or not math.isfinite(span) or not 0<span<=2:
        raise ValueError('Spacing comparison requires four frames and a valid span')
    if pattern=='consecutive':return None  # Actual pack timestamps, no fixed span.
    if pattern=='uniform':young=np.linspace(0,1,frames)
    elif pattern=='golden':
        gaps=((1+math.sqrt(5))/2)**np.arange(frames-1)
        young=np.r_[0,np.cumsum(gaps)/gaps.sum()]
    else:young=(np.geomspace(1,10,frames)-1)/9
    return young[::-1]*span

def history_class(pattern):
    if pattern=='uniform':return ContextHistory
    if pattern not in PATTERNS:raise ValueError('Unknown spacing pattern')
    class SpacedHistory(ContextHistory):
        def __init__(self,frames,span):
            super().__init__(frames,span)
            self.offsets=offsets(pattern,frames,span)

        def add(self,frame,stamp,generation):
            # Legacy-style control stores observation packs, not every pump frame.
            if pattern!='consecutive':super().add(frame,stamp,generation)

        def observation(self,latest,stamp,*,enable=True):
            if not math.isfinite(stamp):raise ValueError('Invalid observation timestamp')
            with self.lock:
                past=[(t,f) for t,f in self.items if t<stamp]
                self.enabled=enable
                if pattern=='consecutive':
                    past=past[-(self.frames-1):]
                    # Do not retain pause/resync frames for the next segment.
                    if enable and stamp>=self.after and (not self.items or stamp>self.items[-1][0]):
                        self.items.append((stamp,latest.copy()))
                        while len(self.items)>self.frames:self.items.popleft()
            past.append((stamp,latest))
            times=np.array([t for t,_ in past])
            if pattern=='consecutive':
                indices=[-1]*(self.frames-len(past))+list(range(len(past)))
                targets=[None]*self.frames
            else:
                targets=stamp-self.offsets
                indices=np.searchsorted(times,targets,side='right')-1
                indices[-1]=len(past)-1
            images,ages,valid,used=[],[],[],set()
            for target,i in zip(targets,indices):
                good=i>=0 and i not in used and (target is None or target-times[i]<=.060000001)
                images.append(past[i][1] if good else np.zeros_like(latest))
                ages.append(min(1.,max(0.,(stamp-times[i])/(self.span+.06))) if good else 1.)
                valid.append(float(good))
                if good:used.add(i)
            return np.concatenate(images),np.array(ages+valid,np.float32)
    return SpacedHistory
