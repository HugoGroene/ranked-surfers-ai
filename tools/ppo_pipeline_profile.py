"""Opt-in buffered pipeline timings. No extra captures, actions or per-step fences."""
import json
from pathlib import Path
import time
import types


class PipelineProfiler:
    def __init__(self,raw,output,device='cpu',max_events=100000):
        self.raw,self.output=raw,Path(output)
        self.events=[]
        self.max_events=max_events
        self.dropped_events=0
        self.gpu_events=None
        if str(device).startswith('cuda'):
            import torch
            self.gpu_events=(torch.cuda.Event(enable_timing=True),torch.cuda.Event(enable_timing=True))
            # Warm event allocation before collection, never fence each decision.
            self.gpu_events[0].record();self.gpu_events[1].record()
            self.gpu_events[1].synchronize()
        raw._pipeline_profiler=self
        for name in ('_capture','_headless_image','_step_browser'):
            self._wrap(name,True)
        for name in ('_native_frames','_read_screen_state','_pack'):
            self._wrap(name,False)
        self._wrap('screenshot',True,getattr(raw,'page',None),'page_screenshot')
        self._wrap('bounding_box',True,getattr(raw,'canvas',None),'canvas_bounding_box')

    def record(self,row):
        if len(self.events)<self.max_events:self.events.append(row)
        else:self.dropped_events+=1

    def _wrap(self,name,asynchronous,target=None,label=None):
        target=self.raw if target is None else target
        original=getattr(target,name,None)
        if original is None:return
        def finished(started):
            row=dict(kind=label or name,start=started,end=time.perf_counter(),
                frame_serial=getattr(self.raw,'_headless_capture_timing',{}).get('electron_serial'))
            if name=='_pack':
                row['source']=dict(getattr(self.raw,'_learning_observation_capture',{}) or {})
            if name=='_step_browser':
                row['guard_source']=dict(getattr(self.raw,'_reaction_guard_capture',{}) or {})
            self.record(row)
        if asynchronous:
            async def measured(_env,*args,**kwargs):
                began=time.perf_counter()
                try:return await original(*args,**kwargs)
                finally:finished(began)
        else:
            def measured(_env,*args,**kwargs):
                began=time.perf_counter()
                try:return original(*args,**kwargs)
                finally:finished(began)
        setattr(target,name,types.MethodType(measured,target))

    def begin_policy(self,transition,source,passive):
        row=dict(kind='policy',transition=transition,source=dict(source or {}),passive=bool(passive),
                 start=time.perf_counter())
        if self.gpu_events:self.gpu_events[0].record()
        return row

    def tensor_ready(self,row):
        row['tensor_ready']=time.perf_counter()

    def forward_enqueued(self,row):
        if self.gpu_events:self.gpu_events[1].record()
        row['forward_returned']=time.perf_counter()

    def policy_ready(self,row):
        row['end']=time.perf_counter()
        # On ordinary policy steps actions.cpu() already waits on this stream.
        # Passive steps may still be asynchronous: query, never add a fence.
        row['gpu_stream_ms']=(float(self.gpu_events[0].elapsed_time(self.gpu_events[1]))
            if self.gpu_events and self.gpu_events[1].query() else None)
        self.record(row)

    def save(self):
        with (self.output/'pipeline_profile.jsonl').open('x',encoding='utf8') as handle:
            for row in sorted(self.events,key=lambda row:row['start']):
                handle.write(json.dumps(row)+'\n')
        (self.output/'pipeline_profile_metadata.json').write_text(json.dumps(dict(
            events=len(self.events),dropped_events=self.dropped_events,
            gpu_events=self.gpu_events is not None,
            note='Host timings and optional CUDA stream elapsed time. No per-decision GPU synchronize. '
                 'Paint wait includes rendering/readback/scheduling; pure GPU readback is not isolated.'),indent=2),encoding='utf8')
