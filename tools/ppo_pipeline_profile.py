"""Opt-in buffered pipeline timings. No extra captures, actions or per-step fences."""
import json
import math
from collections import deque
from pathlib import Path
import time
import types


class PipelineProfiler:
    def __init__(self,raw,output,device='cpu',max_events=100000):
        self.raw,self.output=raw,Path(output)
        self.events=[]
        self.max_events=max_events
        self.dropped_events=0
        self.recent_events=deque(maxlen=256)
        self.spike_count=0
        self.spike_context_count=0
        self.last_spike_context_at=float('-inf')
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
        # Continue retaining recent context after the initial full profile fills.
        self.recent_events.append(row)
        if len(self.events)<self.max_events:self.events.append(row)
        else:self.dropped_events+=1

    def command_diagnostic(self,policy,latency,dispatch_at,episode_start=False):
        """Attach rare delays to the already-persisted transition ledger.

        No extra file writes/captures/fences. Context is bounded and rate limited;
        every >100ms command still gets a compact record, for the entire run.
        """
        delay=(latency or {}).get('policy_request_to_command_ms')
        if (dispatch_at is None or delay is None or not math.isfinite(delay)
                or delay<=100 or policy.get('passive')):
            return None
        self.spike_count+=1
        source=policy.get('source') or {}
        result=dict(threshold_ms=100,latency_ms=delay,
            phase='episode_start' if episode_start else 'ordinary',
            policy_start=policy['start'],policy_end=policy['end'],dispatch_at=dispatch_at,
            policy_host_ms=(policy['end']-policy['start'])*1000,
            policy_ready_to_command_ms=(dispatch_at-policy['end'])*1000,
            latency=dict(latency),context_included=False)
        for field in ('request_started_at','image_received_at','captured_at'):
            if source.get(field) is not None:
                result[field+'_to_policy_ms']=(policy['start']-source[field])*1000
        if dispatch_at-self.last_spike_context_at>=10:
            self.last_spike_context_at=dispatch_at
            self.spike_context_count+=1
            # deque.copy is a bounded snapshot; capture callbacks may append.
            result['context']=[{k:v for k,v in row.items() if k in (
                'kind','start','end','transition','frame_serial','tensor_ready',
                'forward_returned','gpu_stream_ms','passive')}
                for row in self.recent_events.copy()]
            result['context_included']=True
        return result

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
            latency_spikes=self.spike_count,spike_contexts=self.spike_context_count,
            spike_threshold_ms=100,spike_context_interval_seconds=10,
            spike_location='learning_transitions.jsonl: latency_spike (continues after profile fills)',
            gpu_events=self.gpu_events is not None,
            note='Host timings and optional CUDA stream elapsed time. No per-decision GPU synchronize. '
                 'Paint wait includes rendering/readback/scheduling; pure GPU readback is not isolated.'),indent=2),encoding='utf8')
