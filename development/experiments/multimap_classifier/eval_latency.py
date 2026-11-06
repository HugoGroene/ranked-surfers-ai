"""Bounded, capture-free evaluation timing; no changes to actions or guard decisions."""
from collections import deque
import json
import time
from pathlib import Path
from tools.ppo_reaction_metrics import command_latency,latency_summary


def install(trainer):
    original=trainer.create_live_env
    def create(*args,**kwargs):
        monitored,raw=original(*args,**kwargs)
        step,close=monitored.step,monitored.close
        samples=deque(maxlen=20000);intervals=deque(maxlen=20000)
        previous=[None];counts=dict(decisions=0,commands=0)
        def measured(action):
            now=time.perf_counter();episode=raw.episode_number
            if previous[0] is not None and previous[0][0]==episode:
                intervals.append(dict(decision_interval_ms=(now-previous[0][1])*1000))
            previous[0]=(episode,now);counts['decisions']+=1
            source=dict(getattr(raw,'_learning_observation_capture',{}) or {})
            before=raw.step_timings[-1] if raw.step_timings else None
            result=step(action)
            after=raw.step_timings[-1] if raw.step_timings else None
            if after is not None and after is not before:
                value=command_latency(source,getattr(raw,'_reaction_guard_capture',{}),
                    after.get('press_timestamp'),after.get('press_roundtrip_ms'))
                if value:samples.append(value);counts['commands']+=1
            return result
        def finish():
            try:return close()
            finally:
                report=dict(**counts,latency=latency_summary(samples),cadence=latency_summary(intervals),
                    retained_limit=20000,note='Command issue, not Unity response. Command timings include startup/reset '
                    'but omit NONE/no-dispatch steps. Cadence excludes across-episode reset gaps. No extra captures.')
                dest=Path(raw.output_dir)/'evaluation_latency.json'
                with dest.open('x',encoding='utf8') as f:json.dump(report,f,indent=2)
                print('EVALUATION TIMING:',dest,flush=True)
        monitored.step,monitored.close=measured,finish
        return monitored,raw
    trainer.create_live_env=create
