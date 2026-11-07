"""Bounded, capture-free evaluation timing; no changes to actions or guard decisions."""
from collections import deque
import json
import time
from pathlib import Path
from tools.ppo_reaction_metrics import command_latency,latency_summary


def install(trainer):
    if getattr(trainer,'_evaluation_latency_installed',False):return
    trainer._evaluation_latency_installed=True
    original=trainer.create_live_env
    def create(*args,**kwargs):
        monitored,raw=original(*args,**kwargs)
        step,close=monitored.step,monitored.close
        samples=deque(maxlen=20000);intervals=deque(maxlen=20000);traces=deque(maxlen=20000)
        previous=[None];counts=dict(decisions=0,commands=0)
        episode_state=dict(number=None,step=0,movement=False,first=None,previous_action=None)
        pending_prediction=[None]
        from tools import ppo_eval_prediction
        predict=ppo_eval_prediction.predict_with_optional_probabilities
        def timed_prediction(*args,**kwargs):
            began=time.perf_counter()
            source=dict(getattr(raw,'_learning_observation_capture',{}) or {})
            try:return predict(*args,**kwargs)
            finally:pending_prediction[0]=dict(start=began,end=time.perf_counter(),source=source)
        ppo_eval_prediction.predict_with_optional_probabilities=timed_prediction
        guard_fields=('guard_rechecks','prepared_guard_hits','prepared_guard_fallbacks','prepared_guard_late_fallbacks')
        def measured(action):
            now=time.perf_counter();episode=raw.episode_number
            if episode_state['number']!=episode:
                episode_state.update(number=episode,step=0,movement=False,previous_action=None)
                if episode_state['first'] is None:episode_state['first']=episode
            episode_state['step']+=1
            phase=('startup' if episode==episode_state['first'] else 'reset') if episode_state['step']==1 else 'ordinary'
            if previous[0] is not None and previous[0][0]==episode:
                intervals.append(dict(decision_interval_ms=(now-previous[0][1])*1000))
            previous[0]=(episode,now);counts['decisions']+=1
            source=dict(getattr(raw,'_learning_observation_capture',{}) or {})
            before=raw.step_timings[-1] if raw.step_timings else None
            guard_before={name:getattr(raw,name,0) for name in guard_fields}
            waiter=getattr(raw,'_latency_waiter',None)
            waits_before=waiter.calls if waiter is not None else 0
            prediction=pending_prediction[0];pending_prediction[0]=None
            result=step(action)
            after=raw.step_timings[-1] if raw.step_timings else None
            value=None
            if after is not None and after is not before:
                value=command_latency(source,getattr(raw,'_reaction_guard_capture',{}),
                    after.get('press_timestamp'),after.get('press_roundtrip_ms'))
                if value:samples.append(value);counts['commands']+=1
            trace=dict(episode=episode,episode_step=episode_state['step'],phase=phase,action=int(action),
                previous_action=episode_state['previous_action'],first_movement=bool(value and not episode_state['movement']),
                step_started_at=now,step_finished_at=time.perf_counter(),prediction=prediction,
                policy_source=source,guard_source=dict(getattr(raw,'_reaction_guard_capture',{}) or {}),
                latency=value,dispatch_at=after.get('press_timestamp') if after is not None and after is not before else None,
                guard_deltas={name:getattr(raw,name,0)-guard_before[name] for name in guard_fields})
            trace['rate_wait']=dict(waiter.rows[-1]) if waiter is not None and waiter.calls>waits_before else None
            if prediction is not None:
                trace['inference_ms']=(prediction['end']-prediction['start'])*1000
                trace['inference_ready_to_step_ms']=(now-prediction['end'])*1000
            traces.append(trace)
            if value:episode_state['movement']=True
            episode_state['previous_action']=int(action)
            return result
        def finish():
            try:return close()
            finally:
                if ppo_eval_prediction.predict_with_optional_probabilities is timed_prediction:
                    ppo_eval_prediction.predict_with_optional_probabilities=predict
                groups={phase:latency_summary([t['latency'] for t in traces if t['phase']==phase and t['latency']])
                        for phase in ('startup','reset','ordinary')}
                groups['first_movement']=latency_summary([t['latency'] for t in traces if t['first_movement']])
                groups['ordinary_after_none']=latency_summary([t['latency'] for t in traces
                    if t['phase']=='ordinary' and t['previous_action']==0 and t['latency']])
                report=dict(**counts,latency=latency_summary(samples),cadence=latency_summary(intervals),
                    phase_latency=groups,trace_samples=len(traces),
                    retained_limit=20000,note='Command issue, not Unity response. Command timings include startup/reset '
                    'but omit NONE/no-dispatch steps. Cadence excludes across-episode reset gaps. No extra captures.')
                dest=Path(raw.output_dir)/'evaluation_latency.json'
                with (dest.parent/'evaluation_latency_trace.jsonl').open('x',encoding='utf8') as f:
                    for trace in traces:f.write(json.dumps(trace)+'\n')
                with dest.open('x',encoding='utf8') as f:json.dump(report,f,indent=2)
                print('EVALUATION TIMING:',dest,flush=True)
        monitored.step,monitored.close=measured,finish
        return monitored,raw
    trainer.create_live_env=create
