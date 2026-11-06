"""Passive classifier diagnostics; never controls keys or approves BC samples."""
from collections import Counter
import json
from pathlib import Path


class GuardMonitor:
    def __init__(self,session):
        self.session=Path(session)
        self.log=(self.session/'guard_observations.jsonl').open('x',encoding='utf8')
        self.states=Counter();self.active_states=Counter();self.runs={}
        self.count=0;self.errors=0;self.last_flush=float('-inf')
        self.last_warning=float('-inf');self.last_at=None

    def observe(self,*,at,state,probabilities,run=None,run_age=None,frame=None,
                capture_ms=None,classifier_ms=None,error=None):
        self.count+=1;self.states[state]+=1
        reason=None
        if run:
            self.active_states[state]+=1
            counts=self.runs.setdefault(run,Counter());counts[state]+=1
            if state!='GAMEPLAY':
                reason=('menu_during_candidate_review_death_or_false_positive'
                        if state=='MENU' else 'non_gameplay_during_candidate')
        if error:self.errors+=1;reason='classifier_exception'
        row=dict(at=at,state=state,probabilities=probabilities,run=run,run_age_seconds=run_age,
            frame=frame,warning=reason,error=error,capture_ms=capture_ms,classifier_ms=classifier_ms,
            capture_interval_ms=None if self.last_at is None else (at-self.last_at)*1000)
        self.last_at=at
        self.log.write(json.dumps(row,allow_nan=False)+'\n')
        if at-self.last_flush>=1:
            self.log.flush();self.last_flush=at
        if reason and at-self.last_warning>=5:
            self.last_warning=at
            print(f'[BC STATE CHECK] {run or "outside candidate"} age={run_age} state={state}; '
                  'saved for review, not an approved label or PPO guard verdict.',flush=True)
        return row

    def close(self):
        self.log.close()
        result=dict(frames=self.count,states=dict(self.states),
            states_during_candidates=dict(self.active_states),classifier_errors=self.errors,
            runs={k:dict(v) for k,v in self.runs.items()},
            note='Passive raw classifier monitoring, not full PPO guard replay. MENU at death and '
                 'OTHER at pause can be correct. Inspect frame references before correcting labels. '
                 'No training guard thresholds or automatic inputs changed.')
        path=self.session/'guard_summary.json'
        with path.open('x',encoding='utf8') as f:json.dump(result,f,indent=2)
        print('CLASSIFIER DIAGNOSTICS:',path,flush=True)
