"""Terminal-only stage progress, using existing stdout; no game queries."""
from collections import Counter
import re
import time

def clock_text(seconds):
    seconds=max(0,int(seconds))
    return f'{seconds//3600:02d}:{seconds//60%60:02d}:{seconds%60:02d}'

class BatchProgress:
    def __init__(self,total,*,clock=time.monotonic,emit=print):
        self.total=total;self.clock=clock;self.emit=emit
        self.started=clock();self.last_print=float('-inf')
        self.phase='';self.counts=Counter();self.stage_started=None;self.episodes=None

    def __call__(self,job,event,line=None):
        now=self.clock()
        if event=='START':self.stage_started=now;self.episodes=None
        if event=='OUTPUT':
            if line and '[V61 EVAL] episode ' in line:
                match=re.search(r'episode (\d+)/(\d+)',line)
                if match:self.episodes=tuple(map(int,match.groups()))
            if now-self.last_print<30:return
        if event in ('PASSED','FAILED','SKIPPED','RETAINED'):self.counts[event]+=1
        finished=sum(self.counts.values())
        active=event in ('START','OUTPUT')
        command=job['command']
        detail=''
        if active:
            stage_elapsed=now-self.stage_started
            if job['mode']=='train':
                minutes=float(command[command.index('--minutes')+1])
                detail=f' stage_wall={clock_text(stage_elapsed)} training_budget={minutes:g}m (excludes startup)'
            else:
                total=int(command[command.index('--episodes')+1])
                current=self.episodes[0] if self.episodes else 0
                detail=f' episodes={current}/{total} stage_wall={clock_text(stage_elapsed)}'
        self.emit(f'[BATCH PROGRESS] stages={finished}/{self.total} finished '
            f'passed={self.counts["PASSED"]} retained={self.counts["RETAINED"]} '
            f'failed={self.counts["FAILED"]} skipped={self.counts["SKIPPED"]} '
            f'elapsed={clock_text(now-self.started)} | {self.phase} {job["mode"]} '
            f'{job["name"]} {event if event!="OUTPUT" else "RUNNING"}{detail}',flush=True)
        self.last_print=now
