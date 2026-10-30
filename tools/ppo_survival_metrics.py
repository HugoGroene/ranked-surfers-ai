"""Background dashboard aggregation; never changes rewards or PPO collection."""
from collections import deque
import json
import math
from pathlib import Path
import threading
import time


def survival_curve(events, started, requested_seconds, window=100, interval=10, elapsed_seconds=None):
    valid = sorted((e for e in events if math.isfinite(e['duration']) and e['duration'] >= 0
                    and math.isfinite(e['at']) and e['at'] >= started), key=lambda e:e['at'])
    recent = deque(maxlen=window)
    points = []
    horizon=min(requested_seconds,elapsed_seconds if elapsed_seconds is not None else
                max((e['at']-started for e in valid),default=0))
    boundaries=list(range(interval,int(horizon)+1,interval))
    if horizon >= requested_seconds and (not boundaries or boundaries[-1] != requested_seconds):
        boundaries.append(requested_seconds)
    index=0
    for boundary in boundaries:
        while index < len(valid) and valid[index]['at']-started <= boundary:
            recent.append(valid[index]['duration'])
            index+=1
        if recent:
            points.append(dict(seconds=boundary,mean_seconds=sum(recent)/len(recent),samples=len(recent)))
    truncated=sum(e.get('reason') == 'time_limit' for e in valid)
    return dict(requested_seconds=requested_seconds,window_episodes=window,interval_seconds=interval,
                completed_episodes=len(valid),time_limited_episodes=truncated,
                overall_mean_seconds=sum(e['duration'] for e in valid)/len(valid) if valid else None,
                best_seconds=max((e['duration'] for e in valid),default=None),
                points=points,metric='Rolling mean of last 100 completed episodes across all workers; immutable completed-time samples; active reward duration; time-limited episodes included')


class SurvivalMetrics:
    def __init__(self,folder,workers,started,requested_seconds):
        self.folder,self.workers,self.started,self.requested = Path(folder),workers,started,requested_seconds
        self.stop = threading.Event()
        self.points = {}
        self.thread = threading.Thread(target=self.run,daemon=True,name='survival-metrics')

    def update(self):
        events = []
        for worker in range(self.workers):
            path = self.folder/f'worker_{worker}'/'survival_episodes.jsonl'
            if not path.exists():
                continue
            for line in path.read_text(encoding='utf8').splitlines(keepends=True):
                if not line.endswith('\n'):
                    continue
                try:
                    events.append(json.loads(line))
                except (ValueError,TypeError):
                    continue
        result = survival_curve(events,self.started,self.requested,elapsed_seconds=time.perf_counter()-self.started)
        for point in result['points']:
            self.points.setdefault(point['seconds'],point)
        result['points']=[self.points[key] for key in sorted(self.points)]
        tmp = self.folder/'survival.partial.json'
        tmp.write_text(json.dumps(result),encoding='utf8')
        tmp.replace(self.folder/'survival.json')

    def run(self):
        while not self.stop.wait(5):
            try:
                self.update()
            except OSError:
                pass

    def start(self):
        self.update()
        self.thread.start()

    def close(self):
        self.stop.set()
        self.thread.join(timeout=6)
        self.update()
