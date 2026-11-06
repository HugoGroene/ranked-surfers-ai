"""Short actual-PPO worker-count benchmark; no dashboard or input migration."""
import argparse
import json
import os
from pathlib import Path
import sys
import time

from parallel_adapter import ROOT,DEFAULT_CHECKPOINT,configuration,sources,install

# Windows multiprocessing imports this file as __mp_main__. Install the exact
# environment in each child before its production worker target is unpickled.
if __name__=='__main__':os.environ['RS_FIXED_SCALING_20261103']='1'
if os.environ.get('RS_FIXED_SCALING_20261103')=='1':install()


def command(checkpoint,output,workers,minutes):
    return ['--displays',*[str(i+1) for i in range(workers)],'--headless','--headless-capture','electron',
        '--electron-transport','shared_memory','--rollout-transport','shared_memory','--reuse-frame-state',
        '--opencv-threads','1','--profile-reaction','--checkpoint',str(checkpoint),
        '--detector-mode','off','--framing-profile','canvas_v1','--canvas-policy-input',
        '--worker-seed-base','123','--steps',str(128*workers*100000),'--max-minutes',str(minutes),
        '--save-every-minutes','2','--worker-timeout','180','--telemetry','--end-plot','off','--report',str(output)]


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--workers',type=int,choices=range(1,6),required=True)
    p.add_argument('--minutes',type=float,default=5)
    p.add_argument('--checkpoint',type=Path,required=True)
    p.add_argument('--output',type=Path,required=True)
    p.add_argument('--expected-sources',type=Path,required=True)
    a=p.parse_args()
    if not .25<=a.minutes<=15:p.error('Budget must be .25..15 minutes')
    a.output=a.output.resolve();a.checkpoint=a.checkpoint.resolve()
    if not a.output.is_relative_to(ROOT/'reports') or a.output.exists():p.error('Use a new reports folder')
    expected=json.loads(a.expected_sources.read_text())
    if expected!=sources():p.error('Source manifest changed')
    from tools import train_ppo_parallel as parallel
    original_atomic=parallel.atomic_json
    last_round=[0]
    def atomic(path,data):
        original_atomic(path,data)
        if Path(path).name=='parallel_training.json' and data.get('rounds',0)>last_round[0]:
            runtime_path=a.output/'runtime.json'
            if runtime_path.exists():
                runtime=json.loads(runtime_path.read_text())
                if runtime.get('rounds')==data['rounds']:
                    row={key:runtime.get(key) for key in ('rounds','collection_seconds','update_seconds',
                        'elapsed_seconds','optimized_per_second','processed_per_second')}
                    with (a.output/'round_timings.jsonl').open('a',encoding='utf8') as handle:
                        handle.write(json.dumps(row)+'\n')
                    last_round[0]=data['rounds']
    parallel.atomic_json=atomic
    os.chdir(ROOT)
    started=time.perf_counter()
    try:parallel.main(command(a.checkpoint,a.output,a.workers,a.minutes))
    finally:
        if a.output.exists():
            (a.output/'hybrid_benchmark.json').write_text(json.dumps(dict(configuration=configuration(),
                requested_workers=a.workers,requested_minutes=a.minutes,decision_rate_mode='capped',
                capture_backend='gpu_100',torch_threads=1,opencv_threads=1,dashboard=False,
                profiled=True,input_previews=os.environ.get('RS_FIXED_INPUTS')=='1',total_stage_seconds=time.perf_counter()-started,
                source_unchanged=sources()==expected),indent=2))
    if sources()!=expected:raise RuntimeError('Source changed during probe')
    for i in range(a.workers):
        worker=a.output/f'worker_{i}'
        if json.loads((worker/'hybrid_close.json').read_text()).get('status')!='COMPLETE':
            raise RuntimeError(f'Worker {i} did not close cleanly')
        if json.loads((worker/'result.json').read_text()).get('error'):
            raise RuntimeError(f'Worker {i} reported an error')


if __name__=='__main__':main()
