"""Sequential frozen-policy detector ablation. User runs this after training."""
import argparse
import hashlib
import json
from pathlib import Path
import subprocess
import sys


def command(args,mode,folder):
    result=[sys.executable,'-B','-s','-u','-m','tools.train_ppo_parallel','--benchmark-only',
        '--headless','--headless-capture','electron','--electron-transport','shared_memory',
        '--displays',*[str(i+1) for i in range(args.workers)],
        '--framing-profile','canvas_v1','--canvas-policy-input','--reuse-frame-state',
        '--checkpoint',str(args.checkpoint),'--state-model',str(args.state_model),
        '--detector',str(args.detector),'--detector-mode',mode,
        '--steps','768000','--max-minutes',str(args.minutes),'--worker-timeout','160',
        '--telemetry','--report',str(folder)]
    for name in ['compact_observations','cache_detector_features','shared_inference']:
        if getattr(args,name):result.append('--'+name.replace('_','-'))
    result+=['--rollout-transport',args.rollout_transport,'--electron-host',args.electron_host]
    if args.opencv_threads is not None:result+=['--opencv-threads',str(args.opencv_threads)]
    return result


def main(argv=None):
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--checkpoint',type=Path,required=True)
    parser.add_argument('--state-model',type=Path,default=Path('checkpoints/v612_snapshot_20261021/screen_state_candidate_v2.pt'))
    parser.add_argument('--detector',type=Path,default=Path('ai_data/v5_hazards/runs/segment_round04/weights/best.pt'))
    parser.add_argument('--workers',type=int,default=4)
    parser.add_argument('--minutes',type=float,default=1.)
    parser.add_argument('--order',nargs='+',choices=['on','off'],default=['on','off','off','on'])
    parser.add_argument('--rollout-transport',choices=['pipe','shared_memory'],default='pipe')
    parser.add_argument('--compact-observations',action='store_true')
    parser.add_argument('--cache-detector-features',action='store_true')
    parser.add_argument('--shared-inference',action='store_true')
    parser.add_argument('--opencv-threads',type=int,default=None)
    parser.add_argument('--electron-host',choices=['per_worker','shared'],default='per_worker')
    parser.add_argument('--report',type=Path,required=True)
    args=parser.parse_args(argv)
    if not 1<=args.workers<=6 or not .25<=args.minutes<=5:parser.error('Use 1..6 workers and .25..5 minutes per stage')
    if args.compact_observations and args.rollout_transport!='shared_memory':parser.error('Compact observations require shared-memory rollouts')
    if not {'on','off'}.issubset(args.order):parser.error('Include both detector on and off')
    for asset in [args.checkpoint,args.state_model,args.detector]:
        if not asset.is_file():parser.error('Missing asset: '+str(asset))
    if args.report.exists():parser.error('Choose a new report directory')
    digest=hashlib.sha256(args.checkpoint.read_bytes()).hexdigest()
    args.report.mkdir(parents=True)
    summary=dict(status='RUNNING',source_sha256=digest,model_updated=False,stages=[],
        note='Stochastic frozen-policy ablation, including collection barriers but no optimization. Different detector inputs can change trajectories. This is not a 100-episode evaluation or training throughput.')
    try:
        for i,mode in enumerate(args.order):
            folder=args.report/f'{i+1:02d}_detector_{mode}'
            argv=command(args,mode,folder)
            print('Detector '+mode+' stage '+str(i+1),flush=True)
            with (args.report/f'{i+1:02d}_{mode}.log').open('w',encoding='utf8') as log:
                completed=subprocess.run(argv,stdout=log,stderr=subprocess.STDOUT)
            result=json.loads((folder/'parallel_training.json').read_text())
            summary['stages'].append(dict(detector_mode=mode,report=str(folder),result=result))
            if (completed.returncode or result['status']!='COMPLETE_FROZEN_POLICY_DIAGNOSTIC' or
                    result['optimized_transitions'] or not result.get('policy_unchanged') or not result['source_unchanged']):
                raise RuntimeError('Diagnostic failed; inspect '+str(folder))
        summary['status']='COMPLETE'
    except BaseException as exc:
        summary.update(status='STOPPED',error=repr(exc))
        raise
    finally:
        summary['source_unchanged']=hashlib.sha256(args.checkpoint.read_bytes()).hexdigest()==digest
        (args.report/'comparison.json').write_text(json.dumps(summary,indent=2),encoding='utf8')
    print(json.dumps(summary,indent=2))


if __name__=='__main__':main()
