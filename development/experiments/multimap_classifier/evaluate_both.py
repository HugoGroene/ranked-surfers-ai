"""Sequential visible local evaluation; stateful guard and candidate classifier."""
import argparse
import json
from datetime import datetime
from functools import partial
import os
from pathlib import Path
import subprocess
import sys
import threading
from record import ROOT,MAPS
from classifier_config import classifier_path

def latest_checkpoint():
    for run in sorted((ROOT/'reports').glob('multimap_stability_*'),reverse=True):
        plan_file=run/'plan.json'
        if plan_file.exists():
            plan=json.loads(plan_file.read_text())
            if plan.get('diagnostic_only') or plan.get('visible_worker') is not None:continue
        report=run/'stability/train/parallel_training.json'
        checkpoint=run/'stability/train/latest.zip'
        recovery_file=run/'recovery.json'
        if recovery_file.exists():
            recovery=json.loads(recovery_file.read_text())
            if recovery.get('status')!='COMPLETE':continue
            checkpoint=Path(recovery['latest_checkpoint'])
            report=checkpoint.parent/'parallel_training.json'
        if not report.is_file() or not checkpoint.is_file():continue
        value=json.loads(report.read_text())
        if (value.get('status')=='COMPLETE' and not value.get('error') and
                not value.get('finalization_errors') and
                value.get('final_checkpoint_transitions') is not None and
                value['final_checkpoint_transitions']==value.get('optimized_transitions')):
            return checkpoint
    raise RuntimeError('No completed multimap checkpoint found; supply --checkpoint')

def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--episodes',type=int,default=15)
    p.add_argument('--headless',action='store_true',help='Use the training GPU/shared-memory capture path instead of a visible desktop window')
    p.add_argument('--maps',choices=tuple(MAPS),nargs='+',default=list(MAPS))
    p.add_argument('--checkpoint',type=Path)
    p.add_argument('--check-only',action='store_true')
    p.add_argument('--state-model',type=Path)
    p.add_argument('--reuse-pump-crop',action=argparse.BooleanOptionalAction,default=True)
    p.add_argument('--threaded-preprocess',action=argparse.BooleanOptionalAction,default=True)
    p.add_argument('--defer-startup-previews',action=argparse.BooleanOptionalAction,default=True)
    a=p.parse_args()
    if not 1<=a.episodes<=10000:p.error('episodes must be 1..10000')
    checkpoint=(a.checkpoint or latest_checkpoint()).resolve()
    print('EVALUATING CHECKPOINT:',checkpoint,flush=True)
    model=classifier_path(a.state_model)
    env=dict(os.environ,RS_STATEFUL_GUARD='1',RS_VISIBLE_CAPTURE='desktop',
             RS_REUSE_PUMP_CROP='1' if a.reuse_pump_crop else '0',
             RS_DEFER_STARTUP_PREVIEWS='1' if a.defer_startup_previews else '0',
             RS_PUMP_PREPROCESS_THREAD='1' if a.threaded_preprocess else '0')
    output=ROOT/'reports'/datetime.now().strftime(('multimap_headless_' if a.headless else 'multimap_visible_')+'%Y%m%d_%H%M%S_%f')
    runner=ROOT/'development/experiments/command_history/evaluate.py'
    def command(name):
        return [sys.executable,'-B','-s','-u',str(runner),'--layout','fixed4',
            '--checkpoint',str(checkpoint),'--state-model',str(model),
            '--output',str(output/name),'--episodes',str(a.episodes),'--minutes','180',
            *([] if a.headless else ['--visible-browser'])]
    subprocess.run(command('preflight')+['--check-only'],cwd=ROOT,env=env,check=True)
    if a.check_only:return
    sys.path.insert(0,str(ROOT/'development/staged/map_import_20261102'))
    from record_validation import Handler,ThreadingHTTPServer
    for name in dict.fromkeys(a.maps):
        folder=MAPS[name]
        print('HEADLESS EVALUATION:' if a.headless else 'VISIBLE EVALUATION:',name,a.episodes,
              'episodes'+('' if a.headless else '; keep game foreground'),flush=True)
        server=ThreadingHTTPServer(('127.0.0.1',8000),partial(Handler,directory=str(folder)))
        threading.Thread(target=server.serve_forever,daemon=True).start()
        try:subprocess.run(command(name),cwd=ROOT,env=env,check=True)
        finally:server.shutdown();server.server_close()
    print('EVALUATIONS COMPLETE:',output)

if __name__=='__main__':main()
