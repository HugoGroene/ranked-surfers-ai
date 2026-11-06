"""Compare offscreen caps with identical starting weights and actual PPO updates."""
import argparse
from datetime import datetime
import hashlib
import json
import os
from pathlib import Path
import subprocess
import sys

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[2]

def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--checkpoint', type=Path, default=ROOT/'reports/command_history_resume_20261104_093033/train/latest.zip')
    p.add_argument('--minutes', type=float, default=3)
    p.add_argument('--execute', action='store_true')
    a = p.parse_args()
    if not .25 <= a.minutes <= 10: p.error('Minutes must be .25..10 per case')
    checkpoint = a.checkpoint.resolve()
    if not checkpoint.is_file(): p.error('Checkpoint not found')
    source = (HERE.parent/'capture_pipeline/host.cjs').read_text(encoding='utf8')
    replacements = {
        "require('./d3d_bridge.cjs')": "require('../capture_pipeline/d3d_bridge.cjs')",
        'win.webContents.setFrameRate(60);': "const testFPS=Number(process.env.RS_OFFSCREEN_TEST_FPS);\n  if (![30,60].includes(testFPS)) throw new Error('Invalid test FPS');\n  win.webContents.setFrameRate(testFPS);",
        'backend:gpu?"gpu":"bitmap",outputWidth': 'requested_fps:testFPS,configured_fps:win.webContents.getFrameRate(),backend:gpu?"gpu":"bitmap",outputWidth',
    }
    for old, new in replacements.items():
        if source.count(old) != 1: raise RuntimeError('Host changed; review isolated patch: '+old)
        source = source.replace(old, new)
    cases = [(60,1,.5),(30,1,.5)] + [(fps,4,a.minutes) for fps in (60,30,30,60)]
    print('Cases (FPS, workers, active minutes):', cases, flush=True)
    if not a.execute: return
    (HERE/'host.cjs').write_text(source, encoding='utf8')
    output = ROOT/'reports'/datetime.now().strftime('offscreen_fps_%Y%m%d_%H%M%S_%f')
    output.mkdir()
    digest = hashlib.sha256(checkpoint.read_bytes()).hexdigest()
    report = dict(checkpoint=str(checkpoint), checkpoint_sha256=digest,
                  host_sha256=hashlib.sha256(source.encode()).hexdigest(), cases=[])
    def save():
        temp = output/'summary.partial'
        temp.write_text(json.dumps(report, indent=2), encoding='utf8')
        temp.replace(output/'summary.json')
    save()
    print('REPORT:', output/'summary.json', flush=True)
    for i,(fps,workers,minutes) in enumerate(cases):
        name = f'{i+1:02d}_{fps}fps_w{workers}'
        dest = output/name
        env = dict(os.environ, RS_OFFSCREEN_TEST_FPS=str(fps), RS_COMPARISON_LAYOUT='fixed4')
        command = [sys.executable,'-B','-s','-u',str(HERE/'run_case.py'),
                   '--checkpoint',str(checkpoint),'--workers',str(workers),
                   '--minutes',str(minutes),'--output',str(dest)]
        print(f'[{i+1}/{len(cases)}] {name}; log: {output/(name+".log")}', flush=True)
        row = dict(name=name,fps=fps,workers=workers,minutes=minutes,status='RUNNING')
        report['cases'].append(row); save()
        with (output/(name+'.log')).open('w', encoding='utf8') as log:
            code = subprocess.call(command, cwd=ROOT, env=env, stdout=log, stderr=subprocess.STDOUT)
        row.update(exit_code=code,status='PASS' if code==0 else 'FAILED')
        result = dest/'train/parallel_training.json'
        if result.exists(): row['training'] = json.loads(result.read_text())
        ready = list(dest.glob('train/worker_*/electron/ready.json'))
        row['host_settings'] = [json.loads(f.read_text()) for f in ready]
        row['capture_metrics'] = [dict(worker=f.parent.name, **json.loads(f.read_text()))
                                  for f in dest.glob('train/worker_*/temporal_capture.json')]
        if code == 0 and (len(ready) != workers or any(
                h.get('configured_fps') != fps or not h.get('offscreen')
                for h in row['host_settings'])):
            row.update(status='FAILED', error='Offscreen FPS verification failed')
            code = 1
        save()
        print(name, row['status'], flush=True)
        if code and i < 2: break  # Fail fast on smoke; retain subsequent performance failures.
    report['source_unchanged'] = hashlib.sha256(checkpoint.read_bytes()).hexdigest()==digest
    save()
    print('REPORT:', output/'summary.json', flush=True)
    if any(r['status']!='PASS' for r in report['cases']): raise SystemExit(1)

if __name__=='__main__': main()
