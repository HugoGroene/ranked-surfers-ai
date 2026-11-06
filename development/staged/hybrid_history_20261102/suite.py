"""Short smoke/latency tests, never a model-quality selection sweep."""
import argparse
from datetime import datetime
import html
import json
import os
from pathlib import Path
import random
import subprocess
import sys
from urllib.parse import quote

from run_probe import HERE,ROOT,ROAD,sources
from hybrid import LAYOUTS
import batch as engine
from batch_progress import BatchProgress

def read(path):return json.loads(Path(path).read_text())
def write(path,value):
    with Path(path).open('x',encoding='utf-8') as f:json.dump(value,f,indent=2)

def jobs(cases,folder,minutes,smoke):
    result=[]
    for size,layout in cases:
        name=f'road{size}_{layout}'
        common=[sys.executable,'-B','-s','-u',str(HERE/'run_probe.py'),
                '--size',str(size),'--layout',layout,'--seed','123',
                '--expected-sources',str(folder/'sources.json')]
        train=folder/name/'train'
        result.append({'name':name,'mode':'train','output':str(train),
            'command':common+['--mode','train','--minutes',str(minutes),'--output',str(train)]+
                      (['--input-previews'] if smoke else [])})
    if smoke:
        for size,layout in cases:
            name=f'road{size}_{layout}';train=folder/name/'train';evaluation=folder/name/'eval'
            common=[sys.executable,'-B','-s','-u',str(HERE/'run_probe.py'),
                    '--size',str(size),'--layout',layout,'--seed','123',
                    '--expected-sources',str(folder/'sources.json')]
            result.append({'name':name,'mode':'eval','output':str(evaluation),
                'command':common+['--mode','eval','--minutes','3','--episodes','3',
                                   '--checkpoint',str(train/'latest.zip'),'--output',str(evaluation)]})
    return result

def summary(output):
    result=[]
    gallery=['<!doctype html><meta charset="utf-8"><title>Actual policy inputs</title>',
             '<style>body{font-family:sans-serif;background:#202124;color:#eee}img{max-width:100%;height:auto}a{color:#8ab4f8}section{margin:2em 0}</style>',
             '<h1>Actual policy inputs from smoke rollouts</h1><p>Raw PNGs and NPZ tensors preserve the actor observations. '
             'Labels are outside the contact-sheet pixels. Ages use capture-request time, not Unity time.</p>']
    for phase in ('smoke','performance'):
        plan=read(output/phase/'plan.json')
        for job in plan['jobs']:
            if job['mode']!='train':continue
            folder=Path(job['output']);file=folder/'trial_report.json'
            if not file.exists():continue
            r=read(file);h=read(folder/'hybrid_history.json') if (folder/'hybrid_history.json').exists() else {}
            row={'phase':phase,'name':job['name'],'status':r.get('status'),
                 'audit':r.get('audit',{}).get('status'),'optimized_s':r.get('optimized_per_second'),
                 'guard_hard_stopped':r.get('guard_hard_stopped'),
                 'reaction_latency':r.get('reaction_latency'),'history':h}
            capture=folder/'temporal_capture.json'
            row['capture']=read(capture) if capture.exists() else None
            usage=folder/'system_usage.jsonl'
            row['resource_samples_file']=str(usage) if usage.exists() else None
            evidence=folder/'policy_inputs'
            row['input_evidence']=str(evidence) if (evidence/'index.json').exists() else None
            if row['input_evidence']:
                for sample in read(evidence/'index.json')['samples']:
                    name=sample['name']
                    def link(file):return quote(Path(os.path.relpath(file,output)).as_posix(),safe='/')
                    gallery.append(f'<section><h2>{html.escape(job["name"])} / {html.escape(name)}</h2>'
                        f'<p>Actor action: {sample["action"]}; episode start: {sample["episode_start"]}. '
                        f'<a href="{link(evidence/(name+".json"))}">Metadata</a> | '
                        f'<a href="{link(evidence/(name+".npz"))}">Exact input arrays</a></p>'
                        f'<img loading="lazy" src="{link(evidence/(name+"_sheet.png"))}"></section>')
            result.append(row)
    (output/'performance_summary.json').write_text(json.dumps({'rows':result,
        'note':'Short fresh-model probes; different random behavior may affect timing. No learning winner, concurrency claim or automatic default change.'},indent=2))
    (output/'policy_input_gallery.html').write_text('\n'.join(gallery),encoding='utf8')
    table=['# Hybrid performance probes','',
           'Single-worker fresh-model probes. Command timing is not Unity response time. '
           'Inspect latency sample counts, audits and history availability in the JSON before choosing a configuration.','',
           '| Configuration | Status | Optimized/s | Request-to-command median ms | p95 ms | Measured capture Hz |',
           '|---|---|---:|---:|---:|---:|']
    def fmt(value):return f'{value:.2f}' if isinstance(value,(int,float)) else 'n/a'
    for row in result:
        if row['phase']!='performance':continue
        latency=(row.get('reaction_latency') or {}).get('metrics',{}).get('policy_request_to_command_ms',{})
        table.append(f'| {row["name"]} | {row["status"]} | {fmt(row["optimized_s"])} | '
                     f'{fmt(latency.get("median_ms"))} | {fmt(latency.get("p95_ms"))} | '
                     f'{fmt((row.get("capture") or {}).get("achieved_capture_hz"))} |')
    (output/'performance_summary.md').write_text('\n'.join(table),encoding='utf8')

def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--execute',action='store_true')
    p.add_argument('--minutes',type=float,default=4)
    p.add_argument('--sizes',type=int,nargs='+',choices=[256,384],default=[256,384])
    p.add_argument('--layouts',nargs='+',choices=list(LAYOUTS),default=list(LAYOUTS))
    p.add_argument('--resume',type=Path)
    p.add_argument('--plan',type=Path)
    a=p.parse_args()
    if not 1<=a.minutes<=10:p.error('Short performance budget must be 1..10 minutes')
    engine.source_hashes=sources
    if a.plan:
        output=a.plan.resolve().parent;plan=read(a.plan)
    else:
        output=ROOT/'reports'/datetime.now().strftime('hybrid_short_%Y%m%d_%H%M%S_%f')
        cases=[(s,l) for s in a.sizes for l in a.layouts]
        random.Random(218).shuffle(cases)
        plan={'cases':cases,'performance_minutes':a.minutes,'workers':1,
              'note':'All smoke training + checkpoint reload evaluations precede performance probes.'}
        if a.resume:
            old=a.resume.resolve()
            if read(old/'sources.json')!=sources():p.error('Resume requires identical sources')
            plan=read(old/'plan.json')
        output.mkdir(parents=True,exist_ok=False);write(output/'sources.json',sources())
        write(output/'plan.json',plan)
        for phase in ('smoke','performance'):
            folder=output/phase;folder.mkdir();write(folder/'sources.json',sources())
            if a.resume:
                phase_plan=engine.continue_smoke(old/phase/'plan.json',folder,same_sources=True)
            else:
                phase_plan={'jobs':jobs(cases,folder,1 if phase=='smoke' else a.minutes,phase=='smoke')}
            write(folder/'plan.json',phase_plan)
    print('HYBRID SUITE:',output/'plan.json',flush=True)
    print('Cases:',len(plan['cases']),'| one worker, detector off, spatial encoder, GPU capture',flush=True)
    print('Scheduled training:',len(plan['cases'])*(1+plan.get('performance_minutes',4)),
          'minutes, plus smoke evaluation/startup/finalization.',flush=True)
    if not a.execute:
        print('Plan only: no live tests started.');return
    for directory,pattern in ((HERE,'test_*.py'),(ROAD,'test_*.py')):
        subprocess.run([sys.executable,'-B','-s','-m','unittest','discover','-s',str(directory),
                        '-p',pattern,'-q'],cwd=ROOT,check=True)
    for layout in ('recent4_context4','sparse5','sparse4'):
        subprocess.run([sys.executable,'-B','-s',str(HERE/'offline_integration.py'),
                        '--layout',layout],cwd=ROOT,check=True)
    progress=BatchProgress(sum(len(read(output/phase/'plan.json')['jobs']) for phase in ('smoke','performance')))
    try:
        for phase in ('smoke','performance'):
            progress.phase=phase
            engine.execute(read(output/phase/'plan.json'),output/phase,
                           keep_going=phase=='performance',progress=progress)
    finally:
        summary(output)
        print('SUMMARY:',output/'performance_summary.json',flush=True)
        print('POLICY INPUT GALLERY:',output/'policy_input_gallery.html',flush=True)

if __name__=='__main__':main()
