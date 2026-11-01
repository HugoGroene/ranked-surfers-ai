"""Plan or execute sequential scratch trials, then matched deterministic evaluation."""
import argparse
from datetime import datetime
import json
from pathlib import Path
import random
import subprocess
import sys
import copy
import zipfile

from experiment_config import ROOT,HERE,PRESETS,source_hashes,file_hash


def jobs_for(configs,seeds,minutes,episodes,eval_minutes,output,order_seed):
    arms=[(name,seed) for seed in seeds for name in configs]
    random.Random(order_seed).shuffle(arms)
    training=[]
    evaluation=[]
    for name,seed in arms:
        common=['--frames',str(PRESETS[name]['frames']),
                '--span-ms',str(PRESETS[name]['span_ms']),
                '--capture-hz',str(PRESETS[name]['capture_hz']),
                '--decision-hz',str(PRESETS[name]['decision_hz']),'--seed',str(seed)]
        trial=f'{name}_seed{seed}'
        train_dir=output/trial/'train'
        eval_dir=output/trial/'eval'
        base=[sys.executable,'-B','-s','-u',str(HERE/'run_trial.py')]
        expected=['--expected-sources',str(output/'sources.json')]
        training.append(dict(name=trial,mode='train',output=str(train_dir),
            command=base+['--mode','train','--output',str(train_dir),'--minutes',str(minutes)]+common+expected))
        evaluation.append(dict(name=trial,mode='eval',output=str(eval_dir),
            command=base+['--mode','eval','--output',str(eval_dir),'--minutes',str(eval_minutes),
                         '--episodes',str(episodes),'--checkpoint',str(train_dir/'latest.zip')]+common+expected))
    return training+evaluation


def checked_result(job):
    folder=Path(job['output'])
    name='trial_report.json' if job['mode']=='train' else 'v61_eval_report.json'
    report=json.loads((folder/name).read_text())
    if report.get('status')!='COMPLETE':raise RuntimeError(f'{job["name"]}: incomplete {job["mode"]}')
    if job['mode']=='train':
        if report.get('audit',{}).get('status')!='PASS_SETTLED_LEARNING':raise RuntimeError('Learning audit failed')
        if not (folder/'latest.zip').is_file():raise RuntimeError('Checkpoint missing')
    elif not report.get('completed_requested_episodes'):raise RuntimeError('Evaluation count incomplete')
    return report


def summarize(plan):
    rows=[]
    for job in plan['jobs']:
        if job['mode']!='train':continue
        train_dir=Path(job['output'])
        row=dict(name=job['name'],training_status='NOT_RUN',evaluation_status='NOT_RUN')
        if (train_dir/'trial_report.json').exists():
            train=json.loads((train_dir/'trial_report.json').read_text())
            row.update(training_status=train['status'],optimized_transitions=train.get('optimized_transitions'),
                       optimized_per_second=train.get('optimized_per_second'),
                       training_seconds=train.get('training_seconds'),reaction_latency=train.get('reaction_latency'))
        capture_path=train_dir/'temporal_capture.json'
        if capture_path.exists():row['capture']=json.loads(capture_path.read_text())
        eval_job=next((j for j in plan['jobs'] if j['name']==job['name'] and j['mode']=='eval'),None)
        eval_path=(Path(eval_job['output']) if eval_job else train_dir.parent/'eval')/'v61_eval_report.json'
        if eval_path.exists():
            evaluation=json.loads(eval_path.read_text())
            row.update(evaluation_status=evaluation['status'],episodes=evaluation['episodes'],
                       mean_seconds=evaluation['mean_survival_seconds'],median_seconds=evaluation['median_survival_seconds'],
                       max_seconds=evaluation['best_survival_seconds'],actions=evaluation['action_counts'],
                       terminations=evaluation['eval_termination_counts'])
            if evaluation['status']=='COMPLETE' and evaluation.get('durations_seconds'):
                values=evaluation['durations_seconds']
                rng=random.Random(371)
                means=sorted(sum(rng.choices(values,k=len(values)))/len(values) for _ in range(1000))
                row['mean_bootstrap_95pct_seconds']=[means[25],means[974]]
        row['eligible_for_comparison']=row['training_status']=='COMPLETE' and row['evaluation_status']=='COMPLETE'
        rows.append(row)
    return dict(rows=rows,note='Equal wall budgets, not equal transitions. Unity track seeds are not controlled. '
                'Episode bootstrap intervals do not capture between-training-seed uncertainty. '
                'Do not adopt a winner from overlapping intervals or one short scratch run.')


class StageFailures(RuntimeError):
    """All runnable independent stages finished, but at least one failed."""

def execute(plan,output,keep_going=False,before_job=None,progress=None):
    expected=json.loads((output/'sources.json').read_text())
    if source_hashes()!=expected:raise RuntimeError('Sources changed since planning; make a new plan')
    for job in plan['jobs']:
        folder=Path(job['output'])
        if folder.exists() and not job.get('reused'):raise RuntimeError(f'Output already exists; refusing to overwrite {folder}')
    completed=[]
    failed=[]
    failed_training=set()
    try:
        for job in plan['jobs']:
            if source_hashes()!=expected:raise RuntimeError('Sources changed between stages')
            for retained in plan.get('retained_artifacts',[]):
                if file_hash(retained['path'])!=retained['sha256']:raise RuntimeError('Retained smoke artifact changed')
            if job['mode']=='eval' and job['name'] in failed_training:
                print(f'[SKIPPED] {job["name"]} eval: training incomplete',flush=True)
                if progress:progress(job,'SKIPPED')
                continue
            if job.get('reused'):
                checked_result(job)
                print(f'[RETAINED] {job["name"]} {job["mode"]}: {job["output"]}',flush=True)
                completed.append(dict(name=job['name'],mode=job['mode'],reused=True))
                if progress:progress(job,'RETAINED')
                (output/'completed_stages.json').write_text(json.dumps(completed,indent=2))
                continue
            log=output/f'{job["name"]}_{job["mode"]}.log'
            if before_job is not None:before_job(job)
            if progress:progress(job,'START')
            print(f'[{job["mode"].upper()}] {job["name"]}; log: {log}',flush=True)
            with log.open('x',encoding='utf8') as handle:
                process=subprocess.Popen(job['command'],cwd=ROOT,stdout=subprocess.PIPE,
                                         stderr=subprocess.STDOUT,text=True,encoding='utf8',errors='replace')
                try:
                    for line in process.stdout:
                        print(line,end='',flush=True)
                        handle.write(line);handle.flush()
                        if progress:progress(job,'OUTPUT',line)
                    code=process.wait()
                except BaseException:
                    # Parent and child share the terminal Ctrl+C. Wait for child's
                    # checkpoint/finalization instead of blindly killing it.
                    print('Batch interrupted; waiting for trial finalization.',flush=True)
                    process.wait()
                    raise
            validation_error=None
            if not code:
                try:checked_result(job)
                except (OSError,ValueError,KeyError,RuntimeError) as exc:validation_error=repr(exc)
            if code or validation_error:
                if progress:progress(job,'FAILED')
                failed.append(dict(name=job['name'],mode=job['mode'],exit_code=code,output=job['output'],
                                   validation_error=validation_error))
                (output/'failed_stages.json').write_text(json.dumps(failed,indent=2))
                # Only continue after the child recorded successful cleanup.
                # Missing/malformed reports or cleanup errors may leave a live
                # browser behind and would contaminate subsequent timing tests.
                report_name='trial_report.json' if job['mode']=='train' else 'v61_eval_report.json'
                report_path=Path(job['output'])/report_name
                report=json.loads(report_path.read_text()) if report_path.exists() else {}
                interrupted=(code in (130,-1073741510,3221225786) or
                             'KeyboardInterrupt' in str(report.get('error') or '') or
                             report.get('stop_reason')=='interrupted' or
                             report.get('eval_stop_reason')=='interrupted')
                if interrupted or not keep_going or report.get('finalization_errors')!=[]:
                    raise RuntimeError(f'{job["name"]} {job["mode"]} exited {code}; batch stopped')
                if job['mode']=='train':failed_training.add(job['name'])
                print(f'[FAILED] {job["name"]} {job["mode"]}; outputs preserved; continuing independent stages',flush=True)
                continue
            completed.append(dict(name=job['name'],mode=job['mode']))
            if progress:progress(job,'PASSED')
            (output/'completed_stages.json').write_text(json.dumps(completed,indent=2))
    finally:
        (output/'comparison.json').write_text(json.dumps(summarize(plan),indent=2))
    if failed:raise StageFailures(f'{len(failed)} stages failed; completed results preserved in {output}')


def continue_smoke(parent_path,output,*,same_sources=False):
    """Reuse only audited smoke stages; never mix sources in full training trials."""
    parent_path=Path(parent_path).resolve()
    parent=json.loads(parent_path.read_text())
    if same_sources:
        if json.loads((parent_path.parent/'sources.json').read_text())!=source_hashes():
            raise ValueError('Full batch continuation requires unchanged sources; preserve old results as separate evidence')
    elif not 0<parent['train_minutes']<=1 or parent['eval_episodes']>3:
        raise ValueError('Cross-version continuation is restricted to smoke validation, not full learning comparisons')
    plan=copy.deepcopy(parent)
    plan.update(parent_plan=str(parent_path),source_mixed_smoke_validation=not same_sources,retained_artifacts=[],jobs=[])
    train_paths={}
    retained_trains=set()
    for old in parent['jobs']:
        job=copy.deepcopy(old);job.pop('reused',None)
        try:
            if job['mode']=='eval' and job['name'] not in retained_trains:
                raise RuntimeError('Evaluation belongs to a training stage that must be rerun')
            checked_result(job)
            artifact=Path(job['output'])/('latest.zip' if job['mode']=='train' else 'v61_eval_report.json')
            if job['mode']=='train':
                with zipfile.ZipFile(artifact) as z:
                    if z.testzip() is not None:raise ValueError('Corrupt completed checkpoint')
            job['reused']=True
            if job['mode']=='train':retained_trains.add(job['name'])
            files=[artifact,Path(job['output'])/('trial_report.json' if job['mode']=='train' else 'trial_config.json')]
            for f in files:
                if f.exists():plan['retained_artifacts'].append(dict(path=str(f),sha256=file_hash(f)))
        except (OSError,ValueError,RuntimeError,KeyError,zipfile.BadZipFile):
            job['output']=str(output/job['name']/job['mode'])
            cmd=job['command'][:]
            cmd[cmd.index('--output')+1]=job['output']
            cmd[cmd.index('--expected-sources')+1]=str(output/'sources.json')
            if job['mode']=='eval':cmd[cmd.index('--checkpoint')+1]=str(train_paths[job['name']]/'latest.zip')
            job['command']=cmd
        if job['mode']=='train':train_paths[job['name']]=Path(job['output'])
        plan['jobs'].append(job)
    return plan


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--configs',nargs='+',choices=list(PRESETS),default=list(PRESETS))
    p.add_argument('--seeds',nargs='+',type=int,default=[123])
    p.add_argument('--train-minutes',type=float,default=30)
    p.add_argument('--eval-episodes',type=int,default=100)
    p.add_argument('--eval-max-minutes',type=float,default=60)
    p.add_argument('--order-seed',type=int,default=42)
    p.add_argument('--output',type=Path)
    p.add_argument('--plan',type=Path,help='Execute or display a previously generated plan')
    p.add_argument('--continue-smoke',type=Path,help='Retain completed audited smoke stages in a new continuation plan')
    p.add_argument('--continue-batch',type=Path,help='Retain completed full trials, requiring identical source hashes')
    p.add_argument('--keep-going',action='store_true',help='Continue independent stages after a cleanly finalized failure')
    p.add_argument('--execute',action='store_true',help='Actually launch browsers/training; default writes a plan only')
    a=p.parse_args()
    if sum(x is not None for x in [a.plan,a.continue_smoke,a.continue_batch])>1:p.error('Choose one plan/continuation option')
    if a.plan:
        plan=json.loads(a.plan.read_text());output=a.plan.resolve().parent
    else:
        if not 0<a.train_minutes<=2880 or not 0<a.eval_max_minutes<=2880 or a.eval_episodes<1:
            p.error('Invalid budget')
        if len(set(a.seeds))!=len(a.seeds) or any(not 0<=s<2**32 for s in a.seeds):p.error('Seeds must be distinct uint32 values')
        if len(set(a.configs))!=len(a.configs):p.error('Configurations must be distinct')
        output=(a.output or ROOT/'reports'/('temporal_batch_'+datetime.now().strftime('%Y%m%d_%H%M%S_%f'))).resolve()
        if not output.is_relative_to(ROOT/'reports'):p.error('Output must be a new directory under reports')
        output.mkdir(parents=True,exist_ok=False)
        (output/'sources.json').write_text(json.dumps(source_hashes(),indent=2))
        plan=dict(version=1,initialization='scratch',worker_count=1,configs={n:PRESETS[n] for n in a.configs},
                  seeds=a.seeds,train_minutes=a.train_minutes,eval_episodes=a.eval_episodes,
                  order_seed=a.order_seed,jobs=jobs_for(a.configs,a.seeds,a.train_minutes,
                        a.eval_episodes,a.eval_max_minutes,output,a.order_seed),
                  note='Run after other training stops. No checkpoint migration. All training stages precede evaluations.')
        if a.continue_smoke:plan=continue_smoke(a.continue_smoke,output)
        if a.continue_batch:plan=continue_smoke(a.continue_batch,output,same_sources=True)
        (output/'plan.json').write_text(json.dumps(plan,indent=2))
    print('PLAN:',output/'plan.json')
    print('Stages:',len(plan['jobs']),'; scratch training minutes:',len(plan['configs'])*len(plan['seeds'])*plan['train_minutes'])
    if a.execute:execute(plan,output,keep_going=a.keep_going)
    else:print('Plan only; no browser or policy was launched. Add --plan <printed path> --execute to run.')


if __name__=='__main__':main()
