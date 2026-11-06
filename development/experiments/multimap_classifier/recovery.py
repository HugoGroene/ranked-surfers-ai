"""Bounded restart of known guard stops; never continue an uncertain episode."""
import hashlib
import json
import os
from pathlib import Path
import subprocess
import time
import zipfile
import psutil


def save(path,value):
    temporary=path.with_suffix('.partial.json')
    temporary.write_text(json.dumps(value,indent=2),encoding='utf8')
    temporary.replace(path)


def run_child(command,env,dest,*,cwd):
    """Track only our own process descendants and finish cleanup before returning."""
    owned={};process=None;forced=[];remaining=[]
    try:
        process=subprocess.Popen(command,cwd=cwd,env=env)
        while True:
            try:
                parent=psutil.Process(process.pid)
                for child in [parent,*parent.children(recursive=True)]:
                    owned[(child.pid,child.create_time())]=child
            except psutil.NoSuchProcess:pass
            code=process.poll()
            if code is not None:return code
            time.sleep(.25)
    finally:
        # A cooperative trainer normally leaves nothing. On cancellation or a
        # leaked host, terminate only identity-checked descendants we observed.
        for child in reversed(list(owned.values())):
            try:
                if child.is_running():child.kill();forced.append(child)
            except psutil.NoSuchProcess:pass
        _,alive=psutil.wait_procs(forced,timeout=5)
        remaining=[child.pid for child in alive]
        if process is not None:process.wait(timeout=5)
        dest.mkdir(parents=True,exist_ok=True)
        save(dest/'attempt_processes.json',dict(forced_cleanup_pids=[c.pid for c in forced],
                                               remaining_owned_pids=remaining))
        if remaining:raise RuntimeError('Owned processes remain; refusing to start another cohort')


def checkpoint_data(path):
    with zipfile.ZipFile(path) as archive:
        for name in ('data','policy.pth','policy.optimizer.pth'):
            if name not in archive.namelist():raise ValueError('Incomplete checkpoint: '+name)
        if archive.testzip() is not None:raise ValueError('Checkpoint CRC failed')
        return json.loads(archive.read('data'))


def verified_checkpoint(dest,source):
    """Trust only the final atomic save after a fully completed optimizer update."""
    report=json.loads((dest/'train/parallel_training.json').read_text())
    if report.get('finalization_errors') or report.get('source_unchanged') is not True:
        raise ValueError('Finalization/source integrity failed')
    if report.get('source_sha256')!=hashlib.sha256(source.read_bytes()).hexdigest():
        raise ValueError('Source checkpoint hash mismatch')
    optimized=report.get('optimized_transitions',0)
    if not optimized:return source,report
    if (report.get('final_checkpoint_transitions')!=optimized or
            report.get('processed_transitions')!=optimized or report.get('final_checkpoint_skipped')):
        raise ValueError('Final checkpoint does not cover the completed optimizer updates')
    candidate=dest/'train/latest.zip'
    data=checkpoint_data(candidate);before=checkpoint_data(source)
    if data.get('num_timesteps')!=before.get('num_timesteps',0)+optimized:
        raise ValueError('Checkpoint step count mismatch')
    if data.get('observation_contract')!=before.get('observation_contract'):
        raise ValueError('Checkpoint input contract changed')
    return candidate,report


def recoverable(report):
    error=report.get('error','')
    markers=('No four consecutive GAMEPLAY frames', 'unsafe post-resume state=UNCERTAIN',
             'MENU not independently confirmed', 'unresolved captures',
             'No fresh offscreen paint within 500ms')
    return ('GuardUncertainError' in error and any(marker in error for marker in markers)
            and not report.get('finalization_errors'))


def supervise(output,source,minutes,max_restarts,launch,*,clock=time.time,verify=verified_checkpoint):
    """launch(dest, source, remaining_minutes, absolute_deadline) -> exit code.

    Smoke is deliberately outside this function. Deadline includes launch,
    excluded startup episodes, cleanup and every recovery; the trainer finishes
    its current optimizer boundary before stopping, so this is a soft deadline.
    """
    started=clock();deadline=started+minutes*60
    status=dict(status='RUNNING',requested_minutes=minutes,max_restarts=max_restarts,
                source_checkpoint=str(source),deadline_unix=deadline,attempts=[])
    path=output/'recovery.json';save(path,status)
    current=source;rapid_failures=0;last=output/'stability';code=1
    try:
        for index in range(max_restarts+1):
            remaining=deadline-clock()
            if remaining<15:
                status['status']='BUDGET_EXHAUSTED_AFTER_STOP';break
            last=output/('stability' if index==0 else f'recovery_{index:02d}')
            row=dict(index=index,folder=str(last),source_checkpoint=str(current),
                     remaining_minutes=remaining/60,status='RUNNING')
            status['attempts'].append(row);save(path,status)
            print(f'[RECOVERY] Attempt {index+1}/{max_restarts+1}; {remaining/60:.1f} minutes remain; checkpoint={current}',flush=True)
            code=launch(last,current,remaining/60,deadline)
            row['exit_code']=code
            candidate,report=verify(last,current)
            row.update(status=report.get('status'),error=report.get('error'),
                       checkpoint=str(candidate),optimized_transitions=report.get('optimized_transitions',0))
            status['latest_checkpoint']=str(candidate)
            if code==0:
                if report.get('status')!='COMPLETE' or report.get('error'):
                    raise ValueError('Successful process has an inconsistent training report')
                status['status']='COMPLETE';break
            if not recoverable(report):
                status['status']='STOPPED_NONRECOVERABLE';break
            rapid_failures=(rapid_failures+1 if report.get('active_wall_seconds',0)<60
                            or not report.get('optimized_transitions') else 0)
            if rapid_failures>=2:
                status['status']='STOPPED_REPEATED_EARLY_FAILURE';break
            if index==max_restarts:
                status['status']='STOPPED_RETRY_LIMIT';break
            current=candidate
            print('[RECOVERY] Guard stop preserved. Starting fresh workers from the verified saved checkpoint.',flush=True)
    except BaseException as exc:
        status.update(status='CANCELLED' if isinstance(exc,KeyboardInterrupt) else 'FAILED',error=repr(exc))
        raise
    finally:
        status['elapsed_seconds']=clock()-started
        save(path,status)
    return code,last,[Path(row['folder']) for row in status['attempts']]
