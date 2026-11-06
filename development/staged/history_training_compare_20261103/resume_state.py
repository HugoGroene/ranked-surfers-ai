"""Recover only saved optimizer work, with explicit per-arm wall-time accounting."""
import hashlib
import json
import math
from pathlib import Path
import zipfile


def checkpoint_data(path):
    with zipfile.ZipFile(path) as archive:
        return json.loads(archive.read('data'))


def saved_elapsed(folder,steps,cohort):
    report=folder/'parallel_training.json'
    if report.exists():
        r=json.loads(report.read_text(encoding='utf8'))
        if r.get('optimized_transitions')==steps:
            elapsed=r.get('active_wall_seconds')
            if isinstance(elapsed,(int,float)) and math.isfinite(elapsed) and elapsed>0:
                return float(elapsed)
    rounds=folder/'round_timings.jsonl'
    if rounds.exists():
        for line in rounds.read_text(encoding='utf8').splitlines():
            try:r=json.loads(line)
            except ValueError:continue  # Last append may have been interrupted.
            elapsed=r.get('elapsed_seconds')
            if (r.get('rounds')*cohort==steps if isinstance(r.get('rounds'),int) else False):
                if isinstance(elapsed,(int,float)) and math.isfinite(elapsed) and elapsed>0:
                    return float(elapsed)
    return None


def continuation(previous,new_job):
    """New output folder; preserve source and all earlier attempts.

    Retry failed smoke from fresh initialization; this helper handles training only.
    Only time corresponding to recovered model updates is counted toward the budget.
    """
    folder=Path(previous['output']);source=Path(previous['command'][previous['command'].index('--checkpoint')+1])
    expected=checkpoint_data(source)['observation_contract']
    base=previous.get('source_timesteps',0)
    candidates=[]
    files=[folder/'latest.zip',folder/'latest.prev.zip',*folder.glob('checkpoint_*.zip')]
    for path in files:
        if not path.exists():continue
        try:
            data=checkpoint_data(path);total=data['num_timesteps'];steps=total-base
            if data.get('observation_contract')!=expected or not isinstance(total,int) or steps<=0:continue
            cohort=128*previous['workers']
            if steps%cohort:continue
            elapsed=saved_elapsed(folder,steps,cohort)
            if elapsed is not None:candidates.append((total,elapsed,path))
        except (OSError,ValueError,KeyError,zipfile.BadZipFile):continue
    for total,elapsed,path in sorted(candidates,key=lambda x:x[0],reverse=True):
        try:
            with zipfile.ZipFile(path) as archive:
                if archive.testzip() is not None:continue
        except (OSError,zipfile.BadZipFile):continue
        consumed=previous.get('carried_seconds',0.)+elapsed
        result=dict(new_job)
        result.update(carried_seconds=consumed,source_timesteps=total,
            prior_attempts=previous.get('prior_attempts',[])+[dict(output=str(folder),
                recovered_checkpoint=str(path),recovered_active_seconds=elapsed,
                recovered_optimized_transitions=total-base)],
            resumed_from=str(path),resume_sha256=hashlib.sha256(path.read_bytes()).hexdigest())
        # If a failure happened at the budget boundary, one short verified update
        # is still required; the explicit 15-second minimum can overrun the budget.
        result['minutes']=max(.25,new_job['total_budget_minutes']-consumed/60.)
        args=list(result['command'])
        args[args.index('--checkpoint')+1]=str(path)
        args[args.index('--minutes')+1]=str(result['minutes'])
        result['command']=args
        return result
    if any(path.exists() for path in files):
        raise ValueError(f'No compatible intact checkpoint with matching time accounting in {folder}; preserve files for review')
    # No update ever saved: retry the same input checkpoint and carried budget.
    result=dict(new_job)
    result.update(carried_seconds=previous.get('carried_seconds',0.),source_timesteps=base,
        prior_attempts=previous.get('prior_attempts',[])+[dict(output=str(folder),recovered_active_seconds=0.)])
    result['minutes']=previous['minutes']
    args=list(result['command']);args[args.index('--checkpoint')+1]=str(source)
    args[args.index('--minutes')+1]=str(result['minutes']);result['command']=args
    result['resume_sha256']=hashlib.sha256(source.read_bytes()).hexdigest()
    result['resumed_from']=str(source)
    return result
