"""Collect local detector-off action and terminal evidence without training."""
import argparse
from collections import deque
from datetime import datetime
import json
import math
from pathlib import Path
import time


def settle_startup(step, seconds, clock=time.perf_counter):
    """Diagnostic-only passive lead-in; never reset or send movement on failure."""
    deadline=clock()+seconds
    result=None
    while clock()<deadline:
        result=step(0)
        if result[2] or result[3]:
            raise RuntimeError('Episode ended during startup lead-in; no test movement sent')
    return result


def summarize(trials, expected, error):
    complete = len(trials) == expected and error is None
    return dict(status='COLLECTED_REVIEW_REQUIRED' if complete else 'INCOMPLETE',
                expected_trials=expected, collected_trials=len(trials), error=error,
                training=False, checkpoint_loaded=False,
                movement_verified=False, unity_clock_verified=False,
                pending=['Inspect action sequences for actual LEFT/RIGHT/JUMP/ROLL effects',
                         'Inspect terminal sequences for false death detections',
                         'Compare visible HUD progression with timestamps; step time is not Unity time',
                         'Visible-browser crop parity requires a separate foreground check',
                         'BC recorder deferred; long-run reliability not tested'])


def main(argv=None):
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--run-name',default='live_action_checks_'+datetime.now().strftime('%Y%m%d_%H%M%S'))
    parser.add_argument('--trials',type=int,default=2)
    parser.add_argument('--max-minutes',type=float,default=5)
    parser.add_argument('--startup-wait-seconds',type=float,default=1.5,
                        help='Passive diagnostic lead-in after guarded reset; images still require review')
    args=parser.parse_args(argv)
    if Path(args.run_name).name != args.run_name or args.run_name in ('.','..'):
        parser.error('run-name must be a single folder name')
    if not 1 <= args.trials <= 10 or not 0 < args.max_minutes <= 15:
        parser.error('Use 1–10 trials and a duration up to 15 minutes')
    if not math.isfinite(args.startup_wait_seconds) or not 0 < args.startup_wait_seconds <= 5:
        parser.error('startup-wait-seconds must be greater than zero and at most five')
    import cv2
    import train_rs_v612_ppo as trainer
    from tools.ppo_headless_env import install
    from train_rs_v2_verify import RUN_ROOT
    dest=RUN_ROOT/args.run_name
    if dest.exists():
        parser.error(f'Output already exists: {dest}')
    config=trainer.parse_args(['--mode','probe','--run-name',args.run_name,
        '--background-display','1','--canvas-policy-input','--framing-profile','canvas_v1',
        '--detector-mode','off','--reuse-observation-guard','--press-delay-ms','5',
        '--probe-delays-ms','5','--probe-trials',str(args.trials),
        '--state-model','checkpoints/v612_snapshot_20261021/screen_state_candidate_v2.pt'])
    install(trainer,'electron',electron_transport='shared_memory')
    original=trainer.create_live_env
    started=time.perf_counter()
    # No additional capture: retain the exact observations already returned by step.
    def create(*a,**kw):
        monitored,raw=original(*a,**kw)
        evidence=dest/'sequences';evidence.mkdir()
        step,reset=monitored.step,monitored.reset
        ring=deque(maxlen=16)
        episode=0;index=0;phase='startup';action_index=0
        def save(frames,kind):
            for record,pixels in frames:
                filename=f'ep{record["episode"]:03d}_{kind}_{record["step"]:04d}.jpg'
                if not cv2.imwrite(str(evidence/filename),pixels):
                    raise OSError(f'Could not save {filename}')
                with (evidence/'frames.jsonl').open('a',encoding='utf8') as handle:
                    handle.write(json.dumps(dict(record,picture=filename))+'\n')
        def checked_reset(*ra,**rk):
            nonlocal episode,index,phase,action_index
            result=reset(*ra,**rk);episode+=1;index=0;ring.clear()
            phase='startup';action_index=0
            save([(dict(episode=episode,step=0,elapsed_seconds=time.perf_counter()-started,
                        action=None),trainer._rgb_latest(result[0]).copy())],'start')
            settled=settle_startup(checked_step,args.startup_wait_seconds)
            phase='action';ring.clear()
            if settled is not None:
                result=(settled[0],settled[4])
            save([(dict(episode=episode,step=index,elapsed_seconds=time.perf_counter()-started,
                        action=None,phase='pre_action'),trainer._rgb_latest(result[0]).copy())],'pre_action')
            print(f'[INPUT PROBE] Episode {episode}: passive startup lead-in complete; '
                  'saved pre-action frame for visual verification.',flush=True)
            return result
        def checked_step(action):
            nonlocal index,action_index
            if time.perf_counter()-started > args.max_minutes*60:
                raise TimeoutError('Diagnostic time budget reached; partial evidence preserved')
            before=time.perf_counter()
            result=step(action);after=time.perf_counter();index+=1
            obs,reward,terminated,truncated,info=result
            record=dict(episode=episode,step=index,action=int(action),phase=phase,
                        elapsed_seconds=after-started,step_wall_ms=(after-before)*1000,
                        reward=float(reward),terminated=bool(terminated),truncated=bool(truncated),
                        terminal_evidence_source=info.get('terminal_evidence_source'))
            ring.append((record,trainer._rgb_latest(obs).copy()))
            with (dest/'step_trace.jsonl').open('a',encoding='utf8') as handle:
                handle.write(json.dumps(record)+'\n')
            if phase=='startup':
                if index==1 or index%5==0:save([ring[-1]],'startup')
            else:
                action_index+=1
                if action_index<=24:save([ring[-1]],'action')
            if terminated or truncated:save(ring,'terminal')
            return result
        monitored.reset=checked_reset;monitored.step=checked_step
        return monitored,raw
    trainer.create_live_env=create
    error=None
    try:
        trainer.input_probe(config)
    except (Exception,KeyboardInterrupt) as exc:
        error=repr(exc)
    finally:
        trainer.create_live_env=original
    dest.mkdir(parents=True,exist_ok=True)
    probe=dest/'v61_input_probe.json'
    trials=json.loads(probe.read_text(encoding='utf8')).get('trials',[]) if probe.exists() else []
    report=summarize(trials,args.trials*5,error)
    report['startup_wait_seconds']=args.startup_wait_seconds
    report['startup_note']='Fixed passive lead-in, not automatic recognition of controllable gameplay. Review pre_action frames.'
    report['note']='Diagnostic image writes add overhead. These timings are not a training-speed benchmark.'
    (dest/'live_checks_report.json').write_text(json.dumps(report,indent=2),encoding='utf8')
    print(json.dumps(report,indent=2),flush=True)
    print(f'REPORT: {dest / "live_checks_report.json"}',flush=True)
    return 0 if report['status']=='COLLECTED_REVIEW_REQUIRED' else 2


if __name__=='__main__':
    raise SystemExit(main())
