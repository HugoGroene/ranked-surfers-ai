"""Play one guarded episode outside Monitor, PPO collection and callbacks."""
import time
from tools.ppo_run_integrity import atomic_json


def discard_first_episode(raw, predict, abort, output, *, max_seconds=360):
    """Leave the finished raw episode for the normal Monitor/_setup_learn reset.

    No rollout buffer, learner counter, callback or Monitor is touched here.
    Environment diagnostics remain available. Failure propagates; it is never
    converted to a completed warm-up or a synthetic terminal transition.
    """
    started=time.perf_counter()
    report=dict(status='RUNNING',excluded_from_training=True,
                excluded_from_learning_statistics=True,steps=0,reward=0.,actions={})
    raw._training_warmup_active=True
    atomic_json(output/'startup_episode.json',report)
    try:
        if abort.is_set():raise RuntimeError('Startup episode aborted')
        obs,_=raw.reset()
        report['episode']=raw.episode_number
        while True:
            if abort.is_set():raise RuntimeError('Startup episode aborted')
            if time.perf_counter()-started>=max_seconds:
                raise TimeoutError('Startup episode did not finish within its bounded allowance')
            action=int(predict(obs))
            obs,reward,terminated,truncated,info=raw.step(action)
            report['steps']+=1
            report['reward']+=float(reward)
            report['actions'][str(action)]=report['actions'].get(str(action),0)+1
            if terminated or truncated:
                report.update(status='EXCLUDED',terminated=bool(terminated),truncated=bool(truncated),
                              duration=info.get('reward_duration_s',info.get('duration')),
                              termination_reason=info.get('termination_reason'))
                break
    except BaseException as exc:
        report.update(status='FAILED',error=repr(exc))
        raise
    finally:
        raw._training_warmup_active=False
        report['wall_seconds']=time.perf_counter()-started
        atomic_json(output/'startup_episode.json',report)
    print(f'[STARTUP EXCLUDED] Episode {report["episode"]}: {report["steps"]} steps; '
          'no PPO samples or learning statistics. Training starts after normal reset.',flush=True)
    return report
