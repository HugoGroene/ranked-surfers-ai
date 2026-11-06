"""Visible-eval diagnostics only; no changes to observations, actions or guards."""
from collections import Counter
import time
from tools.ppo_run_integrity import atomic_json


def install(trainer):
    create = trainer.create_live_env

    def traced_create(a, dest, **kwargs):
        monitored, raw = create(a, dest, **kwargs)
        reset, step, close = monitored.reset, monitored.step, monitored.close
        episodes = []
        current = None
        reset_done = None

        def traced_reset(*args, **kwargs):
            nonlocal current, reset_done
            began = time.perf_counter()
            current = dict(episode=len(episodes)+1, status='RESETTING', decisions=0,
                           requested_actions=Counter(), dispatched_keys=Counter(), first_steps=[])
            episodes.append(current)
            print('[VISIBLE START] Preparing guarded episode start; '
                  'policy controls begin after reset completes.', flush=True)
            result = reset(*args, **kwargs)
            reset_done = time.perf_counter()
            current.update(status='PLAYING', reset_wall_seconds=reset_done-began)
            print(f'[VISIBLE START] Episode {current["episode"]}: policy ready '
                  f'(reset {reset_done-began:.2f}s).', flush=True)
            return result

        def traced_step(action):
            began = time.perf_counter()
            result = step(action)
            obs, reward, done, truncated, info = result
            if current is not None:
                current['decisions'] += 1
                current['requested_actions'][str(int(action))] += 1
                key = info.get('learning_dispatched_key')
                # Evaluation does not enable settled-learning info fields. Its
                # timing probe still records whether a key press was dispatched.
                if key is None:
                    timings = getattr(raw,'step_timings',None)
                    latest = timings[-1] if timings else {}
                    if latest.get('press') and latest.get('press_timestamp',0) >= began:
                        from train_rs_rl_early_death import ACTIONS
                        key = ACTIONS[int(action)]
                if key is not None:
                    current['dispatched_keys'][str(key)] += 1
                if len(current['first_steps']) < 8:
                    current['first_steps'].append(dict(action=int(action), dispatched_key=key,
                        since_reset_seconds=began-reset_done,
                        step_wall_ms=(time.perf_counter()-began)*1000))
                if current['decisions']==1:
                    print(f'[VISIBLE START] First requested action={int(action)}; '
                          f'prediction/setup delay={(began-reset_done)*1000:.1f}ms.',flush=True)
                if done or truncated:
                    current['status']='COMPLETE'
                    print(f'[VISIBLE EPISODE] {current["episode"]}: '
                          f'requested actions={dict(current["requested_actions"])}',flush=True)
            return result

        def traced_close():
            try:
                return close()
            finally:
                atomic_json(dest/'visible_episode_trace.json',dict(episodes=episodes,
                    note='Requested actions are policy outputs; dispatched keys are recorded only '
                         'when reported by the environment. Neither proves Unity consumed input.'))

        monitored.reset, monitored.step, monitored.close = traced_reset, traced_step, traced_close
        return monitored, raw

    trainer.create_live_env = traced_create
