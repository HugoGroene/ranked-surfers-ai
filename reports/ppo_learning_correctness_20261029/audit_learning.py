"""Offline transition audit. Never opens a browser or loads a working checkpoint.

Uses the installed SB3 collector and repository callback with a tiny synthetic
environment. Synthetic observations/time gaps expose ordering, not Unity physics.
Writes new audit artifacts only; never changes trainer or reward semantics.
"""
from collections import Counter, defaultdict
import argparse
import contextlib
import hashlib
import inspect
import io
import json
import math
from pathlib import Path
import sys
import tempfile
import time
from types import SimpleNamespace

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))
import gymnasium as gym
import numpy as np
import stable_baselines3
from stable_baselines3 import PPO
from stable_baselines3.common.monitor import Monitor
from stable_baselines3.common.on_policy_algorithm import OnPolicyAlgorithm
from stable_baselines3.common.vec_env import DummyVecEnv
from rs_v61_rewards import ProgressRewards
from rs_v612_gameplay_guard import GuardUncertainError
from train_rs_v612_ppo import build_callback

HERE = Path(__file__).resolve().parent


def digest(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


class SyntheticPractice(gym.Env):
    """Models only the callback-facing protocol; no actual browser/guard claims."""
    def __init__(self, mode):
        self.mode = mode
        self.action_space = gym.spaces.Discrete(5)
        self.observation_space = gym.spaces.Dict({
            'image': gym.spaces.Box(-10000, 10000, (2,), np.float32),
            'aux': gym.spaces.Box(-10000, 10000, (1,), np.float32)})
        self.total = self.boundaries = 0
        self.events = []
        self.code = 0
        self.gap = 0.
        self.last_capture_end = time.perf_counter()
        self._ppo_boundary_menu_pending = False
        self._ppo_boundary_menu_duration_at = None
        self._ppo_boundary_menu_started_at = None
        self._ppo_pause_pending = None
        self.survival_rewards = ProgressRewards()
        self.action_context = SimpleNamespace(last_action_times=[None]*5)

    def observation(self):
        return {'image': np.array([self.code, 0], np.float32),
                'aux': np.array([self.duration], np.float32)}

    def reset(self, *, seed=None, options=None):
        super().reset(seed=seed)
        self.duration = 0.
        self.code = 0
        self.episode_start = self.last_step = time.monotonic()
        self.episode_started = time.perf_counter()
        self.survival_rewards.reset()
        self.events.append({'event': 'reset', 'after_total': self.total})
        return self.observation(), {}

    def step(self, action):
        self.total += 1
        handoff = self._ppo_boundary_menu_pending
        terminal = handoff or (self.mode == 'death' and self.total == 2)
        truncated = self.mode == 'timeout' and self.total == 2
        source_code = self.code
        self.code += 1
        self.duration += .05 + self.gap
        gap = self.gap
        self.gap = 0.
        if handoff:
            self.duration = max(self.survival_rewards.last_duration,
                                self._ppo_boundary_menu_duration_at)
            self._ppo_boundary_menu_pending = False
        reward, pieces = self.survival_rewards.step(self.duration, terminal)
        self.last_capture_end = time.perf_counter()
        info = dict(duration=self.duration, reward_survival=pieces['survival'],
                    reward_milestones=pieces['milestones'], reward_death=pieces['death'],
                    reward_v61=reward, ppo_menu_handoff_terminal=handoff)
        self.events.append(dict(event='step', step=self.total, input_code=source_code,
            next_code=self.code, action=int(action), movement_dispatched=not handoff and int(action)!=0,
            suppressed_by_handoff=handoff, reward=reward, done=terminal,
            truncated=truncated, passive_gap=gap, duration=self.duration))
        return self.observation(), reward, terminal, truncated, info

    def pause_for_ppo_optimizer(self):
        self.boundaries += 1
        self.events.append({'event': 'rollout_end', 'step': self.total})
        if ((self.mode == 'boundary_mid' and self.boundaries == 1) or
                (self.mode == 'boundary_final' and self.boundaries == 2)):
            self._ppo_boundary_menu_duration_at = self.duration + .01
            raise GuardUncertainError('Not verified GAMEPLAY at rollout end: MENU, menu=0.900')
        self._ppo_pause_pending = True
        return 0.

    def confirm_post_optimizer_menu(self):
        return True, {'checks': [{}, {}, {}, {}]}

    def resume_after_ppo_optimizer(self, monitor):
        self._ppo_pause_pending = None
        return dict(paused_wall_seconds=.01, corrected_episode_timer=True,
                    corrected_action_timer=True, corrected_monitor_timer=True,
                    active_episode_clock_drift_s=0.)

    def refresh_after_ppo_optimizer(self):
        if self.mode == 'post_resume_mid' and self.boundaries == 1:
            self._ppo_post_resume_menu_candidate = (
                time.monotonic()-.001, time.perf_counter()-.001, time.time()-.001)
            return None, 'synthetic MENU', None
        self.code += 100
        self.gap = .25
        self.last_capture_end = time.perf_counter()
        self.events.append({'event': 'passive_resync', 'step': self.total,
                            'code': self.code, 'unacted_seconds': .25})
        return self.observation(), None, self.last_capture_end


def synthetic_case(mode):
    raw = SyntheticPractice(mode)
    vec = DummyVecEnv([lambda: Monitor(raw)])
    model = PPO('MultiInputPolicy', vec, n_steps=2, batch_size=2, n_epochs=1,
                learning_rate=.0003, gamma=.995, gae_lambda=.95, seed=123,
                policy_kwargs={'net_arch': [8]}, device='cpu', verbose=0)
    stored = []
    returns = []
    original_add = model.rollout_buffer.add
    original_returns = model.rollout_buffer.compute_returns_and_advantage

    def add(obs, action, reward, episode_start, value, log_prob):
        stored.append(dict(step=raw.total, input_code=float(obs['image'][0, 0]),
            action=int(np.asarray(action).flat[0]), reward=float(reward[0]),
            episode_start=bool(episode_start[0])))
        return original_add(obs, action, reward, episode_start, value, log_prob)

    def compute(last_values, dones):
        result = original_returns(last_values, dones)
        returns.append(dict(step=raw.total, boundary_callback_count=raw.boundaries,
            last_done=bool(dones[0]), bootstrap_value=float(last_values.cpu().numpy().flat[0]),
            stored_last_reward=float(model.rollout_buffer.rewards[-1, 0]),
            last_return=float(model.rollout_buffer.returns[-1, 0])))
        return result

    model.rollout_buffer.add = add
    model.rollout_buffer.compute_returns_and_advantage = compute
    original_train = model.train

    def train_with_visible_clock_tick():
        # Tiny fake optimizer phases can fit within Windows' monotonic clock
        # resolution. Allow one tick so the real clock validator sees >0 time.
        time.sleep(.02)
        return original_train()

    model.train = train_with_visible_clock_tick
    args = SimpleNamespace(pause_ppo_optimizer=True, resync_rollouts=True,
        recover_rollout_menu=False, audit_reward_timing=True, log_every_seconds=1e9,
        save_every_minutes=1e9, max_minutes=0 if mode=='time_budget_stop' else 1000)
    try:
        with tempfile.TemporaryDirectory(prefix='ppo_learning_audit_') as temp:
            callback = build_callback(args, Path(temp), raw=raw)
            try:
                with contextlib.redirect_stdout(io.StringIO()) as output:
                    model.learn(total_timesteps=4, callback=callback)
            finally:
                for name in ('log_handle', 'reward_step_audit_handle'):
                    handle = getattr(callback, name, None)
                    if handle is not None and not handle.closed:
                        handle.close()
            audit = [json.loads(x) for x in
                     (Path(temp)/'ppo_reward_step_audit.jsonl').read_text().splitlines()]
        return dict(mode=mode, events=raw.events, buffer_additions=stored,
            return_computations=returns, audit_rows=len(audit),
            pending_final_menu=raw._ppo_boundary_menu_pending,
            total_timesteps=model.num_timesteps, callback_log=output.getvalue())
    finally:
        vec.close()


def recorded_rewards(run):
    directory = ROOT/'ai_data/rl_v2_runs'/run
    config = json.loads((directory/'v61_config.json').read_text())
    effective_path = directory/'v61_effective_ppo_settings.json'
    settings = (json.loads(effective_path.read_text())['effective_settings']
                if effective_path.exists() else config)
    rollout = settings['n_steps']
    durations = []
    groups = defaultdict(lambda: dict(count=0, increments=[], cap_excess_seconds=0.,
                                    logged_survival=0., death_penalties=0))
    previous_duration = 0.
    previous_done = True
    first_step = None
    last = None
    for line in (directory/'ppo_reward_step_audit.jsonl').open():
        row = json.loads(line)
        if first_step is None:
            first_step = row['step']
        increment = max(0., row['episode_duration_s']-(0. if previous_done else previous_duration))
        labels = ['all', 'terminal' if row['done'] else 'nonterminal']
        if previous_done:
            labels.append('first_after_reset')
        if (row['step']-first_step) % rollout == 0 and row['step'] != first_step:
            labels.append('first_after_optimizer')
        if row['ppo_menu_handoff_terminal']:
            labels.append('menu_handoff')
        for label in labels:
            g = groups[label]
            g['count'] += 1
            g['increments'].append(increment)
            g['cap_excess_seconds'] += max(0., increment-.15)
            g['logged_survival'] += row['reward_survival']
            g['death_penalties'] += row['reward_death'] < 0
        durations.append(increment)
        previous_duration, previous_done, last = row['episode_duration_s'], row['done'], row
    for g in groups.values():
        increments = np.asarray(g.pop('increments'))
        g.update(increment_median_s=float(np.median(increments)),
            increment_p95_s=float(np.percentile(increments,95)),
            increment_max_s=float(increments.max()),
            exact_cap_count=int((increments > .150000001).sum()),
            auditor_tolerant_cap_count=int((increments > .162000001).sum()))
        g['counterfactual_uncapped_survival_difference'] = .8*g['cap_excess_seconds']
    report = json.loads((directory/'v61_report.json').read_text())
    return dict(run=run, effective_settings=settings, groups=dict(groups),
        last_audit_row=last, final_requested_steps_completed=report['completed_requested_steps'],
        full_rollout_steps=(len(durations)//rollout)*rollout,
        trailing_incomplete_rollout_steps=len(durations)%rollout,
        note='Durations are trainer-observed, rounded to 0.01s; cap excess is not verified lost Unity gameplay time.')


def dispatch_evidence(run):
    """Check existing instrumentation; no claim about Unity key application."""
    mapping = {1:'ArrowLeft', 2:'ArrowRight', 3:'ArrowUp', 4:'ArrowDown'}
    rows = [json.loads(line) for line in
            (ROOT/'ai_data/rl_v2_runs'/run/'reaction_latency_steps.jsonl').open()]
    failures = []
    commands = 0
    no_command_actions = Counter()
    suppressed_nonzero_terminal = 0
    for i, row in enumerate(rows):
        capture = row['observation']['capture']
        if capture['grab_end'] > row['prediction_start']:
            failures.append([i,'observation captured after prediction'])
        if row.get('command_start') is not None:
            commands += 1
            if row.get('key') != mapping.get(row['action']):
                failures.append([i,'selected action and requested key disagree'])
            if not (row['prediction_start'] <= row['prediction_end'] <= row['command_start']
                    <= row['command_end'] <= row['step_end']):
                failures.append([i,'prediction/dispatch ordering'])
            post = [c for c in row['captures'] if c['phase']=='post_action']
            if not post or post[-1]['grab_start'] < row['command_end']:
                failures.append([i,'missing or premature post-action capture'])
        else:
            no_command_actions[str(row['action'])] += 1
            if row['action']:
                if row['done'] and not row['truncated']:
                    suppressed_nonzero_terminal += 1
                else:
                    failures.append([i,'nonzero action has no dispatch and no death'])
    links = 0
    for previous, current in zip(rows, rows[1:]):
        post = [c for c in previous['captures'] if c['phase']=='post_action']
        if (not previous['done'] and not previous['truncated'] and post
                and previous['episode']==current['episode']):
            links += 1
            if post[-1]['id'] != current['observation']['capture']['id']:
                failures.append([links,'next policy source is not preceding post-action capture'])
    return dict(run=run,decisions=len(rows),commands=commands,
                no_command_actions=dict(no_command_actions),failures=failures,
                suppressed_nonzero_terminal=suppressed_nonzero_terminal,
                linked_next_observations=links,
                note='Command start/return are Playwright timings, not browser-event receipt or Unity response.')


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output', type=Path, default=HERE/'results_v2.json')
    output = parser.parse_args().output
    if output.exists():
        raise SystemExit('Refusing to overwrite existing audit results')
    sources = [ROOT/'train_rs_v612_ppo.py', ROOT/'rs_v61_rewards.py',
               ROOT/'tools/ppo_optimizer_pause_clock.py', ROOT/'tools/ppo_rollout_resync.py']
    before = {str(p.relative_to(ROOT)): digest(p) for p in sources}
    cases = {mode: synthetic_case(mode) for mode in
             ['normal_resync','boundary_mid','boundary_final','post_resume_mid',
              'death','timeout','time_budget_stop']}
    assert cases['boundary_final']['pending_final_menu']
    assert not any(e.get('suppressed_by_handoff') for e in cases['boundary_final']['events'])
    assert any(e.get('suppressed_by_handoff') for e in cases['boundary_mid']['events'])
    assert cases['time_budget_stop']['audit_rows'] == 1
    assert len(cases['time_budget_stop']['buffer_additions']) == 0
    assert cases['normal_resync']['buffer_additions'][2]['input_code'] == 102
    assert cases['normal_resync']['events'][3]['event'] == 'rollout_end'
    assert cases['boundary_mid']['buffer_additions'][2]['reward'] < -.9
    assert cases['boundary_mid']['buffer_additions'][3]['episode_start']
    assert cases['post_resume_mid']['buffer_additions'][2]['reward'] == -1.
    assert abs(cases['death']['buffer_additions'][1]['reward'] + .96) < 1e-6
    assert cases['timeout']['buffer_additions'][1]['reward'] > .04
    for case in cases.values():
        for i, r in enumerate(case['return_computations']):
            assert r['boundary_callback_count'] == i
    runs = ['v612_paused_state_v179_8h_20261028_045909'] + [
        f'ppoctrl_20261028_{i:02d}_train' for i in (2,4,6)]
    after = {str(p.relative_to(ROOT)): digest(p) for p in sources}
    assert before == after
    gamma, lam = .995, .95
    discounts = [dict(interval_s=dt, decisions_per_second=1/dt,
        discount_after_5_seconds=gamma**(5/dt),
        discount_half_life_seconds=dt*math.log(.5)/math.log(gamma),
        gae_trace_half_life_seconds=dt*math.log(.5)/math.log(gamma*lam))
        for dt in (.05192,.04019,.03538,.05,.04)]
    result = dict(sb3_version=stable_baselines3.__version__,
        collector_source_sha256=hashlib.sha256(inspect.getsource(OnPolicyAlgorithm.collect_rollouts).encode()).hexdigest(),
        runtime_sources_unchanged=before==after, source_sha256=before,
        synthetic_cases=cases, recorded_runs=[recorded_rewards(run) for run in runs],
        dispatch_evidence=[dispatch_evidence(run) for run in
            ['v612_latency_pilot_30_20261028','v612_latency_pilot_detect1_30_20261028',
             'v612_pipeline_baseline10_v2_20261028','v612_pipeline_prepared10_v2_20261028']],
        cadence_illustrations=dict(gamma=gamma, gae_lambda=lam, rows=discounts,
            note='Constant-interval illustrations, not measured per-episode discount schedules.'))
    assert all(not item['failures'] for item in result['dispatch_evidence'])
    output.write_text(json.dumps(result,indent=2)+'\n',encoding='utf-8')
    print('Offline audit written:', output)
    print('All seven synthetic ordering scenarios completed; runtime/reward sources unchanged.')


if __name__ == '__main__':
    main()
