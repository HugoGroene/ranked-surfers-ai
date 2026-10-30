"""Offline regression: actual V6.1 reward math, pause clock shifts, log replay.

No browser, GPUs, network or model files needed. Uses *installed* project
rs_v61_rewards.py and tools/ppo_optimizer_pause_clock.py.
"""
import copy
import json
import math
import unittest

from rs_v61_rewards import ProgressRewards
from tools.ppo_optimizer_pause_clock import compensate_pause_clocks
from tools.ppo_reward_timing_audit import (audit_training_log, audit_reward_steps,
                                           synthetic_reward_regression)
from tools.install_ppo_reward_step_audit_v16 import patch


class Context:
    def __init__(self):
        self.last_action_times = [None, 9.0, None, 10.0, 11.0]


class Env:
    def __init__(self):
        self.episode_start = 10.0
        self.last_step = 10.5
        self.episode_started = 10.0
        self.suspected_since = None
        self.previous_observation_at = 10.5
        self._prev_step_start = 10.5
        self.action_context = Context()
        self.last_detection = {'captured_at': 10.4}


class Monitor:
    def __init__(self, env):
        self.env = env
        self.t_start = 10.0


class ClockAndRewards(unittest.TestCase):
    def test_paused_vs_unpaused_match_both_episodes_and_death(self):
        result = synthetic_reward_regression()
        self.assertEqual(result['status'], 'PASS_SYNTHETIC_REWARD_CLOCKS')
        self.assertEqual(result['matched_step_rewards'], 12)
        self.assertEqual(result['verified_episode_resets'], 1)
        self.assertEqual(result['verified_death_penalties'], 2)

    def test_pausing_near_milestone_creates_no_bonus(self):
        c = ProgressRewards()
        c.step(.49, False)
        before = c.last_milestone
        e, monitor = Env(), None
        monitor = Monitor(e)
        e.episode_start = 10.0
        compensate_pause_clocks(e, 2.187, 2.187, monitor=monitor, elapsed_wall=2.187)
        next_duration = round(10.49 + 2.187 - e.episode_start, 2)
        got, parts = c.step(next_duration, False)
        self.assertEqual(before, c.last_milestone)
        self.assertEqual(parts['milestones'], 0.0)
        self.assertEqual(parts['survival'], 0.0)
        self.assertEqual(got, 0.0)

    def test_real_milestone_after_resume_awards_once(self):
        c = ProgressRewards()
        c.step(.49, False)
        _, parts = c.step(.51, False)
        self.assertAlmostEqual(parts['milestones'], .15)
        _, parts = c.step(.55, False)
        self.assertEqual(parts['milestones'], 0.0)

    def test_terminal_death_penalty_exactly_once(self):
        c = ProgressRewards()
        _, parts = c.step(.60, True)
        self.assertAlmostEqual(parts['death'], -1.0)
        self.assertEqual(parts['milestones'], 0.0)
        c.reset()
        _, parts = c.step(.10, False)
        self.assertEqual(parts['death'], 0.0)
        self.assertEqual(parts['milestones'], 0.0)

    def test_reward_cap_is_real_and_must_not_be_claimed_as_fixed(self):
        c = ProgressRewards()
        _, parts = c.step(.40, False)
        self.assertAlmostEqual(parts['survival'], .8 * .15)
        self.assertNotAlmostEqual(parts['survival'], .8 * .4)

    def test_action_age_and_monitor_clock_dont_advance_while_paused(self):
        e = Env()
        monitor = Monitor(e)
        prev = 12.0 - e.action_context.last_action_times[4]
        compensate_pause_clocks(e, 1.4, 1.4, monitor=monitor, elapsed_wall=1.4)
        self.assertAlmostEqual(13.4 - e.action_context.last_action_times[4], prev)
        self.assertAlmostEqual(13.4 - e.episode_start, 2.0)
        self.assertAlmostEqual(13.4 - monitor.t_start, 2.0)

    def test_old_yolo_timestamp_is_not_relabelled_fresh(self):
        e = Env()
        monitor = Monitor(e)
        prior = e.last_detection['captured_at']
        compensate_pause_clocks(e, 1.1, 1.1, monitor=monitor, elapsed_wall=1.1)
        self.assertEqual(e.last_detection['captured_at'], prior)

    def test_atomic_failure_on_nan_action_time(self):
        e = Env()
        monitor = Monitor(e)
        e.action_context.last_action_times[0] = float('nan')
        with self.assertRaises(ValueError):
            compensate_pause_clocks(e, 1, 1, monitor=monitor, elapsed_wall=1)
        self.assertEqual(e.episode_start, 10.0)
        self.assertEqual(monitor.t_start, 10.0)

    def test_reset_clears_progress_milestone_state(self):
        c = ProgressRewards()
        c.step(1.8, False)
        self.assertGreater(c.last_milestone, 0)
        c.reset()
        self.assertEqual(c.last_milestone, 0)
        self.assertEqual(c.last_duration, 0)
        _, parts = c.step(.11, False)
        self.assertEqual(parts['milestones'], 0)

    def test_no_negative_reward_for_repeated_time(self):
        c = ProgressRewards()
        c.step(.45, False)
        val, parts = c.step(.45, False)
        self.assertEqual(val, 0.0)
        self.assertEqual(parts['survival'], 0)


def mk_log(resumes=3, *, steps=512, final_extra=True, completed=True):
    lines = ['[V612 TRAIN] valid config']
    for i in range(resumes):
        lines.append('[PPO PAUSE TRACE] pre_pause_classifier=GAMEPLAY menu_score=0.000 ml_gameplay=1.000')
        lines.append('[PPO PAUSE] Freeze verified after 4 captures; track_deltas=[0.0, 0.0, 0.0]')
        lines.append('[PPO PAUSE] Verified frozen gameplay before optimizer; paused-track delta=0.000')
        lines.append('[PPO GAP] '+json.dumps({'boundary':i+1}))
        lines.append('[PPO PAUSE] Resumed after optimizer with pause-aware clocks: '+json.dumps({
            'paused_wall_seconds':1.2, 'corrected_episode_timer':True,
            'corrected_action_timer':True, 'corrected_monitor_timer':True,
            'active_episode_clock_drift_s':0.0}))
        for j in range(4):
            lines.append(f'[PPO RESYNC TRACE] frame={j+1}/4 since_resume_ms={21+j*60:.1f} '
                         'state=GAMEPLAY menu_score=0.000 ml_gameplay=1.000 ml_menu=0.000')
        lines.append('[PPO RESYNC] refreshed 4-frame stack; cached age now 8.4ms')
    if final_extra:
        lines.extend(['[PPO PAUSE TRACE] pre_pause_classifier=GAMEPLAY menu_score=0.000 ml_gameplay=1.000',
                      '[PPO PAUSE] Freeze verified after 4 captures; track_deltas=[0.0, 0.0, 0.0]',
                      '[PPO PAUSE] Verified frozen gameplay before optimizer; paused-track delta=0.000'])
    lines.append('[V612 REPORT] '+json.dumps({'steps':steps, 'completed_requested_steps':completed,
                                              'episodes':6,'mean_survival_seconds':4.0}))
    return '\n'.join(lines)


def mk_step_log(*, corrupt_reward=False, skip_step=False):
    calc = ProgressRewards()
    events = []
    counter = 100
    for ep in range(2):
        calc.reset()
        for si, seconds in enumerate([.10,.35,.59,.88,1.16]):
            done = si == 4
            val, parts = calc.step(seconds, done)
            step = counter
            counter += 1
            if skip_step and si == 3 and ep == 0:
                counter += 1
            row = {'step':step, 'episode_duration_s':seconds,
                   'done':done, 'truncated':False, 'environment_reward':val,
                   'reward_survival':round(parts['survival'],4),
                   'reward_milestones':round(parts['milestones'],4),
                   'reward_death':round(parts['death'],4),
                   'reward_v61':round(val,4)}
            events.append(row)
    if corrupt_reward:
        events[6]['reward_milestones'] = .99
    return '\n'.join(json.dumps(event) for event in events)


class AuditParsers(unittest.TestCase):
    def test_realistic_complete_rollouts(self):
        v = audit_training_log(mk_log(), min_resumes=3, min_steps=512)
        self.assertEqual(v['status'], 'PASS_STABILITY_CANDIDATE')
        self.assertEqual(v['verified_optimizer_pauses'], 4)
        self.assertEqual(v['optimizer_resumes'], 3)

    def test_powershell_select_string_duplicate_does_not_pollute(self):
        original = mk_log()
        echo = ('\nPS C:\\> Select-String -Path $log -Pattern "Safe stop"\n' +
                'ai_data\\foo.log:23:[PPO PAUSE] Verified frozen gameplay before optimizer\n')
        v = audit_training_log('diff preview [PPO PAUSE] Freeze verified after\n'+original+echo,
                               min_resumes=3,min_steps=512)
        self.assertEqual(v['status'], 'PASS_STABILITY_CANDIDATE')

    def test_safe_stop_fails(self):
        v = audit_training_log(mk_log().replace('[V612 REPORT]',
          '[PPO RESYNC] Safe stop: MENU\n[V612 REPORT]'))
        self.assertEqual(v['status'], 'FAIL_SAFE_AUDIT')

    def test_abandoned_episode_fails(self):
        v = audit_training_log(mk_log().replace('[V612 REPORT]',
          '[PPO RESYNC] ABANDONED partial episode\n[V612 REPORT]'))
        self.assertEqual(v['status'], 'FAIL_SAFE_AUDIT')

    def test_stale_frame_fails(self):
        v = audit_training_log(mk_log().replace('age now 8.4ms', 'age now 450.0ms'))
        self.assertEqual(v['status'], 'FAIL_SAFE_AUDIT')

    def test_missing_frozen_confirmation_fails(self):
        v = audit_training_log(mk_log().replace(
            '[PPO PAUSE] Freeze verified after 4 captures; track_deltas=[0.0, 0.0, 0.0]',
            'not frozen',1))
        self.assertEqual(v['status'], 'FAIL_SAFE_AUDIT')

    def test_incomplete_rollout_fails(self):
        v = audit_training_log(mk_log(completed=False))
        self.assertEqual(v['status'], 'FAIL_SAFE_AUDIT')

    def test_replay_matches_real_progress_rewards(self):
        v = audit_reward_steps(mk_step_log(), minimum_steps=10)
        self.assertEqual(v['status'], 'PASS_REWARD_REPLAY')
        self.assertEqual(v['completed_episodes'], 2)

    def test_replay_detects_forged_milestone(self):
        v = audit_reward_steps(mk_step_log(corrupt_reward=True),minimum_steps=10)
        self.assertEqual(v['status'], 'FAIL_REWARD_REPLAY')
        self.assertIn('reward_milestones', '\n'.join(v['failures']))

    def test_replay_detects_missing_training_step(self):
        v = audit_reward_steps(mk_step_log(skip_step=True),minimum_steps=10)
        self.assertEqual(v['status'], 'FAIL_REWARD_REPLAY')

    def test_replay_rejects_corrupt_json(self):
        v = audit_reward_steps('{bad',minimum_steps=1)
        self.assertEqual(v['status'], 'FAIL_REWARD_REPLAY')

    def test_replay_rejects_short_recording(self):
        v = audit_reward_steps(mk_step_log(),minimum_steps=100)
        self.assertEqual(v['status'], 'FAIL_REWARD_REPLAY')


SYNTHETIC_CALLBACK = '''
import argparse
from pathlib import Path
class T:
    def pause_for_ppo_optimizer(self):
        compensate_pause_clocks(self, 1., 1.)
        pass

def build_callback(a, output):
    from stable_baselines3.common.callbacks import BaseCallback
    class V61Callback(BaseCallback):
        def __init__(self):
            self.log_handle = None
        def _on_training_start(self):
            self.log_handle = self.log_path.open('a',encoding='utf-8')
        def _on_step(self):
            now = time.perf_counter()
            for info in self.locals.get('infos', []):
                for key, dest in (('reward_survival','survival'),('reward_milestones','milestones'),
                                  ('reward_death','death')):
                    pass
        def _on_training_end(self):
            if self.log_handle is not None:
                self.log_handle.close()

def parse_args():
    p=argparse.ArgumentParser()
    p.add_argument('--save-every-minutes',type=float,default=4.)
    a=p.parse_args()
    return a
# [PPO RESYNC TRACE] must be installed
# --pause-ppo-optimizer
'''


class Installer(unittest.TestCase):
    def test_patcher_compiles_and_audit_defaults_off(self):
        changed = patch(SYNTHETIC_CALLBACK)
        self.assertIn("p.add_argument('--audit-reward-timing', action='store_true'", changed)
        self.assertIn("'ppo_reward_step_audit.jsonl'", changed)
        self.assertIn("self.reward_step_audit_handle.write", changed)
        compile(changed,'synthetic_trainer.py','exec')

    def test_patcher_adds_log_not_reward_modification(self):
        changed = patch(SYNTHETIC_CALLBACK)
        self.assertEqual(changed.count("for info in self.locals.get('infos', []):"), 1)
        self.assertIn('self.locals.get(\'rewards\'', changed)
        self.assertNotIn('self.model.predict(', changed)
        self.assertNotIn('self.model.env.reset(', changed)

    def test_patcher_refuses_unknown_or_duplicate_installs(self):
        changed = patch(SYNTHETIC_CALLBACK)
        with self.assertRaises(ValueError):
            patch(changed)
        with self.assertRaises(ValueError):
            patch('not a trainer')


if __name__ == '__main__':
    unittest.main()
