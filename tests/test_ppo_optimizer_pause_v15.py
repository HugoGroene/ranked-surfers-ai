"""Offline tests for experimental PPO pause and reward-time compensation."""
import unittest

from tools.ppo_optimizer_pause_clock import compensate_pause_clocks
from tools.install_ppo_optimizer_pause import patch


class Context:
    def __init__(self, times=None):
        self.last_action_times = [None, 50.0, None, 70.0, 95.0] if times is None else times


class Env:
    def __init__(self):
        self.episode_start = 100.0
        self.last_step = 102.0
        self.episode_started = 200.0
        self.suspected_since = 103.0
        self.previous_observation_at = 205.0
        self._prev_step_start = 205.5
        self.action_context = Context()


class Monitor:
    def __init__(self, env):
        self.env = env
        self.t_start = 10000.0


class ClockTests(unittest.TestCase):
    def setUp(self):
        self.env = Env()
        self.monitor = Monitor(self.env)

    def correct(self, dm=1.0, dp=1.0, dw=1.0):
        return compensate_pause_clocks(self.env, dm, dp, monitor=self.monitor, elapsed_wall=dw)

    def test_survival_clock_excludes_pause(self):
        # Game is paused at actual monotonic=104, resumed at 106.
        # At 106.1 seconds the gameplay duration must be 4.1, not 6.1.
        self.correct(2., 2., 2.)
        self.assertAlmostEqual(106.1 - self.env.episode_start, 4.1)
        self.assertAlmostEqual(self.env.last_step, 104.)

    def test_reward_no_survival_or_milestone_bonus_for_pause(self):
        # Equivalent to ProgressRewards from rs_v61_rewards.py: reward by
        # positive duration delta plus 0.5s milestones. Both unchanged.
        # Source file is available in real project; this synthetic assertion
        # isolates the exact duration argument supplied to ProgressRewards.
        elapsed_before = 4.0
        self.env.episode_start = 100.0
        before = 104.0 - self.env.episode_start
        self.correct(2., 2., 2.)
        after = 106.0 - self.env.episode_start
        reward_step_seconds = min(0.15, max(0.0, after-before))
        self.assertAlmostEqual(before, elapsed_before)
        self.assertAlmostEqual(after, elapsed_before)
        self.assertEqual(reward_step_seconds, 0)
        self.assertEqual(int(after//0.5)-int(before//0.5), 0)

    def test_all_clocks_shift_together(self):
        r = self.correct(1.25, 1.25, 1.25)
        self.assertEqual(self.env.episode_start, 101.25)
        self.assertEqual(self.env.last_step, 103.25)
        self.assertEqual(self.env.episode_started, 201.25)
        self.assertEqual(self.env.suspected_since, 104.25)
        self.assertEqual(self.env.previous_observation_at, 206.25)
        self.assertEqual(self.env._prev_step_start, 206.75)
        self.assertEqual(self.env.action_context.last_action_times,
                         [None,51.25,None,71.25,96.25])
        self.assertEqual(self.monitor.t_start, 10001.25)
        self.assertTrue(r['corrected_monitor_timer'])

    def test_optional_clocks_none(self):
        self.env.suspected_since = None
        self.env.previous_observation_at = None
        self.env._prev_step_start = None
        self.correct()
        self.assertIsNone(self.env.suspected_since)
        self.assertIsNone(self.env.previous_observation_at)
        self.assertIsNone(self.env._prev_step_start)

    def test_action_timing_feature_excludes_pause(self):
        self.env.action_context.last_action_times = [None, None, None, None, 195.]
        before = 205. - 195.
        self.correct(2.,2.,2.)
        after = 207. - self.env.action_context.last_action_times[4]
        self.assertEqual(after, before)

    def test_wrong_monitor_does_not_mutate(self):
        wrong = Monitor(Env())
        with self.assertRaises(ValueError):
            compensate_pause_clocks(self.env,1.,1.,monitor=wrong,elapsed_wall=1.)
        self.assertEqual(self.env.episode_start,100.)
        self.assertEqual(self.monitor.t_start,10000.)

    def test_malformed_context_does_not_partially_shift(self):
        self.env.action_context.last_action_times[1] = float('nan')
        with self.assertRaises(ValueError):
            self.correct()
        self.assertEqual(self.env.episode_start,100.)
        self.assertEqual(self.monitor.t_start,10000.)

    def test_malformed_monitor_does_not_partially_shift(self):
        self.monitor.t_start = float('nan')
        with self.assertRaises(ValueError):
            self.correct()
        self.assertEqual(self.env.episode_start,100.)

    def test_rejects_excessive_pause_and_clocks_disagree(self):
        for args in [(0.,1.,1.), (31.,31.,31.), (2.,1.,2.), (1.,1.,3.), (1.,1.,float('nan'))]:
            with self.subTest(args=args), self.assertRaises(ValueError):
                self.correct(*args)
            self.assertEqual(self.env.episode_start,100.)


SYNTHETIC = '''import argparse
import time
import json
class GuardUncertainError(RuntimeError):
    pass
def build_env_class():
    class PPOV61Env:
        def refresh_after_ppo_optimizer(self):
            pass
        def confirm_post_optimizer_menu(self):
            from tools.ppo_menu_guard_alignment_v14 import (
                accept_menu_after_transient, MAX_CONFIRM_CAPTURES)
        def _guard_diagnostic(self, readings, *, reason, name=None):
            pass

def build_callback(a, output, raw=None):
    from tools.ppo_rollout_gap_audit import RolloutGapTracker
    class V61Callback:
        def _on_rollout_end(self):
            try:
                self.rollout_gap_tracker.finish_rollout(time.perf_counter())
            except Exception as exc:
                print(f"[PPO GAP] rollout-end diagnostic failed: {exc}", flush=True)

        def _on_rollout_start(self):
            # This runs after the preceding PPO gradient update.
            try:
                record = self.rollout_gap_tracker.begin_rollout(
                    time.perf_counter(), timesteps=int(self.num_timesteps))
                if record is not None:
                    print("[PPO GAP] " + json.dumps(record), flush=True)
                    with self.rollout_gap_log_path.open('a', encoding='utf-8') as handle:
                        handle.write(json.dumps(record) + "\\n")
            except Exception as exc:
                print(f"[PPO GAP] rollout-start diagnostic failed: {exc}", flush=True)
            # Run after the optimizer, before SB3 predicts its first action.
            # Only after a completed rollout and only when explicitly enabled.
            if a.resync_rollouts and record is not None:
                try:
                    raw.refresh_after_ppo_optimizer()
                except Exception as exc:
                    raise GuardUncertainError(f"[PPO RESYNC] Safe stop: {exc}") from exc

        def _on_step(self):
            pass

def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument('--resync-rollouts', action='store_true',
                   help='EXPERIMENTAL: passively rebuild 4-frame observation after PPO updates')
    a=p.parse_args()
    a.crop=bench.validate_crop(a.crop)
    return a
'''


class InstallerTests(unittest.TestCase):
    def test_patch_compiles_synthetic_instrumented_trainer(self):
        patched = patch(SYNTHETIC)
        compile(patched, 'train_rs_v612_ppo.py', 'exec')
        self.assertIn('def pause_for_ppo_optimizer', patched)
        self.assertIn('def resume_after_ppo_optimizer', patched)
        self.assertIn("monitor = vec.envs[0]", patched)
        self.assertIn('a.pause_ppo_optimizer and record is not None', patched)

    def test_rollout_hooks_are_in_correct_order(self):
        patched = patch(SYNTHETIC)
        end = patched[patched.index('def _on_rollout_end(self):'):
                      patched.index('def _on_rollout_start(self):')]
        start = patched[patched.index('def _on_rollout_start(self):'):
                        patched.index('def _on_step(self):')]
        self.assertLess(end.index('finish_rollout'), end.index('pause_for_ppo_optimizer'))
        self.assertLess(start.index('resume_after_ppo_optimizer'),
                        start.index('refresh_after_ppo_optimizer'))
        self.assertIn('record = None', start)

    def test_pause_uses_site_controls_never_escape(self):
        patched = patch(SYNTHETIC)
        self.assertIn("button[aria-label=\"Pause game\"]", patched)
        self.assertIn(".game-pause-panel[aria-label=\"Game paused\"]", patched)
        self.assertIn("get_by_role('button', name='Resume', exact=True)", patched)
        self.assertNotIn('keyboard.press(\'Escape\')', patched)
        self.assertNotIn('force=True', patched)
        self.assertIn('mean_change > .25', patched)

    def test_rejects_double_install_and_unexpected_source(self):
        with self.assertRaises(ValueError):
            patch(patch(SYNTHETIC))
        with self.assertRaises(ValueError):
            patch('def foo(): pass')

    def test_opt_in_off_by_default(self):
        patched = patch(SYNTHETIC)
        self.assertIn("p.add_argument('--pause-ppo-optimizer', action='store_true'", patched)
        self.assertIn("a.mode != 'train' or not a.resync_rollouts", patched)
        self.assertNotIn('train_rs_v612_ppo.py.pre_ppo_timing_audit.bak', patched)


if __name__ == '__main__':
    unittest.main()
