"""Offline-only tests against the unchanged 4.61s-compatible reward and audit code."""
import unittest

from tools.ppo_rollout_gap_audit import RolloutGapTracker
from tools.analyze_ppo_rollout_gaps import percentile
from tools.ppo_reward_timing_probe import evaluate
from rs_v61_rewards import ProgressRewards


class RolloutGapTests(unittest.TestCase):
    def test_first_rollout_is_not_optimizer_gap(self):
        t = RolloutGapTracker()
        self.assertIsNone(t.begin_rollout(15, timesteps=0))

    def test_timing_after_ppo_optimizer_update(self):
        t = RolloutGapTracker()
        t.record_step(10.10, capture_at=10.08, done=False)
        t.finish_rollout(10.12)
        r = t.begin_rollout(12.15, timesteps=128)
        self.assertEqual(r['rollout_end_to_next_start_ms'], 2030.0)
        self.assertEqual(r['last_env_step_to_next_start_ms'], 2050.0)
        self.assertEqual(r['cached_frame_to_next_start_ms'], 2070.0)
        self.assertTrue(r['potentially_stale'])
        self.assertFalse(r['last_env_step_done'])

    def test_short_pause_does_not_signal_warning(self):
        t = RolloutGapTracker()
        t.record_step(5.02, capture_at=5.0, done=True)
        t.finish_rollout(5.03)
        r = t.begin_rollout(5.09, timesteps=128)
        self.assertFalse(r['potentially_stale'])
        self.assertTrue(r['last_env_step_done'])

    def test_gap_consumed_once(self):
        t = RolloutGapTracker()
        t.record_step(5.0)
        t.finish_rollout(5.1)
        self.assertIsNotNone(t.begin_rollout(6.0, timesteps=128))
        self.assertIsNone(t.begin_rollout(6.1, timesteps=128))

    def test_missing_capture_is_not_falsely_stale(self):
        t = RolloutGapTracker()
        t.record_step(2.0)
        t.finish_rollout(2.1)
        r = t.begin_rollout(4.0, timesteps=128)
        self.assertIsNone(r['cached_frame_to_next_start_ms'])
        self.assertFalse(r['potentially_stale'])

    def test_percentile(self):
        self.assertEqual(percentile([10, 20, 30], 95), 29.0)


class RewardTimingTests(unittest.TestCase):
    def test_same_survival_has_same_undiscounted_reward_below_cap(self):
        self.assertAlmostEqual(evaluate(5, .05)['total_reward'],
                               evaluate(5, .10)['total_reward'], places=4)

    def test_per_step_cap_can_change_total_reward(self):
        self.assertLess(evaluate(5, .20)['total_reward'],
                        evaluate(5, .10)['total_reward'])

    def test_discounting_depends_on_number_of_decisions(self):
        self.assertNotAlmostEqual(evaluate(5, .05)['discounted_step_return'],
                                  evaluate(5, .10)['discounted_step_return'])

    def test_death_penalty_is_not_action_dependent(self):
        r = ProgressRewards()
        reward, parts = r.step(.05, died=True)
        self.assertEqual(parts['death'], -1.0)
        self.assertEqual(parts['milestones'], 0.0)
        self.assertAlmostEqual(reward, -0.96)


if __name__ == '__main__':
    unittest.main()
