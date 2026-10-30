"""Offline, side-effect-free regressions for fail-closed patch wiring."""
import sys
import unittest
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from tools.harden_ppo_rollout_resync import OLD, NEW, patch


class ResyncHardeningTests(unittest.TestCase):
    def sample(self):
        return (
            'def refresh_after_ppo_optimizer(self):\n'
            '    pass\n'
            'def helper(a):\n'
            '    # --resync-rollouts\n'
            '    if True:\n'
            '        try:\n'
            + OLD +
            '        except Exception:\n'
            '            pass\n'
        )

    def test_reject_skipped_refresh(self):
        out = patch(self.sample())
        self.assertNotIn('existing pre-action guard remains active', out)
        self.assertIn('Stopping instead of acting on the stale cached image', out)
        self.assertIn('if observation is None:', out)

    def test_check_age_before_replacement(self):
        out = patch(self.sample())
        self.assertLess(out.index('if not (0 <= age_ms < 100.0):'),
                        out.index('replace_cached_observation(self.model, observation)'))

    def test_fails_when_prior_patch_missing(self):
        with self.assertRaisesRegex(ValueError, 'not installed'):
            patch('def something_else():\n    pass\n')

    def test_fails_closed_when_patch_twice(self):
        with self.assertRaisesRegex(ValueError, 'already installed'):
            patch(patch(self.sample()))

    def test_compiles(self):
        compile(patch(self.sample()), '<resync-hardened-fixture>', 'exec')

    def test_does_not_change_unrelated_source(self):
        out = patch(self.sample())
        self.assertEqual(out.replace(NEW, OLD), self.sample())


if __name__ == '__main__':
    unittest.main()
