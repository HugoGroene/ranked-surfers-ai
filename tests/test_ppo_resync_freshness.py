from types import SimpleNamespace
import unittest
from unittest.mock import Mock
from rs_v612_gameplay_guard import GuardUncertainError
from tools.ppo_resync_freshness import verified_refresh


class FreshnessTests(unittest.TestCase):
    def test_exact_failed_age_retries_without_accepting_old_stack(self):
        raw=SimpleNamespace(refresh_after_ppo_optimizer=Mock(side_effect=[
            ('old',None,1-.1178),('new',None,1-.015)]))
        self.assertEqual(verified_refresh(raw,clock=lambda:1),('new',None,.985))
        self.assertEqual(raw.refresh_after_ppo_optimizer.call_count,2)

    def test_persistent_stale_stops_at_bound(self):
        raw=SimpleNamespace(refresh_after_ppo_optimizer=Mock(return_value=('old',None,.8)))
        with self.assertRaises(GuardUncertainError):
            verified_refresh(raw,clock=lambda:1)
        self.assertEqual(raw.refresh_after_ppo_optimizer.call_count,3)
        self.assertTrue(raw._guard_hard_stopped)

    def test_menu_or_uncertain_is_not_blindly_retried(self):
        raw=SimpleNamespace(refresh_after_ppo_optimizer=Mock(return_value=(None,'MENU',.9)))
        self.assertEqual(verified_refresh(raw,clock=lambda:1),(None,'MENU',.9))
        self.assertEqual(raw.refresh_after_ppo_optimizer.call_count,1)

    def test_pause_visual_reference_survives_stale_stack(self):
        reference=object()
        raw=SimpleNamespace(_ppo_pause_visual_reference=reference)
        def refresh():
            self.assertIs(raw._ppo_pause_visual_reference,reference)
            raw._ppo_pause_visual_reference=None
            return ('fresh' if fn.call_count==2 else 'old',None,.99 if fn.call_count==2 else .8)
        fn=Mock(side_effect=refresh)
        raw.refresh_after_ppo_optimizer=fn
        self.assertEqual(verified_refresh(raw,clock=lambda:1)[0],'fresh')

    def test_future_timestamp_stops_without_retry(self):
        raw=SimpleNamespace(refresh_after_ppo_optimizer=Mock(return_value=('invalid',None,2)))
        with self.assertRaises(GuardUncertainError):
            verified_refresh(raw,clock=lambda:1)
        self.assertEqual(raw.refresh_after_ppo_optimizer.call_count,1)


if __name__ == '__main__':
    unittest.main()
