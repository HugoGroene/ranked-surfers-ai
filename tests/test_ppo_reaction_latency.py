import unittest
from types import SimpleNamespace
from unittest.mock import AsyncMock

from tools.audit_ppo_reaction_latency import Audit, GrabAudit, row_metrics


class ReactionLatencyTests(unittest.TestCase):
    def test_audit_detaches_from_browser_resource_retained_between_stages(self):
        original_grabber = object()
        original_press = AsyncMock()
        session = SimpleNamespace(resources={})
        raw = SimpleNamespace(grabber=original_grabber, _pack=lambda *args: None,
                              _capture=AsyncMock(), step=lambda action: None,
                              page=SimpleNamespace(keyboard=SimpleNamespace(press=original_press)),
                              _persistent_browser_session=session)
        audit = Audit()
        audit.attach(raw)
        session.resources['grabber'] = raw.grabber  # environment released its lease
        keyboard = raw.page.keyboard
        raw.grabber = None
        audit.detach()
        self.assertIs(session.resources['grabber'], original_grabber)
        self.assertIsNone(raw.grabber)
        self.assertIs(keyboard.press, original_press)

    def test_new_guard_frame_does_not_replace_policy_source_age(self):
        row = {'observation': {'capture': {'grab_end': 10.0}, 'detector_at': 9.9},
               'prediction_start': 10.002, 'prediction_end': 10.006,
               'step_start': 10.007, 'inner_start': 10.030,
               'command_start': 10.031, 'command_end': 10.040,
               'captures': [{'grab_end': 10.025, 'phase': 'guard'},
                            {'grab_end': 10.050, 'phase': 'post_action'}]}
        measured = row_metrics(row)
        self.assertAlmostEqual(measured['policy_rgb_age_at_command_ms'], 31.)
        self.assertAlmostEqual(measured['guard_rgb_age_at_command_ms'], 6.)
        self.assertAlmostEqual(measured['policy_inference_ms'], 4.)

    def test_suppressed_action_is_not_counted_as_a_command(self):
        row = {'observation': {'capture': {'grab_end': 10.0}},
               'prediction_start': 10.002, 'prediction_end': 10.006,
               'step_start': 10.007, 'captures': []}
        self.assertIsNone(row_metrics(row)['policy_rgb_age_at_command_ms'])
        row['metrics'] = row_metrics(row)
        audit = Audit()
        audit.rows.append(row)
        self.assertEqual(audit.summary()['movement_commands']['steps'], 0)

    def test_grab_proxy_returns_identical_object_and_forwards_region(self):
        sentinel = object()
        class FakeGrabber:
            def grab(self, region):
                self.region = region
                return sentinel
        original, audit = FakeGrabber(), Audit()
        proxy = GrabAudit(original, audit)
        region = {'left': 12}
        self.assertIs(proxy.grab(region), sentinel)
        self.assertIs(original.region, region)
        self.assertEqual(audit.latest_grab['id'], 1)
        self.assertLessEqual(audit.latest_grab['grab_start'], audit.latest_grab['grab_end'])
