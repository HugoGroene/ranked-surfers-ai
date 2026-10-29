from types import SimpleNamespace
import unittest
from unittest.mock import AsyncMock, Mock

from tools.ppo_browser_session import PersistentBrowserSession
from tools.run_ppo_batch import manifest_runs, with_run_prefix


def environment():
    env = SimpleNamespace(args=SimpleNamespace(url='http://localhost:8000'))
    for key in ('playwright', 'browser', 'context', 'page', 'grabber', 'canvas'):
        setattr(env, key, None)
    return env


class BrowserSessionTests(unittest.TestCase):
    def test_resources_and_worker_survive_stage_release(self):
        session = PersistentBrowserSession()
        try:
            a = environment()
            session.acquire(a)
            a.browser = SimpleNamespace(is_connected=lambda: True, close=AsyncMock())
            a.page = SimpleNamespace(is_closed=lambda: False)
            a.context = SimpleNamespace(close=AsyncMock())
            a.playwright = SimpleNamespace(stop=AsyncMock())
            a.grabber = SimpleNamespace(close=Mock())
            original = dict((key, getattr(a, key)) for key in ('browser', 'page', 'context', 'playwright', 'grabber'))
            session.remember(a)
            thread = a.worker_thread
            session.release(a)
            self.assertIsNone(a.page)
            self.assertFalse(original['browser'].close.called)
            b = environment()
            session.acquire(b)
            self.assertIs(b.worker_thread, thread)
            self.assertTrue(session.restore(b))
            for key, value in original.items():
                self.assertIs(getattr(b, key), value)
            session.release(b)
            self.assertEqual(session.launches, 1)
            self.assertEqual(session.reuses, 1)
        finally:
            session.close()
        original['browser'].close.assert_awaited_once()
        original['context'].close.assert_awaited_once()
        original['playwright'].stop.assert_awaited_once()
        original['grabber'].close.assert_called_once()
        self.assertFalse(thread.is_alive())

    def test_concurrent_leases_are_rejected(self):
        session = PersistentBrowserSession()
        a = environment()
        try:
            session.acquire(a)
            with self.assertRaisesRegex(RuntimeError, 'already leased'):
                session.acquire(environment())
            session.release(a)
        finally:
            session.close()

    def test_external_site_rejected_before_worker_creation(self):
        session = PersistentBrowserSession()
        a = environment()
        a.args.url = 'https://example.com'
        with self.assertRaises(ValueError):
            session.acquire(a)
        self.assertIsNone(session.loop)
        session.close()

    def test_disconnected_browser_is_not_silently_replaced(self):
        session = PersistentBrowserSession()
        a = environment()
        try:
            session.acquire(a)
            a.browser = SimpleNamespace(is_connected=lambda: False, close=AsyncMock())
            a.page = SimpleNamespace(is_closed=lambda: False)
            session.remember(a)
            with self.assertRaisesRegex(RuntimeError, 'disconnected'):
                session.restore(a)
            session.release(a)
        finally:
            session.close()

    def test_batch_rejects_duplicate_runs_and_non_training_commands(self):
        with self.assertRaises(ValueError):
            manifest_runs({'stages': [{'args': ['--mode', 'probe', '--run-name', 'a']}]})
        with self.assertRaises(ValueError):
            manifest_runs({'stages': [{'args': ['--mode', 'eval', '--run-name', 'a']}]*2})

    def test_prefix_remaps_only_dependent_generated_checkpoints(self):
        manifest = {'stages': [
            {'args': ['--mode', 'train', '--run-name', 'old_train', '--resume', 'checkpoints/source.zip']},
            {'args': ['--mode', 'eval', '--run-name', 'old_eval', '--resume',
                      'ai_data/rl_v2_runs/old_train/latest.zip']}]}
        changed = with_run_prefix(manifest, 'new_run')
        self.assertEqual(manifest_runs(changed), ['new_run_01_train', 'new_run_02_eval'])
        self.assertIn('checkpoints/source.zip', changed['stages'][0]['args'])
        self.assertEqual(changed['stages'][1]['args'][-1].replace('\\', '/'),
                         'ai_data/rl_v2_runs/new_run_01_train/latest.zip')
        self.assertEqual(manifest_runs(manifest), ['old_train', 'old_eval'])

    def test_prefix_cannot_escape_run_directory(self):
        with self.assertRaises(ValueError):
            with_run_prefix({'stages': [{'args': ['--mode', 'eval', '--run-name', 'a']}]}, '../escape')
