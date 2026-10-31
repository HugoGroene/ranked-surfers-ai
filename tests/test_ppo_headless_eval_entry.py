import sys
import unittest
from types import SimpleNamespace
from unittest.mock import Mock, patch

from tools.evaluate_ppo_headless import main


class EntryTests(unittest.TestCase):
    def test_frozen_evaluation_and_failure_exit(self):
        trainer = SimpleNamespace(parse_args=Mock(return_value=SimpleNamespace(mode='eval')),
                                  evaluate=Mock(return_value=False))
        adapter = SimpleNamespace(install=Mock())
        with patch.dict(sys.modules, {'train_rs_v612_ppo': trainer,
                                     'tools.ppo_headless_env': adapter}):
            self.assertEqual(main(['--eval-episodes', '3']), 2)
        adapter.install.assert_called_once_with(trainer, 'electron', electron_transport='shared_memory')
        trainer.evaluate.assert_called_once()

    def test_training_rejected_before_browser(self):
        trainer = SimpleNamespace(parse_args=Mock(return_value=SimpleNamespace(mode='train')))
        adapter = SimpleNamespace(install=Mock())
        with patch.dict(sys.modules, {'train_rs_v612_ppo': trainer,
                                     'tools.ppo_headless_env': adapter}):
            with self.assertRaises(ValueError):
                main(['--mode', 'train'])
        adapter.install.assert_not_called()
