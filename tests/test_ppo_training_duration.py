import contextlib
import io
import unittest
from unittest.mock import patch
import train_rs_v612_ppo as trainer
import tools.train_ppo_parallel as parallel

class DurationTests(unittest.TestCase):
    def test_single_accepts_48_hours_and_rejects_more(self):
        with patch('pathlib.Path.is_file', return_value=True):
            self.assertEqual(trainer.parse_args(['--mode','train','--max-minutes','2880']).max_minutes,2880)
            with contextlib.redirect_stderr(io.StringIO()), self.assertRaises(SystemExit):
                trainer.parse_args(['--mode','train','--max-minutes','2881'])

    def test_parallel_accepts_48_hours_before_asset_loading(self):
        args=['--headless','--displays','1','--checkpoint','source.zip','--report','new_report','--max-minutes','2880']
        with patch('pathlib.Path.is_file',return_value=True), patch('pathlib.Path.exists',return_value=False), patch.object(parallel,'validate_checkpoint',side_effect=RuntimeError('preflight reached')):
            with self.assertRaisesRegex(RuntimeError,'preflight reached'):parallel.main(args)
            args[-1]='2881'
            with contextlib.redirect_stderr(io.StringIO()), self.assertRaises(SystemExit):parallel.main(args)
