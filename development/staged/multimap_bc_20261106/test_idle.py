from types import SimpleNamespace
from unittest.mock import Mock,patch
import unittest
from common import require_idle


class IdleTests(unittest.TestCase):
    def fixtures(self):
        command=['base/python.exe','-B','-s','record.py','--map','halloween','--split','train']
        shell=Mock(pid=1);shell.name.return_value='powershell.exe'
        launcher=Mock(pid=2);launcher.name.return_value='python.exe'
        launcher.cmdline.return_value=['venv/python.exe',*command[1:]]
        launcher.parent.return_value=shell
        current=Mock(pid=3);current.cmdline.return_value=command;current.parent.return_value=launcher
        entries=[SimpleNamespace(pid=pid,info={'name':'python.exe','cmdline':command}) for pid in (2,3)]
        return current,entries,command

    def test_own_windows_launcher_does_not_block(self):
        current,entries,_=self.fixtures()
        with patch('psutil.Process',return_value=current),patch('psutil.process_iter',return_value=entries):
            require_idle()

    def test_identical_sibling_recorder_still_blocks(self):
        current,entries,command=self.fixtures()
        entries.append(SimpleNamespace(pid=4,info={'name':'python.exe','cmdline':command}))
        with patch('psutil.Process',return_value=current),patch('psutil.process_iter',return_value=entries):
            with self.assertRaisesRegex(RuntimeError,r'PIDs: \[4\]'):require_idle()

    def test_different_parent_training_invocation_still_blocks(self):
        current,entries,_=self.fixtures()
        current.parent().cmdline.return_value=['python.exe','train.py']
        entries[0].info['cmdline']=['python.exe','train.py']
        with patch('psutil.Process',return_value=current),patch('psutil.process_iter',return_value=entries):
            with self.assertRaisesRegex(RuntimeError,r'PIDs: \[2\]'):require_idle()

    def test_export_exception_is_only_for_classifier(self):
        from common import ROOT
        current,entries,_=self.fixtures()
        entries.append(SimpleNamespace(pid=4,info={'name':'python.exe','cmdline':[
            'python.exe',str(ROOT/'development/experiments/multimap_classifier/train.py')]}))
        with patch('psutil.Process',return_value=current),patch('psutil.process_iter',return_value=entries):
            with self.assertRaises(RuntimeError):require_idle()
            require_idle(allow_classifier_training=True)
            entries[-1].info['cmdline'][1]=str(ROOT/'development/staged/multimap_bc_20261106/train.py')
            with self.assertRaises(RuntimeError):require_idle(allow_classifier_training=True)


if __name__=='__main__':unittest.main()
