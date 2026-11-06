import copy
import json
from pathlib import Path
import tempfile
import unittest
import zipfile

import capacity_suite as capacity


class CapacityTests(unittest.TestCase):
    def test_venv_launcher_excluded_but_identical_sibling_still_blocks(self):
        from types import SimpleNamespace
        from unittest.mock import Mock,patch
        command=['base/python.exe','-B','-s','capacity_suite.py','--execute']
        shell=Mock(pid=1);shell.name.return_value='powershell.exe'
        launcher=Mock(pid=2);launcher.name.return_value='python.exe'
        launcher.cmdline.return_value=['venv/python.exe',*command[1:]];launcher.parent.return_value=shell
        current=Mock(pid=3);current.cmdline.return_value=command;current.parent.return_value=launcher
        entries=[SimpleNamespace(pid=pid,info={'name':'python.exe','cmdline':command}) for pid in (2,3,4)]
        with patch('psutil.Process',return_value=current),patch('psutil.process_iter',return_value=entries):
            self.assertEqual(capacity.busy_processes(),[4])

    def test_different_parent_training_invocation_is_not_excluded(self):
        from types import SimpleNamespace
        from unittest.mock import Mock,patch
        current=Mock(pid=3);current.cmdline.return_value=['python.exe','capacity_suite.py','--execute']
        parent=Mock(pid=2);parent.name.return_value='python.exe'
        parent.cmdline.return_value=['python.exe','suite.py','--execute'];current.parent.return_value=parent
        entries=[SimpleNamespace(pid=2,info={'name':'python.exe','cmdline':parent.cmdline()})]
        with patch('psutil.Process',return_value=current),patch('psutil.process_iter',return_value=entries):
            self.assertEqual(capacity.busy_processes(),[2])

    def test_smoke_covers_every_spacing_briefly_performance_once_per_frame_count(self):
        smoke=capacity.phase_jobs('smoke',Path('smoke'))['jobs']
        performance=capacity.phase_jobs('performance',Path('performance'))['jobs']
        self.assertEqual(len(smoke),36);self.assertEqual(len(performance),10)
        self.assertEqual([j['mode'] for j in smoke],['train']*18+['eval']*18)
        self.assertTrue(all(capacity.value(j['command'],'--minutes')=='.5' or
                            capacity.value(j['command'],'--minutes')=='0.5' for j in smoke[:18]))
        self.assertTrue(all(capacity.value(j['command'],'--episodes')=='1' for j in smoke[18:]))
        for size in ('256','384'):
            counts=[]
            for job in performance:
                self.assertNotIn('--input-previews',job['command'])
                if capacity.value(job['command'],'--size')!=size:continue
                recent,older=capacity.LAYOUTS[capacity.value(job['command'],'--layout')]
                counts.append(recent+len(older))
            self.assertEqual(sorted(counts),[4,5,6,7,8])

    def fixture(self,root,cleanup=None,status='COMPLETE',minutes=1):
        cases=[(256,'recent4')]
        old={'jobs':capacity.full.jobs(cases,root/'old',minutes,True)}
        wanted={'jobs':capacity.full.jobs(cases,root/'new',.5,True)}
        for job in old['jobs']:
            folder=Path(job['output']);folder.mkdir(parents=True)
            report=dict(status=status,finalization_errors=[] if cleanup is None else cleanup,
                        audit={'status':'PASS_SETTLED_LEARNING'},completed_requested_episodes=True)
            name='trial_report.json' if job['mode']=='train' else 'v61_eval_report.json'
            (folder/name).write_text(json.dumps(report))
            if job['mode']=='train':
                with zipfile.ZipFile(folder/'latest.zip','w') as archive:archive.writestr('test','fixture')
        return old,wanted

    def test_reuse_completed_training_and_matching_evaluation_without_old_mutation(self):
        with tempfile.TemporaryDirectory() as tmp:
            old,wanted=self.fixture(Path(tmp));before=copy.deepcopy(old)
            result=capacity.reuse_completed(wanted,old)
            self.assertTrue(all(j.get('reused') for j in result['jobs']))
            self.assertEqual(result['jobs'][0]['output'],old['jobs'][0]['output'])
            self.assertTrue(result['retained_artifacts']);self.assertEqual(old,before)

    def test_incomplete_or_cleanup_failed_stages_are_not_reused(self):
        for kwargs in ({'status':'STOPPED'},{'cleanup':['environment failure']},{'minutes':.25}):
            with self.subTest(kwargs=kwargs),tempfile.TemporaryDirectory() as tmp:
                old,wanted=self.fixture(Path(tmp),**kwargs)
                result=capacity.reuse_completed(wanted,old)
                self.assertTrue(all(not j.get('reused') for j in result['jobs']))
                checkpoint=capacity.value(result['jobs'][1]['command'],'--checkpoint')
                self.assertEqual(checkpoint,str(Path(wanted['jobs'][0]['output'])/'latest.zip'))

    def test_changed_seed_cannot_be_reused(self):
        with tempfile.TemporaryDirectory() as tmp:
            old,wanted=self.fixture(Path(tmp))
            for job in old['jobs']:
                cmd=job['command'];cmd[cmd.index('--seed')+1]='456'
            result=capacity.reuse_completed(wanted,old)
            self.assertTrue(all(not j.get('reused') for j in result['jobs']))

    def test_longer_completed_performance_probe_is_retained(self):
        with tempfile.TemporaryDirectory() as tmp:
            old,wanted=self.fixture(Path(tmp),minutes=4)
            old['jobs']=old['jobs'][:1];wanted['jobs']=wanted['jobs'][:1]
            for plan in (old,wanted):plan['jobs'][0]['command'].remove('--input-previews')
            cmd=wanted['jobs'][0]['command'];cmd[cmd.index('--minutes')+1]='3'
            result=capacity.reuse_completed(wanted,old)
            self.assertTrue(result['jobs'][0]['reused'])
            self.assertEqual(capacity.value(result['jobs'][0]['command'],'--minutes'),'4')


if __name__=='__main__':unittest.main()
