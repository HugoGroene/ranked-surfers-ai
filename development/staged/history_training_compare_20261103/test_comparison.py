import json
import os
from pathlib import Path
import tempfile
from types import SimpleNamespace
import unittest
from unittest.mock import Mock,patch
import zipfile

import numpy as np
from parallel_adapter import LAYOUTS,configuration,make_history,history_report
import run_comparison as batch
from run_workers import command
from resume_state import continuation


def fake_zip(path,steps,contract=None):
    with zipfile.ZipFile(path,'w') as archive:
        archive.writestr('data',json.dumps(dict(num_timesteps=steps,observation_contract=contract or {'layout':'test'})))


class ComparisonTests(unittest.TestCase):
    def test_correct_two_fixed_schedules(self):
        expected={'fixed4':(4,[.05,.25,.75],4),'fixed5':(5,[.05,.20,.45,.95],3)}
        for layout,(count,context,workers) in expected.items():
            with patch.dict(os.environ,RS_COMPARISON_LAYOUT=layout):
                c=configuration();self.assertEqual(c['size'],[384,384]);self.assertEqual(c['encoder'],'spatial')
                self.assertEqual(c['temporal']['frames'],count);self.assertEqual(c['temporal']['recent_frames'],1)
                self.assertEqual(c['temporal']['context_offsets_seconds'],context)
                self.assertEqual(LAYOUTS[layout]['workers'],workers)
                history=make_history(count,max(context))
                frame=np.zeros((3,2,2),np.uint8);history.observation(frame,0.)
                for i,stamp in enumerate(np.arange(.027,1.08,.027)):
                    history.add(frame+i+1,float(stamp),history.token())
                image,aux=history.observation(frame+200,1.05)
                self.assertEqual(image.shape,(count*3,2,2));self.assertTrue(np.all(aux[-count:]==1))
                actual=aux[:count]*(max(context)+.06);targets=np.array(context[::-1]+[0])
                self.assertTrue(np.all(actual>=targets-1e-6));self.assertTrue(np.all(actual-targets<.02701))
                self.assertEqual(history_report(history)['requested_slot_ages_ms'],list(targets*1000))

    def test_both_smokes_before_long_runs_and_fresh_sources(self):
        checkpoints={name:Path(name+'.zip') for name in LAYOUTS}
        jobs=batch.jobs(Path('out'),checkpoints,260)
        self.assertEqual([j['phase'] for j in jobs],['smoke','smoke','train','train'])
        self.assertEqual([j['workers'] for j in jobs],[4,3,4,3])
        self.assertEqual([j['minutes'] for j in jobs],[1,1,260,260])
        self.assertEqual(sum(j['minutes'] for j in jobs),522)
        for job in jobs:
            self.assertEqual(job['command'][job['command'].index('--checkpoint')+1],str(checkpoints[job['layout']]))
            self.assertEqual(job['source_timesteps'],0)
        args=command('initial.zip','out',4,260)
        for forbidden in ('--benchmark-only','--preview-port','--allow-input-migration'):
            self.assertNotIn(forbidden,args)
        self.assertEqual(args[args.index('--end-plot')+1],'save')
        self.assertEqual(args[args.index('--detector-mode')+1],'off')
        self.assertEqual(args[args.index('--save-every-minutes')+1],'2')

    def test_startup_padding_and_generation_reset(self):
        with patch.dict(os.environ,RS_COMPARISON_LAYOUT='fixed5'):
            history=make_history(5,.95);history.clock=lambda:10.
            frame=np.ones((3,2,2),np.uint8)
            _,aux=history.observation(frame,0.)
            np.testing.assert_array_equal(aux[-5:],[0,0,0,0,1])
            token=history.token();history.clear();history.add(frame,10.1,token)
            image,aux=history.observation(frame,10.2)
            self.assertTrue(np.all(image[:-3]==0));np.testing.assert_array_equal(aux[-5:],[0,0,0,0,1])

    def test_resume_budget_counts_only_recovered_checkpoint(self):
        with tempfile.TemporaryDirectory() as tmp:
            folder=Path(tmp);initial=folder/'initial.zip';fake_zip(initial,0)
            previous=batch.jobs(folder,dict(fixed4=initial,fixed5=initial),260)[2]
            run=Path(previous['output']);run.mkdir();fake_zip(run/'latest.zip',1024)
            (run/'parallel_training.json').write_text(json.dumps(dict(status='STOPPED',optimized_transitions=1536,active_wall_seconds=90.)))
            (run/'round_timings.jsonl').write_text(json.dumps(dict(rounds=2,elapsed_seconds=60.))+'\n{"partial')
            new=batch.jobs(folder/'resumed',dict(fixed4=initial,fixed5=initial),260)[2]
            resumed=continuation(previous,new)
            self.assertEqual(resumed['minutes'],259);self.assertEqual(resumed['carried_seconds'],60.)
            self.assertEqual(resumed['source_timesteps'],1024)
            self.assertEqual(resumed['output'],new['output'])
            self.assertEqual(resumed['command'][resumed['command'].index('--checkpoint')+1],str(run/'latest.zip'))
            # A second interruption must retain previous saved progress in the budget.
            second=Path(resumed['output']);second.mkdir(parents=True);fake_zip(second/'latest.zip',1536)
            (second/'parallel_training.json').write_text(json.dumps(dict(status='STOPPED',optimized_transitions=512,active_wall_seconds=30.)))
            again=batch.jobs(folder/'again',dict(fixed4=initial,fixed5=initial),260)[2]
            recovered=continuation(resumed,again)
            self.assertEqual(recovered['minutes'],258.5);self.assertEqual(recovered['carried_seconds'],90.)
            self.assertEqual(recovered['source_timesteps'],1536)

    def test_resume_rejects_unaccounted_or_wrong_schema_checkpoint(self):
        with tempfile.TemporaryDirectory() as tmp:
            folder=Path(tmp);initial=folder/'initial.zip';fake_zip(initial,0)
            previous=batch.jobs(folder,dict(fixed4=initial,fixed5=initial),260)[2]
            run=Path(previous['output']);run.mkdir();fake_zip(run/'latest.zip',512,{'different':True})
            with self.assertRaises(ValueError):continuation(previous,previous)
            fake_zip(run/'latest.zip',512)
            with self.assertRaises(ValueError):continuation(previous,previous)

    def test_venv_launcher_not_confused_with_competing_run(self):
        argv=['python.exe','run_comparison.py','--execute']
        parent=Mock(pid=1);parent.name.return_value='python.exe';parent.cmdline.return_value=argv;parent.parent.return_value=None
        current=Mock(pid=2);current.cmdline.return_value=argv;current.parent.return_value=parent
        processes=[SimpleNamespace(pid=n,info={'name':'python.exe','cmdline':argv}) for n in (1,2,3)]
        with patch('psutil.Process',return_value=current),patch('psutil.process_iter',return_value=processes):
            self.assertEqual(batch.busy_processes(),[3])


if __name__=='__main__':unittest.main()
