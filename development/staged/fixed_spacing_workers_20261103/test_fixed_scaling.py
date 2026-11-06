import json
import os
from pathlib import Path
import tempfile
from types import SimpleNamespace
import unittest
import zipfile
from unittest.mock import Mock,patch

import numpy as np
from parallel_adapter import LAYOUTS,configuration,make_history,history_report
from run_workers import command
import benchmark_workers as batch
import results


class FixedScalingTests(unittest.TestCase):
    def test_contracts_and_flags(self):
        for layout,ages in LAYOUTS.items():
            with patch.dict(os.environ,RS_FIXED_LAYOUT=layout):
                c=configuration()
                self.assertEqual(c['size'],[384,384]);self.assertEqual(c['encoder'],'spatial')
                self.assertEqual(c['temporal']['frames'],len(ages))
                self.assertEqual(c['temporal']['context_offsets_seconds'],list(ages[1:]))
                self.assertEqual(c['temporal']['span_ms'],ages[-1]*1000)
        for n in range(1,6):
            args=command('source.zip','out',n,5)
            for absent in ('--preview-port','--benchmark-only','--allow-input-migration','--compact-observations'):
                self.assertNotIn(absent,args)
            self.assertEqual(int(args[args.index('--steps')+1])%(128*n),0)
            self.assertEqual(args[args.index('--end-plot')+1],'off')

    def test_smoke_first_then_every_count_and_repeat(self):
        checkpoints={name:Path(name+'.zip') for name in LAYOUTS}
        jobs=batch.jobs(Path('output'),checkpoints,5)
        self.assertEqual(len(jobs),7);self.assertEqual(sum(j['minutes'] for j in jobs),31)
        self.assertEqual(jobs[0]['phase'],'smoke')
        self.assertTrue(all(j['phase']!='smoke' for j in jobs[1:]))
        for layout in LAYOUTS:
            matching=[j for j in jobs if j['layout']==layout]
            self.assertEqual([j['workers'] for j in matching],[1,1,2,3,4,5,1])
            self.assertTrue(all(j['command'][j['command'].index('--checkpoint')+1]==str(checkpoints[layout]) for j in matching))

    def test_timestamp_selection_actual_ages_not_index_assumptions(self):
        for layout,ages in LAYOUTS.items():
            with patch.dict(os.environ,RS_FIXED_LAYOUT=layout):
                h=make_history(len(ages),ages[-1]);frame=np.zeros((3,2,2),np.uint8)
                h.observation(frame,0.)
                # Irregular 27 ms capture intervals and a pump frame newer than the anchor.
                stamps=np.arange(.027,1.06,.027)
                for i,stamp in enumerate(stamps):h.add(frame+i+1,float(stamp),h.token())
                image,aux=h.observation(frame+250,1.03)
                self.assertEqual(image.shape,(3*len(ages),2,2))
                self.assertTrue(np.all(aux[-len(ages):]==1))
                actual=aux[:len(ages)]*(ages[-1]+.06)
                targets=np.array(ages[::-1])
                self.assertTrue(np.all(actual>=targets-1e-6))
                self.assertTrue(np.all(actual-targets<=.027001))
                self.assertTrue(np.all(image[-3:]==250))
                for index,target in enumerate(targets[:-1]):
                    expected=int(np.flatnonzero(stamps<=1.03-target)[-1])+1
                    self.assertTrue(np.all(image[3*index:3*index+3]==expected))
                report=history_report(h)
                self.assertEqual(report['requested_slot_ages_ms'],list(targets*1000))
                self.assertLess(report['last_1000_target_lateness_ms'][-2]['max'],27.001)

    def test_startup_reset_and_inflight_generation_isolation(self):
        with patch.dict(os.environ,RS_FIXED_LAYOUT='fixed4'):
            h=make_history(4,.75);h.clock=lambda:10.
            frame=np.ones((3,2,2),np.uint8)
            _,aux=h.observation(frame,1.)
            np.testing.assert_equal(aux[-4:],[0,0,0,1])
            old=h.token();h.add(frame,1.1,old)
            h.clear();h.add(frame,10.1,old)
            self.assertEqual(len(h.items),0)
            image,aux=h.observation(frame*9,10.2)
            self.assertTrue(np.all(image[:-3]==0));np.testing.assert_equal(aux[-4:],[0,0,0,1])
            h.clear();h.observation(frame,10.3,enable=False)
            self.assertIsNone(h.token())

    def test_missing_history_not_faked_and_no_wait(self):
        with patch.dict(os.environ,RS_FIXED_LAYOUT='fixed4'):
            h=make_history(4,.75);frame=np.ones((3,2,2),np.uint8)
            h.observation(frame,0.);h.add(frame,.70,h.token())
            image,aux=h.observation(frame*2,1.)
            np.testing.assert_equal(aux[-4:],[0,1,0,1])
            self.assertTrue(np.all(image[:3]==0));self.assertEqual(h.packs,2)
            with self.assertRaises(ValueError):make_history(5,.75)

    def test_venv_parent_excluded_sibling_detected(self):
        argv=['python.exe','benchmark_workers.py','--execute']
        parent=Mock(pid=1);parent.name.return_value='python.exe';parent.cmdline.return_value=argv;parent.parent.return_value=None
        current=Mock(pid=2);current.cmdline.return_value=argv;current.parent.return_value=parent
        processes=[SimpleNamespace(pid=n,info={'name':'python.exe','cmdline':argv}) for n in (1,2,3)]
        with patch('psutil.Process',return_value=current),patch('psutil.process_iter',return_value=processes):
            self.assertEqual(batch.busy_processes(),[3])

    def test_cleanup_and_worst_worker_percentile(self):
        with tempfile.TemporaryDirectory() as tmp:
            folder=Path(tmp);(folder/'parallel_training.json').write_text('{"status":"STOPPED","finalization_errors":[]}')
            for i,(median,tail) in enumerate([(30,40),(50,90)]):
                worker=folder/f'worker_{i}';worker.mkdir()
                (worker/'hybrid_close.json').write_text('{"status":"COMPLETE"}')
                (worker/'result.json').write_text(json.dumps({'reaction_latency':{'metrics':{
                    'policy_request_to_command_ms':{'median_ms':median,'p95_ms':tail}}}}))
            self.assertTrue(results.clean(folder,2));self.assertFalse(results.clean(folder,3))
            row=results.summarize_case(dict(name='test',output=str(folder),workers=2,phase='performance'))
            self.assertEqual(row['worst_worker_p95_ms'],90)
            self.assertEqual(row['worker_median_range_ms'],[30,50])

    def test_real_update_and_exact_history_required_for_retention(self):
        with tempfile.TemporaryDirectory() as tmp:
            folder=Path(tmp);worker=folder/'worker_0';worker.mkdir()
            r=dict(status='COMPLETE',model_updated=True,source_unchanged=True,
                optimized_transitions=128,audits=[{'status':'PASS_SETTLED_LEARNING'}],finalization_errors=[])
            (folder/'parallel_training.json').write_text(json.dumps(r))
            (folder/'hybrid_benchmark.json').write_text(json.dumps(dict(source_unchanged=True,
                configuration=configuration(),input_previews=False)))
            (worker/'result.json').write_text('{}');(worker/'hybrid_close.json').write_text('{"status":"COMPLETE"}')
            history=dict(recent_frames=1,context_offsets_seconds=[.75,.25,.05])
            (worker/'hybrid_history.json').write_text(json.dumps(history))
            with zipfile.ZipFile(folder/'latest.zip','w') as archive:archive.writestr('test','payload')
            self.assertEqual(results.validate(folder,1)['optimized_transitions'],128)
            r['model_updated']=False;(folder/'parallel_training.json').write_text(json.dumps(r))
            with self.assertRaises(ValueError):results.validate(folder,1)
            r['model_updated']=True;(folder/'parallel_training.json').write_text(json.dumps(r))
            history['context_offsets_seconds']=[1.,.5,.05]
            (worker/'hybrid_history.json').write_text(json.dumps(history))
            with self.assertRaises(ValueError):results.validate(folder,1)


if __name__=='__main__':unittest.main()
