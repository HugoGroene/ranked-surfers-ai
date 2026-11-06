import unittest
import numpy as np
from hybrid import HybridHistory,LAYOUTS

def pixel(value):return np.full((3,2,2),value,np.uint8)

class HybridTests(unittest.TestCase):
    def test_sparse4_uses_only_current_plus_timestamp_targets(self):
        h=HybridHistory(*LAYOUTS['sparse4'])
        h.observation(pixel(1),1.)
        for i in range(1,121):h.add(pixel(i),1+i*.01,h.token())
        # Extra decision observations must not enter the fixed-age slots.
        for stamp in (2.1,2.15):h.observation(pixel(250),stamp)
        image,aux=h.observation(pixel(200),2.2)
        np.testing.assert_array_equal(image[-3:],pixel(200))
        self.assertNotIn(250,image)
        self.assertEqual(sum(aux[-4:]),4)
        np.testing.assert_allclose(aux[:4]*1060,[1000,500,100,0],atol=10.01)

    def test_all_startup_shapes_and_immediate_latest(self):
        for recent,old in LAYOUTS.values():
            h=HybridHistory(recent,old)
            image,aux=h.observation(pixel(9),1.)
            self.assertEqual(image.shape,(3*(recent+len(old)),2,2))
            self.assertEqual(aux[-1],1)
            self.assertEqual(sum(aux[h.frames:]),1)
            np.testing.assert_array_equal(image[-3:],pixel(9))
            self.assertFalse(image[:-3].any())

    def test_exact_older_context_and_recent_sequence(self):
        h=HybridHistory(3,(.2,.6,1.))
        h.observation(pixel(1),1.)
        for i in range(1,101):h.add(pixel(i),1+i*.01,h.token())
        h.observation(pixel(101),1.94);h.observation(pixel(102),1.97)
        image,aux=h.observation(pixel(103),2.)
        # Earliest 1.0 source was only a decision frame, not pumped. Mark missing.
        self.assertEqual(aux[6],0)
        self.assertEqual(image[3,0,0],40)
        self.assertEqual(image[6,0,0],80)
        np.testing.assert_array_equal(image[-9::3,0,0],[101,102,103])
        self.assertEqual(aux[-1],1)

    def test_no_future_capture_leaks(self):
        h=HybridHistory(3,(.2,.6,1.));h.observation(pixel(1),1.)
        h.add(pixel(99),2.,h.token())
        image,_=h.observation(pixel(3),1.5)
        self.assertNotIn(99,image)

    def test_generation_reset_clears_both_histories(self):
        h=HybridHistory(4,(.2,.6,1.),clock=lambda:2.)
        h.observation(pixel(7),1.);old=h.token();h.add(pixel(8),1.1,old)
        h.clear();h.add(pixel(9),2.1,old)
        image,aux=h.observation(pixel(10),2.2)
        self.assertFalse(image[:-3].any());self.assertEqual(sum(aux[h.frames:]),1)

    def test_disabled_resync_does_not_seed_history(self):
        h=HybridHistory(4,(.2,.6,1.))
        h.observation(pixel(9),1.,enable=False)
        self.assertIsNone(h.token())
        image,_=h.observation(pixel(3),1.1)
        self.assertNotIn(9,image)

    def test_overlap_does_not_duplicate_recent_into_context(self):
        h=HybridHistory(3,(.2,))
        h.observation(pixel(1),1.);h.add(pixel(2),1.1,h.token())
        h.observation(pixel(2),1.1)
        _,aux=h.observation(pixel(3),1.3)
        self.assertEqual(aux[4],0)

    def test_context_tolerance_rejects_wrong_age(self):
        h=HybridHistory(3,(.2,));h.observation(pixel(1),1.)
        h.add(pixel(2),1.1,h.token());h.observation(pixel(3),1.9)
        _,aux=h.observation(pixel(4),2.)
        self.assertEqual(aux[4],0)

    def test_context_capacity_bounded(self):
        h=HybridHistory(4,(.2,.4,.7,1.));h.observation(pixel(1),1.)
        for i in range(1,2001):h.add(pixel(2),1+i*.005,h.token())
        self.assertLessEqual(len(h.items),256)
        self.assertGreaterEqual(h.items[0][0],11-1.06)

    def test_offset_validation(self):
        for offsets in ((1.,.2),(.2,.2),(float('nan'),),(-.2,),(3.,)):
            with self.assertRaises(ValueError):HybridHistory(4,offsets)

    def test_report_and_latest_age(self):
        h=HybridHistory(4,())
        for i in range(50):h.observation(pixel(i),1+i*.04)
        report=h.report()
        self.assertEqual(report['valid_fraction_by_slot'][-1],1)
        self.assertEqual(report['last_1000_slot_ages'][-1]['median_ms'],0)
        self.assertAlmostEqual(report['last_1000_slot_ages'][0]['median_ms'],120)

    def test_duplicate_observation_not_extra_motion(self):
        h=HybridHistory(4,());h.observation(pixel(1),1.);h.observation(pixel(1),1.)
        _,aux=h.observation(pixel(2),1.1)
        self.assertEqual(sum(aux[4:]),2)

    def test_four_frame_baseline_does_not_store_unused_pump_images(self):
        h=HybridHistory(4,());h.observation(pixel(1),1.);h.add(pixel(2),1.1,h.token())
        self.assertEqual(len(h.items),0)

class PlanTests(unittest.TestCase):
    def test_smoke_failure_prevents_performance_phase(self):
        import json
        from pathlib import Path
        import tempfile
        from unittest.mock import patch
        import suite
        with tempfile.TemporaryDirectory() as tmp:
            folder=Path(tmp)
            (folder/'plan.json').write_text(json.dumps({'cases':[]}))
            for phase in ('smoke','performance'):
                (folder/phase).mkdir()
                (folder/phase/'plan.json').write_text(json.dumps({'jobs':[]}))
            with patch('sys.argv',['suite','--plan',str(folder/'plan.json'),'--execute']), \
                    patch.object(suite.subprocess,'run'), \
                    patch.object(suite.engine,'execute',side_effect=RuntimeError('smoke failure')) as execute:
                with self.assertRaisesRegex(RuntimeError,'smoke failure'):suite.main()
            self.assertEqual(execute.call_count,1)
            self.assertEqual(execute.call_args.args[1],folder/'smoke')
            self.assertFalse(execute.call_args.kwargs['keep_going'])

    def test_resume_retains_completed_checkpoints_and_reroutes_missing_evaluations(self):
        import json
        from pathlib import Path
        import tempfile
        from unittest.mock import patch
        import zipfile
        import suite
        with tempfile.TemporaryDirectory() as tmp:
            folder=Path(tmp)/'old';folder.mkdir();destination=Path(tmp)/'new'
            plan={'jobs':suite.jobs([(256,'recent4_context4')],folder,1,True)}
            train=Path(plan['jobs'][0]['output']);train.mkdir(parents=True)
            (train/'trial_report.json').write_text(json.dumps({'status':'COMPLETE','audit':{'status':'PASS_SETTLED_LEARNING'}}))
            with zipfile.ZipFile(train/'latest.zip','w') as archive:archive.writestr('fake','offline fixture')
            (folder/'plan.json').write_text(json.dumps(plan))
            (folder/'sources.json').write_text('{}')
            with patch.object(suite.engine,'source_hashes',return_value={}):
                result=suite.engine.continue_smoke(folder/'plan.json',destination,same_sources=True)
            self.assertTrue(result['jobs'][0]['reused'])
            self.assertEqual(result['jobs'][0]['output'],str(train))
            evaluation=result['jobs'][1];cmd=evaluation['command']
            self.assertEqual(cmd[cmd.index('--checkpoint')+1],str(train/'latest.zip'))
            self.assertTrue(Path(evaluation['output']).is_relative_to(destination))

    def test_all_smoke_training_before_checkpoint_evals(self):
        from pathlib import Path
        from suite import jobs
        cases=[(256,k) for k in LAYOUTS]
        result=jobs(cases,Path('example'),1,True)
        count=len(cases)
        self.assertEqual([j['mode'] for j in result],['train']*count+['eval']*count)
        self.assertTrue(all('--checkpoint' in j['command'] for j in result[count:]))
        self.assertTrue(all('--input-previews' in j['command'] for j in result[:count]))

    def test_performance_probes_have_no_incompatible_checkpoint(self):
        from pathlib import Path
        from suite import jobs
        for job in jobs([(384,'recent4_context4')],Path('example'),3,False):
            self.assertNotIn('--checkpoint',job['command'])
            self.assertNotIn('--input-previews',job['command'])

class EvidenceTests(unittest.TestCase):
    def test_suite_summary_links_real_input_gallery_and_latency_table(self):
        import json
        from pathlib import Path
        import tempfile
        from suite import summary
        from input_evidence import InputEvidence
        with tempfile.TemporaryDirectory() as tmp:
            root=Path(tmp)
            for phase in ('smoke','performance'):
                folder=root/phase;folder.mkdir()
                train=folder/'candidate'/'train';train.mkdir(parents=True)
                (folder/'plan.json').write_text(json.dumps({'jobs':[{'mode':'train','name':'candidate','output':str(train)}]}))
                (train/'trial_report.json').write_text(json.dumps(dict(status='COMPLETE',optimized_per_second=12.5,
                    reaction_latency={'metrics':{'policy_request_to_command_ms':{'median_ms':33.,'p95_ms':42.}}})))
                if phase=='smoke':
                    recorder=InputEvidence(4,1.06)
                    recorder.samples=[dict(image=np.zeros((12,2,2),np.uint8),aux=np.zeros(565,np.float32),
                        rollout=1,index=0,action=3,episode_start=True)]
                    recorder.write(train/'policy_inputs')
            summary(root)
            gallery=(root/'policy_input_gallery.html').read_text()
            self.assertIn('smoke/candidate/train/policy_inputs/01_rollout1_index0_sheet.png',gallery)
            table=(root/'performance_summary.md').read_text()
            self.assertIn('12.50 | 33.00 | 42.00',table)

    def test_actor_samples_export_exact_pixels_aux_and_bounded_copies(self):
        import json
        from pathlib import Path
        import tempfile
        from types import SimpleNamespace
        from PIL import Image
        from input_evidence import InputEvidence
        n=5
        frames=np.arange(8*15*2*2,dtype=np.uint8).reshape(8,1,15,2,2)
        aux=np.zeros((8,1,557+2*n),np.float32)
        aux[:,:,-n:]=1;aux[:,:,-2*n:-n]=np.linspace(1,0,n)
        buffer=SimpleNamespace(buffer_size=8,full=False,pos=7,
            actor_masks=np.array([0,1,1,1,1,1,1,1]).reshape(8,1),
            observations={'image':frames,'aux':aux},actions=np.ones((8,1,1)),
            episode_starts=np.zeros((8,1)))
        recorder=InputEvidence(n,1.06)
        for _ in range(5):recorder.collect(buffer)
        self.assertEqual(len(recorder.samples),6)
        expected=frames[1,0].copy();frames[:]=0
        with tempfile.TemporaryDirectory() as tmp:
            folder=Path(tmp)/'inputs';recorder.write(folder)
            index=json.loads((folder/'index.json').read_text())
            for sample in index['samples']:
                self.assertIn(sample['index'],range(1,7))
                self.assertEqual(sample['action'],1)
            name=index['samples'][0]['name']
            with np.load(folder/f'{name}.npz') as data:
                np.testing.assert_array_equal(data['image'],expected)
                np.testing.assert_array_equal(data['aux'],aux[1,0])
            png=np.array(Image.open(folder/f'{name}_slot01.png'))
            np.testing.assert_array_equal(png,expected[:3].transpose(1,2,0))
            self.assertTrue((folder/f'{name}_sheet.png').exists())

if __name__=='__main__':unittest.main()
