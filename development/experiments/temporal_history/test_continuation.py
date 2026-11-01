"""Regression checks for bounded freshness recovery and immutable smoke reuse."""
import asyncio
import json
from pathlib import Path
import tempfile
import time
import unittest
from unittest.mock import patch
import zipfile

import numpy as np
from batch import continue_smoke, execute, jobs_for
from live_adapter import CapturePump


class DeliveryTests(unittest.IsolatedAsyncioTestCase):
    async def test_source_retry_exhaustion_can_recover_without_publishing_stale_pixels(self):
        await self.source_case(3)

    async def test_source_recovery_is_bounded_to_nine_future_paints(self):
        await self.source_case(100)

    async def source_case(self, failures):
        from tools.ppo_electron_capture import ElectronCapture,StaleElectronFrame
        class Source(ElectronCapture):
            def __init__(self):
                self.calls=0;self.stale_frame_retries=0;self.last_timing={}
            async def _capture_once(self):
                self.calls+=1
                await asyncio.sleep(0)
                if self.calls<=failures:raise StaleElectronFrame('simulated 74ms consumed reply')
                self.last_timing={'electron_serial':self.calls}
                return np.full((2,2,4),self.calls,np.uint8)
        source=Source();published=[]
        pump=CapturePump(source,lambda *a:published.append(a),60)
        try:
            if failures>9:
                with self.assertRaises(StaleElectronFrame):await pump.capture()
                self.assertEqual(source.calls,9)
                self.assertEqual(published,[])
                self.assertEqual(pump.delivery_serial,-1)
            else:
                image=await pump.capture()
                self.assertEqual(source.calls,4)
                self.assertEqual(image[0,0,0],4)
                self.assertEqual(len(published),1)
                self.assertEqual(pump.source_recoveries,1)
                self.assertEqual(pump.source_stale_exhaustions,1)
        finally:await pump.close()

    async def test_protocol_error_never_retried_as_transient_staleness(self):
        from rs_v612_gameplay_guard import GuardUncertainError
        from unittest.mock import AsyncMock
        source=type('Source',(),{})()
        source.capture=AsyncMock(side_effect=GuardUncertainError('geometry mismatch'))
        pump=CapturePump(source,lambda *a:None,60)
        try:
            with self.assertRaisesRegex(GuardUncertainError,'geometry mismatch'):await pump.capture()
            self.assertEqual(source.capture.await_count,1)
            self.assertEqual(pump.source_stale_exhaustions,0)
        finally:await pump.close()

    async def exercise(self, stale_count):
        class Source:
            serial=0
            last_timing={}
            async def capture(self):
                stamp=time.perf_counter()
                await asyncio.sleep(0)
                self.serial+=1
                self.last_timing=dict(request_started_at=stamp,electron_serial=self.serial)
                return np.full((2,2,4),self.serial,np.uint8)
        source=Source()
        def callback(*args):
            if source.serial<=stale_count:time.sleep(.06)
        pump=CapturePump(source,callback,60)
        try:
            if stale_count>=3:
                with self.assertRaisesRegex(RuntimeError,'after 3 fresh requests'):
                    await pump.capture()
                self.assertEqual(pump.delivery_serial,-1)
                self.assertEqual(pump.delivery_stale_discards,3)
                self.assertEqual(source.serial,3)
            else:
                frame=await pump.capture()
                self.assertEqual(frame[0,0,0],2)
                self.assertEqual(pump.delivery_stale_discards,1)
                self.assertEqual(pump.last_timing['temporal_delivery_retries'],1)
                self.assertLessEqual(pump.last_timing['temporal_delivery_age_ms'],50)
                self.assertGreater(pump.last_timing['request_started_at'],
                                   pump.last_timing['temporal_consumer_requested_at'])
        finally:await pump.close()

    async def test_transient_stale_frame_discarded_before_fresh_delivery(self):
        await self.exercise(1)

    async def test_persistent_staleness_stops_without_delivery(self):
        await self.exercise(3)


class ContinuationTests(unittest.TestCase):
    def test_before_job_can_stop_without_launching_child(self):
        with tempfile.TemporaryDirectory() as temp:
            output=Path(temp);(output/'sources.json').write_text('{}')
            jobs=jobs_for(['wide8_60'],[123],30,100,60,output,42)
            with patch('batch.source_hashes',return_value={}),patch('batch.subprocess.Popen') as launch,patch('batch.summarize',return_value={}):
                with self.assertRaisesRegex(RuntimeError,'budget'):
                    execute({'jobs':jobs},output,before_job=lambda job:(_ for _ in ()).throw(RuntimeError('budget')))
                launch.assert_not_called()
                self.assertTrue((output/'comparison.json').exists())

    def test_keep_going_still_aborts_when_cleanup_is_not_confirmed(self):
        self.abort_case([{'stage':'environment','error':'close failed'}],'RuntimeError()')

    def test_keep_going_respects_user_cancellation(self):
        self.abort_case([],'KeyboardInterrupt()')

    def abort_case(self, errors, error):
        from types import SimpleNamespace
        with tempfile.TemporaryDirectory() as temp:
            output=Path(temp);(output/'sources.json').write_text('{}')
            jobs=jobs_for(['wide8_60','wide12_60'],[123],30,100,60,output,42)
            def wait():
                folder=Path(jobs[0]['output']);folder.mkdir(parents=True)
                (folder/'trial_report.json').write_text(json.dumps(dict(status='STOPPED',
                    finalization_errors=errors,error=error)))
                return 1
            with patch('batch.source_hashes',return_value={}),patch('batch.subprocess.Popen',return_value=SimpleNamespace(stdout=[],wait=wait)) as launch,patch('batch.summarize',return_value={}):
                with self.assertRaisesRegex(RuntimeError,'batch stopped'):execute({'jobs':jobs},output,keep_going=True)
                self.assertEqual(launch.call_count,1)

    def test_full_continuation_requires_exact_sources(self):
        with tempfile.TemporaryDirectory() as temp:
            root=Path(temp);path,jobs=self.fixture(root)
            parent=json.loads(path.read_text());parent['train_minutes']=30
            path.write_text(json.dumps(parent));(path.parent/'sources.json').write_text('{}')
            with patch('batch.source_hashes',return_value={}):
                plan=continue_smoke(path,root/'new',same_sources=True)
                self.assertTrue(plan['jobs'][0]['reused'])
                self.assertFalse(plan['source_mixed_smoke_validation'])
            with patch('batch.source_hashes',return_value={'changed':'hash'}):
                with self.assertRaisesRegex(ValueError,'unchanged sources'):
                    continue_smoke(path,root/'other',same_sources=True)

    def test_keep_going_skips_failed_model_eval_but_runs_other_arm(self):
        from types import SimpleNamespace
        with tempfile.TemporaryDirectory() as temp:
            output=Path(temp);(output/'sources.json').write_text('{}')
            jobs=jobs_for(['wide8_60','wide12_60'],[123],30,100,60,output,42)
            launched=[]
            def child(command,**kwargs):
                job=next(j for j in jobs if j['command']==command);launched.append(job)
                def wait():
                    folder=Path(job['output']);folder.mkdir(parents=True)
                    fail=job is jobs[0]
                    report=dict(status='STOPPED' if fail else 'COMPLETE',finalization_errors=[],
                                audit={'status':'PASS_SETTLED_LEARNING'},completed_requested_episodes=True)
                    name='trial_report.json' if job['mode']=='train' else 'v61_eval_report.json'
                    (folder/name).write_text(json.dumps(report));(folder/'latest.zip').write_bytes(b'fixture')
                    return int(fail)
                return SimpleNamespace(stdout=[],wait=wait)
            with patch('batch.source_hashes',return_value={}),patch('batch.subprocess.Popen',side_effect=child),patch('batch.summarize',return_value={}):
                with self.assertRaisesRegex(RuntimeError,'1 stages failed'):execute({'jobs':jobs},output,keep_going=True)
            self.assertEqual([(j['name'],j['mode']) for j in launched],[(j['name'],j['mode']) for j in [jobs[0],jobs[1],jobs[3]]])
            self.assertEqual(len(json.loads((output/'completed_stages.json').read_text())),2)

    def fixture(self, root):
        old=root/'old';old.mkdir()
        jobs=jobs_for(['wide8_60','wide12_60'],[123],1,3,3,old,42)
        completed=jobs[0]
        folder=Path(completed['output']);folder.mkdir(parents=True)
        (folder/'trial_report.json').write_text(json.dumps(dict(status='COMPLETE',
            audit={'status':'PASS_SETTLED_LEARNING'},optimized_transitions=640)))
        with zipfile.ZipFile(folder/'latest.zip','w') as z:z.writestr('data','preserved')
        parent=dict(train_minutes=1,eval_episodes=3,jobs=jobs)
        path=old/'plan.json';path.write_text(json.dumps(parent))
        return path,jobs

    def test_reuses_only_passed_stages_and_routes_evaluations_to_correct_models(self):
        with tempfile.TemporaryDirectory() as temp:
            root=Path(temp);path,old=self.fixture(root)
            before=(Path(old[0]['output'])/'latest.zip').read_bytes()
            plan=continue_smoke(path,root/'new')
            self.assertTrue(plan['jobs'][0]['reused'])
            self.assertNotIn('reused',plan['jobs'][1])
            for training,evaluation in zip(plan['jobs'][:2],plan['jobs'][2:]):
                cmd=evaluation['command']
                self.assertEqual(cmd[cmd.index('--checkpoint')+1],str(Path(training['output'])/'latest.zip'))
            self.assertEqual((Path(old[0]['output'])/'latest.zip').read_bytes(),before)
            self.assertTrue(plan['source_mixed_smoke_validation'])
            self.assertEqual(len(plan['retained_artifacts']),2)

    def test_full_learning_comparison_cannot_mix_source_versions(self):
        with tempfile.TemporaryDirectory() as temp:
            root=Path(temp);path,_=self.fixture(root)
            parent=json.loads(path.read_text());parent['train_minutes']=30
            path.write_text(json.dumps(parent))
            with self.assertRaisesRegex(ValueError,'smoke validation'):continue_smoke(path,root/'new')

    def test_retained_corruption_aborts_before_any_child_launch(self):
        with tempfile.TemporaryDirectory() as temp:
            root=Path(temp);path,old=self.fixture(root)
            output=root/'new';output.mkdir()
            plan=continue_smoke(path,output)
            (output/'sources.json').write_text('{}')
            (Path(old[0]['output'])/'latest.zip').write_bytes(b'changed')
            with patch('batch.source_hashes',return_value={}),patch('batch.subprocess.Popen') as launch:
                with self.assertRaisesRegex(RuntimeError,'artifact changed'):execute(plan,output)
                launch.assert_not_called()

    def test_corrupt_checkpoint_invalidates_its_completed_evaluation(self):
        with tempfile.TemporaryDirectory() as temp:
            root=Path(temp);path,jobs=self.fixture(root)
            evaluation=jobs[2];folder=Path(evaluation['output']);folder.mkdir(parents=True)
            (folder/'v61_eval_report.json').write_text(json.dumps(dict(status='COMPLETE',completed_requested_episodes=True)))
            (Path(jobs[0]['output'])/'latest.zip').write_bytes(b'bad zip')
            plan=continue_smoke(path,root/'new')
            self.assertFalse(any(j.get('reused') for j in plan['jobs']))


if __name__=='__main__':unittest.main()
