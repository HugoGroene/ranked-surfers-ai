import asyncio
from pathlib import Path
import tempfile
from types import SimpleNamespace
import unittest
from unittest.mock import patch

import torch
from tools.ppo_pipeline_profile import PipelineProfiler
from tools.analyze_ppo_pipeline_profile import overlap_ms,decisions,ADDITIVE
from tests import test_ppo_settled_learning as learning


class ProfileTests(unittest.TestCase):
    def test_wrapping_preserves_values_exceptions_and_capture_count(self):
        calls=[]
        async def capture():calls.append('capture');return 'same image'
        def pack(*_args):raise ValueError('same error')
        raw=SimpleNamespace(_capture=capture,_pack=pack)
        profiler=PipelineProfiler(raw,Path('.'))
        self.assertEqual(asyncio.run(raw._capture()),'same image')
        with self.assertRaisesRegex(ValueError,'same error'):raw._pack(None)
        self.assertEqual(calls,['capture'])
        self.assertEqual({r['kind'] for r in profiler.events},{'_capture','_pack'})

    def test_buffer_is_bounded_and_records_are_written_only_on_save(self):
        with tempfile.TemporaryDirectory() as temp:
            p=PipelineProfiler(SimpleNamespace(),Path(temp),max_events=1)
            p.record(dict(kind='test',start=1,end=2));p.record(dict(kind='test',start=2,end=3))
            self.assertFalse(list(Path(temp).iterdir()))
            p.save()
            self.assertEqual(p.dropped_events,1)
            self.assertTrue((Path(temp)/'pipeline_profile.jsonl').is_file())

    def test_real_collector_actions_rewards_and_final_weights_are_unchanged(self):
        original_env=learning.ProtocolEnv
        original_ppo=learning.PPO
        models=[];profiles=[]
        def construct(*args,**kwargs):
            model=original_ppo(*args,**kwargs);models.append(model);return model
        class ProfiledEnv(original_env):
            def __init__(self,mode):
                super().__init__(mode)
                profiles.append(PipelineProfiler(self,Path('.')))
        with patch.object(learning,'PPO',side_effect=construct):
            baseline,_=learning.SettledTests().run_case('normal')
            with patch.object(learning,'ProtocolEnv',ProfiledEnv):
                profiled,_=learning.SettledTests().run_case('normal')
        keys=('sampled_action','actor_mask','kind','environment_reward','buffer_reward','done','active_seconds')
        extract=lambda rows:[{k:r[k] for k in keys} for r in rows if r['event']=='collected']
        self.assertEqual(extract(baseline),extract(profiled))
        for key,value in models[0].policy.state_dict().items():
            self.assertTrue(torch.equal(value,models[1].policy.state_dict()[key]),key)
        self.assertEqual(sum(r['kind']=='policy' for r in profiles[0].events),8)


class AnalysisTests(unittest.TestCase):
    def test_overlapping_nested_events_are_not_double_counted(self):
        rows=[dict(kind='capture',start=.08,end=.11),dict(kind='capture',start=.09,end=.10)]
        self.assertAlmostEqual(overlap_ms(rows,'capture',.085,.105),20.)

    def test_additive_partition_and_frame_lineage(self):
        source=dict(serial=7,electron_serial=17,request_started_at=0.,image_received_at=.03,captured_at=.04)
        ledger=[dict(event='collected',transition=1,dispatch_at=.12,sampled_action=3,source_capture=source)]
        events=[dict(kind='policy',transition=1,source=source,passive=False,
            start=.05,tensor_ready=.06,forward_returned=.07,end=.08,gpu_stream_ms=3.),
            dict(kind='_capture',start=.085,end=.105),
            dict(kind='_step_browser',start=.081,end=.2,guard_source=dict(electron_serial=18,request_started_at=.085))]
        row=decisions(ledger,events)[0]
        self.assertAlmostEqual(sum(row[k] for k in ADDITIVE),120.)
        self.assertAlmostEqual(row['guard_capture_ms'],20.)
        self.assertTrue(row['guard_source_changed'])
        self.assertEqual(row['guard_source_serial'],18)
        events[0]['source']=dict(source,serial=99)
        with self.assertRaisesRegex(ValueError,'lineage'):decisions(ledger,events)


class ReaderBufferTests(unittest.IsolatedAsyncioTestCase):
    async def test_frame_sized_buffer_avoids_flow_control_without_changing_bytes(self):
        payload=bytes(range(256))*8192
        async def receive(limit):
            reader=asyncio.StreamReader(limit=limit)
            ready=asyncio.Event();ready.set()
            pauses=[]
            def pause():pauses.append(1);ready.clear()
            transport=SimpleNamespace(pause_reading=pause,resume_reading=ready.set)
            reader.set_transport(transport)
            async def produce():
                for offset in range(0,len(payload),262144):
                    await ready.wait()
                    reader.feed_data(payload[offset:offset+262144])
                    await asyncio.sleep(0)
                reader.feed_eof()
            task=asyncio.create_task(produce())
            result=await asyncio.wait_for(reader.readexactly(len(payload)),1.)
            await task
            return result,len(pauses)
        small,small_pauses=await receive(65536)
        large,large_pauses=await receive(len(payload)+4096)
        self.assertEqual(small,payload)
        self.assertEqual(large,payload)
        self.assertGreater(small_pauses,0)
        self.assertEqual(large_pauses,0)


if __name__=='__main__':unittest.main()
