import json
import tempfile
import unittest
from pathlib import Path
from types import SimpleNamespace

from tools.ppo_pipeline_profile import PipelineProfiler


class SpikeTests(unittest.TestCase):
    def test_late_spike_context_survives_full_initial_profile(self):
        with tempfile.TemporaryDirectory() as folder:
            profiler=PipelineProfiler(SimpleNamespace(),folder,max_events=1)
            for i in range(300):
                profiler.record(dict(kind='step_callback',start=i,end=i+.001))
            policy=dict(start=300.4,end=300.405,source=dict(request_started_at=300.,
                image_received_at=300.01),passive=False)
            row=profiler.command_diagnostic(policy,dict(policy_request_to_command_ms=420),300.42)
            self.assertAlmostEqual(row['image_received_at_to_policy_ms'],390)
            self.assertAlmostEqual(row['policy_host_ms'],5)
            self.assertAlmostEqual(row['policy_ready_to_command_ms'],15)
            self.assertEqual(len(row['context']),256)
            self.assertEqual(row['context'][-1]['start'],299)
            self.assertEqual(row['phase'],'ordinary')
            self.assertEqual(len(profiler.events),1)
            profiler.save()
            metadata=json.loads((Path(folder)/'pipeline_profile_metadata.json').read_text())
            self.assertEqual(metadata['latency_spikes'],1)
            self.assertEqual(metadata['dropped_events'],299)

    def test_context_rate_limit_does_not_drop_spike_records(self):
        profiler=PipelineProfiler(SimpleNamespace(),'.')
        policy=dict(start=1.,end=1.005,source={},passive=False)
        metric=dict(policy_request_to_command_ms=150)
        first=profiler.command_diagnostic(policy,metric,1.1,True)
        second=profiler.command_diagnostic(policy,metric,2.)
        third=profiler.command_diagnostic(policy,metric,12.)
        self.assertEqual(first['phase'],'episode_start')
        self.assertTrue(first['context_included'])
        self.assertFalse(second['context_included'])
        self.assertTrue(third['context_included'])
        self.assertEqual(profiler.spike_count,3)

    def test_no_fabricated_command_spikes(self):
        profiler=PipelineProfiler(SimpleNamespace(),'.')
        policy=dict(start=1.,end=1.005,source={},passive=False)
        for delay in (None,99,100,float('nan')):
            self.assertIsNone(profiler.command_diagnostic(policy,
                dict(policy_request_to_command_ms=delay),1.1))
        self.assertIsNone(profiler.command_diagnostic(policy,
            dict(policy_request_to_command_ms=500),None))
        policy['passive']=True
        self.assertIsNone(profiler.command_diagnostic(policy,
            dict(policy_request_to_command_ms=500),1.1))
        self.assertEqual(profiler.spike_count,0)


if __name__=='__main__':unittest.main()
