import unittest
import json
from pathlib import Path
from tempfile import TemporaryDirectory
from tools.analyze_ppo_speed import compare,distribution,summarize


def row(name,latency,throughput,valid=True,source='same',frozen=True):
    return dict(report=name,valid=valid,source_sha256=source,benchmark_only=frozen,
        worst_worker_p95_ms=latency,steady=dict(median_ms=latency/2),throughput=throughput)


class SpeedAnalysisTests(unittest.TestCase):
    def test_latency_and_throughput_tradeoff_is_not_hidden_by_one_score(self):
        result=compare([row('fast_reactions',40,25),row('more_steps',70,40),row('dominated',80,20)])
        self.assertEqual(result['lowest_latency'],'fast_reactions')
        self.assertEqual(result['highest_throughput'],'more_steps')
        self.assertEqual(result['pareto_candidates'],['fast_reactions','more_steps'])

    def test_aborted_fast_run_is_excluded_and_mixed_sources_or_modes_refused(self):
        self.assertEqual(compare([row('invalid',1,1000,False),row('valid',50,20)])['lowest_latency'],'valid')
        for other in [row('other',40,30,source='different'),row('other',40,30,frozen=False)]:
            with self.assertRaises(ValueError):compare([row('first',50,20),other])

    def test_stalls_are_reported_and_invalid_values_rejected(self):
        result=distribution([10,20,101,151])
        self.assertEqual(result['above_100ms'],2)
        self.assertEqual(result['above_150ms'],1)
        self.assertEqual(result['max_ms'],151)
        with self.assertRaises(ValueError):distribution([float('nan')])

    def test_saved_training_requires_every_audit_and_keeps_startup_samples(self):
        with TemporaryDirectory() as temp:
            folder=Path(temp)
            config=dict(workers=1,benchmark_only=False,detector_mode='off',compact_observations=False,
                shared_inference=False,electron_host='per_worker',source_sha256='source')
            report=dict(status='COMPLETE',source_unchanged=True,optimized_transitions_per_second=20,
                audits=[dict(status='PASS_SETTLED_LEARNING',failures=[])])
            (folder/'config.json').write_text(json.dumps(config))
            worker=folder/'worker_0';worker.mkdir()
            (worker/'result.json').write_text(json.dumps(dict(learning_transitions=dict(discarded_partial_transitions=0))))
            records=[dict(command='LEFT',reaction_latency=dict(policy_request_to_command_ms=value,
                guard_capture_complete_to_command_ms=10)) for value in [120,30,40]]
            (worker/'learning_transitions.jsonl').write_text('\n'.join(json.dumps(r) for r in records))
            def check():
                (folder/'parallel_training.json').write_text(json.dumps(report))
                return summarize(folder,1)
            result=check()
            self.assertTrue(result['valid'])
            self.assertEqual(result['all']['above_100ms'],1)
            self.assertEqual(result['steady']['above_100ms'],0)
            self.assertEqual(result['steady']['count'],2)
            report['audits']=[]
            self.assertFalse(check()['valid'])
            report['audits']=[dict(status='FAILED',failures=[])]
            self.assertFalse(check()['valid'])
