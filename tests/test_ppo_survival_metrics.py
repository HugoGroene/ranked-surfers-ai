import json
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch
from tools.ppo_survival_metrics import survival_curve,SurvivalMetrics
from tools.ppo_terminal_stats import format_terminal_stats


class SurvivalTests(unittest.TestCase):
    def test_pools_episodes_in_time_order_not_mean_of_worker_means(self):
        events=[dict(at=150,duration=9),dict(at=110,duration=1),dict(at=120,duration=2)]
        result=survival_curve(events,100,3600)
        self.assertEqual(result['requested_seconds'],3600)
        self.assertAlmostEqual(result['points'][-1]['mean_seconds'],4)
        self.assertEqual(result['completed_episodes'],3)
        self.assertEqual(result['points'][0]['seconds'],10)

    def test_rolling_window_is_not_lifetime_mean(self):
        events=[dict(at=100+i,duration=1 if i<50 else 10,reason='death') for i in range(150)]
        result=survival_curve(events,100,3600,elapsed_seconds=150)
        self.assertEqual(result['points'][-1]['mean_seconds'],10)
        self.assertEqual(result['points'][-1]['samples'],100)
        self.assertEqual(result['recent_mean_seconds'],10)
        self.assertEqual(result['recent_samples'],100)
        self.assertEqual(result['overall_mean_seconds'],7)

    def test_no_zero_for_missing_episodes_and_overrun_stays_inside_requested_axis(self):
        self.assertEqual(survival_curve([],100,60)['points'],[])
        result=survival_curve([dict(at=150,duration=3),dict(at=170,duration=5,reason='time_limit')],100,60)
        self.assertEqual(result['points'][-1]['seconds'],60)
        self.assertEqual(result['points'][-1]['mean_seconds'],3)
        self.assertEqual(result['time_limited_episodes'],1)

    def test_new_samples_append_without_replacing_old_dots(self):
        first=[dict(at=105,duration=2),dict(at=115,duration=4)]
        before=survival_curve(first,100,1800,elapsed_seconds=20)['points']
        after=survival_curve(first+[dict(at=125,duration=9)],100,1800,elapsed_seconds=30)['points']
        self.assertEqual(after[:len(before)],before)
        self.assertEqual([p['seconds'] for p in after],[10,20,30])

    def test_incomplete_line_is_ignored_and_final_data_persists(self):
        with tempfile.TemporaryDirectory() as directory:
            folder=Path(directory)
            worker=folder/'worker_0'
            worker.mkdir()
            (worker/'survival_episodes.jsonl').write_text('{"at":110,"duration":4}\n{"at":')
            metrics=SurvivalMetrics(folder,1,100,3600)
            metrics.update()
            result=json.loads((folder/'survival.json').read_text())
            self.assertEqual(result['completed_episodes'],1)
            self.assertEqual(result['points'][0]['mean_seconds'],4)

    def test_terminal_snapshot_is_cached_and_final_mean_uses_latest_completions(self):
        with tempfile.TemporaryDirectory() as directory:
            folder=Path(directory)
            for index,events in enumerate([[dict(at=110,duration=2)],
                    [dict(at=120,duration=4),dict(at=125,duration=9,reason='time_limit')]]):
                worker=folder/f'worker_{index}';worker.mkdir()
                (worker/'survival_episodes.jsonl').write_text(''.join(json.dumps(e)+'\n' for e in events))
            metrics=SurvivalMetrics(folder,2,100,20)
            self.assertIsNone(metrics.snapshot())
            metrics.update()
            with patch.object(Path,'read_text',side_effect=AssertionError('Terminal must not read files')):
                summary=metrics.snapshot()
                self.assertEqual(summary['completed_episodes'],3)
                self.assertEqual(summary['recent_mean_seconds'],5)
                self.assertEqual(summary['overall_mean_seconds'],5)
                text=format_terminal_stats(summary,60,1800,summary['updated_at']+2)
                self.assertIn('overall_mean=5.00s recent100_mean=5.00s',text)
                self.assertIn('recent_n=3 max=9.00s time_limits=1 optimized/s=30.0',text)
                self.assertIn('elapsed=01:00 stats_age=2.0s',text)
                summary['completed_episodes']=999
                self.assertEqual(metrics.snapshot()['completed_episodes'],3)
            saved=json.loads((folder/'survival.json').read_text())
            self.assertEqual(saved['points'][-1]['mean_seconds'],3)

    def test_missing_episodes_and_frozen_collection_are_labeled(self):
        result=survival_curve([],100,60)
        self.assertIsNone(result['recent_mean_seconds'])
        self.assertEqual(result['recent_samples'],0)
        text=format_terminal_stats(result,0,0,100)
        self.assertIn('episodes=0 overall_mean=n/a recent100_mean=n/a',text)
        self.assertIn('optimized/s=0.0',text)
        self.assertIn('[FROZEN STATS]',format_terminal_stats(result,10,100,100,True))
        self.assertIn('processed/s=10.0',format_terminal_stats(result,10,100,100,True))


if __name__ == '__main__':
    unittest.main()
