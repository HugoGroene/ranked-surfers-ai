import json
from pathlib import Path
import tempfile
from types import SimpleNamespace
import unittest
from unittest.mock import Mock,patch

from parallel_adapter import configuration
import benchmark_workers as batch
from run_workers import command
import results


class ScalingTests(unittest.TestCase):
    def test_selected_contract_and_all_worker_flags(self):
        config=configuration()
        self.assertEqual(config['size'],[384,384]);self.assertEqual(config['encoder'],'spatial')
        self.assertEqual(config['temporal']['frames'],5)
        self.assertEqual(config['temporal']['context_offsets_seconds'],[.5,1.])
        for n in range(1,6):
            args=command('model.zip','new_report',n,5)
            self.assertNotIn('--preview-port',args);self.assertNotIn('--benchmark-only',args)
            self.assertNotIn('--compact-observations',args);self.assertNotIn('--allow-input-migration',args)
            self.assertEqual(args[args.index('--rollout-transport')+1],'shared_memory')
            self.assertEqual(int(args[args.index('--steps')+1])%(128*n),0)
            self.assertEqual(args[args.index('--end-plot')+1],'off')

    def test_same_checkpoint_and_ascending_counts_after_smoke(self):
        jobs=batch.jobs(Path('out'),Path('source.zip'),5)
        self.assertEqual([j['workers'] for j in jobs],[1,1,2,3,4,5,1])
        self.assertEqual(jobs[0]['phase'],'smoke');self.assertEqual(jobs[-1]['phase'],'repeat')
        self.assertEqual(sum(j['minutes'] for j in jobs),31)
        self.assertTrue(all(j['command'][j['command'].index('--checkpoint')+1]=='source.zip' for j in jobs))

    def test_python_launcher_is_not_competing_worker_but_sibling_is(self):
        argv=['python.exe','benchmark_workers.py','--execute']
        parent=Mock(pid=1);parent.name.return_value='python.exe';parent.cmdline.return_value=argv;parent.parent.return_value=None
        current=Mock(pid=2);current.cmdline.return_value=argv;current.parent.return_value=parent
        processes=[SimpleNamespace(pid=n,info={'name':'python.exe','cmdline':argv}) for n in (1,2,3)]
        with patch('psutil.Process',return_value=current),patch('psutil.process_iter',return_value=processes):
            self.assertEqual(batch.busy_processes(),[3])

    def test_cleanup_requires_all_workers_and_successful_parent_finalization(self):
        with tempfile.TemporaryDirectory() as tmp:
            folder=Path(tmp);(folder/'parallel_training.json').write_text(json.dumps({'finalization_errors':[]}))
            (folder/'worker_0').mkdir();(folder/'worker_0/hybrid_close.json').write_text('{"status":"COMPLETE"}')
            self.assertTrue(results.clean(folder,1));self.assertFalse(results.clean(folder,2))
            (folder/'parallel_training.json').write_text('{"finalization_errors":["timeout"]}')
            self.assertFalse(results.clean(folder,1))

    def test_worst_worker_latency_is_not_average_of_percentiles(self):
        with tempfile.TemporaryDirectory() as tmp:
            folder=Path(tmp);(folder/'parallel_training.json').write_text('{"status":"STOPPED"}')
            for i,(median,tail) in enumerate([(30,40),(50,90)]):
                worker=folder/f'worker_{i}';worker.mkdir()
                (worker/'result.json').write_text(json.dumps({'reaction_latency':{'metrics':{
                    'policy_request_to_command_ms':{'median_ms':median,'p95_ms':tail}}}}))
            row=results.summarize_case(dict(name='test',output=str(folder),workers=2,phase='performance'))
            self.assertEqual(row['worker_median_range_ms'],[30,50])
            self.assertEqual(row['worst_worker_p95_ms'],90)
            self.assertEqual(row['optimizer_seconds']['count'],0)


if __name__=='__main__':unittest.main()
