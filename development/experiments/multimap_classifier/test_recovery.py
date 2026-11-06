import hashlib
import json
import os
from pathlib import Path
import sys
import tempfile
import unittest
import zipfile
from unittest.mock import patch
from recovery import supervise,verified_checkpoint,recoverable,run_child

GUARD="RuntimeError: Worker 3 GuardUncertainError: [PPO RESYNC] No four consecutive GAMEPLAY frames within 8 captures"


def archive(path,steps,contract=None):
    with zipfile.ZipFile(path,'w') as z:
        z.writestr('data',json.dumps(dict(num_timesteps=steps,observation_contract=contract or {'version':1})))
        z.writestr('policy.pth',b'fixture')
        z.writestr('policy.optimizer.pth',b'fixture')


class RecoveryTests(unittest.TestCase):
    def test_verified_final_checkpoint_and_reject_mismatch_corruption(self):
        with tempfile.TemporaryDirectory() as folder:
            root=Path(folder);source=root/'source.zip';archive(source,100)
            dest=root/'attempt';(dest/'train').mkdir(parents=True)
            candidate=dest/'train/latest.zip';archive(candidate,612)
            report=dict(source_sha256=hashlib.sha256(source.read_bytes()).hexdigest(),
                source_unchanged=True,finalization_errors=[],optimized_transitions=512,
                processed_transitions=512,final_checkpoint_transitions=512,error=GUARD)
            file=dest/'train/parallel_training.json';file.write_text(json.dumps(report))
            self.assertEqual(verified_checkpoint(dest,source)[0],candidate)
            archive(candidate,613)
            with self.assertRaisesRegex(ValueError,'step count'):verified_checkpoint(dest,source)
            archive(candidate,612,{'version':2})
            with self.assertRaisesRegex(ValueError,'contract'):verified_checkpoint(dest,source)
            candidate.write_bytes(b'invalid zip')
            with self.assertRaises(zipfile.BadZipFile):verified_checkpoint(dest,source)
            report['finalization_errors']=['save failed'];file.write_text(json.dumps(report))
            with self.assertRaisesRegex(ValueError,'integrity'):verified_checkpoint(dest,source)

    def scenario(self,errors,elapsed,limit=3,minutes=10):
        folder=tempfile.TemporaryDirectory();self.addCleanup(folder.cleanup)
        root=Path(folder.name);now=[1000.];calls=[];reports={}
        def launch(dest,source,remaining,deadline):
            index=len(calls);calls.append((dest,source,remaining,deadline))
            now[0]+=elapsed[index]
            error=errors[index]
            reports[dest]=(dest/'train/latest.zip',dict(status='STOPPED' if error else 'COMPLETE',
                error=error,optimized_transitions=512,active_wall_seconds=elapsed[index],finalization_errors=[]))
            return int(bool(error))
        result=supervise(root,root/'source.zip',minutes,limit,launch,clock=lambda:now[0],
                         verify=lambda dest,source:reports[dest])
        return result,json.loads((root/'recovery.json').read_text()),calls

    def test_restarts_from_saved_checkpoint_and_deducts_elapsed_time(self):
        (code,last,attempts),report,calls=self.scenario([GUARD,None],[120,480])
        self.assertEqual(code,0);self.assertEqual(report['status'],'COMPLETE')
        self.assertEqual(calls[1][1],calls[0][0]/'train/latest.zip')
        self.assertEqual(calls[1][2],8)
        self.assertEqual(calls[0][3],calls[1][3])
        self.assertEqual(len(attempts),2)

    def test_limits_repeated_early_failures_and_total_restarts(self):
        _,report,calls=self.scenario([GUARD,GUARD],[10,10])
        self.assertEqual(report['status'],'STOPPED_REPEATED_EARLY_FAILURE')
        self.assertEqual(len(calls),2)
        _,report,calls=self.scenario([GUARD,GUARD],[120,120],limit=1)
        self.assertEqual(report['status'],'STOPPED_RETRY_LIMIT')

    def test_unknown_error_or_exhausted_time_never_restarts(self):
        _,report,calls=self.scenario(['CUDA out of memory'],[120])
        self.assertEqual(report['status'],'STOPPED_NONRECOVERABLE')
        self.assertEqual(len(calls),1)
        _,report,calls=self.scenario([GUARD],[600])
        self.assertEqual(report['status'],'BUDGET_EXHAUSTED_AFTER_STOP')
        self.assertEqual(len(calls),1)
        self.assertFalse(recoverable(dict(error='GuardUncertainError: geometry changed')))
        self.assertFalse(recoverable(dict(error=GUARD,finalization_errors=['checkpoint failed'])))

    def test_cancel_never_restarts(self):
        with tempfile.TemporaryDirectory() as folder:
            root=Path(folder)
            def cancel(*args):raise KeyboardInterrupt()
            with self.assertRaises(KeyboardInterrupt):supervise(root,root/'source.zip',30,3,cancel)
            self.assertEqual(json.loads((root/'recovery.json').read_text())['status'],'CANCELLED')

    def test_child_cleanup_returns_exit_code_and_saves_report(self):
        with tempfile.TemporaryDirectory() as folder:
            root=Path(folder);dest=root/'attempt'
            code=run_child([sys.executable,'-B','-s','-c','raise SystemExit(7)'],os.environ.copy(),dest,cwd=root)
            self.assertEqual(code,7)
            self.assertEqual(json.loads((dest/'attempt_processes.json').read_text())['remaining_owned_pids'],[])

    def test_plots_pool_all_attempts_by_map(self):
        from stability import plots
        with tempfile.TemporaryDirectory() as folder:
            root=Path(folder)
            attempts=[root/'stability',root/'recovery_01']
            for j,attempt in enumerate(attempts):
                for worker in range(4):
                    wd=attempt/f'train/worker_{worker}';wd.mkdir(parents=True)
                    (wd/'survival_episodes.jsonl').write_text(json.dumps(dict(at=10*j+worker,duration=1+j))+'\n')
            plots(attempts[0],4,show=False,attempts=attempts)
            result=json.loads((attempts[0]/'per_map_summary.json').read_text())
            self.assertEqual(result['halloween']['episodes'],4)
            self.assertEqual(result['tokyo']['overall_mean'],1.5)


if __name__=='__main__':unittest.main()
