import contextlib
import io
import json
from pathlib import Path
import tempfile
from types import SimpleNamespace
import unittest
from unittest.mock import patch
import zipfile

import numpy as np

from tools.ppo_observation_contract import requested_contract, validate_checkpoint, attach_contract
from tools.ppo_run_integrity import atomic_json, finalize_run, worker_seed
from tools.ppo_eval_summary import episode_metrics, completion
from tools.ppo_detector_off_features import detector_off_aux


class ContractTests(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.addCleanup(self.temp.cleanup)
        self.path = Path(self.temp.name)/'source.zip'
        self.contract = requested_contract(SimpleNamespace())

    def source(self, data):
        with zipfile.ZipFile(self.path, 'w') as archive:
            archive.writestr('data', json.dumps(data))

    def test_current_saved_contract_round_trip(self):
        model = SimpleNamespace()
        attach_contract(model, self.contract)
        self.source(vars(model))
        result = validate_checkpoint(self.path, self.contract)
        self.assertFalse(result['migration'])
        self.assertFalse(result['legacy_metadata'])

    def test_known_old_off_canvas_source_is_accepted_with_limitation(self):
        self.source(dict(framing_profile='canvas_v1', detector_mode='off'))
        self.assertTrue(validate_checkpoint(self.path, self.contract)['legacy_metadata'])

    def test_historical_source_cannot_silently_become_off_canvas(self):
        self.source({})
        before = self.path.read_bytes()
        with self.assertRaisesRegex(ValueError, 'input contract mismatch'):
            validate_checkpoint(self.path, self.contract)
        result = validate_checkpoint(self.path, self.contract, allow_migration=True)
        self.assertEqual(set(result['mismatches']), {'framing_profile', 'detector_mode'})
        self.assertEqual(before, self.path.read_bytes())

    def test_same_shape_different_channel_order_or_detector_rejected(self):
        for key, value in [('image_order', 'BGR'), ('detector_mode', 'on'), ('resize_method', 'area')]:
            with self.subTest(key=key):
                self.source({'observation_contract':dict(self.contract, **{key:value})})
                with self.assertRaises(ValueError):validate_checkpoint(self.path, self.contract)

    def test_unknown_contract_version_and_missing_fields_rejected_even_for_migration(self):
        for contract in [dict(self.contract, version=99), {'version':1}]:
            self.source({'observation_contract':contract})
            with self.assertRaises(ValueError):
                validate_checkpoint(self.path, self.contract, allow_migration=True)


class FinalizationTests(unittest.TestCase):
    def test_cleanup_failure_cannot_skip_checkpoint_or_final_status(self):
        with tempfile.TemporaryDirectory() as directory:
            folder=Path(directory); events=[]; report={'status':'COMPLETE'}
            def checkpoint():
                events.append('save'); (folder/'latest').write_text('committed')
            def bad():
                events.append('bad'); raise OSError('cleanup failure')
            errors=finalize_run(report, checkpoint,
                [('broken',bad),('later',lambda:events.append('later'))],
                lambda:atomic_json(folder/'report.json',report))
            self.assertEqual(events,['save','bad','later'])
            self.assertEqual((folder/'latest').read_text(),'committed')
            self.assertEqual(json.loads((folder/'report.json').read_text())['status'],'STOPPED')
            self.assertEqual(errors[0]['stage'],'broken')

    def test_save_failure_and_second_interrupt_still_record_and_cleanup(self):
        report={'status':'STOPPED','error':'original worker failure'}; persisted=[]; events=[]
        def bad_save():raise OSError('disk full')
        def interrupt():raise KeyboardInterrupt()
        errors=finalize_run(report,bad_save,[('interrupt',interrupt),('later',lambda:events.append(1))],
                            lambda:persisted.append(json.loads(json.dumps(report))))
        self.assertEqual(report['error'],'original worker failure')
        self.assertEqual(len(errors),2)
        self.assertEqual(events,[1])
        self.assertEqual(len(persisted),2)

    def test_failed_atomic_write_preserves_previous_report(self):
        with tempfile.TemporaryDirectory() as directory:
            path=Path(directory)/'report.json';atomic_json(path,{'value':1})
            with self.assertRaises(ValueError):atomic_json(path,{'value':float('nan')})
            self.assertEqual(json.loads(path.read_text()),{'value':1})
            self.assertEqual(list(Path(directory).iterdir()),[path])


class SamplingAndFeatureTests(unittest.TestCase):
    def test_worker_streams_are_reproducible_and_distinct(self):
        seeds=[worker_seed(123,i) for i in range(6)]
        streams=[np.random.default_rng(s).integers(0,2**31,size=20) for s in seeds]
        self.assertEqual(len(set(seeds)),6)
        for i,seed in enumerate(seeds):
            np.testing.assert_array_equal(streams[i],np.random.default_rng(seed).integers(0,2**31,size=20))
        self.assertFalse(np.array_equal(streams[0],streams[1]))

    def test_detector_removal_matches_live_encoder_without_touching_context_or_input(self):
        from rs_v6_ppo_features import encode_hazards
        from rs_v61_features import encode_aux, ActionContext, AUX_DIM, V6_AUX_DIM, SCREEN_COLUMN_DIM
        from rs_v6_ppo_features import HAZARD_DIM
        # Start with exactly the live unavailable feature encoding.
        hazards, age, fresh=encode_hazards(None,episode_tag=1,now=2.,max_stale_ms=300.)
        context=ActionContext();context.note_action(1,1.5)
        expected=encode_aux(hazards,[0,1],2.,[.04,.05],age,snapshot=None,
            episode_tag=1,now=2.,action_context=context)
        recorded=expected.copy()
        recorded[:HAZARD_DIM]=.6
        recorded[V6_AUX_DIM-1]=.2
        recorded[V6_AUX_DIM:V6_AUX_DIM+SCREEN_COLUMN_DIM]=.7
        before=recorded.copy()
        np.testing.assert_array_equal(detector_off_aux(recorded),expected)
        np.testing.assert_array_equal(recorded,before)
        self.assertEqual(detector_off_aux(np.stack([recorded,recorded])).shape,(2,AUX_DIM))

    def test_bad_aux_is_rejected(self):
        with self.assertRaises(ValueError):detector_off_aux(np.zeros(4))


class EvaluationTests(unittest.TestCase):
    def test_clocks_are_separate_and_missing_verified_clock_fails(self):
        self.assertEqual(episode_metrics({'duration':4.,'reward_duration_s':3.}), (3.,4.,3.))
        self.assertEqual(episode_metrics({'duration':4.},legacy=True),(4.,4.,None))
        with self.assertRaises(KeyError):episode_metrics({'duration':4.})

    def test_partial_batch_is_not_success(self):
        self.assertFalse(completion(100,20,'ui_safety_stop')['completed_requested_episodes'])
        self.assertTrue(completion(100,100,'episodes')['completed_requested_episodes'])

    def test_evaluator_reports_verified_metric_and_safe_stop_status(self):
        import train_rs_v612_ppo as trainer
        from rs_v612_gameplay_guard import GuardUncertainError
        class FakeEnv:
            def __init__(self, fail):self.fail=fail;self.n=0
            def reset(self):return {},{}
            def step(self,action):
                self.n+=1
                if self.fail and self.n==2:raise GuardUncertainError('test stop')
                return {},1.,True,False,dict(duration=4.,reward_duration_s=3.,
                    episode={'r':1.},terminal_evidence_source='inherited_template')
            def close(self):pass
        for fail in (False,True):
            with self.subTest(fail=fail),tempfile.TemporaryDirectory() as directory:
                with patch('pathlib.Path.is_file',return_value=True):
                    args=trainer.parse_args(['--mode','eval','--eval-policy','none','--eval-episodes','2','--run-name','test'])
                env=FakeEnv(fail)
                with patch('train_rs_v2_verify.RUN_ROOT',Path(directory)), \
                     patch.object(trainer,'create_live_env',return_value=(env,env)), \
                     patch.object(trainer,'build_report',return_value={}),contextlib.redirect_stdout(io.StringIO()):
                    success=trainer.evaluate(args)
                report=json.loads((Path(directory)/'test/v61_eval_report.json').read_text())
                self.assertEqual(success,not fail)
                self.assertEqual(report['mean_survival_seconds'],3.)
                self.assertEqual(report['wall_durations_seconds'][0],4.)
                self.assertEqual(report['terminal_evidence_counts']['inherited_template'],1 if fail else 2)
                self.assertEqual(report['status'],'INCOMPLETE' if fail else 'COMPLETE')

    def test_defaults_are_detector_off_canvas_without_disabling_safety_model(self):
        import train_rs_v612_ppo as trainer
        with patch('pathlib.Path.is_file',return_value=True):
            args=trainer.parse_args(['--mode','eval','--eval-policy','none'])
        self.assertEqual(args.detector_mode,'off')
        self.assertEqual(args.framing_profile,'canvas_v1')
        self.assertIsNotNone(args.state_model)

    def test_small_offline_check_remains_available(self):
        import train_rs_v612_ppo as trainer
        args=trainer.parse_args(['--mode','check','--resolution','160x96'])
        self.assertEqual(args.resolution,(160,96))

    def test_mismatched_evaluation_stops_before_browser_creation(self):
        import train_rs_v612_ppo as trainer
        with tempfile.TemporaryDirectory() as directory:
            source=Path(directory)/'old.zip'
            with zipfile.ZipFile(source,'w') as archive:archive.writestr('data','{}')
            with patch('pathlib.Path.is_file',return_value=True):
                args=trainer.parse_args(['--mode','eval','--resume',str(source)])
            with patch.object(trainer,'create_live_env') as create:
                with self.assertRaisesRegex(ValueError,'input contract mismatch'):trainer.evaluate(args)
                create.assert_not_called()


class CoordinatorTests(unittest.TestCase):
    def run_coordinator(self, failure=None):
        """Actual CPU learner/coordinator, fake worker transport; no game/process."""
        import torch
        from stable_baselines3 import PPO
        from tests.test_ppo_parallel_learning import SmallEnv, local_packet
        from tools.ppo_parallel_learning import policy_arrays, policy_hash, SharedRolloutPPO
        import tools.train_ppo_parallel as runner
        from unittest.mock import MagicMock
        torch.set_num_threads(1)
        model=PPO('MultiInputPolicy',SmallEnv(),device='cpu',n_steps=16,batch_size=8,n_epochs=1,seed=123)
        model.framing_profile='canvas_v1';model.detector_mode='off'
        with tempfile.TemporaryDirectory() as directory:
            folder=Path(directory); source=folder/'source.zip';model.save(source)
            state=folder/'state.pt';state.write_bytes(b'fixture')
            output=folder/'run';parent=MagicMock();child=MagicMock()
            process=MagicMock();process.is_alive.return_value=False
            if failure=='cleanup':process.join.side_effect=OSError('join failed')
            ctx=MagicMock();ctx.Pipe.return_value=(parent,child);ctx.Process.return_value=process
            def receive(connections, expected, timeout, abort):
                if expected=='rollout':
                    packet=local_packet(0)
                    packet.update(version=0,policy_hash=policy_hash(policy_arrays(model.policy)))
                    return [packet]
                return [{'op':expected,'worker':0}]
            argv=['--headless','--displays','1','--checkpoint',str(source),'--state-model',str(state),
                  '--steps','16','--max-minutes','1','--report',str(output),'--end-plot','off']
            with contextlib.ExitStack() as stack:
                stack.enter_context(patch('stable_baselines3.PPO.load',return_value=model))
                stack.enter_context(patch.object(runner.mp,'get_context',return_value=ctx))
                stack.enter_context(patch.object(runner,'receive_all',side_effect=receive))
                stack.enter_context(patch('tools.ppo_settled_learning.audit_settled_ledger',return_value={'status':'PASS_SETTLED_LEARNING'}))
                stack.enter_context(patch.object(runner,'source_manifest',return_value={'test':True}))
                stack.enter_context(contextlib.redirect_stdout(io.StringIO()))
                if failure=='optimizer':
                    stack.enter_context(patch.object(SharedRolloutPPO,'train',side_effect=KeyboardInterrupt()))
                try:runner.main(argv)
                except BaseException as exc:
                    if failure is None:raise
                    self.assertIsInstance(exc,KeyboardInterrupt if failure=='optimizer' else RuntimeError)
                else:
                    self.assertIsNone(failure)
            report=json.loads((output/'parallel_training.json').read_text())
            self.assertTrue(process.join.called)
            if failure=='optimizer':
                self.assertFalse((output/'latest.zip').exists())
                self.assertIn('optimizer interrupted',report['final_checkpoint_skipped'])
                self.assertEqual(report['status'],'STOPPED')
            else:
                with zipfile.ZipFile(output/'latest.zip') as archive:
                    data=json.loads(archive.read('data'))
                self.assertEqual(data['detector_mode'],'off')
                self.assertEqual(data['observation_contract']['framing_profile'],'canvas_v1')
                self.assertEqual(report['final_checkpoint_transitions'],16)
                self.assertEqual(report['status'],'STOPPED' if failure else 'COMPLETE')
            model.logger.close()

    def test_full_shared_update_and_checkpoint(self):self.run_coordinator()
    def test_cleanup_failure_preserves_updated_model(self):self.run_coordinator('cleanup')
    def test_interrupted_optimizer_is_not_saved_as_committed(self):self.run_coordinator('optimizer')


if __name__=='__main__':unittest.main()
