"""Evaluation tracing must not add inference, actions, or captures."""
import json
from pathlib import Path
import tempfile
import time
from types import SimpleNamespace
import unittest
from unittest.mock import patch
from development.experiments.multimap_classifier.eval_latency import install
from tools import ppo_eval_prediction
from development.experiments.latency_tail_20261107.pacing import DeadlineWait,replace_wait


class Parent:
    def step(self,action):return action+1


class WaitFixture(Parent):
    def step(self,action):
        remaining=.002
        time.sleep(remaining)
        return super().step(action)


class TraceTests(unittest.TestCase):
    def test_runtime_bypass_and_buffer_restoration(self):
        import os
        from development.experiments.latency_tail_20261107 import runtime
        from tools.ppo_settled_learning import SettledPPO
        with patch.dict(os.environ,{'RS_PIPELINE_EXPERIMENT':''},clear=True):
            self.assertFalse(runtime.settings()['enabled'])
        buffer=SimpleNamespace(n_envs=1,full=True,generator_ready=False,device='cpu')
        optimizer=object()
        model=SimpleNamespace(rollout_buffer=buffer,_settled_stats={},batch_size=32,
                              policy=SimpleNamespace(optimizer=optimizer))
        for fail in (False,True):
            calls=[]
            def original(value):
                calls.append(value.rollout_buffer)
                if fail:raise RuntimeError('optimizer fixture')
                return 'trained'
            with patch.dict(os.environ,{},clear=True),patch.object(SettledPPO,'train',original),\
                    patch('development.experiments.latency_tail_20261107.pacing.install'),\
                    patch('development.staged.pipeline_optimizations_20261106.observation.install'),\
                    patch('development.experiments.latency_tail_20261107.resync_spacing.install'),\
                    patch('development.experiments.latency_tail_20261107.adam_steps.restore_cpu_steps',
                          return_value=dict(moved=0,skipped=None)) as counters:
                runtime.install(SimpleNamespace())
                if fail:
                    with self.assertRaisesRegex(RuntimeError,'optimizer fixture'):SettledPPO.train(model)
                else:self.assertEqual(SettledPPO.train(model),'trained')
                self.assertIs(model.rollout_buffer,buffer)
                self.assertEqual(len(calls),1)
                counters.assert_called_once_with(optimizer)

    def test_wait_patch_preserves_super_and_one_wait(self):
        env=WaitFixture();calls=[];env._latency_rate_wait=calls.append
        step=replace_wait(WaitFixture.step,WaitFixture)
        self.assertEqual(step(env,4),5);self.assertEqual(calls,[.002])

    def test_precise_timer_deadline_and_close(self):
        timer=DeadlineWait(True)
        try:
            timer.wait(.002);timer.wait(0)
            self.assertGreaterEqual(timer.rows[0]['actual_ms'],2)
            self.assertEqual(timer.calls,2)
            with self.assertRaises(ValueError):timer.wait(float('nan'))
        finally:timer.close();timer.close()
        self.assertIsNone(timer.handle)

    def test_passthrough_phases_none_and_restore(self):
        with tempfile.TemporaryDirectory() as temp:
            raw=SimpleNamespace(output_dir=temp,episode_number=1,step_timings=[],
                _learning_observation_capture={},_reaction_guard_capture={},guard_rechecks=0)
            calls=[];predictions=[];closed=[]
            def predict(*args,**kwargs):
                predictions.append((args,kwargs));return (2,'unchanged')
            def step(action):
                calls.append(action)
                raw.step_timings.append(dict(press_timestamp=time.perf_counter() if action else None,
                                            press_roundtrip_ms=1))
                raw.guard_rechecks+=int(action==2)
                return ('obs',1,False,{})
            env=SimpleNamespace(step=step,close=lambda:closed.append(True))
            trainer=SimpleNamespace(create_live_env=lambda:(env,raw))
            with patch.object(ppo_eval_prediction,'predict_with_optional_probabilities',predict):
                install(trainer);factory=trainer.create_live_env;install(trainer)
                self.assertIs(factory,trainer.create_live_env)
                monitored,_=trainer.create_live_env()
                for episode,action in [(1,0),(1,2),(1,0),(2,3)]:
                    raw.episode_number=episode
                    raw._learning_observation_capture={'request_started_at':time.perf_counter()}
                    self.assertEqual(ppo_eval_prediction.predict_with_optional_probabilities('model'),(2,'unchanged'))
                    self.assertEqual(monitored.step(action),('obs',1,False,{}))
                monitored.close()
                self.assertIs(ppo_eval_prediction.predict_with_optional_probabilities,predict)
            self.assertEqual(calls,[0,2,0,3]);self.assertEqual(len(predictions),4);self.assertEqual(closed,[True])
            rows=[json.loads(s) for s in (Path(temp)/'evaluation_latency_trace.jsonl').read_text().splitlines()]
            self.assertEqual([r['phase'] for r in rows],['startup','ordinary','ordinary','reset'])
            self.assertEqual([r['first_movement'] for r in rows],[False,True,False,True])
            self.assertEqual(rows[1]['guard_deltas']['guard_rechecks'],1)
            summary=json.loads((Path(temp)/'evaluation_latency.json').read_text())
            self.assertEqual(summary['commands'],2);self.assertEqual(summary['decisions'],4)
            self.assertEqual(summary['phase_latency']['ordinary']['window_samples'],1)


if __name__=='__main__':unittest.main()
