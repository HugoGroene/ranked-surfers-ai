"""Stdlib-only routing/serialization tests; no Torch import or GPU allocation."""
import tempfile
import sys
from types import ModuleType, SimpleNamespace
import unittest
from unittest.mock import patch
import model_runtime


class TensorStub:
    def detach(self): return self
    def cpu(self): return self
    def clone(self): return TensorStub()


class StagingTests(unittest.TestCase):
    def test_checkpoint_export_preserves_runtime_flags_and_moments(self):
        step = TensorStub(); moment = object()
        state = dict(param_groups=[dict(params=[0],lr=.0003,fused=True,foreach=False)],
                     state={0:dict(step=step,exp_avg=moment)})
        saved = model_runtime.portable_optimizer_state(state,[dict(fused=None,foreach=None)])
        self.assertTrue(state['param_groups'][0]['fused'])
        self.assertIsNone(saved['param_groups'][0]['fused'])
        self.assertEqual(saved['param_groups'][0]['lr'],.0003)
        self.assertIs(state['state'][0]['step'],step)
        self.assertIsNot(saved['state'][0]['step'],step)
        self.assertIs(saved['state'][0]['exp_avg'],moment)

    def test_group_drift_rejected(self):
        with self.assertRaises(ValueError):
            model_runtime.portable_optimizer_state(dict(param_groups=[{}],state={}),[])

    def test_combination_keeps_worker_and_learner_entrypoints_separate(self):
        calls = []
        class Settled:
            def configure_settled(self,*args,**kwargs): calls.append('worker'); return 17
        def learner(model,*args,**kwargs): calls.append('learner'); return 23
        tools = ModuleType('tools'); tools.__path__ = []
        settled = ModuleType('tools.ppo_settled_learning'); settled.SettledPPO = Settled
        learning = ModuleType('tools.ppo_parallel_learning'); learning.configure_learner = learner
        tools.ppo_parallel_learning = learning
        modules = {'tools':tools,'tools.ppo_settled_learning':settled,'tools.ppo_parallel_learning':learning}
        with tempfile.TemporaryDirectory() as folder, patch.dict(sys.modules,modules), \
                patch.object(model_runtime,'policy_channels_last',side_effect=lambda p:calls.append('layout')), \
                patch.object(model_runtime,'fused_adam',side_effect=lambda p:'fused'):
            model_runtime.install_model_features('policy_fused')
            model_runtime.install_model_features('policy_fused')  # Idempotent.
            worker = Settled(); worker._settled_output = folder
            worker.policy = SimpleNamespace(optimizer='worker optimizer')
            worker.observation_space = {'image':SimpleNamespace(shape=(12,384,384))}
            coordinator = SimpleNamespace(policy=SimpleNamespace(optimizer='normal'),_settled_output=folder)
            self.assertEqual(worker.configure_settled(),17)
            self.assertEqual(learning.configure_learner(coordinator),23)
            self.assertEqual(calls,['worker','layout','learner'])
            self.assertEqual(worker.policy.optimizer,'worker optimizer')
            self.assertEqual(coordinator.policy.optimizer,'fused')


if __name__=='__main__': unittest.main()
