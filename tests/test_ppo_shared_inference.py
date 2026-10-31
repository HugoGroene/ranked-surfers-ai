import hashlib
import multiprocessing as mp
from pathlib import Path
import tempfile
import unittest
import numpy as np
import torch
from stable_baselines3 import PPO
from stable_baselines3.common.utils import obs_as_tensor
from tests.test_ppo_scaling_storage import TinyImageEnv,TinyFeatures
from tools.ppo_parallel_learning import policy_arrays,policy_hash,install_policy
from tools.ppo_shared_inference import InferenceHost,InferenceClient,forward_with_probabilities,RemotePolicy
from types import SimpleNamespace
from rs_v612_gameplay_guard import GuardUncertainError


class SharedInferenceProcessTests(unittest.TestCase):
    def test_distribution_diagnostics_preserve_forward_outputs_and_random_stream(self):
        model=PPO('MultiInputPolicy',TinyImageEnv(),n_steps=16,batch_size=8,device='cpu',seed=12,
                  policy_kwargs={'features_extractor_class':TinyFeatures})
        model.policy.set_training_mode(False)
        obs=obs_as_tensor({'image':np.zeros((1,12,16,16),np.uint8),'aux':np.zeros((1,3),np.float32)},'cpu')
        with torch.no_grad():
            torch.manual_seed(93);original=model.policy(obs);original_tail=torch.rand(4)
            torch.manual_seed(93);actual,probabilities=forward_with_probabilities(model.policy,obs);actual_tail=torch.rand(4)
            for a,b in zip(original,actual):torch.testing.assert_close(a,b,rtol=0,atol=0)
            torch.testing.assert_close(original_tail,actual_tail,rtol=0,atol=0)
            torch.testing.assert_close(probabilities,model.policy.get_distribution(obs).distribution.probs,rtol=0,atol=0)
        self.assertNotIn('_get_action_dist_from_latent',model.policy.__dict__)

    def test_cached_diagnostic_is_labelled_as_prior_decision_and_rejected_after_policy_change(self):
        import time
        client=SimpleNamespace(version=4,last_probabilities=(4,12,time.perf_counter(),np.array([.1,.2,.3,.2,.2],np.float32)))
        policy=RemotePolicy(client,{})
        snapshot=policy.diagnostic_snapshot()
        self.assertEqual(snapshot['policy_version'],4)
        self.assertEqual(snapshot['request_serial'],12)
        self.assertIn('not a prediction on the following observation',snapshot['source'])
        client.version=5
        self.assertIsNone(policy.diagnostic_snapshot())

    def test_spawned_service_matches_actual_policy_values_log_probs_and_versioned_updates(self):
        torch.set_num_threads(1)
        with tempfile.TemporaryDirectory() as directory:
            model=PPO('MultiInputPolicy',TinyImageEnv(),n_steps=16,batch_size=8,device='cpu',seed=22,
                      policy_kwargs={'features_extractor_class':TinyFeatures})
            source=Path(directory)/'source.zip';model.save(source)
            digest_before=hashlib.sha256(source.read_bytes()).hexdigest()
            host=InferenceHost(mp.get_context('spawn'),model.observation_space,2,
                dict(checkpoint=str(source),detector_mode='off',inference_device='cpu',inference_batch_wait_ms=0))
            clients=[]
            try:
                host.expect('ready',30)
                arrays=policy_arrays(model.policy);digest=policy_hash(arrays)
                host.install(0,arrays,digest,5)
                for resource in host.clients:
                    client=InferenceClient(resource,1000);client.version=0;client.digest=digest;clients.append(client)
                observation={'image':np.full((1,12,16,16),60,np.uint8),'aux':np.array([[.1,.3,.8]],np.float32)}
                for client in clients:
                    actions,values,log_probs=client.predict(observation)
                    with torch.no_grad():
                        expected,log_prob,_=model.policy.evaluate_actions(obs_as_tensor(observation,'cpu'),torch.from_numpy(actions))
                    np.testing.assert_allclose(values,expected.numpy(),rtol=1e-6,atol=1e-6)
                    np.testing.assert_allclose(log_probs,log_prob.numpy(),rtol=1e-6,atol=1e-6)
                    self.assertIsNotNone(client.last_probabilities)
                    np.testing.assert_allclose(client.predict(observation,kind='value'),values,rtol=1e-6,atol=1e-6)
                clients[0].version=1
                with self.assertRaises(GuardUncertainError):clients[0].predict(observation)
                arrays['action_net.bias'][1]+=.3
                digest=policy_hash(arrays);install_policy(model.policy,arrays,digest)
                host.install(1,arrays,digest,5)
                for client in clients:
                    client.version=1;client.digest=digest
                    action,values,log_probs=client.predict(observation,deterministic=True)
                    with torch.no_grad():expected=model.policy(obs_as_tensor(observation,'cpu'),deterministic=True)
                    np.testing.assert_array_equal(action,expected[0].numpy())
                    np.testing.assert_allclose(log_probs,expected[2].numpy(),rtol=1e-6,atol=1e-6)
                self.assertEqual(hashlib.sha256(source.read_bytes()).hexdigest(),digest_before)
            finally:
                for client in clients:client.close()
                host.close()
            self.assertFalse(host.process.is_alive())
            self.assertEqual(host.process.exitcode,0)
