import tempfile
import unittest
from pathlib import Path
from tools.ppo_reaction_metrics import command_latency,latency_summary
from tools.ppo_worker_preview import previews_enabled,write_status


class ReactionTests(unittest.TestCase):
    def test_warmup_does_not_train_change_weights_or_sample_rng(self):
        import torch
        from stable_baselines3 import PPO
        from tests.test_ppo_parallel_learning import SmallEnv
        from tools.ppo_reaction_metrics import warm_policy
        model=PPO('MultiInputPolicy',SmallEnv(),n_steps=16,batch_size=8,device='cpu',seed=42)
        before={key:value.clone() for key,value in model.policy.state_dict().items()}
        rng=torch.get_rng_state().clone()
        warm_policy(model.policy,model.observation_space)
        self.assertTrue(torch.equal(rng,torch.get_rng_state()))
        for key,value in model.policy.state_dict().items():
            self.assertTrue(torch.equal(before[key],value),key)
        self.assertEqual(model.num_timesteps,0)
        self.assertFalse(model.policy.optimizer.state)

    def test_fresh_guard_does_not_hide_old_policy_image(self):
        policy=dict(request_started_at=.8,image_received_at=.86,captured_at=.87)
        guard=dict(request_started_at=.95,image_received_at=.98,captured_at=.99)
        result=command_latency(policy,guard,1,7)
        self.assertAlmostEqual(result['policy_request_to_command_ms'],200)
        self.assertAlmostEqual(result['policy_capture_complete_to_command_ms'],130)
        self.assertAlmostEqual(result['guard_capture_complete_to_command_ms'],10)
        self.assertEqual(result['command_roundtrip_ms'],7)

    def test_suppressed_or_noop_action_is_not_reported_as_dispatch(self):
        self.assertIsNone(command_latency(dict(captured_at=.5),{},None))

    def test_percentiles_include_slow_tail(self):
        result=latency_summary([dict(age=v) for v in [1,2,3,100]])['metrics']['age']
        self.assertEqual(result['median_ms'],2.5)
        self.assertEqual(result['max_ms'],100)
        self.assertGreater(result['p95_ms'],80)

    def test_status_only_updates_do_not_create_or_modify_image(self):
        with tempfile.TemporaryDirectory() as directory:
            folder=Path(directory)
            self.assertTrue(previews_enabled(folder))
            (folder/'preview_control.json').write_text('{"enabled":false}')
            self.assertFalse(previews_enabled(folder))
            (folder/'live.jpg').write_bytes(b'existing image')
            self.assertTrue(write_status(folder,dict(phase='collecting',image_updated_at=1)))
            self.assertEqual((folder/'live.jpg').read_bytes(),b'existing image')


if __name__ == '__main__':
    unittest.main()
