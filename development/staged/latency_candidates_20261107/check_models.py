"""Idle-only policy/Adam parity. No game, no source checkpoint replacement."""
import argparse
import copy
import json
from pathlib import Path
import sys
import unittest
from validation import ROOT, MODEL_RESULT, digest, fingerprints, require_idle


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--checkpoint',type=Path,required=True)
    parser.add_argument('--cuda',action='store_true')
    a = parser.parse_args()
    require_idle()  # Must precede Torch import/device initialization.
    sys.path.append(str(ROOT/'development/staged/pipeline_optimizations_20261106'))
    from development.staged.multimap_bc_20261106.common import contract
    contract(a.checkpoint)
    import numpy as np
    import torch
    from stable_baselines3 import PPO
    from tools.ppo_parallel_learning import install_policy, policy_arrays, policy_hash
    from tools.ppo_settled_learning import SettledPPO
    from development.staged.pipeline_optimizations_20261106.test_runtime import Learner
    from model_runtime import policy_channels_last, fused_adam
    from development.experiments.latency_tail_20261107.adam_steps import restore_cpu_steps
    import test_model_staging
    torch.set_num_threads(1)
    if a.cuda and not torch.cuda.is_available(): raise RuntimeError('CUDA unavailable; cannot certify CUDA candidates')
    device = 'cuda' if a.cuda else 'cpu'
    source_hash = digest(a.checkpoint)
    report = dict(passed=False,cuda=a.cuda,checkpoint_sha256=source_hash,sources=fingerprints(),
                  torch_version=torch.__version__,device=device,checks=[])
    MODEL_RESULT.parent.mkdir(parents=True,exist_ok=True)
    try:
        result = unittest.TextTestRunner(verbosity=2).run(unittest.defaultTestLoader.loadTestsFromModule(test_model_staging))
        if not result.wasSuccessful(): raise AssertionError('Staging tests failed')
        model = PPO.load(str(a.checkpoint),device=device)
        original = model.policy
        candidate = copy.deepcopy(original)
        original.set_training_mode(False); candidate.set_training_mode(False)
        before = policy_hash(policy_arrays(candidate))
        policy_channels_last(candidate)
        assert before == policy_hash(policy_arrays(candidate)), 'Layout altered weight values'
        generator = torch.Generator(device=device).manual_seed(462)
        observations = {'image':torch.randint(0,256,(2,*model.observation_space['image'].shape),
                                             dtype=torch.uint8,device=device,generator=generator),
                        'aux':torch.rand((2,*model.observation_space['aux'].shape),device=device,generator=generator)}
        retained = {k:v.clone() for k,v in observations.items()}
        def compare():
            with torch.no_grad():
                x = original.get_distribution(observations).distribution.probs
                y = candidate.get_distribution(observations).distribution.probs
                torch.testing.assert_close(x,y,atol=2e-6,rtol=2e-5)
                torch.testing.assert_close(original.predict_values(observations),candidate.predict_values(observations),atol=2e-6,rtol=2e-5)
                for seed in (17,29,41):
                    torch.manual_seed(seed); old = original(observations)
                    torch.manual_seed(seed); new = candidate(observations)
                    assert torch.equal(old[0],new[0]), 'Sampled action changed'
                    for v,w in zip(old[1:],new[1:]): torch.testing.assert_close(v,w,atol=2e-6,rtol=2e-5)
                assert torch.equal(x.argmax(-1),y.argmax(-1)), 'Deterministic action changed'
        compare()
        # Real worker version installation must keep the new layout AND values.
        with torch.no_grad(): original.action_net.bias.add_(.01*torch.arange(5,device=device))
        arrays = policy_arrays(original)
        install_policy(candidate,arrays,policy_hash(arrays))
        compare()
        assert candidate.features_extractor.rgb[0].weight.is_contiguous(memory_format=torch.channels_last)
        for key,value in retained.items(): assert torch.equal(value,observations[key]), 'Observation mutated'
        report['checks'].append('policy probabilities/value/sampling/weight install/input ownership')
        if a.cuda:
            for masks in ([1]*8,[0,1,0,1,1,0,1,0],[0]*8):
                old,new = Learner(masks),Learner(masks)
                for learner in (old,new):
                    learner.policy.to(device); learner.rollout_buffer.device=torch.device(device)
                    for p in learner.policy.parameters(): p.grad=torch.full_like(p,.01)
                    learner.policy.optimizer.step();learner.policy.optimizer.zero_grad(set_to_none=True)
                    # Reproduce SB3's map_location='cuda' restore of CPU steps.
                    for state in learner.policy.optimizer.state.values(): state['step']=state['step'].to(device)
                moments={id(p):(s['exp_avg'],s['exp_avg_sq']) for p,s in new.policy.optimizer.state.items()}
                moved=restore_cpu_steps(new.policy.optimizer)
                assert moved['moved']==len(new.policy.optimizer.state)
                for p,state in new.policy.optimizer.state.items():
                    assert state['exp_avg'] is moments[id(p)][0] and state['exp_avg_sq'] is moments[id(p)][1]
                    assert state['step'].device.type=='cpu'
                assert restore_cpu_steps(new.policy.optimizer)['moved']==0
                for iteration in range(2):
                    np.random.seed(93+iteration); SettledPPO.train(old)
                    np.random.seed(93+iteration); SettledPPO.train(new)
                    for key,value in old.policy.state_dict().items():
                        torch.testing.assert_close(value,new.policy.state_dict()[key],rtol=0,atol=0)
                    for p,q in zip(old.policy.parameters(),new.policy.parameters()):
                        for key,value in old.policy.optimizer.state[p].items():
                            torch.testing.assert_close(value.cpu(),new.policy.optimizer.state[q][key].cpu(),rtol=0,atol=0)
                    assert old.records==new.records and old._settled_stats==new._settled_stats
                report['checks'].append('CPU Adam counters exact masked update parity: '+str(masks))
            # Also exercise migration of this actual checkpoint's optimizer.
            converted = fused_adam(original.optimizer)
            assert restore_cpu_steps(converted)['moved']==0, 'Fused steps must stay on GPU'
            exported = converted.state_dict(); existing = original.optimizer.state_dict()
            for i,state in existing['state'].items():
                for key,value in state.items():
                    if torch.is_tensor(value): torch.testing.assert_close(value.cpu(),exported['state'][i][key].cpu(),rtol=0,atol=0)
                    else: assert value == exported['state'][i][key]
            del converted
            for masks in ([1]*8,[0,1,0,1,1,0,1,0],[0]*8):
                old,new = Learner(masks),Learner(masks)
                for learner in (old,new):
                    learner.policy.to(device)
                    learner.rollout_buffer.device = torch.device(device)
                    for parameter in learner.policy.parameters(): parameter.grad = torch.full_like(parameter,.01)
                    learner.policy.optimizer.step(); learner.policy.optimizer.zero_grad(set_to_none=True)
                new.policy.optimizer = fused_adam(new.policy.optimizer)
                actor_before = copy.deepcopy(new.policy.actor.state_dict())
                actor_state = {id(p):copy.deepcopy(new.policy.optimizer.state[p]) for p in new.policy.actor.parameters()}
                # Two consecutive updates exercise the migrated state and continuity.
                for iteration in range(2):
                    np.random.seed(75+iteration); SettledPPO.train(old)
                    np.random.seed(75+iteration); SettledPPO.train(new)
                    for key,value in old.policy.state_dict().items():
                        torch.testing.assert_close(value,new.policy.state_dict()[key],atol=2e-6,rtol=2e-5)
                    assert old.records == new.records and old._settled_stats == new._settled_stats
                    assert old._n_updates == new._n_updates
                    for key in old.logs:
                        np.testing.assert_allclose(old.logs[key],new.logs[key],atol=2e-6,rtol=2e-5)
                    for p,q in zip(old.policy.parameters(),new.policy.parameters()):
                        for key,value in old.policy.optimizer.state[p].items():
                            torch.testing.assert_close(value.cpu(),new.policy.optimizer.state[q][key].cpu(),atol=2e-6,rtol=2e-5)
                if not any(masks):
                    for key,value in actor_before.items(): assert torch.equal(value,new.policy.actor.state_dict()[key])
                    for p in new.policy.actor.parameters():
                        for key,value in actor_state[id(p)].items():
                            assert torch.equal(value,new.policy.optimizer.state[p][key]), 'Passive actor momentum changed'
                # Standard Adam must be able to resume exported state, without
                # silently inheriting the staged fused setting or CUDA step clock.
                saved = new.policy.optimizer.state_dict()
                assert all(not g.get('fused') for g in saved['param_groups'])
                assert all(v['step'].device.type=='cpu' for v in saved['state'].values())
                resumed = torch.optim.Adam(new.policy.parameters())
                resumed.load_state_dict(saved)
                for p in new.policy.parameters(): p.grad = torch.full_like(p,.001)
                resumed.step(); resumed.zero_grad(set_to_none=True)
                report['checks'].append('masked PPO + momentum/export: '+str(masks))
        report['source_unchanged'] = digest(a.checkpoint)==source_hash
        report['passed'] = report['source_unchanged']
    except BaseException as exc:
        report['error'] = repr(exc)
        raise
    finally:
        MODEL_RESULT.write_text(json.dumps(report,indent=2))
        print('MODEL CHECK REPORT:',MODEL_RESULT,flush=True)
    if not report['passed']: raise SystemExit(1)


if __name__=='__main__': main()
