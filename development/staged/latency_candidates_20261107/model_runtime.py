"""Lazy Torch imports: isolated policy-layout and learner-optimizer candidates."""
import copy
import json
from pathlib import Path


def policy_channels_last(policy):
    import torch
    from development.staged.latency_candidates_20261107.runtime import channels_last
    if hasattr(policy, '_candidate_rgb_hook'):
        return False
    extractor = policy.features_extractor
    if (not policy.share_features_extractor or type(extractor).__name__ != 'SpatialExtractor'
            or not hasattr(extractor, 'rgb')):
        raise ValueError('Policy layout candidate requires the shared SpatialExtractor')
    parameters = list(extractor.rgb.parameters())
    if any(p.dtype != torch.float32 for p in parameters):
        raise ValueError('Policy layout candidate is FP32 only')
    identities = [id(p) for p in policy.parameters()]
    policy._candidate_rgb_hook = channels_last(extractor.rgb, torch)
    if identities != [id(p) for p in policy.parameters()]:
        raise RuntimeError('Layout conversion replaced optimizer parameter identities')
    return True


def portable_optimizer_state(state, original_flags):
    """Export normal Adam flags/CPU steps without mutating the fused runtime."""
    if len(state['param_groups']) != len(original_flags):
        raise ValueError('Optimizer parameter groups changed during the experiment')
    exported = dict(state)
    exported['param_groups'] = [dict(group, **flags)
                               for group, flags in zip(state['param_groups'], original_flags)]
    exported['state'] = {}
    for index, value in state['state'].items():
        exported['state'][index] = dict(value)
        if 'step' in value:
            exported['state'][index]['step'] = value['step'].detach().cpu().clone()
    return exported


def fused_adam(original):
    """Clone existing momentum/steps; no reset of Adam or PPO hyperparameters."""
    import torch
    if type(original) is not torch.optim.Adam:
        raise ValueError('Fused candidate requires ordinary torch.optim.Adam')
    for group in original.param_groups:
        if group.get('capturable') or group.get('differentiable') or group.get('fused'):
            raise ValueError('Unsupported source Adam execution mode')
        if not group['params'] or any(p.device.type != 'cuda' or p.dtype != torch.float32
                                      for p in group['params']):
            raise ValueError('Fused candidate requires CUDA FP32 parameter groups')
    # Deep copy prevents the source optimizer from sharing mutable moment tensors.
    state = copy.deepcopy(original.state_dict())
    flags = [{key: group.get(key) for key in ('fused','foreach')} for group in original.param_groups]
    groups = [dict(group, params=list(group['params']), fused=True, foreach=False)
              for group in original.param_groups]
    options = dict(original.defaults, fused=True, foreach=False)
    candidate = torch.optim.Adam(groups, **options)
    for group in state['param_groups']:
        group.update(fused=True, foreach=False)
    # PyTorch moves the step tensors to their parameter's device for fused Adam.
    candidate.load_state_dict(state)
    for parameter, value in candidate.state.items():
        if value and value['step'].device != parameter.device:
            raise RuntimeError('Fused Adam step tensor is on the wrong device')
    def export(optimizer, snapshot):
        return portable_optimizer_state(snapshot, flags)
    candidate.register_state_dict_post_hook(export)
    return candidate


def install_model_features(arm):
    if arm not in ('policy_layout','fused_adam','policy_fused'):
        raise ValueError('Unknown model candidate')
    if arm in ('policy_layout','policy_fused'):
        from tools.ppo_settled_learning import SettledPPO
        if not getattr(SettledPPO, '_candidate_policy_layout', False):
            configure_worker_original = SettledPPO.configure_settled
            def configure(model, *args, **kwargs):
                result = configure_worker_original(model, *args, **kwargs)
                policy_channels_last(model.policy)
                (Path(model._settled_output)/'policy_layout_candidate.json').write_text(json.dumps(dict(
                    mode='channels_last', scope='worker RGB encoder only', dtype='float32',
                    input_shape=list(model.observation_space['image'].shape))))
                return result
            SettledPPO.configure_settled = configure
            SettledPPO._candidate_policy_layout = True
    if arm in ('fused_adam','policy_fused'):
        from tools import ppo_parallel_learning as learning
        if not getattr(learning, '_candidate_fused_adam', False):
            configure_learner_original = learning.configure_learner
            def configure(model, *args, **kwargs):
                result = configure_learner_original(model, *args, **kwargs)
                model.policy.optimizer = fused_adam(model.policy.optimizer)
                (Path(model._settled_output)/'fused_adam_candidate.json').write_text(json.dumps(dict(
                    mode='fused', scope='central learner only', restored_momentum=True,
                    checkpoint_export='original Adam flags and CPU step tensors')))
                return result
            learning.configure_learner = configure
            learning._candidate_fused_adam = True
