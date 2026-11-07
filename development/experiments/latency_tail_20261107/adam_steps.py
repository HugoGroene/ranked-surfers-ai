"""Keep ordinary Adam's scalar counters on CPU after CUDA checkpoint loading.

SB3's map_location can move these counters onto CUDA along with the moments.
Ordinary Adam then reads them back as Python scalars on every minibatch. Restore
the location used by Adam's own initialization, without changing values/moments.
Fused, capturable and differentiable variants have different requirements.
"""


def restore_cpu_steps(optimizer):
    import torch
    if type(optimizer) is not torch.optim.Adam:
        return dict(moved=0, skipped='not ordinary Adam')
    moved = 0
    for group in optimizer.param_groups:
        if group.get('capturable') or group.get('fused') or group.get('differentiable'):
            continue
        for parameter in group['params']:
            state = optimizer.state.get(parameter)
            if not state or 'step' not in state:
                continue
            step = state['step']
            if not torch.is_tensor(step) or step.numel() != 1 or step.requires_grad:
                raise ValueError('Unexpected ordinary Adam step counter')
            if step.device.type != 'cpu':
                state['step'] = step.detach().cpu()
                moved += 1
    return dict(moved=moved, skipped=None)
