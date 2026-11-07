"""Restore ordinary Adam's CPU scalar counters after CUDA checkpoint loading."""
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
            if not state or 'step' not in state: continue
            step = state['step']
            if not torch.is_tensor(step) or step.numel()!=1 or step.requires_grad:
                raise ValueError('Unexpected ordinary Adam step counter')
            if step.device.type != 'cpu':
                state['step'] = step.detach().cpu()
                moved += 1
    return dict(moved=moved, skipped=None)


def install():
    import json
    from pathlib import Path
    from tools import ppo_parallel_learning as learning
    if getattr(learning,'_candidate_cpu_steps',False): return
    original = learning.configure_learner
    def configure(model,*args,**kwargs):
        result = original(model,*args,**kwargs)
        report = restore_cpu_steps(model.policy.optimizer)
        (Path(model._settled_output)/'adam_cpu_steps.json').write_text(json.dumps(report))
        print('[ADAM CPU STEPS] '+json.dumps(report),flush=True)
        return result
    learning.configure_learner = configure
    learning._candidate_cpu_steps = True
