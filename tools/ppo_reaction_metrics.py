"""Command timing diagnostics; these do not measure Unity input consumption."""
import math
import numpy as np


def warm_policy(policy,observation_space):
    """Warm inference before gameplay reset; never sample/send an action or learn."""
    import torch
    from stable_baselines3.common.utils import obs_as_tensor
    policy.set_training_mode(False)
    observation={name:np.zeros((1,*space.shape),dtype=space.dtype)
                 for name,space in observation_space.spaces.items()}
    with torch.no_grad():
        policy(obs_as_tensor(observation,policy.device),deterministic=True)
    if policy.device.type == 'cuda':
        torch.cuda.synchronize(policy.device)


def command_latency(policy,guard,dispatch_at,roundtrip_ms=None):
    if dispatch_at is None:
        return None
    result={}
    for name,source in [('policy',policy or {}),('guard',guard or {})]:
        for field,label in [('request_started_at','request_to_command_ms'),
                            ('browser_frame_at','browser_swap_to_command_ms'),
                            ('paint_callback_at','paint_callback_to_command_ms'),
                            ('image_received_at','image_reply_to_command_ms'),
                            ('captured_at','capture_complete_to_command_ms')]:
            stamp=source.get(field)
            if stamp is not None:
                elapsed=(dispatch_at-stamp)*1000
                if math.isfinite(elapsed) and elapsed>=0:
                    result[name+'_'+label]=elapsed
    if roundtrip_ms is not None:
        result['command_roundtrip_ms']=float(roundtrip_ms)
    return result or None


def latency_summary(samples):
    result={}
    for key in {key for sample in samples for key in sample}:
        values=[s[key] for s in samples if key in s and math.isfinite(s[key])]
        if values:
            result[key]=dict(count=len(values),median_ms=float(np.median(values)),
                             p95_ms=float(np.percentile(values,95)),max_ms=float(max(values)))
    return dict(window_samples=len(samples),metrics=result,
                note='Command issue time, not Unity response. Browser swap timing is available only for validated stream metadata; screenshot presentation time is otherwise unknown.')
