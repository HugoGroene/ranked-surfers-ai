"""Read evaluation probabilities from the action's existing model forward."""
import numpy as np


def predict_with_optional_probabilities(model, observation, *, deterministic, sample_probabilities):
    if not sample_probabilities:
        action, _ = model.predict(observation, deterministic=deterministic)
        return action, None
    policy = model.policy
    original = policy.get_distribution
    had_instance_method = 'get_distribution' in policy.__dict__
    previous_instance_method = policy.__dict__.get('get_distribution')
    captured = []

    def observe(*args, **kwargs):
        distribution = original(*args, **kwargs)
        captured.append(distribution)
        return distribution

    policy.get_distribution = observe
    try:
        action, _ = model.predict(observation, deterministic=deterministic)
    finally:
        if had_instance_method:
            policy.get_distribution = previous_instance_method
        else:
            delattr(policy, 'get_distribution')
    if not captured:
        return action, None
    probabilities = captured[-1].distribution.probs.detach().cpu().numpy()[0]
    if not np.isfinite(probabilities).all():
        raise RuntimeError('Non-finite evaluation probabilities')
    return action, {'probabilities': [round(float(p), 5) for p in probabilities],
                    'max_probability': round(float(probabilities.max()), 5)}
