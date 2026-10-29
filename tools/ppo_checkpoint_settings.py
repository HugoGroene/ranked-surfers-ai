"""Explicit, weight-preserving overrides of saved PPO training settings."""
import json
import zipfile


SETTING_NAMES = ('n_steps', 'batch_size', 'n_epochs', 'learning_rate', 'gamma', 'ent_coef', 'target_kl', 'seed')


def load_training_checkpoint(args, env, *, device):
    from stable_baselines3 import PPO

    requested = getattr(args, 'override_ppo_settings', [])
    if any(key not in SETTING_NAMES for key in requested):
        raise ValueError('Unsupported saved PPO setting override')
    updates = {key: getattr(args, key) for key in requested}
    with zipfile.ZipFile(args.resume) as archive:
        saved = json.loads(archive.read('data'))
    before = {key: saved.get(key) for key in SETTING_NAMES}
    # Avoid copying serialized callable payloads into human-readable reports.
    before = {key: ('serialized callable' if isinstance(value, dict) else value)
              for key, value in before.items()}
    model = PPO.load(str(args.resume), env=env, device=device, **updates)
    if args.steps % model.n_steps:
        raise ValueError(f'Requested {args.steps} steps is not divisible by the effective {model.n_steps}-step rollout')
    if model.n_steps % model.batch_size:
        raise ValueError('Effective rollout must be divisible by effective minibatch size')
    after = {key: getattr(model, key) for key in SETTING_NAMES}
    if callable(after['learning_rate']):
        after['learning_rate'] = 'callable'
    after['learning_rate_at_start'] = float(model.lr_schedule(1.))
    return model, {'explicit_overrides': updates, 'saved_settings': before, 'effective_settings': after}
