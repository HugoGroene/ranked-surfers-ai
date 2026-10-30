#!/usr/bin/env python3
"""Opt-in *strict source-matched* v1.7 rollout-boundary MENU handoff install.

Usage:
    python -B -m tools.install_ppo_rollout_menu_handoff_v17 --check
    python -B -m tools.install_ppo_rollout_menu_handoff_v17 --apply

Will NOT overwrite a trainer/auditor different from the exact local v1.6
bundle supplied for this fix. Everything is validated before either write.
"""
from pathlib import Path
import argparse
import hashlib
import os
import tempfile


ROOT = Path(__file__).resolve().parents[1]
PAYLOAD = Path(__file__).resolve().parent / '_ppo_menu_handoff_v17_payload'
SPECS = (
    ('train_rs_v612_ppo.py', 'trainer.txt',
     '518de1205923165a31a7f26e52d2dbb52fa35dab1dcb2701dc4f342313b4d263'),
    ('tools/ppo_reward_timing_audit.py', 'auditor.txt',
     '9a44e6a0e7b5b948a0ff347caf74e24676074646a9646a8a6f7555a4c80ec616'),
)


def digest(data):
    return hashlib.sha256(data).hexdigest()


def validate(root=ROOT, payload=PAYLOAD):
    prepared = []
    for name, asset, expected in SPECS:
        target = root / name
        source = payload / asset
        if not target.is_file() or not source.is_file():
            raise RuntimeError(f'Missing required trainer, audit, or packaged payload: {name}/{asset}')
        old = target.read_bytes()
        if digest(old) != expected:
            raise RuntimeError(
                f'Refusing to replace {name}: installed source hash differs from the '
                f'exact uploaded v1.6 bundle. Actual SHA256={digest(old)}. '
                'Do NOT overwrite or git restore the patched trainer.')
        new = source.read_bytes()
        if old == new:
            raise RuntimeError(f'Already installed or identical: {name}')
        compile(new.decode('utf-8'), str(target), 'exec')
        backup = target.with_name(target.name + '.pre_ppo_menu_handoff_v17.bak')
        if backup.exists():
            raise RuntimeError(f'Backup already exists; refusing to overwrite: {backup}')
        prepared.append((target, old, new, backup))
    helper = root / 'tools' / 'ppo_rollout_menu_handoff_v17.py'
    if not helper.is_file():
        raise RuntimeError('Missing tools/ppo_rollout_menu_handoff_v17.py')
    compile(helper.read_text(encoding='utf-8'), str(helper), 'exec')
    return prepared


def apply(root=ROOT, payload=PAYLOAD):
    prepared = validate(root, payload)
    # Backups preserve the exact bytes supplied by the user; avoid partial
    # installation if a write fails. No training data or model files touched.
    for target, old, new, backup in prepared:
        backup.write_bytes(old)
    completed = []
    try:
        for target, old, new, backup in prepared:
            fd, tmp = tempfile.mkstemp(prefix=target.name + '.v17.', dir=str(target.parent))
            try:
                with os.fdopen(fd, 'wb') as f:
                    f.write(new)
                    f.flush()
                    os.fsync(f.fileno())
                os.replace(tmp, target)
            finally:
                if os.path.exists(tmp):
                    os.unlink(tmp)
            completed.append(target)
    except BaseException:
        for target, old, new, backup in prepared:
            if target in completed:
                target.write_bytes(old)
        raise
    return [str(x[0].relative_to(root)) for x in prepared]


def main():
    p = argparse.ArgumentParser(description=__doc__)
    mode = p.add_mutually_exclusive_group(required=True)
    mode.add_argument('--check', action='store_true')
    mode.add_argument('--apply', action='store_true')
    a = p.parse_args()
    try:
        if a.check:
            items = validate()
            print('[PPO MENU V17] DRY RUN OK. Strict source hashes matched.')
            print('[PPO MENU V17] Will change: ' + ', '.join(str(x[0].relative_to(ROOT)) for x in items))
            print('[PPO MENU V17] No source changes made.')
        else:
            changed = apply()
            print('[PPO MENU V17] Installed boundary MENU terminal handoff + UTF-16 log auditor.')
            print('[PPO MENU V17] Updated: ' + ', '.join(changed))
            print('[PPO MENU V17] Original trainer/auditor backed up. Model weights unchanged.')
    except Exception as exc:
        p.exit(2, '[PPO MENU V17] REFUSED: ' + str(exc) + '\n')


if __name__ == '__main__':
    main()
