#!/usr/bin/env python3
"""Auditor-only correction: allow corroborated MENU boundary after pre-pause GAMEPLAY.

No changes to trainer, PPO, Unity, gameplay guard, model checkpoints, or rewards.
Usage: python -B -m tools.patch_ppo_reward_precheck_v1751 --check|--apply
"""
import argparse
import ast
import hashlib
import os
from pathlib import Path
import tempfile

ROOT = Path(__file__).resolve().parents[1]
TARGET = 'tools/ppo_reward_timing_audit.py'
SHA_OLD = '32688924c0f7eb2fe16bfb7c8c70c5edeaa6dd171389f5e232a618dd68343300'
OLD = '''    if prechecks and prechecks != verified:
        failures.append(f'Pre-pause classifier checks {prechecks} != verified freezes {verified}')
'''
NEW = '''    # A verified GAMEPLAY pre-pause screenshot can race with an actual death:
    # pause_for_ppo_optimizer() logged the screenshot, but the control vanished
    # and the independently corroborated MENU handoff replaced the freeze.
    # Each confirmed boundary MENU can explain AT MOST one such extra check.
    # Missing or otherwise unexplained checks still fail closed.
    if prechecks and not (verified <= prechecks <= verified + menu_boundaries):
        failures.append(
            f'Pre-pause classifier checks {prechecks} outside expected '
            f'{verified}..{verified + menu_boundaries} '
            f'(verified freezes plus corroborated boundary MENU races)')
'''


def sha(blob):
    return hashlib.sha256(blob).hexdigest()


def prepare(root=ROOT):
    target = Path(root) / TARGET
    if not target.exists():
        raise RuntimeError(f'Missing {target}')
    raw = target.read_bytes()
    if sha(raw) != SHA_OLD:
        raise RuntimeError(f'Unexpected auditor SHA256 {sha(raw)}; expected {SHA_OLD}; no changes made')
    source = raw.decode('utf-8')
    if source.count(OLD) != 1:
        raise RuntimeError('Expected auditor precheck condition missing or duplicated')
    replacement = source.replace(OLD, NEW).encode('utf-8')
    compile(replacement, str(target), 'exec')
    # Other code and behavior remain byte-identical outside the specified block.
    assert replacement != raw
    backup = target.with_name(target.name + '.pre_ppo_reward_precheck_v1751.bak')
    if backup.exists():
        raise RuntimeError(f'Backup already exists: {backup}; no changes made')
    return target, backup, raw, replacement


def apply(root=ROOT):
    target, backup, raw, replacement = prepare(root)
    with backup.open('xb') as output:
        output.write(raw)
    fd, path = tempfile.mkstemp(prefix='.ppo_reward_precheck_v1751.', dir=str(target.parent))
    try:
        with os.fdopen(fd, 'wb') as output:
            output.write(replacement)
            output.flush()
            os.fsync(output.fileno())
        os.replace(path, target)
    finally:
        if os.path.exists(path):
            os.unlink(path)
    return backup


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    mode = parser.add_mutually_exclusive_group(required=True)
    mode.add_argument('--check', action='store_true')
    mode.add_argument('--apply', action='store_true')
    args = parser.parse_args()
    try:
        if args.check:
            prepare()
            print('[PPO AUDIT 1.7.5.1] DRY RUN PASS; exact v1.7.5 auditor source matched. No files changed.')
        else:
            backup = apply()
            print('[PPO AUDIT 1.7.5.1] Installed auditor-only MENU precheck counting correction.')
            print(f'[PPO AUDIT 1.7.5.1] Backup: {backup.name}; trainer and checkpoints unchanged.')
    except Exception as exc:
        parser.exit(2, f'[PPO AUDIT 1.7.5.1] REFUSED: {exc}\n')


if __name__ == '__main__':
    main()
