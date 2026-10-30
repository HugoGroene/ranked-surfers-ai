#!/usr/bin/env python3
"""Source-hashed, opt-in v1.7.3 -> v1.7.4 React Pause click race fix.

Changes only pause_for_ppo_optimizer in the current trainer; no model or
checkpoint files are ever accessed. Verifies complete trainer SHA256 and
exact pre/post-click method anchors before replacement.

    python -B -m tools.patch_ppo_pause_click_v174 --check
    python -B -m tools.patch_ppo_pause_click_v174 --apply
"""
from pathlib import Path
import argparse
import ast
import hashlib
import os
import tempfile

ROOT = Path(__file__).resolve().parents[1]
TARGET = ROOT / 'train_rs_v612_ppo.py'
PAYLOAD = Path(__file__).resolve().parent / '_ppo_pause_click_v174_payload'
EXPECTED_V173_SHA256 = 'a65a5e0e384823cfc4f6d3bf08491b09ddf0202e997684520e107ff5893cfb23'
BACKUP_SUFFIX = '.pre_ppo_pause_click_v174.bak'

def digest(blob):
    return hashlib.sha256(blob).hexdigest()

def payloads(payload=PAYLOAD):
    before = (payload / 'old.txt').read_bytes()
    after = (payload / 'new.txt').read_bytes()
    assert before != after
    return before, after

def only_pause_method_changed(before, after):
    def reduced(text):
        tree = ast.parse(text)
        matches = [n for n in ast.walk(tree) if isinstance(n, ast.FunctionDef)
                   and n.name == 'pause_for_ppo_optimizer']
        if len(matches) != 1:
            raise RuntimeError('Expected one pause_for_ppo_optimizer method')
        matches[0].body = [ast.Pass()]
        return ast.dump(tree, include_attributes=False)
    if reduced(before) != reduced(after):
        raise RuntimeError('Patch unexpectedly changes code outside the pause method')

def prepare(root=ROOT, payload=PAYLOAD):
    target = root / 'train_rs_v612_ppo.py'
    helper = root / 'tools' / 'ppo_pause_control_recheck_v171.py'
    if not target.is_file() or not helper.is_file():
        raise RuntimeError('Missing v1.7.3 trainer or v1.7.1 pause recheck helper')
    old = target.read_bytes()
    if digest(old) != EXPECTED_V173_SHA256:
        raise RuntimeError(
            'Trainer source does not match exact installed v1.7.3: '
            f'SHA256={digest(old)} expected={EXPECTED_V173_SHA256}; nothing changed')
    anchor, replacement = payloads(payload)
    if old.count(anchor) != 1:
        raise RuntimeError(f'Expected one old pause-click block; found {old.count(anchor)}')
    updated = old.replace(anchor, replacement)
    compile(updated.decode('utf8'), str(target), 'exec')
    only_pause_method_changed(old.decode('utf8'), updated.decode('utf8'))
    if updated.count(b'from playwright.async_api import TimeoutError as PlaywrightTimeoutError') != 1:
        raise RuntimeError('Expected one Playwright timeout handler')
    backup = target.with_name(target.name + BACKUP_SUFFIX)
    if backup.exists():
        raise RuntimeError(f'Backup already exists: {backup.name}; no file changed')
    return target, backup, old, updated

def apply(root=ROOT, payload=PAYLOAD):
    target, backup, old, updated = prepare(root, payload)
    try:
        with backup.open('xb') as f:
            f.write(old)
        fd, temp = tempfile.mkstemp(prefix=target.name + '.v174.', dir=str(target.parent))
        try:
            with os.fdopen(fd, 'wb') as f:
                f.write(updated)
                f.flush()
                os.fsync(f.fileno())
            os.replace(temp, target)
        finally:
            if os.path.exists(temp):
                os.unlink(temp)
    except BaseException:
        if target.read_bytes() == old:
            backup.unlink(missing_ok=True)
        raise
    return backup

def main():
    p = argparse.ArgumentParser(description=__doc__)
    modes = p.add_mutually_exclusive_group(required=True)
    modes.add_argument('--check', action='store_true')
    modes.add_argument('--apply', action='store_true')
    args = p.parse_args()
    try:
        if args.check:
            _t, _b, old, after = prepare()
            print('[PPO PAUSE CLICK V174] DRY RUN PASS; exact installed v1.7.3 hash matched.')
            print('[PPO PAUSE CLICK V174] Only one guarded pause-button click block changes; '
                  f'+{len(after)-len(old)} bytes. No model or rewards modified.')
        else:
            backup = apply()
            print('[PPO PAUSE CLICK V174] Applied guarded Pause click timeout recovery.')
            print(f'[PPO PAUSE CLICK V174] Backup: {backup.name}')
    except Exception as exc:
        p.exit(2, f'[PPO PAUSE CLICK V174] REFUSED: {exc}\n')

if __name__ == '__main__':
    main()
