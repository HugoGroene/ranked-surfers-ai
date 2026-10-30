#!/usr/bin/env python3
"""Minimal fail-closed hardening for the opt-in PPO rollout-resync patch.

Use --check first, then --apply. Only changes train_rs_v612_ppo.py in CWD;
keeps a byte-for-byte backup; refuses unexpected input or repeat application.
"""
import argparse
import difflib
import os
from pathlib import Path

TARGET = Path('train_rs_v612_ppo.py')
BACKUP = Path('train_rs_v612_ppo.py.pre_ppo_resync_hardening.bak')

OLD = '''                    if observation is None:
                        print(f"[PPO RESYNC] skipped: {skip_reason}; existing pre-action guard remains active", flush=True)
                    else:
                        replace_cached_observation(self.model, observation)
                        age_ms = (time.perf_counter() - capture_at) * 1000
                        print(f"[PPO RESYNC] refreshed 4-frame stack; cached age now {age_ms:.1f}ms", flush=True)
'''

NEW = '''                    if observation is None:
                        # Never allow the first action after an optimizer pause to
                        # consume a stale observation when refreshing was unsafe.
                        raise GuardUncertainError(
                            f"Cannot verify gameplay after PPO update: {skip_reason}. "
                            "Stopping instead of acting on the stale cached image.")
                    age_ms = (time.perf_counter() - capture_at) * 1000
                    if not (0 <= age_ms < 100.0):
                        raise GuardUncertainError(
                            f"Refresh itself is stale ({age_ms:.1f}ms). "
                            "Stopping instead of acting on an old observation.")
                    replace_cached_observation(self.model, observation)
                    print(f"[PPO RESYNC] refreshed 4-frame stack; cached age now {age_ms:.1f}ms", flush=True)
'''


def patch(text):
    if 'def refresh_after_ppo_optimizer(self):' not in text or '--resync-rollouts' not in text:
        raise ValueError('Expected rollout resync v1 is not installed; refusing')
    count = text.count(OLD)
    if count != 1:
        if NEW in text:
            raise ValueError('Resync hardening already installed; refusing duplicate application')
        raise ValueError(f'Expected one exact resync callback block, found {count}; refusing to modify')
    changed = text.replace(OLD, NEW, 1)
    compile(changed, str(TARGET), 'exec')
    return changed


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    flags = parser.add_mutually_exclusive_group(required=True)
    flags.add_argument('--check', action='store_true')
    flags.add_argument('--apply', action='store_true')
    a = parser.parse_args()
    raw = TARGET.read_bytes()
    crlf = b'\r\n' in raw
    text = raw.decode('utf-8').replace('\r\n', '\n')
    hardened = patch(text)
    output = hardened.replace('\n', '\r\n') if crlf else hardened
    if a.check:
        print(''.join(difflib.unified_diff(
            text.splitlines(True), hardened.splitlines(True),
            fromfile='a/train_rs_v612_ppo.py', tofile='b/train_rs_v612_ppo.py')))
        print('DRY RUN ONLY; no changes made')
        return
    if BACKUP.exists():
        parser.error(f'Backup exists; refusing overwrite: {BACKUP}')
    BACKUP.write_bytes(raw)
    tmp = TARGET.with_name(TARGET.name + '.hardening_tmp')
    try:
        tmp.write_bytes(output.encode('utf-8'))
        os.replace(tmp, TARGET)
    finally:
        if tmp.exists():
            tmp.unlink()
    print('Applied fail-closed PPO resync hardening. Original backup:', BACKUP)
    print('The --resync-rollouts flag is STILL opt-in and OFF by default.')


if __name__ == '__main__':
    main()
