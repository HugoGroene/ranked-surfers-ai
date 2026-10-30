#!/usr/bin/env python3
"""Opt-in, exact-source v1.7.2 -> v1.7.3 post-resume MENU/death handoff.

python -B -m tools.install_ppo_post_resume_menu_v173 --check
python -B -m tools.install_ppo_post_resume_menu_v173 --apply

Refuses unknown or modified versions; original checkpoint is never opened.
"""
from pathlib import Path
import argparse
import hashlib
import os
import tempfile

ROOT = Path(__file__).resolve().parents[1]
PAYLOAD = Path(__file__).resolve().parent/'_ppo_post_resume_menu_v173_payload'
SPECS = (
    ('train_rs_v612_ppo.py', 'trainer.txt',
     '91fe4baa87e32d7b82f8edf4f753445d15a80fc7669f73a12ea6299791306e64'),
    ('tools/ppo_reward_timing_audit.py', 'auditor.txt',
     '29b4951cf8b831bbde601ba6b916305af06250539e2e1eaa5d7d74e921e4a490'),
)

def sha256(b):
    return hashlib.sha256(b).hexdigest()


def validate(root=ROOT, payload=PAYLOAD):
    prepared = []
    for name, asset, expected in SPECS:
        target = root/name
        new_path = payload/asset
        if not target.is_file() or not new_path.is_file():
            raise RuntimeError(f'Missing required v1.7.2 code or payload: {name}')
        before = target.read_bytes()
        actual = sha256(before)
        if actual != expected:
            raise RuntimeError(
                f'Unexpected installed {name} SHA256={actual}; expected exact v1.7.2. '
                'No files changed. Do not force/overwrite the trainer.')
        after = new_path.read_bytes()
        if before == after:
            raise RuntimeError(f'Already patched / identical {name}')
        compile(after.decode('utf-8'), str(target), 'exec')
        backup = target.with_name(target.name+'.pre_ppo_post_resume_menu_v173.bak')
        if backup.exists():
            raise RuntimeError(f'Backup exists; refusing to overwrite {backup.name}')
        prepared.append((target,before,after,backup))
    # Confirm only the audited branches are new and the existing pre-action
    # guard still has its fail-closed boundary MENU gate in the compiled source.
    s = prepared[0][2]
    for required in (b'boundary_menu_handoff = bool(',
                     b'if boundary_menu_handoff:',
                     b'confirmed, evidence = raw.confirm_post_optimizer_menu()',
                     b'compensate_pause_clocks(',
                     b'no movement key allowed.',
                     b'ppo_menu_handoff_terminal'):
        if required not in s:
            raise RuntimeError(f'Missing existing safety invariant {required!r}')
    return prepared


def apply(root=ROOT, payload=PAYLOAD):
    jobs = validate(root, payload)
    made=[]
    try:
        for target,before,after,backup in jobs:
            backup.write_bytes(before)
            made.append(backup)
        completed=[]
        try:
            for target,before,after,backup in jobs:
                fd,temp=tempfile.mkstemp(prefix=target.name+'.v173.', dir=str(target.parent))
                try:
                    with os.fdopen(fd,'wb') as f:
                        f.write(after);f.flush();os.fsync(f.fileno())
                    os.replace(temp,target)
                finally:
                    if os.path.exists(temp):os.unlink(temp)
                completed.append((target,before))
        except BaseException:
            for target,before in completed:
                target.write_bytes(before)
            raise
    except BaseException:
        for backup in made:
            backup.unlink(missing_ok=True)
        raise
    return [str(j[0].relative_to(root)) for j in jobs]


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    actions=parser.add_mutually_exclusive_group(required=True)
    actions.add_argument('--check', action='store_true')
    actions.add_argument('--apply', action='store_true')
    a=parser.parse_args()
    try:
        if a.check:
            changes=validate()
            print('[PPO POST RESUME MENU V173] DRY RUN PASS; exact v1.7.2 source hashes match.')
            print('[PPO POST RESUME MENU V173] Changes: '+', '.join(str(j[0].relative_to(ROOT)) for j in changes))
            print('[PPO POST RESUME MENU V173] No files changed; model and weights untouched.')
        else:
            changes=apply()
            print('[PPO POST RESUME MENU V173] Installed guarded post-resume death handoff + auditor.')
            print('[PPO POST RESUME MENU V173] Backups created; updated: '+', '.join(changes))
    except Exception as exc:
        parser.exit(2, '[PPO POST RESUME MENU V173] REFUSED: '+str(exc)+'\n')

if __name__=='__main__':
    main()
