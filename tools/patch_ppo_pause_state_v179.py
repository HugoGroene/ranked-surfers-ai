#!/usr/bin/env python3
"""Install v1.7.9 only on the exact v1.7.8 trainer; back up before replacement."""
from __future__ import annotations
import argparse
import hashlib
import os
from pathlib import Path
import tempfile

ROOT = Path(__file__).resolve().parents[1]
TRAINER = ROOT/'train_rs_v612_ppo.py'
PAYLOAD = ROOT/'tools/_ppo_pause_state_v179_payload/trainer.txt'
VISUAL = ROOT/'tools/ppo_pause_visual_v179.py'
BACKUP = ROOT/'train_rs_v612_ppo.py.pre_ppo_pause_state_v179.bak'

BEFORE_SHA='3bafc769ae917bbf0a7566da53de0ba80d7c7423ccc3105fad7aefdf18a97d33'
AFTER_SHA='6c77bf4c6e69bf8e493095020a4b3d4fda1d7e5b5b61440eea18cb0feccd4a2e'
VISUAL_SHA='a293202c43bbcf71a29551a9bbef42f0d32cd7c120200adc30f46fb0dcb73fea'


def canonical(data):
    if b'\r' in data.replace(b'\r\n', b''):
        raise ValueError('Unrecognized line endings')
    return data.replace(b'\r\n', b'\n')


def sha(data):
    return hashlib.sha256(data).hexdigest()


def prepare():
    if not TRAINER.is_file() or not VISUAL.is_file() or not PAYLOAD.is_file():
        raise ValueError('Missing trainer, Pause visual helper, or packaged trainer payload')
    before=TRAINER.read_bytes()
    normalized=canonical(before)
    if sha(normalized)!=BEFORE_SHA:
        raise ValueError('Trainer is not the exact v1.7.8 source; refusing to modify')
    candidate=PAYLOAD.read_bytes()
    if sha(candidate)!=AFTER_SHA:
        raise ValueError('Packaged trainer payload checksum mismatch')
    if sha(canonical(VISUAL.read_bytes()))!=VISUAL_SHA:
        raise ValueError('Pause-visual helper checksum mismatch')
    if BACKUP.exists():
        raise ValueError('Existing v1.7.9 backup; second install refused')
    compile(candidate,str(TRAINER),'exec')
    compile(VISUAL.read_bytes(),str(VISUAL),'exec')
    result=candidate.replace(b'\n', b'\r\n') if b'\r\n' in before else candidate
    return before,result


def apply():
    original,replacement=prepare()
    with BACKUP.open('xb') as f:
        f.write(original);f.flush();os.fsync(f.fileno())
    handle,temp=tempfile.mkstemp(prefix='.ppo-paused-v179-',suffix='.py',dir=TRAINER.parent)
    try:
        with os.fdopen(handle,'wb') as f:
            f.write(replacement);f.flush();os.fsync(f.fileno())
        os.replace(temp,TRAINER)
    finally:
        if os.path.exists(temp): os.unlink(temp)


def main():
    ap=argparse.ArgumentParser(description=__doc__)
    group=ap.add_mutually_exclusive_group(required=True)
    group.add_argument('--check',action='store_true')
    group.add_argument('--apply',action='store_true')
    args=ap.parse_args()
    try:
        if args.check:
            prepare()
            print('[PPO PAUSED V179] DRY RUN PASS; exact v1.7.8 trainer and helper payload matched. No trainer changes.')
        else:
            apply()
            print('[PPO PAUSED V179] Installed DOM + rendered-Pause exclusion in post-optimizer resync.')
            print('[PPO PAUSED V179] Backup created; original model, rewards and screen classifier untouched.')
    except Exception as exc:
        ap.exit(2,'[PPO PAUSED V179] REFUSED: '+str(exc)+'\n')

if __name__=='__main__': main()
