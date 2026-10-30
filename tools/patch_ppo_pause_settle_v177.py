#!/usr/bin/env python3
"""Source-locked v1.7.6 -> v1.7.7 Pause overlay settle race patch.

Only PPOV61Env.pause_for_ppo_optimizer changes. Leaves weights, training,
rewards, detectors, audit and checkpoint logic untouched.
"""
import argparse
import ast
import hashlib
import os
from pathlib import Path
import tempfile

ROOT=Path(__file__).resolve().parents[1]
SOURCE_SHA='c8e8a4f4774c4081219c7cfb48772d5c36412e2f20c691cbe842bba44c4640fe'
TARGET=ROOT/'train_rs_v612_ppo.py'
PAYLOAD=ROOT/'tools'/'_ppo_pause_settle_v177_payload'/'trainer.txt'
BACKUP_SUFFIX='.pre_ppo_pause_settle_v177.bak'

def sha(blob):return hashlib.sha256(blob).hexdigest()

def _only_pause_changed(before,after):
    def strip_method(src):
        tree=ast.parse(src)
        owner=next(n for n in ast.walk(tree) if isinstance(n,ast.ClassDef) and n.name=='PPOV61Env')
        matching=[n for n in owner.body if isinstance(n,ast.FunctionDef) and n.name=='pause_for_ppo_optimizer']
        if len(matching)!=1:raise RuntimeError('Unexpected pause method count')
        matching[0].body=[ast.Pass()]
        return ast.dump(tree,include_attributes=False)
    if strip_method(before)!=strip_method(after):
        raise RuntimeError('Patch modifies code outside pause_for_ppo_optimizer')

def prepare(root=ROOT,payload=PAYLOAD):
    trainer=root/'train_rs_v612_ppo.py'
    for relative in ('tools/ppo_pause_control_recheck_v171.py',
                     'tools/ppo_pause_freeze_window.py',
                     'tools/ppo_atomic_checkpoint_v176.py'):
        if not (root/relative).is_file():
            raise RuntimeError('Required runtime module missing: '+relative)
    original=trainer.read_bytes()
    digest=sha(original)
    if digest!=SOURCE_SHA:
        raise RuntimeError(f'Unexpected trainer SHA256 {digest}; requires {SOURCE_SHA}, no changes')
    replacement=payload.read_bytes()
    if replacement==original:
        raise RuntimeError('Replacement and original are identical')
    compile(replacement,str(trainer),'exec')
    _only_pause_changed(original,replacement)
    backup=trainer.with_name(trainer.name+BACKUP_SUFFIX)
    if backup.exists():raise RuntimeError(f'Backup already exists: {backup}; refusing duplicate install')
    return trainer,backup,original,replacement

def apply(root=ROOT,payload=PAYLOAD):
    target,backup,original,replacement=prepare(root,payload)
    with backup.open('xb') as out:
        out.write(original);out.flush();os.fsync(out.fileno())
    fd,tmp=tempfile.mkstemp(prefix='.pause-settle-v177-',suffix='.py',dir=str(target.parent))
    try:
        with os.fdopen(fd,'wb') as out:
            out.write(replacement);out.flush();os.fsync(out.fileno())
        os.replace(tmp,target)
    finally:
        if os.path.exists(tmp):os.unlink(tmp)
    return backup

def main():
    p=argparse.ArgumentParser(description=__doc__)
    mode=p.add_mutually_exclusive_group(required=True)
    mode.add_argument('--check',action='store_true')
    mode.add_argument('--apply',action='store_true')
    a=p.parse_args()
    try:
        if a.check:
            prepare()
            print('[PPO PAUSE SETTLE V177] DRY RUN PASS: exact v1.7.6 trainer matched, no changes.')
        else:
            saved=apply()
            print('[PPO PAUSE SETTLE V177] Installed bounded overlay-disappearance recovery.')
            print(f'[PPO PAUSE SETTLE V177] Backup: {saved.name}; models/checkpoints unchanged.')
    except Exception as exc:
        p.exit(2,f'[PPO PAUSE SETTLE V177] REFUSED: {exc}\n')

if __name__=='__main__':main()
