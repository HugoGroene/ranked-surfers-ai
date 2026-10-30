#!/usr/bin/env python3
"""Source-hashed v1.7.5 -> v1.7.6 overnight safety audit fixes.

Changes only the local V6.1.2 trainer. Requires
 tools/ppo_atomic_checkpoint_v176.py and existing v1.7.5 dependencies.
Read-only --check, opt-in --apply with backed-up source.
"""
from pathlib import Path
import argparse
import ast
import hashlib
import os
import tempfile

ROOT = Path(__file__).resolve().parents[1]
SOURCE_SHA = 'ddd3146dfa5916e8a74f3c672a1353f62b60603aca69d845cac0f918972926fa'
TARGET = ROOT / 'train_rs_v612_ppo.py'
PAYLOAD = Path(__file__).resolve().parent / '_ppo_overnight_v176_payload' / 'trainer.txt'
BACKUP_SUFFIX = '.pre_ppo_overnight_v176.bak'


def sha(blob):
    return hashlib.sha256(blob).hexdigest()


def _scope_limited(old, new):
    """Only permitted methods or train() may change, never PPO/reward logic."""
    allowed = {
        ('PPOV61Env','pause_for_ppo_optimizer'),
        ('PPOV61Env','resume_after_ppo_optimizer'),
        ('V61Callback','__init__'),
        ('V61Callback','_on_rollout_end'),
        ('V61Callback','_on_step'),
        ('','train'),
    }
    def reduce(code):
        tree=ast.parse(code)
        seen=[]
        for node in ast.walk(tree):
            if isinstance(node,ast.FunctionDef) and node.name=='train' and node in tree.body:
                node.body=[ast.Pass()];seen.append(('', 'train'))
            if isinstance(node,ast.ClassDef) and node.name in ('PPOV61Env','V61Callback'):
                for fn in node.body:
                    if isinstance(fn,ast.FunctionDef) and (node.name,fn.name) in allowed:
                        fn.body=[ast.Pass()];seen.append((node.name,fn.name))
        if set(seen)!=allowed or len(seen)!=len(allowed):
            raise RuntimeError(f'Unexpected patch scope: {seen}')
        return ast.dump(tree,include_attributes=False)
    if reduce(old) != reduce(new):
        raise RuntimeError('Modification outside approved trainer methods')


def prepare(root=ROOT, payload=PAYLOAD):
    trainer = root / 'train_rs_v612_ppo.py'
    needed = ('ppo_pause_control_recheck_v171.py', 'ppo_pause_freeze_window.py',
              'ppo_atomic_checkpoint_v176.py')
    for dep in needed:
        if not (root/'tools'/dep).is_file():
            raise RuntimeError(f'Missing required helper: tools/{dep}')
    old = trainer.read_bytes()
    if sha(old)!=SOURCE_SHA:
        raise RuntimeError(f'Trainer source mismatch: {sha(old)}; expected {SOURCE_SHA}; unchanged')
    new=payload.read_bytes()
    if old==new:
        raise RuntimeError('Payload matches old trainer')
    compile(new,str(trainer),'exec')
    compile((root/'tools'/'ppo_atomic_checkpoint_v176.py').read_bytes(),'checkpoint','exec')
    _scope_limited(old,new)
    backup = trainer.with_name(trainer.name + BACKUP_SUFFIX)
    if backup.exists():
        raise RuntimeError(f'Backup already exists: {backup.name}; refusing repeat install')
    return trainer, backup, old, new


def apply(root=ROOT, payload=PAYLOAD):
    trainer,backup,old,new=prepare(root,payload)
    with backup.open('xb') as file:
        file.write(old)
        file.flush()
        os.fsync(file.fileno())
    fd,name=tempfile.mkstemp(prefix='.trainer.v176.', suffix='.py',dir=str(trainer.parent))
    try:
        with os.fdopen(fd,'wb') as file:
            file.write(new)
            file.flush()
            os.fsync(file.fileno())
        os.replace(name,trainer)
    finally:
        if os.path.exists(name):os.unlink(name)
    return backup


def main():
    ap=argparse.ArgumentParser(description=__doc__)
    m=ap.add_mutually_exclusive_group(required=True)
    m.add_argument('--check',action='store_true')
    m.add_argument('--apply',action='store_true')
    a=ap.parse_args()
    try:
        if a.check:
            prepare()
            print('[PPO OVERNIGHT V176] DRY RUN PASS: exact v1.7.5 trainer SHA matched; no changes.')
        else:
            backup=apply()
            print('[PPO OVERNIGHT V176] Installed guarded boundary, Resume and atomic-save fixes.')
            print(f'[PPO OVERNIGHT V176] Trainer backup: {backup.name}; no checkpoints touched.')
    except Exception as exc:
        ap.exit(2, f'[PPO OVERNIGHT V176] REFUSED: {exc}\n')

if __name__=='__main__':main()
