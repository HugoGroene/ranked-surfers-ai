#!/usr/bin/env python3
"""Source-hashed, opt-in V1.7.4 -> V1.7.5 V6-only distance HUD mask patch.

Usage: python -B -m tools.patch_ppo_guard_hud_v175 --check|--apply
Never changes checkpoint, original V5 classifier, rewards, or input gating.
"""
import argparse
import ast
import hashlib
import os
from pathlib import Path
import tempfile

ROOT = Path(__file__).resolve().parents[1]
TRAINER = ROOT / 'train_rs_v612_ppo.py'
HELPER = ROOT / 'tools' / 'ppo_guard_hud_v175.py'
SOURCE_SHA = '4c65d35adf3f5b7fe14264b222437754cf0d4e08986536b8bc53a2c1ca7b4960'
BACKUP_SUFFIX = '.pre_ppo_guard_hud_v175.bak'
ANCHOR = '        def _guard_diagnostic(self, readings, *, reason, name=None):'
INSERTION = '''        @staticmethod
        def _meter_mask(bgra):
            """Guard-only distance HUD mask; supports visible Chrome or fullscreen."""
            from tools.ppo_guard_hud_v175 import meter_mask
            return meter_mask(bgra)

'''


def sha(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


def prepare(root=ROOT):
    target = root / 'train_rs_v612_ppo.py'
    helper = root / 'tools' / 'ppo_guard_hud_v175.py'
    if not target.is_file() or not helper.is_file():
        raise RuntimeError('Expected trainer and V1.7.5 helper in tools/')
    old = target.read_bytes()
    if sha(old) != SOURCE_SHA:
        raise RuntimeError('Exact V1.7.4 trainer SHA256 mismatch: '
                           f'{sha(old)} (expected {SOURCE_SHA}); no changes made')
    text = old.decode('utf-8')
    if text.count(ANCHOR) != 1:
        raise RuntimeError('Missing exact V1.7.4 guard anchor')
    updated = text.replace(ANCHOR, INSERTION + ANCHOR).encode('utf-8')
    compile(updated, str(target), 'exec')
    compile(helper.read_bytes(), str(helper), 'exec')
    # Verify the only AST change is the new static _meter_mask method.
    original_ast = ast.parse(old)
    updated_ast = ast.parse(updated)
    def get_class(module):
        build = next(n for n in module.body if isinstance(n, ast.FunctionDef)
                     and n.name == 'build_env_class')
        return next(n for n in build.body if isinstance(n, ast.ClassDef)
                    and n.name == 'PPOV61Env')
    original_cls = get_class(original_ast)
    new_cls = get_class(updated_ast)
    diff = [n for n in new_cls.body if isinstance(n, ast.FunctionDef)
            and n.name == '_meter_mask']
    if len(diff) != 1:
        raise RuntimeError('New mask override missing or duplicated')
    new_cls.body.remove(diff[0])
    if ast.dump(original_ast, include_attributes=False) != ast.dump(updated_ast, include_attributes=False):
        raise RuntimeError('Unexpected AST change outside mask override')
    backup = target.with_name(target.name + BACKUP_SUFFIX)
    if backup.exists():
        raise RuntimeError(f'Backup already exists: {backup.name}; no changes made')
    return target, backup, old, updated


def apply(root=ROOT):
    target, backup, old, updated = prepare(root)
    with backup.open('xb') as stream:
        stream.write(old)
    fd, temp = tempfile.mkstemp(prefix=target.name + '.v175.', dir=str(target.parent))
    try:
        with os.fdopen(fd,'wb') as stream:
            stream.write(updated)
            stream.flush()
            os.fsync(stream.fileno())
        os.replace(temp,target)
    finally:
        if os.path.exists(temp):
            os.unlink(temp)
    return backup


def main():
    p=argparse.ArgumentParser(description=__doc__)
    modes=p.add_mutually_exclusive_group(required=True)
    modes.add_argument('--check',action='store_true')
    modes.add_argument('--apply',action='store_true')
    a=p.parse_args()
    try:
        if a.check:
            _, _, old, updated=prepare()
            print('[PPO GUARD HUD V175] DRY RUN PASS: exact V1.7.4 source.')
            print('[PPO GUARD HUD V175] V6-only _meter_mask override added; no other AST changes; '
                  f'{len(updated)-len(old)} bytes')
        else:
            backup=apply()
            print('[PPO GUARD HUD V175] Applied browser/fullscreen HUD support. '
                  f'Backup: {backup.name}')
    except Exception as error:
        p.exit(2,f'[PPO GUARD HUD V175] REFUSED: {error}\n')

if __name__ == '__main__':
    main()
