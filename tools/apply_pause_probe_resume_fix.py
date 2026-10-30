#!/usr/bin/env python3
"""Apply safe, narrowly-scoped pause probe v1.1 selector correction.

Run from repository root as: python -B -m tools.apply_pause_probe_resume_fix --check/--apply
Changes only tools/probe_unity_pause.py. Never touches trainer or checkpoints.
"""
from __future__ import annotations

import argparse
import ast
from pathlib import Path
import shutil
import sys

FILE = Path('tools/probe_unity_pause.py')
OLD = '''            resume_button = page.locator('button[aria-label="Resume game"]')'''
NEW = '''            # The page-level Resume game button is behind the active pause layer.
            # Click the visible Resume action *inside* that layer instead;
            # never use force=True to bypass Playwright's pointer safety checks.
            resume_button = overlay.get_by_role('button', name='Resume', exact=True)'''


def apply_patch(source: str) -> str:
    if OLD not in source:
        if NEW in source:
            raise ValueError('Probe already has the v1.1 resume fix; nothing to apply')
        raise ValueError('Expected unmodified v1 pause-probe Resume selector not found; refusing patch')
    if source.count(OLD) != 1:
        raise ValueError('Resume selector is not unique; refusing patch')
    if "overlay = page.locator('.game-pause-panel[aria-label=\"Game paused\"]')" not in source:
        raise ValueError('The expected pause overlay locator is missing; refusing patch')
    if 'await overlay.wait_for(state=\'hidden\'' not in source:
        raise ValueError('Missing hidden-overlay verification; refusing patch')
    patched = source.replace(OLD, NEW, 1)
    ast.parse(patched, filename=str(FILE))
    return patched


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    group = parser.add_mutually_exclusive_group(required=True)
    group.add_argument('--check', action='store_true', help='Validate installation without changing any file')
    group.add_argument('--apply', action='store_true', help='Back up and patch exactly the pause probe')
    args = parser.parse_args(argv)
    if not FILE.is_file():
        parser.error(f'Missing {FILE}. Extract the original pause-probe v1 package first.')
    original = FILE.read_text(encoding='utf-8')
    try:
        revised = apply_patch(original)
    except ValueError as exc:
        print(f'[PPO PAUSE FIX] Refused: {exc}', file=sys.stderr)
        return 2
    if args.check:
        print('[PPO PAUSE FIX] DRY RUN OK: only one Resume locator will change.')
        print('[PPO PAUSE FIX] Old: page-level Resume game button (blocked by overlay)')
        print('[PPO PAUSE FIX] New: visible Resume button inside Game paused overlay')
        print('[PPO PAUSE FIX] No changes made.')
        return 0
    backup = FILE.with_name(FILE.name + '.pre_resume_fix_v11.bak')
    if backup.exists():
        print(f'[PPO PAUSE FIX] Refusing to overwrite existing backup: {backup}', file=sys.stderr)
        return 2
    shutil.copy2(FILE, backup)
    FILE.write_text(revised, encoding='utf-8')
    print(f'[PPO PAUSE FIX] Applied v1.1. Backup: {backup}')
    print('[PPO PAUSE FIX] The PPO trainer, rewards, guard, and checkpoints are unchanged.')
    return 0


if __name__ == '__main__':
    raise SystemExit(main())
