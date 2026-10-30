#!/usr/bin/env python3
"""Fix v1.5.1 paused-overlay GAMEPLAY-classifier false stop, fail closed.

Use as module from Claude development repository:
  python -B -m tools.patch_ppo_optimizer_pause_overlay_v152 --check
  python -B -m tools.patch_ppo_optimizer_pause_overlay_v152 --apply

Does NOT change pre-pause GAMEPLAY verification or the existing post-resume
classifier-based fresh-observation guard.
"""
import argparse
import difflib
import os
from pathlib import Path

TARGET = Path('train_rs_v612_ppo.py')
BACKUP = Path('train_rs_v612_ppo.py.pre_ppo_pause_overlay_v152.bak')

OLD = '''                    state, _probs = self._read_screen_state()
                    if state != 'GAMEPLAY' or score >= self.args.threshold * .85:
                        raise GuardUncertainError(
                            f'Not verified GAMEPLAY during pause: {state}, template={score:.3f}')
'''

NEW = '''                    # The site's pause overlay changes the screenshot that the
                    # screen classifier sees.  In the live probe a verified
                    # frozen GAMEPLAY track classified as UNCERTAIN, so an ML
                    # GAMEPLAY vote here is not meaningful.  GAMEPLAY must
                    # still be confirmed BEFORE clicking Pause (above).  While
                    # paused, require the visible overlay, a frozen *track*,
                    # and no positive independent menu-template evidence.
                    # After resume, the existing four-frame resync rechecks
                    # unoccluded GAMEPLAY before the first PPO action.
                    if score >= self.args.threshold * .85:
                        raise GuardUncertainError(
                            f'MENU template detected during pause: {score:.3f}')
'''


def patch(source):
    if '--pause-ppo-optimizer' not in source or 'FreezeWindow(required_pairs=3, max_delta=.25)' not in source:
        raise ValueError('Requires installed v1.5.1 optimizer pause with settling verifier')
    if source.count(OLD) != 1:
        raise ValueError(f'Expected exactly one v1.5.1 paused classifier check, found {source.count(OLD)}')
    updated = source.replace(OLD, NEW, 1)
    # Do NOT weaken independent GAMEPLAY gate before Pause or change resume safety.
    method = updated.split('def pause_for_ppo_optimizer(self):', 1)[1].split('def resume_after_ppo_optimizer(self, monitor):', 1)[0]
    if ("if state != 'GAMEPLAY' or menu_score >= self.args.threshold * .85:" not in method
        or 'await pause_button.click(timeout=1800)' not in method
        or 'if not await overlay.is_visible():' not in method
        or 'verified = window.observe(' not in method
        or 'if verified is None:' not in method):
        raise ValueError('Required pre-pause GAMEPLAY or frozen-scene checks missing')
    if 'refresh_after_ppo_optimizer()' not in updated or 'replace_cached_observation' not in updated:
        raise ValueError('Required post-optimizer observation refresh missing')
    compile(updated, str(TARGET), 'exec')
    return updated


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    mode = parser.add_mutually_exclusive_group(required=True)
    mode.add_argument('--check', action='store_true')
    mode.add_argument('--apply', action='store_true')
    args = parser.parse_args()
    data = TARGET.read_bytes()
    crlf = b'\r\n' in data
    old = data.decode('utf-8').replace('\r\n', '\n')
    updated = patch(old)
    if args.check:
        print(''.join(difflib.unified_diff(old.splitlines(True), updated.splitlines(True),
                                          fromfile='a/'+str(TARGET), tofile='b/'+str(TARGET))))
        print('DRY RUN ONLY. No source changes.')
        return
    if BACKUP.exists():
        parser.error('Backup already exists: '+str(BACKUP))
    BACKUP.write_bytes(data)
    tmp = TARGET.with_name(TARGET.name + '.pause_overlay_v152_tmp')
    try:
        tmp.write_bytes((updated.replace('\n', '\r\n') if crlf else updated).encode('utf-8'))
        os.replace(tmp, TARGET)
    finally:
        if tmp.exists():
            tmp.unlink()
    print('Installed v1.5.2: pause-overlay classifier false stop fixed.')
    print('Pre-pause GAMEPLAY gate, four-frame resync, and action guard unchanged.')
    print('Backup:', BACKUP)
    print('--pause-ppo-optimizer remains OPT-IN, OFF by default.')


if __name__ == '__main__':
    main()
