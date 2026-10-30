#!/usr/bin/env python3
"""Align post-optimizer MENU recovery with the *existing* V6.1 gameplay guard.

Run only in the Claude development clone after v1.2 recovery was installed.
--check is read-only. --apply makes a distinct backup, never modifies weights.
"""
import argparse
import difflib
import os
from pathlib import Path

from tools.install_ppo_optimizer_menu_recovery import METHOD as OLD_METHOD

TARGET = Path('train_rs_v612_ppo.py')
BACKUP = Path('train_rs_v612_ppo.py.pre_menu_guard_alignment.bak')

NEW_METHOD = '''        def confirm_post_optimizer_menu(self):
            """Confirm death with the SAME conservative ML/template guard as normal play.

            Four independent passive captures; no keypress until the inherited
            safe reset path runs. ML-only MENU is not enough for recovery.
            """
            import asyncio
            from rs_v612_gameplay_guard import screen_gate
            from tools.ppo_menu_guard_alignment import accept_corroborated_menu

            async def confirm():
                if self._guard_hard_stopped:
                    return False, 'gameplay guard already hard-stopped'
                try:
                    focused = bool(await self.page.evaluate(
                        'document.hasFocus() && document.visibilityState === "visible"'))
                except Exception as exc:
                    return False, f'cannot check browser focus: {exc}'
                if not focused:
                    return False, 'browser not focused/visible'
                readings = []
                for i in range(4):
                    if i:
                        await asyncio.sleep(.075)
                    _legacy, menu_score, _unused = await self._capture()
                    state, probs = self._read_screen_state()
                    readings.append(screen_gate(
                        state, probs, menu_score,
                        confidence=self.state_confidence,
                        menu_threshold=float(self.args.threshold),
                        motion_threshold=self.startup_motion_threshold))
                confirmed = accept_corroborated_menu(readings, focused=focused)
                evidence = {
                    'checks': [
                        {'state': r.state,
                         'menu_probability': round(r.menu_probability, 4),
                         'menu_score': round(r.menu_score, 5),
                         'corroborated': bool(r.ml_template_corroborated)}
                        for r in readings],
                    'required': '4 consecutive confident ML MENU + >=80% template corroboration',
                }
                return confirmed, evidence

            return self._call(confirm(), timeout=12)

'''


def patch(source):
    if 'from tools.ppo_menu_guard_alignment import accept_corroborated_menu' in source:
        raise ValueError('PPO MENU guard alignment already installed')
    if '--recover-rollout-menu' not in source or '--resync-rollouts' not in source:
        raise ValueError('Install the resync and MENU recovery v1.2 patches first')
    if source.count(OLD_METHOD) != 1:
        raise ValueError(f'Expected the unchanged v1.2 MENU confirmer exactly once (found {source.count(OLD_METHOD)}); no changes made')
    updated = source.replace(OLD_METHOD, NEW_METHOD, 1)
    compile(updated, str(TARGET), 'exec')
    return updated


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    mode = parser.add_mutually_exclusive_group(required=True)
    mode.add_argument('--check', action='store_true')
    mode.add_argument('--apply', action='store_true')
    a = parser.parse_args()
    original = TARGET.read_bytes()
    crlf = b'\r\n' in original
    s = original.decode('utf-8').replace('\r\n', '\n')
    new = patch(s)
    if a.check:
        print(''.join(difflib.unified_diff(s.splitlines(True), new.splitlines(True),
                     fromfile='a/'+str(TARGET), tofile='b/'+str(TARGET))))
        print('DRY RUN ONLY; no files changed')
        return
    if BACKUP.exists():
        parser.error(f'Backup already exists; refusing to overwrite: {BACKUP}')
    BACKUP.write_bytes(original)
    tmp = TARGET.with_name(TARGET.name + '.guard_alignment_tmp')
    try:
        tmp.write_bytes((new.replace('\n', '\r\n') if crlf else new).encode('utf-8'))
        os.replace(tmp, TARGET)
    finally:
        if tmp.exists():
            tmp.unlink()
    print('Installed opt-in MENU guard alignment. ML-only MENU still cannot restart the game.')
    print('Original saved to', BACKUP)


if __name__ == '__main__':
    main()
