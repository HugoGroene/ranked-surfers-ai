#!/usr/bin/env python3
"""Opt-in v1.4 patch for post-optimizer MENU verification.

Install AFTER v1.3, in the Claude dev clone. Invoke as a module:
  python -B -m tools.extend_ppo_menu_recovery_window --check
  python -B -m tools.extend_ppo_menu_recovery_window --apply

Changes only confirm_post_optimizer_menu() in train_rs_v612_ppo.py. Does not
modify PPO, action logic, rewards, guard thresholds or original checkpoints.
"""
import argparse
import difflib
import os
from pathlib import Path

from tools.align_ppo_menu_recovery_guard import NEW_METHOD as V13_METHOD

TARGET = Path('train_rs_v612_ppo.py')
BACKUP = Path('train_rs_v612_ppo.py.pre_menu_window_v14.bak')

V14_METHOD = '''        def confirm_post_optimizer_menu(self):
            """Passively verify a death after PPO updates using four recent MENU frames.

            Allow up to eight fresh captures to outlast a brief early animation dip.
            NEVER lower any guard threshold or restart based on ML alone.
            No keys are sent here; a separate inherited safe reset handles R.
            """
            import asyncio
            from rs_v612_gameplay_guard import screen_gate
            from tools.ppo_menu_guard_alignment_v14 import (
                accept_menu_after_transient, MAX_CONFIRM_CAPTURES)

            async def confirm():
                if self._guard_hard_stopped:
                    return False, 'gameplay guard already hard-stopped'
                readings = []
                for i in range(MAX_CONFIRM_CAPTURES):
                    if i:
                        await asyncio.sleep(.075)
                    # Focus can change while waiting. Do not authorize a reset
                    # using only the initial focus state.
                    try:
                        focused = bool(await self.page.evaluate(
                            'document.hasFocus() && document.visibilityState === "visible"'))
                    except Exception as exc:
                        return False, f'cannot verify browser focus: {exc}'
                    if not focused:
                        return False, 'browser not focused/visible'
                    if self._guard_hard_stopped:
                        return False, 'gameplay guard hard-stopped during confirmation'
                    _legacy, menu_score, _unused = await self._capture()
                    state, probs = self._read_screen_state()
                    reading = screen_gate(
                        state, probs, menu_score,
                        confidence=self.state_confidence,
                        menu_threshold=float(self.args.threshold),
                        motion_threshold=self.startup_motion_threshold)
                    readings.append(reading)
                    # Requires 4 most recent independent captures to corroborate
                    # MENU. A weak or gameplay capture breaks the streak.
                    if accept_menu_after_transient(readings, focused=focused):
                        confirmed = True
                        break
                else:
                    confirmed = False
                evidence = {
                    'checks': [
                        {'state': r.state,
                         'menu_probability': round(r.menu_probability, 4),
                         'menu_score': round(r.menu_score, 5),
                         'corroborated': bool(r.ml_template_corroborated)}
                        for r in readings],
                    'required': '4 consecutive confident ML MENU + >=80% template corroboration',
                    'max_checks': MAX_CONFIRM_CAPTURES,
                }
                return confirmed, evidence

            return self._call(confirm(), timeout=15)

'''


def patch(source):
    if 'from tools.ppo_menu_guard_alignment_v14 import (' in source:
        raise ValueError('v1.4 rolling MENU confirmation already installed')
    if source.count(V13_METHOD) != 1:
        raise ValueError('Expected exactly one unmodified v1.3 MENU confirmer; refusing to edit unknown source')
    if '--resync-rollouts' not in source or '--recover-rollout-menu' not in source:
        raise ValueError('Missing required opt-in PPO resync/recovery flags')
    updated = source.replace(V13_METHOD, V14_METHOD, 1)
    compile(updated, str(TARGET), 'exec')
    return updated


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    group = parser.add_mutually_exclusive_group(required=True)
    group.add_argument('--check', action='store_true')
    group.add_argument('--apply', action='store_true')
    opts = parser.parse_args()
    raw = TARGET.read_bytes()
    crlf = b'\r\n' in raw
    original = raw.decode('utf-8').replace('\r\n', '\n')
    updated = patch(original)
    if opts.check:
        print(''.join(difflib.unified_diff(
            original.splitlines(True), updated.splitlines(True),
            fromfile='a/'+str(TARGET), tofile='b/'+str(TARGET))))
        print('DRY RUN ONLY; no files modified')
        return
    if BACKUP.exists():
        parser.error(f'Backup exists; refusing to overwrite: {BACKUP}')
    BACKUP.write_bytes(raw)
    tmp = TARGET.with_name(TARGET.name+'.v14_menu_tmp')
    try:
        tmp.write_bytes((updated.replace('\n', '\r\n') if crlf else updated).encode('utf-8'))
        os.replace(tmp, TARGET)
    finally:
        if tmp.exists():
            tmp.unlink()
    print('Installed v1.4: up to 8 MENU captures, 4 consecutive corroborated required.')
    print('No thresholds, rewards, checkpoints or policy actions changed.')
    print('Original saved at', BACKUP)


if __name__ == '__main__':
    main()
