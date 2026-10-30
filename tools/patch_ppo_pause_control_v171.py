#!/usr/bin/env python3
"""Strict v1.7 -> v1.7.1 Pause button race fix for local Claude clone.

    python -B -m tools.patch_ppo_pause_control_v171 --check
    python -B -m tools.patch_ppo_pause_control_v171 --apply

Only changes the exact 3-line Pause button eligibility check in the current
v1.7 trainer. Does not alter a screen classifier, reward, restart, or PPO model.
"""
from pathlib import Path
import argparse
import hashlib
import os
import tempfile

ROOT = Path(__file__).resolve().parents[1]
TARGET = ROOT / 'train_rs_v612_ppo.py'
HELPER = ROOT / 'tools' / 'ppo_pause_control_recheck_v171.py'
EXPECTED_V17_SHA256 = '9eb5c01a8dbb8d913a384b297219ba9c1fad2e156a02802844cf71b2184d3666'
BACKUP_SUFFIX = '.pre_ppo_pause_control_v171.bak'

OLD = b'''                if await pause_button.count() != 1 or not await pause_button.is_visible():
                    raise GuardUncertainError('Unique visible site Pause game button missing')
                await pause_button.click(timeout=1800)
'''

NEW = b'''                # React can remove the Pause control AFTER the pre-pause
                # GAMEPLAY capture (e.g. on an emerging death transition).
                # Recover only with fresh, independently guarded observations.
                if await pause_button.count() != 1 or not await pause_button.is_visible():
                    from tools.ppo_pause_control_recheck_v171 import (
                        PauseControlRecheck, MAX_RECHECK_CAPTURES)
                    window = PauseControlRecheck(
                        confidence=self.state_confidence,
                        threshold=self.args.threshold,
                        motion_threshold=self.startup_motion_threshold)
                    for recheck_index in range(MAX_RECHECK_CAPTURES):
                        if recheck_index:
                            await asyncio.sleep(.075)
                        if await overlay.is_visible():
                            raise GuardUncertainError(
                                'Unexpected Pause overlay during missing-control recheck')
                        focused = bool(await page.evaluate(
                            'document.hasFocus() && document.visibilityState === "visible"'))
                        count = await pause_button.count()
                        if count > 1:
                            raise GuardUncertainError(
                                'Ambiguous multiple site Pause game buttons; NO click')
                        visible = count == 1 and await pause_button.is_visible()
                        _legacy, score, _unused = await self._capture()
                        new_state, new_probs = self._read_screen_state()
                        decision = window.observe(
                            state=new_state, probabilities=new_probs,
                            menu_score=score, observed_mono=_time.monotonic(),
                            focused=focused, button_count=count,
                            button_visible=visible)
                        if decision == 'MENU':
                            # Anchor survival at the FIRST corroborated MENU
                            # observation, not the last confirmation; no menu
                            # waiting time is credited to the terminal reward.
                            first_menu = window.first_menu_observed_mono
                            if first_menu is None:
                                raise GuardUncertainError(
                                    'Corroborated MENU missing observation-time anchor')
                            self._ppo_boundary_menu_duration_at = max(
                                float(self.survival_rewards.last_duration),
                                first_menu - float(self.episode_start))
                            print('[PPO PAUSE CONTROL] Missing button; MENU '
                                  f'corroborated after {len(window.readings)} frames; '
                                  'no Pause click, no movement, no restart.', flush=True)
                            # The existing v1.7 callback repeats independent
                            # confirmation and defers a REAL terminal reward.
                            raise GuardUncertainError(
                                f'Not verified GAMEPLAY at rollout end: MENU, '
                                f'menu={score:.3f}')
                        if decision == 'READY':
                            print('[PPO PAUSE CONTROL] Pause button reappeared '
                                  f'with two positive GAMEPLAY frames after '
                                  f'{len(window.readings)} checks.', flush=True)
                            break
                        if decision == 'ABORT':
                            break
                    else:
                        decision = 'ABORT'
                    if decision != 'READY':
                        # Preserve the latest screenshot for inspection without
                        # trusting it for an R or any movement command.
                        import cv2
                        diagnostic = (self.output_dir /
                            f'ppo_pause_control_ep{self.episode_number:04d}.jpg')
                        if self._v2_last_bgra is not None:
                            cv2.imwrite(str(diagnostic), cv2.cvtColor(
                                self._v2_last_bgra, cv2.COLOR_BGRA2BGR))
                        raise GuardUncertainError(
                            'Pause control unavailable/ambiguous after bounded '
                            f'guarded checks; {window.diagnostic()}; '
                            f'diagnostic={diagnostic}; NO inputs sent')
                await pause_button.click(timeout=1800)
'''


def digest(data):
    return hashlib.sha256(data).hexdigest()


def prepare(root=ROOT):
    target = root / 'train_rs_v612_ppo.py'
    helper = root / 'tools' / 'ppo_pause_control_recheck_v171.py'
    if not target.is_file() or not helper.is_file():
        raise RuntimeError('Trainer or v1.7.1 helper missing')
    old = target.read_bytes()
    sha = digest(old)
    if sha != EXPECTED_V17_SHA256:
        raise RuntimeError(f'Unexpected trainer source hash {sha}; require exact installed v1.7. No changes made.')
    if old.count(OLD) != 1:
        raise RuntimeError(f'Expected exactly one Pause button anchor, found {old.count(OLD)}')
    updated = old.replace(OLD, NEW)
    compile(updated.decode('utf-8'), str(target), 'exec')
    compile(helper.read_text(encoding='utf-8'), str(helper), 'exec')
    backup = target.with_name(target.name + BACKUP_SUFFIX)
    if backup.exists():
        raise RuntimeError(f'Backup already exists: {backup}')
    return target, backup, old, updated


def apply(root=ROOT):
    target, backup, old, updated = prepare(root)
    backup.write_bytes(old)
    fd, temp = tempfile.mkstemp(prefix=target.name+'.v171.', dir=str(target.parent))
    try:
        with os.fdopen(fd, 'wb') as output:
            output.write(updated)
            output.flush()
            os.fsync(output.fileno())
        os.replace(temp, target)
    finally:
        if os.path.exists(temp):
            os.unlink(temp)
    return backup


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    mode = parser.add_mutually_exclusive_group(required=True)
    mode.add_argument('--check', action='store_true')
    mode.add_argument('--apply', action='store_true')
    args = parser.parse_args()
    try:
        if args.check:
            target, backup, old, updated = prepare()
            print('[PPO PAUSE CONTROL] DRY RUN OK: exact v1.7 trainer hash matched.')
            print(f'[PPO PAUSE CONTROL] Changes: 1 Pause button guard ({len(updated)-len(old)} bytes added).')
            print('[PPO PAUSE CONTROL] No file changed; classifier, rewards and model untouched.')
        else:
            backup = apply()
            print('[PPO PAUSE CONTROL] Applied v1.7.1 missing Pause-button recheck.')
            print(f'[PPO PAUSE CONTROL] Backup: {backup.name}')
            print('[PPO PAUSE CONTROL] Pause/ML guards retained; default flags unchanged.')
    except Exception as exc:
        parser.exit(2, f'[PPO PAUSE CONTROL] REFUSED: {exc}\n')


if __name__ == '__main__':
    main()
