#!/usr/bin/env python3
"""Add observation-only timing evidence for suspected death during PPO unpause.

Run from CLAUDE development clone (NOT original repo) using -m:
  python -B -m tools.trace_ppo_pause_resume_v153 --check
  python -B -m tools.trace_ppo_pause_resume_v153 --apply

Only touches train_rs_v612_ppo.py (backup made). No action, reward,
classifier threshold, PPO buffer, or restart behavior is changed.
"""
from pathlib import Path
import argparse
import difflib
import os

TARGET = Path('train_rs_v612_ppo.py')
BACKUP = Path('train_rs_v612_ppo.py.pre_ppo_pause_resume_trace_v153.bak')
MARKER = '[PPO RESYNC TRACE]'

RESYNC_OLD = '''                    legacy, menu_score, _ = await self._capture()
                    state, _probs = self._read_screen_state()
                    if state != 'GAMEPLAY' or menu_score >= self.args.threshold * .85:
                        return None, f'unsafe screen state={state} menu_score={menu_score:.3f}'
                    observation, _fresh, _age = self._pack(legacy, start_detection=(i == 0))
'''

RESYNC_NEW = '''                    legacy, menu_score, _ = await self._capture()
                    state, _probs = self._read_screen_state()
                    # Zero additional captures: trace only the four frames
                    # that the existing resync already consumes.
                    resumed_at = getattr(self, '_ppo_resume_trace_start_perf', None)
                    since_resume_ms = ((time.perf_counter() - resumed_at) * 1000.0
                                       if resumed_at is not None else None)
                    if since_resume_ms is not None:
                        print(f'[PPO RESYNC TRACE] frame={i+1}/4 '
                              f'since_resume_ms={since_resume_ms:.1f} '
                              f'state={state} menu_score={menu_score:.3f} '
                              f'ml_gameplay={_probs.get("GAMEPLAY", 0.):.3f} '
                              f'ml_menu={_probs.get("MENU", 0.):.3f}', flush=True)
                    if state != 'GAMEPLAY' or menu_score >= self.args.threshold * .85:
                        # Save precisely the frame that caused the safe stop,
                        # without issuing input or weakening screen-state gating.
                        if since_resume_ms is not None and self._v2_last_bgra is not None:
                            try:
                                import cv2
                                image_path = (self.output_dir /
                                    f'ppo_resume_failure_ep{self.episode_number:04d}_frame{i+1}.jpg')
                                wrote = cv2.imwrite(str(image_path),
                                    cv2.cvtColor(self._v2_last_bgra, cv2.COLOR_BGRA2BGR))
                                if wrote:
                                    print(f'[PPO RESYNC TRACE] failure_screenshot={image_path}', flush=True)
                            except Exception as trace_exc:
                                print(f'[PPO RESYNC TRACE] diagnostic screenshot unavailable: '
                                      f'{trace_exc}', flush=True)
                        return None, (f'unsafe screen state={state} '
                                      f'menu_score={menu_score:.3f} '
                                      f'frame={i+1}/4 '
                                      f'since_resume_ms={since_resume_ms}')
                    observation, _fresh, _age = self._pack(legacy, start_detection=(i == 0))
'''

RESUME_OLD = '''            finished_mono, finished_perf, finished_wall = self._call(resume(), timeout=8)
            dm = finished_mono - token[0]
'''
RESUME_NEW = '''            finished_mono, finished_perf, finished_wall = self._call(resume(), timeout=8)
            # The timestamp is recorded immediately after the confirmed
            # overlay disappearance. This measures how long the game was
            # running before each of the existing four resync captures.
            self._ppo_resume_trace_start_perf = finished_perf
            dm = finished_mono - token[0]
'''

PRE_OLD = '''                if state != 'GAMEPLAY' or menu_score >= self.args.threshold * .85:
                    raise GuardUncertainError(
                        f'Not verified GAMEPLAY at rollout end: {state}, menu={menu_score:.3f}')
                if await pause_button.count() != 1 or not await pause_button.is_visible():
'''
PRE_NEW = '''                if state != 'GAMEPLAY' or menu_score >= self.args.threshold * .85:
                    raise GuardUncertainError(
                        f'Not verified GAMEPLAY at rollout end: {state}, menu={menu_score:.3f}')
                print('[PPO PAUSE TRACE] pre_pause_classifier=GAMEPLAY '
                      f'menu_score={menu_score:.3f} '
                      f'ml_gameplay={probs.get("GAMEPLAY", 0.):.3f}', flush=True)
                if await pause_button.count() != 1 or not await pause_button.is_visible():
'''


def patch(source):
    if MARKER in source:
        raise ValueError('PPO pause/resume trace already installed')
    markers = ('--pause-ppo-optimizer',
               'FreezeWindow(required_pairs=3, max_delta=.25)',
               'MENU template detected during pause',
               'refresh_after_ppo_optimizer()',
               'replace_cached_observation',
               'compensate_pause_clocks(')
    missing = [m for m in markers if m not in source]
    if missing:
        raise ValueError(f'Requires installed v1.5.2 and resync. Missing: {missing}')
    for old in (RESYNC_OLD, RESUME_OLD, PRE_OLD):
        if source.count(old) != 1:
            raise ValueError(f'Expected ONE exact anchor. Found {source.count(old)}: {old[:90]!r}')
    changed = source.replace(RESYNC_OLD, RESYNC_NEW, 1)
    changed = changed.replace(RESUME_OLD, RESUME_NEW, 1)
    changed = changed.replace(PRE_OLD, PRE_NEW, 1)
    compile(changed, str(TARGET), 'exec')
    return changed


def main():
    p = argparse.ArgumentParser(description=__doc__)
    mode = p.add_mutually_exclusive_group(required=True)
    mode.add_argument('--check', action='store_true')
    mode.add_argument('--apply', action='store_true')
    args = p.parse_args()
    raw = TARGET.read_bytes()
    crlf = b'\r\n' in raw
    original = raw.decode('utf-8').replace('\r\n', '\n')
    revised = patch(original)
    if args.check:
        print(''.join(difflib.unified_diff(original.splitlines(True), revised.splitlines(True),
                fromfile='a/train_rs_v612_ppo.py', tofile='b/train_rs_v612_ppo.py')))
        print('DRY RUN ONLY: no files changed.')
        return
    if BACKUP.exists():
        p.error(f'Backup exists; refusing to overwrite: {BACKUP}')
    BACKUP.write_bytes(raw)
    temp = TARGET.with_name(TARGET.name + '.ppo_resume_trace_v153_tmp')
    try:
        temp.write_bytes((revised.replace('\n', '\r\n') if crlf else revised).encode('utf-8'))
        os.replace(temp, TARGET)
    finally:
        if temp.exists():
            temp.unlink()
    print('Installed observation-only PPO pause/resume trace v1.5.3.')
    print('Unchanged: action timing, step count, pause/resume logic, rewards, screen guards, checkpoint.')
    print('Backup:', BACKUP)


if __name__ == '__main__':
    main()
