#!/usr/bin/env python3
"""Narrow v1.5.1 fix: wait for a VERIFIED frozen scene, not overlay alone.

From the Claude development clone:
  python -B -m tools.patch_ppo_optimizer_pause_settle_v151 --check
  python -B -m tools.patch_ppo_optimizer_pause_settle_v151 --apply

Requires that opt-in --pause-ppo-optimizer v1.5 is already installed.
"""
import argparse
import difflib
import os
from pathlib import Path

TARGET = Path('train_rs_v612_ppo.py')
BACKUP = Path('train_rs_v612_ppo.py.pre_ppo_pause_settle_v151.bak')

OLD = '''                start_mono = _time.monotonic()
                start_perf = _time.perf_counter()
                start_wall = _time.time()
                # Verifying the freeze while the game is paused does not create
                # an artificial PPO transition or alter the model observation.
                await self._capture()
                if self._v2_last_bgra is None:
                    raise GuardUncertainError('Missing paused game capture')
                first = self._motion_view(self._v2_last_bgra, self.crop)
                await asyncio.sleep(.18)
                await self._capture()
                if self._v2_last_bgra is None:
                    raise GuardUncertainError('Missing second paused game capture')
                second = self._motion_view(self._v2_last_bgra, self.crop)
                if first.shape != second.shape or first.size == 0:
                    raise GuardUncertainError('Invalid paused track images')
                mean_change = float(_np.mean(_np.abs(
                    first.astype(_np.int16) - second.astype(_np.int16))))
                if not _np.isfinite(mean_change) or mean_change > .25:
                    raise GuardUncertainError(
                        f'Game did not freeze after Pause: mean pixel delta={mean_change:.3f}')
                if not await overlay.is_visible():
                    raise GuardUncertainError('Pause overlay vanished while checking freeze')
                return start_mono, start_perf, start_wall, mean_change
'''
NEW = '''                # The React pause overlay can become visible before Unity has
                # consumed the pause message. Unlike v1.5, do not infer a
                # frozen game from the overlay or a single capture pair.
                # Bound the settling interval and NEVER relax the .25 threshold.
                from tools.ppo_pause_freeze_window import FreezeWindow
                window = FreezeWindow(required_pairs=3, max_delta=.25)
                verified = None
                for attempt in range(16):
                    if not await overlay.is_visible():
                        raise GuardUncertainError('Pause overlay vanished while settling Unity')
                    await asyncio.sleep(.09)
                    _legacy, score, _unused = await self._capture()
                    state, _probs = self._read_screen_state()
                    if state != 'GAMEPLAY' or score >= self.args.threshold * .85:
                        raise GuardUncertainError(
                            f'Not verified GAMEPLAY during pause: {state}, template={score:.3f}')
                    if self._v2_last_bgra is None:
                        raise GuardUncertainError('Missing paused Unity capture')
                    track = self._motion_view(self._v2_last_bgra, self.crop)
                    verified = window.observe(
                        track, (_time.monotonic(), _time.perf_counter(), _time.time()))
                    if verified is not None:
                        break
                if verified is None:
                    raise GuardUncertainError(
                        'Unity did not demonstrate three consecutive frozen track '
                        f'intervals within 16 checks; changes={window.changes}')
                if not await overlay.is_visible():
                    raise GuardUncertainError('Pause overlay vanished after freeze verification')
                start_mono, start_perf, start_wall, mean_change = verified
                print(f'[PPO PAUSE] Freeze verified after {attempt + 1} captures; '
                      f'track_deltas={window.changes}', flush=True)
                return start_mono, start_perf, start_wall, mean_change
'''


def patch(source):
    if '--pause-ppo-optimizer' not in source or 'def pause_for_ppo_optimizer(self):' not in source:
        raise ValueError('Missing installed v1.5 optimizer pause')
    if 'FreezeWindow(required_pairs=3' in source:
        raise ValueError('v1.5.1 settling verification already installed')
    if source.count(OLD) != 1:
        raise ValueError(f'Expected one exact v1.5 freeze verifier; found {source.count(OLD)}')
    updated = source.replace(OLD, NEW, 1)
    compile(updated, str(TARGET), 'exec')
    return updated


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    mode=parser.add_mutually_exclusive_group(required=True)
    mode.add_argument('--check', action='store_true')
    mode.add_argument('--apply', action='store_true')
    args=parser.parse_args()
    data=TARGET.read_bytes()
    crlf = b'\r\n' in data
    old=data.decode('utf-8').replace('\r\n','\n')
    updated=patch(old)
    if args.check:
        print(''.join(difflib.unified_diff(old.splitlines(True), updated.splitlines(True),
              fromfile='a/'+str(TARGET), tofile='b/'+str(TARGET))))
        print('DRY RUN ONLY. No source changes.')
        return
    if BACKUP.exists():
        parser.error('Refusing to overwrite existing backup: '+str(BACKUP))
    BACKUP.write_bytes(data)
    tmp=TARGET.with_name(TARGET.name+'.pause_settle_v151_tmp')
    try:
        tmp.write_bytes((updated.replace('\n','\r\n') if crlf else updated).encode('utf-8'))
        os.replace(tmp,TARGET)
    finally:
        if tmp.exists():tmp.unlink()
    print('Installed v1.5.1 optimizer pause-settling verifier. --pause-ppo-optimizer remains OPT-IN.')
    print('Backing up original installed v1.5 trainer at:', BACKUP)
    print('No changes to PPO actions, rewards, classifier, checkpoint or restart policy.')

if __name__=='__main__':
    main()
