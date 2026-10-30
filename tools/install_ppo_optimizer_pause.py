#!/usr/bin/env python3
"""Install experimental, opt-in PPO gradient-update pause with safe timing.

Run from Claude development clone (NOT the original repository):
 python -B -m tools.install_ppo_optimizer_pause --check
 python -B -m tools.install_ppo_optimizer_pause --apply

Requires timing audit, resync hardening, menu recovery v1.4 installed.
Exact single-occurrence anchors, AST compilation, separate backup, atomic write.
"""
import argparse
import difflib
import os
from pathlib import Path

TARGET = Path('train_rs_v612_ppo.py')
BACKUP = Path('train_rs_v612_ppo.py.pre_ppo_optimizer_pause_v15.bak')
MARKER = 'def pause_for_ppo_optimizer(self):'


METHODS = '''        def pause_for_ppo_optimizer(self):
            """Freeze the *local* Practice game before PPO.train(), without keys.

            UI must report GAMEPLAY; verify the site pause overlay and that the
            captured track remains static. Never send Escape (fullscreen risk).
            The old cached PPO observation stays untouched until rollout-start
            refresh, after the game resumes.
            """
            import asyncio
            import time as _time
            import numpy as _np

            async def pause():
                if self._guard_hard_stopped or getattr(self, '_ppo_pause_pending', None):
                    raise GuardUncertainError('Guard stopped or pause already pending')
                if self.suspected_since is not None:
                    raise GuardUncertainError('Unresolved menu suspicion at rollout end')
                page = self.page
                if not bool(await page.evaluate(
                        'document.hasFocus() && document.visibilityState === "visible"')):
                    raise GuardUncertainError('Browser not focused/visible before pause')
                overlay = page.locator('.game-pause-panel[aria-label="Game paused"]')
                pause_button = page.locator('button[aria-label="Pause game"]')
                if await overlay.is_visible():
                    raise GuardUncertainError('Game unexpectedly already paused')
                _legacy, menu_score, _ = await self._capture()
                state, probs = self._read_screen_state()
                if state != 'GAMEPLAY' or menu_score >= self.args.threshold * .85:
                    raise GuardUncertainError(
                        f'Not verified GAMEPLAY at rollout end: {state}, menu={menu_score:.3f}')
                if await pause_button.count() != 1 or not await pause_button.is_visible():
                    raise GuardUncertainError('Unique visible site Pause game button missing')
                await pause_button.click(timeout=1800)
                await overlay.wait_for(state='visible', timeout=2500)
                start_mono = _time.monotonic()
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

            # Never silently proceed if the pause fails. Trainer catches the
            # exception, saves partial model, and closes the Practice browser.
            paused_mono, paused_perf, paused_wall, delta = self._call(pause(), timeout=15)
            self._ppo_pause_pending = (paused_mono, paused_perf, paused_wall,
                                       paused_mono - float(self.episode_start))
            return delta

        def resume_after_ppo_optimizer(self, monitor):
            """Resume, correct clocks, then let existing v1.4 resync the image."""
            import time as _time
            from tools.ppo_optimizer_pause_clock import compensate_pause_clocks

            token = getattr(self, '_ppo_pause_pending', None)
            if not (isinstance(token, tuple) and len(token) == 4):
                raise GuardUncertainError('No validated pause token at rollout start')

            # Validate the exact SB3 Monitor BEFORE unpausing.
            if getattr(monitor, 'env', None) is not self:
                raise GuardUncertainError('Expected exact Monitor(raw) wrapper')

            async def resume():
                page = self.page
                if not bool(await page.evaluate(
                        'document.hasFocus() && document.visibilityState === "visible"')):
                    raise GuardUncertainError('Browser not focused/visible before resume')
                overlay = page.locator('.game-pause-panel[aria-label="Game paused"]')
                button = overlay.get_by_role('button', name='Resume', exact=True)
                if not await overlay.is_visible() or not await button.is_visible():
                    raise GuardUncertainError('Verified pause overlay/Resume button missing')
                await button.click(timeout=1800)
                await overlay.wait_for(state='hidden', timeout=2500)
                return _time.monotonic(), _time.perf_counter(), _time.time()

            finished_mono, finished_perf, finished_wall = self._call(resume(), timeout=8)
            dm = finished_mono - token[0]
            dp = finished_perf - token[1]
            # Adjust wall-time origins, but never fabricate a step/reward or
            # relax the screen-state classifier. If anything goes wrong,
            # fail closed BEFORE the next PPO action.
            report = compensate_pause_clocks(
                self, dm, dp, monitor=monitor,
                elapsed_wall=finished_wall - token[2])
            corrected_duration = _time.monotonic() - float(self.episode_start)
            duration_drift = corrected_duration - token[3]
            if not (-.10 <= duration_drift <= .20):
                raise GuardUncertainError(
                    f'Pause-aware episode timer drifted {duration_drift:.3f}s')
            report['active_episode_clock_drift_s'] = round(duration_drift, 3)
            self._ppo_pause_pending = None
            return report

'''

# Current callback from timing audit v1; exact anchors are a deliberate safety
# feature. If a local trainer differs, the installer will refuse to change it.
END_ANCHOR = '''            except Exception as exc:
                print(f"[PPO GAP] rollout-end diagnostic failed: {exc}", flush=True)

        def _on_rollout_start(self):
'''
END_NEW = '''            except Exception as exc:
                print(f"[PPO GAP] rollout-end diagnostic failed: {exc}", flush=True)
            if a.pause_ppo_optimizer:
                try:
                    if raw is None:
                        raise RuntimeError('No raw environment for PPO pause')
                    delta = raw.pause_for_ppo_optimizer()
                    print(f'[PPO PAUSE] Verified frozen gameplay before optimizer; '
                          f'paused-track delta={delta:.3f}', flush=True)
                except Exception as exc:
                    raise GuardUncertainError(f'[PPO PAUSE] Safe stop at rollout-end: {exc}') from exc

        def _on_rollout_start(self):
'''
START_ANCHOR = '''            except Exception as exc:
                print(f"[PPO GAP] rollout-start diagnostic failed: {exc}", flush=True)
            # Run after the optimizer, before SB3 predicts its first action.
'''
START_NEW = '''            except Exception as exc:
                print(f"[PPO GAP] rollout-start diagnostic failed: {exc}", flush=True)
            # Resume BEFORE the existing fresh-stack capture and PPO prediction.
            if a.pause_ppo_optimizer and record is not None:
                try:
                    if raw is None:
                        raise RuntimeError('No raw environment for PPO resume')
                    vec = self.model.env
                    if self.model.n_envs != 1 or not hasattr(vec, 'envs') or len(vec.envs) != 1:
                        raise RuntimeError('Pause clock correction requires one DummyVecEnv')
                    monitor = vec.envs[0]
                    if getattr(monitor, 'env', None) is not raw:
                        raise RuntimeError('Pause requires Monitor(raw) without other wrappers')
                    clock_report = raw.resume_after_ppo_optimizer(monitor)
                    print('[PPO PAUSE] Resumed after optimizer with pause-aware clocks: '
                          + json.dumps(clock_report), flush=True)
                except Exception as exc:
                    raise GuardUncertainError(f'[PPO PAUSE] Safe stop at rollout-start: {exc}') from exc
            # Run after the optimizer, before SB3 predicts its first action.
'''
FLAG_ANCHOR = '''    p.add_argument('--resync-rollouts', action='store_true',
                   help='EXPERIMENTAL: passively rebuild 4-frame observation after PPO updates')
'''
FLAG_NEW = FLAG_ANCHOR + '''    p.add_argument('--pause-ppo-optimizer', action='store_true',
                   help='EXPERIMENTAL: freeze local Practice during PPO gradient updates, correct wall clocks; requires --resync-rollouts')
'''
ARGS_ANCHOR = '''    a=p.parse_args()
    a.crop=bench.validate_crop(a.crop)
'''
ARGS_NEW = '''    a=p.parse_args()
    if a.pause_ppo_optimizer and (a.mode != 'train' or not a.resync_rollouts):
        p.error('--pause-ppo-optimizer requires --mode train --resync-rollouts')
    a.crop=bench.validate_crop(a.crop)
'''


def once(source, original, changed):
    count = source.count(original)
    if count != 1:
        raise ValueError(f'Expected exactly one known patch anchor; found {count}: {original[:100]!r}')
    return source.replace(original, changed, 1)


def patch(source):
    if MARKER in source or '--pause-ppo-optimizer' in source:
        raise ValueError('Experimental optimizer pause already installed')
    for marker in ('def refresh_after_ppo_optimizer(self):',
                   'def confirm_post_optimizer_menu(self):',
                   'from tools.ppo_menu_guard_alignment_v14 import (',
                   'from tools.ppo_rollout_gap_audit import RolloutGapTracker'):
        if marker not in source:
            raise ValueError('Missing installed v1.4 resync/menu/timing requirement: '+marker)
    s = once(source,
             '        def _guard_diagnostic(self, readings, *, reason, name=None):\n',
             METHODS + '        def _guard_diagnostic(self, readings, *, reason, name=None):\n')
    s = once(s, END_ANCHOR, END_NEW)
    s = once(s, START_ANCHOR, START_NEW)
    s = once(s, '        def _on_rollout_start(self):\n'
             '            # This runs after the preceding PPO gradient update.',
             '        def _on_rollout_start(self):\n'
             '            record = None\n'
             '            # This runs after the preceding PPO gradient update.')
    s = once(s, FLAG_ANCHOR, FLAG_NEW)
    s = once(s, ARGS_ANCHOR, ARGS_NEW)
    compile(s, str(TARGET), 'exec')
    return s


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    flags = parser.add_mutually_exclusive_group(required=True)
    flags.add_argument('--check', action='store_true')
    flags.add_argument('--apply', action='store_true')
    args = parser.parse_args()
    old_bytes = TARGET.read_bytes()
    use_crlf = b'\r\n' in old_bytes
    old = old_bytes.decode('utf-8').replace('\r\n', '\n')
    new = patch(old)
    if args.check:
        print(''.join(difflib.unified_diff(old.splitlines(True), new.splitlines(True),
              fromfile='a/'+str(TARGET), tofile='b/'+str(TARGET))))
        print('DRY RUN ONLY — no changes made')
        return
    if BACKUP.exists():
        parser.error('Backup already exists; not overwriting: '+str(BACKUP))
    BACKUP.write_bytes(old_bytes)
    temp = TARGET.with_name(TARGET.name+'.ppo_optimizer_pause_tmp')
    try:
        temp.write_bytes((new.replace('\n','\r\n') if use_crlf else new).encode('utf-8'))
        os.replace(temp, TARGET)
    finally:
        if temp.exists():
            temp.unlink()
    print('Installed OPT-IN --pause-ppo-optimizer (default OFF).')
    print('Existing classifier, guarded restarts, rewards and model weights unchanged.')
    print('Backup:', BACKUP)


if __name__ == '__main__':
    main()
