#!/usr/bin/env python3
"""Add opt-in PPO rollout-start *passive* resync atop timing audit v1.

  python tools/apply_ppo_rollout_resync.py --check  # no write
  python tools/apply_ppo_rollout_resync.py --apply  # backs up then edits

Run in the CLAUDE clone only. Does NOT touch the original repository.
"""
import argparse
import difflib
import os
from pathlib import Path

TARGET = Path('train_rs_v612_ppo.py')
BACKUP = Path('train_rs_v612_ppo.py.pre_ppo_resync.bak')
MARKER = 'def refresh_after_ppo_optimizer(self):'


def once(text, old, new):
    count = text.count(old)
    if count != 1:
        raise ValueError(f'Expected exactly one patch anchor, found {count}: {old[:90]!r}')
    return text.replace(old, new, 1)


METHOD = '''        def refresh_after_ppo_optimizer(self):
            """Capture four new frames without keys after a PPO gradient update.

            This is an experimental, passive observation-only operation. Refuse
            uncertain/menu states and let the existing pre-action guard decide
            what happens next. Do not reset, alter rewards or inject inputs.
            """
            import asyncio

            async def capture_fresh_stack():
                if self._guard_hard_stopped:
                    raise GuardUncertainError('Cannot refresh after guard hard stop')
                self.native_history.clear()  # do not mix pre-/post-optimizer frames
                self.frame_intervals.clear()
                self.previous_observation_at = None
                observation = None
                for i in range(4):
                    if i:
                        await asyncio.sleep(.033)
                    legacy, menu_score, _ = await self._capture()
                    state, _probs = self._read_screen_state()
                    if state != 'GAMEPLAY' or menu_score >= self.args.threshold * .85:
                        return None, f'unsafe screen state={state} menu_score={menu_score:.3f}'
                    observation, _fresh, _age = self._pack(legacy, start_detection=(i == 0))
                return observation, None

            obs, reason = self._call(capture_fresh_stack(), timeout=15)
            return obs, reason, self.last_capture_end

'''


def patch(source):
    if MARKER in source:
        raise ValueError('PPO resync appears already installed. Refusing double-install.')
    if 'from tools.ppo_rollout_gap_audit import RolloutGapTracker' not in source:
        raise ValueError('Install and verify the PPO timing audit v1 first.')
    source = once(source,
        '        def _guard_diagnostic(self, readings, *, reason, name=None):\n',
        METHOD + '        def _guard_diagnostic(self, readings, *, reason, name=None):\n')
    source = once(source,
        '    from tools.ppo_rollout_gap_audit import RolloutGapTracker\n',
        '    from tools.ppo_rollout_gap_audit import RolloutGapTracker\n'
        '    from tools.ppo_rollout_resync import replace_cached_observation\n')
    source = once(source,
        '            except Exception as exc:\n'
        '                print(f"[PPO GAP] rollout-start diagnostic failed: {exc}", flush=True)\n\n'
        '        def _on_step(self):\n',
        '            except Exception as exc:\n'
        '                print(f"[PPO GAP] rollout-start diagnostic failed: {exc}", flush=True)\n'
        '            # Run after the optimizer, before SB3 predicts its first action.\n'
        '            # Only active with the explicit experimental flag.\n'
        '            if a.resync_rollouts and record is not None:\n'
        '                try:\n'
        '                    if raw is None:\n'
        '                        raise RuntimeError("No raw environment for resync")\n'
        '                    observation, skip_reason, capture_at = raw.refresh_after_ppo_optimizer()\n'
        '                    if observation is None:\n'
        '                        print(f"[PPO RESYNC] skipped: {skip_reason}; existing pre-action guard remains active", flush=True)\n'
        '                    else:\n'
        '                        replace_cached_observation(self.model, observation)\n'
        '                        age_ms = (time.perf_counter() - capture_at) * 1000\n'
        '                        print(f"[PPO RESYNC] refreshed 4-frame stack; cached age now {age_ms:.1f}ms", flush=True)\n'
        '                except Exception as exc:\n'
        '                    raise GuardUncertainError(f"[PPO RESYNC] Safe stop: {exc}") from exc\n\n'
        '        def _on_step(self):\n')
    source = once(source,
        "    p.add_argument('--rate-mode',choices=['uncapped','capped'],default='uncapped')\n",
        "    p.add_argument('--rate-mode',choices=['uncapped','capped'],default='uncapped')\n"
        "    p.add_argument('--resync-rollouts', action='store_true',\n"
        "                   help='EXPERIMENTAL: passively rebuild 4-frame observation after PPO updates')\n")
    return source


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    flag = parser.add_mutually_exclusive_group(required=True)
    flag.add_argument('--check', action='store_true')
    flag.add_argument('--apply', action='store_true')
    a = parser.parse_args()
    raw = TARGET.read_bytes()
    crlf = b'\r\n' in raw
    original = raw.decode('utf-8')
    normalized = original.replace('\r\n', '\n')
    updated = patch(normalized)
    compile(updated, str(TARGET), 'exec')
    updated_disk = updated.replace('\n', '\r\n') if crlf else updated
    if a.check:
        print(''.join(difflib.unified_diff(original.splitlines(True), updated_disk.splitlines(True),
              fromfile='a/'+str(TARGET), tofile='b/'+str(TARGET))))
        print('DRY RUN: no files changed')
        return
    if BACKUP.exists():
        parser.error(f'Refusing to overwrite backup {BACKUP}')
    BACKUP.write_bytes(raw)
    tmp = TARGET.with_name(TARGET.name+'.resync_tmp')
    try:
        tmp.write_bytes(updated_disk.encode('utf-8'))
        os.replace(tmp, TARGET)
    finally:
        if tmp.exists():
            tmp.unlink()
    print('Installed opt-in --resync-rollouts. Existing behavior unchanged without flag.')
    print('Backup:', BACKUP)


if __name__ == '__main__':
    main()
