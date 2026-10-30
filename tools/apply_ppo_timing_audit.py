#!/usr/bin/env python3
"""Strict, reversible instrumentation-only edit of train_rs_v612_ppo.py.

Run from repository root:
  python tools/apply_ppo_timing_audit.py --check
  python tools/apply_ppo_timing_audit.py --apply
No PPO behavior or checkpoint loading/saving is modified.
"""
import argparse
import difflib
from pathlib import Path
import os

TARGET = Path('train_rs_v612_ppo.py')
BACKUP = Path('train_rs_v612_ppo.py.pre_ppo_timing_audit.bak')
MARKER = "from tools.ppo_rollout_gap_audit import RolloutGapTracker"


def once(content, old, new):
    if content.count(old) != 1:
        raise RuntimeError(f'Expected exactly one anchor, found {content.count(old)}: {old[:90]!r}. '
                           'Do not force this installer onto a changed training script.')
    return content.replace(old, new, 1)


def patch(original):
    if MARKER in original:
        raise RuntimeError('PPO timing instrumentation already installed; no changes needed.')

    content = once(original,
        'def build_callback(a, output):\n    from stable_baselines3.common.callbacks import BaseCallback\n',
        'def build_callback(a, output, raw=None):\n    from stable_baselines3.common.callbacks import BaseCallback\n'
        '    from tools.ppo_rollout_gap_audit import RolloutGapTracker\n')

    content = once(content,
        '            self.log_path = output/\'progress.jsonl\'\n'
        '            self.log_handle = None\n',
        '            self.log_path = output/\'progress.jsonl\'\n'
        '            self.log_handle = None\n'
        '            self.rollout_gap_tracker = RolloutGapTracker()\n'
        '            self.rollout_gap_log_path = output / \'ppo_rollout_gaps.jsonl\'\n')

    content = once(content,
        '        def _on_training_start(self):\n'
        '            self.log_handle = self.log_path.open(\'a\',encoding=\'utf-8\')\n\n'
        '        def _on_step(self):\n',
        '        def _on_training_start(self):\n'
        '            self.log_handle = self.log_path.open(\'a\',encoding=\'utf-8\')\n\n'
        '        def _on_rollout_end(self):\n'
        '            # Before PPO.train() starts. No environment interaction.\n'
        '            try:\n'
        '                self.rollout_gap_tracker.finish_rollout(time.perf_counter())\n'
        '            except Exception as exc:\n'
        '                print(f"[PPO GAP] rollout-end diagnostic failed: {exc}", flush=True)\n\n'
        '        def _on_rollout_start(self):\n'
        '            # This runs after the preceding PPO gradient update. SB3 still\n'
        '            # holds its previous _last_obs. Measure the gap, but DO NOT\n'
        '            # alter observations, weights, rewards, or actions.\n'
        '            try:\n'
        '                record = self.rollout_gap_tracker.begin_rollout(\n'
        '                    time.perf_counter(), timesteps=int(self.num_timesteps))\n'
        '                if record is not None:\n'
        '                    print("[PPO GAP] " + json.dumps(record), flush=True)\n'
        '                    with self.rollout_gap_log_path.open(\'a\', encoding=\'utf-8\') as handle:\n'
        '                        handle.write(json.dumps(record) + "\\n")\n'
        '            except Exception as exc:\n'
        '                print(f"[PPO GAP] rollout-start diagnostic failed: {exc}", flush=True)\n\n'
        '        def _on_step(self):\n')

    content = once(content,
        '        def _on_step(self):\n'
        '            now = time.perf_counter()\n'
        '            for act in np.asarray(',
        '        def _on_step(self):\n'
        '            now = time.perf_counter()\n'
        '            try:\n'
        '                # raw is the same unwrapped env supplied to Monitor,\n'
        '                # preserving original source-frame timing even if SB3\n'
        '                # adds VecEnv observation wrappers.\n'
        '                capture_at = getattr(raw, \'last_capture_end\', None) if raw is not None else None\n'
        '                done = bool(np.any(self.locals.get(\'dones\', [])))\n'
        '                self.rollout_gap_tracker.record_step(now, capture_at=capture_at, done=done)\n'
        '            except Exception as exc:\n'
        '                if not getattr(self, \'_rollout_gap_step_error_shown\', False):\n'
        '                    self._rollout_gap_step_error_shown = True\n'
        '                    print(f"[PPO GAP] step diagnostic failed: {exc}", flush=True)\n'
        '            for act in np.asarray(')

    content = once(content,
        '        callback=build_callback(a,dest)\n',
        '        callback=build_callback(a,dest,raw=raw)\n')
    return content


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    mode = parser.add_mutually_exclusive_group(required=True)
    mode.add_argument('--check', action='store_true', help='show planned change, no write')
    mode.add_argument('--apply', action='store_true', help='back up file and apply once')
    args = parser.parse_args()
    if not TARGET.is_file():
        parser.error(f'Missing {TARGET}. Run from repo root.')
    original_bytes = TARGET.read_bytes()
    # Retain CRLF on Windows so the one-file diff stays minimal.
    crlf = b'\r\n' in original_bytes
    original = original_bytes.decode('utf-8')
    normalized = original.replace('\r\n', '\n') if crlf else original
    updated = patch(normalized)
    if crlf:
        updated = updated.replace('\n', '\r\n')
    diff = ''.join(difflib.unified_diff(original.splitlines(True), updated.splitlines(True),
                                       fromfile='a/'+str(TARGET), tofile='b/'+str(TARGET)))
    if args.check:
        print(diff)
        print('DRY RUN ONLY — no files modified')
        return
    if BACKUP.exists():
        parser.error(f'Backup already exists: {BACKUP}; resolve manually to avoid clobbering it.')
    BACKUP.write_bytes(original_bytes)
    temp = TARGET.with_name(TARGET.name + '.ppo_timing_tmp')
    try:
        temp.write_bytes(updated.encode('utf-8'))
        # Syntax check without emitting .pyc in the project.
        compile(updated, str(TARGET), 'exec')
        os.replace(temp, TARGET)
    finally:
        if temp.exists():
            temp.unlink()
    print(f'Instrumentation installed. Original backed up at {BACKUP}')
    print('Only train_rs_v612_ppo.py was changed. No game or training was launched.')


if __name__ == '__main__':
    main()
