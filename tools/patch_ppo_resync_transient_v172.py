#!/usr/bin/env python3
"""Strict opt-in v1.7.1 -> v1.7.2 first-frame uncertainty fix.

 python -B -m tools.patch_ppo_resync_transient_v172 --check
 python -B -m tools.patch_ppo_resync_transient_v172 --apply

Only replaces the refresh_after_ppo_optimizer method on the exact v1.7.1
trainer. Does not edit rewards, PPO actions, screen classifier, pause clocks,
menu handoff, or any checkpoint. Fail closed on unexpected source changes.
"""
from pathlib import Path
import argparse
import hashlib
import os
import tempfile

ROOT = Path(__file__).resolve().parents[1]
TRAINER = ROOT / 'train_rs_v612_ppo.py'
HELPER = ROOT / 'tools' / 'ppo_resync_transient_v172.py'
EXPECTED_SHA256 = '02e1d840d59504a30709b259e6c9189fa100026c399a2a7f2bce71ec2de21b21'
BACKUP_SUFFIX = '.pre_ppo_resync_transient_v172.bak'

OLD = b'''        def refresh_after_ppo_optimizer(self):
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
                return observation, None

            obs, reason = self._call(capture_fresh_stack(), timeout=15)
            return obs, reason, self.last_capture_end

'''

NEW = b'''        def refresh_after_ppo_optimizer(self):
            """Rebuild four consecutive confident GAMEPLAY images, without keys.

            A near-GAMEPLAY UNCERTAIN frame just after overlay disappearance is
            neither gameplay proof nor MENU proof. Discard it and any partial
            image stack; allow a bounded number of independent fresh retries.
            No ML threshold changes, speculative restarts or reward changes.
            """
            import asyncio
            from tools.ppo_resync_transient_v172 import (
                ResyncWindow, MAX_RESYNC_CAPTURES)

            async def capture_fresh_stack():
                if self._guard_hard_stopped:
                    raise GuardUncertainError('Cannot refresh after guard hard stop')
                window = ResyncWindow()
                self.native_history.clear()  # never mix pre-/post-optimizer frames
                self.frame_intervals.clear()
                self.previous_observation_at = None
                observation = None
                accepted_traces = []
                for attempt in range(MAX_RESYNC_CAPTURES):
                    if attempt:
                        await asyncio.sleep(.033)
                    # Only the foreground local Practice may feed the PPO image.
                    focused = bool(await self.page.evaluate(
                        'document.hasFocus() && document.visibilityState === "visible"'))
                    if not focused:
                        return None, 'browser focus/visibility lost while resynchronizing; no keys sent'
                    legacy, menu_score, _ = await self._capture()
                    state, probs = self._read_screen_state()
                    decision = window.observe(
                        state=state, probs=probs, menu_score=menu_score,
                        confidence=self.state_confidence,
                        menu_threshold=self.args.threshold)
                    resumed_at = getattr(self, '_ppo_resume_trace_start_perf', None)
                    since_resume_ms = ((time.perf_counter() - resumed_at) * 1000.0
                                       if resumed_at is not None else None)
                    if decision == 'RETRY':
                        print(f'[PPO RESYNC TRANSIENT] skipped={window.skipped} '
                              f'capture={attempt+1}/{MAX_RESYNC_CAPTURES} '
                              f'discarded_valid={window.discarded_before_last_retry} '
                              f'state={state} menu_score={menu_score:.3f} '
                              f'ml_gameplay={probs.get("GAMEPLAY", 0.):.3f} '
                              f'since_resume_ms={since_resume_ms}', flush=True)
                        # _pack has already added any earlier valid frames; they
                        # cannot be used across an uncertain intervening image.
                        self.native_history.clear()
                        self.frame_intervals.clear()
                        self.previous_observation_at = None
                        observation = None
                        accepted_traces.clear()  # rejected partial stacks must not appear in the audit
                        continue
                    if decision == 'ABORT':
                        # Exact unsafe frame, for manual inspection only.
                        if self._v2_last_bgra is not None:
                            try:
                                import cv2
                                image_path = (self.output_dir /
                                    f'ppo_resume_failure_ep{self.episode_number:04d}_frame{attempt+1}.jpg')
                                wrote = cv2.imwrite(str(image_path),
                                    cv2.cvtColor(self._v2_last_bgra, cv2.COLOR_BGRA2BGR))
                                if wrote:
                                    print(f'[PPO RESYNC TRACE] failure_screenshot={image_path}', flush=True)
                            except Exception as trace_exc:
                                print(f'[PPO RESYNC TRACE] diagnostic screenshot unavailable: '
                                      f'{trace_exc}', flush=True)
                        return None, (f'{window.last_reason}; '
                                      f'capture={attempt+1}/{MAX_RESYNC_CAPTURES} '
                                      f'since_resume_ms={since_resume_ms}')
                    # Only confirmed GAMEPLAY frames are packed or counted in
                    # the existing four-frame audit. Never infer from motion.
                    if since_resume_ms is not None:
                        accepted_traces.append(
                            f'[PPO RESYNC TRACE] frame={window.accepted}/4 '
                            f'since_resume_ms={since_resume_ms:.1f} '
                            f'state={state} menu_score={menu_score:.3f} '
                            f'ml_gameplay={probs.get("GAMEPLAY", 0.):.3f} '
                            f'ml_menu={probs.get("MENU", 0.):.3f}')
                    observation, _fresh, _age = self._pack(
                        legacy, start_detection=(window.accepted == 1))
                    if decision == 'DONE':
                        # Only the four COMPLETE consecutive valid frames get
                        # audit TRACE events. Rejected partial stacks do not.
                        for line in accepted_traces:
                            print(line, flush=True)
                        return observation, None
                return None, (f'No four consecutive GAMEPLAY frames within '
                              f'{MAX_RESYNC_CAPTURES} captures; '
                              f'transient_skips={window.skipped}; no keys sent')

            obs, reason = self._call(capture_fresh_stack(), timeout=15)
            return obs, reason, self.last_capture_end

'''


def sha256(raw):
    return hashlib.sha256(raw).hexdigest()


def prepare(root=ROOT):
    target = root / 'train_rs_v612_ppo.py'
    helper = root / 'tools' / 'ppo_resync_transient_v172.py'
    if not (target.is_file() and helper.is_file()):
        raise RuntimeError('Missing trainer or v1.7.2 helper')
    source = target.read_bytes()
    digest = sha256(source)
    if digest != EXPECTED_SHA256:
        raise RuntimeError(f'Unexpected trainer hash {digest}; require exact v1.7.1. No changes made.')
    if source.count(OLD) != 1:
        raise RuntimeError(f'Expected exactly one old resync method, got {source.count(OLD)}')
    changed = source.replace(OLD, NEW)
    compile(changed.decode('utf-8'), str(target), 'exec')
    compile(helper.read_text(encoding='utf-8'), str(helper), 'exec')
    backup = target.with_name(target.name + BACKUP_SUFFIX)
    if backup.exists():
        raise RuntimeError(f'Backup already exists: {backup.name}')
    return target, backup, source, changed


def apply(root=ROOT):
    target, backup, before, after = prepare(root)
    backup.write_bytes(before)
    fd, tmp = tempfile.mkstemp(prefix=target.name+'.v172.', dir=str(target.parent))
    try:
        with os.fdopen(fd, 'wb') as out:
            out.write(after)
            out.flush()
            os.fsync(out.fileno())
        os.replace(tmp, target)
    finally:
        if os.path.exists(tmp):
            os.unlink(tmp)
    return backup


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    actions = parser.add_mutually_exclusive_group(required=True)
    actions.add_argument('--check', action='store_true')
    actions.add_argument('--apply', action='store_true')
    a = parser.parse_args()
    try:
        if a.check:
            target, backup, before, after = prepare()
            print('[PPO RESYNC V172] DRY RUN OK: exact v1.7.1 source hash matched.')
            print('[PPO RESYNC V172] Changes: one post-resume observation method only.')
            print('[PPO RESYNC V172] No files changed; existing classifier/rewards/weights untouched.')
        else:
            backup = apply()
            print('[PPO RESYNC V172] Installed opt-in transient resume window v1.7.2.')
            print('[PPO RESYNC V172] Backup: '+str(backup.name))
            print('[PPO RESYNC V172] Screen guard, death handoff, clocks, original checkpoint unchanged.')
    except Exception as exc:
        parser.exit(2, '[PPO RESYNC V172] REFUSED: '+str(exc)+'\n')


if __name__ == '__main__':
    main()
