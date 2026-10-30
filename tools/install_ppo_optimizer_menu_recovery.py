#!/usr/bin/env python3
"""Fail-closed patch installer for opt-in verified menu recovery after PPO updates.

Requires previously installed timing audit, resync v1, and hardening v1.1.
Run inside the CLAUDE DEVELOPMENT CLONE only; --check never modifies files.
"""
import argparse
import difflib
import os
from pathlib import Path

TARGET = Path('train_rs_v612_ppo.py')
BACKUP = Path('train_rs_v612_ppo.py.pre_menu_recovery.bak')

METHOD_ANCHOR = '        def _guard_diagnostic(self, readings, *, reason, name=None):\n'
METHOD = '''        def confirm_post_optimizer_menu(self):
            """Passive, repeated menu confirmation before any new R/restart.

            Never inject keys here. Returning False must cause a safe stop.
            """
            import asyncio
            from tools.ppo_optimizer_menu_recovery import accept_menu_checks

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
                checks = []
                for i in range(3):
                    if i:
                        await asyncio.sleep(.075)
                    _legacy, menu_score, _unused = await self._capture()
                    state, _probs = self._read_screen_state()
                    checks.append((str(state), float(menu_score)))
                threshold = float(self.args.threshold) * .85
                allowed = accept_menu_checks(checks, threshold=threshold, focused=focused)
                return allowed, {'checks': checks, 'threshold': round(threshold, 3)}

            return self._call(confirm(), timeout=12)

'''

IMPORT_ANCHOR = '    from tools.ppo_rollout_resync import replace_cached_observation\n'
IMPORT_NEW = (IMPORT_ANCHOR +
    '    from tools.ppo_optimizer_menu_recovery import candidate_is_menu, validated_new_vec_episode\n')

FLAG_ANCHOR = ("    p.add_argument('--resync-rollouts', action='store_true',\n"
               "                   help='EXPERIMENTAL: passively rebuild 4-frame observation after PPO updates')\n")
FLAG_NEW = (FLAG_ANCHOR +
    "    p.add_argument('--recover-rollout-menu', action='store_true',\n"
    "                   help='EXPERIMENTAL: after verified death during PPO update, reset local Practice with episode flag; skipped death reward is logged')\n")

CB_OLD = '''                    if observation is None:
                        # Never allow the first action after an optimizer pause to
                        # consume a stale observation when refreshing was unsafe.
                        raise GuardUncertainError(
                            f"Cannot verify gameplay after PPO update: {skip_reason}. "
                            "Stopping instead of acting on the stale cached image.")
'''
CB_NEW = '''                    if observation is None:
                        # If the game died *during the optimizer*, the preceding
                        # rollout was already trained on. Safely reset a NEW episode
                        # without inventing an action or a synthetic terminal reward.
                        # An abandoned in-progress episode is logged explicitly.
                        if not (a.recover_rollout_menu and candidate_is_menu(skip_reason)):
                            raise GuardUncertainError(
                                f"Cannot verify gameplay after PPO update: {skip_reason}. "
                                "Stopping instead of acting on the stale cached image.")
                        verified, evidence = raw.confirm_post_optimizer_menu()
                        if not verified:
                            raise GuardUncertainError(
                                f"Post-update MENU not independently confirmed: {evidence}. "
                                "Stopping without restarting active gameplay.")
                        # Use the *monitored VecEnv*, not raw.reset(), so monitor
                        # and SB3 wrappers remain synchronized. Only confirmed MENU
                        # can reach the inherited safe R-restart routine.
                        new_vec_obs = self.model.env.reset()
                        reset_obs, episode_starts = validated_new_vec_episode(
                            self.model._last_obs, new_vec_obs, self.model.n_envs)
                        fresh_at = getattr(raw, 'last_capture_end', None)
                        reset_age_ms = ((time.perf_counter() - fresh_at) * 1000
                                        if fresh_at is not None else float('inf'))
                        if not (0 <= reset_age_ms < 100.0):
                            raise GuardUncertainError(
                                f"Reset gameplay capture is stale ({reset_age_ms:.1f}ms).")
                        self.model._last_obs = reset_obs
                        self.model._last_episode_starts = episode_starts
                        print('[PPO RESYNC] Verified MENU during optimizer; '
                              f'restarted local Practice with fresh observation '
                              f'({reset_age_ms:.1f}ms old), new_episode=True; '
                              'ABANDONED partial episode without death reward.', flush=True)
                        return
'''


def replace_one(s, a, b):
    matches = s.count(a)
    if matches != 1:
        raise ValueError(f'Expected exactly one installation anchor (found {matches}): {a[:70]!r}')
    return s.replace(a, b, 1)


def patch(source):
    if 'def confirm_post_optimizer_menu(self):' in source or '--recover-rollout-menu' in source:
        raise ValueError('MENU recovery appears already installed; refusing duplicate installation')
    if ('from tools.ppo_rollout_gap_audit import RolloutGapTracker' not in source
            or 'def refresh_after_ppo_optimizer(self):' not in source):
        raise ValueError('Required prior PPO timing and resync patches are missing')
    s = replace_one(source, METHOD_ANCHOR, METHOD + METHOD_ANCHOR)
    s = replace_one(s, IMPORT_ANCHOR, IMPORT_NEW)
    s = replace_one(s, FLAG_ANCHOR, FLAG_NEW)
    s = replace_one(s, CB_OLD, CB_NEW)
    compile(s, str(TARGET), 'exec')
    return s


def main():
    p = argparse.ArgumentParser(description=__doc__)
    g = p.add_mutually_exclusive_group(required=True)
    g.add_argument('--check', action='store_true')
    g.add_argument('--apply', action='store_true')
    args = p.parse_args()
    original_bytes = TARGET.read_bytes()
    had_crlf = b'\r\n' in original_bytes
    normalized = original_bytes.decode('utf-8').replace('\r\n', '\n')
    patched = patch(normalized)
    if args.check:
        print(''.join(difflib.unified_diff(normalized.splitlines(True), patched.splitlines(True),
              fromfile='a/'+str(TARGET), tofile='b/'+str(TARGET))))
        print('DRY RUN ONLY — no files changed')
        return
    if BACKUP.exists():
        p.error(f'Backup exists; refusing overwrite: {BACKUP}')
    BACKUP.write_bytes(original_bytes)
    temp = TARGET.with_name(TARGET.name+'.menu_recovery_tmp')
    try:
        temp.write_bytes((patched.replace('\n','\r\n') if had_crlf else patched).encode('utf-8'))
        os.replace(temp, TARGET)
    finally:
        if temp.exists():
            temp.unlink()
    print('Installed opt-in --recover-rollout-menu; enabled ONLY together with --resync-rollouts.')
    print('Original source backup:', BACKUP)
    print('WARNING: Death during optimizer is detected after the previous rollout trained; no terminal reward is invented.')

if __name__ == '__main__':
    main()
