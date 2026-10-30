# PPO resync v1.1: fail-closed hardening

Use only after the opt-in resync v1 patch has been applied and its eight tests have passed.

This upgrade addresses one gap: resync v1 **skips** updating the cached SB3 observation when the game is not safely confirmed as GAMEPLAY, then lets PPO select an action using the old screenshot. The regular input guard checks whether *now* is safe to press keys, but that check does not make the *old PPO observation* fresh. This upgrade stops the training process safely instead of proceeding from that stale input.

It also refuses to install an observation already >= 100 ms old, and fails closed if the fresh observation is unavailable. The trained policy, reward function, action selection, checkpoint format and `--resync-rollouts` default (disabled) are unchanged. If gameplay dies while the optimizer is running, training may stop early; that is deliberate until we implement and validate a proper episode transition.

## Install in the Claude clone, with the game/trainer stopped

```powershell
& $py -B tools\harden_ppo_rollout_resync.py --check
& $py -B -m unittest discover -s tests -p 'test_ppo_resync_hardening.py' -v
& $py -B tools\harden_ppo_rollout_resync.py --apply
& $py -B -m unittest discover -s tests -p 'test_ppo_resync_hardening.py' -v
& $py -B -m unittest discover -s tests -p 'test_ppo_rollout_resync.py' -v
& $py -B -m unittest discover -s tests -p 'test_ppo_pipeline_audit.py' -v
& $py -B train_rs_v612_ppo.py --mode check --resolution 160x96
& $py -B -m py_compile train_rs_v612_ppo.py

git diff --check
```

Do not run overnight until a short `--resync-rollouts` run logs `[PPO RESYNC] refreshed 4-frame stack; cached age now ...ms` multiple times and a matched evaluation confirms the policy is no worse. The original `[PPO GAP]` entries report **pre-resync** image ages (~840 ms); that number does not need to fall. Keep the original checkpoint untouched.
