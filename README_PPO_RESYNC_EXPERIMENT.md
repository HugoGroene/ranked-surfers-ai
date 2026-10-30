# PPO rollout observation resync v1 — experimental

This is an **opt-in** correction experiment atop the existing PPO timing-audit instrumentation. It is **not** a proven survival improvement or an overnight-ready modification. Code must be tested on the user's Windows checkout before any live experiment.

## Scope

`--resync-rollouts` (off by default) tries to rebuild a four-frame RGB stack from four passive physical captures at every PPO optimizer boundary, immediately before SB3 chooses the first action in the next rollout. It does not send keys, reset games, modify rewards, change model architecture, change the rollout buffer, or change PPO optimizer states. It updates SB3's `_last_obs` to the validated new Dict observation (single-environment case).

If gameplay is not confirmed in all four frames, it skips replacement and relies on the existing guard to block any unsafe action. Unexpected resync errors fail closed and stop training. It does not solve missed rewards during optimizer updates, and the extra ~100–150ms of passive sampling might still change gameplay timing. A run should never be compared against the baseline as a completely controlled policy experiment without accounting for that difference.

## Installation (in clone root; no original repo changes)

1. Extract this zip into the existing Claude clone.
2. Use your original-repo venv Python, **without** starting the game:

```powershell
& $py -B tools\apply_ppo_rollout_resync.py --check
& $py -B -m unittest discover -s tests -p "test_ppo_rollout_resync.py" -v
```

The tests read the currently installed timing-audit source and check the planned patch, without writing source. Only if both pass:

```powershell
& $py -B tools\apply_ppo_rollout_resync.py --apply
& $py -B -m unittest discover -s tests -p "test_ppo_pipeline_audit.py" -v
& $py -B -m py_compile train_rs_v612_ppo.py
& $py -B train_rs_v612_ppo.py --help

git diff --check
```

**IMPORTANT:** Before live testing, separately verify in your installed Stable-Baselines3 version that `on_rollout_start` occurs *before* the next action is sampled, and that the `Dict` observation has the assumed one-env shape. The relevant SB3 v2.7.0 source does so, but your Windows package may be a different version. This procedure is intentionally conservative.

## Two-run comparison

Use a short **baseline diagnostic run** without `--resync-rollouts` and an **experimental** run with it, starting from the same preserved checkpoint, separate run names and otherwise matched settings. Stop immediately on guard uncertainty or repeated skipped resyncs. Look for `[PPO RESYNC] refreshed 4-frame stack` and the new cached-age figure (<100ms target). The original `[PPO GAP]` measure still reports the pre-resync age and should remain ~840ms; that is expected. Record timing, episode distributions, action counts and new nonterminal menu screens. Do not run overnight until confirmed with a short live check.

## Caveat

Only fixes the *first action's source image* after PPO training pauses when gameplay can safely be confirmed. The optimization pause still advances the Unity world, and rewards across the skipped time remain an unresolved issue. A long-run-safe fix may need reward normalization or episode-handling changes. Do not automatically restart from unverified gameplay.
