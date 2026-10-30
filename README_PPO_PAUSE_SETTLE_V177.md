# PPO 1.7.7: disappearing Pause overlay during Unity freeze settling

## Live trigger

The 8-hour v1.7.6 trial stopped safely at 6,656 steps after
`[PPO PAUSE] Safe stop at rollout-end: Pause overlay vanished while settling Unity`.
The final partial checkpoint was saved atomically.

## Change

Only `PPOV61Env.pause_for_ppo_optimizer()` in the trainer is changed.
The overlay's disappearance no longer automatically aborts on its first occurrence.
A bounded recheck (existing `PauseControlRecheck`, maximum 8 captures) determines:

* OVERLAY returns: retry the complete freeze verification from scratch, without clicking.
* Two new classifier-confirmed GAMEPLAY frames, no corroborated MENU, unique Pause button:
  at most **one** additional verified Pause click across the whole method, then
  require visible Pause overlay and three frozen-track pairs again.
* Four corroborated MENU frames: delegate to existing rollout-terminal handoff,
  recording the death using the original guarded env.step, with no movement or R.
* Ambiguous UI, multiple Pause buttons, loss of focus, or second disappearance:
  **stop safely**. A diagnostic JPG is saved on unresolved rechecks.

The .25 freeze threshold, game-state classifier thresholds, rewards, PPO actions,
post-resume four-frame resync, resume-click behavior, and atomic checkpointing
are unchanged. This is regression-tested offline, not yet live-verified.

## Installation

Stop training and keep an unmodified copy of the current partial checkpoint.
From the repository root:

```powershell
$py = "$HOME\Documents\ranked-surfers-ai\.venv\Scripts\python.exe"
Expand-Archive -LiteralPath "$HOME\Downloads\ranked_surfers_ppo_pause_settle_v177.zip" -DestinationPath (Get-Location).Path -Force
& $py -B -m unittest discover -s tests -p "test_ppo_pause_settle_v177.py" -v
& $py -B -m tools.patch_ppo_pause_settle_v177 --check
& $py -B -m tools.patch_ppo_pause_settle_v177 --apply
& $py -B -m unittest discover -s tests -p "test_ppo_pause_settle_v177.py" -v
& $py -B -m unittest discover -s tests -p "test_ppo_current_runtime.py" -v
& $py -B train_rs_v612_ppo.py --mode check --resolution 160x96
& $py -B -m tools.ppo_reward_timing_audit --self-test
git diff --check
```

Use PowerShell `$LASTEXITCODE` to verify each step; do not start training after any failure.
The installer verifies the exact v1.7.6 SHA256, backs up the trainer, and modifies
no model or checkpoint. No `--recover-rollout-menu` flag.

## Verification

Run a new 2,048-step controlled live test from the partial overnight checkpoint
in a unique run folder, audit logs and reward JSONL, and **do not** launch another
8-hour run until it passes. Not every transient recovery is guaranteed to be
encountered during a short live run. If the UI remains unverified, a safe stop
is still expected and should be investigated from its diagnostic screenshot.
