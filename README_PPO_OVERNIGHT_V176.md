# Ranked Surfers PPO v1.7.6 — guarded overnight reliability patch

Built from Claude's independent read-only audit and the exact successful
v1.7.5 trainer **SHA256 `ddd3146dfa5916e8a74f3c672a1353f62b60603aca69d845cac0f918972926fa`**.
The v1.7.5.1 reward auditor is retained unchanged.

## Scope (do not conflate offline checks with real Unity testing)

- **A1:** If the rollout-end classifier is transiently uncertain or a menu suspicion
  is pending, run the existing bounded ≤8-frame recheck. Only two *new* confident
  GAMEPLAY+unique-Pause-button observations permit Pause. Four recent independently
  corroborated MENU observations permit the existing guarded death handoff.
  Anything else stops. No movement or restart is sent while deciding.
- **A2:** If the Resume-button click times out, recheck the pause overlay. A hidden
  overlay is not enough to authorize gameplay: the unchanged 4-frame post-resume
  refresh remains mandatory. If overlay is visible, a second click requires a
  unique visible Resume button, browser focus, and independently verified frozen
  track. At most one retry. If unresolved, stop. No threshold relaxation.
- **A3:** Test actual v1.7.6 method bodies from the current trainer (after install)
  or source-matched payload (before install), rather than reconstructing the old
  v1.7.3 version. Synthetic browser fakes, not a live Unity test.
- **A6:** Each checkpoint is written to a temporary local ZIP, CRC checked,
  backed up into `latest.prev.zip`, then atomically renamed to `latest.zip`.
  Periodic saves with `--pause-ppo-optimizer` happen at the verified frozen
  rollout boundary (or confirmed terminal MENU), not mid-gameplay. Final save is
  also atomic. This does not modify the source checkpoint passed with `--resume`.

**Not changed:** policy actions, PPO algorithm, rewards, menu corroboration,
classifier thresholds, live restart, HUD detector, 4-frame resync, reward auditor.

**Not fixed in this patch:** A4 (non-menu transient death animation during resync)
remains safe-stop, not automatic recovery; A5 (`--recover-rollout-menu`) remains
DISABLED because it can abandon an unrewarded death. Keep that flag OFF.

## Contents

- `tools/patch_ppo_overnight_v176.py`: source-hashed check / opt-in installer
- `tools/_ppo_overnight_v176_payload/trainer.txt`: complete patched trainer bytes
- `tools/ppo_atomic_checkpoint_v176.py`: atomic save helper
- `tests/test_ppo_overnight_v176.py`: 20 offline regression tests

## On Windows: from the local project directory, training STOPPED

```powershell
cd "$HOME\Documents\ranked-surfers-ai-claude\ranked-surfers-ai-new"
$py = "$HOME\Documents\ranked-surfers-ai\.venv\Scripts\python.exe"
Expand-Archive -LiteralPath "$HOME\Downloads\ranked_surfers_ppo_overnight_audit_v176.zip" -DestinationPath (Get-Location).Path -Force
& $py -B -m unittest discover -s tests -p "test_ppo_overnight_v176.py" -v
if ($LASTEXITCODE -ne 0) { throw 'New tests failed; do not install.' }
& $py -B -m tools.patch_ppo_overnight_v176 --check
if ($LASTEXITCODE -ne 0) { throw 'Source verification failed; do not install.' }
& $py -B -m tools.patch_ppo_overnight_v176 --apply
if ($LASTEXITCODE -ne 0) { throw 'Installation failed.' }
& $py -B -m unittest discover -s tests -p "test_ppo_overnight_v176.py" -v
& $py -B train_rs_v612_ppo.py --mode check --resolution 160x96
& $py -B -m tools.ppo_reward_timing_audit --self-test
git diff --check
```

Expected offline tests before install: **20 OK**. After install: **19 OK,
1 skip** (the preinstall hash test). A `.pre_ppo_overnight_v176.bak` trainer
backup is created. The installer refuses an unexpected trainer SHA, and does
not modify original weights, the run folder, or Github.

## Live verification BEFORE overnight

Use existing `--mode train --pause-ppo-optimizer --resync-rollouts
--audit-reward-timing`, `--steps 2048`, a NEW run folder, and the current
verified pretrained continuation. Audit with `--min-resumes 15 --min-steps 2048`
and reward replay with `--min-steps 2048`.

Only after that completes without UI stops should you attempt a longer monitored
trial. Do not push to GitHub before reviewing the exact `git diff` and Windows
live result. Original 7h checkpoint stays untouched.

## Constraints / uncertainty

Playwright/Unity UI races were tested offline with fake pages; success there
cannot prove all live DOM/camera transitions are safe. The `PASS_REWARD_REPLAY`
checker replays arithmetic, not independent Unity wall time. `latest.prev.zip`
is the *previous successfully saved same-run checkpoint*, not the original 7h
pretrained model.
