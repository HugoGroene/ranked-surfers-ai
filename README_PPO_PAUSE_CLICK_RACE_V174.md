# Ranked Surfers PPO v1.7.4 – Pause-button click/DOM race

**Scope:** optional source-matched patch for exact v1.7.3 `train_rs_v612_ppo.py` in the Claude development clone. Does not modify reward arithmetic, classifier, PPO actions, checkpoint, optimizer clock compensation, or evaluation logic.

## Evidence

A 4,096-step v1.7.3 live test stopped at 640 actions, during episode 6. The screen classifier had confirmed GAMEPLAY before a pause attempt, but Playwright spent 1,800 ms waiting for `button[aria-label="Pause game"]` to be stable. It reported the node was detached. This can happen while React re-renders or on a death transition; it is **not proof of death**. The pause verifier fail-closed, correctly.

## New behavior

When the Playwright **Pause button click itself** times out:

1. Don't assume the click failed. If the pause overlay appeared, do not click again; verify **three frozen track intervals** and the usual template threshold before declaring Unity paused.
2. If no overlay, observe up to eight *new* screen/template captures via the unchanged `PauseControlRecheck` helper.
3. Four corroborated `MENU` frames: anchor survival duration to the first corroborated menu observation and raise the existing guarded `MENU` handoff candidate. The unchanged rollout callback reconfirms `MENU` after this method returns, and the next normal guarded env step must record a real terminal reward and send **no movement key**.
4. Two verified `GAMEPLAY` frames plus one unique visible Pause button: allow **one** new Pause click attempt. If that also times out, re-evaluate the overlay/screen once; never click a third time.
5. Focus loss, multiple Pause controls, uncorroborated ML-only MENU, unresolved state, missing freeze evidence: **safe stop**. No blind restart, no blind key.

This reuses the previous missing-button guard and leaves all reward and classifier rules intact.

## Install (training stopped)

From `C:\Users\Hugo\Documents\ranked-surfers-ai-claude\ranked-surfers-ai-new`:

```powershell
$py = "$HOME\Documents\ranked-surfers-ai\.venv\Scripts\python.exe"
Expand-Archive -LiteralPath "$HOME\Downloads\ranked_surfers_ppo_pause_click_race_v174.zip" -DestinationPath (Get-Location).Path -Force
& $py -B -m unittest discover -s tests -p "test_ppo_pause_click_v174.py" -v
if ($LASTEXITCODE -ne 0) { throw "v1.7.4 tests failed – DO NOT APPLY" }
& $py -B -m tools.patch_ppo_pause_click_v174 --check
if ($LASTEXITCODE -ne 0) { throw "Source hash check failed – DO NOT APPLY" }
& $py -B -m tools.patch_ppo_pause_click_v174 --apply
if ($LASTEXITCODE -ne 0) { throw "v1.7.4 installation failed" }
& $py -B train_rs_v612_ppo.py --mode check --resolution 160x96
& $py -B -m tools.ppo_reward_timing_audit --self-test
git diff --check
```

The installer requires the exact installed v1.7.3 trainer SHA256 `a65a5e0e384823cfc4f6d3bf08491b09ddf0202e997684520e107ff5893cfb23`; it backs up the trainer as `train_rs_v612_ppo.py.pre_ppo_pause_click_v174.bak`. It will refuse subsequent installation attempts rather than overwrite backups. No push required.

## Controlled live test

Keep your Practice server running, Chromium visible/focused, and other windows away from the game crop. Train again from **the original pretrained checkpoint** (never overwrite it):

```powershell
$checkpoint = "$HOME\Documents\ranked-surfers-ai\ai_data\rl_v2_runs\v612_overnight_next_420min_press5_20261027_094116\latest.zip"
$state = "checkpoints\v612_snapshot_20261021\screen_state_candidate_v2.pt"
$detector = "ai_data\v5_hazards\runs\segment_round04\weights\best.pt"
$run = "v612_pause_click_v174_" + (Get-Date -Format "yyyyMMdd_HHmmss")
$log = "ai_data\ppo_pause_$run.log"
& $py -B -u train_rs_v612_ppo.py `
  --mode train --pause-ppo-optimizer --resync-rollouts --audit-reward-timing `
  --resume $checkpoint --run-name $run --steps 2048 --max-minutes 10 `
  --fps 30 --rate-mode uncapped --press-delay-ms 5 `
  --state-model $state --detector $detector --imgsz 960 `
  --detect-every 3 --max-stale-ms 300 `
  2>&1 | Tee-Object -FilePath $log
& $py -B -m tools.ppo_reward_timing_audit --log $log --min-resumes 15 --min-steps 2048
$rewardLog = "ai_data\rl_v2_runs\$run\ppo_reward_step_audit.jsonl"
& $py -B -m tools.ppo_reward_timing_audit --reward-steps $rewardLog --min-steps 2048
Select-String -Path $log -Pattern "\[PPO PAUSE CLICK\]|\[PPO PAUSE CONTROL\]|\[PPO MENU HANDOFF\]|Safe stop|Traceback"
```

For a 2,048-step run, 15 continuations are expected; the final 16th freeze may be recorded without a matching resume. Both `PASS_STABILITY_CANDIDATE` and `PASS_REWARD_REPLAY` are required. Absence of a click timeout in a passing run means the specific new branch was not live-tested.

## Evaluation

Evaluation does **not** need optimizer pauses. The `--mode eval` implementation uses `model.predict` and `monitored.step` but never calls PPO gradient optimization or the rollout callbacks. The argument parser explicitly forbids `--pause-ppo-optimizer` except in `--mode train` with `--resync-rollouts`. Keep the ordinary menu/input safety guard during evaluation; omit `--pause-ppo-optimizer`, `--resync-rollouts`, and `--audit-reward-timing`.

Example deterministic evaluation:

```powershell
$evalRun = "v612_eval_deterministic_" + (Get-Date -Format "yyyyMMdd_HHmmss")
& $py -B -u train_rs_v612_ppo.py `
  --mode eval --eval-policy deterministic --eval-episodes 20 `
  --resume $checkpoint --run-name $evalRun --max-minutes 15 `
  --fps 30 --rate-mode uncapped --press-delay-ms 5 `
  --state-model $state --detector $detector --imgsz 960 `
  --detect-every 3 --max-stale-ms 300
```

Evaluation uses the checkpoint given in `--resume`, e.g. original baseline or new experimental model; neither path gets trained or overwritten in evaluation.

## Verification

17 new integration/installer tests passed pre-install and post-install. The existing v1.7.3 test suite also passed on the patched trainer: 18 tests passed, 1 expected installer skip. These are **offline mocked UI tests**, not a live browser test. Windows has not yet exercised this new branch live.
