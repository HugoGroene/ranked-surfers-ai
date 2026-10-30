# PPO optimizer-pause MENU recovery — experimental v1.2

## Why this exists

The prior live smoke test verified ~1,022 ms cached observation age after PPO's optimizer update, and the game was already at a confirmed `MENU` screen. The fail-closed resync correctly stopped before sending a stale action. Repeating that safety stop is not a long-run training strategy.

This **optional** patch introduces a conservative, additional mode for the specific case where Unity has died while PPO was updating: repeated passive confirmation of a menu screen, then **reset the monitored vectorized environment** and mark SB3's next observation as the start of a new episode. It never sends movement keys during verification, doesn't disable the existing gameplay guard, and doesn't modify rewards, weights, or the original 4.61s checkpoint.

**Important statistical caveat:** the interrupted episode ends while PPO is doing optimizer work, *after* its previous rollout was trained. There is no observed terminal transition or death penalty for that episode. The patch explicitly reports `ABANDONED partial episode without death reward`; it does not pretend training semantics are fully repaired. Frequent abandonments bias learning. If they are common, focus on preventing the optimizer pause/game mismatch instead of running overnight with this workaround.

## Install in Claude clone only

Keep local game/server/training stopped. You need previously installed PPO timing audit v1, opt-in resync v1, and resync hardening v1.1.

```powershell
cd "$HOME\Documents\ranked-surfers-ai-claude\ranked-surfers-ai-new"
$py = "$HOME\Documents\ranked-surfers-ai\.venv\Scripts\python.exe"
Expand-Archive -LiteralPath "$HOME\Downloads\ranked_surfers_ppo_optimizer_menu_recovery_v12.zip" -DestinationPath (Get-Location).Path -Force
& $py -B tools\install_ppo_optimizer_menu_recovery.py --check
& $py -B tools\install_ppo_optimizer_menu_recovery.py --apply
& $py -B -m unittest discover -s tests -p 'test_ppo_optimizer_menu_recovery.py' -v
& $py -B -m unittest discover -s tests -p 'test_ppo_resync_hardening.py' -v
& $py -B -m unittest discover -s tests -p 'test_ppo_rollout_resync.py' -v
& $py -B -m unittest discover -s tests -p 'test_ppo_pipeline_audit.py' -v
& $py -B train_rs_v612_ppo.py --mode check --resolution 160x96
git diff --check
```

The installer makes a single byte-identical backup `train_rs_v612_ppo.py.pre_menu_recovery.bak`, rejects repeats and unexpected edits; `--check` is read-only. The *existing* `--resync-rollouts` alone still stops on MENU. The new behavior only occurs when **both** `--resync-rollouts` and `--recover-rollout-menu` are enabled.

## Short local Practice-only smoke test

Start the existing `server.py` in a second PowerShell window. Keep Chrome visible and do not manually press arrows.

```powershell
$checkpoint = "$HOME\Documents\ranked-surfers-ai\ai_data\rl_v2_runs\v612_overnight_next_420min_press5_20261027_094116\latest.zip"
$state = 'checkpoints\v612_snapshot_20261021\screen_state_candidate_v2.pt'
$detector = 'ai_data\v5_hazards\runs\segment_round04\weights\best.pt'
$run = 'v612_resync_menu_smoke_' + (Get-Date -Format 'yyyyMMdd_HHmmss')
$log = 'ai_data\ppo_resync_menu_smoke_' + $run + '.log'
& $py -B -u train_rs_v612_ppo.py `
  --mode train --resync-rollouts --recover-rollout-menu `
  --resume $checkpoint --run-name $run `
  --steps 512 --max-minutes 5 `
  --fps 30 --rate-mode uncapped --press-delay-ms 5 `
  --state-model $state --detector $detector `
  --imgsz 960 --detect-every 3 --max-stale-ms 300 `
  2>&1 | Tee-Object -FilePath $log

Select-String -Path $log -Pattern '\[PPO GAP\]|\[PPO RESYNC\]|Safe stop|Traceback|ABANDONED'
```

## Interpretation / stop criteria

- `[PPO RESYNC] refreshed 4-frame stack; cached age now ...ms`: fresh gameplay path succeeded (must be <100 ms).
- `[PPO RESYNC] Verified MENU during optimizer; restarted local Practice ... new_episode=True; ABANDONED ...`: recovered from *verified* death without using cached stale image. **Count these; frequent recoveries are a training-quality problem.**
- `[PPO RESYNC] Safe stop: ...`: failed closed; do **not** extend to overnight.
- Neither of the first two events before training ends: test didn't establish resync behavior.

This is a **short controlled experiment, not an approved overnight training configuration**. Check the log and report before longer runs. Don't stage `ai_data` or `.bak` files in Git.
