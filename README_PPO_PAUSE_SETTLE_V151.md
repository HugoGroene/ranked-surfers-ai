# PPO optimizer pause: delayed Unity freeze confirmation v1.5.1

**Status:** experimental, opt-in, requires installed v1.5 `--pause-ppo-optimizer` patch in Claude development clone.

## Why the initial 512-step test stopped

The React `Game paused` overlay became visible, but the first two track captures still differed by **44.558** mean pixel values. The site can show the overlay before the Unity iframe has actually processed the pause state. **This single failed capture pair does not demonstrate that the browser cannot pause Unity.** The earlier standalone pause/resume probes found moving → zero moving → moving scene checks.

## Fix

The new `FreezeWindow` waits up to **16 captures** spaced by 90 ms, requiring **three consecutive unchanged track intervals** (4 independent images), with the same **0.25 pixel-delta limit**. While waiting, the existing ML screen-state classifier must continue to report `GAMEPLAY` and the original menu template must remain below its existing 85%-of-threshold limit. Pause overlay must remain visible. If any check fails or freezing never stabilizes, the trainer **stops safely**. **No tolerance is relaxed**.

**Clock correctness:** v1.5 started subtracting pause time when the overlay appeared. This update starts the pause clock at the first verified stationary capture of the sustained stable sequence. The period in which Unity may still be moving is **not** counted as paused. The existing v1.5 `compensate_pause_clocks` function still shifts wall-time origins only after verified resume, and v1.4 resync refreshes four frames before the next PPO action.

**Unchanged:** checkpoint/architecture, reward coefficients, actions, classifier-based death/restarts, strict opt-in flags and optimizer resume ordering. The original repo and checkpoint are untouched. This does not solve independent step-cadence reward/discounting bias. Menu recovery stays disabled in the initial smoke test.

## Commands

From `C:\Users\Hugo\Documents\ranked-surfers-ai-claude\ranked-surfers-ai-new` with PPO stopped and `$py` set:

```powershell
Expand-Archive -LiteralPath "$HOME\Downloads\ranked_surfers_ppo_pause_settle_v151.zip" -DestinationPath (Get-Location).Path -Force
& $py -B -m unittest discover -s tests -p 'test_ppo_optimizer_pause_settle_v151.py' -v
& $py -B -m unittest discover -s tests -p 'test_ppo_optimizer_pause_v15.py' -v
& $py -B -m tools.patch_ppo_optimizer_pause_settle_v151 --check
& $py -B -m tools.patch_ppo_optimizer_pause_settle_v151 --apply
& $py -B train_rs_v612_ppo.py --mode check --resolution 160x96
git diff --check
```

Only if every step succeeds, start local server in another PowerShell, then:

```powershell
$checkpoint = "$HOME\Documents\ranked-surfers-ai\ai_data\rl_v2_runs\v612_overnight_next_420min_press5_20261027_094116\latest.zip"
$state = 'checkpoints\v612_snapshot_20261021\screen_state_candidate_v2.pt'
$detector = 'ai_data\v5_hazards\runs\segment_round04\weights\best.pt'
$run = 'v612_pause_settle_v151_' + (Get-Date -Format 'yyyyMMdd_HHmmss')
$log = "ai_data\ppo_pause_$run.log"
& $py -B -u train_rs_v612_ppo.py `
  --mode train --pause-ppo-optimizer --resync-rollouts `
  --resume $checkpoint --run-name $run --steps 512 --max-minutes 5 `
  --fps 30 --rate-mode uncapped --press-delay-ms 5 `
  --state-model $state --detector $detector `
  --imgsz 960 --detect-every 3 --max-stale-ms 300 `
  2>&1 | Tee-Object -FilePath $log
Select-String -Path $log -Pattern '\[PPO PAUSE\]|\[PPO RESYNC\]|Safe stop|Traceback|ABANDONED'
```

## Success signal

- `[PPO PAUSE] Freeze verified after N captures; track_deltas=[...]` with the last three values `<=0.25`.
- `[PPO PAUSE] Verified frozen gameplay before optimizer; paused-track delta=...`.
- `[PPO GAP]` shows the natural optimizer pause, while `[PPO PAUSE] Resumed after optimizer ... active_episode_clock_drift_s` is close to zero (target -0.10..+0.20 s).
- `[PPO RESYNC] refreshed 4-frame stack; cached age now ...ms` below 100ms.
- All 512 steps, zero safety stops, zero `ABANDONED` during paused optimizer updates.

**Fail-closed:** If Unity never freezes, a classifier disagrees, the overlay disappears, or clocks become inconsistent, the run stops and saves a separate partial checkpoint. Never run overnight until the 512-step smoke run succeeds.
