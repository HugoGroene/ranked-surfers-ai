# Ranked Surfers AI — experimental PPO optimizer pause v1.5

**Purpose:** stop the LOCAL Unity Practice game from advancing during PPO's
optimizer updates instead of merely recovering after those updates. This is
an opt-in, fail-closed experiment. It has **not** been validated in a live
training run yet. Never enable it in ranked/online play.

## What the previous live probes established

- Moving Practice track: mean pixel delta 44.87; paused track: 0; resumed
  track: 31.72. Pause and Resume UI buttons worked.
- Site UI uses `.game-pause-panel[aria-label="Game paused"]` and
  `button[aria-label="Pause game"]` and a **Resume** button inside the overlay.
- The original trainer's raw episode duration increased 2.218 seconds during
  a measured 2.187-second pause. The distance HUD ROI was unreadable (no
  yellow pixels), so *do not claim we measured Unity's internal game clock*.
- Resync v1.4 already obtains fresh images (~8 ms old) or recovers verified
  menu screens, but one of three rollout boundaries abandoned an episode.

## What v1.5 changes

- New `--pause-ppo-optimizer` flag, **OFF by default** and valid only with
  `--mode train --resync-rollouts`. Existing `--recover-rollout-menu` remains
  independent; do **not** enable it on the initial live test.
- At PPO's `_on_rollout_end` (before gradient optimization), verify
  a fresh GAMEPLAY capture, click site's **Pause game** button, wait for paused
  overlay, and verify two samples of the game track are frozen.
- At `_on_rollout_start` (after optimizer, before next action), click the
  visible **Resume** control inside the overlay. Verify disappearance.
- Compensate all identified training-sensitive clocks for verified pause time:
  `episode_start` and `last_step` (monotonic), `episode_started`,
  `previous_observation_at` and `_prev_step_start` (performance counter),
  `suspected_since` if present, `action_context.last_action_times` (5-action
  timing features), and the exact SB3 Monitor wrapper's cumulative time origin
  `t_start` (wall clock). Validate the entire update before mutating clocks.
- Enforce a residual active episode clock drift below 0.2 seconds after
  compensation. Log pause duration and the measured residual drift.
- Existing v1.4 `refresh_after_ppo_optimizer()` then rebuilds the 4-frame
  observation and updates SB3's cached observation. Screen-state classifier,
  guarded restarts, actions, base reward function, learning rate, model
  architecture, and checkpoint bytes are NOT changed.
- If focus, GAMEPLAY, the Pause/Resume overlay, freeze evidence, the
  single-environment Monitor mapping, or clock correction cannot be verified,
  abort training without making another PPO movement command.

**Known limitations:** The pause overlay and frozen scene are strong visual
indicators, but they do not prove the game's internal simulation clock stops.
Pause/Resume itself can introduce a few milliseconds of live time. The
existing per-step 0.15-second survival-reward cap, cadence dependence, and
PPO discounting have not been changed by this patch. A verified MENU still
uses your original guarded restart; live timeout/death behavior remains to
be validated. The last rollout can remain paused while the trainer saves and
closes its browser, since there is no next rollout to resume.

## Installation — only in CLAUDE DEVELOPMENT CLONE

PowerShell. Keep the browser and trainer stopped while installing.

```powershell
cd "$HOME\Documents\ranked-surfers-ai-claude\ranked-surfers-ai-new"
$py = "$HOME\Documents\ranked-surfers-ai\.venv\Scripts\python.exe"
Expand-Archive -LiteralPath "$HOME\Downloads\ranked_surfers_ppo_optimizer_pause_v15.zip" -DestinationPath (Get-Location).Path -Force
& $py -B -m unittest discover -s tests -p "test_ppo_optimizer_pause_v15.py" -v
& $py -B -m tools.check_pause_ppo_sb3
& $py -B -m tools.install_ppo_optimizer_pause --check
```

Review output. If checks succeed:

```powershell
& $py -B -m tools.install_ppo_optimizer_pause --apply
& $py -B train_rs_v612_ppo.py --mode check --resolution 160x96
git diff --check
& $py -B train_rs_v612_ppo.py --help | Select-String 'pause-ppo-optimizer'
```

The installer **refuses unknown source layouts**, detects double-installation,
uses one exact source file, and writes backup
`train_rs_v612_ppo.py.pre_ppo_optimizer_pause_v15.bak`.

## First LIVE smoke test (only after every check passes)

Launch the existing local `server.py` in a separate terminal and keep the
Chromium Practice window visible and unobstructed. Do not use a ranked server.

```powershell
$checkpoint = "$HOME\Documents\ranked-surfers-ai\ai_data\rl_v2_runs\v612_overnight_next_420min_press5_20261027_094116\latest.zip"
$state = "checkpoints\v612_snapshot_20261021\screen_state_candidate_v2.pt"
$detector = "ai_data\v5_hazards\runs\segment_round04\weights\best.pt"
$run = "v612_pause_optimizer_smoke_" + (Get-Date -Format "yyyyMMdd_HHmmss")
$log = "ai_data\ppo_pause_optimizer_$run.log"
& $py -B -u train_rs_v612_ppo.py `
  --mode train `
  --pause-ppo-optimizer `
  --resync-rollouts `
  --resume $checkpoint `
  --run-name $run `
  --steps 512 `
  --max-minutes 5 `
  --fps 30 `
  --rate-mode uncapped `
  --press-delay-ms 5 `
  --state-model $state `
  --detector $detector `
  --imgsz 960 `
  --detect-every 3 `
  --max-stale-ms 300 `
  2>&1 | Tee-Object -FilePath $log
```

**Do not include** `--recover-rollout-menu` in the first smoke test. We need
unexpected death/menu after pausing to stop training so the failure is visible.

After the run:

```powershell
Select-String -Path $log -Pattern '\[PPO PAUSE\]|\[PPO RESYNC\]|\[PPO GAP\]|Safe stop|Traceback|ABANDONED'
Get-Content "ai_data\rl_v2_runs\$run\v61_report.json" | ConvertFrom-Json | Select-Object steps, episodes, mean_survival_seconds, completed_requested_steps | Format-List
```

Expect: repeated `[PPO PAUSE] Verified frozen gameplay ...`, each followed
at the next rollout by `[PPO PAUSE] Resumed ...`, with approximately 0.8-1.2
seconds of paused time but `active_episode_clock_drift_s` near zero. Then
`[PPO RESYNC]` should refresh the image below 100 ms. Any `[PPO PAUSE] Safe stop`
means stop, save diagnostics, and do not run overnight training. Last rollout
need not resume, because the training process closes its browser.

**How to roll back v1.5 only (trainer stopped):**

```powershell
Copy-Item 'train_rs_v612_ppo.py.pre_ppo_optimizer_pause_v15.bak' 'train_rs_v612_ppo.py' -Force
```

This leaves v1.4 instrumentation and recovery intact. Original project and
original 4.61s checkpoint are never overwritten by this package.
