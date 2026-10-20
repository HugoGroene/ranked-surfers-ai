# V4 high-speed / jump-only controller benchmark

This script is a **copy** of the already working `rs_v4_low_latency_benchmark_v4.py`. The ML screen-state start/restart logic remains unchanged. YOLO, the 512x288 RGB observation stack, PPO inference, and PPO updates still run.

What is new:

* `--benchmark-action jump` replaces **every** sampled PPO action with action index **3 (ArrowUp/JUMP)**. It never sends left, right, or roll during gameplay. `--benchmark-action none` sends no arrow action. `--benchmark-action ppo` (default) retains the original behavior.
* Distinct `gameplay_only_decision_intervals_ms`, `ppo_update_boundary_intervals_ms`, and `episode_boundary_intervals_ms` in the saved JSON report.
* All timing summaries now report `p99_ms` in addition to median/p95/mean.

**Important:** fixed-action runs are **only** valid for performance profiling. PPO still proposes actions and performs training updates, but the environment receives a different action. This violates the on-policy assumption; any PPO weights/rewards from these runs are not meaningful evidence of learning. No PPO model checkpoint is saved.

## Install

Copy `rs_v4_jump_speed_benchmark.py` into the root of `C:\Users\Hugo\Documents\ranked-surfers-ai`, beside the working V4 script and the already trained screen-state model. Start the local server at `http://localhost:8000` as before.

## A — baseline, 20 ms presses + quality resizing

```powershell
.\.venv\Scripts\python.exe -u rs_v4_jump_speed_benchmark.py live `
  --resolution 512x288 --fps 30 --rate-mode uncapped `
  --crop 0 0.055 1 0.965 `
  --steps 256 --max-minutes 4 --n-steps 128 --batch-size 32 --n-epochs 2 `
  --capture-mode fast --resize-method area --torch-threads 1 `
  --benchmark-action jump --press-delay-ms 20 `
  --state-model ai_data\v4_screen_states\screen_state_best.pt `
  --state-confidence 0.90 --menu-ready-timeout 30 `
  --startup-timeout 8 --startup-grace 0.10 --restart-grace 0.10 `
  --detector checkpoints\yolo_hazards_best.pt --detect-every 3 `
  --save-overlays 0 --run-name v4_jump_20ms_area `
  2>&1 | Tee-Object -FilePath ai_data\v4_jump_20ms_area.log
```

## B — test shorter 10 ms presses

Repeat the same command, but replace `--press-delay-ms 20` with `--press-delay-ms 10` and change `--run-name` and `.log` to `v4_jump_10ms_area`. **Watch the game**: if jump key events stop registering correctly, return to 20 ms (or try 15 ms). Do not infer keyboard reliability from FPS alone.

## C — test faster image resizing (optional)

Repeat B but change `--resize-method area` to `--resize-method linear`, and name the run `v4_jump_10ms_linear`. This can make tiny hazards less visually clear; don't adopt it for real PPO training solely because it benchmarks faster.

## Print comparable results

```powershell
$results = foreach ($name in @('v4_jump_20ms_area','v4_jump_10ms_area','v4_jump_10ms_linear')) {
  $path = "ai_data\rl_v2_runs\$name\v4_low_latency_report.json"
  if (!(Test-Path $path)) { continue }
  $r = Get-Content $path -Raw | ConvertFrom-Json
  $g = $r.gameplay_only_decision_intervals_ms
  [pscustomobject]@{
    Run=$name; Complete=$r.complete
    Hz=if ($g.median_ms) {[math]::Round(1000/$g.median_ms,1)} else {$null}
    P95ms=$g.p95_ms; P99ms=$g.p99_ms
    KeypressMs=$r.keypress_roundtrip_ms.median_ms
    CaptureMs=$r.capture_and_browser_step_ms.median_ms
    CropMs=$r.game_crop_resize_ms.median_ms
    Resets=$r.episode_resets_ms.count
    GuardFails=$r.startup_guard_failures
    Jumps=$r.action_counts.JUMP
    Lefts=$r.action_counts.LEFT
    Rights=$r.action_counts.RIGHT
  }
}
$results | Format-Table -AutoSize
```

Decision-rate target for the next iterations: approximately 30 Hz median and gameplay-only p95 under 50 ms, **subject to actual jump responsiveness and stable restarts**. Keep the native/inter-area path if faster resizing weakens hazard visibility.
