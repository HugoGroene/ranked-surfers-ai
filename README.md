# Ranked Surfers AI — local Practice experiments

An experimental screenshot-driven agent for a **locally served Unity WebGL copy** of Ranked Surfers. The project combines Playwright/Chromium input, screen capture, an Ultralytics YOLO hazard detector, and Stable-Baselines3 PPO with stacked RGB observations. All documented automated runs are intended for the **local Practice environment**, not the live/ranked website.

> **Status (2026-10-20):** V4 startup/restart detection and the low-latency benchmark are working on the Windows development machine. The current V4 scripts are **benchmarks**, not a trained high-performing agent. The next phase is improved YOLO labels, observations, and PPO rewards.

## Project map

| File or directory | Purpose |
|---|---|
| `rs_v4_jump_speed_benchmark.py` | **Current V4 benchmark entry point**: fast capture, asynchronous YOLO, 512×288 PPO observations, ML startup guard; reports gameplay-only latency. |
| `rs_v4_low_latency_benchmark_v4.py` | Earlier working V4 benchmark with screen-state guard (before fixed-action performance mode). |
| `rs_v4_screen_states.py` | Screen-state network and live classifier used by the V4 controller; `record`, `label`, and `watch` commands. |
| `rs_v4_screen_states_v2.py` | **Preferred training CLI**: permits validation across multiple complete sessions rather than requiring all classes in one session. |
| `rs_practice_controller.py` | Shared game control and menu-template detection. |
| `train_rs_v3_ppo.py` | Earlier V3 PPO system and reusable observation/feature code. |
| `rs_v2_live_hazards_v2.py` | YOLO hazard feature extraction. |
| `checkpoints/yolo_hazards_best.pt` | Existing YOLO detector checkpoint; **verify it is present**. |
| `ai_data/v4_screen_states/screen_state_best.pt` | **Trained ML menu/loading/gameplay classifier**; required for live V4 benchmarks. |
| `ai_data/start_templates/` | Legacy menu templates (Windows-calibrated changes may be tracked). |
| `ai_data/rl_v2_runs/<run-name>/v4_low_latency_report.json` | Per-run performance report; generated data, normally not committed. |

Earlier V2/V3 documentation and V4 diagnostic scripts are retained for reference. The downloaded Unity game is at `rankedsurfers-downloaded/` on the development machine; ensure you have permission to store or redistribute those assets before publishing a repository.

## Windows setup

Run commands from the repository root in PowerShell:

```powershell
cd "$HOME\Documents\ranked-surfers-ai"

# First-time installation only; skip if .venv is already configured.
py -3.10 -m venv .venv
.\.venv\Scripts\python.exe -m pip install --upgrade pip
.\.venv\Scripts\python.exe -m pip install "gymnasium==1.2.2" stable-baselines3 ultralytics playwright mss opencv-python numpy psutil torch torchvision
.\.venv\Scripts\python.exe -m playwright install chromium
```

The PyTorch line is a convenience starting point. For NVIDIA GPU acceleration, use the **appropriate CUDA-enabled PyTorch build** for your platform and check `torch.cuda.is_available()`. The original machine used an RTX 4060 Ti. Other transitive requirements might be needed if a historical script reports a missing dependency.

A PowerShell activation option is `Set-ExecutionPolicy -Scope Process Bypass; .\.venv\Scripts\Activate.ps1`, but none of the commands below requires activation.

Check the required model files:

```powershell
Test-Path .\checkpoints\yolo_hazards_best.pt
Test-Path .\ai_data\v4_screen_states\screen_state_best.pt
.\.venv\Scripts\python.exe -c "import torch; print('CUDA available:', torch.cuda.is_available())"
```

## Start the local game

In a separate PowerShell terminal, leave the HTTP server running:

```powershell
cd "$HOME\Documents\ranked-surfers-ai"
.\.venv\Scripts\python.exe -m http.server 8000 --directory .\rankedsurfers-downloaded
```

The game is served at `http://localhost:8000/`. In the Practice menu, pressing **R** starts/restarts the run. For manually collected screen-state screenshots, use Chromium with the **same window framing as the automated controller** (browser bars, window size, and any fullscreen configuration must match). Avoid collecting labels in a different fullscreen layout and assuming they will generalize.

## Screen-state classifier: capture → label → train → watch

This is a **whole-screen state classifier**, separate from YOLO. The controller waits for confirmed MENU, sends `R`, and suppresses arrow inputs until confident GAMEPLAY is confirmed.

```powershell
# Record a new session; during the 5-second delay focus the local game.
# The recorder never sends inputs. Manually start, die, and restart with R.
.\.venv\Scripts\python.exe rs_v4_screen_states_v2.py record `
  --session session_05 --seconds 40 --fps 20 --delay 5

# Label frames: M=menu, L=loading/transition, G=gameplay,
# O=other, K=skip, Q=save/quit (see README_RS_V4_SCREEN_STATES.md).
.\.venv\Scripts\python.exe rs_v4_screen_states_v2.py label `
  --session ai_data\v4_screen_states\raw\session_05

# Train using labeled sessions. V2 can use multiple full sessions for validation;
# each class must appear in at least two separate sessions.
.\.venv\Scripts\python.exe -u rs_v4_screen_states_v2.py train --epochs 20

# Test classifications live WITHOUT any automatic keyboard input.
.\.venv\Scripts\python.exe -u rs_v4_screen_states_v2.py watch `
  --seconds 60 --fps 10 --delay 5 --verbose
```

`train` saves `ai_data/v4_screen_states/screen_state_best.pt` and `screen_state_best.json`. Do not trust validation accuracy alone: confirm the live watch does **not** call LOADING/MENU `GAMEPLAY` before enabling automation. Raw screenshots/labels can be large and are normally kept locally, not checked into Git.

## V4 complete-pipeline benchmark (local Practice only)

Start the HTTP server first. This runs the full capture + asynchronous YOLO + PPO inference/update + screen-state restart stack at **512×288**, using the faster **linear** image resize. PPO updates run, but this is a short experimental benchmark; no reusable PPO model is saved.

```powershell
.\.venv\Scripts\python.exe -u rs_v4_jump_speed_benchmark.py live `
  --resolution 512x288 --fps 30 --rate-mode uncapped `
  --crop 0 0.055 1 0.965 `
  --steps 256 --max-minutes 4 --n-steps 128 --batch-size 32 --n-epochs 2 `
  --capture-mode fast --resize-method linear --torch-threads 1 `
  --benchmark-action jump --press-delay-ms 10 `
  --state-model ai_data\v4_screen_states\screen_state_best.pt `
  --state-confidence 0.90 --menu-ready-timeout 30 `
  --startup-timeout 8 --startup-grace 0.10 --restart-grace 0.10 `
  --detector checkpoints\yolo_hazards_best.pt --detect-every 3 `
  --save-overlays 0 --run-name v4_jump_10ms_linear `
  2>&1 | Tee-Object -FilePath ai_data\v4_jump_10ms_linear.log
```

Action modes:

- `--benchmark-action jump`: override all sampled PPO actions with JUMP (avoids deliberate lateral steering into starting trains). **Speed test only:** PPO's action/reward pairing is invalid for learning in this mode.
- `--benchmark-action none`: send no movement input, another speed-control baseline.
- `--benchmark-action ppo`: execute actual PPO decisions; required for evaluating action-control behavior, but the current benchmark uses a fresh throwaway policy.

To test the fastest measured configuration, **change** `--press-delay-ms 10` to `5`, and use a new `--run-name` and log filename such as `v4_jump_5ms_linear`. The 5 ms press was faster in timing measurements, but **missed key events and lane-change reliability have not been established**. Prefer 10 ms if uncertain. Avoid automating the live ranked website without permission.

## Analyze benchmark timings

The report is saved at `ai_data/rl_v2_runs/<run-name>/v4_low_latency_report.json`. This PowerShell snippet compares completed run names without piping directly from a `foreach` statement (which is a PowerShell syntax error):

```powershell
$names = @('v4_jump_20ms_area', 'v4_jump_10ms_area',
           'v4_jump_10ms_linear', 'v4_jump_5ms_linear')

$results = foreach ($name in $names) {
    $path = "ai_data\rl_v2_runs\$name\v4_low_latency_report.json"
    if (-not (Test-Path $path)) { continue }
    $r = Get-Content $path -Raw | ConvertFrom-Json
    $g = $r.gameplay_only_decision_intervals_ms
    [pscustomobject]@{
        Run       = $name
        Hz        = if ($g.median_ms) { [math]::Round(1000 / $g.median_ms, 1) } else { $null }
        P95ms     = $g.p95_ms
        P99ms     = $g.p99_ms
        KeypressMs = $r.keypress_roundtrip_ms.median_ms
        CaptureMs = $r.capture_and_browser_step_ms.median_ms
        GuardFails = $r.startup_guard_failures
    }
}
$results | Format-Table -AutoSize
```

Recorded results on the Windows RTX 4060 Ti machine (same 512×288 resolution):

| Action mode | Press | Resize | Median gameplay decisions/s | Gameplay p95 | Gameplay p99 |
|---|---:|---|---:|---:|---:|
| JUMP | 20 ms | area | 20.7 Hz | 60.27 ms | 61.03 ms |
| JUMP | 10 ms | area | 24.1 Hz | 51.74 ms | 52.97 ms |
| JUMP | 10 ms | linear | 26.1 Hz | 46.15 ms | 47.34 ms |
| JUMP | 5 ms | linear | **31.4 Hz** | **38.36 ms** | **39.01 ms** |

**These are median gameplay decision rates, not overall training steps/sec.** Frequent deaths/restarts dominate overall throughput; earlier complete runs saw 40 restarts of approximately 1.66 seconds each. The working ML guard reported zero startup failures and zero watchdog timeouts in the earlier 512-step V4 run. Latency measurements do not establish that 5 ms keypresses register reliably in Unity.

## Next development phase

1. Improve the YOLO dataset using actual native-resolution cropped game screenshots; investigate missed coins and barriers and false-positive trains.
2. Improve observation freshness (previous YOLO age was ~104 ms median), anticipatory features, and PPO rewards.
3. Test action reliability, especially 5 ms lane changes, before relying on the fastest benchmark setting.
4. Evaluate learned survival distributions, not just controller speed; fixed-action benchmarks **do not** demonstrate PPO learning.

## Version-control policy

Track source files, documentation, selected small reference templates, and model checkpoints needed to reproduce the controller. **Do not indiscriminately `git add .`**: `ai_data/` contains screenshots, episode captures, logs, potentially sensitive browser imagery, and large generated outputs. Stage the intended trained classifier explicitly and keep the raw recordings backed up separately. See `README_RS_V4_JUMP_SPEED.md` and `README_RS_V4_SCREEN_STATES.md` for additional details.

## V5.4 � Nine-Class Segmentation Checkpoint

The V5.4 detector is a YOLO11n segmentation model trained on
native-resolution Ranked Surfers gameplay screenshots.

Classes:
- coin
- train_front
- train_side
- train_roof
- ramp
- barrier_jump
- barrier_roll
- barrier_either
- tunnel_entrance

The V5 detector distinguishes traversable ramps and train roofs
from hazardous train fronts and different barrier geometries.

### Trained model

ai_data/v5_hazards/runs/segment_round04/weights/best.pt

### Dataset

139 manually reviewed and annotated images:
- Training: 87 images (sessions 02 and 03)
- Validation: 20 images (session 04)
- Held-out test: 32 images (session 05)

### Live benchmark

Settings:
- PPO image resolution: 512x288
- Linear image resizing
- 5 ms keyboard presses
- Asynchronous YOLO segmentation at imgsz=960
- Jump-only action override for repeatable speed testing

Measured:
- Gameplay decision rate: 31.7 Hz
- Gameplay p95: 38.79 ms
- Gameplay p99: 46.82 ms
- YOLO inference median: 20.8 ms
- Hazard age median: 64.4 ms
- Startup guard failures: 0

This is a shadow-segmentation benchmark. The new nine-class
detections are not yet integrated into PPO observations.

### Preview predictions

.\.venv\Scripts\python.exe rs_v5_preview_predictions.py `
  --model ai_data\v5_hazards\runs\segment_round04\weights\best.pt `
  --source ai_data\v5_hazards\raw\v5_capture_04 `
  --sample 12 --imgsz 960 --conf 0.20 --device 0 `
  --output ai_data\v5_hazards\preview_round04

See README_RS_V5_LIVE_SEG_BENCHMARK.md for the full live commands.

### Next milestone

Integrate the nine-class segmentation observations into PPO,
redesign rewards, and train a new policy that can learn advanced
movement and hazard anticipation.

