# Ranked Surfers AI - V6.1.2 PPO

Screenshot-driven reinforcement learning for a **locally served Practice copy** of Ranked Surfers. The agent plays through the visible Chromium/Unity WebGL game, using Stable-Baselines3 PPO with screen capture, a nine-class YOLO segmentation detector, and a gameplay/menu safety guard. These experiments are intended **only for local Practice** (`http://localhost:8000`), not live or ranked play.

## Current status (26 October 2026)

The current model trains **independently using PPO**, without human demonstrations or scripted obstacle-avoidance actions. The trained policy selects one of five actions: `NONE`, `LEFT`, `RIGHT`, `JUMP`, or `ROLL`.

- **Training and evaluation:** `train_rs_v612_ppo.py`
- **Gameplay safety guard:** `rs_v612_gameplay_guard.py`
- **Runtime screen-state classifier code:** `rs_v4_screen_states.py`
- **Screen-state model:** `ai_data/v4_screen_states_rebuild_01/screen_state_candidate_v2.pt`
- **Nine-class YOLO segmentation model:** `ai_data/v5_hazards/runs/segment_round04/weights/best.pt`
- **Observation:** four stacked RGB frames at 512 x 288 (12 channels), plus a 557-dimensional numerical auxiliary observation
- **Operating configuration:** 5 ms keypresses, target FPS 30, `--rate-mode uncapped`, fast capture, linear resize, YOLO inference every three decisions
- **Latest selected PPO checkpoint backup:** `checkpoints/v612_after_120min_press5_20261026.zip` (Git LFS)
- **Original local checkpoint:** `ai_data/rl_v2_runs/v612_resume_120min_press5_20261026_133230/latest.zip`

### Recorded evaluation results

A controlled 100-episode evaluation using the 5 ms configuration produced:

| Evaluated policy | Episodes | Mean survival | Median survival | Survived >= 5 s |
| --- | ---: | ---: | ---: | ---: |
| Earlier PPO checkpoint | 100 | 3.11 s | 3.35 s | 0 |
| PPO checkpoint after 120-minute continuation | 100 | **3.84 s** | **4.16 s** | **15** |
| No-action reference | 100 | 3.48 s | 3.37 s | 12 |

The corresponding JSON data is preserved at `reports/v612_learning_proof_20261026_153745/`. These results suggest PPO improvement but do not guarantee better performance on every procedurally generated episode. **Do not repeat the no-action reference for every checkpoint**; retain it for comparisons under unchanged game and evaluation conditions.

## Project structure

| Path | Purpose |
| --- | --- |
| `train_rs_v612_ppo.py` | Active PPO trainer, evaluator and offline check |
| `rs_v612_gameplay_guard.py` | Fail-closed visual gate before game inputs |
| `rs_v61_features.py`, `rs_v6_ppo_features.py`, `rs_v61_rewards.py` | Active observation and reward logic |
| `rs_v5_segmentation_live_benchmark.py`, `rs_v5_shadow_segmentation.py` | Live capture and segmentation dependencies |
| `rs_v4_screen_states.py` | Current runtime state-classifier implementation |
| `rs_practice_controller.py`, `train_rs_v3_ppo.py` and other older-named root modules | **Still required** via the inherited controller/import chain |
| `server.py` | Local Unity WebGL server with required content-encoding and cross-origin headers |
| `test_rs_v612_guard.py`, `test_rs_v61.py` | Active regression tests |
| `checkpoints/` | Selected PPO/classifier/detector weights backed up in Git LFS |
| `reports/` | Selected JSON experiment evidence committed to Git |
| `ai_data/rl_v2_runs/` | Local, Git-ignored training and evaluation outputs |
| `development/archive/docs/`, `development/archive/python/` | Historical documentation and standalone tools |
| `development/local/` | Ignored local diagnostics and backups |
| `rankedsurfers-downloaded/` | Local Unity WebGL game assets |

**Important:** Several root files beginning with `v2`, `v3`, `v4`, or `color` are still imported by the current runtime. Do not move or remove them without checking transitive imports and rerunning tests. Do not refactor the active pipeline in the middle of a training experiment.

## Windows setup

Run commands in **Windows PowerShell**, from the repository root:

```powershell
cd "$HOME\Documents\ranked-surfers-ai"
```

A working `.venv` and required model files are assumed. To verify essential packages and hardware:

```powershell
.\.venv\Scripts\python.exe -c "import torch, stable_baselines3, ultralytics, playwright, mss; print('CUDA:', torch.cuda.is_available())"
```

To restore Git LFS objects in an existing clone, when needed:

```powershell
git lfs install
git lfs pull
```

### Start the local game server

Open a **separate PowerShell terminal**, and leave it running:

```powershell
cd "$HOME\Documents\ranked-surfers-ai"
.\.venv\Scripts\python.exe server.py
```

Open `http://localhost:8000` in the local Practice environment. Use the bundled `server.py` rather than a generic `python -m http.server`; it configures the response headers expected by Unity WebGL.

## Before training: stable-build checks

```powershell
cd "$HOME\Documents\ranked-surfers-ai"

.\.venv\Scripts\python.exe -m unittest test_rs_v612_guard test_rs_v61 -q
if ($LASTEXITCODE -ne 0) { throw "Unit tests failed" }

.\.venv\Scripts\python.exe train_rs_v612_ppo.py --mode check --resolution 160x96
if ($LASTEXITCODE -ne 0) { throw "PPO architecture check failed" }
```

The October 26 cleanup passed 48 tests and the offline PPO check. Re-run them after code or import-path changes; no live game control is required for these checks.

## Continue training for four hours (5 ms)

Keep the local server running and **do not run two training/evaluation processes at the same time**. This command continues from the saved 120-minute checkpoint without modifying architecture, rewards, perception or guard thresholds.

```powershell
cd "$HOME\Documents\ranked-surfers-ai"

$candidate = "ai_data\v4_screen_states_rebuild_01\screen_state_candidate_v2.pt"
$detector   = "ai_data\v5_hazards\runs\segment_round04\weights\best.pt"
$checkpoint = "ai_data\rl_v2_runs\v612_resume_120min_press5_20261026_133230\latest.zip"

foreach ($file in @($candidate, $detector, $checkpoint)) {
    if (-not (Test-Path $file)) { throw "Missing required file: $file" }
}

$runName = "v612_extended_240min_press5_" + (Get-Date -Format "yyyyMMdd_HHmmss")
Write-Host "Resuming from: $checkpoint"
Write-Host "Writing run: $runName"

.\.venv\Scripts\python.exe -u train_rs_v612_ppo.py `
  --mode train `
  --run-name $runName `
  --resume $checkpoint `
  --steps 1048576 `
  --max-minutes 240 `
  --resolution 512x288 `
  --n-steps 128 `
  --batch-size 32 `
  --n-epochs 5 `
  --fps 30 `
  --rate-mode uncapped `
  --press-delay-ms 5 `
  --capture-mode fast `
  --resize-method linear `
  --state-model $candidate `
  --state-confidence 0.90 `
  --detector $detector `
  --imgsz 960 `
  --conf 0.20 `
  --detect-every 3 `
  --save-every-minutes 4 `
  2>&1 | Tee-Object -FilePath "ai_data\$runName.log"
```

The step ceiling is intentionally higher than the likely experience collected in four hours; the normal stopping condition should be the **240-minute time limit**. This generally leaves `completed_requested_steps: false` in the JSON report, which is expected. The guard may stop earlier if game state cannot be verified; **never disable the guard just to keep a run going**. Saved checkpoints do not imply a successful or completed run unless the report and guard state agree.

During training, keep Windows awake and unlocked, and the browser/game visible and unobstructed for screen capture. Do not change the display layout or minimize the game.

## Inspect a training run

Run this in the **same PowerShell session** that set `$runName` above, or set `$runName` to the saved run folder's name:

```powershell
$runDir = "ai_data\rl_v2_runs\$runName"
$r = Get-Content "$runDir\v61_report.json" -Raw | ConvertFrom-Json
$r | Select-Object steps, episodes, wall_seconds, mean_survival_seconds, `
  max_survival_seconds, first_25_mean_seconds, last_25_mean_seconds, `
  avg_steps_per_sec_including_resets, completed_requested_steps | Format-List
$r.detector.v612_live_input_guard | Format-List
Get-Item "$runDir\latest.zip" | Select-Object FullName, Length, LastWriteTime
```

Check that `guard_hard_stopped` is `false` and `uncertain_state_abort` is `0` before treating the checkpoint as a normal finished continuation.

## Evaluate a trained checkpoint: 100 deterministic episodes

Use exactly the same perception and 5 ms input configuration used for the prior 100-episode evaluation. This example finds the newest four-hour checkpoint, **including potentially interrupted runs**; verify its training report before evaluating it.

```powershell
cd "$HOME\Documents\ranked-surfers-ai"

$candidate = "ai_data\v4_screen_states_rebuild_01\screen_state_candidate_v2.pt"
$detector  = "ai_data\v5_hazards\runs\segment_round04\weights\best.pt"

$latestRun = Get-ChildItem "ai_data\rl_v2_runs" -Directory |
  Where-Object {
    $_.Name -like "v612_extended_240min_press5_*" -and
    (Test-Path (Join-Path $_.FullName "latest.zip"))
  } |
  Sort-Object LastWriteTime -Descending |
  Select-Object -First 1

if (-not $latestRun) { throw "No four-hour-run checkpoint found" }

$checkpoint = Join-Path $latestRun.FullName "latest.zip"
$evalName = "v612_eval_after_240min_press5_" + (Get-Date -Format "yyyyMMdd_HHmmss")

.\.venv\Scripts\python.exe -u train_rs_v612_ppo.py `
  --mode eval `
  --run-name $evalName `
  --resume $checkpoint `
  --eval-policy deterministic `
  --eval-episodes 100 `
  --max-minutes 35 `
  --resolution 512x288 `
  --fps 30 `
  --rate-mode uncapped `
  --press-delay-ms 5 `
  --capture-mode fast `
  --resize-method linear `
  --state-model $candidate `
  --state-confidence 0.90 `
  --detector $detector `
  --imgsz 960 `
  --conf 0.20 `
  --detect-every 3 `
  2>&1 | Tee-Object -FilePath "ai_data\$evalName.log"
```

Summarize the saved result:

```powershell
$e = Get-Content "ai_data\rl_v2_runs\$evalName\v61_eval_report.json" -Raw |
  ConvertFrom-Json
$durations = @($e.durations_seconds)

[PSCustomObject]@{
    Episodes     = $e.eval_episodes
    MeanSeconds  = $e.mean_survival_seconds
    MedianSeconds = $e.median_survival_seconds
    BestSeconds  = $e.best_survival_seconds
    Survived5s   = @($durations | Where-Object { $_ -ge 5 }).Count
    Survived10s  = @($durations | Where-Object { $_ -ge 10 }).Count
    GuardAborts  = $e.detector.v612_live_input_guard.uncertain_state_abort
} | Format-List
```

Compare the new checkpoint with the already saved **3.84-second / 100-episode** trained-policy reference in `reports/v612_learning_proof_20261026_153745/after.json`. A fresh no-action benchmark is not required for every training milestone.

## Results, backups and version control

- Training runs: `ai_data/rl_v2_runs/<run-name>/latest.zip`, `v61_report.json`, and `progress.jsonl`.
- Evaluation runs: `ai_data/rl_v2_runs/<eval-name>/v61_eval_report.json`.
- The `ai_data/rl_v2_runs/` tree is **Git-ignored**. A normal `git push` does not automatically back it up.
- Copy **selected milestone checkpoints** into `checkpoints/` and **selected JSON reports** into `reports/`, then commit and push those explicit copies.
- `.gitattributes` tracks `.pt` and `.zip` files with **Git LFS**; keep meaningful model checkpoints rather than every four-minute autosave.
- `development/archive/` contains historical scripts and docs. `development/local/` is for ignored local diagnostics; never `git add -f` its contents unintentionally.

## Safety and reproducibility

The gameplay guard is designed to block action when it cannot verify gameplay on the most recent visual evidence; it cannot guarantee that the game state will not change between observation and keypress. Limit automation to the **local Practice** environment. Model performance, throughput, and survival vary across procedural levels; sample averages are not guarantees. The native trainer defaults to a **10 ms** press delay, so pass `--press-delay-ms 5` explicitly to retain the measured operating configuration. Avoid changing core controller imports, rewards, detector, guard or PPO architecture immediately before a long run.

The downloaded game assets remain in `rankedsurfers-downloaded/` because the local server needs them. Confirm you have permission to redistribute those assets if the repository is made public.
