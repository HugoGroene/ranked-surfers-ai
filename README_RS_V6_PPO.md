# V6 — PPO learns by itself from RGB history + YOLO V5.4 segmentation

**Local Practice only. No demonstration learning, no fixed jump action, no live ranked website.**

This is a NEW trainer. It reuses the tested V4 Chromium/menu/loading/restart controller and V5.4 async YOLO code without editing them. It trains a new SB3 PPO model and saves `latest.zip`, progress logs, and a run report. V3/V4 PPO checkpoints are **not** observation-compatible with V6; start V6 from scratch, then resume **V6** checkpoints.

## Observation and action design

- `image`: 4 recent actual **512×288 RGB frames**, 12 channels CHW; **PPO still sees the game pixels directly**.
- `aux`: 467 normalized numerical values: 442 from nine V5 segmentation classes, 20 last-four-actions one-hot features, three capture-interval features, episode time and YOLO age.
- Hazard format: 9 classes × 3 **image-height regions** (upper/middle/lower) × 2 visible objects per class/region × 8 values (box, confidence, mask area, box area, present), plus per-class counts and a freshness flag. Regions **aren't actual track lanes or calibrated distances**; YOLO doesn't choose movements.
- Old or previous-episode YOLO predictions are zeroed and explicitly flagged as stale. Async inference is scheduled every three observations by default, matching the previously measured ~64 ms median hazard age.
- Actions from PPO: `NONE`, `LEFT`, `RIGHT`, `JUMP`, `ROLL`. PPO's sampled action is **the one executed**. No demonstration data, no hard-coded obstacle responses, no action override. Repeated actions can yield quick sequences, but learned advanced maneuvers are **not guaranteed**.

### Reward function (action-neutral)

By default per environment step:

`reward = 0.4 × min(actual elapsed gameplay seconds, 0.15) + 0.5 × newly passed 2-second milestones − 1.2 if confirmed dead`

Rewards are based on observed local-game survival/death, **not YOLO confidence, coins detected, object avoidance guesses, movement type, or a presumed safe lane**. The 0.15-s cap prevents capture/menu stalls from generating excessive single-step reward. No reward for collecting coins (coins are fatal in this challenge). Long runs naturally earn more reward. All coefficients are CLI-adjustable for later experiments. Watchdog truncations are **not** counted as confirmed deaths; examine their screenshots.

## Install

Extract `rs_v6_ppo_package.zip` into the root of your existing repository, without removing old scripts:

```powershell
cd "$HOME\Documents\ranked-surfers-ai"
Expand-Archive "$HOME\Downloads\rs_v6_ppo_package.zip" -DestinationPath . -Force
```

Requires the earlier working `rs_v5_segmentation_live_benchmark.py`, `rs_v5_shadow_segmentation.py`, `rs_v4_screen_states.py`, `train_rs_v3_ppo.py`, `train_rs_v2_verify.py`, `rs_practice_controller.py` and their existing dependencies. Your `.venv` already has SB3, Ultralytics, GPU PyTorch, Gymnasium, Playwright, MSS, OpenCV. Both of these should be `True`:

```powershell
Test-Path 'ai_data\v5_hazards\runs\segment_round04\weights\best.pt'
Test-Path 'ai_data\v4_screen_states\screen_state_best.pt'
```

## 1. Offline PPO architecture check — no game needed

```powershell
.\.venv\Scripts\python.exe -u train_rs_v6_ppo.py --mode check --resolution 160x96
```

This checks nine-class encoding, age/episode gating, 12-channel RGB + 467-value auxiliary observations, an actual PPO learning update, and model save/reload. It isn't a skill test. This particular test **must pass before starting live training**.

## 2. Start local Practice server

In a separate PowerShell window (leave open):

```powershell
cd "$HOME\Documents\ranked-surfers-ai"
.\.venv\Scripts\python.exe -m http.server 8000 --directory .\rankedsurfers-downloaded
```

The trainer opens its own Chromium and activates the same site fullscreen as V4. **Don't use Chrome F11; don't obstruct or minimize Chromium.**

## 3. First real 128-step PPO smoke test

```powershell
.\.venv\Scripts\python.exe -u train_rs_v6_ppo.py --mode train `
  --run-name v6_smoke_128 `
  --steps 128 --max-minutes 5 `
  --resolution 512x288 --n-steps 128 --batch-size 32 --n-epochs 3 `
  --fps 30 --rate-mode uncapped `
  --press-delay-ms 5 --capture-mode fast --resize-method linear `
  --state-model ai_data\v4_screen_states\screen_state_best.pt `
  --detector ai_data\v5_hazards\runs\segment_round04\weights\best.pt `
  --imgsz 960 --conf 0.20 --detect-every 3 `
  2>&1 | Tee-Object -FilePath ai_data\v6_smoke_128.log
```

What to check: `[SCREEN STATE] GAMEPLAY_CONFIRMED`, actions include a **mix** rather than only JUMP, `[V6 PROGRESS]`, one PPO update, `[V6] Saved model`. The actual PPO may die quickly; this is expected early on. A 5-ms tap is fast but **input reliability has not been fully verified**—try `--press-delay-ms 10` if lane changes/jumps appear missed.

Outputs in `ai_data\rl_v2_runs\v6_smoke_128\`:

- `latest.zip`: saved PPO policy/critic (not a YOLO model)
- `progress.jsonl`: periodic episode lengths, action distribution, reward parts
- `episodes.monitor.csv`: SB3 episode returns/lengths
- `v6_report.json`: real decision interval, YOLO freshness, action counts, reset failures
- `matched_detection_previews`: same-frame segmentation overlays when enabled

## 4. Initial autonomous PPO training run

After the smoke test passes, start a new run with more time:

```powershell
.\.venv\Scripts\python.exe -u train_rs_v6_ppo.py --mode train `
  --run-name v6_learning_01 `
  --steps 6000 --max-minutes 45 `
  --resolution 512x288 --n-steps 128 --batch-size 32 --n-epochs 3 `
  --fps 30 --rate-mode uncapped `
  --press-delay-ms 5 --capture-mode fast --resize-method linear `
  --detector ai_data\v5_hazards\runs\segment_round04\weights\best.pt `
  --state-model ai_data\v4_screen_states\screen_state_best.pt `
  --imgsz 960 --conf 0.20 --detect-every 3 `
  --save-every-minutes 4 --log-every-seconds 30 `
  2>&1 | Tee-Object -FilePath ai_data\v6_learning_01.log
```

The time limit may stop the run before it reaches `--steps`. PPO throughput includes repeated 1.5-second restarts when the initial random policy dies. `latest.zip` is still saved on ordinary completion, time limit, or Ctrl+C. Don't judge learning from a few hundred steps: log survival and action distributions over multiple runs. Avoid changing rewards mid-experiment if you want clean comparisons.

### Resume from the V6 checkpoint

Specify a **new** run name and existing `latest.zip`; you can't resume old V3 models due to the new aux shape:

```powershell
.\.venv\Scripts\python.exe -u train_rs_v6_ppo.py --mode train `
  --run-name v6_learning_02 `
  --resume ai_data\rl_v2_runs\v6_learning_01\latest.zip `
  --steps 6000 --max-minutes 45 `
  --resolution 512x288 --n-steps 128 --batch-size 32 --n-epochs 3 `
  --press-delay-ms 5
```

A loaded checkpoint retains its saved PPO learning-rate/rollout hyperparameters; the trainer reports that if CLI rollout/batch settings don't match. The **environment observation schema and image resolution must match**. Keep the original model, rewards and checkpoints for reproducibility.

## 5. Evaluate the learned policy (no PPO updates)

```powershell
.\.venv\Scripts\python.exe -u train_rs_v6_ppo.py --mode eval `
  --run-name v6_learning_01_eval `
  --resume ai_data\rl_v2_runs\v6_learning_01\latest.zip `
  --resolution 512x288 --eval-episodes 20 --max-minutes 15 `
  --press-delay-ms 5
```

Saves `v6_eval_report.json` with durations and action counts. Evaluation uses deterministic actions by default; `--stochastic-eval` optionally samples. Compare **matched evaluation settings and number of episodes**, not just a training-episode record.

## 6. Inspect progress in PowerShell

```powershell
$r = Get-Content 'ai_data\rl_v2_runs\v6_learning_01\v6_report.json' -Raw | ConvertFrom-Json
$r.steps
$r.episodes
$r.mean_survival_seconds
$r.max_survival_seconds
$r.action_counts
$r.gameplay_only_intervals_ms
$r.detector.age_at_decision_ms
$r.detector.fresh_fraction
$r.detector.v5_segmentation.hazard_obs_nonzero

Get-Content 'ai_data\rl_v2_runs\v6_learning_01\progress.jsonl' -Tail 5
```

Red flags: action collapse to repeated jump/no-op, many bad screen-state restarts, stale detections for most observations, rewards rising but episode survival not rising. Survival improvements need repeatable evaluations because starting obstacles vary.

## Limitations

- Running this once does **not** prove PPO can master the game. Pure on-policy PPO may still be sample-inefficient with frequent early deaths.
- The numeric YOLO representation keeps the two bottom-most detections per class and image band, plus counts; it is intentionally bounded and may omit dense coin lines. The **raw RGB frames remain visible to PPO**.
- Polygon mask area is encoded, not the full raster mask. Running YOLO mask segmentation directly into the CNN would be a different and more expensive architecture.
- Image-height bands are *not* map lanes. PPO learns whether train roof/ramp/barrier contacts are useful or dangerous through consequences, not scripted collision rules.
- Rewards are intentionally action-neutral. We don't yet infer exact distance, speed, position, or collision geometry from the Unity state.
- We have tested Python syntax and synthetic feature/reward behavior locally. **Full SB3 training and Windows Chromium/game execution must be verified on your own PC**; the sandbox here lacks Gymnasium/SB3 and your locally served game.
