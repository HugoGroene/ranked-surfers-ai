# Ranked Surfers V6.1 — local PPO learning diagnostics and improvements

**Run ONLY against local Practice at `http://localhost:8000`.** No scripted hazard avoidance, no demonstration learning, and no modification to the WebGL game. This update is a **new trainer**; it leaves `train_rs_v6_ppo.py` and all V4/V5 scripts untouched.

## Why V6.1

Your V6 evaluation (25 deterministic episodes) selected LEFT on all 333 steps and survived 0.70 seconds on average. The stochastic run mixed five actions and averaged 0.96 seconds, so the previous policy didn't demonstrate reliable obstacle avoidance. This version prioritizes **input verification, observation clarity, reward feedback, probability diagnostics, and matched baselines** before long training.

## Files

- `train_rs_v61_ppo.py` — new command-line trainer, evaluator, and real-game input probe
- `rs_v61_features.py` — preserves four RGB frames plus all nine YOLO segmentation features, adds **81 screen-column measurements**, uncertain commanded-lane context, and action timing; `aux` = **557** vs V6's 467
- `rs_v61_rewards.py` — action-neutral dense survival reward with frequent verified-time milestones and confirmed-death penalty
- `test_rs_v61.py` — seven unit tests for feature dimensions, class ordering, stale/episode gating, lane uncertainty, reward bookkeeping and no repeated-duration farming

Requires the existing V6 and V4/V5/V3 project files, especially `rs_v6_ppo_features.py`. Install the four Python scripts plus this README in the repository root. No additional Python packages required beyond your working V6 `.venv`.

**V6 PPO checkpoints cannot be resumed as V6.1 checkpoints:** the auxiliary observation changed size. Keep V6's `latest.zip` as a baseline. Train a new V6.1 checkpoint.

## 0. Run offline validation (no browser or server)

```powershell
cd "$HOME\Documents\ranked-surfers-ai"
.\.venv\Scripts\python.exe -m unittest -v test_rs_v61.py
.\.venv\Scripts\python.exe -u train_rs_v61_ppo.py --mode check --resolution 160x96
```

`--mode check` additionally verifies a real SB3 PPO gradient update and checkpoint round-trip. Unit tests and `--help` were exercised in the packaging environment, but full SB3 and live browser checks **must** run in your installed Windows environment.

## 1. Start the **local** game server

In its own PowerShell window:

```powershell
cd "$HOME\Documents\ranked-surfers-ai"
.\.venv\Scripts\python.exe -m http.server 8000 --directory .\rankedsurfers-downloaded
```

Keep the trainer-controlled Chromium fullscreen and unobstructed, exactly as with V6.

## 2. Input reliability probe — run before training

```powershell
.\.venv\Scripts\python.exe -u train_rs_v61_ppo.py `
  --mode probe --run-name v61_probe_01 `
  --probe-trials 1 --probe-delays-ms 5,10,20 `
  --resolution 512x288 --fps 30 --rate-mode uncapped `
  --capture-mode fast --resize-method linear `
  --state-model ai_data\v4_screen_states\screen_state_best.pt `
  --detector ai_data\v5_hazards\runs\segment_round04\weights\best.pt `
  --imgsz 960 --conf 0.20 --detect-every 3 `
  2>&1 | Tee-Object -FilePath ai_data\v61_probe_01.log
```

Probe outputs under `ai_data\rl_v2_runs\v61_probe_01\`:

- `v61_input_probe.json`: requested commands, returned keyboard **keydown/keyup** counts (when event hooks work), terminal status, image filename
- `action_pairs\*.jpg`: real RGB **before/after** screenshots for LEFT, RIGHT, JUMP, ROLL and LEFT-TWICE using 5-, 10- and 20-ms taps.

**Important:** a browser keyboard event does **not** prove that Unity accepted the command. Confirm lane change or jump/roll **visually** from images/live view. Also, immediately after pressing a key an animation may not be finished; look at the full result before concluding it missed. Early death or obstacles can make a trial uninformative. The controller waits for confirmed death before restart; it refuses unsafe resets from active gameplay.

Compare delivery + visual success rate of 5 vs 10 vs 20 ms. Pick the shortest **reliable** hold. For training, default is 10 ms, but set `--press-delay-ms 20` if 10-ms taps visibly fail.

## 3. First real V6.1 PPO smoke (128 steps)

```powershell
.\.venv\Scripts\python.exe -u train_rs_v61_ppo.py `
  --mode train --run-name v61_smoke_128 --steps 128 --max-minutes 5 `
  --resolution 512x288 --n-steps 128 --batch-size 32 --n-epochs 5 `
  --fps 30 --rate-mode uncapped --press-delay-ms 10 `
  --capture-mode fast --resize-method linear `
  --state-model ai_data\v4_screen_states\screen_state_best.pt `
  --detector ai_data\v5_hazards\runs\segment_round04\weights\best.pt `
  --imgsz 960 --conf 0.20 --detect-every 3 `
  2>&1 | Tee-Object -FilePath ai_data\v61_smoke_128.log
```

The script still sends the PPO-selected actions unmodified through the existing V4 controller. It never rewards YOLO confidence, moving sideways, seeing obstacles, or guessed safe lanes. No hardcoded obstacle-avoidance policy is introduced.

## 4. Measured V6.1 training after smoke/probe passes

```powershell
.\.venv\Scripts\python.exe -u train_rs_v61_ppo.py `
  --mode train --run-name v61_learning_01 `
  --steps 4096 --max-minutes 20 `
  --resolution 512x288 --n-steps 128 --batch-size 32 --n-epochs 5 `
  --fps 30 --rate-mode uncapped --press-delay-ms 10 `
  --capture-mode fast --resize-method linear `
  --state-model ai_data\v4_screen_states\screen_state_best.pt `
  --detector ai_data\v5_hazards\runs\segment_round04\weights\best.pt `
  --imgsz 960 --conf 0.20 --detect-every 3 --save-every-minutes 4 `
  2>&1 | Tee-Object -FilePath ai_data\v61_learning_01.log
```

The `--steps` value MUST be divisible by `--n-steps` (4096 is 32×128). Stops at 4096 steps or 20 minutes, whichever happens first. Saved `latest.zip`, `v61_report.json`, `progress.jsonl` and `episodes.monitor.csv` are under `ai_data\rl_v2_runs\v61_learning_01\`.

PPO defaults modified from V6: entropy coefficient **0.002** (was 0.02), training epochs **5** (was 3), learning rate **0.0003** (was 0.00025). These are *experiments*, not established improvements. The progress report includes snapshot categorical policy probabilities and entropy, plus first/last-25 episode means. Actual PPO `approx_kl` and `clip_fraction` remain in normal SB3 logs. Neither the improved reward nor lower entropy guarantees learning.

## 5. Evaluate the checkpoint **and** simple baselines

Run these in succession with unique names, not in parallel on one desktop:

```powershell
# Deterministic learned policy
.\.venv\Scripts\python.exe -u train_rs_v61_ppo.py --mode eval `
  --run-name v61_eval_det_01 --eval-policy deterministic --eval-episodes 25 `
  --resume ai_data\rl_v2_runs\v61_learning_01\latest.zip `
  --max-minutes 10 --press-delay-ms 10

# Stochastic learned policy
.\.venv\Scripts\python.exe -u train_rs_v61_ppo.py --mode eval `
  --run-name v61_eval_sample_01 --eval-policy stochastic --eval-episodes 25 `
  --resume ai_data\rl_v2_runs\v61_learning_01\latest.zip `
  --max-minutes 10 --press-delay-ms 10

# Uniform-random ACTION baseline. No checkpoint required.
.\.venv\Scripts\python.exe -u train_rs_v61_ppo.py --mode eval `
  --run-name v61_eval_random_01 --eval-policy random --eval-episodes 25 `
  --max-minutes 10 --press-delay-ms 10

# Do-nothing baseline. No checkpoint required.
.\.venv\Scripts\python.exe -u train_rs_v61_ppo.py --mode eval `
  --run-name v61_eval_none_01 --eval-policy none --eval-episodes 25 `
  --max-minutes 10 --press-delay-ms 10
```

These use the default V5 detector checkpoint, screen classifier, resolution and input options; you can explicitly provide the same flags as training if changed. Outputs are `v61_eval_report.json` in each run folder. Record average/median/best survival, action mix, mean policy probabilities and probability of the top action. These baseline runs exercise the game but **never train or save a PPO policy**.

## Architecture / caveats

- **RGB:** 4 genuine RGB screenshots at 512×288; CNN branch unchanged.
- **V5:** Original 467-value aux remains intact, including all nine YOLO classes, polygon area and timing.
- **New screen-column summary:** 9 classes × 3 **image** columns × 3 geometric measurements. Values overlap columns based on detection boxes. These are image-space hints only, not projected real-world 3D lanes or collision probabilities. No new labels or YOLO retraining needed.
- **Player lane:** No reliable pixel-based player detector exists in the supplied V6 source. V6.1 records unverified commanded-lane and time-since-action; lane input **defaults to all zeros** instead of inventing the player's actual position. Optional `--assume-center-lane` adds low-confidence command odometry but is **not recommended** until action registration is demonstrated. It never masks or changes PPO actions.
- **Key presses:** We do not skip repeated LEFT/RIGHT actions, because that would silently corrupt PPO's (action, observation, reward) tuples. Probe checks actual Playwright events and saves frame pairs for visual review. Action duration is configured through existing `--press-delay-ms`.
- **Reward:** `+0.8 × elapsed gameplay seconds (capped at 0.15 per step) +0.15 every verified 0.5s survival milestone −1.0 on confirmed death`. No reward for YOLO predictions or action type. An optional distance-based reward is **not included**: a reliable, validated game-distance reading was not present in V6, and guessing distance from the screen would make training feedback unreliable.
- **No guarantee:** Training in a single asynchronous Unity environment with very short episodes is inherently challenging. Actual improvement must be demonstrated against uniform random and no-op policies under matched conditions.
- **Run duration:** The 20-minute limit is wall time. `--steps 4096` may finish sooner.
