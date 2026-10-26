# Ranked Surfers V2 — verify before recording demonstrations

This package is for **LOCAL Practice at http://localhost:8000 only**.
Keep your previous V2 scripts and checkpoints safe. This is a fresh version
whose model input is 4 x RGB 256x144 = 12x144x256.

## Installation

Unzip the package directly into `~/Documents/rankedsurflocal/` alongside the
existing dependencies:

- rs_practice_controller.py
- train_rs_rl_early_death.py
- train_rs_color_human_named.py
- train_rs_color_fast.py
- train_rs_color_quick_restart.py
- train_rs_color_anchor_cache.py

Activate your existing `.venv` and run your local HTTP server as normal.

These new files are in the archive:

- `train_rs_v2_verify.py` — fullscreen startup **R-first**, smoke/verify/collect/train
- `verify_rs_v2_offline.py` — no game browser; tests synthetic imitation **and** PPO
- `evaluate_rs_v2_verify.py` — local Practice evaluator using R-first fullscreen
- `rs_v2_vision.py` — image conversions
- `inspect_rs_v2.py` — only needed later if collecting demos

## 1. Fullscreen startup/restart smoke test (NO model training)

```bash
cd ~/Documents/rankedsurflocal
source .venv/bin/activate
python -u train_rs_v2_verify.py --mode smoke --fullscreen \
  --run-name v2_r_smoke --max-minutes 2 --episode-max-seconds 12 \
  2>&1 | tee ai_data/v2_r_smoke.log
```

Look for `[SCREEN] Entering SITE fullscreen before the first R`,
`[RESTART] R succeeded for episode 1`, multiple episode restarts, and an
`actual_model_input.png` in `ai_data/rl_v2_runs/v2_r_smoke/`.
If R fails, it retries three times, exits fullscreen, uses the original HTML
Practice button, then tries to return to site fullscreen. **Stop and report**
any repeated fallbacks or missing game input screenshots.

## 2. Offline learning sanity test (NO browser, NO human demos)

```bash
python -u verify_rs_v2_offline.py --updates 100 --ppo-steps 1536 \
  2>&1 | tee ai_data/v2_offline_test.log
```

This generates fake images with five color-coded markers. It tests:

1. Disk writing/loading of the V2 .npz demonstration shape, including
   train/validation partitions.
2. A CNN's supervised updates on artificial labeled frames; compares losses
   and checks that weights actually change.
3. Independently initializes PPO **without imitation** and gives it an
   image-dependent synthetic reward task. It compares before/after accuracy.
4. Checkpoint save and load preserve action probabilities.

Read `ai_data/v2_pipeline_offline.json`.
A pass confirms technical learning/gradient flow on artificial images only —
it does NOT prove skill at Ranked Surfers. If synthetic PPO does not improve,
try `--ppo-steps 4096` before assuming the architecture cannot learn.

## 3. Live, demonstration-free PPO end-to-end verification

```bash
python -u train_rs_v2_verify.py --mode verify --fullscreen \
  --run-name v2_pipeline_check --steps 512 --max-minutes 6 \
  --fps 20 --episode-max-seconds 12 \
  2>&1 | tee ai_data/v2_pipeline_check.log
```

This **automatically disables** `--pretrain-updates` and `--bc-updates`.
It deliberately creates a temporary fresh PPO model. It captures actual
fullscreen gameplay and runs actual PPO updates on the game reward.
It records action counts, rewards, varied observations, changes in both
visual CNN weights and action-head weights, and saved checkpoint reload
compatibility. Results in:

- `ai_data/rl_v2_runs/v2_pipeline_check/pipeline_verification.json`
- `ai_data/rl_v2_runs/v2_pipeline_check/latest.zip`
- `ai_data/rl_v2_runs/v2_pipeline_check/audits/`
- `ai_data/rl_v2_runs/v2_pipeline_check/episodes.monitor.csv`

The temporary policy is NOT the final model and is not expected to perform
well after 512 steps. A `[VERIFY] PASS` means that the mechanical loop works,
not that it understands obstacles or coins.

After stopping, print the report with:

```bash
python -m json.tool ai_data/rl_v2_runs/v2_pipeline_check/pipeline_verification.json
```

The game's site fullscreen should remain visible and unobstructed during
steps 1 and 3. DO NOT use Chrome F11. `R` is sent through Playwright.

## 4. Only after verification, consider demonstration collection

```bash
python -u train_rs_v2_verify.py --mode collect --fullscreen \
  --run-name v2_collect --max-minutes 20 --human-fps 25 \
  2>&1 | tee ai_data/v2_collect.log
python inspect_rs_v2.py
```

Wait to do this until steps 1–3 pass and you inspect timing/policy audit data.

## Limitations

The synthetic test does not validate the game reward design or that the CNN
recognizes actual coins and trains. The live test confirms learning updates
happen, but cannot guarantee improvement from survival-only rewards. The coin
detector remains diagnostic-only, because its false positives have not been
fully audited. Obstacle-avoidance reward shaping needs separate validation.
