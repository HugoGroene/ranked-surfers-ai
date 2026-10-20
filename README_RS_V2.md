# Ranked Surfers LOCAL Practice — V2 experiment

This experiment intentionally starts a **new** PPO model. The old 12×90×160
checkpoint and old demonstration files remain untouched.

### Install

Copy the following **four** new Python files into `~/Documents/rankedsurflocal/`:

- `train_rs_v2.py`
- `rs_v2_vision.py`
- `inspect_rs_v2.py`
- `evaluate_rs_v2.py`

The new trainer **imports your existing working controllers** in the same folder:
`train_rs_color_anchor_cache.py`, `train_rs_color_quick_restart.py`,
`train_rs_color_fast.py`, `train_rs_color_human_named.py`,
`train_rs_rl_early_death.py`, and `rs_practice_controller.py`.
Use your previously working Python virtual environment. Keep your local web
server serving `http://localhost:8000`.

**Do not start with an overnight run.** Fullscreen and `R` restart have not
been validated against your local running Unity game.

### 1. Smoke test: 3 minutes maximum

```bash
cd ~/Documents/rankedsurflocal
source .venv/bin/activate
python -u train_rs_v2.py --mode smoke --fullscreen \
  --run-name v2_smoke --max-minutes 3 \
  2>&1 | tee ai_data/v2_smoke.log
```

Watch for `[SCREEN] Site fullscreen ...`, `fullscreen=True`,
`[RESTART] R succeeded` after a death. The existing HTML Practice button
starts the **first** attempt; the trainer tries `R` after subsequent deaths.
If `R` does not work, the existing cached HTML Practice path is the fallback.

**After stopping**, open these:

```bash
xdg-open ai_data/rl_v2_runs/v2_smoke/capture_preview.jpg
xdg-open ai_data/rl_v2_runs/v2_smoke/actual_model_input.png
```

The second file is the **exact last RGB frame** of the observation sent to PPO,
256×144 pixels (four such frames are stacked as 12×144×256).
Confirm that coins, trains, and the player are visible. If the crop shows
Chrome tabs / desktop / blank pixels, **do not train**; report the
`[SCREEN]`, `[RESTART]`, and `[RL]` lines first.

If the automatic fullscreen click fails, you can click the site's **blue game
fullscreen icon** yourself, but the controller should be retested to verify
its screenshot crop afterward. Do not use F11.

If `R` never works, run with `--no-r-restart` to fall back to the working HTML
Practice button.

### 2. Fresh, timestamped demonstrations

```bash
python -u train_rs_v2.py --mode collect --fullscreen \
  --run-name v2_collect --max-minutes 20 \
  --human-fps 25 \
  2>&1 | tee ai_data/v2_collect.log
```

You control the character with the arrow keys. The script timestamps actual
browser keydowns and matches them with the most recent **prior** screenshot.
Samples with excessive timing ambiguity are skipped; if two different
movements occur between successive video frames, the later ambiguous label
is skipped rather than teaching contradictory actions for identical pixels. After each death it
attempts an `R` restart (with HTML fallback). The new recordings go in
`ai_data/rl_v2_demos/` and do not overwrite the old recordings.

Check the number of examples, labels and recording latency:

```bash
python inspect_rs_v2.py
xdg-open ai_data/v2_demo_preview.jpg
```

Aim initially for **~1,000 varied, correctly timed decisions**. Sample quality
matters more than raw quantity. Re-run `--mode collect` as often as needed.

### 3. New model, short 10-minute check

```bash
python -u train_rs_v2.py --mode train --fullscreen \
  --run-name v2_model --max-minutes 10 --steps 100000 \
  --fps 20 --demo-max 2200 --pretrain-updates 500 \
  --bc-updates 1 --bc-batch-size 32 \
  --milestone-seconds 2 --milestone-bonus 0.10 \
  --audit-seconds 30 --save-every-minutes 2 \
  2>&1 | tee ai_data/v2_training.log
```

The trainer prints imitation-learning loss every 100 updates. It also prints
an **episode-held-out** classification accuracy (only if enough distinct
recording files exist). It saves `after_pretraining.zip` and `latest.zip` in
`ai_data/rl_v2_runs/v2_model/`, plus screenshot/probability audits under
`v2_model/audits/`.

The reward gives small survival-time bonuses at 2-second progress milestones
and still penalizes death. It does **not** give a bonus for merely pressing an
arrow key. Candidate coin outlines are *diagnostic only*; they are too
error-prone to be trusted as rewards without more validation.

### 4. Evaluate (no training)

```bash
python evaluate_rs_v2.py --fullscreen \
  --checkpoint ai_data/rl_v2_runs/v2_model/latest.zip \
  --episodes 10 --policies model,noop --fps 20
```

This compares the deterministic model to doing nothing. Outputs are written
to `ai_data/rl_v2_evaluations/<timestamp>/summary.json` and `episodes.csv`.

### 5. Resume without repeating pretraining

```bash
python -u train_rs_v2.py --mode train --fullscreen \
  --run-name v2_more --resume ai_data/rl_v2_runs/v2_model/latest.zip \
  --max-minutes 60 --steps 250000 --fps 20 \
  --bc-updates 1 --audit-seconds 120 --save-every-minutes 5 \
  2>&1 | tee ai_data/v2_more.log
```

### Examine different screenshots *offline* after training

```bash
python inspect_rs_v2.py \
  --checkpoint ai_data/rl_v2_runs/v2_model/latest.zip \
  --screenshots /path/to/screenshot1.png /path/to/screenshot2.png
```

It writes `ai_data/v2_visual_audit/probabilities.csv` and
`input_audit.jpg`, compares real/frozen scenes with blank and flipped images,
and reports action-probability variation. This verifies *visual sensitivity*;
it **cannot** prove the resulting decisions are correct. For motion
sensitivity, use the real 4-frame gameplay traces in training audits.

### Limitations / safety checks

- The existing menu/death detector uses color-filtered text templates saved
  from the old screen. **Fullscreen may change its scores.** Do not leave the
  new program unattended before checking several natural deaths and early
  `R` restarts during a smoke test.
- Site fullscreen may require interacting with a particular blue icon. The
  click is based on accessible selectors and, if necessary, on the observed
  bottom-right control. The game's UI is not available to this environment
  for live end-to-end validation.
- MSS captures the **visible desktop**; do not minimize Chromium, cover it,
  let the screen blank, or lock the workstation.
- 256×144 observations use substantially more memory and compute than the old
  160×90 model; throughput may decline. Measure actual steps/second.
- Visually candidate coins can be confused with other yellow/orange things,
  particularly pumpkins and signs. Don't use candidate counts as rewards.
- PPO can plateau or forget demonstrations. A larger input and milestone
  rewards are experiments, **not guaranteed improvements**.
- This only runs the user's own **localhost Practice** environment.
