# Ranked Surfers V6.1.1 — live-input safety gate

This is a **safety patch** for local Practice runs. Keep the original V6.1 trainer and V4/V5 scripts as backups. The new CLI file is `train_rs_v611_ppo.py` so no old source files are overwritten. This package adds no game-play strategy and no change to the 557-element V6.1 observation or the reward architecture. It requires your installed V6.1 files (`rs_v61_features.py`, `rs_v61_rewards.py`) and all the existing V6/V5/V4 dependencies.

## Why this is needed

The V4/V5 restart guard requires two positive GAMEPLAY readings after pressing R, but during normal gameplay its inherited action loop mostly uses the **menu text template**, not the full-screen state classifier. If the template misses a menu frame, a subsequent PPO movement might be sent while the visible game is not running.

The attached log shows regular successful startup confirmations and menu detections for early episodes, **but it ends before the user-reported failure**. So this is a identified safety gap in code, not a proven root cause for the later observations.

## The change

- Every decision gets a fresh **full-screen capture immediately before the action**. Movement is allowed only when the learned classifier independently says `GAMEPLAY` with the configured confidence and the menu-text similarity remains low.
- `MENU`, `LOADING`, `OTHER`, and uncertain frames block all movement; two positive menu captures end the episode even when the old template is inconclusive.
- A transient observation must have **two later consecutive GAMEPLAY** confirmations before the requested action is executed. The controller does not secretly replace a PPO movement with `NONE` and then train on the result.
- If the UI cannot be reconfirmed, the process **aborts**, saves a full-screen `v611_guard_episode_NNNN.jpg` and JSON evidence, and the training wrapper saves a checkpoint. It does **not** repeatedly press R or blindly keep playing.
- The final report includes `detector.v611_live_input_guard` counts. Extra pre-action screen captures will slow gameplay decisions; correctness is the priority.

The added gate uses the *existing* trained screen classifier and template, which can both have mistakes. This reduces the observed risk but does **not** establish a mathematical guarantee that a keypress can never land after an asynchronous death transition. We still need a live smoke test.

## Install

```powershell
cd "$HOME\Documents\ranked-surfers-ai"
Expand-Archive "$HOME\Downloads\rs_v611_live_guard.zip" -DestinationPath . -Force
```

## Offline guard checks

```powershell
.\.venv\Scripts\python.exe -m unittest -v test_rs_v611_guard.py
.\.venv\Scripts\python.exe -u train_rs_v611_ppo.py --mode check --resolution 160x96
```

## Live guard smoke (128 steps)

Keep the existing local game HTTP server running and Chromium visible/unobstructed.

```powershell
.\.venv\Scripts\python.exe -u train_rs_v611_ppo.py `
  --mode train --run-name v611_guard_smoke_128 `
  --steps 128 --max-minutes 5 `
  --resolution 512x288 --n-steps 128 --batch-size 32 --n-epochs 5 `
  --fps 30 --rate-mode uncapped --press-delay-ms 10 `
  --capture-mode fast --resize-method linear `
  --state-model ai_data\v4_screen_states\screen_state_best.pt `
  --detector ai_data\v5_hazards\runs\segment_round04\weights\best.pt `
  --imgsz 960 --conf 0.20 --detect-every 3 `
  2>&1 | Tee-Object -FilePath ai_data\v611_guard_smoke_128.log
```

If the trainer stops with `[V611 GUARD] ABORT`, **do not disable the guard** or launch the long training experiment. Inspect the screenshot/JSON path printed in the error, plus the most recent terminal lines. Send those files for diagnosis. This may indicate a menu-frame recognition issue, incorrect crop, or a genuine transient. It is the intended safe behavior.

If the smoke passes without incorrect menu controls, run a 256- or 512-step guarded experiment next before attempting another 4096-step run. Do not resume the interrupted V6.1 checkpoint yet: its trajectories may contain UI misclassification, and the current focus is proving correct controller behavior.

## Offline verification performed in package environment

The 10 standalone screen-gate unit tests pass and the new trainer compiles. Real Playwright/Unity capture and full PPO gradient/checkpoint testing are not executable in this package environment and must be verified on the user's Windows project before further training.
