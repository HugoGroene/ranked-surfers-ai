# V4: Screen state recognition / safe Practice startup (Windows)

This is **separate** from object detection (YOLO) and PPO. YOLO detects coins/trains; this model classifies the **entire game screen** as MENU / LOADING / GAMEPLAY / optional OTHER. A controller state machine—not PPO—chooses when to press `R`. Future Ranked UI automation can add explicit states and action sequences without changing PPO.

## Why this is different

The previous startup guard failed on Windows because it assumed a blue pause icon / yellow distance text appears in hardcoded screen coordinates. It did not. Screen-state inference is trained on **actual Windows screenshots** instead. Until trained and independently verified, the updated controller will refuse to run.

## Installation

Copy **both** scripts from `Downloads` into the project. In PowerShell:

```powershell
cd "$HOME\Documents\ranked-surfers-ai"
Copy-Item "$HOME\Downloads\rs_v4_screen_states.py" . -Force
Copy-Item "$HOME\Downloads\rs_v4_low_latency_benchmark_v4.py" . -Force
```

Use the existing working venv (`.\.venv\Scripts\python.exe`). Requirements are already covered by PyTorch, OpenCV, NumPy, MSS, and SB3.

## A. Record real screens (human manually operates local Practice)

Start the local game and get it into fullscreen. Then run:

```powershell
.\.venv\Scripts\python.exe rs_v4_screen_states.py record --session sample_01 --seconds 30 --fps 10 --delay 5
```

During the 5-second countdown, focus the browser. Keep the character on the menu for ~2 seconds, press `R` yourself, wait through loading, let the character run/die, pause on the menu a moment, and press `R` again. Capture **a mixture of all three screens**. This script takes screenshots but never sends inputs.

View images in `ai_data\v4_screen_states\raw\sample_01` to ensure they contain the game.

Repeat twice as `sample_02` and `sample_03` on fresh game attempts. **At least two independent sessions with each class** are required; three are recommended. If a session is all menu, repeat it while manually pressing `R`. Source image size is limited to 960 pixels in width because these are for state classification, **not** YOLO object detector training.

## B. Human label each session

```powershell
.\.venv\Scripts\python.exe rs_v4_screen_states.py label --session ai_data\v4_screen_states\raw\sample_01
.\.venv\Scripts\python.exe rs_v4_screen_states.py label --session ai_data\v4_screen_states\raw\sample_02
.\.venv\Scripts\python.exe rs_v4_screen_states.py label --session ai_data\v4_screen_states\raw\sample_03
```

OpenCV will show an image. Use `M` = visible **FIND RANKED MATCH** menu ready for R; `L` = game loading/transition; `G` = clearly **active gameplay**; `O` = other/uncertain screens; `K` = skip; `B` = back; `Q` = save/quit. After labeling a frame, `3`, `5`, or `0` can label the **next** 3/5/10 frames with the same class. Only bulk-label a segment if you checked it has no transitions.

If a frame could be either loading or gameplay, **skip** rather than guess. Don't label the menu's animated disappearance as GAMEPLAY.

The labeler stores human decisions in `raw\sample_XX\labels.csv` and copies labeled screenshots to `dataset\CLASS\sample_XX`. Re-running the labeler is supported; it removes stale copies of frames that you previously labeled differently in that same session.

## C. Train screen classifier

```powershell
.\.venv\Scripts\python.exe -u rs_v4_screen_states.py train --epochs 20 2>&1 | Tee-Object -FilePath ai_data\v4_state_train.log
```

It holds out an **entire recording session**, not random adjacent video frames. If one class has too few examples or is missing from the validation session it will refuse to train. The `OTHER` class is optional; you can skip ambiguous images until you have enough `OTHER` examples across multiple sessions.

Check `ai_data\v4_screen_states\screen_state_best.json` for held-out confusion matrix. Aim for **zero `LOADING`/`MENU` → `GAMEPLAY` mistakes** before allowing automated action. Note that high validation accuracy alone is insufficient if sessions are too similar.

## D. Live WATCH test with **no automated inputs**

```powershell
.\.venv\Scripts\python.exe -u rs_v4_screen_states.py watch --seconds 45 --fps 10 --delay 5 --verbose 2>&1 | Tee-Object -FilePath ai_data\v4_state_watch.log
```

Switch to the local game during countdown, manually press `R` and observe the predictions. Confirm MENU→LOADING→GAMEPLAY→MENU on at least a few attempts. If the classifier announces GAMEPLAY while the screen is loading or on the menu, **DO NOT** run the automated benchmark; add examples and retrain. Use `--monitor 2` for a different monitor.

## E. Integrated short V4 benchmark (only AFTER section D)

```powershell
.\.venv\Scripts\python.exe -u rs_v4_low_latency_benchmark_v4.py live `
  --resolution 512x288 `
  --fps 30 `
  --rate-mode uncapped `
  --crop 0 0.055 1 0.965 `
  --steps 128 `
  --n-steps 128 `
  --batch-size 32 `
  --n-epochs 2 `
  --capture-mode fast `
  --resize-method area `
  --torch-threads 1 `
  --state-model ai_data\v4_screen_states\screen_state_best.pt `
  --state-confidence 0.90 `
  --menu-ready-timeout 30 `
  --startup-timeout 8 `
  --startup-grace 0.10 `
  --restart-grace 0.10 `
  --detector checkpoints\yolo_hazards_best.pt `
  --detect-every 3 `
  --press-delay-ms 20 `
  --run-name v4_screen_state_smoke_01 `
  2>&1 | Tee-Object -FilePath ai_data\v4_screen_state_smoke_01.log
```

Only use on local Practice. The controller waits for `MENU` confirmed by classifier AND existing Windows menu template before sending R; then requires **two confident consecutive GAMEPLAY** predictions with the menu absent. It stops instead of pressing keys on ambiguous states. No models are overwritten.

Diagnostic images: `ai_data\rl_v2_runs\v4_screen_state_smoke_01\state_ep0001_menu_ready.jpg`, `state_ep0001_gameplay.jpg`, and `state_trace_ep0001\`. A gameplay screenshot does **not** exist if the safety check fails. Do not interpret missing screenshot as a separate error.

## Future ranked mode

Keep a separate explicit UI state/action orchestration system (MENU, MODE_SELECT, QUEUE, READY, LOADING, GAMEPLAY, RESULTS). The orchestrator can click a mode, wait for the verified new state, and retry/abort safely. **Do not** let PPO choose menu buttons. Any live Ranked workflows require a separately tested set of screenshots and transitions; this package controls only local Practice with `R`.

## Limitations / test status

- Tested: Python syntax, crop preview, synthetic supervised classifier training + held-out evaluation + inference, and CLI argument parsing.
- Not tested: actual Windows Unity gameplay, state generalization on your game's menu/loading/gameplay, or real controller latency. You must complete the watch and short live checks.
- Native screenshots vary with Windows scaling, fullscreen and monitor. Use the same monitor and fullscreen configuration for capture, training, watch, and gameplay.
- The state classifier is not a replacement for obstacle YOLO. PPO still uses RGB+YOLO for hazard information.
