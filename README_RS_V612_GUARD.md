# Ranked Surfers V6.1.2 — robust menu/play pre-key guard

## Why this patch exists

Your V6.1.1 run safely blocked actions on the menu but aborted after 34 steps with
`states=UNCERTAIN,UNCERTAIN; menu_scores=0.57,1.00`. The second strong menu-template
score was not counted when the ML state classifier returned `UNCERTAIN`. That is
**a guard decision weakness**, not evidence that PPO learned badly or that YOLO failed.

V6.1.2 preserves the 557-feature PPO architecture and the existing V4/V5 state
classifier/controller. It changes only the conservative *pre-key screen-state
confirmation* and its diagnostics. It does not impose any hardcoded obstacle
avoidance. It does not automatically learn from demonstrations.

## Improvements

- Every requested PPO movement checks a fresh full-screen capture. Already
  confidently classified `GAMEPLAY` with low menu-template score allows the key
  immediately, unless the inherited controller already suspects a menu.
- When the first reading is ambiguous, gather up to **six independent frames**
  about 45 ms apart. Two menu indicators in the last three readings (strong
  template matches and/or high-confidence ML menu predictions) confirm `MENU`.
  A single template match **never** suffices to claim death.
- Two consecutive verified `GAMEPLAY` readings permit the requested action.
  If the old controller is still suppressing keys because of a stale menu
  suspicion, clear it **only after** these gameplay confirmations. This fixes
  a potential silent-no-op issue in inherited control.
- If six readings cannot confirm either state, **no key is sent**. The run
  stops, saves the current screenshot and six-reading JSON evidence, and
  saves any partial model/report **without a long Python traceback**.
- Capture the first six confirmed-menu samples for review. These are written
  to the unique run directory.

This is a fail-closed heuristic, not a guarantee that an unseen menu transition
cannot occur between a screen capture and the following keypress. Inspect the
menu screenshots after the smoke test; if a confirmed menu screenshot actually
shows gameplay, the template/classifier needs more calibration.

## Install

Download `rs_v612_guard_fix.zip` and extract its contents into
`C:\Users\Hugo\Documents\ranked-surfers-ai`. The archive contains only **new**
files and does **not** overwrite the V6.1, V6.1.1, V4 or V5 scripts.

Requirements: previously installed V6.1 and V4/V5 dependencies, working
`ai_data/v4_screen_states/screen_state_best.pt` and nine-class V5 model.

## Offline verification

```powershell
.\.venv\Scripts\python.exe -m unittest -v test_rs_v612_guard.py
.\.venv\Scripts\python.exe -u train_rs_v612_ppo.py --mode check --resolution 160x96
```

The guard tests require only Python stdlib and pass on the packaging host. Full
PPO/browser integration is **not tested here**; run the two commands above on
Windows before a live smoke test.

## Small live smoke test (128 PPO decisions)

Keep `http://localhost:8000/` hosted and Chromium fully visible and unobstructed.

```powershell
.\.venv\Scripts\python.exe -u train_rs_v612_ppo.py `
  --mode train `
  --run-name v612_guard_smoke_128 `
  --steps 128 `
  --max-minutes 5 `
  --resolution 512x288 `
  --n-steps 128 `
  --batch-size 32 `
  --n-epochs 5 `
  --fps 30 `
  --rate-mode uncapped `
  --press-delay-ms 10 `
  --capture-mode fast `
  --resize-method linear `
  --state-model ai_data\v4_screen_states\screen_state_best.pt `
  --detector ai_data\v5_hazards\runs\segment_round04\weights\best.pt `
  --imgsz 960 `
  --conf 0.20 `
  --detect-every 3 `
  2>&1 | Tee-Object -FilePath ai_data\v612_guard_smoke_128.log
```

Observe the game. Stop immediately if it still appears to press movement on a
visible menu. A fail-closed `[V612 GUARD] SAFETY STOP` is a deliberate safety
mechanism, not a successful test.

Look in `ai_data/rl_v2_runs/v612_guard_smoke_128/` for:
- `v612_guard_episode_XXXX.jpg` and matching JSON, for menu transitions or
  an ambiguous-screen safety stop.
- `v61_report.json` (retains existing V6.1 report format), with the guard
  counters under `detector.v612_live_input_guard`.
- `latest.zip`, which is just a partial untrained checkpoint if fewer than
  128 steps were collected; don't assume useful PPO training occurred.

After the 128-step run, confirm no movement on menu and whether the saved
menu screenshots really show a menu. **Only then** start longer experiments
with a fresh run name.
