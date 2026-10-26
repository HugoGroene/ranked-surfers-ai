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
  about 45 ms apart. Two strong template matches in the last three readings,
  or four consecutive ML MENU predictions supported by moderate template
  matches, confirm `MENU`. Uncorroborated ML MENU votes cannot confirm death.
  A single template match **never** suffices to claim death.
- Two consecutive verified `GAMEPLAY` readings, or two independent visual
  confirmations of active gameplay, permit the requested action.
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

## Episode 76 correction (false ML MENU during gameplay)

In `v612_learning_01`, the rollout stopped during visible gameplay at 41 m.
The V4 classifier swung toward MENU (up to 98.8%) while the separate menu
template stayed at 0.00–0.16. The previous V6.1.2 guard incorrectly counted
two *ML-only* MENU predictions as independent confirmation of death.

The corrected guard keeps the V4 checkpoint, capture region and PPO architecture
unchanged. Now:

- An ML-only MENU prediction is **disagreement**, not confirmed death. Two
  strong menu-template matches among the last three captures still confirm
  MENU, including when the classifier returns UNCERTAIN.
- If two consecutive fresh captures show textured central-track motion *and*
  changes in the yellow distance HUD, with no recent menu-template hit, this
  is independent positive evidence for continuing gameplay. A low template
  score, motion alone, or a static yellow HUD **cannot** authorize movement.
- Two consecutive high-confidence V4 GAMEPLAY readings can also recover as
  before, but only when recent captures have no menu-template evidence.
- Unresolved disagreement still stops after six captures without sending a
  movement action. The pending PPO action is never silently replaced with a
  no-op, and interrupted PPO rollouts are not passed off as trained updates.

Visual continuity uses the inherited `_motion_view`, `_meter_mask` and
`startup_motion_threshold` helpers on the existing raw BGRA capture; it does
not modify the crop. The distance-HUD criterion may be too strict if the HUD
placement or digit color differs. In that case the system fails closed and
saves the new `track_motion`, `track_texture`, `hud_pixels`, and
`hud_changed_pixels` diagnostic fields, which must be reviewed before tuning.

The first six successful recoveries from ML-menu disagreement also save
`v612_disagreement_episode_XXXX_NN.jpg` plus a JSON log for auditing.
Report counters `ml_menu_without_template` and `visual_gameplay_recoveries`
are included under `detector.v612_live_input_guard`.

## Episode 12 correction (true menu with moderate template score)

The longer smoke test stopped at episode 12 with six near-99.95% ML MENU
predictions and an unchanging template score of 0.577. The saved screenshot
unambiguously shows the title menu. The startup detector already accepts
two ML MENU captures with template >= `0.8 * threshold` (0.544 when the
configured threshold is 0.68). The pre-key guard previously required two
template scores >= 0.68 even when ML and the weaker template agreed.

The corrected pre-key guard now accepts four **consecutive** captures with
high-confidence ML MENU and template score >= `0.8 * threshold` as another
way to confirm a menu. It still confirms two strong template matches as
before. A MENU prediction on its own cannot end the episode, so the original
episode 76 gameplay case with template scores 0.00–0.16 remains blocked from
false death confirmation. This fixes the corroboration mismatch without
altering the V4 classifier, screenshot crop, or raw captures.

Offline regression tests cover episode 76 probabilities, genuine menus,
loading states, motion without HUD progression, HUD changes without track
motion, and fail-closed timeouts:

```powershell
.\.venv\Scripts\python.exe -m unittest -v test_rs_v612_guard.py
```

These synthetic tests do **not** validate the real scene-motion/HUD thresholds.
Do not resume `v612_learning_01` yet. First run the smaller live smoke test
below and inspect any MENU, disagreement and safety-stop screenshots.

## Install

Extract `v612_episode76_guard_patch.zip` into your project directory
(`C:\Users\Hugo\Documents\ranked-surfers-ai`), replacing the V6.1.2
trainer, guard and test files plus this README. Back up those files first if
you have uncommitted local changes. The archive does **not** overwrite the
V6.1, V6.1.1, V4, or V5 scripts or the existing trained model checkpoints.

If the episode 76 patch is already installed, apply
`v612_episode12_menu_confirm_patch.zip` instead. This smaller update replaces
only `rs_v612_gameplay_guard.py`, `test_rs_v612_guard.py`, and this README.

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
