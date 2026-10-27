# Human demonstration recorder + behavioral cloning — V1.3

**Updated: 27 October 2026 · Windows · local Practice only · experimental.**

Source branch: [`demo-recorder-debug-20261027`](https://github.com/HugoGroene/ranked-surfers-ai/tree/demo-recorder-debug-20261027), user-confirmed latest commit `943909d`. The tracked `rs_v612_demo.py` is **V1.3** (calibrated fast MSS capture), not the V1.1 version described in the old README. The separate `rs_v612_demo_bc.py` was introduced with the original demonstration integration. This recorder does **not** play the game, send movement controls, or automatically restart it.

## Why collect demonstrations?

The latest 7-hour PPO continuation (local checkpoint, **not** yet on GitHub) scored **mean 4.61 s, median 4.21 s over 100 deterministic episodes**, with **44/100 ≥5 s** and **2/100 ≥10 s**. Yet deterministic evaluation selected **zero LEFT and RIGHT actions**. Average LEFT+RIGHT policy probability was only **0.71%** (`0.56%` and `0.15%`). Human demonstrations are intended to show it successful lane movements, **not** replace PPO or assume all human actions are ideal. Whether BC improves game survival has **not been evaluated**.

The correct learning sequence is: **verified human observation/action pairs → approved good intervals → conservative actor-only BC on a copy of the best PPO → matched 100-episode deterministic gameplay evaluation → optional separate PPO continuation**.

## Status and known issues

| Item | Current evidence |
| --- | --- |
| V1.3 installation on Windows | User installed successfully on 27 Oct 2026 |
| Demo unit tests | **12/12 PASS** locally (offline tests only) |
| V1.1/V1.2 action capture | Real trials found missing or invalid labels; **do not train on these broken runs** |
| V1.2 capture | Real trial `demo_20261027_185600_980494`: **51 frames / 14.07 s ≈ 3.55 FPS**, median interval **272 ms**, **0 valid action samples**, despite real keydowns being recorded |
| V1.3 approach | One-time browser screenshot / full-desktop MSS template matching to find physical canvas; then fast `mss.grab` every capture (no per-frame Chromium screenshot) |
| V1.3 browser/display behavior | Detects viewport/fullscreen changes about once/second; recalibrates and **discards an incomplete run** when geometry changes |
| V1.3 real performance and input labels | **NOT YET CONFIRMED** with a recorded real Windows trial. A new short candidate test is essential |
| Behavior-cloned agent | No measured 100-episode result yet; preserve source PPO checkpoint |

## 1. Setup and checks

Run inside the **original** repo or whichever clone has the game, `.venv` and local model assets:

```powershell
cd "$HOME\Documents\ranked-surfers-ai"

.\.venv\Scripts\python.exe -m unittest test_rs_v612_demo -v

$state = "ai_data\v4_screen_states_rebuild_01\screen_state_candidate_v2.pt"
$detector = "ai_data\v5_hazards\runs\segment_round04\weights\best.pt"
foreach ($path in @($state,$detector)) {
  if (-not (Test-Path $path)) { throw "Required local model missing: $path" }
}
```

Stop other browser automation / PPO train/eval before recording. The GitHub clone alone does **not** include a working `.venv`, most `ai_data` assets, or the 4.61-second local checkpoint. Check those separately if working in `ranked-surfers-ai-new`.

## 2. Start the local Practice game

**PowerShell #1:**

```powershell
cd "$HOME\Documents\ranked-surfers-ai"
.\.venv\Scripts\python.exe server.py
```

Leave it open. Practice URL: `http://localhost:8000`. Only local Practice is supported. The server's custom Unity WebGL headers are required.

## 3. Start recording V1.3 into a new local-only folder

**PowerShell #2:**

```powershell
cd "$HOME\Documents\ranked-surfers-ai"
.\.venv\Scripts\python.exe -u rs_v612_demo.py record `
  --root "ai_data\demonstrations_v13" `
  --fps 20 `
  --detect-every 3 `
  --imgsz 960
```

The recorder launches a separate Chromium browser. It only captures screen pixels and logs **your physical keys**: arrow keys, WASD and Space. Press **R in the game** to restart manually after death. It never injects gameplay inputs.

For a reliable trial:

1. Keep the browser uncovered and foreground while playing. Enter fullscreen **before** starting a run, then allow any `[CALIBRATION]` message to finish. If you resize/fullscreen mid-run, V1.3 detects that change and discards the incomplete run while recalibrating. Do not reposition the window while playing.
2. Wait for `[RUN ...] Gameplay confirmed` before expecting training samples. Try deliberate **LEFT, RIGHT, JUMP and ROLL** movements; look for `[HUMAN INPUT]` messages from Win32 and/or `[BROWSER INPUT]` diagnostic messages.
3. Check the periodic `[CAPTURE] Measured frame interval ... FPS` line. The requested `--fps 20` is a **target**, not a guarantee. Sustained low FPS produces invalid action labels.
4. After a good run, die and wait for `Completed`, **or Alt+Tab after a good segment** to freeze the current run in memory. Alt+Tab to the terminal and type a command followed by Enter.

| Terminal command | Meaning |
| --- | --- |
| `S` | Save most recent buffered run as **candidate** (recommended first; excluded from BC) |
| `P` | Save candidate and open preview video/contact sheet |
| `G` | Save + approve, default excluding final **0.7 seconds** (not enough if the error occurred earlier) |
| `T` | Enter explicit good intervals, e.g. `0.25-4.8,5.3-8.6`, and save + approve |
| `R` | Discard the latest unsaved buffered run; **not** the same as the game's restart key |
| `Q` | Quit recorder |

The **latest buffered completed run** is kept in RAM for save/reject; save it before another run replaces it. Each saved replay receives a distinct timestamped folder. `G` refuses to approve all-NONE/no-action examples, but **the existence of key events is not proof their timing is correct**.

## 4. Location of recordings and what's inside

Your old demos remain in `ai_data/demonstrations_v1/` or `ai_data/demonstrations_v12/`; current V1.3 recordings go under **`ai_data/demonstrations_v13/` in the same repository**:

```text
ranked-surfers-ai/
  ai_data/
    demonstrations_v13/
      capture_preview_full_canvas.png   # current calibration reference
      runs/
        demo_YYYYmmdd_HHMMSS_micros/
          manifest.json                 # candidate/approved/rejected + selected intervals
          frames.npz                    # single RGB model frames, 3×288×512 each
          aux.npy                       # V6.1.2 557-D auxiliary features
          actions.npy                   # NONE/LEFT/RIGHT/JUMP/ROLL sample labels
          valid.npy                     # label alignment/interval validity mask
          timestamps.npy                # gameplay-relative seconds
          events.jsonl                  # original Win32 + browser keyboard records
          preview.mp4 (or preview.avi)  # uncropped full-canvas human review video
          contact_sheet.jpg             # thumbnail overview of model input frames
      bc_runs/
        bc_YYYYmmdd_HHMMSS/
          model.zip                     # new fine-tuned PPO model, NOT source replacement
```

The human `preview.mp4` preserves full-canvas content and aspect ratio. `frames.npz` deliberately contains the **same cropped/letterboxed 512×288 game image** as the PPO training pipeline. These are different on purpose. Four-frame image history is reconstructed by the dataset loader to avoid storing every RGB frame four times.

The recorder captures asynchronous YOLO detections with staleness/episode gating, then reconstructs the action/timing history for the 557-D auxiliary observation. This is designed for model compatibility, but keydown-to-observation discretization remains approximate and **must** be checked with real data. An approved manifest does not guarantee correctness.

### Open and inspect a replay

```powershell
cd "$HOME\Documents\ranked-surfers-ai"
explorer "ai_data\demonstrations_v13\capture_preview_full_canvas.png"

$run = Get-ChildItem "ai_data\demonstrations_v13\runs" -Directory |
  Sort-Object LastWriteTime -Descending | Select-Object -First 1
if (-not $run) { throw 'No V1.3 demonstration runs saved yet' }
explorer $run.FullName

$m = Get-Content "$($run.FullName)\manifest.json" -Raw | ConvertFrom-Json
$m | Select-Object status,duration_seconds,frames,recorded_keydown_count,valid_action_samples,action_counts_valid,keyboard_sources | Format-List

$video = Join-Path $run.FullName $m.preview_video
if ($m.preview_video -and (Test-Path $video)) { Invoke-Item $video }
```

Do not approve until you verify ALL of the following: the preview shows the correct unobstructed canvas, it plays at plausible speed, actual physical key presses appear in events and in **valid** action labels, LEFT and RIGHT aren't accidentally zero when you performed lane changes, and the intended approved segments contain good decisions. If frame intervals are frequently `>130 ms`, labels may be invalid. Look at `valid_action_samples` and `action_counts_valid` **not** merely `recorded_keydown_count`.

### Edit the replay later — including one already saved with G

`review` updates **only `manifest.json`**; raw RGB, actions, aux, video and timestamps remain unchanged. You can exclude wrong moves and death portions long after saving.

```powershell
# Keep 0.2 to 5.5 seconds only:
.\.venv\Scripts\python.exe rs_v612_demo.py review `
  "$($run.FullName)" --status approved --segments "0.2-5.5"

# Or keep two separate good intervals and exclude a mistake between them:
.\.venv\Scripts\python.exe rs_v612_demo.py review `
  "$($run.FullName)" --status approved --segments "0.2-3.5,4.2-7.0"

# Or stop the replay from being used in BC at all:
.\.venv\Scripts\python.exe rs_v612_demo.py review `
  "$($run.FullName)" --status candidate
```

A saved `candidate` can later be changed to `approved`, once verified. Bad or uncertain runs should stay `candidate` or `rejected`. Merely deleting the last 0.7 s may still leave the decision that caused death; manually inspect the time of the wrong input and adjust the cutoff or intervals accordingly.

## 5. Behavioral cloning: separate PPO model, same architecture

The current PPO input is a `Dict(image, aux)` with `12×288×512 uint8` stacked RGB and `557` numeric features. The action labels map to `0=NONE`, `1=LEFT`, `2=RIGHT`, `3=JUMP`, `4=ROLL`.

Before BC:

- Collect multiple **approved, varied** episodes, not only one long recording. At least **3 approved runs** are required for the current holdout workflow; aim for a meaningful number of correctly timed LEFT and RIGHT examples across different obstacle situations.
- Ensure each run's preview and timestamps are sensible. Review correct segments manually; a long duration does not prove optimal behavior.
- Preserve the full original **4.61-second** PPO checkpoint; **do not** train into its original path.

Then:

```powershell
cd "$HOME\Documents\ranked-surfers-ai"
$best = "ai_data\rl_v2_runs\v612_overnight_next_420min_press5_20261027_094116\latest.zip"
if (-not (Test-Path $best)) { throw "Missing best local checkpoint: $best" }

.\.venv\Scripts\python.exe -u rs_v612_demo_bc.py `
  --root "ai_data\demonstrations_v13" `
  --checkpoint $best `
  --epochs 3 `
  --batch-size 12 `
  --lr 0.00002
```

`rs_v612_demo_bc.py` accepts only **approved** runs and segments, splits holdout by **entire run**, preserves human action examples while subsampling some NONE actions for training, fine-tunes the **policy network/action head** with existing visual/aux representation frozen initially, keeps critic weights fixed, and saves a **separate** V6.1.2-compatible checkpoint. Existing original PPO weights are never replaced. Best holdout imitation loss is **not equivalent** to better gameplay survival.

Run `train_rs_v612_ppo.py --mode eval --resume <new-bc-model.zip> --eval-policy deterministic --eval-episodes 100` with exactly the same settings as baseline. A ready-to-paste full 100-episode command and report summary are in **[README.md](README.md#benchmark-or-evaluate-a-checkpoint-100-deterministic-episodes)**. Compare survival mean, median, ≥5s and ≥10s, LEFT/RIGHT action counts and policies, as well as guard aborts. **No BC gameplay improvement has been demonstrated yet.**

Only if evaluation is acceptable, optionally continue PPO from the **new** BC model in another training directory. The BC helper initializes fresh PPO Adam statistics after fine-tuning so previous PPO moments are not blindly reused. Do not simultaneously change YOLO resolution, PPO reward, LSTM, keypress length and demonstration learning; make controlled tests one variable at a time.

## 6. Troubleshooting / reliability

**Video still cropped / shows browser borders:** The V1.3 start-of-session calibration is designed to locate the Unity canvas on the *physical desktop* using image matching and fail if uncertain. Inspect `capture_preview_full_canvas.png`. Recalibrate with the game visible/uncovered and still at a stable screen; do not trust a misframed preview. The PPO's game-only crop is intentionally tighter than the preview.

**Input feels laggy or windowed gameplay glitches:** V1.2 used `page.screenshot(clip=...)` *for every frame*; its real trial measured only **3.55 FPS** and zero usable action samples. V1.3 requests one calibration screenshot then records with MSS, avoiding that compositor work. Do not claim V1.3 is fixed until you confirm actual FPS, latency and validity on Windows. If still slow, quantify YOLO GPU/CPU load, detector freshness and the screen classifier before changing observation semantics. Use fullscreen before play; window changes invalidate the current partial run.

**Keypress counts look right but valid samples are zero:** Input events are not enough: the observation preceding each action must have a sufficiently short, unambiguous decision interval. V1.3 prefers Win32 physical key timestamps to delayed browser callbacks and records both for diagnosis. If sampling is slow, do not relax the `130ms` label-validity bound just to make bad data appear valid.

**Canvas moves during recording / multi-monitor scaling:** V1.3 checks window/canvas geometry and recalibrates on detected changes. A move or occlusion not detectable through its view signature is still a risk, and desktop/template screenshots may disagree during active animation. Favor a stationary, uncovered display and verify each session's capture preview. Changing fullscreen may discard the unsaved partial run.

**No detection or classifier asset in a fresh Git clone:** Most `ai_data` model assets are local/ignored. Check the local paths and copy the necessary assets from your original project rather than committing enormous training directories.

**Don't train on old broken runs:** At least one V1.1 run with 297 frames had **zero physical actions**. A V1.2 run had input events but **zero valid action samples**. Keep these candidate/rejected or archive them; action inference from playback alone is unreliable.

**Safety:** This system is only for `http://localhost:8000` local Practice. The recorder never submits movement controls; the PPO controller uses a fail-closed gameplay guard. Don't run PPO training/evaluation alongside manual recording. Fullscreen, calibration and hardware speed influence the quality of every recorded example.

## 7. Version control

The recorder/BC source and tests are on the debug branch. The newest 4.61-second checkpoint, `ai_data/demonstrations_v13/` recordings and many trained models are **local only**. Your `.gitignore` may not exclude the V1.3 demonstration data; do **not** run `git add .` or accidentally commit raw replays/backups. Stage documentation and code by explicit filename:

```powershell
git add README.md README_DEMONSTRATIONS_V612.md
git diff --cached --stat
git commit -m "Update docs for PPO 4.61s and V1.3 demonstrations"
git push origin demo-recorder-debug-20261027
```

This documentation describes implemented code and observed results; **a 12-test offline PASS is not a substitute for a real Windows live-capture test**, and a 4.61-second reference is not proof that BC will improve it.
