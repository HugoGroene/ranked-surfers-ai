# Human demonstrations for Ranked Surfers V6.1.2 (local Practice only)

**Experimental integration (V1.1 keyboard + preview fix).** Uses the same game-only crop/letterbox preprocessing, four RGB frames, V5 nine-class YOLO asynchronous detector, V6.1.2 557-D auxiliary schema, and five discrete actions. It does not alter old training code, checkpoint files, reward shaping, or the UI safety guard. No ranked/live service connections are permitted.

## Install

Copy `rs_v612_demo.py`, `rs_v612_demo_bc.py`, and `test_rs_v612_demo.py` into the **repository root**, next to `train_rs_v612_ppo.py`. Existing venv packages (OpenCV, numpy, mss, Playwright, Ultralytics, SB3, PyTorch) are reused; no new pip packages.

```powershell
cd "$HOME\Documents\ranked-surfers-ai"
.\.venv\Scripts\python.exe -m unittest test_rs_v612_demo -v
# Another terminal: .\.venv\Scripts\python.exe server.py
.\.venv\Scripts\python.exe -u rs_v612_demo.py record --fps 20 --detect-every 3 --imgsz 960
```

Close any *PPO* evaluation/training Chromium first. The recorder launches **its own** Chromium on `http://localhost:8000`; wait for the visible game canvas, click Practice manually, and use arrow keys normally. **The recorder never injects any arrow keys or restart keys.** Restart manually with R after dying, as usual. Keep the captured Chromium game visible and unobstructed during gameplay, as with PPO MSS training. You can Alt+Tab after a good run: focus loss pauses capture and buffers its last gameplay frames. If MENU was detected first, the completed run is also buffered. Wait for a `[RUN] ... buffered` or `Completed` message before saving.

After a completed run, Alt+Tab to the PowerShell recorder and type one of:

- `G` Enter: immediately save **approved**, excluding the last 0.7 seconds by default (an initial safety margin, not a guarantee of optimality).
- `S` Enter: save **candidate** for later review; **not** included in BC.
- `T` Enter: type exact approved intervals, e.g. `0.2-5.4,5.8-8.3`, to save **approved** with mistakes/excluded periods removed without modifying original observations.
- `P` Enter: save **candidate** and open the timestamped `preview.mp4` / `preview.avi` showing the **entire game canvas** (or contact sheet if video codecs unavailable), for review before approval.
- `R` Enter: discard latest completed unsaved run from RAM.
- `Q` Enter: quit.

You can manually restart a new run even after saving an earlier run. Only **the most recent completed run** is retained temporarily in RAM; save/approve it before finishing another run. If you accidentally discard a good run, it cannot be recovered unless previously saved. The recorder does not automatically start/restart Practice. Avoid interacting with the browser or using automated PPO simultaneously.

All outputs are in different uniquely named directories, under:

```text
ai_data/demonstrations_v1/runs/demo_YYYYmmdd_HHMMSS_micros/
    manifest.json           # only approved, selected time ranges can train BC
    frames.npz              # immutable raw, unique 3x288x512 RGB frames
    aux.npy                 # matching 557-D model auxiliary features
    actions.npy             # physical human action BEFORE subsequent observation
    timestamps.npy          # gameplay-relative capture times
    valid.npy               # ambiguous/missing interval labels excluded
    events.jsonl            # raw trusted keyboard events with timestamps
    preview.mp4 or .avi     # timestamped full-canvas replay (if codec available)
    contact_sheet.jpg       # overview of selected frames
```

`manifest.json` can be updated without cutting or deleting any frames. For a previously saved candidate:

```powershell
.\.venv\Scripts\python.exe rs_v612_demo.py review `
  "ai_data\demonstrations_v1\runs\demo_YYYYmmdd_HHMMSS_micros" `
  --status approved --segments "0.2-4.2,4.8-8.3"
```

A run with questionable decisions should be marked candidate/rejected. Manually review the timestamped replay video or contact sheet and selected time ranges; **do not blindly approve the last 0.7-second cutoff**, because a mistake can start well before death. More detailed inspection is possible using individual frame arrays from `frames.npz` (and timestamps/actions); the video can be scrubbed by time in your normal Windows player, while the contact sheet summarizes the whole run. If you need frame-by-frame curation, load `.npz` with NumPy and review it manually before approval.

## First actor-only behavioral cloning experiment

Collect at least three approved **varied** runs; more and richer LEFT/RIGHT situations are better. We specifically require at least some LEFT and RIGHT actions in *training* runs. BC splits **entire runs** for holdout rather than adjacent frames; it preserves all non-NONE actions and subsamples NONE so the action imbalance doesn't swamp lanes. All five action types remain learnable. It freezes the CNN/aux feature extractor and critic, updates only the PPO policy MLP/action head, selects the best validation epoch, and resets PPO Adam optimizer. The existing checkpoint is never modified.

```powershell
$best = "ai_data\rl_v2_runs\v612_overnight_next_420min_press5_20261027_094116\latest.zip"
.\.venv\Scripts\python.exe -u rs_v612_demo_bc.py --checkpoint $best --epochs 3 --batch-size 12 --lr 0.00002
```

A distinct checkpoint is created at `ai_data\demonstrations_v1\bc_runs\bc_YYYYmmdd_HHMMSS\model.zip`. This is **not proven better** until you run your existing controlled deterministic **100-episode** evaluation using that path with `train_rs_v612_ppo.py --mode eval --resume <model.zip> ...` and compare with baseline **mean 4.61 s, 44/100 survive 5s, 0 deterministic LEFT/RIGHT**. Stop if it damages survival or action timing. A short PPO continuation can subsequently resume from *this separate BC checkpoint*, without touching original.

## Safety and data fidelity

- Pre-action causal labeling: observation `i` is paired with physical keydowns *between* observation `i` and `i+1`, never an image after the action. Auto-repeat keydowns and keyups are excluded from action labels. Simultaneous/conflicting keydowns and capture gaps are marked invalid.
- Gameplay-only capture is decided by the **existing** screen state classifier; no menu/death screens used for positive-action BC.
- The terminal is not captured as gameplay. Focus loss mid-run discards the incomplete run. `R` manually restarts game and is **never a PPO action**.
- Observation schema matches V6.1.2, but **human event→decision discretization, asynchronous detector scheduling, and browser→Python timestamp offset** are approximate. Inspect early data; do not assume they are bit-identical to PPO-internal observations. In particular, browser callbacks are only timestamped to the expected observation window; unmatched/late events can be missed.
- Browser–desktop coordinate mapping may need `--screen-x-offset` / `--screen-y-offset` on Windows/DPI-scaled monitors. Inspect `contact_sheet.jpg` before recording substantial data. If it shows Chrome UI/terminal/incorrect crop, discard that run and fix the capture region. The recorder uses the same canvas-region helper as the old manual collector.
- The collector currently identifies a completed run using three consecutive confident `MENU` predictions. If misclassification causes early/late endings, keep as candidate, inspect, and adjust state classifier/capture alignment. It never sends keys to recover.
- More advanced future upgrades: full frame-by-frame preview, event-lag diagnostics, and PPO+BC auxiliary loss. Don't alter reward, fps, LSTM, or PPO hyperparameters during the first BC experiment.

## V1.1 hotfix: important keyboard-data integrity issue

The original V1 recorder used a browser JavaScript callback to log human arrows. One
real-world 15.1-second recording had **zero** keyboard events and all 297 action
labels were `NONE` despite visible gameplay movements. Do NOT train on it.
V1.1 switches to Windows `GetAsyncKeyState` polling every 4 ms in a separate
thread, using `time.perf_counter` timestamps shared with frame captures. Each
physical arrow keydown prints `[HUMAN INPUT] ArrowLeft/Right/Up/Down captured`.
Run one short recording and VERIFY that LEFT and RIGHT occur in its manifest.
This remains **unverified in live Windows gameplay** until you test it locally.
V1.1 refuses to approve any run or interval without a valid non-NONE action;
BC also fails closed on already-approved all-NONE runs. It additionally reads
actual YOLO episode tags instead of accidentally assuming `episode=1` forever.

Old `preview.mp4` files used cropped model frames with a 24-pixel footer and
video aspect ratio 512x312. New videos use **uncropped game-canvas frames**,
letterboxed to the normal 512x288 (16:9) aspect ratio, and draw a small
label inside the image. The full-canvas frames are used for *review video* only;
the PPO dataset still uses its exact original game-only crop. Existing cropped
recordings cannot recover pixels outside the saved crop. A broken run's missing
arrow timestamps also cannot be reconstructed reliably from video alone.

To install the V1.1 ZIP into an existing V1 install, stop the recorder and run
`INSTALL_DEMO_FIX.ps1` from the extracted ZIP. It backs up previous demo files
under `ai_data/demo_patch_backup_YYYYMMDD_HHMMSS` before replacing them; it
never changes PPO training code, YOLO weights or checkpoints. Then rerun tests.
