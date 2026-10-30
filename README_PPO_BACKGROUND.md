# Experimental background-display workers

29 October 2026. Local Practice only. This is an opt-in capture/input backend
and a fixed-policy throughput diagnostic. Initial background gameplay and a
256-transition corrected training smoke test passed. It is not yet validated
for unattended long training and is not a multi-environment PPO learner. Existing
foreground defaults, policy settings, rewards and guards are unchanged.

## Initial live results

The VDD display is extended at 1920x1080 / 144 Hz. A read-only check confirmed
the physical primary still uses NVIDIA at 1920x1080 / 144 Hz. The driver adds a
separate virtual adapter; it does not replace the physical monitor's NVIDIA
driver. A reported physical brightness change has not been diagnosed; physical
brightness, HDR and colour settings were not intentionally changed by these tools.

`reports/background_workarea_01/benchmark.json`: 20-second fixed-policy run,
262 valid decisions, three completed episodes, no guard stop. All 24 native
foreground samples reported the game window in the background; 23 samples
showed motion and the canvas document recorded R/JUMP/ROLL key events. The
source checkpoint hash stayed unchanged. This proves short background transport
and gameplay, not exact per-action Unity causality or a parallel speedup.

`ai_data/rl_v2_runs/v612_background_smoke_20261029_workarea01/`: 256 transitions
collected and optimized, two optimizer pauses, one continuation/resync, three
terminal transitions, no discarded transitions, no guard stop. The corrected
ledger audit returned PASS_SETTLED_LEARNING. This was a new disposable smoke
checkpoint, not a promoted gameplay model. Saved settings remained rollout 128
and learning rate .0003. Final optimization leaves the game paused as intended.

No click on the game was required. Startup may initially activate a newly
created Chromium window, but there is no ongoing Windows activation loop. The
successful diagnostic verified that it subsequently ran in the background.
The test window fits the selected monitor's work area, excluding its taskbar.
The owned page's existing game-stage is expanded to the viewport and unrelated
site UI is hidden. Native fullscreen is avoided. Pause/UI controls inside the
game-stage stay present; classifier thresholds are unchanged.

Earlier failed diagnostics remain preserved under `reports/background_live_*`.
They exposed a native-title presentation race, a minimized window, the embedded
game's incompatible layout, and calibration failures when the taskbar obscured
the bottom of an ordinary window. None of those failed fixed-policy attempts
trained or modified a checkpoint.

## Windows setup

VirtualDrivers VDD Control was installed through winget. Winget verified the
installer hash; the installed executable has a valid SignPath signature.
Installing the control app does not itself prove the Windows display driver is
active. Open VDD Control, install/enable its display driver, and extend Windows
to one additional display. Keep the physical display primary. For the prototype
use 1920x1080 at 144 Hz, 100% scaling and SDR. Do not duplicate/mirror displays. The tool
lists actual MSS indexes; Windows Settings display numbers may differ.

The official project and installation instructions are at
https://github.com/VirtualDrivers/Virtual-Display-Driver . The existing physical
display remains usable; Chromium may activate during initial window creation,
but this backend never performs ongoing native foreground activation/retries.
Keep the game window unminimized and unobstructed on its dedicated display.

```powershell
cd "$HOME\Documents\ranked-surfers-ai-claude\ranked-surfers-ai-new"
$py = "$HOME\Documents\ranked-surfers-ai\.venv\Scripts\python.exe"
& $py -B -s -m tools.ppo_background_display
```

If only index 1 (the physical primary) appears, background-display testing cannot
start. Do not select the primary to bypass this check.

## New trainer flags

- `--background-display N`: opt into a dedicated non-primary MSS display. Own
  Chromium is positioned there through CDP; page focus is emulated and background
  throttling switches are passed at launch. No `SetForegroundWindow`, SendKeys or
  global input hook is used by the background backend. Gameplay keys still use
  the owned page's existing Playwright interface.
- `--canvas-policy-input`: with the display flag, use a pixel-calibrated game
  canvas, then the usual fractional crop/letterbox for policy and detector. This
  matches the BC recorder's preprocessing for equivalent canvas pixels and
  dimensions. It changes the framing seen by historical PPO checkpoints, so is
  explicit rather than silently enabled. Without this flag the old display-crop
  policy source remains in use.

The state/menu guard still receives raw selected-display images and retains all
confidence thresholds. Policy canvas calibration does not change classifier
preprocessing or imply the existing classifier is validated on every new layout.
Window/display geometry is rechecked every 0.5 seconds. Hidden/minimized/moved
windows, unavailable displays, unsupported scaling and changed canvas geometry
stop the run. Pixel matching is performed at startup against the selected
display; unrelated windows obscuring the game later are handled by the existing
visual guard, not by a claim of continuous pixel-match calibration.

Persistent batches reject changing capture mode, display or policy-source mode
mid-session. Use a fresh session when changing these settings.

## First short fixed-policy diagnostic

Start the server in a separate terminal unless already running:

```powershell
cd "$HOME\Documents\ranked-surfers-ai-claude\ranked-surfers-ai-new"
$py = "$HOME\Documents\ranked-surfers-ai\.venv\Scripts\python.exe"
& $py -B -u server.py
```

Then run this in your own training terminal, after confirming display 2 exists:

```powershell
cd "$HOME\Documents\ranked-surfers-ai-claude\ranked-surfers-ai-new"
$py = "$HOME\Documents\ranked-surfers-ai\.venv\Scripts\python.exe"
$report = "reports\background_one_" + (Get-Date -Format "yyyyMMdd_HHmmss")
& $py -B -s -u -m tools.benchmark_ppo_background `
  --displays 2 `
  --checkpoint "ai_data\demonstrations_v13\bc_runs\bc_sampled_batch_20261028_1740_v2\latest.zip" `
  --canvas-policy-input `
  --seconds 90 `
  --report $report
Write-Host "REPORT: $report\benchmark.json"
```

Use your normal browser on the physical monitor during the diagnostic. This
does not perform optimizer updates or overwrite/save a checkpoint. Models load
before the common measurement starts; the measured interval includes guarded
episode starts and resets. Workers use independent policy/detector copies, not
shared inference. The parent verifies the source checkpoint hash afterward.

Inspect `worker_0/background_display.png`, `background_canvas.png` and
`background_policy.png` for exact framing. `trace.jsonl` samples native foreground
status, visual motion and keydown events in the canvas's own document. Native
background status is separate from emulated `document.hasFocus()`. Keydown
receipt and motion are transport evidence, not proof Unity executed each action;
review the saved gameplay captures before approving background training.

## Two-worker benchmark and shared PPO gate

After one worker passes, create a second dedicated extended display and confirm
its index. Repeat the diagnostic with `--displays 2 3`, otherwise identical
settings. Do not launch two workers on the same display. Compare aggregate valid
decisions per second from `benchmark.json`, motion/input evidence, observation
latencies and resource pressure. Failed workers produce STOPPED status and no
successful throughput metric. This diagnostic reports a partial rate separately.

These measurements omit PPO optimizer/pause overhead. Multiple model copies may
increase memory/compute contention compared with a future shared service. They
establish whether the rendering/capture/input backend scales, not the eventual
learning speedup. Testing three/four workers is deferred until two work well.

`SettledPPO` still deliberately rejects `n_envs != 1`. Shared learning requires
separate work on per-worker boundary settlement, reward clocks, passive masks,
terminal observations, pause barriers and frame lineage. A worker resetting must
not cause unrepresented elapsed time in another worker. Aggregate rollout size
must be declared: 128 steps per worker gives 256 transitions for two workers.
Do not remove the one-environment check and assume correctness follows.

Virtual refresh diagnostic: tools.configure_ppo_virtual_modes --apply --refresh-hz 60 sets only active MTT virtual monitors to 1080p60. Supported rates are 30, 60 and 144; the default remains 144. Physical monitor settings are verified unchanged. This is a display diagnostic, not an adopted training configuration. The current browsing comparison uses one virtual monitor rather than two, so any improvement cannot yet be attributed to refresh rate alone.

Headless prototype (no PPO or movement inputs): python -B -s -m tools.probe_ppo_headless --workers 1 --seconds 15 --report reports/NEW_UNIQUE_DIRECTORY. Use --workers 2 for concurrent browser capture. Each isolated browser uses a 1904x937 viewport, local Practice only, GPU requested, and canvas-clipped JPEG capture. Saves renderer, capture timings and sample images. Rendering/capture rates are not policy decisions or optimized PPO throughput; motion is not automatic gameplay verification. Existing training defaults and checkpoints are untouched. Virtual displays are not used by this prototype; their current enabled state is not changed.

Initial headless results (reports/headless_prototype_20261030_01 and headless_prototype_20261030_two_01): NVIDIA ANGLE D3D11 confirmed; saved early images visually show Practice gameplay. One browser captured 18.27 images/s (median 50.77ms); two captured about 15 images/s each (median 66.36/66.53ms). These are short passive browser-image benchmarks, not guarded policy or learning benchmarks. Both virtual displays remained enabled during these checks, so desktop-lag isolation without VDD remains outstanding. All prototype browsers closed at completion.

Experimental guarded headless PPO: tools.train_ppo_parallel --headless --displays 1 2 --checkpoint SOURCE.zip --canvas-policy-input --steps 256 --max-minutes 2 --worker-timeout 90 --report reports/NEW_UNIQUE_DIRECTORY. In --headless mode, --displays values are distinct worker IDs rather than monitor indices. Each worker captures its own 1904x937 browser canvas, places it at (8,87) in a black 1920x1080 guard image to retain current ROI coordinates, and uses the existing calibrated-canvas policy crop. No desktop pixels are read. Existing classifier/menu/pause guards and settled-learning collector remain enabled. The source rollout length is retained per worker, so two workers double the transitions per shared update; this is not an identical learning configuration to one worker. Source checkpoint is read-only; only new experimental outputs are saved. Native GPU presentation timestamps are unavailable with browser screenshots, so capture-age semantics and pause/resume timing still need validation before long training.

Guarded headless PPO smoke results, 2026-10-30: reports/headless_shared_ppo_20261030_01 completed two workers x128 transitions, one shared update, all 256 optimized in 26.693 active seconds (9.591 optimized transitions/s). Both worker ledger audits passed, no dropped transitions or guard stops. reports/headless_shared_ppo_one_20261030_01 completed one worker x256 transitions, two updates in 38.226 seconds (6.697/s), audit passed including one passive post-resume transition. Source SHA256 unchanged in both. Observed short-run ratio 1.43x, not a settled benchmark: different aggregate update sizes/counts, stochastic episodes, startup/load excluded, and two-worker post-update continuation not exercised. Virtual displays disabled. No promotion or long-run validation implied.

Shared headless scaling diagnostics: --headless now permits up to four distinct --displays worker IDs (monitor-backed mode still permits two). --telemetry records whole-system CPU/core usage, RAM, process CPU and NVIDIA samples in system_usage.jsonl at approximately 1 Hz. --preview-port 8001 serves a local grid at http://127.0.0.1:8001/ while the runner lives; it shows existing policy frames at ~1 Hz, phase, transition count and image age, plus shared-update runtime/throughput after each completed update. No extra game captures. --save-every-minutes controls atomic saves at shared update barriers. Each additional worker retains the source rollout length; aggregate update size therefore changes with worker count. Compare throughput separately from learning quality. VDD is currently disabled at the device level (problem code 22); no VDD-named startup entry/task was found. Reboot persistence has not been empirically tested.

Repeated headless 2/3/4-worker results and limitations are recorded in reports/HEADLESS_SCALING_20261030.md. All completed repeated runs passed worker ledger audits. Measured optimized transitions/s: 11.29, 15.03, 16.97 respectively; these short checks do not establish learning quality or sustained scaling. See the report for the preserved failed first three-worker startup attempt and its bounded startup handling changes.

Experimental headless worker limit extended to six for the requested scaling check: --headless --displays 1 2 3 4 5 6. Monitor-backed mode still supports at most two. Six retains 128 transitions per worker from the source, yielding 768 transitions per shared update; use --steps 1536 for the same two-update smoke check. This is a throughput experiment, not an adopted learning configuration.

Six-worker smoke check passed all six learning audits: 18.58 optimized transitions/s, versus 16.97 for four in prior short checks. Peak VRAM 6954 MiB and system RAM 85.6%; marginal throughput gain about 9.5%. Results recorded in reports/HEADLESS_SCALING_20261030.md. Six is supported experimentally, not selected as the default.

Framing audit: reports/framing_audit_20261030_02/README.md documents legacy desktop vs BC/current headless vs whole-canvas candidate inputs, same-frame comparisons and recoverability. Existing headless mode still applies DEFAULT_CROP inside the canvas; it is not yet the proposed whole-canvas profile. No runtime framing changes were made by this audit. BC preview videos contain burned-in action labels and must not be used directly as training data. PPO updates do not train the frozen detector.

Whole-canvas opt-in implementation and validation: README_PPO_CANVAS_V1.md. Add --framing-profile canvas_v1 to the shared headless command; headless canvas becomes 1920x945 and policy/detector consume its entire contents. Prior scaling results used legacy canvas fractions, not this new profile.

Dashboard update: six previews use three columns/two rows on desktop, two columns below 1000px and one below 620px. A live SVG survival chart uses requested --max-minutes as its fixed x-axis, seconds of active run duration as y-axis, and the rolling last 100 completed episodes pooled across workers. One point per 30-second bucket; the current bucket updates live. Time-limit episodes are included and counted explicitly. Empty periods are not filled with zero-duration episodes. Rollout overrun points are clamped to the requested x-axis endpoint. Worker episode records (survival_episodes.jsonl) and aggregate survival.json are saved, with aggregation every ~5 seconds independently of PPO updates. The chart is a training trend, not matched evaluation evidence. Reopen a saved new-run chart without training: python -B -s -m tools.ppo_worker_preview --report reports/YOUR_RUN --port 8001. Stop with Ctrl+C. Older runs without these metrics are not automatically backfilled. The UI demo screenshots under reports/dashboard_ui_check_20261030_01 use explicitly synthetic graph data; the real single-rollout metrics smoke test is reports/dashboard_metrics_smoke_20261030_01 (three completed episodes, one update, learning audit passed).

Diagnostic dashboard: graph points are red and connected by a thicker teal line. The status strip shows live elapsed/requested time, shared updates and optimized throughput; last-update KL, clipping fraction, entropy loss, explained variance and value loss; per-worker and aggregate rollout sizes, learning rate, framing, collection/transport and optimizer time, and last checkpoint transition count. Worker tiles show completed episodes and recent-25 mean duration. The survival summary includes overall mean and best duration. CPU/core/RAM/available memory/GPU/VRAM/power reuse existing telemetry samples; the UI reads telemetry/graph every five seconds and previews once per second, with no extra screen captures or GPU queries. These measurements can include unrelated desktop activity.
Shared training now retains checkpoint_NNNNNNNNN.zip snapshots at configured save barriers in addition to latest.zip, so a diagnostic run can compare intermediate models. The initial source stays unchanged. --max-minutes counts active training after worker readiness and stops after a whole shared update; allow startup and a final rollout beyond the requested duration.

Selected source for the planned 30-minute diagnostic: ai_data/rl_v2_runs/v612_largebc_defaults_3h_20261029_204226/latest.zip (SHA256 66dc546f04253d9d2cf4aeb5af0764d9500c70ce78439159e518d5a96553bc9e). ZIP checked; num_timesteps 1713054, n_steps 128, batch 32, epochs 5, LR .0003, gamma .995, lambda .95, target_kl None. This is a legacy-framing source explicitly adapted with --framing-profile canvas_v1. Six workers give aggregate rollout 768. Use --steps 768000 --max-minutes 30 --save-every-minutes 5 --telemetry --preview-port 8001 with a new report path. The high step ceiling lets the wall-time budget control the run. The real preflight (reports/dashboard_diagnostic_preflight_20261030_01) optimized 128 transitions, passed the learning audit, saved both latest and numbered snapshot, and preserved the source. No 30-minute run was launched by the assistant.
Worker action counts in the dashboard are sampled PPO actions, not confirmed Unity responses; guards may suppress actions. Counts are shown to diagnose action collapse or poor lateral coverage. Dashboard progress uses optimized transitions; elapsed time updates in the browser between shared updates. Snapshot/run comparisons and logged diagnostics remain necessary before claiming improved policy quality.

Persistent graph fix: the previous in-progress 30-second bucket replaced its newest point. New runs now append immutable samples at fixed 10-second boundaries. All previous points remain visible for the entire requested runtime, even though each point averages the recent 100 completed episodes. The first point appears after 10 seconds if episodes have completed. Late rollout completion does not rewrite the endpoint; later completed episodes still count in overall summaries. Existing point values are preserved during metric refreshes. The x-axis remains the full requested runtime.
