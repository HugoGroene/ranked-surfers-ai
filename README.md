# Ranked Surfers AI: multimap PPO and behavioral cloning

Updated 7 November 2026. Development branch: **headless-parallel**.

Train a visual policy for the Subway Surfers No Coin challenge using local Unity
WebGL Practice copies. Current work supports **Halloween and Tokyo**, four
independent Electron workers, a shared PPO optimizer, a stateful screen guard,
and an optional offline behavioral-cloning (BC) phase from human demonstrations.

The current configuration is a measured engineering choice, not a proven global
learning optimum. BC gameplay improvement has not yet been established.

## Current configuration

| Component | Setting |
|---|---|
| Workers | Four headless Electron instances; two Halloween, two Tokyo |
| Rendering/capture | 60 FPS offscreen target; GPU shared-texture bridge and shared-memory transport |
| Policy image | Road crop removing 15% per side; RGB 384 x 384, aspect-preserving padding |
| Visual history | Four frames at requested ages 0, 0.05, 0.25, 0.75 seconds; stored oldest first |
| Encoder | Spatial CNN plus auxiliary MLP, feeding actor and critic |
| Auxiliary input | 789 values: legacy features, actual frame ages/validity, 32 timestamped command entries |
| Command history | One second; five-way action encoding, age and validity; worker-local |
| Object detector | Off; screen-state classification remains active |
| PPO | 128 steps/worker, 512 aggregate, minibatch 32, 5 epochs, LR 0.0003 |
| PPO discount/exploration | Gamma 0.995, GAE lambda 0.95, entropy coefficient 0.002, clip 0.2; settled collector uses elapsed-time discounts |
| Inputs | NONE / LEFT / RIGHT / JUMP / ROLL; 5 ms keyboard press setting |
| Dashboard | Off in the continuation command; graphs shown after cleanup |

Requested frame ages and capture/decision rates are targets, not achieved timing
guarantees. Command history describes dispatched inputs, not confirmation that
Unity executed them. Verified optimizer pauses retain image/command history with
clock compensation; episode resets or uncertain continuity clear it. One startup
episode per worker is excluded from PPO collection, not from low-level diagnostics.

The active classifier is selected centrally in
[`maps.json`](development/experiments/multimap_classifier/maps.json). Current
multimap training, evaluation and recording (including BC recording) use it.
Historical standalone experiment scripts retain their own explicit defaults.

## Local setup and assets

Use Windows, the project's existing Python environment with PyTorch/SB3, OpenCV,
Playwright, psutil and Gymnasium, and the local Electron/native capture build.
See [Electron setup](README_ELECTRON_CAPTURE.md) and the
[capture pipeline](development/experiments/capture_pipeline/README.md).

A clone **does not include** current checkpoints, classifier weights, demonstration
recordings, the 50 GB BC export, new map downloads, local Electron dependencies or
run logs. Restore these from your own local storage. Map roots/releases and model
paths are inventoried in `maps.json`; the baseline classifier manifest is also
needed to train a dataset expansion. Missing assets fail preflight checks.
The scripts below use this workstation's existing Python path; adjust `$py` on a
new machine. Development targets the local Practice copies.

## Continue PPO training

Use a compatible four-frame, 789-aux checkpoint with retained-pause history semantics.
The explicit source below is the latest completed 15-minute continuation of the
30-minute post-BC run (5,424,128 cumulative transitions). It remains local and is
not a claim that it is the best evaluated policy. Because this run was marked
diagnostic-only, specify it explicitly rather than relying on automatic selection.

```powershell
cd "$HOME\Documents\ranked-surfers-ai-claude\ranked-surfers-ai-new"
$py = "$HOME\Documents\ranked-surfers-ai\.venv\Scripts\python.exe"
$checkpoint = "reports\multimap_stability_20261107_035054_502059\stability\train\latest.zip"

powershell.exe -NoProfile -ExecutionPolicy Bypass -File .\development\experiments\multimap_classifier\resume.ps1 -Checkpoint $checkpoint -Hours 10 -MaxRestarts 3
```

This starts its own local map servers (ports 8000 and 8010), runs a two-minute
smoke check, then the requested training window. Smoke failure stops everything.
During training, only recognized capture/guard failures can trigger bounded
recovery from a verified checkpoint. Integrity failures, unknown errors and Ctrl+C
stop recovery. Each invocation writes a new report/checkpoint; the source is preserved.
Three restarts means at most three additional attempts after the initial attempt,
not three separate ten-hour runs. Repeated early failures can stop recovery sooner.
The budget includes training-attempt startup and recovery; smoke/setup and final
optimizer-boundary cleanup add time. Graphs and chronological quarter means are
reported per map. Close plot windows before starting another session.

Omitting `-Checkpoint` selects the latest completed compatible multimap run, not
an arbitrary diagnostic or interrupted checkpoint. An explicit BC checkpoint must
be supplied for the first PPO continuation after BC.

## Evaluate both maps

```powershell
& $py -B -s -u .\development\experiments\multimap_classifier\evaluate_both.py --checkpoint $checkpoint --episodes 30
```

Default is visible local gameplay; keep the game foreground and unobscured.
Add `--headless` to use the training capture path. Timing can differ between those
paths. The evaluator starts its own server. With no `--checkpoint`, it selects the
latest completed multimap PPO run; it does not automatically select a BC candidate.

## Human demonstrations and BC

Follow the [complete BC workflow](development/staged/multimap_bc_20261106/README.md)
for recording, review, export and training. Record both maps with arrow keys or
WASD, retain separate validation sessions, select/trim good intervals, and inspect
the exact model-input gallery. Guard warnings are evidence for human labeling,
not automatically approved classifier or BC labels.

The current prepared export has 206 training runs (101,755 examples) and 15
separate validation runs (8,214 examples). Selected runs were at least 12 seconds,
with the last three seconds removed. All five action classes are present, but
NONE dominates; matching validation labels alone does not prove better gameplay.

BC is offline GPU training, not four running games. The original full-encoder BC
and balanced retries remain available for research, but are not the source of the
current continuation. The selected repair uses a fresh actor with fixed feature
normalization, leaving the encoder and critic unchanged during BC. It resets PPO
optimizer moments and saves a separate checkpoint. See the workflow's actor-repair
section for the exact training and verification commands.

The selected local candidate was
`reports/bc_actor_repair_20261107_014610_719776/bc_best.zip`; the current checkpoint
continues it through 30 and 15 minutes of ordinary PPO. PPO trains normally again,
including the encoder and critic. Visual inspection was encouraging, but this does
not establish an improvement over the older PPO policy. The last diagnostic's
training means were 3.54 seconds on Halloween and 3.72 seconds on Tokyo.

Evaluate BC and source checkpoints on both maps using matched settings before
continuing PPO. Subsequent PPO has **no ongoing BC loss or demonstration replay**.
Do not launch simultaneous game training/evaluation or another GPU trainer during BC.

## Classifier maintenance and evidence

[Classifier workflow](development/experiments/multimap_classifier/README.md): keep
old labeled sessions and their splits, add new-map or reviewed guard-error examples,
train a separate candidate, inspect per-map validation and test live startup,
deaths, pause/resume and cleanup before changing the shared selection.
Adding an arbitrary third map still requires worker-routing and release validation;
the current trainer's assignment is explicitly Halloween/Tokyo.

The 6 November candidate retained all 1,731 baseline examples and used 3,566 total.
Validation was 349/349 Halloween and 284/285 Tokyo with no uncertain predictions at
0.90. The one mismatch visually appears to be a menu mislabeled GAMEPLAY; the old
label is preserved. Four identical static menu images occur across classifier
splits. These validation sessions selected the epoch; they are not an independent
test set.

A four-worker live integration check completed in 86 seconds: 13 completed training
episodes, two shared updates, passing learning audits and pause/history restoration
on every worker, with no hard guard stops or leaked processes. Ordinary
image-request-to-command latency was **34.0 ms median / 38.3 ms p95**; reset/resume
tails were higher. This measures command dispatch, not Unity response, and does not
establish overnight reliability or policy improvement.

## Validation and development

Normal training now skips the redundant legacy frame-stack allocation, reuses
bounded pinned transfer buffers, uses precise decision-cap waits, restores plain
Adam step counters to CPU after checkpoint loading, and spaces post-resume captures
without an extra sleep after every capture. Four verification frames are still
required. Evaluation shares the observation, pacing and resume changes.

The completed 15-minute run passed all four learning audits and cleanup checks:
40.96 optimized transitions/s, ordinary request-to-command median 33.44 ms and
p95 37.71 ms. A subsequent short four-worker test reduced median post-resume
verification from about 293 to 195 ms while preserving the checks. These are
different timing windows; command dispatch does not measure Unity's visible response.
Rare capture stalls remain: four workers are preferred to the five-worker probe,
which gained about 9% throughput but had a 40.46 ms p95 and a 256 ms outlier.

GPU rollout caching, deferred optimizer metrics, channels-last layouts, fused Adam
and the alternative capture-pump timer remain experimental and disabled. Native
GPU crop/resize and asynchronous readback have not been implemented. See
[measured changes, limits and rollback](development/experiments/latency_tail_20261107/README.md)
and [publication checks](PUBLICATION_CHECKS_20261107.md).

See [publication checks and known historical test failures](PUBLICATION_CHECKS_20261106.md).
The full legacy test discovery is not green; the same failing cases reproduce on
the previous commit. Current workflow suites passed.

### Development and history

- [Current status and research handoff](PROJECT_STATUS_AND_RESEARCH.md)
- [One-second command history](development/experiments/command_history/README.md)
- [Four/five-frame comparison](development/staged/history_training_compare_20261103/README.md)
- [Latency research and limits](development/staged/multimap_bc_20261106/LATENCY_REVIEW.md)
- [Earlier README and historical reports](README_HISTORY_20261101.md)

Source and tests are versioned. Recordings, maps, model weights, generated reports
and local backup snapshots are excluded from new commits. Existing tracked
historical artifacts remain in Git. Avoid treating old experiment defaults as the
current continuation configuration.
