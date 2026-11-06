# Ranked Surfers AI: multimap PPO and behavioral cloning

Updated 6 November 2026. Development branch: **headless-parallel**.

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
The explicit source below was the successful 30-minute diagnostic and the BC source.
It remains local and is not a claim that it is the best evaluated policy.

```powershell
cd "$HOME\Documents\ranked-surfers-ai-claude\ranked-surfers-ai-new"
$py = "$HOME\Documents\ranked-surfers-ai\.venv\Scripts\python.exe"
$checkpoint = "reports\multimap_stability_20261106_171018_244012\stability\train\latest.zip"

powershell.exe -NoProfile -ExecutionPolicy Bypass -File .\development\experiments\multimap_classifier\resume.ps1 -Checkpoint $checkpoint -Minutes 120 -MaxRestarts 3
```

This starts its own local map servers (ports 8000 and 8010), runs a two-minute
smoke check, then the requested training window. Smoke failure stops everything.
During training, only recognized capture/guard failures can trigger bounded
recovery from a verified checkpoint. Integrity failures, unknown errors and Ctrl+C
stop recovery. Each invocation writes a new report/checkpoint; the source is preserved.
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

```powershell
$bc = ".\development\staged\multimap_bc_20261106"
$dataset = "ai_data\multimap_bc_v1\exports\20261106_223203_345469"
& $py -B -s -u "$bc\train.py" --dataset $dataset --checkpoint $checkpoint --epochs 20 --minutes 60
```

BC is offline GPU training, batch size 16, not four running games. It updates the
shared encoder and actor, freezes critic-head weights, balances map sampling and
uses correctly mirrored actions/history. Changed shared features can still change
critic predictions. It selects by held-out map/class-balanced loss and saves a
separate `bc_best.zip`; the source remains a fallback. PPO optimizer moments are
reset. Dataset checksumming and baseline validation happen before epoch output;
there may be several quiet minutes before the first printed epoch. Validation and
saving add time beyond the optimization budget.

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
